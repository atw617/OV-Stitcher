from collections import OrderedDict
import math
from typing import Callable, Optional, Sequence, Tuple
from functools import partial

import cv2
import torch
from sklearn.cluster import DBSCAN
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint
import numpy as np

from .utils import to_2tuple
from .pos_embed import get_2d_sincos_pos_embed


class LayerNormFp32(nn.LayerNorm):
    """Subclass torch's LayerNorm to handle fp16 (by casting to float32 and back)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x.to(torch.float32), self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class LayerNorm(nn.LayerNorm):
    """Subclass torch's LayerNorm (with cast back to input dtype)."""

    def forward(self, x: torch.Tensor):
        orig_type = x.dtype
        x = F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        return x.to(orig_type)


class QuickGELU(nn.Module):
    # NOTE This is slower than nn.GELU or nn.SiLU and uses more GPU memory
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))

    def forward(self, x):
        return x.mul_(self.gamma) if self.inplace else x * self.gamma


class PatchDropout(nn.Module):
    """
    https://arxiv.org/abs/2212.00794
    """

    def __init__(self, prob, exclude_first_token=True):
        super().__init__()
        assert 0 <= prob < 1.
        self.prob = prob
        self.exclude_first_token = exclude_first_token  # exclude CLS token

    def forward(self, x):
        if not self.training or self.prob == 0.:
            return x

        if self.exclude_first_token:
            cls_tokens, x = x[:, :1], x[:, 1:]
        else:
            cls_tokens = torch.jit.annotate(torch.Tensor, x[:, :1])

        batch = x.size()[0]
        num_tokens = x.size()[1]

        batch_indices = torch.arange(batch)
        batch_indices = batch_indices[..., None]

        keep_prob = 1 - self.prob
        num_patches_keep = max(1, int(num_tokens * keep_prob))

        rand = torch.randn(batch, num_tokens)
        patch_indices_keep = rand.topk(num_patches_keep, dim=-1).indices

        x = x[batch_indices, patch_indices_keep]

        if self.exclude_first_token:
            x = torch.cat((cls_tokens, x), dim=1)

        return x


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=True,
            scaled_cosine=False,
            scale_heads=False,
            logit_scale_max=math.log(1. / 0.01),
            attn_drop=0.,
            proj_drop=0.
    ):
        super().__init__()
        self.scaled_cosine = scaled_cosine
        self.scale_heads = scale_heads
        assert dim % num_heads == 0, 'dim should be divisible by num_heads'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        self.logit_scale_max = logit_scale_max

        # keeping in_proj in this form (instead of nn.Linear) to match weight scheme of original
        self.in_proj_weight = nn.Parameter(torch.randn((dim * 3, dim)) * self.scale)
        if qkv_bias:
            self.in_proj_bias = nn.Parameter(torch.zeros(dim * 3))
        else:
            self.in_proj_bias = None

        if self.scaled_cosine:
            self.logit_scale = nn.Parameter(torch.log(10 * torch.ones((num_heads, 1, 1))))
        else:
            self.logit_scale = None
        self.attn_drop = nn.Dropout(attn_drop)
        if self.scale_heads:
            self.head_scale = nn.Parameter(torch.ones((num_heads, 1, 1)))
        else:
            self.head_scale = None
        self.out_proj = nn.Linear(dim, dim)
        self.out_drop = nn.Dropout(proj_drop)

    def forward(self, x, attn_mask: Optional[torch.Tensor] = None):
        L, N, C = x.shape
        q, k, v = F.linear(x, self.in_proj_weight, self.in_proj_bias).chunk(3, dim=-1)
        q = q.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        k = k.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)
        v = v.contiguous().view(L, N * self.num_heads, -1).transpose(0, 1)

        if self.logit_scale is not None:
            attn = torch.bmm(F.normalize(q, dim=-1), F.normalize(k, dim=-1).transpose(-1, -2))
            logit_scale = torch.clamp(self.logit_scale, max=self.logit_scale_max).exp()
            attn = attn.view(N, self.num_heads, L, L) * logit_scale
            attn = attn.view(-1, L, L)
        else:
            q = q * self.scale
            attn = torch.bmm(q, k.transpose(-1, -2))

        if attn_mask is not None:
            if attn_mask.dtype == torch.bool:
                new_attn_mask = torch.zeros_like(attn_mask, dtype=q.dtype)
                new_attn_mask.masked_fill_(attn_mask, float("-inf"))
                attn_mask = new_attn_mask
            attn += attn_mask

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = torch.bmm(attn, v)
        if self.head_scale is not None:
            x = x.view(N, self.num_heads, L, C) * self.head_scale
            x = x.view(-1, L, C)
        x = x.transpose(0, 1).reshape(L, N, C)
        x = self.out_proj(x)
        x = self.out_drop(x)
        return x


class AttentionalPooler(nn.Module):
    def __init__(
            self,
            d_model: int,
            context_dim: int,
            n_head: int = 8,
            n_queries: int = 256,
            norm_layer: Callable = LayerNorm
    ):
        super().__init__()
        self.query = nn.Parameter(torch.randn(n_queries, d_model))
        self.attn = nn.MultiheadAttention(d_model, n_head, kdim=context_dim, vdim=context_dim)
        self.ln_q = norm_layer(d_model)
        self.ln_k = norm_layer(context_dim)

    def forward(self, x: torch.Tensor):
        x = self.ln_k(x).permute(1, 0, 2)  # NLD -> LND
        N = x.shape[1]
        q = self.ln_q(self.query)
        out = self.attn(q.unsqueeze(1).expand(-1, N, -1), x, x, need_weights=False)[0]
        return out.permute(1, 0, 2)  # LND -> NLD


class ResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            is_cross_attention: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_head)
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()
        if is_cross_attention:
            self.ln_1_kv = norm_layer(d_model)

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def attention(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = k_x if k_x is not None else q_x
        v_x = v_x if v_x is not None else q_x

        if attn_mask.dtype != torch.bool:
            attn_mask = attn_mask.to(q_x.dtype) if attn_mask is not None else None
        return self.attn(
            q_x, k_x, v_x, need_weights=False, attn_mask=attn_mask
        )[0]

    def forward(
            self,
            q_x: torch.Tensor,
            k_x: Optional[torch.Tensor] = None,
            v_x: Optional[torch.Tensor] = None,
            attn_mask: Optional[torch.Tensor] = None,
    ):
        k_x = self.ln_1_kv(k_x) if hasattr(self, "ln_1_kv") and k_x is not None else None
        v_x = self.ln_1_kv(v_x) if hasattr(self, "ln_1_kv") and v_x is not None else None
        x = q_x + self.ls_1(self.attention(q_x=self.ln_1(q_x), k_x=k_x, v_x=v_x, attn_mask=attn_mask))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


class CustomResidualAttentionBlock(nn.Module):
    def __init__(
            self,
            d_model: int,
            n_head: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            scale_cosine_attn: bool = False,
            scale_heads: bool = False,
            scale_attn: bool = False,
            scale_fc: bool = False,
    ):
        super().__init__()

        self.ln_1 = norm_layer(d_model)
        self.attn = Attention(
            d_model, n_head,
            scaled_cosine=scale_cosine_attn,
            scale_heads=scale_heads,
        )
        self.ln_attn = norm_layer(d_model) if scale_attn else nn.Identity()
        self.ls_1 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

        self.ln_2 = norm_layer(d_model)
        mlp_width = int(d_model * mlp_ratio)
        self.mlp = nn.Sequential(OrderedDict([
            ("c_fc", nn.Linear(d_model, mlp_width)),
            ("gelu", act_layer()),
            ('ln', norm_layer(mlp_width) if scale_fc else nn.Identity()),
            ("c_proj", nn.Linear(mlp_width, d_model))
        ]))
        self.ls_2 = LayerScale(d_model, ls_init_value) if ls_init_value is not None else nn.Identity()

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        x = x + self.ls_1(self.ln_attn(self.attn(self.ln_1(x), attn_mask=attn_mask)))
        x = x + self.ls_2(self.mlp(self.ln_2(x)))
        return x


def _expand_token(token, batch_size: int):
    return token.view(1, 1, -1).expand(batch_size, -1, -1)


class Transformer(nn.Module):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
    ):
        super().__init__()
        self.width = width
        self.layers = layers
        self.grad_checkpointing = False

        self.resblocks = nn.ModuleList([
            ResidualAttentionBlock(
                width, heads, mlp_ratio, ls_init_value=ls_init_value, act_layer=act_layer, norm_layer=norm_layer)
            for _ in range(layers)
        ])

    def get_cast_dtype(self) -> torch.dtype:
        if hasattr(self.resblocks[0].mlp.c_fc, 'int8_original_dtype'):
            return self.resblocks[0].mlp.c_fc.int8_original_dtype
        return self.resblocks[0].mlp.c_fc.weight.dtype

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None):
        for r in self.resblocks:
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                x = checkpoint(r, x, None, None, attn_mask)
            else:
                x = r(x, attn_mask=attn_mask)
        return x


class VisionTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            image_size: int,
            patch_size: int,
            width: int,
            layers: int,
            heads: int,
            mlp_ratio: float,
            ls_init_value: float = None,
            attentional_pool: bool = False,
            attn_pooler_queries: int = 256,
            attn_pooler_heads: int = 8,
            output_dim: int = 512,
            patch_dropout: float = 0.,
            no_ln_pre: bool = False,
            pos_embed_type: str = 'learnable',
            pool_type: str = 'tok',
            final_ln_after_pool: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('tok', 'avg', 'none')
        self.output_tokens = output_tokens
        image_height, image_width = self.image_size = to_2tuple(image_size)
        patch_height, patch_width = self.patch_size = to_2tuple(patch_size)
        self.grid_size = (image_height // patch_height, image_width // patch_width)
        self.final_ln_after_pool = final_ln_after_pool  # currently ignored w/ attn pool enabled
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=width, kernel_size=patch_size, stride=patch_size, bias=False)

        # class embeddings and positional embeddings
        scale = width ** -0.5
        self.class_embedding = nn.Parameter(scale * torch.randn(width))
        if pos_embed_type == 'learnable':
            self.positional_embedding = nn.Parameter(
                scale * torch.randn(self.grid_size[0] * self.grid_size[1] + 1, width))
        elif pos_embed_type == 'sin_cos_2d':
            # fixed sin-cos embedding
            assert self.grid_size[0] == self.grid_size[1], \
                'currently sin cos 2d pos embedding only supports square input'
            self.positional_embedding = nn.Parameter(
                torch.zeros(self.grid_size[0] * self.grid_size[1] + 1, width), requires_grad=False)
            pos_embed_type = get_2d_sincos_pos_embed(width, self.grid_size[0], cls_token=True)
            self.positional_embedding.data.copy_(torch.from_numpy(pos_embed_type).float())
        else:
            raise ValueError

        # setting a patch_dropout of 0. would mean it is disabled and this function would be the identity fn
        self.patch_dropout = PatchDropout(patch_dropout) if patch_dropout > 0. else nn.Identity()

        self.ln_pre = nn.Identity() if no_ln_pre else norm_layer(width)
        self.transformer = Transformer(
            width,
            layers,
            heads,
            mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )

        if attentional_pool:
            if isinstance(attentional_pool, str):
                self.attn_pool_type = attentional_pool
                self.pool_type = 'none'
                if attentional_pool in ('parallel', 'cascade'):
                    self.attn_pool = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=attn_pooler_queries,
                    )
                    self.attn_pool_contrastive = AttentionalPooler(
                        output_dim,
                        width,
                        n_head=attn_pooler_heads,
                        n_queries=1,
                    )
                else:
                    assert False
            else:
                self.attn_pool_type = ''
                self.pool_type = pool_type
                self.attn_pool = AttentionalPooler(
                    output_dim,
                    width,
                    n_head=attn_pooler_heads,
                    n_queries=attn_pooler_queries,
                )
                self.attn_pool_contrastive = None
            pool_dim = output_dim
        else:
            self.attn_pool = None
            pool_dim = width
            self.pool_type = pool_type

        self.ln_post = norm_layer(pool_dim)
        self.proj = nn.Parameter(scale * torch.randn(pool_dim, output_dim))

        self.init_parameters()

    def lock(self, unlocked_groups=0, freeze_bn_stats=False):
        for param in self.parameters():
            param.requires_grad = False

        if unlocked_groups != 0:
            groups = [
                [
                    self.conv1,
                    self.class_embedding,
                    self.positional_embedding,
                    self.ln_pre,
                ],
                *self.transformer.resblocks[:-1],
                [
                    self.transformer.resblocks[-1],
                    self.ln_post,
                ],
                self.proj,
            ]

            def _unlock(x):
                if isinstance(x, Sequence):
                    for g in x:
                        _unlock(g)
                else:
                    if isinstance(x, torch.nn.Parameter):
                        x.requires_grad = True
                    else:
                        for p in x.parameters():
                            p.requires_grad = True

            _unlock(groups[-unlocked_groups:])

    def init_parameters(self):
        # FIXME OpenAI CLIP did not define an init for the VisualTransformer
        # TODO experiment if default PyTorch init, below, or alternate init is best.

        # nn.init.normal_(self.class_embedding, std=self.scale)
        # nn.init.normal_(self.positional_embedding, std=self.scale)
        #
        # proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        # attn_std = self.transformer.width ** -0.5
        # fc_std = (2 * self.transformer.width) ** -0.5
        # for block in self.transformer.resblocks:
        #     nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
        #     nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
        #     nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
        #     nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        #
        # if self.text_projection is not None:
        #     nn.init.normal_(self.text_projection, std=self.scale)
        pass

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def _global_pool(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        if self.pool_type == 'avg':
            pooled, tokens = x[:, 1:].mean(dim=1), x[:, 1:]
        elif self.pool_type == 'tok':
            pooled, tokens = x[:, 0], x[:, 1:]
        else:
            pooled = tokens = x

        return pooled, tokens

    def stitch_features_from_batch(self, feature, img_shape, pixel_locs, paddings, encoder='clip'):
        if encoder == 'dino':
            patch_size = self.dino_patch_size
        elif encoder == 'clip':
            patch_size = self.patch_size[0]
        else:
            raise ValueError("Unsupported encoder")

        # feature: [num_crop, num_token, D]
        num_crop, D, num_token = feature.shape

        H_feat = img_shape[2] // patch_size
        W_feat = img_shape[3] // patch_size

        preds = torch.zeros((1, D, H_feat, W_feat), device=feature.device, dtype=feature.dtype)
        count_mat = torch.zeros((1, 1, H_feat, W_feat), device=feature.device, dtype=feature.dtype)

        for i, (loc, pad) in enumerate(zip(pixel_locs, paddings)):
            y1, x1, y2, x2 = loc
            pad_l, pad_r, pad_t, pad_b = pad

            padded_h = (y2 - y1) + pad_t + pad_b
            padded_w = (x2 - x1) + pad_l + pad_r

            crop_feat_h = math.ceil(padded_h / patch_size)
            crop_feat_w = math.ceil(padded_w / patch_size) 

            y1_patch = y1 // patch_size
            x1_patch = x1 // patch_size
            y2_patch = y2 // patch_size
            x2_patch = x2 // patch_size
            
            pad_t_patch = math.ceil(pad_t / patch_size)
            pad_b_patch = math.ceil(pad_b / patch_size)
            pad_l_patch = math.ceil(pad_l / patch_size)
            pad_r_patch = math.ceil(pad_r / patch_size)

            valid_h = (y2 - y1) // patch_size
            valid_w = (x2 - x1) // patch_size

            feat = feature[i].reshape(D, crop_feat_h, crop_feat_w)

            feat = feat[:, pad_t_patch:pad_t_patch+valid_h, pad_l_patch:pad_l_patch+valid_w]

            preds[:, :, y1_patch:y1_patch+valid_h, x1_patch:x1_patch+valid_w] += feat.unsqueeze(0)

            count_mat[:, :, y1_patch:y1_patch+valid_h, x1_patch:x1_patch+valid_w] += 1

        assert (count_mat == 0).sum() == 0, "Some positions not filled."

        stitched_feature = preds / count_mat

        return stitched_feature

    def get_windowed_imgs(self, img):
        if isinstance(img, list):
            img = img[0].unsqueeze(0)
        if isinstance(self.slide_stride, int):
            stride = (self.slide_stride, self.slide_stride)
        if isinstance(self.slide_crop, int):
            crop_size = (self.slide_crop, self.slide_crop)

        self.img_shape = img.shape
        h_stride, w_stride = stride
        h_crop, w_crop = crop_size
        batch_size, _, h_img, w_img = img.shape
        h_grids = max(h_img - h_crop + h_stride - 1, 0) // h_stride + 1
        w_grids = max(w_img - w_crop + w_stride - 1, 0) // w_stride + 1
        crop_imgs, paddings, pixel_locs = [], [], []
        for h_idx in range(h_grids):
            for w_idx in range(w_grids):
                y1 = h_idx * h_stride
                x1 = w_idx * w_stride
                y2 = min(y1 + h_crop, h_img)
                x2 = min(x1 + w_crop, w_img)
                y1 = max(y2 - h_crop, 0)
                x1 = max(x2 - w_crop, 0)
                crop_img = img[:, :, y1:y2, x1:x2]
                pixel_locs.append(torch.tensor([y1, x1, y2, x2]))

                H, W = crop_img.shape[2:]  # original image shape
                pad = self.compute_padsize(H, W, 56)
                if any(pad):
                    crop_img = nn.functional.pad(crop_img, pad)  # zero padding
                crop_imgs.append(crop_img)
                paddings.append(pad)
        batched_imgs = torch.cat(crop_imgs, dim=0) # [n_patches, 3, h, w]

        return batched_imgs, pixel_locs, paddings

    @torch.inference_mode()
    def forward(self, img_shape, x: torch.Tensor, dino_feats, dino_patch_size, feat_shape, masks, coords, paddings, slide_stride, slide_crop):
        B, nc, h, w = x.shape
        token_size = img_shape[-2] // self.patch_size[0], img_shape[-1] // self.patch_size[1]
        cropped_token_size = h // self.patch_size[0], w // self.patch_size[1]

        self.slide_stride = slide_stride
        self.slide_crop = slide_crop
        self.dino_patch_size = dino_patch_size

        x = self.conv1(x)  # shape = [*, width, grid, grid]
        x = x.reshape(x.shape[0], x.shape[1], -1)  # shape = [*, width, grid ** 2]
        x = x.permute(0, 2, 1)  # shape = [*, grid ** 2, width]

        # class embeddings and positional embeddings
        x = torch.cat([_expand_token(self.class_embedding, x.shape[0]).to(x.dtype), x], dim=1)

        if x.shape[1] != self.positional_embedding.shape[0]:
            x = x + self.interpolate_pos_encoding(x, h, w).to(x.dtype)
        else:
            x = x + self.positional_embedding.to(x.dtype)

        x = self.patch_dropout(x)
        x = self.ln_pre(x)
        x = x.permute(1, 0, 2)  # NLD -> LND

        # Some hyperparameters
        ALPHA, BETA = 1, 0.5  # Coefficients that balance two additional branches
        num_layers = len(self.transformer.resblocks)    # Number of lower layers in Spatial Branch for different sizes of CLIP
        if num_layers == 12:
            NUM_SPATIAL = 5
        elif num_layers == 24:
            NUM_SPATIAL = 12
        elif num_layers == 32:
            NUM_SPATIAL = 28
        else:
            NUM_SPATIAL = num_layers - 4
        TAU = 4 # Temperature coefficient
        EPS, MIN_SAMPLES = 0.2, 1  # DBSCAN

        # Value Reconstruction
        similarity = torch.einsum("b m c, b n c -> b m n", dino_feats, dino_feats)

        # Mask Merging
        masks = F.interpolate(masks.float(), size=feat_shape, mode='nearest').int().squeeze(1)
        
        dino_feats = dino_feats.reshape(1, *feat_shape, -1)
        for batch_idx in range(1):
            mask_values = torch.unique(masks[batch_idx])
            if mask_values.shape[0] > 2:
                region_masks = [idx == masks[batch_idx] for idx in mask_values]
                dino_feat = dino_feats[batch_idx]
                instance_feats = torch.stack([dino_feat[mask].mean(dim=0) for mask in region_masks])
                instance_feats = F.normalize(instance_feats, dim=1)[1:]
                instance_sim = torch.einsum("m c, n c -> m n", instance_feats, instance_feats)

                dbscan = DBSCAN(metric='precomputed', eps=EPS, min_samples=MIN_SAMPLES)
                cluster = dbscan.fit_predict(((instance_sim.max() - instance_sim).cpu().numpy()))
                cluster = torch.from_numpy(cluster).to(x.device)
                cluster = cluster.to(masks.dtype)
                for idx, cluster_v in enumerate(cluster, start=1):
                    if cluster_v != -1:
                        masks[batch_idx][region_masks[idx]] = cluster_v + 10000

        # Scope Reconstruction # 이건 마지막에 쓰일 attention_weights
        for batch_idx in range(1):
            region_masks = [(idx == masks[batch_idx]).view(-1) for idx in torch.unique(masks[batch_idx])] # 47개의 전체 N
            equal_idx = torch.zeros_like(similarity[batch_idx]).bool()
            for mask in region_masks[1:]:
                equal_idx += (torch.outer(mask, mask))

            # For unsegmented regions, there are two implementation methods: 1. `mask.unsqueeze(0) + mask.unsqueeze(1)` and 2. `torch.outer(mask, mask)`.
            # Method 1: Allows patches from segmented and unsegmented regions to attend to each other (if similarity > mean).
            # Method 2: Disallows any attention between patches from segmented and unsegmented regions.
            # Experiments show that the first method performs slightly better.
            mask = region_masks[0]
            equal_idx += ((mask.unsqueeze(0) + mask.unsqueeze(1)) * (similarity[batch_idx] > similarity[batch_idx].mean()))
            similarity[batch_idx][~equal_idx] = float('-inf')
        attn_weights = (similarity * TAU).softmax(dim=-1)

        # Cropped_attn_mask for Semantic Branch
        # Please note that each image has a different number of masks, so Semantic Branch only considered the case where the batch size is 1.

        cropped_masks, _, _ = self.get_windowed_imgs(F.interpolate(masks.float().unsqueeze(1), size=(img_shape[-2], img_shape[-1]), mode='nearest')) # ([21, 1, 224, 224])
        cropped_masks = F.interpolate(cropped_masks.float(), size=cropped_token_size, mode='nearest').int().squeeze(1)
    
        cropped_region_masks = [idx == cropped_masks for idx in torch.unique(masks)]

        mct = x[:1].repeat(len(cropped_region_masks), 1, 1)
        num_mct = mct.shape[0]
        x = torch.cat([mct, x], dim=0) 
        cropped_attn_mask = torch.zeros((x.shape[0], x.shape[0]), device=x.device).bool()
        cropped_attn_mask[num_mct:, num_mct:] = True
        cropped_attn_mask[:num_mct, num_mct] = True
        cropped_attn_mask = cropped_attn_mask.repeat(B, 1, 1)
        cropped_region_masks = torch.stack(cropped_region_masks, dim=1) # 21, 47, 14, 14 겠지?

        cropped_attn_mask[:, :num_mct, num_mct + 1:] = cropped_region_masks.view(cropped_region_masks.shape[0], cropped_region_masks.shape[1], -1)
        cropped_attn_mask = ~cropped_attn_mask

        # Stitched attention mask for Semantic Branch
        stitched_region_masks = F.interpolate(masks.float().unsqueeze(1), size=token_size, mode='nearest').int().squeeze(1)
        stitched_region_masks = stitched_region_masks[0] # 1, 1, 28, 56
        stitched_region_masks = [idx == stitched_region_masks for idx in torch.unique(masks)]
        stitched_attn_mask = torch.zeros((token_size[0]*token_size[1] + num_mct + 1, token_size[0]*token_size[1] + num_mct + 1), device=x.device).bool()
        stitched_attn_mask[num_mct:, num_mct:] = True
        stitched_attn_mask[:num_mct, num_mct] = True

        stitched_region_masks = torch.stack(stitched_region_masks, dim=0)
        stitched_attn_mask[:num_mct, num_mct + 1:] = stitched_region_masks.view(stitched_region_masks.shape[0], -1)
        stitched_attn_mask = ~stitched_attn_mask

        feats_spatial = []

        for idx, blk in enumerate(self.transformer.resblocks[:-1]):
            output = []

            if num_layers - 2 - NUM_SPATIAL <= idx <= num_layers - 3:
                num_heads = blk.attn.num_heads
                head_dim = blk.attn.embed_dim // num_heads
                scale = head_dim ** -0.5

                stitched_x_inter = self.stitch_features_from_batch(x[num_mct+1:].permute(1, 2, 0), img_shape, coords, paddings, encoder='clip').reshape(1, blk.attn.embed_dim, -1).permute(0, 2, 1)
                q_inter, k_inter, v_inter = F.linear(blk.ln_1(x), blk.attn.in_proj_weight, blk.attn.in_proj_bias)[num_mct+1:].chunk(3, dim=-1)

                stitched_q_inter = self.stitch_features_from_batch(q_inter.permute(1, 2, 0), img_shape, coords, paddings, encoder='clip').reshape(1, blk.attn.embed_dim, -1).permute(0, 2, 1)
                stitched_k_inter = self.stitch_features_from_batch(k_inter.permute(1, 2, 0), img_shape, coords, paddings, encoder='clip').reshape(1, blk.attn.embed_dim, -1).permute(0, 2, 1)
                stitched_v_inter = self.stitch_features_from_batch(v_inter.permute(1, 2, 0), img_shape, coords, paddings, encoder='clip').reshape(1, blk.attn.embed_dim, -1).permute(0, 2, 1)

                attn = torch.bmm(stitched_q_inter, stitched_k_inter.transpose(1, 2)) * scale  + stitched_attn_mask[num_mct+1:, num_mct+1:].unsqueeze(0)
                attn = F.softmax(attn, dim=-1)
                attn = torch.bmm(attn, stitched_v_inter)

                stitched_x_inter = stitched_x_inter + blk.attn.out_proj(attn)

                stitched_x_inter = stitched_x_inter + blk.mlp(blk.ln_2(stitched_x_inter))

                feats_spatial.append(stitched_x_inter)

            for input, attn_mask in zip(x.permute(1, 0, 2), cropped_attn_mask):
                x = blk(input.unsqueeze(1), attn_mask=attn_mask)
                output.append(x)
            
            x = torch.stack(output, dim=1).squeeze(2)
            

        for blk in self.transformer.resblocks[-1:]:
            # Main Branch and Spatial Branch
            x_main = x[num_mct:]
            x_spatial = torch.stack(feats_spatial).mean(0)
            attn_weights = attn_weights.expand(2, -1 ,-1)
            embed_dim = blk.attn.embed_dim
            v_proj_weight = blk.attn.in_proj_weight.narrow(0, 2 * embed_dim, embed_dim)
            v_proj_bias = blk.attn.in_proj_bias.narrow(0, 2 * embed_dim, embed_dim)

            v_main = F.linear(blk.ln_1(x_main), v_proj_weight, v_proj_bias)
            v_spatial = F.linear(blk.ln_1(x_spatial.permute(1, 0, 2)), v_proj_weight, v_proj_bias)

            stitched_v_main = self.stitch_features_from_batch(v_main[1:].permute(1, 2, 0), img_shape, coords, paddings, encoder='clip')

            v = torch.cat([stitched_v_main, v_spatial.permute(1, 2, 0).reshape(stitched_v_main.shape)], dim=0)
            v = F.interpolate(v, size=feat_shape, mode='bilinear', align_corners=False)
            v = v.flatten(2, 3).transpose(1, 2)  # B, L, C
            x_main_spatial = torch.bmm(attn_weights, v)
            x_main_spatial = x_main_spatial.transpose(0, 1).contiguous()
            x_main_spatial = blk.attn.out_proj(x_main_spatial)

            mct = []
            for input, attn_mask in zip(x.permute(1, 0, 2), cropped_attn_mask):
                mct.append(blk(input.unsqueeze(1), attn_mask=attn_mask))
            mct = torch.stack(mct, dim=1).squeeze(2)[:num_mct, :].permute(1, 0, 2)

            expand_masks = F.interpolate(cropped_region_masks.to(mct.dtype),
                                         size=(self.slide_crop // self.dino_patch_size, self.slide_crop // self.dino_patch_size),
                                         mode='nearest').flatten(2, -1)

            x_semantic = expand_masks.transpose(1, -1) @ mct
            x_semantic = self.stitch_features_from_batch(x_semantic.permute(0, 2, 1), img_shape, coords, paddings, encoder='dino')
            x_semantic = x_semantic.flatten(-2, -1).permute(2, 0, 1)

            x = torch.cat([x_main_spatial, x_semantic], dim=1)
        
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_post(x)

        x_main, x_spatial, x_semantic = x.chunk(3, dim=0)
        x = x_main + x_spatial * ALPHA + x_semantic * BETA

        x = x @ self.proj
        
        return x

    def interpolate_pos_encoding(self, x, w, h):
        npatch = x.shape[1] - 1
        N = self.positional_embedding.shape[0] - 1
        if npatch == N and w == h:
            return self.positional_embedding
        class_pos_embed = self.positional_embedding[[0]]
        patch_pos_embed = self.positional_embedding[1:]
        dim = x.shape[-1]
        w0 = w // self.patch_size[0]
        h0 = h // self.patch_size[1]
        w0, h0 = w0 + 0.1, h0 + 0.1
        patch_pos_embed = nn.functional.interpolate(
            patch_pos_embed.reshape(1, int(math.sqrt(N)), int(math.sqrt(N)), dim).permute(0, 3, 1, 2),
            scale_factor=(w0 / math.sqrt(N), h0 / math.sqrt(N)),
            mode='bicubic',
        )
        assert int(w0) == patch_pos_embed.shape[-2] and int(h0) == patch_pos_embed.shape[-1]
        patch_pos_embed = patch_pos_embed.permute(0, 2, 3, 1).view(1, -1, dim)
        return torch.cat((class_pos_embed.unsqueeze(0), patch_pos_embed), dim=1)
    
    def compute_padsize(self, H: int, W: int, patch_size: int):
        l, r, t, b = 0, 0, 0, 0
        if W % patch_size:
            lr = patch_size - (W % patch_size)
            l = lr // 2
            r = lr - l

        if H % patch_size:
            tb = patch_size - (H % patch_size)
            t = tb // 2
            b = tb - t

        return l, r, t, b

def text_global_pool(x, text: Optional[torch.Tensor] = None, pool_type: str = 'argmax'):
    if pool_type == 'first':
        pooled, tokens = x[:, 0], x[:, 1:]
    elif pool_type == 'last':
        pooled, tokens = x[:, -1], x[:, :-1]
    elif pool_type == 'argmax':
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        assert text is not None
        pooled, tokens = x[torch.arange(x.shape[0]), text.argmax(dim=-1)], x
    else:
        pooled = tokens = x

    return pooled, tokens


class TextTransformer(nn.Module):
    output_tokens: torch.jit.Final[bool]

    def __init__(
            self,
            context_length: int = 77,
            vocab_size: int = 49408,
            width: int = 512,
            heads: int = 8,
            layers: int = 12,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            output_dim: int = 512,
            embed_cls: bool = False,
            no_causal_mask: bool = False,
            pad_id: int = 0,
            pool_type: str = 'argmax',
            proj_bias: bool = False,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_tokens: bool = False,
    ):
        super().__init__()
        assert pool_type in ('first', 'last', 'argmax', 'none')
        self.output_tokens = output_tokens
        self.num_pos = self.context_length = context_length
        self.vocab_size = vocab_size
        self.width = width
        self.output_dim = output_dim
        self.heads = heads
        self.pad_id = pad_id
        self.pool_type = pool_type

        self.token_embedding = nn.Embedding(vocab_size, width)
        if embed_cls:
            self.cls_emb = nn.Parameter(torch.empty(width))
            self.num_pos += 1
        else:
            self.cls_emb = None
        self.positional_embedding = nn.Parameter(torch.empty(self.num_pos, width))
        self.transformer = Transformer(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.ln_final = norm_layer(width)

        if no_causal_mask:
            self.attn_mask = None
        else:
            self.register_buffer('attn_mask', self.build_causal_mask(), persistent=False)

        if proj_bias:
            self.text_projection = nn.Linear(width, output_dim)
        else:
            self.text_projection = nn.Parameter(torch.empty(width, output_dim))

        self.init_parameters()

    def init_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        if self.cls_emb is not None:
            nn.init.normal_(self.cls_emb, std=0.01)

        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                nn.init.normal_(self.text_projection.weight, std=self.transformer.width ** -0.5)
                if self.text_projection.bias is not None:
                    nn.init.zeros_(self.text_projection.bias)
            else:
                nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.transformer.grad_checkpointing = enable

    def build_causal_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.num_pos, self.num_pos)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def build_cls_mask(self, text, cast_dtype: torch.dtype):
        cls_mask = (text != self.pad_id).unsqueeze(1)
        cls_mask = F.pad(cls_mask, (1, 0, cls_mask.shape[2], 0), value=True)
        additive_mask = torch.empty(cls_mask.shape, dtype=cast_dtype, device=cls_mask.device)
        additive_mask.fill_(0)
        additive_mask.masked_fill_(~cls_mask, float("-inf"))
        additive_mask = torch.repeat_interleave(additive_mask, self.heads, 0)
        return additive_mask

    def forward(self, text):
        cast_dtype = self.transformer.get_cast_dtype()
        seq_len = text.shape[1]

        x = self.token_embedding(text).to(cast_dtype)  # [batch_size, n_ctx, d_model]
        attn_mask = self.attn_mask
        if self.cls_emb is not None:
            seq_len += 1
            x = torch.cat([x, _expand_token(self.cls_emb, x.shape[0])], dim=1)
            cls_mask = self.build_cls_mask(text, cast_dtype)
            if attn_mask is not None:
                attn_mask = attn_mask[None, :seq_len, :seq_len] + cls_mask[:, :seq_len, :seq_len]

        x = x + self.positional_embedding[:seq_len].to(cast_dtype)
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x, attn_mask=attn_mask)
        x = x.permute(1, 0, 2)  # LND -> NLD

        # x.shape = [batch_size, n_ctx, transformer.width]
        if self.cls_emb is not None:
            # presence of appended cls embed (CoCa) overrides pool_type, always take last token
            pooled, tokens = text_global_pool(x, pool_type='last')
            pooled = self.ln_final(pooled)  # final LN applied after pooling in this case
        else:
            x = self.ln_final(x)
            pooled, tokens = text_global_pool(x, text, pool_type=self.pool_type)

        if self.text_projection is not None:
            if isinstance(self.text_projection, nn.Linear):
                pooled = self.text_projection(pooled)
            else:
                pooled = pooled @ self.text_projection

        if self.output_tokens:
            return pooled, tokens

        return pooled


class MultimodalTransformer(Transformer):
    def __init__(
            self,
            width: int,
            layers: int,
            heads: int,
            context_length: int = 77,
            mlp_ratio: float = 4.0,
            ls_init_value: float = None,
            act_layer: Callable = nn.GELU,
            norm_layer: Callable = LayerNorm,
            output_dim: int = 512,
    ):

        super().__init__(
            width=width,
            layers=layers,
            heads=heads,
            mlp_ratio=mlp_ratio,
            ls_init_value=ls_init_value,
            act_layer=act_layer,
            norm_layer=norm_layer,
        )
        self.context_length = context_length
        self.cross_attn = nn.ModuleList([
            ResidualAttentionBlock(
                width,
                heads,
                mlp_ratio,
                ls_init_value=ls_init_value,
                act_layer=act_layer,
                norm_layer=norm_layer,
                is_cross_attention=True,
            )
            for _ in range(layers)
        ])

        self.register_buffer('attn_mask', self.build_attention_mask(), persistent=False)

        self.ln_final = norm_layer(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))

    def init_parameters(self):
        proj_std = (self.transformer.width ** -0.5) * ((2 * self.transformer.layers) ** -0.5)
        attn_std = self.transformer.width ** -0.5
        fc_std = (2 * self.transformer.width) ** -0.5
        for block in self.transformer.resblocks:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)
        for block in self.transformer.cross_attn:
            nn.init.normal_(block.attn.in_proj_weight, std=attn_std)
            nn.init.normal_(block.attn.out_proj.weight, std=proj_std)
            nn.init.normal_(block.mlp.c_fc.weight, std=fc_std)
            nn.init.normal_(block.mlp.c_proj.weight, std=proj_std)

        if self.text_projection is not None:
            nn.init.normal_(self.text_projection, std=self.transformer.width ** -0.5)

    def build_attention_mask(self):
        # lazily create causal attention mask, with full attention between the tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(self.context_length, self.context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    def forward(self, image_embs, text_embs):
        text_embs = text_embs.permute(1, 0, 2)  # NLD -> LNDsq
        image_embs = image_embs.permute(1, 0, 2)  # NLD -> LND
        seq_len = text_embs.shape[0]

        for resblock, cross_attn in zip(self.resblocks, self.cross_attn):
            if self.grad_checkpointing and not torch.jit.is_scripting():
                # TODO: handle kwargs https://github.com/pytorch/pytorch/issues/79887#issuecomment-1161758372
                text_embs = checkpoint(resblock, text_embs, None, None, self.attn_mask[:seq_len, :seq_len])
                text_embs = checkpoint(cross_attn, text_embs, image_embs, image_embs, None)
            else:
                text_embs = resblock(text_embs, attn_mask=self.attn_mask[:seq_len, :seq_len])
                text_embs = cross_attn(text_embs, k_x=image_embs, v_x=image_embs)

        x = text_embs.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x)

        if self.text_projection is not None:
            x = x @ self.text_projection

        return x

    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
