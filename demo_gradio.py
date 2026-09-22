import gc
import numpy as np
import torch

try:
    from CropFormer.demo_mask2former.demo import get_entityseg
    from detectron2.data.detection_utils import read_image
except:
    print("EntitySeg is not installed")
try:
    from sam2.build_sam import build_sam2
    from sam2.automatic_mask_generator import SAM2AutomaticMaskGenerator
except:
    print("SAM2 is not installed")

np.random.seed(0)
from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from ov_stitcher_segmentor import OVStitcherSegmentation
from scipy.ndimage import label
import gradio as gr
from matplotlib.colors import hsv_to_rgb


def generate_distinct_colors(n):
    hues = np.linspace(0, 1, n, endpoint=False)
    sats = np.ones(n) * 0.8
    vals = np.ones(n) * 0.9
    hsv_colors = np.stack((hues, sats, vals), axis=-1)

    rgb_colors = (hsv_to_rgb(hsv_colors) * 255).astype(np.uint8)
    return rgb_colors


def set_sam_mask_generator(model, sam_model, points_per_side=8, pred_iou_thresh=0.4, stability_score_thresh=0.4, multimask_output=False):
    model.mask_generator_type = 'sam2'
    model.mask_generator = SAM2AutomaticMaskGenerator(
        model=sam_model,
        points_per_side=points_per_side,
        pred_iou_thresh=pred_iou_thresh,
        stability_score_thresh=stability_score_thresh,
        multimask_output=multimask_output,
    )


def set_entity_mask_generator(model, entity_model):
    model.mask_generator_type = 'entityseg'
    model.confidence_threshold = 0.5
    model.mask_generator = entity_model


def clear_memory():
    """Release Python and CUDA caches."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
        return "CUDA memory cache cleared."
    return "CPU memory cleared."


model = OVStitcherSegmentation(clip_type='metaclip_fullcc', model_type='ViT-L-14-quickgelu', dino_type='dino_vitb8',
                             name_path='./configs/my_name.txt', mask_generator=None)

try:
    sam2 = build_sam2("sam2_hiera_l.yaml", "sam2_hiera_large.pt", device='cuda', apply_postprocessing=False)
    sam2 = sam2.half()
except:
    print("SAM2 is not installed")
try:
    entity_model = get_entityseg(cfg_file="mask2former_hornet_3x.yaml", ckpt_path="Mask2Former_hornet_3x_576d0b.pth")
except:
    print("EntitySeg is not installed")

def process_image(img_path, input_text, segmentation_method, points_per_side, pred_iou_thresh, stability_score_thresh, multimask_output):
    if segmentation_method == 'SAM2':
        set_sam_mask_generator(model, sam2, points_per_side, pred_iou_thresh, stability_score_thresh, multimask_output)
    elif segmentation_method == 'EntitySeg':
        set_entity_mask_generator(model, entity_model)
    else:
        raise ValueError("Unknown segmentation method selected.")

    input_img = Image.open(img_path).convert("RGB")

    with open('./configs/my_name.txt', 'r') as file:
        lines = file.readlines()
    name_list = [line.strip() for line in lines]

    if set(name_list) != set(list(dict.fromkeys(item.strip() for item in input_text.split(',')))):
        name_list = list(dict.fromkeys(item.strip() for item in input_text.split(',')))
        with open('./configs/my_name.txt', 'w') as writers:
            for i in range(len(name_list)):
                if i == len(name_list) - 1:
                    writers.write(name_list[i])
                else:
                    writers.write(name_list[i] + '\n')
        writers.close()
        model.generate_category_embeddings('./configs/my_name.txt')

    class_names = {index: value for index, value in enumerate(name_list)}

    img_tensor = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.48145466, 0.4578275, 0.40821073], [0.26862954, 0.26130258, 0.27577711]),
    ])(input_img)
    img_tensor = img_tensor.unsqueeze(0).to('cuda')

    seg_pred = model.predict([img_tensor, img_path], data_samples=None)
    seg_pred = seg_pred.data.cpu().numpy().squeeze()

    regions = {}
    for label_id in np.unique(seg_pred):
        labeled_array, _ = label(seg_pred == label_id)
        sizes = np.bincount(labeled_array.ravel())
        max_idx = np.argmax(sizes[1:]) + 1

        region = (labeled_array == max_idx)
        y, x = np.where(region)
        center_y, center_x = int(np.mean(y)), int(np.mean(x))

        if y[0] + 10 < region.shape[0] and x[0] + 10 < region.shape[1]:
            regions[label_id] = (center_y, x[0] + 10)
        else:
            regions[label_id] = (center_y, x[0])
    output_str = '; '.join(f"{class_names[idx]}" for idx, coor in regions.items())

    palette = generate_distinct_colors(len(name_list))
    seg_pred_color = palette[seg_pred]

    seg_pred_color = (seg_pred_color * 0.7 + np.array(input_img) * 0.3).astype(np.uint8)

    seg_pred_color = Image.fromarray(seg_pred_color)
    draw = ImageDraw.Draw(seg_pred_color)
    font = ImageFont.load_default(size=max(seg_pred.shape[0], seg_pred.shape[1]) // 50)
    for class_id, (center_y, center_x) in regions.items():
        if class_id in class_names:
            text = class_names[class_id]
            draw.text((center_x, center_y), text, font=font, fill="white", anchor="lt")

    return seg_pred_color, output_str


example_list = [
    ["samples/Golden Retriever,Husky,background.jpg", "golden retriever,husky,background"],
    ["samples/cat.jpg", "cat, background"],
    ["samples/animals.png", "cheetah, zebra, rhinoceros, elephant, buffalo, giraffe, antelope, lion, leopard, background"],
    ["samples/fruit.jpg", "background, banana, pineapple, broccoli, potato, tomato, chili pepper, kiwi, avocado, orange, lemon, strawberry, cherry tomato, parsley, lime"],
    ["samples/giraffe.png", "giraffe, tree, grass, mountain, sky"],
]


# 创建 Gradio 接口
with gr.Blocks() as iface:
    gr.Markdown("# OV-Stitcher Segmentation")
    gr.Markdown("Upload an image, provide class names, choose a mask generator, and adjust parameters.")

    with gr.Row():
        with gr.Column(scale=1):
            img_input = gr.Image(type="filepath", label="Input Image")
            text_input = gr.Textbox(label="Class Names (comma-separated)", info="e.g., 'cat,dog,background'")


            method_selector = gr.Radio(
                choices=['EntitySeg', 'SAM2'],
                value='SAM2',  
                label="Mask Generator"
            )

            with gr.Accordion("SAM2 Parameters", open=True, visible=False) as sam_params_group:
                points_per_side_slider = gr.Slider(minimum=4, maximum=64, step=4, value=16, label="Points Per Side")
                pred_iou_thresh_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.4, label="Prediction IoU Threshold")
                stability_score_thresh_slider = gr.Slider(minimum=0.0, maximum=1.0, step=0.05, value=0.4, label="Stability Score Threshold")
                multimask_output_checkbox = gr.Checkbox(value=False, label="Multimask Output")

            submit_btn = gr.Button("Segment Image", variant="primary")

        with gr.Column(scale=1):
            img_output = gr.Image(type="pil", label="Segmentation Result")
            label_output = gr.Label(label="Detected Classes")
            memory_btn = gr.Button("Free Memory")
            memory_status = gr.Textbox(label="Memory Status", value="Idle", interactive=False)

    gr.Examples(examples=example_list, inputs=[img_input, text_input])

    def toggle_sam_params(selection):
        if selection == 'SAM2':
            return gr.update(visible=True)
        else:
            return gr.update(visible=False)


    method_selector.change(
        fn=toggle_sam_params,
        inputs=method_selector,
        outputs=sam_params_group
    )

    submit_btn.click(
        fn=process_image,
        inputs=[
            img_input,
            text_input,
            method_selector,
            points_per_side_slider,
            pred_iou_thresh_slider,
            stability_score_thresh_slider,
            multimask_output_checkbox
        ],
        outputs=[img_output, label_output]
    )

    memory_btn.click(
        fn=clear_memory,
        inputs=None,
        outputs=memory_status
    )


iface.launch()
