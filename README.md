# [CVPR 2026 Findings] OV-Stitcher: A Global Context-Aware Framework for Training-Free Open-Vocabulary Semantic Segmentation

## 📄Overview
<div align="center">
   <img src="assets/overview.jpeg" alt="OV-Stitcehr Framework" width="80%"/>
</div>

## ⚙️Datasets
`With background class`: PASCAL VOC (VOC21), PASCAL Context (PC60), and COCO Object (Object),

`Without background class`: VOC20, PC59 (i.e., VOC21 and PC60 without the background category), Cityscapes (City), ADE20k (ADE), and COCO Stuff164k (Stuff).

Please follow the data preparation document of [MMSeg](https://github.com/open-mmlab/mmsegmentation/blob/main/docs/en/user_guides/2_dataset_prepare.md) to download and pre-process
the datasets. Move the datasets to the `data/` directory.
The COCO Object dataset can be converted from COCO Stuff164k by executing the following command:

```
python datasets/cvt_coco_object.py PATH_TO_COCO_STUFF164K -o PATH_TO_COCO164K
```

## 📊Results
| Dataset | ViT-B/16 | ViT-L/14 |
|--------|-------------------|-------------------|
| VOC21 | **75.7** | **74.0** |
| Context60 | **43.9** | **43.4** |
| Object | **42.6** | **46.5** |
| VOC20 | **89.8** | **90.2** |
| City | **48.1** | **50.6** |
| Context59 | **48.8** | **48.6** |
| ADE20K | **24.7** | **27.7** |
| Stuff | **31.8** | **31.6** |
| **Avg.** | **50.7** | **51.6** |

## Code
Code will be released soon.
