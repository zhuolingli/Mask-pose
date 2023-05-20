#!/bin/sh

det_model_path=finetuned/det.pth # path：obj det model
kp_det_model_path=finetuned/kp_det.pth # path: key point det model

img_root=data 
img_name=coco/val2017/img002C1C0.jpg
output_path=vis_results/video_with_mmdet # path to save results


python demo/top_down_img_demo_with_mmdet.py \
configs/detection/mask_rcnn_r50_fpn_rat.py ${det_model_path} \
configs/keypoint/animal/2d_kpt_sview_rgb_img/topdown_heatmap/res50_animalpose_256x256.py ${kp_det_model_path} \
--img-root ${img_root} \
--img ${img_name} \
--out-img-root ${output_path} \
--bbox-thr 0.3 \
--kpt-thr 0.1 \
--radius 3 \
--thickness 3


