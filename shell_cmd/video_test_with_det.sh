#!/bin/sh

det_model_path=finetuned/det.pth # path：obj det model
kp_det_model_path=finetuned/kp_det.pth # path: key point det model
video_path=data/video/GH010188_single.mp4 # path of the video to test
output_path=vis_results/video_with_mmdet # path to save results


python utils/video_test_with_det.py configs/detection/mask_rcnn_r50_fpn_rat.py ${det_model_path} configs/keypoint/animal/2d_kpt_sview_rgb_img/topdown_heatmap/res50_animalpose_256x256.py ${kp_det_model_path} --video-path ${video_path} --out-video-root ${output_path} --bbox-thr 0.3 --kpt-thr 0.1 --radius 3 --thickness 3








