# 在视频上结合mmdet进行测试。

python demo/top_down_video_demo_with_mmdet.py \
my_det_config/mask_rcnn_r50_fpn_rat.py \
/run/media/cv/d/lzl/fsl/from_50/train/mmdetection-master/work_dirs/mask_rcnn_r50_fpn_1x_rat/epoch_400.pth \
configs/animal/2d_kpt_sview_rgb_img/topdown_heatmap/animalpose/my_res50_animalpose.py \
work_dirs/rat_12_28_finetune_2/epoch_60.pth \
--video-path 论文_分析视频/GH010190_dual.mp4 \
--out-video-root vis_results/video_with_mmdet \
--bbox-thr 0.3 \
--kpt-thr 0.1 \
--radius 3 \
--thickness 3


