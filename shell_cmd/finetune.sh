#!/bin/sh
 
 save_dir=work_dirs/finetune # dir of saving finetuned model
 
 
 python tools/train.py configs/keypoint/animal/2d_kpt_sview_rgb_img/topdown_heatmap/res50_animalpose_256x256.py --work-dir ${save_dir}