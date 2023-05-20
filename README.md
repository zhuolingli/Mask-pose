
# Markerless Animal Pose Estimation Method Based on Mask-pose
This is an open-source tool for animal pose estimation based on [MMdetection](https://github.com/open-mmlab/mmdetection) and [MMpose](https://github.com/open-mmlab/mmpose).
> **Abstract:** *The study of animal behavior has evolved from observational data collection to controlled laboratory experiments, incorporating multidisciplinary approaches. Efficient and accurate estimation of animal postures is critical for behavioral studies. Videography allows for easy observation and recording of animal postures, but marker systems can be expensive and potentially disruptive to animal behavior. Markerless posture estimation can simplify experimental workflows and provide more objective measurements. To address this challenge, we introduce a toolkit based on neural network transfer learning, which adapts to animal pose-tracking tasks with minimal training video frames. This method combines two neural network modules derived from Mask-RCNN and SimpleBaseline for object detection and key point detection, respectively. It also includes a data annotation module and script files for data preparation and result visualization. We demonstrate that Mask-pose, leveraging state-of-the-art methods for pose estimation and transfer learning framework, achieves better performance in various laboratory settings with limited training data. Maskpose is released as free and open-source software to promote ongoing contributions and support open science.*

![image](https://github.com/zhuolingli/Mask-pose/assets/67094418/e4b15f34-9b24-4bc4-9e6b-66a03b84aaa6)
![demo](https://github.com/zhuolingli/Mask-pose/assets/67094418/1773be31-b3d6-4689-9dcb-f21321efa1ae)
![demo2](https://github.com/zhuolingli/Mask-pose/assets/67094418/1764fd47-adc9-4980-adc3-e7bed0ea93fb)


## &#x1F527; Usage
### Dependencies
- MMdetection version: 2.25.0
- MMpose version: 0.28.1
- CUDA version: 11.1
- Pytorch version: 1.9.1
### Data Preparation
- Please use [Labelme](https://github.com/wkentaro/labelme) to label data. In Labelme, create “Polygons” and “Points” for annonations of masks and key points, respectively.
- The file format of Labelme is incompatible with the feeding format **(coco)** of MMpose and MMdetection. Run `utils/labelme2coco.py` to convert the format and put them in `./data`. 
  ```
  python utils/labelme2coco.py --input_dir {labelme_data_dir} --output_dir data/coco --labels labels.txt
  ```
### Train 
- Add the config python file for the customized coco dataset in `configs/datasets/.`.
- Download the pretrained models of [Mask-RCNN](https://github.com/open-mmlab/mmdetection/tree/main/configs/mask_rcnn) and [SimpleBaseline](https://mmpose.readthedocs.io/en/latest/papers/algorithms.html#simplebaseline2d-eccv-2018) and put them in `pretrain/.`.
- Modify the model config file (`configs/model/res50_animalpose_256x256.py`). 
- Run `shell_cmd/finetune.sh` to finetune Mask-RCNN and SimpleBaseline, respectively.
  ```
  sh shell_cmd/finetune.sh
  ```
### Pose Estimation
The pipelines of pose estimation include animal detection (bboxes or masks) and key point detection.
- For a single image.
  ```
  sh shell_cmd/img_test_with_det.sh
  ```
- For a video.
  ```
  sh shell_cmd/video_test_with_det.sh
  ```
We also provide a mouse dataset with coco format, and corresponding finetuned models for users. Please click this [link](https://pan.baidu.com/s/1uzTnMlZ06YOg8kqPXfYaxw?pwd=xmsl) (code:xmsl)




### Visualization
Users can utilize `utils/parse_results.py` to parse the results of pose estimation, draw the animal's movement trajectory, and analyzes the time-sequence motion state. 
![image](https://github.com/zhuolingli/Mask-pose/assets/67094418/6e750764-31c3-45f6-af4f-d67d51a40262)
  
 
 
 
 
## BibTeX

If you find our work and this repository useful. Please consider giving a star :star: and citation &#x1F4DA;.

```bibtex
@InProceedings{lang2022bam,
  title={Learning What Not to Segment: A New Perspective on Few-Shot Segmentation},
  author={Lang, Chunbo and Cheng, Gong and Tu, Binfei and Han, Junwei},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  pages={8057--8067},
  year={2022},
  }  
  
@article{lang2023bam,
	title={Base and Meta: A New Perspective on Few-shot Segmentation},
	author={Lang, Chunbo and Cheng, Gong and Tu, Binfei and Li, Chao and Han, Junwei},
	journal={IEEE Transactions on Pattern Analysis and Machine Intelligence (PAMI)},
	pages={1--18},
	year={2023},
	doi={10.1109/TPAMI.2023.3265865},
}
```
