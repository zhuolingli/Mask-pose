# 命令行执行： python labelme2coco.py --input_dir images --output_dir coco --labels labels.txt
# 输出文件夹必须为空文件夹

import argparse
import collections
import datetime
import glob
import json
import os
import os.path as osp
import sys
import uuid
import imgviz
import numpy as np
import labelme
from sklearn.model_selection import train_test_split
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon

try:
    import pycocotools.mask
except ImportError:
    print("Please install pycocotools:\n\n    pip install pycocotools\n")
    sys.exit(1)

# 数据集基础信息定义，注意标签序号要保持一致
keypoints_info = [
    'mouth',
    'left_ear',
    'right_ear',
    'neck',
    'tailstock',
]
kpt2id = dict(zip((keypoints_info),range(len(keypoints_info))))

skeleton_info = [[0, 1],  # 骨架结构
                 [0, 2],
                 [1, 3],
                 [2, 3],
                 [3, 4]]


# labelme文件转coco格式
def to_coco(args, label_files, train):
    # 创建 总标签data
    now = datetime.datetime.now()
    data = dict(
        info=dict(
            description=None,
            url=None,
            version=None,
            year=now.year,
            contributor=None,
            date_created=now.strftime("%Y-%m-%d %H:%M:%S.%f"),
        ),
        licenses=[dict(url=None, id=0, name=None, )],
        images=[
            # license, url, file_name, height, width, date_captured, id
        ],
        type="instances",
        annotations=[
            # segmentation, area, iscrowd, image_id, bbox, category_id, id
        ],
        categories=[
            # supercategory, id, name
        ],
    )

    # 创建一个 {类名 : id} 的字典，并保存到 总标签data 字典中。
    class_name_to_id = {} # 这里存放的是实例类别。即背景、老鼠，猪牛羊等。实例的不同关键点种类隶属于实例类别。
    for i, line in enumerate(open(args.labels).readlines()):
        class_id = i - 1  # starts with -1
        class_name = line.strip()  # strip() 方法用于移除字符串头尾指定的字符(默认为空格或换行符)或字符序列。
        if class_id == -1:
            assert class_name == "__ignore__"  # background:0, class1:1, ,,
            continue
        class_name_to_id[class_name] = class_id
        keypoints = None
        if class_id > 0:
            keypoints = keypoints_info
        data["categories"].append(
            dict(supercategory=None, id=class_id, name=class_name, keypoints=keypoints )
        )

    if train:
        out_ann_file = osp.join(args.output_dir, "annotations", "instances_train2017.json")
    else:
        out_ann_file = osp.join(args.output_dir, "annotations", "instances_val2017.json")

    for image_id, filename in enumerate(label_files):

        label_file = labelme.LabelFile(filename=filename)
        base = osp.splitext(osp.basename(filename))[0]  # 文件名不带后缀
        if train:
            out_img_file = osp.join(args.output_dir, "train2017", base + ".jpg")
        else:
            out_img_file = osp.join(args.output_dir, "val2017", base + ".jpg")

        print("| ", out_img_file)

        # ************************** 对图片的处理开始 *******************************************
        # 将标签文件对应的图片进行保存到对应的 文件夹。train保存到 train2017/ test保存到 val2017/
        img = labelme.utils.img_data_to_arr(label_file.imageData)  # .json文件中包含图像，用函数提出来
        imgviz.io.imsave(out_img_file, img)  # 将图像保存到输出路径

        # ************************** 对图片的处理结束 *******************************************

        # ************************** 对标签的处理开始 *******************************************
        data["images"].append(
            dict(
                license=0,
                url=None,
                file_name='/'.join(osp.relpath(out_img_file, osp.dirname(out_ann_file)).split('\\'))[3:],
                #   out_img_file = "/coco/train2017/1.jpg"
                #   out_ann_file = "/coco/annotations/annotations_train2017.json"
                #   osp.dirname(out_ann_file) = "/coco/annotations"
                #   file_name = ..\train2017\1.jpg   out_ann_file文件所在目录下 找 out_img_file 的相对路径
                height=img.shape[0],
                width=img.shape[1],
                date_captured=None,
                id=image_id,
            )
        )
        masks = {}  # for area
        bboxes = {}
        areas = {}
        segmentations = collections.defaultdict(list)  # for segmentation
        splited_keypoints = collections.defaultdict(dict)

        # 对于一张img的JSON文件处理，先从mask开始。
        for shape in label_file.shapes: # label_file.shapes 表示这一json文件中所有的注释。
            points = shape["points"]
            label = shape["label"]
            group_id = shape.get("group_id")
            shape_type = shape.get("shape_type", "polygon")
            # if shape_type != 'polygon': # 只将多边形注释计入segmentation
            #     continue
            # 根据多边形得到mask
            mask = labelme.utils.shape_to_mask(
                img.shape[:2], points, shape_type
            )

            if group_id is None:
                group_id = uuid.uuid1()

            instance = (label, group_id)
            if instance in masks:
                masks[instance] = masks[instance] | mask
            else:
                masks[instance] = mask
            # 上述代码将分割、关键点注释都归类为mask，且分别存储为独立的instance。但实质上，只有分割注释
            # 才是我们需要的instance。上述操作只是方便后续用masks字典进行可视化。
            if shape_type != 'polygon': # 只将多边形注释计入segmentation
                continue
            # 判断该img图片中的关键点是否处于该mask多边形中。
            polygon = Polygon(points) # 根据mask的各点坐标创建多边形
            for shape in label_file.shapes:
                if shape['shape_type'] == 'point': # 判断shape类型是否为关键点。
                    label = shape["label"]
                    key_point = Point(shape['points'][0]) # 创建 Point对象
                    if polygon.contains(key_point):
                        splited_keypoints[instance].update({label:shape['points'][0]})

            # 根据实例的mask得到其bbox
            mask = np.asfortranarray(mask.astype(np.uint8))
            mask = pycocotools.mask.encode(mask)

            bbox = pycocotools.mask.toBbox(mask).flatten().tolist()   # bbox 左上，长宽
            area = float(pycocotools.mask.area(mask))# 这里计算的是 mask的area ，在mmpose中需要bbox的 area.
            area = float(bbox[2] * bbox[3])

            areas[instance] = area
            bboxes[instance] = bbox

            # mask坐标转为list类型存储
            points = np.asarray(points).flatten().tolist()
            segmentations[instance].append(points)

        segmentations = dict(segmentations)
        splited_keypoints = dict(splited_keypoints)

        # 一张img中可能包含多个实例对象，每个实例作为annotations的一个条目。
        for instance in splited_keypoints.keys(): # 每一个实例的注释为一个条目。一个图片可能包含多个实例。
            cls_name, group_id = instance
            if cls_name not in class_name_to_id:
                continue
            cls_id = class_name_to_id[cls_name]

            bbox = bboxes[instance]  # bbox 左上，长宽
            area = areas[instance]
            keypoints = splited_keypoints.get(instance) # 可能有mask，但是没有标记kpt
            np_points = np.zeros([len(keypoints_info), 3], dtype=np.float32)
            if keypoints: # 若某个实例对象的kpt不可见，则kpt数组直接置零。
                for label, point in keypoints.items():
                    kpt_id = kpt2id[label]
                    np_points[kpt_id, 0] = float(point[0])
                    np_points[kpt_id, 1] = float(point[1])
                    np_points[kpt_id, 2] = 2
            num_keypoints = int(sum(np_points[:, 2] > 0))
            keypoints = np_points.reshape(-1).tolist()

            data["annotations"].append(
                dict(
                    id=len(data["annotations"]),
                    image_id=image_id,
                    category_id=cls_id, # 动物的类别。背景类的id为0，cls_id从1开始。
                    segmentation=segmentations[instance],
                    area=area,
                    bbox=bbox,
                    num_keypoints=num_keypoints,
                    keypoints=keypoints,
                    iscrowd=0,
                )
            )
        # ************************** 对标签的处理结束 *******************************************

        # ************************** 可视化的处理开始 *******************************************
        if not args.noviz:
            class_names = ['rat'] + keypoints_info
            all_class_name_to_id = dict(zip((class_names),range(len(class_names))))

            labels, captions, masks = zip(
                *[(all_class_name_to_id[cnm], cnm, msk) for (cnm, gid), msk in masks.items()]
            )
            viz = imgviz.instances2rgb(
                image=img,
                labels=labels,
                masks=masks,
                captions=captions,
                font_size=15,
                line_width=2,
            )
            out_viz_file = osp.join(
                args.output_dir, "visualization", base + ".jpg"
            )
            imgviz.io.imsave(out_viz_file, viz)
        # ************************** 可视化的处理结束 *******************************************

    with open(out_ann_file, "w") as f:  # 将每个标签文件汇总成data后，保存总标签data文件
        json.dump(data, f)


# 主程序执行
def main():
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--input_dir", default='images', help="input annotated directory")
    parser.add_argument("--output_dir", default='coco', help="output dataset directory")
    parser.add_argument("--labels", default='labels.txt', help="labels file", )
    parser.add_argument("--noviz", help="no visualization", action="store_true")
    args = parser.parse_args()

    if osp.exists(args.output_dir):
        print("Output directory already exists:", args.output_dir)
        sys.exit(1)
    os.makedirs(args.output_dir)
    print("| Creating dataset dir:", args.output_dir)
    if not args.noviz:
        os.makedirs(osp.join(args.output_dir, "visualization"))

    # 创建保存的文件夹
    if not os.path.exists(osp.join(args.output_dir, "annotations")):
        os.makedirs(osp.join(args.output_dir, "annotations"))
    if not os.path.exists(osp.join(args.output_dir, "train2017")):
        os.makedirs(osp.join(args.output_dir, "train2017"))
    if not os.path.exists(osp.join(args.output_dir, "val2017")):
        os.makedirs(osp.join(args.output_dir, "val2017"))

    # 获取目录下所有的.jpg文件列表
    img_files = glob.glob(osp.join(args.input_dir, "*.jpg"))
    print('| Image number: ', len(img_files))



    # 获取目录下所有的joson文件列表
    label_files = glob.glob(osp.join(args.input_dir, "*.json"))
    print('| Json number: ', len(label_files))

    # json清洗，取json和img的交集。
    elements_img = set([img.split('.')[0] for img in img_files])
    elements_json = set([json.split('.')[0] for json in label_files])
    elements = elements_img.intersection(elements_json)

    img_files = [element + '.jpg' for element in elements]
    label_files = [element + '.json' for element in elements]


    # img_files:待划分的样本特征集合    label_files:待划分的样本标签集合    test_size:测试集所占比例
    # x_train:划分出的训练集特征      x_test:划分出的测试集特征     y_train:划分出的训练集标签    y_test:划分出的测试集标签
    x_train, x_test, y_train, y_test = train_test_split(img_files, label_files, test_size=0.3)
    print("| Train number:", len(y_train), '\t Value number:', len(y_test))

    # 把训练集标签转化为COCO的格式，并将标签对应的图片保存到目录 /train2017/
    print("—" * 50)
    print("| Train images:")
    to_coco(args, y_train, train=True)

    # 把测试集标签转化为COCO的格式，并将标签对应的图片保存到目录 /val2017/
    print("—" * 50)
    print("| Test images:")
    to_coco(args, y_test, train=False)


if __name__ == "__main__":
    print("—" * 50)
    main()
    print("—" * 50)