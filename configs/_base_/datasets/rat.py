dataset_info = dict(
    paper_info=dict(
        author='Cao, Jinkun and Tang, Hongyang and Fang, Hao-Shu and '
        'Shen, Xiaoyong and Lu, Cewu and Tai, Yu-Wing',
        title='Cross-Domain Adaptation for Animal Pose Estimation',
        container='The IEEE International Conference on '
        'Computer Vision (ICCV)',
        year='2019',
        homepage='https://sites.google.com/view/animal-pose/',
    ),
    dataset_name='RAT',

    keypoint_info={
        0:
        dict(
            name='mouth', id=0, color=[51, 153, 255],swap=''),
        1:
        dict(
            name='left_ear', id=1, color=[0, 255, 0],swap='right_ear'),
        2:
        dict(
            name='right_ear', id=2, color=[255, 128, 0],swap='left_ear'),
        3:
        dict(
            name='neck', id=3, color=[51, 153, 255], swap=''),
        4:
        dict(
            name='tailstock', id=4, color=[51, 153, 255], swap=''),
    },
    
    skeleton_info={
        0: dict(link=('left_ear', 'right_ear'), id=0, color=[51, 153, 255]),
        1: dict(link=('mouth', 'left_ear'), id=1, color=[0, 255, 0]),
        2: dict(link=('mouth', 'right_ear'), id=2, color=[255, 128, 0]),
        3: dict(link=('left_ear', 'neck'), id=3, color=[0, 255, 0]),
        4: dict(link=('right_ear', 'neck'), id=4, color=[255, 128, 0]),
        5: dict(link=('neck', 'tailstock'), id=5, color=[51, 153, 255]),
    },
    joint_weights=[
        1., 1., 1., 1., 1.
    ],
    sigmas=[
        0.001, 0.001, 0.001,  0.001, 0.001, # 多人标准相同数据集得到的标准差，数据集由我自己标注，所以设置为0.001相当于没有
    ])
