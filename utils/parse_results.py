# -*- coding: utf-8 -*-
"""
Created on Sat Jan  9 14:48:10 2021

@author: gkm0120
"""
import cv2
import numpy as np
import os.path as path
import argparse
import pickle as pkl
import json
import matplotlib.pyplot as plt
import math

kpid2name = ['snout', 'left_ear', 'right_ear', 'neck', 'tail_base']

# 计算处于不同时间点的两个box的iou
def bbox_iou(bbox1, bbox2):
    """
    计算两个 bounding box 的交并比
    :param bbox1: 第一个 bounding box，格式为 [x_min, y_min, x_max, y_max]
    :param bbox2: 第二个 bounding box，格式为 [x_min, y_min, x_max, y_max]
    :return: 交并比
    """
    x_left = max(bbox1[0], bbox2[0])
    y_top = max(bbox1[1], bbox2[1])
    x_right = min(bbox1[2], bbox2[2])
    y_bottom = min(bbox1[3], bbox2[3])

    if x_right < x_left or y_bottom < y_top:
        return 0.0

    intersection_area = (x_right - x_left) * (y_bottom - y_top)
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - intersection_area

    iou = intersection_area / union_area
    return iou

def point_speed(X, Y):
    pass

def center(points):
    """计算给定矩阵的质心"""
    if type(points)== np.ndarray:
        points = points.tolist()
    x = (points[0] + points[2]) / 2.0
    y = (points[1] + points[3]) / 2.0
    return np.array([np.float32(x), np.float32(y)], np.float32)

class Rat():
    """Pedestrian（行人）
    每个行人都由ROI，ID和卡尔曼过滤器组成，因此我们创建了一个步行者类来保存对象状态
    """

    def __init__(self, id, start_id,  bbox_and_key):
        """使用跟踪窗口坐标初始化行人对象"""
        # 设置ROI区域
        self.id = int(id)
        bbox_and_key = {k:v.tolist() for k,v  in bbox_and_key.items()}
        self.current_bbox = bbox_and_key['bbox'][:-1]
        self.history = [{}] * (start_id+1) + [bbox_and_key]
        self.center = center(self.current_bbox)

    def __del__(self):
        print("Pedestrian %d destroyed" % self.id)

    def update(self, bbox_and_key, frame=None):
        # print ("updating %d " % self.id)
        bbox_and_key = {k:v.tolist() for k,v  in bbox_and_key.items()}
        self.history.append(bbox_and_key)
        current_bbox = bbox_and_key.get('bbox', [])
        if current_bbox !=[]:
            self.current_bbox = current_bbox[:-1] # 若更新帧无效， self.current_bbox保持不变。
            self.center = center(self.current_bbox)
        # 在bbox中心画圈。
        if frame is not None:
            cv2.circle(frame, (int(self.center.tolist()[0]),int(self.center.tolist()[1])), 4,  (11, (self.id + 1) * 25 + 1), -1)

def is_stable(idx, N, valid_list):
    """
    判断从idx起的N帧内，稳定检出是否连续
    """
    if valid_list[idx + N] - valid_list[idx] == N:
        return True
    else:
        return False

def tack_rats(video_path, results_file, save_dir):
    with open(results_file, 'rb') as f: # 加载检测结果。
        results = pkl.load(f)

    bbox_keypoint_results = results['result']
    keypoint_id2name = results['dataset_info']
    # fps = results['fps']
    camera = cv2.VideoCapture(video_path)  # 加载视频

    cv2.namedWindow("surveillance",cv2.WINDOW_NORMAL)
    rats = {}
    # fourcc = cv2.VideoWriter_fourcc(*'XVID')
    # out = cv2.VideoWriter('output.avi', fourcc, 20.0, (640, 480))


    # 统计每帧图像检出的bbox个数。将bbox个数最一致的作为真实的目标个数。true_obj_num
    obj_num_dict = {}
    for frame_id, fram_res in enumerate(bbox_keypoint_results):
        obj_num = len(fram_res)
        if not obj_num_dict.get(obj_num):
            obj_num_dict[obj_num] = [frame_id]
        else:
            obj_num_dict[obj_num].append(frame_id)

    true_obj_num = 0
    frame_num = 0
    for k, v in obj_num_dict.items():
        if len(v) > frame_num:
            frame_num = len(v)
            true_obj_num = k
    # 有效检出所有obj的frame_id列表。
    valid_framid_list = obj_num_dict[true_obj_num]

    is_initialized = False # 初始化追踪对象标志位。
    for frame_id, result in enumerate(bbox_keypoint_results):
        print(" -------------------- FRAME %d --------------------" % frame_id)
        grabbed, frame = camera.read()
        if (grabbed is False):
            print("failed to grab frame.")
            break

        # 在有效帧中，寻找一个好的初始帧。该帧满足以下条件：
        # 1）无漏检。2）从此帧开始的后续 N 帧图片内，都不出现漏检。
        if frame_id in valid_framid_list:
            if not is_initialized:
                if is_stable(frame_id, 10, valid_framid_list):
                    for obj_id in range(true_obj_num):
                        rats[obj_id] = Rat(obj_id, frame_id, result[obj_id])
                    is_initialized = True  # 完成跟踪对象初始化。
                    print('initialized!!!!!!')

            else: # 有效帧，且已经完成初始化。判断bbox的分布。
                # iou_list = np.zeros((true_obj_num, true_obj_num))
                for rat_id, rat in rats.items():
                    max_iou = 0
                    current = {}
                    for new_id, new_res in enumerate(result):
                        new_bbox = new_res['bbox'][:-1]
                        iou = bbox_iou(new_bbox, rat.current_bbox)
                        if iou>max_iou:
                            max_iou=iou
                            current = new_res
                    if max_iou!=0: # 不允许过大的跨度
                        rat.update(current, frame)


        elif is_initialized:  # 在无效帧中，若已经进行初始化，则为history添加空列表。
            for k,v in rats.items():
                v.update({}, frame)

        else: # 无效帧，且并未初始化，继续循环。
            continue
        cv2.imshow("surveillance", frame)  # 窗口显示结果
        # out.write(frame)
        if cv2.waitKey(110) & 0xff == 27:
            break
    # out.release()
    camera.release()

    RATS = {k:v.history for k,v in rats.items()}

    with open(save_dir, 'w') as f:
        temp = json.dumps(RATS)
        f.write(temp)

    return RATS


def sample(x, interval,fps): # 根据视频fps和绘图的时间间隔对数据点进行抽样。
    return x[:,::int(fps*interval)]


def draw_trace_lines(x, y, color='red'):
    for t in range(x.shape[1]):
        plt.plot(x[:,t], y[:,t], 'o', markersize=1, color=color, alpha=t/x.shape[1])
        skeleton1 = [0, 1, 2, 0]
        skeleton2 = [1, 3, 4]
        skeleton3 = [2,3]
        X1 = [x[kp_id,t] for kp_id in skeleton1]
        Y1 = [y[kp_id, t] for kp_id in skeleton1]
        X2 = [x[kp_id, t] for kp_id in skeleton2]
        Y2 = [y[kp_id, t] for kp_id in skeleton2]
        X3 = [x[kp_id, t] for kp_id in skeleton3]
        Y3 = [y[kp_id, t] for kp_id in skeleton3]
        plt.plot(X1, Y1, linewidth=1, color=color, alpha=t/x.shape[1])
        plt.plot(X2, Y2, linewidth=1, color=color, alpha=t/x.shape[1])
        plt.plot(X3, Y3, linewidth=1, color=color, alpha=t/x.shape[1])
    plt.xlabel('x')
    plt.ylabel('y')


def parse_speed(x, y, delta_t=0.1):
    v_x = np.diff(x) /delta_t
    v_y = np.diff(y) /delta_t
    speed = np.sqrt(v_x ** 2 + v_y ** 2)
    mean = np.mean(speed, axis=1)
    std = np.std(speed, axis=1)
    threshold = mean + 3 * std
    #filter abnormal data
    valid_ts = []
    for kpid in range(5):
        valid_t = np.argwhere((speed[kpid] <= threshold[kpid]) == True).tolist()
        valid_t = [elem for sublist in valid_t for elem in sublist]
        valid_ts.append(set(valid_t))

    def intersection(sets):
        if not sets:
            return set()
        result = sets[0]
        for s in sets[1:]:
            result &= s
        return result

    valid_ts = intersection(valid_ts)
    speed = speed[:,list(valid_ts)]

    plt.figure()

    time_axis = [delta_t*i for i in range(speed.shape[1])]

    for kp_id in range(5):
        plt.plot(time_axis, speed[kp_id], label=kpid2name[kp_id])
    plt.xlabel('Time (sec)')
    plt.ylabel('Speed (pixel/s)')
    plt.legend(loc=1)
    plt.savefig('single_speed.pdf', dpi=600)
    plt.show()

# 计算向量夹角

def point_position(A, B, C):
    # 计算向量 AB 和 AC 的坐标表示
    x1,x2,x3 = A[0],B[0],C[0]
    y1,y2,y3 = A[1],B[1],C[1]


    BA_x = x2 - x1
    BA_y = y2 - y1
    CA_x = x3 - x1
    CA_y = y3 - y1
    # 计算叉积
    cross_product = BA_x * CA_y - BA_y * CA_x
    # 判断位置
    return cross_product

def cal_angle(point_a, point_b,  point_c ,point_d):
    x1, x2, x3, x4 = point_a[0], point_b[0], point_c[0], point_d[0] # 点a、b、c的x坐标
    y1, y2, y3, y4 = point_a[1], point_b[1], point_c[1], point_d[1]  # 点a、b、c的y坐标
    # 求出斜率
    k1 = (y2 - y1) / (float(x2 - x1))
    if (x2-x1) == 0:
        x2
    k2 = (y4 - y3) / (float(x4 - x3))
    if (x2-x1) == 0:
        x2
    # 方向向量
    x = np.array([1, k1])
    y = np.array([1, k2])
    # 模长
    Lx = np.sqrt(x.dot(x))
    Ly = np.sqrt(y.dot(y))
    # 根据向量之间求其夹角并四舍五入
    # Cobb = int(math.fabs(np.arctan((k1 - k2) / (float(1 + k1 * k2))) * 180 / np.pi))
    Cobb = (np.arccos(x.dot(y) / (float(Lx * Ly))) * 180 / np.pi)
    return Cobb
# 判断直线外一点处于直线的左侧或右侧。叉姬大于零在右侧。

def parse_head_degree(x_s, y_s, delta_t=0.1, color='red'):
    angle_list = []
    for t in range(x_s.shape[1]):
        x_mouse = x_s[0][t]
        y_mouse = y_s[0][t]
        x_neck = x_s[3][t]
        y_neck = y_s[3][t]
        x_tail = x_s[4][t]
        y_tail = y_s[4][t]
        # 计算头颈、颈尾夹角。
        angle = cal_angle([x_mouse,y_mouse], [x_neck,y_neck],[x_neck,y_neck], [x_tail,y_tail])
        # 计算偏转方向
        cross_product = point_position([x_neck,y_neck], [x_tail,y_tail],[x_mouse,y_mouse])
        angle = angle * cross_product / math.fabs(cross_product)
        angle_list.append(angle)

    angles = np.array(angle_list)
    angles = angles[np.logical_not(np.isnan(angles))]
    angles[angles<-100] = angles[angles<-100]  + 30
    angles[angles>100] = angles[angles>100] - 30

    time_axis = [delta_t*i for i in range(angles.shape[0])]
    label = 1 if color=='red' else 2
    plt.plot(time_axis, angles, color=color, label='mouse_' + str(label))
    plt.axhline(y=0, color='blue', linewidth=2, linestyle='--')
    plt.xlabel('Time (sec)')
    plt.ylabel('Head turning (degree)')

def parse_mouse2tail(X,Y, delta_t=0.1):
    mousex1 = X[0][0]
    mousex2 = X[1][0]
    mousey1 = Y[0][0]
    mousey2 = Y[1][0]
    tailx1 = X[0][4]
    tailx2 = X[1][4]
    taily1 = Y[0][4]
    taily2 = Y[1][4]

    mouse2tail12 = np.sqrt((mousex1-tailx2)**2 + (mousey1-taily2)**2)
    mouse2tail21 = np.sqrt((mousex2-tailx1)**2 + (mousey2-taily1)**2)
    mouse2mouse = np.sqrt((mousex1-mousex2)**2 + (mousey1-mousey2)**2)
    time_axis = [delta_t*i for i in range(mouse2mouse.shape[0])]

    return mouse2tail21, mouse2tail12, mouse2mouse, time_axis

def parse(rats_json, time=0.3, fps=30):
    colors = ['red','blue']
    X, Y = [], []
    for rat_id, res in rats_json.items():
        start_id = 0
        fra_num = len(res)
        x = np.empty((5,fra_num))
        y = np.empty((5,fra_num))
        for frame_id, bbox_key in enumerate(res):
            if bbox_key != {}:
                if start_id == 0:
                    start_id = frame_id
                for kp_id in range(5):
                    x[kp_id][frame_id] = bbox_key['keypoints'][kp_id][0]
                    y[kp_id][frame_id] = bbox_key['keypoints'][kp_id][1]
            else:
                for kp_id in range(5):
                    x[kp_id][frame_id] = None
                    y[kp_id][frame_id] = None
        for frame_id in range(fra_num): #将空值补全为前一个邻近有效值。
            if frame_id<start_id:
                x[:,frame_id] = x[:,start_id]
                y[:, frame_id] = y[:, start_id]
            elif np.isnan(x[0,frame_id]):
                x[:, frame_id] = x[:, frame_id-1]
                y[:, frame_id] = y[:, frame_id - 1]
        x[x<0] = 0
        y[y<0] = 0
        x_s = sample(x, time, fps)
        y_s = sample(y, time, fps)
        X.append(x_s)
        Y.append(y_s)
        # 绘制轨迹追踪
        # plt.figure(1)
        # draw_trace_lines(x_s, y_s, colors[int(rat_id)])


        # # 绘制速度分析曲线
        # plt.figure(2)
        # parse_speed(x_s, y_s, time)

        # # 绘制头部躯干偏转角
        plt.figure(3)
        parse_head_degree(x_s, y_s, time, colors[int(rat_id)])

        ## 绘制mouse->tail


    plt.figure(1)
    plt.savefig('motion_trace.png', dpi=600)

    plt.figure(3)
    plt.legend()
    plt.savefig('head_turning.png', dpi=600)


    plt.figure(4)
    mouse2tail21, mouse2tail12, mouse2mouse, time_axis = parse_mouse2tail(X,Y,time)
    plt.plot(time_axis, mouse2tail12, color='red', label='mouse_1->mouse_2')
    plt.plot(time_axis, mouse2tail21, color='blue', label='mouse_2->mouse_1')
    plt.legend()
    plt.ylabel('Distance (pixel)')
    plt.xlabel('Time (sec)')
    plt.savefig('genital_touch_5_8.png', dpi=600)


    plt.figure(5)
    plt.plot(time_axis, mouse2mouse, color='red', label='mouse_to_mouse')
    plt.ylabel('Distance (pixel)')
    plt.xlabel('Time (sec)')
    plt.savefig('facial_touch_5_8.png', dpi=600)
    plt.show()
    pass

    pass
if __name__ == "__main__":
    video_path = "D:\Learning\pose estmation\论文视频\GH010190 (1).mp4"
    result_path = 'results_dual_90_video.pkl'
    SVAE_JSON_FILE = result_path.split('.')[0] + '.json'
    # RATS_json = tack_rats(video_path, result_path, SVAE_JSON_FILE) # 分析bbox序列并对小鼠进行ReID,返回整理好的bbox_keypoints 序列。
    with open(SVAE_JSON_FILE,'r') as f:
        RATS = json.load(f)
    parse(RATS)



