# -*- coding: utf-8 -*-
"""
人体检测模块 - YOLOv8-Pose
========================
基于 YOLOv8-Pose 模型的人体检测与关键点提取
支持 RK3576/RK3588 等平台的 NPU 加速推理

使用方式:
  detector = HumanDetector(model_path='yolov8n-pose.rknn')
  person, ex, ey, keypoints = detector.process(frame)
  if person:
      ctrl.step(ex, ey)  # 控制机械臂
"""

import os
import numpy as np
import cv2

# RKNN 推理
from rknnlite.api import RKNNLite

# ============================================================
# YOLOv8-Pose 配置常量
# ============================================================

CLASSES = ['person']
NMS_THRESH = 0.4       # NMS 阈值
OBJECT_THRESH = 0.5     # 目标置信度阈值
INFER_SIZE = (640, 640)  # 推理输入尺寸

# 姿态可视化调色板
POSE_PALETTE = np.array([
    [255, 128, 0], [255, 153, 51], [255, 178, 102], [230, 230, 0],
    [255, 153, 255], [153, 204, 255], [255, 102, 255], [255, 51, 255],
    [102, 178, 255], [51, 153, 255], [255, 153, 153], [255, 102, 102],
    [255, 51, 51], [153, 255, 153], [102, 255, 102], [51, 255, 51],
    [0, 255, 0], [0, 0, 255], [255, 0, 0], [255, 255, 255]
], dtype=np.uint8)

KPT_COLOR = POSE_PALETTE[[16, 16, 16, 16, 16, 0, 0, 0, 0, 0, 0, 9, 9, 9, 9, 9, 9]]

SKELETON = [
    [16, 14], [14, 12], [17, 15], [15, 13], [12, 13],
    [6, 12], [7, 13], [6, 7], [6, 8], [7, 9],
    [8, 10], [9, 11], [2, 3], [1, 2], [1, 3],
    [2, 4], [3, 5], [4, 6], [5, 7]
]

LIMB_COLOR = POSE_PALETTE[[9, 9, 9, 9, 7, 7, 7, 0, 0, 0, 0, 0, 16, 16, 16, 16, 16, 16, 16]]


# ============================================================
# 辅助类与函数
# ============================================================

class DetectBox:
    """YOLOv8-Pose 检测结果"""
    def __init__(self, classId, score, xmin, ymin, xmax, ymax, keypoint):
        self.classId = classId
        self.score = score
        self.xmin = xmin
        self.ymin = ymin
        self.xmax = xmax
        self.ymax = ymax
        self.keypoint = keypoint  # shape: (51,) = 17 * 3 (x, y, conf)


def letterbox_resize(image, size, bg_color=114):
    """
    Letterbox 缩放，保持宽高比不变

    参数:
        image: BGR 图像
        size: 目标尺寸 (width, height)
        bg_color: 填充颜色（灰度值）

    返回:
        result_image: 缩放后图像
        aspect_ratio: 缩放比例
        offset_x, offset_y: 偏移量
    """
    if isinstance(image, str):
        image = cv2.imread(image)

    target_width, target_height = size
    image_height, image_width, _ = image.shape

    aspect_ratio = min(target_width / image_width, target_height / image_height)
    new_width = int(image_width * aspect_ratio)
    new_height = int(image_height * aspect_ratio)

    image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_AREA)

    result_image = np.ones((target_height, target_width, 3), dtype=np.uint8) * bg_color
    offset_x = (target_width - new_width) // 2
    offset_y = (target_height - new_height) // 2
    result_image[offset_y:offset_y + new_height, offset_x:offset_x + new_width] = image

    return result_image, aspect_ratio, offset_x, offset_y


def sigmoid(x):
    """Sigmoid 激活函数"""
    return 1 / (1 + np.exp(-x))


def softmax(x, axis=-1):
    """Softmax 激活函数（数值稳定版）"""
    exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


def iou(xmin1, ymin1, xmax1, ymax1, xmin2, ymin2, xmax2, ymax2):
    """计算两个检测框的 IOU"""
    xmin = max(xmin1, xmin2)
    ymin = max(ymin1, ymin2)
    xmax = min(xmax1, xmax2)
    ymax = min(ymax1, ymax2)

    innerWidth = max(0, xmax - xmin)
    innerHeight = max(0, ymax - ymin)
    innerArea = innerWidth * innerHeight

    area1 = (xmax1 - xmin1) * (ymax1 - ymin1)
    area2 = (xmax2 - xmin2) * (ymax2 - ymin2)
    total = area1 + area2 - innerArea

    return innerArea / total if total > 0 else 0


def nms(detect_result):
    """非极大值抑制（NMS）"""
    pred_boxes = []
    sorted_boxes = sorted(detect_result, key=lambda x: x.score, reverse=True)

    for i in range(len(sorted_boxes)):
        if sorted_boxes[i].classId != -1:
            pred_boxes.append(sorted_boxes[i])
            for j in range(i + 1, len(sorted_boxes)):
                if sorted_boxes[i].classId == sorted_boxes[j].classId:
                    iou_val = iou(
                        sorted_boxes[i].xmin, sorted_boxes[i].ymin,
                        sorted_boxes[i].xmax, sorted_boxes[i].ymax,
                        sorted_boxes[j].xmin, sorted_boxes[j].ymin,
                        sorted_boxes[j].xmax, sorted_boxes[j].ymax
                    )
                    if iou_val > NMS_THRESH:
                        sorted_boxes[j].classId = -1

    return pred_boxes


def process_feature(out, keypoints, index, model_w, model_h, stride, scale_w=1, scale_h=1):
    """
    后处理：从特征图解析检测框和关键点

    参数:
        out: 模型输出特征 (1, 65, N)
        keypoints: 关键点数据
        index: 关键点起始索引
        model_w, model_h: 特征图宽高
        stride: 特征图步长
    """
    xywh = out[:, :64, :]
    conf = sigmoid(out[:, 64:, :])
    results = []

    for h in range(model_h):
        for w in range(model_w):
            for c in range(len(CLASSES)):
                if conf[0, c, (h * model_w) + w] > OBJECT_THRESH:
                    xywh_ = xywh[0, :, (h * model_w) + w]
                    xywh_ = xywh_.reshape(1, 4, 16, 1)
                    data = np.array([i for i in range(16)]).reshape(1, 1, 16, 1)

                    xywh_ = softmax(xywh_, 2)
                    xywh_ = np.multiply(data, xywh_)
                    xywh_ = np.sum(xywh_, axis=2, keepdims=True).reshape(-1)

                    xywh_temp = xywh_.copy()
                    xywh_temp[0] = (w + 0.5) - xywh_[0]
                    xywh_temp[1] = (h + 0.5) - xywh_[1]
                    xywh_temp[2] = (w + 0.5) + xywh_[2]
                    xywh_temp[3] = (h + 0.5) + xywh_[3]

                    xywh_[0] = (xywh_temp[0] + xywh_temp[2]) / 2
                    xywh_[1] = (xywh_temp[1] + xywh_temp[3]) / 2
                    xywh_[2] = xywh_temp[2] - xywh_temp[0]
                    xywh_[3] = xywh_temp[3] - xywh_temp[1]
                    xywh_ = xywh_ * stride

                    xmin = (xywh_[0] - xywh_[2] / 2) * scale_w
                    ymin = (xywh_[1] - xywh_[3] / 2) * scale_h
                    xmax = (xywh_[0] + xywh_[2] / 2) * scale_w
                    ymax = (xywh_[1] + xywh_[3] / 2) * scale_h

                    keypoint = keypoints[..., (h * model_w) + w + index]
                    keypoint[..., 0:2] = keypoint[..., 0:2] // 1

                    box = DetectBox(
                        c,
                        conf[0, c, (h * model_w) + w],
                        xmin, ymin, xmax, ymax,
                        keypoint
                    )
                    results.append(box)

    return results


# ============================================================
# 人体检测器
# ============================================================

class HumanDetector:
    """
    YOLOv8-Pose 人体检测器

    使用 RKNN 进行 NPU 加速推理，
    检测人体并提取 17 个关键点。

    使用方式:
        detector = HumanDetector('yolov8n-pose.rknn')
        person, ex, ey, keypoints = detector.process(frame)
    """

    def __init__(self, model_path):
        """
        参数:
            model_path: YOLOv8-Pose RKNN 模型路径
        """
        self.model_path = model_path

        # 加载 RKNN 模型
        print(f"  [HumanDetector] 加载模型: {model_path}")
        self.rknn = RKNNLite(verbose=False)
        ret = self.rknn.load_rknn(model_path)
        if ret != 0:
            raise RuntimeError(f"RKNN 模型加载失败: {model_path}")

        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("RKNN 运行时初始化失败")

        print(f"  [HumanDetector] 模型已就绪，推理尺寸: {INFER_SIZE}")

    def detect(self, frame):
        """
        检测图像中的人体

        参数:
            frame: BGR 图像 (H, W, 3)

        返回:
            list of DetectBox: 检测到的人体列表（已 NMS）
        """
        # Letterbox 缩放
        letterbox_img, aspect_ratio, offset_x, offset_y = \
            letterbox_resize(frame, INFER_SIZE, bg_color=114)

        # BGR → RGB，增加 batch 维度
        infer_img = np.expand_dims(letterbox_img[..., ::-1], axis=0)

        # RKNN 推理
        results = self.rknn.inference(inputs=[infer_img])

        # 后处理
        outputs = []
        keypoints = results[3]

        for x in results[:3]:
            if x.shape[2] == 20:
                stride, idx = 8, 0
            elif x.shape[2] == 40:
                stride, idx = 16, 20 * 4 * 20 * 4
            elif x.shape[2] == 80:
                stride, idx = 32, 20 * 4 * 20 * 4 + 20 * 2 * 20 * 2
            else:
                continue

            feature = x.reshape(1, 65, -1)
            outputs += process_feature(
                feature, keypoints, idx,
                x.shape[3], x.shape[2], stride,
                scale_w=1, scale_h=1
            )

        # NMS
        pred_boxes = nms(outputs)

        # 坐标还原到原始图像
        for box in pred_boxes:
            box.xmin = int((box.xmin - offset_x) / aspect_ratio)
            box.ymin = int((box.ymin - offset_y) / aspect_ratio)
            box.xmax = int((box.xmax - offset_x) / aspect_ratio)
            box.ymax = int((box.ymax - offset_y) / aspect_ratio)

            # 关键点也还原
            kpts = box.keypoint.reshape(-1, 3)
            kpts[..., 0] = (kpts[..., 0] - offset_x) / aspect_ratio
            kpts[..., 1] = (kpts[..., 1] - offset_y) / aspect_ratio
            box.keypoint = kpts.reshape(-1)

        return pred_boxes

    def process(self, frame):
        """
        处理一帧图像，返回跟踪目标信息

        参数:
            frame: BGR 图像 (H, W, 3)

        返回:
            tuple: (person_box, ex, ey, keypoints)
              - person_box: (x1, y1, x2, y2) 或 None
              - ex: 水平偏差（人体中心 - 画面中心）像素
              - ey: 垂直偏差（人体中心 - 画面中心）像素
              - keypoints: (17, 3) 关键点数组 或 None
        """
        h, w = frame.shape[:2]
        boxes = self.detect(frame)

        if not boxes:
            return None, 0.0, 0.0, None

        # 选面积最大的检测框
        best_box = max(boxes, key=lambda b: (b.xmax - b.xmin) * (b.ymax - b.ymin))

        # 人体中心
        cx = (best_box.xmin + best_box.xmax) / 2
        cy = (best_box.ymin + best_box.ymax) / 2

        # 画面中心偏差
        ex = cx - w / 2
        ey = cy - h / 2

        # 关键点
        keypoints = best_box.keypoint.reshape(-1, 3)

        person_box = (int(best_box.xmin), int(best_box.ymin),
                      int(best_box.xmax), int(best_box.ymax))

        return person_box, ex, ey, keypoints

    def draw_results(self, frame, boxes):
        """
        在图像上绘制检测结果（关键点和骨架）

        参数:
            frame: BGR 图像
            boxes: DetectBox 列表

        返回:
            frame: 绘制后的图像
        """
        for box in boxes:
            x1, y1 = box.xmin, box.ymin
            x2, y2 = box.xmax, box.ymax

            # 绘制检测框
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, f"Person {box.score:.2f}", (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

            # 绘制关键点
            kpts = box.keypoint.reshape(-1, 3)
            for k, (x, y, conf) in enumerate(kpts):
                if x > 0 and y > 0:
                    cv2.circle(frame, (int(x), int(y)), 4,
                               [int(c) for c in KPT_COLOR[k]], -1, cv2.LINE_AA)

            # 绘制骨架
            for k, sk in enumerate(SKELETON):
                pos1 = (int(kpts[sk[0] - 1, 0]), int(kpts[sk[0] - 1, 1]))
                pos2 = (int(kpts[sk[1] - 1, 0]), int(kpts[sk[1] - 1, 1]))
                if pos1[0] > 0 and pos1[1] > 0 and pos2[0] > 0 and pos2[1] > 0:
                    cv2.line(frame, pos1, pos2,
                             [int(c) for c in LIMB_COLOR[k]], 2, cv2.LINE_AA)

        return frame

    def release(self):
        """释放 RKNN 资源"""
        if self.rknn:
            self.rknn.release()


# ============================================================
# 便捷函数：查找默认模型
# ============================================================

def find_model():
    """查找默认的 YOLOv8-Pose RKNN 模型"""
    candidates = [
        'yolov8n-pose.rknn',
        'yolov8_pose.rknn',
        '../人体跟随/models/yolov8n-pose.rknn',
        'model/yolov8n-pose.rknn',
    ]
    for c in candidates:
        if os.path.exists(c):
            return c
    return None