# -*- coding: utf-8 -*-
"""
人脸检测与姿态估计模块 (板端版本)
=============================
使用 Haar Cascade 进行人脸检测（无需模型转换，开箱即用）
使用 Hopenet 进行头部姿态估计（Yaw/Pitch/Roll）

支持的推理后端（按优先级自动选择）:
  1. RKNNLite (开发板 NPU 加速)
  2. ONNX Runtime (PC 端)
  3. PyTorch (PC 端回退) - 需要 hopenet.py 在同一目录

模型搜索路径:
  - model/hopenet_i8.rknn
  - model/hopenet.onnx
  - hopenet_i8.rknn (当前目录)

检测输出:
  - 人脸框 (x1, y1, x2, y2)
  - 头部姿态角度 (yaw, pitch, roll)
  - 人脸中心像素坐标
"""

import os
import sys
import cv2
import numpy as np

# ============================================================
# RKNN 后端
# ============================================================
try:
    from rknnlite.api import RKNNLite
    HAS_RKNN = True
except ImportError:
    HAS_RKNN = False

# ============================================================
# ONNX Runtime 后端
# ============================================================
try:
    import onnxruntime as ort
    HAS_ONNX = True
except ImportError:
    HAS_ONNX = False


# ============================================================
# Softmax 函数
# ============================================================

def softmax(x):
    """Softmax 激活函数 (数值稳定版)"""
    x_max = np.max(x, axis=-1, keepdims=True)
    exp = np.exp(x - x_max)
    return exp / exp.sum(axis=-1, keepdims=True)


def decode_angles(yaw_prob, pitch_prob, roll_prob):
    """
    将 Hopenet 66-bin 概率分布解码为角度
    公式: sum(prob_i * i) * 3 - 99  (范围 -99° ~ +99°)
    """
    idx = np.arange(66, dtype=np.float32)
    yaw_angle = float((yaw_prob * idx).sum() * 3 - 99)
    pitch_angle = float((pitch_prob * idx).sum() * 3 - 99)
    roll_angle = float((roll_prob * idx).sum() * 3 - 99)
    return yaw_angle, pitch_angle, roll_angle


# ============================================================
# 人脸检测器 - Haar Cascade（零依赖，即时可用）
# ============================================================

class FaceDetector:
    """基于 OpenCV Haar Cascade 的轻量级人脸检测器"""

    def __init__(self, min_face_size=80, scale_factor=1.1, min_neighbors=5):
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        if not os.path.exists(cascade_path):
            cascade_path = 'haarcascade_frontalface_default.xml'

        self.detector = cv2.CascadeClassifier(cascade_path)
        self.min_face_size = min_face_size
        self.scale_factor = scale_factor
        self.min_neighbors = min_neighbors

        if self.detector.empty():
            print("[WARN] Haar Cascade 加载失败，请检查 OpenCV 安装")
        else:
            print(f"  [FaceDetector] Haar Cascade (min_size={min_face_size})")

    def detect(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)
        faces = self.detector.detectMultiScale(
            gray,
            scaleFactor=self.scale_factor,
            minNeighbors=self.min_neighbors,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        result = []
        h, w = frame.shape[:2]
        for (x, y, fw, fh) in faces:
            x2, y2 = x + fw, y + fh
            x2 = min(x2, w - 1)
            y2 = min(y2, h - 1)
            if x2 > x and y2 > y:
                result.append((x, y, x2, y2))
        return result


# ============================================================
# 人脸姿态估计器 - Hopenet
# ============================================================

class PoseEstimator:
    """
    Hopenet 头部姿态估计器

    自动选择推理后端:
      - RKNNLite: 开发板 NPU 加速（最快）
      - ONNX Runtime: PC 端（中速）
      - PyTorch: PC 端回退（最慢）

    输入: 224x224 RGB 人脸图像
    输出: (yaw, pitch, roll) 角度，单位：度
    """

    def __init__(self, model_path=None):
        self.backend = None
        self.model_path = model_path

        # 自动检测可用模型
        rknn_path = model_path or self._find_model('.rknn')
        onnx_path = self._find_model('.onnx')
        pkl_path = self._find_weight()

        if HAS_RKNN and rknn_path and os.path.exists(rknn_path):
            self._load_rknn(rknn_path)
        elif HAS_ONNX and onnx_path and os.path.exists(onnx_path):
            self._load_onnx(onnx_path)
        elif pkl_path and os.path.exists(pkl_path):
            self._load_torch(pkl_path)
        else:
            raise RuntimeError(
                "无法加载 Hopenet 模型！\n"
                f"  RKNN: {rknn_path or 'N/A'} (需要 RKNNLite)\n"
                f"  ONNX: {onnx_path or 'N/A'} (需要 onnxruntime)\n"
                f"  权重: {pkl_path or 'N/A'} (需要 torch + hopenet.py)\n"
                "  请先运行 onnx2rknn.py 或确保模型文件在 model/ 目录下"
            )

    def _find_model(self, ext):
        """查找可用模型文件（板端扁平目录）"""
        candidates = [
            f'model/hopenet_i8{ext}',
            f'model/hopenet{ext}',
            f'hopenet_i8{ext}',
            f'hopenet_fp{ext}',
            f'hopenet{ext}',
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _find_weight(self):
        """查找 PyTorch 权重文件"""
        candidates = [
            'model/hopenet_robust_alpha1.pkl',
            'hopenet_robust_alpha1.pkl',
        ]
        for c in candidates:
            if os.path.exists(c):
                return c
        return None

    def _load_rknn(self, path):
        self.rknn = RKNNLite()
        ret = self.rknn.load_rknn(path)
        if ret != 0:
            raise RuntimeError(f"RKNN 模型加载失败: {path}")
        ret = self.rknn.init_runtime()
        if ret != 0:
            raise RuntimeError("RKNN 运行时初始化失败")
        self.backend = 'rknn'
        print(f"  [PoseEstimator] RKNN: {os.path.basename(path)} (NPU)")

    def _load_onnx(self, path):
        try:
            self.session = ort.InferenceSession(path, providers=['CPUExecutionProvider'])
        except Exception:
            self.session = ort.InferenceSession(path)
        self.backend = 'onnx'
        size_mb = os.path.getsize(path) / 1024**2
        print(f"  [PoseEstimator] ONNX: {os.path.basename(path)} ({size_mb:.1f}MB)")

    def _load_torch(self, path):
        """加载 PyTorch 模型 - 板端扁平目录，hopenet.py 在同一目录"""
        from hopenet import Hopenet
        import torchvision
        import torch

        model = Hopenet(torchvision.models.resnet.Bottleneck, [3, 4, 6, 3], 66)
        state_dict = torch.load(path, map_location='cpu', weights_only=False)
        model.load_state_dict(state_dict)
        model.eval()
        self.torch_model = model
        self.backend = 'torch'
        print(f"  [PoseEstimator] PyTorch: {os.path.basename(path)}")

    def estimate(self, face_roi):
        if face_roi.size == 0:
            return 0.0, 0.0, 0.0

        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        face_resized = cv2.resize(face_rgb, (224, 224))

        if self.backend == 'rknn':
            face_input = np.expand_dims(face_resized, 0).astype(np.float32)
            outputs = self.rknn.inference(inputs=[face_input])
            raw = [outputs[0][0], outputs[1][0], outputs[2][0]]

        elif self.backend == 'onnx':
            face_input = np.expand_dims(face_resized, 0).astype(np.float32)
            face_input = face_input.transpose(0, 3, 1, 2)
            outputs = self.session.run(None, {'input': face_input})
            raw = outputs

        elif self.backend == 'torch':
            import torch
            import torch.nn.functional as F

            face_rgb_norm = face_resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            face_norm = (face_rgb_norm - mean) / std
            tensor = torch.from_numpy(face_norm).permute(2, 0, 1).unsqueeze(0)
            with torch.no_grad():
                yaw_out, pitch_out, roll_out = self.torch_model(tensor)
            yaw_prob_t = F.softmax(yaw_out, dim=1).numpy()[0]
            pitch_prob_t = F.softmax(pitch_out, dim=1).numpy()[0]
            roll_prob_t = F.softmax(roll_out, dim=1).numpy()[0]
            return decode_angles(yaw_prob_t, pitch_prob_t, roll_prob_t)
        else:
            return 0.0, 0.0, 0.0

        yaw_prob = softmax(raw[0])
        pitch_prob = softmax(raw[1])
        roll_prob = softmax(raw[2])
        return decode_angles(yaw_prob, pitch_prob, roll_prob)

    def release(self):
        if self.backend == 'rknn':
            self.rknn.release()


# ============================================================
# 人脸跟踪处理器（检测 + 姿态估计 + 坐标转换）
# ============================================================

class FaceTracker:
    """人脸跟踪处理器"""

    def __init__(self, pose_model_path=None, min_face_size=80):
        self.detector = FaceDetector(min_face_size=min_face_size)
        self.pose_estimator = PoseEstimator(model_path=pose_model_path)

    def process(self, frame):
        h, w = frame.shape[:2]
        faces = self.detector.detect(frame)

        if not faces:
            return None, 0.0, 0.0, 0.0, 0.0, 0.0

        face = max(faces, key=lambda f: (f[2] - f[0]) * (f[3] - f[1]))
        x1, y1, x2, y2 = face

        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        ex = cx - w / 2
        ey = cy - h / 2

        roi = frame[y1:y2, x1:x2]
        yaw, pitch, roll = self.pose_estimator.estimate(roi)

        return face, ex, ey, yaw, pitch, roll

    def release(self):
        self.pose_estimator.release()