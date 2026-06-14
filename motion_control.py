#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运动控制程序 - 人脸跟随 / 人体跟随
==================================
板端部署版本 (RK3576 / RK3588)

架构:
  摄像头 → [人脸检测+姿态估计 | 人体检测+关键点] → 运动控制器 → 机械臂

运行方式:
  # 人体跟随模式（默认）
  python motion_control.py --mode human --port /dev/ttyACM0 --show

  # 人脸跟随模式
  python motion_control.py --mode face --port /dev/ttyACM0 --show

  # 无机械臂测试模式
  python motion_control.py --mode human --no-robot --show

操作:
  SPACE - 切换跟踪开关
  M     - 切换模式 (face/human)
  ESC/Q - 退出程序
"""

import os
import sys
import time
import argparse
import cv2
import numpy as np
import torch

# ============================================================
# 本地模块导入 (板端扁平目录结构)
# ============================================================
from control_utils import (
    HOME_POSE_DEG,
    MotionController,
    SearchController,
    build_home_pose,
    build_zero_pose,
    send_action_to_robot,
    smooth_move_to,
)

# ============================================================
# 参数配置
# ============================================================

MISS_TO_SEARCH = 8
HIT_TO_CONFIRM = 3
LOST_TIMEOUT = 3.0

SEARCH_PAN_RANGE = 45.0
SEARCH_STEP_DEG = 2.0

EMA_ALPHA = 0.15


# ============================================================
# 可视化工具
# ============================================================

class Visualizer:
    """画面叠加信息绘制"""

    @staticmethod
    def draw_center_cross(frame):
        h, w = frame.shape[:2]
        cv2.line(frame, (w // 2 - 20, h // 2), (w // 2 + 20, h // 2), (0, 255, 255), 2)
        cv2.line(frame, (w // 2, h // 2 - 20), (w // 2, h // 2 + 20), (0, 255, 255), 2)

    @staticmethod
    def draw_face_box(frame, face, color=(0, 255, 0)):
        x1, y1, x2, y2 = face
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        return frame

    @staticmethod
    def draw_hud(frame, mode, tracking, search_mode, fps, info_dict):
        h, w = frame.shape[:2]
        overlay = frame.copy()
        cv2.rectangle(overlay, (5, 5), (300, 190), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.5, frame, 0.5, 0)

        y = 25
        cv2.putText(frame, f"Mode: {mode.upper()}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
        y += 22
        status_text = "TRACKING" if tracking else "PAUSED"
        status_color = (0, 255, 0) if tracking else (0, 0, 255)
        cv2.putText(frame, f"Status: {status_text}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, status_color, 1)
        y += 22
        if search_mode:
            cv2.putText(frame, "SEARCHING...", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 165, 255), 1)
            y += 22
        for key, val in info_dict.items():
            cv2.putText(frame, f"{key}: {val}", (12, y),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (200, 200, 200), 1)
            y += 20
        cv2.putText(frame, f"FPS: {fps:.1f}", (12, y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(frame, "SPACE:Toggle | M:Mode | ESC:Quit",
                    (w - 380, h - 12), cv2.FONT_HERSHEY_SIMPLEX,
                    0.42, (140, 140, 140), 1)
        return frame


# ============================================================
# 主控制循环类
# ============================================================

class MotionControlApp:
    """运动控制应用主类"""

    def __init__(self, args):
        self.args = args
        self.mode = args.mode
        self.use_robot = not args.no_robot

        self.tracking = True
        self.running = True

        self.miss_count = 0
        self.hit_count = 0
        self.last_detect_time = time.time()
        self.search_mode = False

        self.fps_counter = []
        self.fps = 0.0

        self.face_tracker = None
        self.human_detector = None

        self.robot_ctrl = None
        self.joint_names = None

        self.motion_ctrl = None
        self.search_ctrl = None

    def setup(self):
        """初始化所有模块"""
        print("=" * 55)
        print(f"  运动控制程序 - {self.mode.upper()} 跟随模式")
        print("=" * 55)

        self._init_camera()
        self._init_detector()

        if self.use_robot:
            self._init_robot()
        else:
            print("[INFO] 无机械臂模式（仅显示检测结果）")

        self._init_controller()

        print()
        print("  System Ready!")
        print(f"  SPACE: Toggle tracking | M: Switch mode | ESC: Quit")
        print("=" * 55)

    def _init_camera(self):
        cap = cv2.VideoCapture(self.args.cam_id)
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.args.width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.args.height)
        if not cap.isOpened():
            print(f"[ERROR] 无法打开摄像头 {self.args.cam_id}")
            sys.exit(1)
        self.cap = cap
        print(f"  [Camera] ID={self.args.cam_id}, {self.args.width}x{self.args.height}")

    def _init_detector(self):
        if self.mode == 'face':
            from face_detector import FaceTracker
            print("  [Detector] 加载人脸跟踪模块...")
            self.face_tracker = FaceTracker(
                pose_model_path=self.args.pose_model,
                min_face_size=self.args.min_face
            )
        elif self.mode == 'human':
            from human_detector import HumanDetector, find_model
            model_path = self.args.model
            if not model_path or not os.path.exists(model_path):
                model_path = find_model()
                if not model_path:
                    print("[ERROR] 找不到 YOLOv8-Pose 模型文件！")
                    print("  请使用 --model 指定 .rknn 模型路径")
                    print("  或把模型放入 model/ 目录")
                    sys.exit(1)
            print(f"  [Detector] 加载人体检测模型: {model_path}")
            self.human_detector = HumanDetector(model_path)

    def _init_robot(self):
        from robot_controller import RobotController
        print(f"  [Robot] 连接机械臂: {self.args.port}")
        self.robot_ctrl = RobotController(
            port=self.args.port,
            robot_id=self.args.robot_id,
        )
        self.robot_ctrl.connect()
        self.joint_names = self.robot_ctrl.joint_names

    def _init_controller(self):
        if self.use_robot:
            joint_names = self.joint_names
        else:
            joint_names = [
                'shoulder_pan.pos', 'shoulder_lift.pos',
                'elbow_flex.pos', 'wrist_flex.pos',
                'wrist_roll.pos', 'gripper.pos'
            ]

        pan_center = HOME_POSE_DEG.get('shoulder_pan.pos', 0.0)
        shoulder_center = HOME_POSE_DEG.get('shoulder_lift.pos', 30.0)
        elbow_center = HOME_POSE_DEG.get('elbow_flex.pos', -65.0)

        self.motion_ctrl = MotionController(
            joint_names=joint_names,
            pan_center=pan_center,
            shoulder_center=shoulder_center,
            elbow_center=elbow_center,
            deadzone_x=25, deadzone_y=35,
            pan_gain=0.01, shoulder_gain=0.01, elbow_gain=0.01,
            ema_alpha=EMA_ALPHA,
        )

        self.search_ctrl = SearchController(
            pan_center=pan_center,
            range_deg=SEARCH_PAN_RANGE,
            step_deg=SEARCH_STEP_DEG,
        )

        home_action = build_home_pose(joint_names)
        self.motion_ctrl.set_current_action(home_action)

        if self.use_robot:
            print("  [Controller] 移动到 Home Pose...")
            target_action = smooth_move_to(
                self.robot_ctrl.robot, joint_names,
                build_home_pose, duration=2.0, steps=50
            )
            self.motion_ctrl.set_current_action(target_action)

    def switch_mode(self):
        old_mode = self.mode
        self.mode = 'human' if self.mode == 'face' else 'face'

        if old_mode == 'face' and self.face_tracker:
            self.face_tracker.release()
            self.face_tracker = None
        elif old_mode == 'human' and self.human_detector:
            self.human_detector.release()
            self.human_detector = None

        self.motion_ctrl.reset_smoothing()
        self.search_ctrl.reset()
        self.hit_count = 0
        self.miss_count = 0
        self.search_mode = False
        self.last_detect_time = time.time()

        self._init_detector()
        print(f"\n[Mode] 切换到: {self.mode.upper()}")

    def process_frame(self, frame):
        h, w = frame.shape[:2]
        target_detected = False
        ex, ey = 0.0, 0.0
        info_dict = {}

        if self.tracking:
            if self.mode == 'face' and self.face_tracker:
                face, ex, ey, yaw, pitch, roll = self.face_tracker.process(frame)
                if face is not None:
                    target_detected = True
                    Visualizer.draw_face_box(frame, face)
                    info_dict = {
                        "Yaw": f"{yaw:+6.2f} deg",
                        "Pitch": f"{pitch:+6.2f} deg",
                        "Roll": f"{roll:+6.2f} deg",
                        "Error X": f"{ex:+.0f} px",
                        "Error Y": f"{ey:+.0f} px",
                    }

            elif self.mode == 'human' and self.human_detector:
                person_box, ex, ey, keypoints = self.human_detector.process(frame)
                if person_box is not None:
                    target_detected = True
                    x1, y1, x2, y2 = person_box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    cv2.putText(frame, "Person", (x1, y1 - 5),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
                    info_dict = {
                        "Box": f"{x1},{y1}-{x2},{y2}",
                        "Error X": f"{ex:+.0f} px",
                        "Error Y": f"{ey:+.0f} px",
                    }

            if target_detected:
                self.hit_count += 1
                self.miss_count = 0
                self.last_detect_time = time.time()
                self.search_mode = False
                self.search_ctrl.stop()
                if self.hit_count >= HIT_TO_CONFIRM and self.use_robot:
                    target_action = self.motion_ctrl.step(ex, ey)
                    send_action_to_robot(self.robot_ctrl.robot, target_action)
            else:
                self.hit_count = 0
                self.miss_count += 1
                if (self.miss_count > MISS_TO_SEARCH and
                        time.time() - self.last_detect_time > LOST_TIMEOUT):
                    if not self.search_mode:
                        self.search_mode = True
                        self.search_ctrl.start()
                        print("[Search] 未检测到目标，开始扫描...")
                if self.search_mode and self.use_robot:
                    current_action = self.motion_ctrl.current_action
                    target_action = self.search_ctrl.step(current_action, self.joint_names)
                    self.motion_ctrl.set_current_action(target_action)
                    send_action_to_robot(self.robot_ctrl.robot, target_action)

        Visualizer.draw_center_cross(frame)
        if target_detected and self.tracking:
            h, w = frame.shape[:2]
            cx, cy = int(w // 2 + ex), int(h // 2 + ey)
            cv2.line(frame, (w // 2, h // 2), (cx, cy), (0, 255, 255), 2)
            cv2.circle(frame, (cx, cy), 8, (0, 255, 255), -1)

        frame = Visualizer.draw_hud(
            frame, self.mode, self.tracking, self.search_mode, self.fps, info_dict
        )
        return frame

    def handle_key(self, key):
        if key in (27, ord('q'), ord('Q')):
            self.running = False
        elif key == 32:
            self.tracking = not self.tracking
            status = "ON" if self.tracking else "OFF"
            print(f"[UI] Tracking: {status}")
            if not self.tracking:
                self.search_mode = False
                self.search_ctrl.stop()
        elif key in (ord('m'), ord('M')):
            self.switch_mode()

    def run(self):
        try:
            while self.running:
                t_start = time.time()
                ret, frame = self.cap.read()
                if not ret:
                    time.sleep(0.05)
                    continue
                frame = cv2.flip(frame, 1)
                frame = self.process_frame(frame)

                dt = time.time() - t_start
                self.fps_counter.append(1.0 / max(dt, 0.001))
                if len(self.fps_counter) > 30:
                    self.fps_counter.pop(0)
                self.fps = sum(self.fps_counter) / len(self.fps_counter)

                if self.args.show:
                    cv2.imshow("Motion Control - Board", frame)
                key = cv2.waitKey(1) & 0xFF
                self.handle_key(key)
        except KeyboardInterrupt:
            print("\n[Shutdown] 用户中断")
        finally:
            self.cleanup()

    def cleanup(self):
        print("[Shutdown] 释放资源...")
        if self.use_robot and self.robot_ctrl and self.robot_ctrl.is_connected:
            try:
                print("  [Robot] 移动到 Zero Pose...")
                smooth_move_to(
                    self.robot_ctrl.robot, self.joint_names,
                    build_zero_pose, duration=2.0, steps=50
                )
            except Exception as e:
                print(f"  [WARN] 归零失败: {e}")
        if self.face_tracker:
            self.face_tracker.release()
        if self.human_detector:
            self.human_detector.release()
        if self.robot_ctrl:
            self.robot_ctrl.disconnect()
        if self.cap:
            self.cap.release()
        cv2.destroyAllWindows()
        print("[Shutdown] 完成。")


def main():
    parser = argparse.ArgumentParser(
        description="运动控制程序 - 板端部署版",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  python motion_control.py --mode human --port /dev/ttyACM0 --show
  python motion_control.py --mode face --port /dev/ttyACM0 --show
  python motion_control.py --mode human --no-robot --show
        """
    )
    parser.add_argument('--mode', type=str, default='human',
                        choices=['face', 'human'],
                        help='跟踪模式 (默认: human)')
    parser.add_argument('--port', type=str, default='/dev/ttyACM0',
                        help='机械臂串口')
    parser.add_argument('--robot-id', type=str, default='follower_arm')
    parser.add_argument('--no-robot', action='store_true',
                        help='无机械臂模式')
    parser.add_argument('--cam-id', type=int, default=11,
                        help='摄像头 ID (板端默认: 11)')
    parser.add_argument('--width', type=int, default=640)
    parser.add_argument('--height', type=int, default=480)
    parser.add_argument('--show', action='store_true',
                        help='显示预览窗口')
    parser.add_argument('--model', type=str, default='model/yolov8n-pose.rknn',
                        help='YOLOv8-Pose RKNN 模型路径')
    parser.add_argument('--pose-model', type=str, default='model/hopenet_i8.rknn',
                        help='Hopenet 姿态估计模型路径')
    parser.add_argument('--min-face', type=int, default=80)
    args = parser.parse_args()

    app = MotionControlApp(args)
    app.setup()
    app.run()


if __name__ == '__main__':
    main()