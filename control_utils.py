# -*- coding: utf-8 -*-
"""
运动控制工具函数
===============
EMA滤波、死区控制、搜索逻辑、姿态定义
可直接用于人脸跟随和人体跟随两种模式

设计理念: 最简单、不需要微调的比例控制
"""

import time
import numpy as np
import torch


# ============================================================
# 机械臂预设姿态（角度制）
# ============================================================

# 启动姿态 - 机械臂看向前方中间位置
HOME_POSE_DEG = {
    'shoulder_pan.pos': 0.0,      # 左右正中
    'shoulder_lift.pos': 30.0,    # 手臂稍微抬起
    'elbow_flex.pos': -65.0,      # 肘部略弯
    'wrist_flex.pos': 50.0,       # 手腕稍微向下
    'wrist_roll.pos': 20.0,       # 手腕水平
    'gripper.pos': 0.0,           # 夹爪不动
}

# 归零姿态 - 程序退出时返回
ZERO_POSE_DEG = {
    'shoulder_pan.pos': -10.0,
    'shoulder_lift.pos': -5.0,
    'elbow_flex.pos': -5.0,
    'wrist_flex.pos': 70.0,
    'wrist_roll.pos': 10.0,
    'gripper.pos': 0.0,
}

# 关节运动范围限制（相对中心的角度偏移）
JOINT_LIMITS = {
    'shoulder_pan.pos': 90.0,      # ±90° 水平旋转范围
    'shoulder_lift.pos': 45.0,     # ±45° 垂直范围
    'elbow_flex.pos': 45.0,        # ±45° 肘部范围
    'wrist_flex.pos': 45.0,        # ±45° 手腕范围
    'wrist_roll.pos': 45.0,        # ±45° 手腕旋转
    'gripper.pos': 45.0,           # 夹爪范围
}


def build_home_pose(joint_names):
    """根据 joint_names 构造 Home Pose 向量"""
    action_dim = len(joint_names)
    home = torch.zeros(1, action_dim, dtype=torch.float32)
    for i, name in enumerate(joint_names):
        if name in HOME_POSE_DEG:
            home[0, i] = HOME_POSE_DEG[name]
    return home


def build_zero_pose(joint_names):
    """根据 joint_names 构造 Zero Pose 向量"""
    action_dim = len(joint_names)
    zero = torch.zeros(1, action_dim, dtype=torch.float32)
    for i, name in enumerate(joint_names):
        if name in ZERO_POSE_DEG:
            zero[0, i] = ZERO_POSE_DEG[name]
    return zero


# ============================================================
# EMA 指数移动平均滤波器
# ============================================================

class EMAFilter:
    """
    一维 EMA 滤波器，用于平滑信号
    alpha 越小越平滑但响应越慢
    """

    def __init__(self, alpha=0.15):
        self.alpha = alpha
        self.value = 0.0
        self.initialized = False

    def update(self, new_value):
        """更新并返回平滑后的值"""
        if not self.initialized:
            self.value = new_value
            self.initialized = True
        else:
            self.value = self.alpha * new_value + (1 - self.alpha) * self.value
        return self.value

    def reset(self):
        self.initialized = False
        self.value = 0.0


# ============================================================
# 死区控制
# ============================================================

def apply_deadzone(value, threshold):
    """
    死区抑制：小于阈值的信号置零，防止机械臂抖动
    """
    return 0.0 if abs(value) < threshold else value


# ============================================================
# 比例运动控制器
# ============================================================

class MotionController:
    """
    运动控制器
    ==========
    最简单的比例控制(P控制) + 死区 + EMA平滑 + 限幅

    控制逻辑:
      - 输入: 画面中心偏差 (ex, ey) 像素
      - 输出: 关节角度偏移量

    使用方式:
      ctrl = MotionController(joint_names, pan_center, shoulder_center, elbow_center)
      action = ctrl.step(current_action, ex, ey)
      send_action_to_robot(robot, action)
    """

    def __init__(self, joint_names, pan_center, shoulder_center, elbow_center,
                 deadzone_x=25, deadzone_y=35,
                 pan_gain=0.01, shoulder_gain=0.01, elbow_gain=0.01,
                 ema_alpha=0.15):
        """
        参数:
            joint_names: 机械臂关节名称列表
            pan_center: shoulder_pan 中心角度 (Home Pose 值)
            shoulder_center: shoulder_lift 中心角度
            elbow_center: elbow_flex 中心角度
            deadzone_x: 水平死区（像素），默认 25
            deadzone_y: 垂直死区（像素），默认 35
            pan_gain: 水平增益系数，默认 0.01
            shoulder_gain: 垂直肩关节增益，默认 0.01
            elbow_gain: 垂肘关节增益，默认 0.01
            ema_alpha: EMA 平滑系数，默认 0.15
        """
        self.joint_names = joint_names
        self.idx_map = {n: i for i, n in enumerate(joint_names)}

        # 中心位置
        self.pan_center = pan_center
        self.shoulder_center = shoulder_center
        self.elbow_center = elbow_center

        # 死区
        self.deadzone_x = deadzone_x
        self.deadzone_y = deadzone_y

        # 增益
        self.pan_gain = pan_gain
        self.shoulder_gain = shoulder_gain
        self.elbow_gain = elbow_gain

        # EMA 滤波器
        self.ex_filter = EMAFilter(alpha=ema_alpha)
        self.ey_filter = EMAFilter(alpha=ema_alpha)

        # 当前目标动作
        self.current_action = None

    def set_current_action(self, action):
        """设置当前动作（用于首次和重置）"""
        self.current_action = action.clone() if isinstance(action, torch.Tensor) else action

    def step(self, ex, ey):
        """
        计算新的关节角度

        参数:
            ex: 水平偏差（目标中心 - 画面中心），像素
            ey: 垂直偏差，像素

        返回:
            target_action: 更新后的目标关节角度 (torch.Tensor)
        """
        target_action = self.current_action.clone()

        # 1. EMA 平滑偏差
        ex_f = self.ex_filter.update(ex)
        ey_f = self.ey_filter.update(ey)

        # 2. 应用死区
        ex_f = apply_deadzone(ex_f, self.deadzone_x)
        ey_f = apply_deadzone(ey_f, self.deadzone_y)

        # 3. 比例控制 - 水平方向
        if 'shoulder_pan.pos' in self.idx_map:
            idx = self.idx_map['shoulder_pan.pos']
            pan_limit = JOINT_LIMITS.get('shoulder_pan.pos', 90.0)
            pan_offset = ex_f * self.pan_gain
            target_action[0, idx] = np.clip(
                self.current_action[0, idx] + pan_offset,
                self.pan_center - pan_limit,
                self.pan_center + pan_limit
            )

        # 4. 比例控制 - 垂直方向（肩关节）
        if 'shoulder_lift.pos' in self.idx_map:
            idx = self.idx_map['shoulder_lift.pos']
            lift_limit = JOINT_LIMITS.get('shoulder_lift.pos', 45.0)
            shoulder_offset = ey_f * self.shoulder_gain
            target_action[0, idx] = np.clip(
                self.current_action[0, idx] + shoulder_offset,
                self.shoulder_center - lift_limit,
                self.shoulder_center + lift_limit
            )

        # 5. 比例控制 - 垂直方向（肘关节）
        if 'elbow_flex.pos' in self.idx_map:
            idx = self.idx_map['elbow_flex.pos']
            elbow_limit = JOINT_LIMITS.get('elbow_flex.pos', 45.0)
            elbow_offset = ey_f * self.elbow_gain
            target_action[0, idx] = np.clip(
                self.current_action[0, idx] + elbow_offset,
                self.elbow_center - elbow_limit,
                self.elbow_center + elbow_limit
            )

        self.current_action = target_action
        return target_action

    def reset_smoothing(self):
        """重置 EMA 滤波器（用于模式切换）"""
        self.ex_filter.reset()
        self.ey_filter.reset()


# ============================================================
# 机械臂搜索控制
# ============================================================

class SearchController:
    """
    搜索模式控制器
    ==============
    当没有检测到目标时，机械臂左右扫描寻找目标

    使用方式:
      search = SearchController(pan_center=0.0, range_deg=45.0, step_deg=2.0)
      if no_target:
          action = search.step(current_action, joint_names)
      else:
          search.reset()
    """

    def __init__(self, pan_center=0.0, range_deg=45.0, step_deg=2.0):
        """
        参数:
            pan_center: 搜索中心角度
            range_deg: 左右扫描范围（度）
            step_deg: 每帧移动步长（度）
        """
        self.pan_center = pan_center
        self.range_deg = range_deg
        self.step_deg = step_deg
        self.search_pan = pan_center
        self.search_dir = 1.0  # +1 向右，-1 向左
        self.active = False

    def start(self, current_pan=None):
        """启动搜索模式"""
        self.active = True
        if current_pan is not None:
            self.search_pan = current_pan
        else:
            self.search_pan = self.pan_center
        self.search_dir = 1.0

    def stop(self):
        """停止搜索模式"""
        self.active = False

    def step(self, current_action, joint_names):
        """
        执行一步搜索

        参数:
            current_action: 当前关节角度 (torch.Tensor)
            joint_names: 关节名称列表

        返回:
            target_action: 更新后的目标关节角度
        """
        target_action = current_action.clone()
        self.search_pan += self.search_dir * self.step_deg

        # 边界反转方向
        if self.search_pan > self.pan_center + self.range_deg:
            self.search_pan = self.pan_center + self.range_deg
            self.search_dir = -1.0
        elif self.search_pan < self.pan_center - self.range_deg:
            self.search_pan = self.pan_center - self.range_deg
            self.search_dir = 1.0

        # 更新 shoulder_pan 关节
        idx_map = {n: i for i, n in enumerate(joint_names)}
        if 'shoulder_pan.pos' in idx_map:
            target_action[0, idx_map['shoulder_pan.pos']] = float(self.search_pan)

        return target_action

    def reset(self):
        """重置搜索状态"""
        self.active = False
        self.search_pan = self.pan_center
        self.search_dir = 1.0


# ============================================================
# 平滑插值移动
# ============================================================

def smooth_move_to(robot, joint_names, target_pose_func, duration=2.0, steps=50):
    """
    平滑移动到目标姿态

    参数:
        robot: LeRobot 机械臂实例
        joint_names: 关节名称列表
        target_pose_func: 返回目标姿态 tensor 的函数
        duration: 移动总时间（秒）
        steps: 插值步数

    返回:
        最终目标姿态 (torch.Tensor)
    """
    action_dim = len(joint_names)
    target = target_pose_func(joint_names)

    # 获取当前姿态
    obs = robot.get_observation()
    current = torch.zeros(1, action_dim, dtype=torch.float32)
    for i, name in enumerate(joint_names):
        if name in obs:
            current[0, i] = float(obs[name])

    # 逐帧插值
    for k in range(steps):
        alpha = float(k + 1) / float(steps)
        action = (1.0 - alpha) * current + alpha * target
        send_action_to_robot(robot, action)
        time.sleep(duration / steps)

    return target


# ============================================================
# 发送动作到机械臂
# ============================================================

def send_action_to_robot(robot, action):
    """
    将关节角度（度）映射成 LeRobot 键值 dict 并发送

    参数:
        robot: LeRobot 机械臂实例
        action: torch.Tensor 或 numpy.ndarray，shape=(1, action_dim)
    """
    if isinstance(action, torch.Tensor):
        action_np = action.detach().cpu().numpy().reshape(-1)
    else:
        action_np = action.reshape(-1)

    joint_names = list(robot.action_features.keys())
    n = min(len(joint_names), len(action_np))

    robot_action = {
        joint_names[i]: float(action_np[i])
        for i in range(n)
    }
    robot.send_action(robot_action)