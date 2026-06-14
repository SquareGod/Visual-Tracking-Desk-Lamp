# -*- coding: utf-8 -*-
"""
机械臂控制封装
=============
基于 LeRobot SO101 的控制接口封装

前置条件:
  pip install lerobot[feetech]
  机械臂已通过 lerobot-calibrate 校准

使用方式:
  from robot_controller import RobotController
  robot = RobotController(port='COM8')
  robot.connect()
  robot.send_action({'shoulder_pan.pos': 10.0, ...})
  robot.disconnect()
"""

import torch
import numpy as np

# LeRobot SO101 从动臂
from lerobot.robots.so101_follower.config_so101_follower import SO101FollowerConfig
from lerobot.robots.so101_follower.so101_follower import SO101Follower


class RobotController:
    """
    SO101 机械臂控制封装

    封装了 LeRobot 的 SO101Follower，提供简化的控制接口。
    所有角度单位为"度"。
    """

    def __init__(self, port, robot_id="follower_arm", use_degrees=True):
        """
        初始化机械臂

        参数:
            port: 串口设备号，如 'COM8' (Windows) 或 '/dev/ttyACM0' (Linux)
            robot_id: 机械臂标识
            use_degrees: 是否使用角度制（默认 True）
        """
        self.port = port
        self.robot_id = robot_id
        self.use_degrees = use_degrees

        # 构造配置
        self.config = SO101FollowerConfig(
            port=port,
            id=robot_id,
            cameras={},
            use_degrees=use_degrees,
        )

        self._robot = None
        self._joint_names = None
        self._action_dim = None
        self._connected = False

    def connect(self):
        """连接机械臂"""
        self._robot = SO101Follower(self.config)
        self._robot.connect()
        self._connected = True

        self._joint_names = list(self._robot.action_features.keys())
        self._action_dim = len(self._joint_names)

        print(f"[RobotController] 已连接: {self.port}")
        print(f"[RobotController] 关节数量: {self._action_dim}")
        print(f"[RobotController] 关节名称: {self._joint_names}")
        return self

    def disconnect(self):
        """断开机械臂"""
        if self._connected and self._robot is not None:
            self._robot.disconnect()
            self._connected = False
            print(f"[RobotController] 已断开: {self.port}")

    @property
    def robot(self):
        """获取原始 LeRobot 机械臂实例（高级用法）"""
        return self._robot

    @property
    def joint_names(self):
        """获取关节名称列表"""
        return self._joint_names

    @property
    def action_dim(self):
        """获取动作维度"""
        return self._action_dim

    @property
    def is_connected(self):
        """是否已连接"""
        return self._connected

    def send_action(self, action):
        """
        发送关节角度指令

        参数:
            action: 支持以下格式:
                    - dict: {'shoulder_pan.pos': 10.0, ...}
                    - torch.Tensor: shape=(1, action_dim)，单位为度
                    - numpy.ndarray: shape=(action_dim,)，单位为度
        """
        if isinstance(action, torch.Tensor):
            # Tensor → dict
            action_np = action.detach().cpu().numpy().reshape(-1)
            n = min(len(self._joint_names), len(action_np))
            robot_action = {
                self._joint_names[i]: float(action_np[i])
                for i in range(n)
            }
        elif isinstance(action, np.ndarray):
            # numpy → dict
            action_np = action.reshape(-1)
            n = min(len(self._joint_names), len(action_np))
            robot_action = {
                self._joint_names[i]: float(action_np[i])
                for i in range(n)
            }
        elif isinstance(action, dict):
            robot_action = action
        else:
            raise TypeError(f"不支持 action 类型: {type(action)}")

        self._robot.send_action(robot_action)

    def get_observation(self):
        """获取当前观测（含各关节角度）"""
        return self._robot.get_observation()

    def get_current_action_tensor(self):
        """
        获取当前关节角度作为 Tensor

        返回:
            torch.Tensor, shape=(1, action_dim)
        """
        obs = self.get_observation()
        action = torch.zeros(1, self._action_dim, dtype=torch.float32)
        for i, name in enumerate(self._joint_names):
            if name in obs:
                action[0, i] = float(obs[name])
        return action

    def __enter__(self):
        """上下文管理器入口"""
        self.connect()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器出口"""
        self.disconnect()