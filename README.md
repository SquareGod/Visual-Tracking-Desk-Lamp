# 运动控制程序 - 板端部署包

适用于 RK3576 / RK3588 开发板的运动控制程序，支持**人脸跟随**和**人体跟随**两种模式。

---

## 📁 文件清单

```
motion_control_board/
├── motion_control.py          ← 主程序入口（人体/人脸双模式）
├── control_utils.py           ← 控制核心（EMA滤波、死区、搜索）
├── robot_controller.py        ← LeRobot SO101 机械臂封装
├── face_detector.py           ← 人脸检测 + Hopenet 姿态估计
├── human_detector.py          ← YOLOv8-Pose 人体检测 + 关键点
├── hopenet.py                 ← Hopenet 模型定义（PyTorch回退）
├── onnx2rknn.py               ← ONNX→RKNN 模型转换工具
├── run.sh                     ← 一键启动脚本 (Linux)
├── requirements.txt           ← Python 依赖
├── README.md                  ← 本文件
└── model/                     ← 模型文件夹（需手动放入 .rknn 文件）
    ├── README.md              ← 模型获取说明
    ├── [yolov8n-pose.rknn]    ← 你放入的 YOLO 模型
    └── [hopenet_i8.rknn]      ← 你放入的 Hopenet 模型
```

---

## 🚀 快速上手

### 1. 把文件夹拷到板端

```bash
# 方式1: SCP
scp -r motion_control_board/ baiwen@dshanpi-a1:~/lerobot/

# 方式2: U盘拷贝
# 将 motion_control_board/ 整个文件夹拷贝到板端 ~/lerobot/ 目录下
```

### 2. 放入模型文件

把你之前转换好的 `.rknn` 模型文件放入 `model/` 目录：

```bash
# 人体检测模型
cp yolov8n-pose.rknn motion_control_board/model/

# 人脸姿态模型
cp hopenet_i8.rknn motion_control_board/model/
```

**如何获取模型？** 参见 `model/README.md`

### 3. 确保机械臂已校准

```bash
# 如果还没校准过（只需要做一次）
lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm
```

### 4. 安装依赖（首次）

```bash
# 激活 lerobot 环境
conda activate lerobot

# 安装 RKNN 推理库（确认对应 Python 版本）
cd ~/Projects/rknn-toolkit2/rknn-toolkit-lite2/packages
pip install rknn_toolkit_lite2-2.3.2-cp310-cp310-manylinux_2_17_aarch64.manylinux2014_aarch64.whl

# 回到程序目录
cd ~/lerobot/motion_control_board

# 安装其他依赖
pip install -r requirements.txt
```

### 5. 运行

```bash
# ✅ 人体跟随模式（推荐，YOLOv8-Pose 检测人体）
python motion_control.py --mode human --port /dev/ttyACM0 --show

# ✅ 人脸跟随模式（Hopenet 姿态估计）
python motion_control.py --mode face --port /dev/ttyACM0 --show

# 🔧 无机械臂测试（先看看检测效果）
python motion_control.py --mode human --no-robot --show

# 🚀 使用启动脚本
bash run.sh human
bash run.sh face
```

---

## 🎮 运行中操作

| 按键 | 功能 |
|------|------|
| `SPACE` | 暂停 / 恢复跟踪 |
| `M` | 切换人脸/人体模式 |
| `ESC` / `Q` | 安全退出（机械臂自动归零） |

---

## ⚙️ 调参指南

所有控制参数都在 `control_utils.py` 文件顶部，直接修改即可：

```python
# control_utils.py - 调整这些值改变控制行为

HOME_POSE_DEG = {           # 启动时的初始姿态
    'shoulder_pan.pos': 0.0,     # 水平角度
    'shoulder_lift.pos': 30.0,   # 抬起高度
    ...
}

class MotionController:
    def __init__(self, ...
                 deadzone_x=25,      # 水平死区（像素）：改大=减少左右微动
                 deadzone_y=35,      # 垂直死区（像素）：改大=减少上下微动
                 pan_gain=0.01,      # 水平增益：改大=水平移动更快
                 shoulder_gain=0.01, # 垂直增益：改大=垂直移动更快
                 ema_alpha=0.15):    # 平滑系数：改小=更平滑但响应慢
```

| 想达到的效果 | 调整方法 |
|-------------|---------|
| 减少抖动 | 增大 `deadzone_x`/`deadzone_y`（如 25→40） |
| 降低灵敏度 | 增大 `ema_alpha`（如 0.15→0.30） |
| 响应更快 | 增大 `pan_gain`/`shoulder_gain`（如 0.01→0.015） |
| 搜索范围更大 | 增大 `SEARCH_PAN_RANGE`（如 45→60） |

---

## ❓ 常见问题

### Q: 提示找不到 RKNN 模型
```
[ERROR] 找不到 YOLOv8-Pose 模型文件！
```
**解决**：把 `.rknn` 文件放入 `model/` 目录，或用 `--model` 参数指定路径。

### Q: 机械臂不跟随/不动
1. 确认连接了从动臂（follower arm），不要连主动臂（leader arm）
2. 确认已校准：`lerobot-calibrate ...`
3. 先用 `--no-robot --show` 模式确认检测正常

### Q: 摄像头打不开
- 运行 `lerobot-find-cameras opencv` 查看可用摄像头
- 用 `--cam-id` 参数指定正确的摄像头编号

### Q: 人脸检测不到
- Haar Cascade 需要正面人脸，确保光线充足
- 降低 `--min-face` 参数（默认 80，可降到 60）

### Q: 人体检测很慢
- 确保模型是 INT8 量化版本
- 检查 RKNN 运行时是否正确初始化

---

## 📊 程序退出流程

按 `ESC` 或 `Ctrl+C` 退出时，程序会自动：
1. 停止检测
2. 机械臂平滑移动到 Zero Pose（安全归零位）
3. 断开机械臂连接
4. 释放摄像头和模型资源