#!/bin/bash
# 运动控制程序 - 一键启动脚本
# ================================
# 用法:
#   bash run.sh human        # 人体跟随模式
#   bash run.sh face         # 人脸跟随模式
#
# 默认参数可在下方修改

set -e

# === 默认配置（根据实际情况修改）===
PORT="/dev/ttyACM0"          # 机械臂串口
CAM_ID=0                     # 摄像头 ID
WIDTH=640                    # 摄像头宽度
HEIGHT=480                   # 摄像头高度

# 模型路径
YOLO_MODEL="yolov8n-pose.rknn"      # YOLOv8-Pose 人体检测模型
HOPENET_MODEL="hopenet_i8.rknn"      # Hopenet 人脸姿态模型

# 激活 LeRobot 环境（如有需要）
# conda activate lerobot

# === 解析参数 ===
MODE="${1:-human}"

if [ "$MODE" != "human" ] && [ "$MODE" != "face" ]; then
    echo "用法: bash run.sh [human|face]"
    echo "  human  - 人体跟随模式 (YOLOv8-Pose)"
    echo "  face   - 人脸跟随模式 (Hopenet)"
    exit 1
fi

echo "============================================="
echo "  运动控制程序 - ${MODE^^} 跟随模式"
echo "============================================="
echo "  串口:   $PORT"
echo "  摄像头: /dev/video$CAM_ID"
echo "  分辨率: ${WIDTH}x${HEIGHT}"
echo "============================================="

if [ "$MODE" = "human" ]; then
    python motion_control.py \
        --mode human \
        --port "$PORT" \
        --cam-id "$CAM_ID" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --model "$YOLO_MODEL" \
        --show
elif [ "$MODE" = "face" ]; then
    python motion_control.py \
        --mode face \
        --port "$PORT" \
        --cam-id "$CAM_ID" \
        --width "$WIDTH" \
        --height "$HEIGHT" \
        --pose-model "$HOPENET_MODEL" \
        --show
fi