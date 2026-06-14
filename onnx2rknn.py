"""
ONNX → RKNN 模型转换脚本
==========================================
用法:
  python onnx2rknn.py <onnx_path> [platform] [i8|fp] [output_path]

示例:
  # 转换 Hopenet (INT8 量化)
  python onnx2rknn.py hopenet.onnx rk3576 i8 hopenet_i8.rknn

  # 转换 MTCNN P-Net
  python onnx2rknn.py mtcnn_pnet.onnx rk3576 i8 mtcnn_pnet_i8.rknn

  # FP32 版本 (精度高, 速度慢)
  python onnx2rknn.py hopenet.onnx rk3576 fp hopenet_fp.rknn

支持的平台: rk3562 rk3566 rk3568 rk3576 rk3588
"""
import sys
import os

# RKNN-Toolkit2 需要在 Rockchip 平台上运行 (Linux + NPU 驱动)
try:
    from rknn.api import RKNN
except ImportError:
    print("=" * 55)
    print("  [ERROR] 未检测到 RKNN-Toolkit2")
    print("  此脚本需要在安装了 RKNN-Toolkit2 的环境中运行")
    print("  参考: https://tuyaopen.100ask.net/docs/TuyaDev/AppDev/part3-3/03-3-1_SetupRKNNEnvironment/")
    print("=" * 55)
    print()
    print("  替代方案:")
    print("    1. 将 ONNX 文件复制到 RK3576/RK3588 开发板")
    print("    2. 在开发板上安装 RKNN-Toolkit2")
    print("    3. 运行本脚本进行转换")
    print()
    print("  当前目录的 ONNX 文件:")
    for f in sorted(os.listdir('.')):
        if f.endswith('.onnx'):
            size_mb = os.path.getsize(f) / 1024**2
            print(f"    {f}  ({size_mb:.1f} MB)")
    sys.exit(1)


# 配置参数
DATASET_PATH = './dataset.txt'            # 校准图片列表 (INT8 量化需要)
DEFAULT_RKNN_PATH = './output.rknn'
DEFAULT_QUANT = True
DEFAULT_PLATFORM = 'rk3576'


def parse_args():
    """解析命令行参数"""
    if len(sys.argv) < 2:
        print('Usage: python onnx2rknn.py <onnx_path> [platform] [i8|fp] [output_rknn_path]')
        print('        platform: rk3562/rk3566/rk3568/rk3576/rk3588 (default: rk3576)')
        print('        i8|fp: i8=INT8量化(推荐), fp=FP32 (default: i8)')
        print()
        print('Examples:')
        print('  python onnx2rknn.py hopenet.onnx')
        print('  python onnx2rknn.py hopenet.onnx rk3576 i8 hopenet_i8.rknn')
        print('  python onnx2rknn.py mtcnn_onet.onnx rk3576 i8 mtcnn_onet_i8.rknn')
        sys.exit(1)

    onnx_path = sys.argv[1]
    platform = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PLATFORM
    quant_type = sys.argv[3] if len(sys.argv) > 3 else 'i8'
    rknn_path = sys.argv[4] if len(sys.argv) > 4 else DEFAULT_RKNN_PATH

    do_quant = quant_type == 'i8'
    return onnx_path, platform, do_quant, rknn_path


def convert_onnx_to_rknn(onnx_path, platform, do_quant, rknn_path):
    """主转换流程"""
    print("=" * 55)
    print(f"  ONNX -> RKNN Converter")
    print(f"  Input:  {onnx_path}")
    print(f"  Output: {rknn_path}")
    print(f"  Platform: {platform}")
    print(f"  Quant: {'INT8' if do_quant else 'FP32'}")
    print("=" * 55)

    # 检查输入文件
    if not os.path.exists(onnx_path):
        print(f"\n[ERROR] ONNX 文件不存在: {onnx_path}")
        sys.exit(1)

    size_mb = os.path.getsize(onnx_path) / 1024**2
    print(f"\n  ONNX size: {size_mb:.1f} MB")

    # 创建 RKNN 对象
    rknn = RKNN(verbose=True)

    # Step 1: 配置
    print('\n--> Config model')
    ret = rknn.config(
        mean_values=[[123.675, 116.28, 103.53]],   # ImageNet BGR mean
        std_values=[[58.395, 57.12, 57.375]],       # ImageNet BGR std
        target_platform=platform
    )
    if ret != 0:
        print(f'Config failed! ret={ret}')
        sys.exit(ret)
    print('  [OK] Config done')

    # Step 2: 加载 ONNX
    print('\n--> Loading ONNX model')
    ret = rknn.load_onnx(model=onnx_path)
    if ret != 0:
        print(f'Load ONNX failed! ret={ret}')
        sys.exit(ret)
    print('  [OK] ONNX loaded')

    # Step 3: 构建 RKNN
    print(f'\n--> Building RKNN model (quant={"INT8" if do_quant else "FP32"})')
    ret = rknn.build(do_quantization=do_quant, dataset=DATASET_PATH)
    if ret != 0:
        print(f'Build failed! ret={ret}')
        print('  提示: INT8 量化需要 dataset.txt (校准图片列表)')
        print('  如无校准数据，请使用 FP32 模式: i8 改为 fp')
        sys.exit(ret)
    print('  [OK] Build done')

    # Step 4: 导出
    print(f'\n--> Exporting RKNN model to {rknn_path}')
    ret = rknn.export_rknn(rknn_path)
    if ret != 0:
        print(f'Export failed! ret={ret}')
        sys.exit(ret)

    out_size = os.path.getsize(rknn_path) / 1024**2
    print(f'  [OK] RKNN saved: {rknn_path} ({out_size:.1f} MB)')

    # Step 5: 释放资源
    rknn.release()
    print('\n' + '=' * 55)
    print(f'  [OK] Conversion complete!')
    print(f'  Output: {os.path.abspath(rknn_path)}')
    print('=' * 55)


if __name__ == '__main__':
    onnx_path, platform, do_quant, rknn_path = parse_args()
    convert_onnx_to_rknn(onnx_path, platform, do_quant, rknn_path)