#!/usr/bin/env python3
"""
系统依赖检查脚本。

检查实时数字人引擎所需的所有依赖是否正确安装。
"""

import sys
from pathlib import Path


def check_python_version():
    """检查 Python 版本"""
    print("=" * 60)
    print("检查 Python 版本...")
    version = sys.version_info
    print(f"  当前版本: Python {version.major}.{version.minor}.{version.micro}")

    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ 需要 Python 3.8 或更高版本")
        return False
    else:
        print("  ✅ Python 版本符合要求")
        return True


def check_module(module_name, package_name=None, optional=False):
    """检查 Python 模块是否安装"""
    if package_name is None:
        package_name = module_name

    try:
        __import__(module_name)
        print(f"  ✅ {package_name}")
        return True
    except ImportError:
        if optional:
            print(f"  ⚠️  {package_name} (可选，某些功能需要)")
        else:
            print(f"  ❌ {package_name} (必需)")
        return False


def check_python_packages():
    """检查 Python 包"""
    print("\n" + "=" * 60)
    print("检查 Python 包...")

    # 必需的包
    required = [
        ("torch", "PyTorch"),
        ("cv2", "opencv-python"),
        ("numpy", "numpy"),
        ("PIL", "Pillow"),
    ]

    # 可选的包
    optional = [
        ("torchaudio", "torchaudio（用于音频文件输入）"),
        ("pyaudio", "pyaudio（用于麦克风输入）"),
        ("mediapipe", "mediapipe（用于数据标注）"),
        ("transformers", "transformers（用于 HuBERT 特征）"),
    ]

    all_ok = True

    print("\n必需的包:")
    for module, package in required:
        if not check_module(module, package, optional=False):
            all_ok = False

    print("\n可选的包:")
    for module, package in optional:
        check_module(module, package, optional=True)

    return all_ok


def check_ffmpeg():
    """检查 ffmpeg 是否安装"""
    import subprocess

    print("\n" + "=" * 60)
    print("检查 ffmpeg...")

    try:
        result = subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
            text=True,
            timeout=5,
        )

        if result.returncode == 0:
            # 提取版本号
            first_line = result.stdout.split('\n')[0]
            print(f"  ✅ {first_line}")
            return True
        else:
            print("  ❌ ffmpeg 未正确安装")
            return False
    except FileNotFoundError:
        print("  ❌ ffmpeg 未安装")
        print("     安装方法:")
        print("       macOS: brew install ffmpeg")
        print("       Ubuntu: sudo apt-get install ffmpeg")
        print("       Windows: 从 https://ffmpeg.org/download.html 下载")
        return False
    except Exception as e:
        print(f"  ❌ 检查 ffmpeg 时出错: {e}")
        return False


def check_gpu():
    """检查 GPU 可用性"""
    print("\n" + "=" * 60)
    print("检查 GPU...")

    try:
        import torch

        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            print(f"  ✅ CUDA 可用")
            print(f"     GPU 数量: {gpu_count}")
            print(f"     GPU 0: {gpu_name}")

            # 检查 CUDA 版本
            cuda_version = torch.version.cuda
            print(f"     CUDA 版本: {cuda_version}")
            return True
        else:
            print("  ⚠️  CUDA 不可用，将使用 CPU（速度较慢）")
            return False
    except Exception as e:
        print(f"  ⚠️  检查 GPU 时出错: {e}")
        return False


def check_avatar_files():
    """检查 Avatar 文件"""
    print("\n" + "=" * 60)
    print("检查 Avatar 文件...")

    avatar_dir = Path("engine/avatars/demo")

    if not avatar_dir.exists():
        print(f"  ❌ Avatar 目录不存在: {avatar_dir}")
        return False

    avatar_json = avatar_dir / "avatar.json"
    if not avatar_json.exists():
        print(f"  ❌ Avatar 配置文件不存在: {avatar_json}")
        return False

    print(f"  ✅ Avatar 目录存在: {avatar_dir}")

    # 读取 avatar.json 检查源图像
    try:
        import json
        with open(avatar_json, 'r') as f:
            config = json.load(f)

        source_image = config.get("source_image")
        if source_image:
            # 相对于 avatar 目录
            source_path = avatar_dir / source_image
            if source_path.exists():
                print(f"  ✅ 源图像存在: {source_image}")
                return True
            else:
                print(f"  ❌ 源图像不存在: {source_path}")
                return False
        else:
            print("  ❌ avatar.json 中未配置 source_image")
            return False
    except Exception as e:
        print(f"  ❌ 读取 avatar.json 失败: {e}")
        return False


def check_liveportrait():
    """检查 LivePortrait 第三方库"""
    print("\n" + "=" * 60)
    print("检查 LivePortrait...")

    liveportrait_dir = Path("engine/third_party/LivePortrait")

    if not liveportrait_dir.exists():
        print(f"  ❌ LivePortrait 目录不存在: {liveportrait_dir}")
        print("     请克隆 LivePortrait 仓库到 engine/third_party/")
        return False

    # 检查关键文件
    key_files = [
        "src/live_portrait_pipeline.py",
        "src/config/inference_config.py",
    ]

    all_exist = True
    for file in key_files:
        file_path = liveportrait_dir / file
        if not file_path.exists():
            print(f"  ❌ 缺少文件: {file}")
            all_exist = False

    if all_exist:
        print(f"  ✅ LivePortrait 文件完整")
        return True
    else:
        return False


def print_summary(results):
    """打印检查结果摘要"""
    print("\n" + "=" * 60)
    print("检查结果摘要:")
    print("=" * 60)

    all_critical_ok = all([
        results["python_version"],
        results["python_packages"],
        results["avatar_files"],
        results["liveportrait"],
    ])

    if all_critical_ok:
        print("✅ 所有必需的组件都已正确安装！")
        print("\n你可以运行以下命令测试系统:")
        print("  python -m engine.demo_realtime_stub")
    else:
        print("❌ 一些必需的组件缺失，请安装后重试。")

    # 可选组件提示
    if not results["ffmpeg"]:
        print("\n⚠️  ffmpeg 未安装，无法进行数据标注和训练。")

    if not results["gpu"]:
        print("\n⚠️  GPU 不可用，推理速度会较慢。考虑使用 GPU 以获得更好性能。")

    print("\n" + "=" * 60)


def main():
    """主函数"""
    print("实时数字人引擎 - 系统依赖检查")

    results = {
        "python_version": check_python_version(),
        "python_packages": check_python_packages(),
        "ffmpeg": check_ffmpeg(),
        "gpu": check_gpu(),
        "avatar_files": check_avatar_files(),
        "liveportrait": check_liveportrait(),
    }

    print_summary(results)

    # 返回退出码
    if all([
        results["python_version"],
        results["python_packages"],
        results["avatar_files"],
        results["liveportrait"],
    ]):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
