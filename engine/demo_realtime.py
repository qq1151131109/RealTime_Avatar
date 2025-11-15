"""
实时数字人 demo（完整版）。

使用训练好的 Audio2Motion 模型，从真实音频驱动数字人：
- 支持麦克风实时采集或音频文件播放
- 使用训练好的 Audio2Motion 模型进行推理
- 如果模型不存在，自动降级到占位符（正弦波）
- 集成配置系统、FPS 监控、眨眼和 PTS 同步

使用方法:

1. 使用训练好的模型 + 音频文件:
   python -m engine.demo_realtime --model checkpoints/best_model.pt --audio speech.wav

2. 使用训练好的模型 + 麦克风:
   python -m engine.demo_realtime --model checkpoints/best_model.pt --microphone

3. 占位符模式（正弦波，无需模型）:
   python -m engine.demo_realtime_stub

"""

import argparse
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .avatar_loader import load_avatar
from .renderer import LivePortraitRenderer
from .motion import MotionVector
from .audio2motion_stub import Audio2MotionStub
from .pts_sync import PTSSync
from .config import EngineConfig, get_default_config
from .audio_input import create_audio_input, AudioInput


def run_demo_with_audio(
    avatar_root: Path,
    model_path: Optional[str] = None,
    audio_mode: str = "file",
    audio_file: Optional[str] = None,
    fps: float = 25.0,
    config: Optional[EngineConfig] = None,
) -> None:
    """
    运行实时数字人 demo（带音频输入）。

    参数:
        avatar_root: Avatar 资源目录
        model_path: Audio2Motion 模型路径（如果为 None，使用占位符）
        audio_mode: 音频输入模式 ("microphone", "file", "silent")
        audio_file: 音频文件路径（audio_mode="file" 时需要）
        fps: 目标帧率
        config: 引擎配置，默认使用全局配置
    """
    if config is None:
        config = get_default_config()

    interval = 1.0 / fps
    avatar_config, source_bgr = load_avatar(avatar_root)
    renderer = LivePortraitRenderer(avatar_config, source_bgr, config.render)

    # 尝试加载 Audio2Motion 模型
    audio2motion = None
    use_trained_model = False

    if model_path and Path(model_path).exists():
        try:
            print(f"[Demo] 尝试加载训练好的模型: {model_path}")
            from .audio2motion import Audio2Motion

            audio2motion = Audio2Motion(
                model_path=model_path,
                config=config.audio2motion,
            )
            audio2motion.start()
            use_trained_model = True
            print("[Demo] ✅ 使用训练好的 Audio2Motion 模型")
        except Exception as e:
            print(f"[Demo] ⚠️  加载模型失败: {e}")
            print("[Demo] 降级到占位符模式")
            audio2motion = None

    # 如果没有模型，使用占位符
    if audio2motion is None:
        print("[Demo] 使用占位符 Audio2Motion（正弦波）")
        audio2motion = Audio2MotionStub(target_fps=fps, config=config.audio2motion)
        use_trained_model = False

    # 初始化 PTS 同步器
    pts_sync = PTSSync(fps=fps, config=config.sync)
    pts_sync.init_sync()

    # 创建音频输入
    audio_input: Optional[AudioInput] = None

    if use_trained_model and audio_mode != "silent":
        try:
            audio_input = create_audio_input(
                mode=audio_mode,
                file_path=audio_file,
                sample_rate=config.audio2motion.sample_rate,
                chunk_size=int(config.audio2motion.sample_rate * 0.1),  # 100ms chunks
                loop=(audio_mode == "file"),  # 文件模式循环播放
            )

            # 启动音频采集，回调将音频推送到 Audio2Motion
            audio_input.start(callback=lambda audio_chunk: audio2motion.push_audio(audio_chunk))

            print(f"[Demo] 音频输入模式: {audio_mode}")
        except Exception as e:
            print(f"[Demo] ⚠️  音频输入初始化失败: {e}")
            print("[Demo] 继续使用占位符模式")
            audio_input = None

    # 渲染循环
    print(f"[Demo] 开始实时渲染 (目标 {fps} FPS)")
    print("[Demo] 按 ESC 退出")

    window_name = "Real-time Digital Human"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    # 统计信息
    frame_count = 0
    dropped_frames = 0
    duplicated_frames = 0
    last_motion = MotionVector.neutral()

    # FPS 计算
    fps_update_interval = 0.5  # 每 0.5 秒更新一次 FPS
    fps_frame_count = 0
    fps_start_time = time.time()
    current_fps = 0.0

    start_time = time.time()

    try:
        while True:
            loop_start = time.time()

            # 获取当前音频时间戳
            audio_pts = time.time() - start_time

            # 获取 MotionVector
            motion = audio2motion.get_motion(audio_pts)

            # PTS 同步：判断是否渲染此帧
            should_render, action = pts_sync.should_render_frame()

            if action == "drop":
                # 丢帧
                dropped_frames += 1
                pts_sync.advance_frame()
                continue
            elif action == "duplicate":
                # 重复上一帧
                motion = last_motion
                duplicated_frames += 1

            # 渲染
            output_bgr = renderer.render(motion)
            frame_count += 1
            last_motion = motion

            # 更新 PTS
            pts_sync.advance_frame()

            # 计算 FPS
            fps_frame_count += 1
            now = time.time()
            if now - fps_start_time >= fps_update_interval:
                current_fps = fps_frame_count / (now - fps_start_time)
                fps_frame_count = 0
                fps_start_time = now

            # 获取漂移信息
            drift_ms = pts_sync.get_drift() * 1000

            # 在图像上绘制 FPS 和漂移
            display_img = output_bgr.copy()
            cv2.putText(
                display_img,
                f"FPS: {current_fps:.1f}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2,
            )
            cv2.putText(
                display_img,
                f"Drift: {drift_ms:.1f} ms",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 255),
                2,
            )

            # 显示模式信息
            mode_text = "Model: Trained" if use_trained_model else "Model: Stub (Sine)"
            cv2.putText(
                display_img,
                mode_text,
                (10, 110),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
            )

            if audio_mode != "silent":
                audio_text = f"Audio: {audio_mode}"
                cv2.putText(
                    display_img,
                    audio_text,
                    (10, 140),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                )

            cv2.imshow(window_name, display_img)

            # 检查退出键
            key = cv2.waitKey(1)
            if key == 27:  # ESC
                break

            # 帧率控制
            elapsed = time.time() - loop_start
            sleep_time = pts_sync.wait_for_next_frame()
            if sleep_time > 0:
                time.sleep(sleep_time)

    except KeyboardInterrupt:
        print("\n[Demo] 用户中断")
    finally:
        # 清理资源
        print("\n[Demo] 正在清理资源...")

        if audio_input:
            audio_input.stop()

        if use_trained_model and audio2motion:
            audio2motion.stop()

        cv2.destroyAllWindows()

        # 打印统计信息
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print(f"\n{'='*50}")
        print(f"运行统计:")
        print(f"  总时长: {total_time:.2f} 秒")
        print(f"  总帧数: {frame_count}")
        print(f"  平均 FPS: {avg_fps:.2f}")
        print(f"  丢帧数: {dropped_frames}")
        print(f"  重复帧数: {duplicated_frames}")
        print(f"{'='*50}")


def main():
    """命令行入口"""
    parser = argparse.ArgumentParser(description="实时数字人 demo（完整版）")

    parser.add_argument(
        "--avatar",
        type=str,
        default="engine/avatars/demo",
        help="Avatar 资源目录路径（默认: engine/avatars/demo）",
    )

    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Audio2Motion 模型路径（如果不提供，使用占位符）",
    )

    parser.add_argument(
        "--audio",
        type=str,
        default=None,
        help="音频文件路径（与 --microphone 互斥）",
    )

    parser.add_argument(
        "--microphone",
        action="store_true",
        help="使用麦克风输入（与 --audio 互斥）",
    )

    parser.add_argument(
        "--fps",
        type=float,
        default=25.0,
        help="目标帧率（默认: 25.0）",
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="配置文件路径（可选）",
    )

    args = parser.parse_args()

    # 确定音频模式
    if args.microphone and args.audio:
        parser.error("--microphone 和 --audio 不能同时使用")

    if args.microphone:
        audio_mode = "microphone"
        audio_file = None
    elif args.audio:
        audio_mode = "file"
        audio_file = args.audio
    else:
        # 没有音频输入，使用静音（仅用于测试占位符）
        audio_mode = "silent"
        audio_file = None

    # 加载配置
    config = None
    if args.config:
        config = EngineConfig.load_from_file(args.config)

    # 运行 demo
    avatar_root = Path(args.avatar)

    run_demo_with_audio(
        avatar_root=avatar_root,
        model_path=args.model,
        audio_mode=audio_mode,
        audio_file=audio_file,
        fps=args.fps,
        config=config,
    )


if __name__ == "__main__":
    main()
