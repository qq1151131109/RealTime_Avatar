"""
最小实时渲染 demo（占位版）：

- 使用 Avatar 配置加载一张源图像；
- 构造一个 LivePortraitRenderer 占位实例；
- 以固定帧率循环渲染源图像到窗口。
- 集成配置系统、FPS 监控、眨眼和 PTS 同步。

后续接入真实 DrivingCode 与 Audio2Motion 后，可在此基础上扩展。
"""

import time
from pathlib import Path
from typing import Optional

import cv2

from .avatar_loader import load_avatar
from .renderer import LivePortraitRenderer
from .motion import MotionVector
from .audio2motion_stub import Audio2MotionStub
from .pts_sync import PTSSync
from .config import EngineConfig, get_default_config


def run_demo(
    avatar_root: Path,
    fps: float = 25.0,
    config: Optional[EngineConfig] = None,
) -> None:
    """
    运行实时数字人 demo。

    参数:
        avatar_root: Avatar 资源目录
        fps: 目标帧率
        config: 引擎配置，默认使用全局配置
    """
    if config is None:
        config = get_default_config()

    interval = 1.0 / fps
    avatar_config, source_bgr = load_avatar(avatar_root)
    renderer = LivePortraitRenderer(avatar_config, source_bgr, config.render)
    a2m = Audio2MotionStub(target_fps=fps, config=config.audio2motion)

    # 初始化 PTS 同步器
    pts_sync = PTSSync(fps=fps, config=config.sync)
    pts_sync.init_sync()

    window_name = f"Avatar Demo - {avatar_config.name}"
    cv2.namedWindow(window_name, cv2.WINDOW_AUTOSIZE)

    # FPS 统计
    fps_update_interval = 1.0  # 每秒更新一次 FPS 显示
    fps_last_update = time.time()
    fps_frame_count = 0
    current_fps = 0.0

    # 上一帧图像（用于重复帧）
    last_frame = None

    try:
        while True:
            # 占位调用：push_audio 当前传入空字节，仅用于维持时间推进
            a2m.push_audio(b"\x00" * 320)

            # 检查是否应该渲染当前帧
            should_render, action = pts_sync.should_render_frame()

            if action == "drop":
                # 跳帧：不渲染，直接推进
                pts_sync.advance_frame()
                if config.verbose:
                    print("[Demo] 跳帧以同步")
                continue
            elif action == "duplicate":
                # 重复帧：使用上一帧
                if last_frame is not None:
                    display_frame = last_frame.copy()
                else:
                    # 没有上一帧，渲染新帧
                    motion = a2m.pop_motion()
                    if motion is None:
                        motion = MotionVector.zeros()
                    frame = renderer.render_frame(motion)
                    display_frame = frame
                    last_frame = frame
            else:
                # 正常渲染
                motion = a2m.pop_motion()
                if motion is None:
                    motion = MotionVector.zeros()

                frame = renderer.render_frame(motion)
                last_frame = frame
                display_frame = frame

            # 更新 FPS 统计
            now = time.time()
            fps_frame_count += 1
            if now - fps_last_update >= fps_update_interval:
                current_fps = fps_frame_count / (now - fps_last_update)
                fps_frame_count = 0
                fps_last_update = now

            # 在画面上叠加信息
            if config.show_fps:
                info_frame = display_frame.copy()
                fps_text = f"FPS: {current_fps:.1f}"
                cv2.putText(
                    info_frame,
                    fps_text,
                    (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2,
                    cv2.LINE_AA,
                )

                # 显示同步信息
                if config.sync.use_pts:
                    stats = pts_sync.get_stats()
                    sync_text = f"Drift: {stats['drift_ms']:.1f}ms"
                    cv2.putText(
                        info_frame,
                        sync_text,
                        (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 255),
                        2,
                        cv2.LINE_AA,
                    )

                display_frame = info_frame

            cv2.imshow(window_name, display_frame)

            if cv2.waitKey(1) & 0xFF == 27:
                break

            # PTS 同步：等待到下一帧
            sleep_time = pts_sync.wait_for_next_frame()
            if sleep_time > 0:
                time.sleep(sleep_time)
            elif sleep_time < -interval and config.warn_on_slow_frame:
                # 渲染落后警告
                print(f"[Warning] 渲染落后 {-sleep_time*1000:.1f}ms")

            pts_sync.advance_frame()

    finally:
        # 打印统计信息
        if config.verbose or config.sync.use_pts:
            stats = pts_sync.get_stats()
            print("\n=== 渲染统计 ===")
            print(f"总帧数: {stats['frame_count']}")
            print(f"平均 FPS: {stats['avg_fps']:.2f}")
            print(f"丢帧数: {stats['dropped_frames']}")
            print(f"重复帧数: {stats['duplicated_frames']}")
            print(f"最终漂移: {stats['drift_ms']:.1f}ms")

        cv2.destroyAllWindows()


if __name__ == "__main__":
    root = Path(__file__).resolve().parent / "avatars" / "demo"
    run_demo(root)
