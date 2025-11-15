"""
音视频同步器（PTS - Presentation Timestamp）。

提供高精度的音视频同步机制，避免音画不同步和时间漂移。
"""

import time
from dataclasses import dataclass
from typing import Optional

from .config import SyncConfig


@dataclass
class SyncState:
    """同步状态"""
    base_time: float = 0.0  # 时间基准（秒）
    frame_count: int = 0  # 已渲染帧数
    audio_pts: float = 0.0  # 当前音频 PTS（秒）
    video_pts: float = 0.0  # 当前视频 PTS（秒）
    drift: float = 0.0  # 音视频漂移（秒）
    dropped_frames: int = 0  # 丢弃帧数
    duplicated_frames: int = 0  # 重复帧数


class PTSSync:
    """
    PTS 时间戳同步器。

    基于 Presentation Timestamp 实现高精度音视频同步，
    自动检测和校正时间漂移。
    """

    def __init__(self, fps: float = 25.0, config: Optional[SyncConfig] = None) -> None:
        if config is None:
            from .config import get_default_config
            config = get_default_config().sync

        self._config = config
        self._fps = fps
        self._frame_duration = 1.0 / fps

        self._state = SyncState()
        self._initialized = False

    def init_sync(self, start_time: Optional[float] = None) -> None:
        """
        初始化同步状态。

        参数:
            start_time: 起始时间（秒），默认使用当前时间
        """
        if start_time is None:
            start_time = time.time()

        self._state = SyncState(base_time=start_time)
        self._initialized = True

    def update_audio_pts(self, audio_pts: float) -> None:
        """
        更新音频 PTS。

        参数:
            audio_pts: 音频当前播放位置的时间戳（秒）
        """
        if not self._initialized:
            self.init_sync()

        self._state.audio_pts = audio_pts
        self._update_drift()

    def get_video_pts(self, frame_index: Optional[int] = None) -> float:
        """
        获取视频 PTS。

        参数:
            frame_index: 帧索引，默认使用内部计数器

        返回:
            video_pts: 视频 PTS（秒）
        """
        if not self._initialized:
            self.init_sync()

        if frame_index is None:
            frame_index = self._state.frame_count

        video_pts = self._state.base_time + frame_index * self._frame_duration
        self._state.video_pts = video_pts
        return video_pts

    def should_render_frame(self) -> tuple[bool, str]:
        """
        判断是否应该渲染当前帧。

        基于音视频 PTS 差异决定是否渲染、跳过或重复帧。

        返回:
            (should_render, action): 是否渲染，动作说明
                action: "render" | "drop" | "duplicate"
        """
        if not self._config.use_pts:
            # 不使用 PTS，始终渲染
            return True, "render"

        if not self._initialized:
            return True, "render"

        # 计算音视频差异（音频领先为正，视频领先为负）
        av_diff = self._state.audio_pts - self._state.video_pts
        tolerance = self._config.pts_tolerance_ms / 1000.0

        if abs(av_diff) < tolerance:
            # 同步良好，正常渲染
            return True, "render"
        elif av_diff > tolerance:
            # 音频领先，视频需要加速（丢帧）
            if av_diff > self._frame_duration * 2:
                # 领先超过2帧，跳过当前帧
                self._state.dropped_frames += 1
                return False, "drop"
            else:
                # 领先较少，正常渲染但不等待
                return True, "render"
        else:
            # 视频领先，需要减速（重复帧）
            if abs(av_diff) > self._frame_duration:
                # 领先超过1帧，重复上一帧
                self._state.duplicated_frames += 1
                return False, "duplicate"
            else:
                # 领先较少，正常渲染
                return True, "render"

    def wait_for_next_frame(self) -> float:
        """
        等待到下一帧的渲染时间。

        返回:
            sleep_time: 需要睡眠的时间（秒），负值表示已经落后
        """
        if not self._initialized:
            return 0.0

        target_pts = self.get_video_pts(self._state.frame_count + 1)
        now = time.time()
        sleep_time = target_pts - now

        return sleep_time

    def advance_frame(self) -> None:
        """推进到下一帧"""
        self._state.frame_count += 1
        self._state.video_pts = self.get_video_pts()

    def _update_drift(self) -> None:
        """更新音视频漂移量"""
        self._state.drift = self._state.audio_pts - self._state.video_pts

        # 检查是否需要校正漂移
        threshold = self._config.drift_correction_threshold_ms / 1000.0
        if abs(self._state.drift) > threshold:
            self._correct_drift()

    def _correct_drift(self) -> None:
        """校正时间漂移"""
        # 重置时间基准，消除累积漂移
        now = time.time()
        self._state.base_time = now - self._state.frame_count * self._frame_duration

        print(
            f"[PTSSync] 检测到漂移 {self._state.drift*1000:.1f}ms，已校正时间基准"
        )

    def get_stats(self) -> dict:
        """
        获取同步统计信息。

        返回:
            stats: 包含帧数、漂移、丢帧等统计信息
        """
        return {
            "frame_count": self._state.frame_count,
            "audio_pts": self._state.audio_pts,
            "video_pts": self._state.video_pts,
            "drift_ms": self._state.drift * 1000,
            "dropped_frames": self._state.dropped_frames,
            "duplicated_frames": self._state.duplicated_frames,
            "avg_fps": (
                self._state.frame_count / (time.time() - self._state.base_time)
                if self._state.base_time > 0
                else 0.0
            ),
        }

    def reset(self) -> None:
        """重置同步状态"""
        self._state = SyncState()
        self._initialized = False
