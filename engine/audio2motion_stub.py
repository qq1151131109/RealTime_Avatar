"""
Audio2Motion 占位实现：

- 接口与缓冲逻辑按照实时使用场景设计；
- 当前实现不真正解析 PCM，仅按时间生成一个周期性张嘴 Motion 序列，便于打通端到端链路；
- 后续可替换内部推理为真实的深度学习模型。
"""

import math
import time
from collections import deque
from typing import Optional

import numpy as np

from .motion import MotionVector, clamp_motion
from .config import Audio2MotionConfig
from .blink_controller import BlinkController


class Audio2MotionStub:
    """
    简化占位版 Audio2Motion：

    - push_audio: 接收 PCM 片段（当前忽略内容，仅用于时间推进）；
    - pop_motion: 按固定帧率产生一个周期性张嘴的 MotionVector。
    """

    def __init__(
        self,
        target_fps: float = 25.0,
        max_queue_size: int = 100,
        config: Optional[Audio2MotionConfig] = None,
    ) -> None:
        if config is not None:
            self._target_fps = config.target_fps
            self._max_queue_size = config.max_queue_size
        else:
            self._target_fps = target_fps
            self._max_queue_size = max_queue_size

        self._frame_interval = 1.0 / self._target_fps
        self._last_frame_time: Optional[float] = None
        self._audio_queue: deque[bytes] = deque()
        self._phase: float = 0.0

        # 初始化眨眼控制器
        self._blink_controller = BlinkController()

    def push_audio(self, pcm_chunk: bytes) -> None:
        """
        推入一段 PCM 数据。
        当前占位实现仅用于记录"有数据到来"，不解析内容。
        队列超过最大长度时自动丢弃旧数据，避免内存泄漏。
        """
        if not pcm_chunk:
            return
        self._audio_queue.append(pcm_chunk)
        # 限制队列大小，避免内存泄漏
        while len(self._audio_queue) > self._max_queue_size:
            self._audio_queue.popleft()

    def pop_motion(self) -> Optional[MotionVector]:
        """
        按 target_fps 输出一帧 MotionVector，如果尚未到时间则返回 None。
        """
        now = time.time()
        if self._last_frame_time is None:
            self._last_frame_time = now

        if now - self._last_frame_time < self._frame_interval:
            return None

        self._last_frame_time = now

        # 使用简单正弦函数生成嘴型开合与轻微头动，模拟说话节奏
        # 注意：MotionVector 所有维度都在 [-1, 1] 范围内
        self._phase += self._frame_interval * 2.0 * math.pi * 1.2
        # jaw_open: -1(闭合) 到 1(张开)，映射正弦波 [-1, 1] -> [0, 1] 再映射到 [-1, 1]
        jaw_open = math.sin(self._phase)  # [-1, 1]，但我们希望 0 为中性，所以保持原值
        # 为了让嘴型有明显的开合，我们将负值视为闭合状态
        jaw_open = max(-1.0, min(1.0, jaw_open))

        head_yaw = 0.3 * math.sin(self._phase * 0.7)
        head_pitch = 0.2 * math.sin(self._phase * 0.5)
        head_roll = 0.15 * math.sin(self._phase * 1.1)

        # 更新眨眼状态
        eye_blink = self._blink_controller.update()

        values = np.zeros(MotionVector.dim(), dtype=np.float32)
        values[0] = float(jaw_open)
        values[3] = float(head_yaw)
        values[4] = float(head_pitch)
        values[5] = float(head_roll)
        values[6] = float(eye_blink)  # 眨眼参数
        motion = MotionVector(values=values)
        return clamp_motion(motion)
