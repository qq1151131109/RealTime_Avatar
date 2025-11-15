"""
自动眨眼控制器。

基于随机时间间隔生成自然的眨眼动作。
"""

import random
import time
from dataclasses import dataclass
from typing import Optional

from .config import BlinkConfig


@dataclass
class BlinkState:
    """眨眼状态"""
    is_blinking: bool = False
    blink_start_time: float = 0.0
    next_blink_time: float = 0.0
    eye_close_ratio: float = 0.0  # 当前眨眼闭眼程度 [0, 1]


class BlinkController:
    """
    眨眼控制器。

    生成随机间隔的眨眼动作，模拟自然眨眼行为。
    """

    def __init__(self, config: Optional[BlinkConfig] = None) -> None:
        if config is None:
            from .config import get_default_config
            config = get_default_config().blink

        self._config = config
        self._state = BlinkState()

        # 初始化下次眨眼时间
        if self._config.enabled:
            self._state.next_blink_time = time.time() + self._get_random_interval()

    def _get_random_interval(self) -> float:
        """获取随机眨眼间隔（秒）"""
        return random.uniform(self._config.min_interval, self._config.max_interval)

    def update(self) -> float:
        """
        更新眨眼状态并返回当前眼睛闭合比例。

        返回:
            eye_close_ratio: 0.0 = 睁眼，1.0 = 完全闭眼
        """
        if not self._config.enabled:
            return 0.0

        now = time.time()

        # 检查是否应该开始新的眨眼
        if not self._state.is_blinking and now >= self._state.next_blink_time:
            self._state.is_blinking = True
            self._state.blink_start_time = now
            self._state.next_blink_time = now + self._get_random_interval()

        # 如果正在眨眼，计算当前闭眼程度
        if self._state.is_blinking:
            elapsed = now - self._state.blink_start_time
            blink_progress = elapsed / self._config.duration

            if blink_progress >= 1.0:
                # 眨眼结束
                self._state.is_blinking = False
                self._state.eye_close_ratio = 0.0
            else:
                # 使用平滑曲线：快速闭眼 -> 短暂停留 -> 快速睁眼
                # 使用三角形或正弦曲线模拟
                if blink_progress < 0.3:
                    # 闭眼阶段（0-30%）
                    self._state.eye_close_ratio = (
                        blink_progress / 0.3 * self._config.eye_close_ratio
                    )
                elif blink_progress < 0.7:
                    # 保持闭眼（30-70%）
                    self._state.eye_close_ratio = self._config.eye_close_ratio
                else:
                    # 睁眼阶段（70-100%）
                    self._state.eye_close_ratio = (
                        (1.0 - blink_progress) / 0.3 * self._config.eye_close_ratio
                    )

        return self._state.eye_close_ratio

    def force_blink(self) -> None:
        """强制触发一次眨眼"""
        if not self._config.enabled:
            return

        now = time.time()
        self._state.is_blinking = True
        self._state.blink_start_time = now
        # 下次眨眼时间在当前眨眼结束后重新计算
        self._state.next_blink_time = now + self._config.duration + self._get_random_interval()

    def reset(self) -> None:
        """重置眨眼状态"""
        self._state = BlinkState()
        if self._config.enabled:
            self._state.next_blink_time = time.time() + self._get_random_interval()
