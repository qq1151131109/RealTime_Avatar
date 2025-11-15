from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass
class MotionVector:
    """
    简化版的 Motion 表达，占位定义（所有维度归一化到 [-1, 1] 范围）：

    - 0: jaw_open        下巴张开程度 [-1, 1]（-1=闭合，1=最大张开）
    - 1: mouth_wide      嘴横向张开程度 [-1, 1]（-1=收缩，1=最大横向张开）
    - 2: mouth_narrow    嘴纵向收缩程度 [-1, 1]（-1=放松，1=最大收缩）
    - 3: head_yaw        水平转头 [-1, 1]（-1=左转，1=右转）
    - 4: head_pitch      竖直抬头低头 [-1, 1]（-1=低头，1=抬头）
    - 5: head_roll       侧倾 [-1, 1]（-1=左倾，1=右倾）
    - 6: eye_blink       眨眼程度 [0, 1]（0=睁眼，1=闭眼）
    """
    values: np.ndarray

    @classmethod
    def dim(cls) -> int:
        return 7  # 增加了眨眼维度

    @classmethod
    def zeros(cls) -> "MotionVector":
        return cls(values=np.zeros(cls.dim(), dtype=np.float32))

    def as_numpy(self) -> np.ndarray:
        return self.values


def clamp_motion(motion: MotionVector) -> MotionVector:
    """
    将 Motion 中各维度约束在合理范围，避免异常值。
    当前实现简单裁剪到 [-1, 1]。
    """
    clamped = np.clip(motion.values, -1.0, 1.0).astype(np.float32)
    return MotionVector(values=clamped)

