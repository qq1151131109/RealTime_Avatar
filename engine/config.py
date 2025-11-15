"""
全局配置管理模块。

提供配置类和默认配置，支持从文件加载和运行时修改。
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional
import json


@dataclass
class RenderConfig:
    """渲染器配置"""
    # 性能相关
    enable_fp16: bool = False  # 启用半精度推理
    enable_tensorrt: bool = False  # 启用 TensorRT 优化
    batch_size: int = 1  # 批处理大小

    # LivePortrait 相关
    max_yaw_deg: float = 20.0  # 最大左右转头角度
    max_pitch_deg: float = 15.0  # 最大上下抬头角度
    max_roll_deg: float = 10.0  # 最大侧倾角度
    jaw_amp_factor: float = 0.02  # 张嘴幅度系数（回退模式）

    # 嘴型映射参数
    lip_source_ratio: float = 1.0  # 源图嘴型基准比例
    lip_min_threshold: float = -0.8  # 最小张嘴阈值
    lip_ratio_multiplier: float = 2.0  # 嘴型比例放大系数


@dataclass
class Audio2MotionConfig:
    """Audio2Motion 配置"""
    target_fps: float = 25.0  # 目标帧率
    max_queue_size: int = 100  # 音频队列最大长度
    look_ahead_ms: int = 100  # 前瞻时长（毫秒）


@dataclass
class BlinkConfig:
    """眨眼配置"""
    enabled: bool = True  # 是否启用眨眼
    min_interval: float = 2.0  # 最小眨眼间隔（秒）
    max_interval: float = 5.0  # 最大眨眼间隔（秒）
    duration: float = 0.15  # 眨眼持续时间（秒）
    eye_close_ratio: float = 0.8  # 眨眼时的闭眼比例


@dataclass
class SyncConfig:
    """音视频同步配置"""
    use_pts: bool = True  # 使用 PTS 时间戳同步
    pts_tolerance_ms: int = 50  # PTS 误差容忍度（毫秒）
    drift_correction_threshold_ms: int = 200  # 漂移校正阈值（毫秒）


@dataclass
class EngineConfig:
    """引擎全局配置"""
    render: RenderConfig = field(default_factory=RenderConfig)
    audio2motion: Audio2MotionConfig = field(default_factory=Audio2MotionConfig)
    blink: BlinkConfig = field(default_factory=BlinkConfig)
    sync: SyncConfig = field(default_factory=SyncConfig)

    # 调试选项
    verbose: bool = False  # 详细日志
    show_fps: bool = True  # 显示 FPS
    warn_on_slow_frame: bool = True  # 帧率过慢时警告

    @classmethod
    def load_from_file(cls, config_path: Path) -> "EngineConfig":
        """从 JSON 文件加载配置"""
        with open(config_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        config = cls()

        # 渲染配置
        if 'render' in data:
            for k, v in data['render'].items():
                if hasattr(config.render, k):
                    setattr(config.render, k, v)

        # Audio2Motion 配置
        if 'audio2motion' in data:
            for k, v in data['audio2motion'].items():
                if hasattr(config.audio2motion, k):
                    setattr(config.audio2motion, k, v)

        # 眨眼配置
        if 'blink' in data:
            for k, v in data['blink'].items():
                if hasattr(config.blink, k):
                    setattr(config.blink, k, v)

        # 同步配置
        if 'sync' in data:
            for k, v in data['sync'].items():
                if hasattr(config.sync, k):
                    setattr(config.sync, k, v)

        # 调试选项
        if 'verbose' in data:
            config.verbose = data['verbose']
        if 'show_fps' in data:
            config.show_fps = data['show_fps']
        if 'warn_on_slow_frame' in data:
            config.warn_on_slow_frame = data['warn_on_slow_frame']

        return config

    def save_to_file(self, config_path: Path) -> None:
        """保存配置到 JSON 文件"""
        data = {
            'render': {
                'enable_fp16': self.render.enable_fp16,
                'enable_tensorrt': self.render.enable_tensorrt,
                'batch_size': self.render.batch_size,
                'max_yaw_deg': self.render.max_yaw_deg,
                'max_pitch_deg': self.render.max_pitch_deg,
                'max_roll_deg': self.render.max_roll_deg,
                'jaw_amp_factor': self.render.jaw_amp_factor,
                'lip_source_ratio': self.render.lip_source_ratio,
                'lip_min_threshold': self.render.lip_min_threshold,
                'lip_ratio_multiplier': self.render.lip_ratio_multiplier,
            },
            'audio2motion': {
                'target_fps': self.audio2motion.target_fps,
                'max_queue_size': self.audio2motion.max_queue_size,
                'look_ahead_ms': self.audio2motion.look_ahead_ms,
            },
            'blink': {
                'enabled': self.blink.enabled,
                'min_interval': self.blink.min_interval,
                'max_interval': self.blink.max_interval,
                'duration': self.blink.duration,
                'eye_close_ratio': self.blink.eye_close_ratio,
            },
            'sync': {
                'use_pts': self.sync.use_pts,
                'pts_tolerance_ms': self.sync.pts_tolerance_ms,
                'drift_correction_threshold_ms': self.sync.drift_correction_threshold_ms,
            },
            'verbose': self.verbose,
            'show_fps': self.show_fps,
            'warn_on_slow_frame': self.warn_on_slow_frame,
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, ensure_ascii=False)


# 全局默认配置实例
_default_config: Optional[EngineConfig] = None


def get_default_config() -> EngineConfig:
    """获取全局默认配置"""
    global _default_config
    if _default_config is None:
        _default_config = EngineConfig()
    return _default_config


def set_default_config(config: EngineConfig) -> None:
    """设置全局默认配置"""
    global _default_config
    _default_config = config
