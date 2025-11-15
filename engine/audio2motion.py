"""
真实的 Audio2Motion 推理引擎。

加载训练好的模型，将流式音频转换为 MotionVector 序列。
"""

from pathlib import Path
from typing import Optional, Deque
from collections import deque
import threading
import time

import numpy as np
import torch
import torch.nn as nn

from engine.motion import MotionVector
from engine.audio_features import AudioFeatureExtractor
from engine.audio2motion_model import create_model
from engine.config import Audio2MotionConfig


class Audio2Motion:
    """
    Audio2Motion 推理引擎。

    加载训练好的模型，实时将音频流转换为 MotionVector 序列。
    """

    def __init__(
        self,
        model_path: str,
        config: Optional[Audio2MotionConfig] = None,
        device: Optional[torch.device] = None,
    ):
        """
        初始化 Audio2Motion 引擎。

        参数:
            model_path: 训练好的模型检查点路径
            config: Audio2Motion 配置
            device: 推理设备
        """
        self.config = config or Audio2MotionConfig()
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # 加载模型
        print(f"[Audio2Motion] 加载模型: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)

        model_config = checkpoint.get('config', {})
        self.model = create_model(
            model_type=model_config.get('model_type', 'tcn'),
            input_dim=model_config.get('input_dim', 80),
            hidden_dim=model_config.get('hidden_dim', 128),
            num_layers=model_config.get('num_layers', 4),
            output_dim=model_config.get('output_dim', 7),
        )

        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(self.device)
        self.model.eval()

        print(f"[Audio2Motion] 模型加载完成 (epoch {checkpoint.get('epoch', 'unknown')})")

        # 初始化音频特征提取器
        self.feature_extractor = AudioFeatureExtractor(
            feature_type=model_config.get('feature_type', 'mel'),
            sample_rate=self.config.sample_rate,
            n_mels=model_config.get('input_dim', 80),
            hop_length=self.config.hop_length,
        )

        # 音频帧率和motion帧率
        self.audio_frame_rate = self.feature_extractor.get_frame_rate()  # ~100 fps
        self.motion_fps = self.config.motion_fps  # 25 fps
        self.downsample_factor = int(self.audio_frame_rate / self.motion_fps)  # ~4

        print(f"[Audio2Motion] 音频帧率: {self.audio_frame_rate:.1f} fps")
        print(f"[Audio2Motion] Motion帧率: {self.motion_fps} fps")
        print(f"[Audio2Motion] 下采样因子: {self.downsample_factor}")

        # 音频缓冲区（原始音频样本）
        self.audio_buffer: Deque[np.ndarray] = deque()
        self.audio_buffer_lock = threading.Lock()
        self.max_buffer_samples = self.config.sample_rate * 10  # 最多缓存10秒

        # MotionVector 输出缓冲区（已预测的motion）
        self.motion_buffer: Deque[tuple[float, MotionVector]] = deque()
        self.motion_buffer_lock = threading.Lock()

        # 推理线程
        self.running = False
        self.inference_thread: Optional[threading.Thread] = None

        # 统计信息
        self.total_audio_samples = 0
        self.total_motion_frames = 0

        # 用于平滑过渡的历史特征
        self.feature_history: Deque[np.ndarray] = deque(maxlen=10)

    def start(self):
        """启动推理线程"""
        if self.running:
            return

        self.running = True
        self.inference_thread = threading.Thread(target=self._inference_loop, daemon=True)
        self.inference_thread.start()
        print("[Audio2Motion] 推理线程已启动")

    def stop(self):
        """停止推理线程"""
        if not self.running:
            return

        self.running = False
        if self.inference_thread:
            self.inference_thread.join(timeout=2.0)

        print(f"[Audio2Motion] 已停止 (处理 {self.total_audio_samples} 音频样本, "
              f"生成 {self.total_motion_frames} motion帧)")

    def push_audio(self, audio_data: np.ndarray):
        """
        推送音频数据到缓冲区。

        参数:
            audio_data: 音频样本，shape (num_samples,)，16kHz 单声道
        """
        with self.audio_buffer_lock:
            self.audio_buffer.append(audio_data)
            self.total_audio_samples += len(audio_data)

            # 限制缓冲区大小，防止内存溢出
            total_buffered = sum(len(chunk) for chunk in self.audio_buffer)
            while total_buffered > self.max_buffer_samples and len(self.audio_buffer) > 1:
                dropped = self.audio_buffer.popleft()
                total_buffered -= len(dropped)

    def get_motion(self, timestamp: float) -> MotionVector:
        """
        获取指定时间戳的 MotionVector。

        参数:
            timestamp: 时间戳（秒）

        返回:
            motion: MotionVector
        """
        with self.motion_buffer_lock:
            if not self.motion_buffer:
                # 缓冲区为空，返回中性姿态
                return MotionVector.neutral()

            # 查找最接近 timestamp 的 motion
            # motion_buffer 是按时间排序的 (timestamp, MotionVector) 对

            # 移除过时的 motion（超过1秒前的）
            min_timestamp = timestamp - 1.0
            while self.motion_buffer and self.motion_buffer[0][0] < min_timestamp:
                self.motion_buffer.popleft()

            if not self.motion_buffer:
                return MotionVector.neutral()

            # 如果 timestamp 早于第一个可用的motion，返回第一个
            if timestamp <= self.motion_buffer[0][0]:
                return self.motion_buffer[0][1]

            # 如果 timestamp 晚于最后一个motion，返回最后一个
            if timestamp >= self.motion_buffer[-1][0]:
                return self.motion_buffer[-1][1]

            # 二分查找最接近的 motion
            # 找到 timestamp 前后的两个motion，进行线性插值
            left_idx = 0
            right_idx = len(self.motion_buffer) - 1

            while right_idx - left_idx > 1:
                mid_idx = (left_idx + right_idx) // 2
                if self.motion_buffer[mid_idx][0] < timestamp:
                    left_idx = mid_idx
                else:
                    right_idx = mid_idx

            t1, motion1 = self.motion_buffer[left_idx]
            t2, motion2 = self.motion_buffer[right_idx]

            # 线性插值
            alpha = (timestamp - t1) / (t2 - t1) if t2 > t1 else 0.0
            alpha = np.clip(alpha, 0.0, 1.0)

            return MotionVector.interpolate(motion1, motion2, alpha)

    def _inference_loop(self):
        """推理线程主循环"""
        print("[Audio2Motion] 推理循环开始")

        # 用于累积足够的音频进行批处理
        accumulated_audio = np.array([], dtype=np.float32)

        # 每次处理的最小样本数（对应约1秒音频）
        min_chunk_samples = self.config.sample_rate  # 16000 samples = 1s

        while self.running:
            # 从缓冲区获取音频
            with self.audio_buffer_lock:
                if self.audio_buffer:
                    chunk = self.audio_buffer.popleft()
                    accumulated_audio = np.concatenate([accumulated_audio, chunk])
                else:
                    chunk = None

            # 如果累积的音频不够，等待更多数据
            if len(accumulated_audio) < min_chunk_samples:
                time.sleep(0.05)  # 50ms
                continue

            try:
                # 提取音频特征
                # 将 accumulated_audio 转为 torch tensor
                waveform = torch.from_numpy(accumulated_audio).unsqueeze(0)  # (1, samples)

                features = self.feature_extractor.extract_from_waveform(
                    waveform, normalize=True
                )  # (num_audio_frames, feature_dim)

                # 下采样到 motion_fps
                # 每 downsample_factor 个音频帧对应一个 motion 帧
                num_audio_frames = features.shape[0]
                num_motion_frames = num_audio_frames // self.downsample_factor

                if num_motion_frames == 0:
                    # 音频太短，等待更多
                    time.sleep(0.05)
                    continue

                # 对特征进行下采样（取平均）
                downsampled_features = []
                for i in range(num_motion_frames):
                    start_idx = i * self.downsample_factor
                    end_idx = start_idx + self.downsample_factor
                    avg_feature = features[start_idx:end_idx].mean(axis=0)
                    downsampled_features.append(avg_feature)

                downsampled_features = np.array(downsampled_features)  # (num_motion_frames, feature_dim)

                # 模型推理
                with torch.no_grad():
                    # 转为 torch tensor
                    input_tensor = torch.from_numpy(downsampled_features).unsqueeze(0).float()  # (1, time, feature_dim)
                    input_tensor = input_tensor.to(self.device)

                    # 前向传播
                    output_tensor = self.model(input_tensor)  # (1, time, 7)

                    # 转回 numpy
                    motion_vectors = output_tensor.squeeze(0).cpu().numpy()  # (num_motion_frames, 7)

                # 计算每个 motion 的时间戳
                # accumulated_audio 的起始时间戳
                samples_processed = self.total_audio_samples - len(accumulated_audio)
                start_timestamp = samples_processed / self.config.sample_rate

                # 将预测的 motion 加入缓冲区
                with self.motion_buffer_lock:
                    for i in range(num_motion_frames):
                        # 每个 motion 对应的时间戳
                        motion_timestamp = start_timestamp + (i / self.motion_fps)

                        # 创建 MotionVector
                        motion = MotionVector(
                            jaw_open=float(np.clip(motion_vectors[i, 0], -1, 1)),
                            mouth_wide=float(np.clip(motion_vectors[i, 1], -1, 1)),
                            mouth_narrow=float(np.clip(motion_vectors[i, 2], -1, 1)),
                            head_yaw=float(np.clip(motion_vectors[i, 3], -1, 1)),
                            head_pitch=float(np.clip(motion_vectors[i, 4], -1, 1)),
                            head_roll=float(np.clip(motion_vectors[i, 5], -1, 1)),
                            eye_blink=float(np.clip(motion_vectors[i, 6], 0, 1)),
                        )

                        self.motion_buffer.append((motion_timestamp, motion))
                        self.total_motion_frames += 1

                    # 限制 motion 缓冲区大小（最多保留5秒）
                    max_motion_buffer_size = int(self.motion_fps * 5)
                    while len(self.motion_buffer) > max_motion_buffer_size:
                        self.motion_buffer.popleft()

                # 移除已处理的音频
                processed_samples = num_motion_frames * self.downsample_factor * self.config.hop_length
                accumulated_audio = accumulated_audio[processed_samples:]

            except Exception as e:
                import traceback
                print(f"[Audio2Motion] 推理错误: {e}")
                traceback.print_exc()
                time.sleep(0.1)

        print("[Audio2Motion] 推理循环结束")

    def __enter__(self):
        """上下文管理器支持"""
        self.start()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """上下文管理器支持"""
        self.stop()


# 便捷函数
def create_audio2motion(
    model_path: str,
    config: Optional[Audio2MotionConfig] = None,
    device: Optional[torch.device] = None,
) -> Audio2Motion:
    """
    创建 Audio2Motion 引擎的便捷函数。

    参数:
        model_path: 模型检查点路径
        config: 配置
        device: 设备

    返回:
        audio2motion: Audio2Motion 实例
    """
    return Audio2Motion(model_path, config, device)
