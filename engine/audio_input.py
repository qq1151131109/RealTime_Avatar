"""
音频输入模块。

支持：
- 麦克风实时采集
- 音频文件播放
- 统一的音频流接口
"""

from pathlib import Path
from typing import Optional, Callable
from collections import deque
import threading
import time

import numpy as np


class AudioInput:
    """
    音频输入基类。

    提供统一的音频流接口，支持麦克风和文件输入。
    """

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1600):
        """
        初始化音频输入。

        参数:
            sample_rate: 采样率（Hz）
            chunk_size: 每次读取的样本数
        """
        self.sample_rate = sample_rate
        self.chunk_size = chunk_size
        self.running = False
        self.callback: Optional[Callable[[np.ndarray], None]] = None

    def start(self, callback: Callable[[np.ndarray], None]):
        """
        开始音频采集。

        参数:
            callback: 音频数据回调函数，接收 np.ndarray (num_samples,)
        """
        raise NotImplementedError

    def stop(self):
        """停止音频采集"""
        raise NotImplementedError


class MicrophoneInput(AudioInput):
    """
    麦克风实时音频输入。

    使用 pyaudio 采集麦克风音频。
    """

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1600):
        """
        初始化麦克风输入。

        参数:
            sample_rate: 采样率（Hz）
            chunk_size: 每次读取的样本数（默认 1600 = 0.1秒 @ 16kHz）
        """
        super().__init__(sample_rate, chunk_size)
        self.stream = None
        self.pyaudio = None

    def start(self, callback: Callable[[np.ndarray], None]):
        """
        开始麦克风采集。

        参数:
            callback: 音频数据回调函数
        """
        try:
            import pyaudio
        except ImportError:
            raise ImportError(
                "麦克风输入需要 pyaudio。请安装: pip install pyaudio"
            )

        self.callback = callback
        self.running = True

        self.pyaudio = pyaudio.PyAudio()

        # 打开音频流
        self.stream = self.pyaudio.open(
            format=pyaudio.paFloat32,
            channels=1,
            rate=self.sample_rate,
            input=True,
            frames_per_buffer=self.chunk_size,
            stream_callback=self._audio_callback,
        )

        print(f"[MicrophoneInput] 麦克风已启动 ({self.sample_rate} Hz)")
        self.stream.start_stream()

    def _audio_callback(self, in_data, frame_count, time_info, status):
        """PyAudio 回调函数"""
        if status:
            print(f"[MicrophoneInput] PyAudio 状态: {status}")

        # 转换为 numpy 数组
        audio_data = np.frombuffer(in_data, dtype=np.float32)

        if self.callback:
            self.callback(audio_data)

        import pyaudio
        return (None, pyaudio.paContinue)

    def stop(self):
        """停止麦克风采集"""
        self.running = False

        if self.stream:
            self.stream.stop_stream()
            self.stream.close()

        if self.pyaudio:
            self.pyaudio.terminate()

        print("[MicrophoneInput] 麦克风已停止")


class AudioFileInput(AudioInput):
    """
    音频文件播放输入。

    从音频文件读取数据，模拟实时流。
    """

    def __init__(
        self,
        file_path: str,
        sample_rate: int = 16000,
        chunk_size: int = 1600,
        loop: bool = False,
    ):
        """
        初始化音频文件输入。

        参数:
            file_path: 音频文件路径
            sample_rate: 目标采样率（Hz）
            chunk_size: 每次读取的样本数
            loop: 是否循环播放
        """
        super().__init__(sample_rate, chunk_size)
        self.file_path = Path(file_path)
        self.loop = loop
        self.audio_thread: Optional[threading.Thread] = None

        # 加载音频文件
        self._load_audio()

    def _load_audio(self):
        """加载音频文件"""
        try:
            import torchaudio
        except ImportError:
            raise ImportError(
                "音频文件输入需要 torchaudio。请安装: pip install torchaudio"
            )

        if not self.file_path.exists():
            raise FileNotFoundError(f"音频文件不存在: {self.file_path}")

        # 加载音频
        waveform, orig_sample_rate = torchaudio.load(str(self.file_path))

        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        # 重采样
        if orig_sample_rate != self.sample_rate:
            resampler = torchaudio.transforms.Resample(
                orig_freq=orig_sample_rate,
                new_freq=self.sample_rate,
            )
            waveform = resampler(waveform)

        # 转为 numpy 数组
        self.audio_data = waveform.squeeze(0).numpy()  # (num_samples,)

        duration = len(self.audio_data) / self.sample_rate
        print(f"[AudioFileInput] 加载音频: {self.file_path.name}")
        print(f"[AudioFileInput] 时长: {duration:.2f} 秒, 采样率: {self.sample_rate} Hz")

    def start(self, callback: Callable[[np.ndarray], None]):
        """
        开始播放音频文件。

        参数:
            callback: 音频数据回调函数
        """
        self.callback = callback
        self.running = True

        # 启动播放线程
        self.audio_thread = threading.Thread(target=self._playback_loop, daemon=True)
        self.audio_thread.start()

        print(f"[AudioFileInput] 开始播放 {'(循环)' if self.loop else ''}")

    def _playback_loop(self):
        """音频播放循环"""
        while self.running:
            # 从头开始播放
            position = 0

            while position < len(self.audio_data) and self.running:
                # 读取一个 chunk
                end_position = min(position + self.chunk_size, len(self.audio_data))
                chunk = self.audio_data[position:end_position]

                # 回调
                if self.callback:
                    self.callback(chunk)

                position = end_position

                # 模拟实时播放，等待相应时长
                chunk_duration = len(chunk) / self.sample_rate
                time.sleep(chunk_duration)

            # 如果不循环，结束
            if not self.loop:
                break

        print("[AudioFileInput] 播放完成")

    def stop(self):
        """停止播放"""
        self.running = False

        if self.audio_thread:
            self.audio_thread.join(timeout=2.0)

        print("[AudioFileInput] 播放已停止")


class SilentInput(AudioInput):
    """
    静音输入（用于测试）。

    生成全零音频流。
    """

    def __init__(self, sample_rate: int = 16000, chunk_size: int = 1600):
        super().__init__(sample_rate, chunk_size)
        self.thread: Optional[threading.Thread] = None

    def start(self, callback: Callable[[np.ndarray], None]):
        """开始生成静音"""
        self.callback = callback
        self.running = True

        self.thread = threading.Thread(target=self._silent_loop, daemon=True)
        self.thread.start()

        print("[SilentInput] 静音输入已启动")

    def _silent_loop(self):
        """静音生成循环"""
        while self.running:
            # 生成全零音频
            chunk = np.zeros(self.chunk_size, dtype=np.float32)

            if self.callback:
                self.callback(chunk)

            # 等待
            chunk_duration = self.chunk_size / self.sample_rate
            time.sleep(chunk_duration)

    def stop(self):
        """停止生成"""
        self.running = False

        if self.thread:
            self.thread.join(timeout=2.0)

        print("[SilentInput] 静音输入已停止")


def create_audio_input(
    mode: str = "file",
    file_path: Optional[str] = None,
    sample_rate: int = 16000,
    chunk_size: int = 1600,
    loop: bool = False,
) -> AudioInput:
    """
    创建音频输入的便捷函数。

    参数:
        mode: 输入模式 ("microphone", "file", "silent")
        file_path: 音频文件路径（mode="file" 时需要）
        sample_rate: 采样率
        chunk_size: chunk 大小
        loop: 是否循环播放（仅 file 模式）

    返回:
        audio_input: AudioInput 实例
    """
    if mode == "microphone":
        return MicrophoneInput(sample_rate, chunk_size)
    elif mode == "file":
        if file_path is None:
            raise ValueError("file 模式需要提供 file_path 参数")
        return AudioFileInput(file_path, sample_rate, chunk_size, loop)
    elif mode == "silent":
        return SilentInput(sample_rate, chunk_size)
    else:
        raise ValueError(f"未知的音频输入模式: {mode}")
