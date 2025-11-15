"""
音频特征提取模块。

支持多种音频特征：
- Mel-spectrogram（梅尔频谱）
- MFCC（梅尔频率倒谱系数）
- HuBERT embeddings（预训练语音模型特征）
"""

from typing import Optional, Literal
import numpy as np
import torch
import torchaudio
import torchaudio.transforms as T


class AudioFeatureExtractor:
    """
    音频特征提取器。

    将原始音频波形转换为适合模型训练的特征表示。
    """

    def __init__(
        self,
        feature_type: Literal["mel", "mfcc", "hubert"] = "mel",
        sample_rate: int = 16000,
        n_mels: int = 80,
        n_mfcc: int = 13,
        hop_length: int = 160,  # 10ms at 16kHz
        win_length: int = 400,  # 25ms at 16kHz
        n_fft: int = 512,
    ):
        """
        初始化音频特征提取器。

        参数:
            feature_type: 特征类型 ("mel", "mfcc", "hubert")
            sample_rate: 采样率（Hz）
            n_mels: Mel频带数量
            n_mfcc: MFCC系数数量
            hop_length: 帧移（样本数）
            win_length: 窗口长度（样本数）
            n_fft: FFT点数
        """
        self.feature_type = feature_type
        self.sample_rate = sample_rate
        self.hop_length = hop_length
        self.win_length = win_length

        # 根据特征类型初始化转换器
        if feature_type == "mel":
            self.transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
            )
            self.feature_dim = n_mels

        elif feature_type == "mfcc":
            self.mel_transform = T.MelSpectrogram(
                sample_rate=sample_rate,
                n_fft=n_fft,
                win_length=win_length,
                hop_length=hop_length,
                n_mels=n_mels,
                power=2.0,
            )
            self.mfcc_transform = T.MFCC(
                sample_rate=sample_rate,
                n_mfcc=n_mfcc,
                melkwargs={
                    "n_fft": n_fft,
                    "n_mels": n_mels,
                    "hop_length": hop_length,
                    "win_length": win_length,
                }
            )
            self.feature_dim = n_mfcc

        elif feature_type == "hubert":
            # HuBERT 需要单独加载
            self.hubert_model = None
            self.feature_dim = 768  # HuBERT base hidden size
            print("[AudioFeatureExtractor] HuBERT 模式：需要调用 load_hubert() 加载模型")

        else:
            raise ValueError(f"不支持的特征类型: {feature_type}")

    def load_hubert(self, model_name: str = "facebook/hubert-base-ls960"):
        """加载 HuBERT 预训练模型（可选）"""
        try:
            from transformers import HubertModel
            self.hubert_model = HubertModel.from_pretrained(model_name)
            self.hubert_model.eval()
            print(f"[AudioFeatureExtractor] HuBERT 模型加载成功: {model_name}")
        except ImportError:
            raise ImportError("需要安装 transformers: pip install transformers")

    def extract_from_waveform(
        self,
        waveform: torch.Tensor,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        从波形提取特征。

        参数:
            waveform: 音频波形，shape (1, samples) 或 (samples,)
            normalize: 是否归一化

        返回:
            features: 特征矩阵，shape (time_frames, feature_dim)
        """
        if waveform.dim() == 1:
            waveform = waveform.unsqueeze(0)

        with torch.no_grad():
            if self.feature_type == "mel":
                # Mel-spectrogram
                mel_spec = self.transform(waveform)  # (1, n_mels, time)
                # 转换到 dB
                mel_spec_db = torchaudio.transforms.AmplitudeToDB()(mel_spec)
                features = mel_spec_db.squeeze(0).T  # (time, n_mels)

            elif self.feature_type == "mfcc":
                # MFCC
                mfcc = self.mfcc_transform(waveform)  # (1, n_mfcc, time)
                features = mfcc.squeeze(0).T  # (time, n_mfcc)

            elif self.feature_type == "hubert":
                if self.hubert_model is None:
                    raise RuntimeError("HuBERT 模型未加载，请先调用 load_hubert()")
                # HuBERT embeddings
                outputs = self.hubert_model(waveform)
                features = outputs.last_hidden_state.squeeze(0)  # (time, 768)

            # 归一化
            if normalize and self.feature_type in ["mel", "mfcc"]:
                mean = features.mean(dim=0, keepdim=True)
                std = features.std(dim=0, keepdim=True) + 1e-8
                features = (features - mean) / std

        return features.cpu().numpy()

    def extract_from_file(
        self,
        audio_path: str,
        normalize: bool = True,
    ) -> np.ndarray:
        """
        从音频文件提取特征。

        参数:
            audio_path: 音频文件路径
            normalize: 是否归一化

        返回:
            features: 特征矩阵，shape (time_frames, feature_dim)
        """
        # 加载音频
        waveform, sr = torchaudio.load(audio_path)

        # 重采样到目标采样率
        if sr != self.sample_rate:
            resampler = T.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        # 转为单声道
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        return self.extract_from_waveform(waveform, normalize)

    def get_frame_rate(self) -> float:
        """
        获取特征帧率（帧/秒）。

        返回:
            frame_rate: 帧率
        """
        return self.sample_rate / self.hop_length

    def samples_to_frames(self, num_samples: int) -> int:
        """将样本数转换为帧数"""
        return (num_samples - self.win_length) // self.hop_length + 1

    def frames_to_samples(self, num_frames: int) -> int:
        """将帧数转换为样本数（近似）"""
        return num_frames * self.hop_length


def extract_mel_spectrogram(
    audio_path: str,
    sample_rate: int = 16000,
    n_mels: int = 80,
    hop_length: int = 160,
) -> np.ndarray:
    """
    快捷函数：提取 Mel-spectrogram。

    参数:
        audio_path: 音频文件路径
        sample_rate: 采样率
        n_mels: Mel频带数
        hop_length: 帧移

    返回:
        mel_spec: Mel频谱，shape (time_frames, n_mels)
    """
    extractor = AudioFeatureExtractor(
        feature_type="mel",
        sample_rate=sample_rate,
        n_mels=n_mels,
        hop_length=hop_length,
    )
    return extractor.extract_from_file(audio_path)


def extract_mfcc(
    audio_path: str,
    sample_rate: int = 16000,
    n_mfcc: int = 13,
    hop_length: int = 160,
) -> np.ndarray:
    """
    快捷函数：提取 MFCC。

    参数:
        audio_path: 音频文件路径
        sample_rate: 采样率
        n_mfcc: MFCC系数数
        hop_length: 帧移

    返回:
        mfcc: MFCC特征，shape (time_frames, n_mfcc)
    """
    extractor = AudioFeatureExtractor(
        feature_type="mfcc",
        sample_rate=sample_rate,
        n_mfcc=n_mfcc,
        hop_length=hop_length,
    )
    return extractor.extract_from_file(audio_path)
