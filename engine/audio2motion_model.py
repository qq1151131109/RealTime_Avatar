"""
Audio2Motion 模型架构。

基于 Temporal Convolutional Network (TCN) 的轻量级模型，
将音频特征序列映射到 MotionVector 序列。
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class TemporalBlock(nn.Module):
    """
    TCN 的基本时间卷积块。

    包含两个因果卷积层 + 残差连接。
    """

    def __init__(
        self,
        n_inputs: int,
        n_outputs: int,
        kernel_size: int,
        stride: int,
        dilation: int,
        dropout: float = 0.2,
    ):
        super().__init__()

        padding = (kernel_size - 1) * dilation

        self.conv1 = nn.Conv1d(
            n_inputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(n_outputs)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            n_outputs, n_outputs, kernel_size,
            stride=stride, padding=padding, dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(n_outputs)
        self.dropout2 = nn.Dropout(dropout)

        # 残差连接的维度匹配
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None

        self.relu = nn.ReLU()

    def forward(self, x):
        """
        前向传播。

        参数:
            x: (batch, channels, time)

        返回:
            out: (batch, channels, time)
        """
        # 第一个卷积
        out = self.conv1(x)
        out = out[:, :, :x.size(2)]  # 裁剪到原始长度（因果性）
        out = self.bn1(out)
        out = self.relu(out)
        out = self.dropout1(out)

        # 第二个卷积
        out = self.conv2(out)
        out = out[:, :, :x.size(2)]  # 裁剪到原始长度（因果性）
        out = self.bn2(out)
        out = self.relu(out)
        out = self.dropout2(out)

        # 残差连接
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class Audio2MotionTCN(nn.Module):
    """
    Audio2Motion TCN 模型。

    输入: 音频特征序列 (batch, time, feature_dim)
    输出: MotionVector 序列 (batch, time, 7)
    """

    def __init__(
        self,
        input_dim: int = 80,  # Mel频谱维度
        hidden_dim: int = 128,
        num_layers: int = 4,
        kernel_size: int = 3,
        dropout: float = 0.2,
        output_dim: int = 7,  # MotionVector 维度
    ):
        """
        初始化模型。

        参数:
            input_dim: 输入特征维度
            hidden_dim: 隐藏层维度
            num_layers: TCN 层数
            kernel_size: 卷积核大小
            dropout: Dropout 概率
            output_dim: 输出维度（MotionVector）
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        # 输入投影
        self.input_proj = nn.Linear(input_dim, hidden_dim)

        # TCN 层
        layers = []
        num_channels = [hidden_dim] * num_layers
        for i in range(num_layers):
            dilation = 2 ** i
            in_channels = hidden_dim
            out_channels = hidden_dim
            layers.append(
                TemporalBlock(
                    in_channels, out_channels, kernel_size,
                    stride=1, dilation=dilation, dropout=dropout
                )
            )

        self.tcn = nn.Sequential(*layers)

        # 输出层
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 音频特征，shape (batch, time, input_dim)

        返回:
            motion: MotionVector，shape (batch, time, output_dim)
        """
        batch_size, time_steps, _ = x.shape

        # 输入投影
        x = self.input_proj(x)  # (batch, time, hidden_dim)

        # TCN 需要 (batch, channels, time) 格式
        x = x.transpose(1, 2)  # (batch, hidden_dim, time)

        # TCN
        x = self.tcn(x)  # (batch, hidden_dim, time)

        # 转回 (batch, time, channels)
        x = x.transpose(1, 2)  # (batch, time, hidden_dim)

        # 输出投影
        motion = self.output_proj(x)  # (batch, time, output_dim)

        return motion


class Audio2MotionLSTM(nn.Module):
    """
    Audio2Motion LSTM 模型（备选）。

    使用双向 LSTM 进行序列建模。
    """

    def __init__(
        self,
        input_dim: int = 80,
        hidden_dim: int = 128,
        num_layers: int = 2,
        dropout: float = 0.2,
        output_dim: int = 7,
        bidirectional: bool = True,
    ):
        """
        初始化模型。

        参数:
            input_dim: 输入特征维度
            hidden_dim: LSTM 隐藏层维度
            num_layers: LSTM 层数
            dropout: Dropout 概率
            output_dim: 输出维度
            bidirectional: 是否使用双向 LSTM
        """
        super().__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim
        self.bidirectional = bidirectional

        # LSTM
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            dropout=dropout if num_layers > 1 else 0,
            bidirectional=bidirectional,
            batch_first=True,
        )

        # 输出层
        lstm_output_dim = hidden_dim * 2 if bidirectional else hidden_dim
        self.output_proj = nn.Sequential(
            nn.Linear(lstm_output_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播。

        参数:
            x: 音频特征，shape (batch, time, input_dim)

        返回:
            motion: MotionVector，shape (batch, time, output_dim)
        """
        # LSTM
        lstm_out, _ = self.lstm(x)  # (batch, time, hidden_dim * 2)

        # 输出投影
        motion = self.output_proj(lstm_out)  # (batch, time, output_dim)

        return motion


def create_model(
    model_type: str = "tcn",
    input_dim: int = 80,
    hidden_dim: int = 128,
    **kwargs
) -> nn.Module:
    """
    创建模型的工厂函数。

    参数:
        model_type: 模型类型 ("tcn" 或 "lstm")
        input_dim: 输入特征维度
        hidden_dim: 隐藏层维度
        **kwargs: 其他模型参数

    返回:
        model: PyTorch 模型
    """
    if model_type == "tcn":
        return Audio2MotionTCN(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
    elif model_type == "lstm":
        return Audio2MotionLSTM(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            **kwargs
        )
    else:
        raise ValueError(f"不支持的模型类型: {model_type}")


# 测试代码
if __name__ == "__main__":
    # 测试 TCN 模型
    model = create_model("tcn", input_dim=80, hidden_dim=128, num_layers=4)
    print(f"TCN 模型参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 测试前向传播
    batch_size = 4
    time_steps = 100
    input_dim = 80

    x = torch.randn(batch_size, time_steps, input_dim)
    output = model(x)

    print(f"输入 shape: {x.shape}")
    print(f"输出 shape: {output.shape}")
    assert output.shape == (batch_size, time_steps, 7), "输出维度不正确"

    print("\n✅ 模型测试通过")
