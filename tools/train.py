"""
Audio2Motion 模型训练脚本。

训练音频到 MotionVector 的映射模型。
"""

import argparse
import json
from pathlib import Path
from typing import Tuple, List
import random

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

import sys
sys.path.append(str(Path(__file__).parent.parent))

from engine.audio_features import AudioFeatureExtractor
from engine.audio2motion_model import create_model


class Audio2MotionDataset(Dataset):
    """
    Audio2Motion 训练数据集。

    加载音频-MotionVector 对。
    """

    def __init__(
        self,
        data_dir: str,
        feature_extractor: AudioFeatureExtractor,
        sequence_length: int = 100,  # 4秒 @ 25fps
        overlap: float = 0.5,
    ):
        """
        初始化数据集。

        参数:
            data_dir: 数据目录（包含 prepare_data.py 输出的文件）
            feature_extractor: 音频特征提取器
            sequence_length: 序列长度（帧数）
            overlap: 序列重叠比例
        """
        self.data_dir = Path(data_dir)
        self.feature_extractor = feature_extractor
        self.sequence_length = sequence_length
        self.overlap = overlap

        # 加载所有数据文件
        self.samples = self._load_samples()

        print(f"加载 {len(self.samples)} 个训练样本")

    def _load_samples(self) -> List[Tuple[Path, Path, dict]]:
        """加载所有音频-motion对"""
        samples = []

        # 查找所有的 meta 文件
        meta_files = list(self.data_dir.glob("*_meta.json"))

        for meta_file in meta_files:
            with open(meta_file, 'r') as f:
                metadata = json.load(f)

            # 对应的音频和motion文件
            base_name = meta_file.stem.replace("_meta", "")
            audio_file = self.data_dir / f"{base_name}_audio.wav"
            motion_file = self.data_dir / f"{base_name}_motion.npy"

            if audio_file.exists() and motion_file.exists():
                samples.append((audio_file, motion_file, metadata))

        return samples

    def __len__(self) -> int:
        # 每个样本可以生成多个序列片段
        total_sequences = 0
        for _, motion_file, _ in self.samples:
            motion = np.load(motion_file)
            num_frames = len(motion)
            step = int(self.sequence_length * (1 - self.overlap))
            num_seqs = max(1, (num_frames - self.sequence_length) // step + 1)
            total_sequences += num_seqs

        return total_sequences

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        获取一个训练样本。

        返回:
            audio_features: (sequence_length, feature_dim)
            motion_vector: (sequence_length, 7)
        """
        # 找到对应的文件和序列索引
        step = int(self.sequence_length * (1 - self.overlap))
        cumulative = 0

        for audio_file, motion_file, metadata in self.samples:
            motion = np.load(motion_file)
            num_frames = len(motion)
            num_seqs = max(1, (num_frames - self.sequence_length) // step + 1)

            if idx < cumulative + num_seqs:
                # 找到了对应的文件
                seq_idx = idx - cumulative

                # 计算起始帧
                start_frame = seq_idx * step
                end_frame = min(start_frame + self.sequence_length, num_frames)

                # 加载音频特征
                audio_features = self.feature_extractor.extract_from_file(str(audio_file))

                # 对齐音频特征和motion
                # 假设音频帧率和motion帧率一致（都是25fps）
                audio_seq = audio_features[start_frame:end_frame]
                motion_seq = motion[start_frame:end_frame]

                # 如果序列不够长，填充
                if len(audio_seq) < self.sequence_length:
                    pad_length = self.sequence_length - len(audio_seq)
                    audio_seq = np.pad(audio_seq, ((0, pad_length), (0, 0)), mode='edge')
                    motion_seq = np.pad(motion_seq, ((0, pad_length), (0, 0)), mode='edge')

                return (
                    torch.FloatTensor(audio_seq),
                    torch.FloatTensor(motion_seq),
                )

            cumulative += num_seqs

        # 不应该到达这里
        raise IndexError(f"索引超出范围: {idx}")


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    optimizer: optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    num_batches = 0

    pbar = tqdm(dataloader, desc="训练")

    for audio_features, motion_vectors in pbar:
        audio_features = audio_features.to(device)
        motion_vectors = motion_vectors.to(device)

        # 前向传播
        predicted_motion = model(audio_features)

        # 计算损失
        loss = criterion(predicted_motion, motion_vectors)

        # 反向传播
        optimizer.zero_grad()
        loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

        pbar.set_postfix({"loss": loss.item()})

    return total_loss / num_batches


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    """验证"""
    model.eval()
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for audio_features, motion_vectors in dataloader:
            audio_features = audio_features.to(device)
            motion_vectors = motion_vectors.to(device)

            predicted_motion = model(audio_features)
            loss = criterion(predicted_motion, motion_vectors)

            total_loss += loss.item()
            num_batches += 1

    return total_loss / num_batches


def main():
    parser = argparse.ArgumentParser(description="训练 Audio2Motion 模型")
    parser.add_argument("--data_dir", type=str, required=True,
                        help="数据目录")
    parser.add_argument("--output_dir", type=str, default="./checkpoints",
                        help="模型保存目录")
    parser.add_argument("--model_type", type=str, default="tcn",
                        choices=["tcn", "lstm"], help="模型类型")
    parser.add_argument("--feature_type", type=str, default="mel",
                        choices=["mel", "mfcc"], help="音频特征类型")
    parser.add_argument("--hidden_dim", type=int, default=128,
                        help="隐藏层维度")
    parser.add_argument("--num_layers", type=int, default=4,
                        help="模型层数")
    parser.add_argument("--batch_size", type=int, default=16,
                        help="批大小")
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="训练轮数")
    parser.add_argument("--lr", type=float, default=1e-3,
                        help="学习率")
    parser.add_argument("--sequence_length", type=int, default=100,
                        help="序列长度（帧）")

    args = parser.parse_args()

    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"使用设备: {device}")

    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 初始化特征提取器
    feature_extractor = AudioFeatureExtractor(
        feature_type=args.feature_type,
        sample_rate=16000,
        n_mels=80,
        hop_length=160,  # 10ms @ 16kHz -> ~100fps, downsample to 25fps
    )

    # 创建数据集
    print("加载数据集...")
    dataset = Audio2MotionDataset(
        data_dir=args.data_dir,
        feature_extractor=feature_extractor,
        sequence_length=args.sequence_length,
    )

    # 分割训练集和验证集
    train_size = int(0.9 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )

    print(f"训练集: {len(train_dataset)} 样本")
    print(f"验证集: {len(val_dataset)} 样本")

    # 创建模型
    model = create_model(
        model_type=args.model_type,
        input_dim=feature_extractor.feature_dim,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        output_dim=7,
    )
    model = model.to(device)

    print(f"\n模型: {args.model_type.upper()}")
    print(f"参数量: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")

    # 损失函数和优化器
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=5, verbose=True
    )

    # 训练循环
    best_val_loss = float('inf')

    for epoch in range(args.num_epochs):
        print(f"\nEpoch {epoch + 1}/{args.num_epochs}")

        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)

        # 验证
        val_loss = validate(model, val_loader, criterion, device)

        print(f"训练损失: {train_loss:.6f}")
        print(f"验证损失: {val_loss:.6f}")

        # 学习率调度
        scheduler.step(val_loss)

        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': val_loss,
                'config': {
                    'model_type': args.model_type,
                    'feature_type': args.feature_type,
                    'input_dim': feature_extractor.feature_dim,
                    'hidden_dim': args.hidden_dim,
                    'num_layers': args.num_layers,
                    'output_dim': 7,
                }
            }
            torch.save(checkpoint, output_dir / "best_model.pt")
            print(f"✅ 保存最佳模型 (val_loss: {val_loss:.6f})")

        # 定期保存检查点
        if (epoch + 1) % 10 == 0:
            checkpoint_path = output_dir / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(checkpoint, checkpoint_path)

    print("\n训练完成！")
    print(f"最佳验证损失: {best_val_loss:.6f}")


if __name__ == "__main__":
    main()
