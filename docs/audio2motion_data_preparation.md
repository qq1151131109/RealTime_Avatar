# Audio2Motion 数据准备指南

本指南介绍如何准备训练数据来训练自己的 Audio2Motion 模型。

## 概述

Audio2Motion 模型需要**成对的音频-面部运动数据**进行训练:

- **输入**: 音频特征序列（Mel-spectrogram 或 MFCC）
- **输出**: MotionVector 序列（7维面部运动参数）

训练数据来源：**说话人脸视频**，我们从中同时提取音频和面部运动标签。

## 数据要求

### 视频质量要求

1. **分辨率**: 建议 ≥ 480p，人脸清晰可见
2. **帧率**: 建议 ≥ 25 FPS
3. **光照**: 均匀明亮，避免强烈阴影或背光
4. **人脸**:
   - 单人正面或半侧面（±30度内）
   - 人脸占画面的 20-40%
   - 无遮挡（眼镜可以，口罩不行）
5. **音频**:
   - 清晰的语音，无明显噪音
   - 建议为单声道或立体声
6. **内容**:
   - 包含丰富的面部表情和口型变化
   - 建议有多样的说话风格（平静、激动、疑问等）

### 数据量建议

- **最小数据量**: 至少 10 分钟的高质量视频
- **推荐数据量**: 30-60 分钟
- **理想数据量**: 2-3 小时

数据量越大，模型泛化能力越强。

### 视频来源建议

1. **自己录制**:
   - 使用手机或网络摄像头录制
   - 阅读新闻稿、书籍或自由说话
   - 确保良好的光照和音质

2. **公开视频**:
   - YouTube 访谈、演讲视频
   - 新闻主播视频
   - 教学视频
   - **注意**: 确保有权使用这些数据，遵守版权法规

3. **专业数据集**（如果可用）:
   - GRID 数据集
   - LRW (Lip Reading in the Wild)
   - VoxCeleb

## 数据准备流程

### 步骤 1: 准备视频文件

将收集的视频文件放入一个目录，例如:

```
raw_videos/
├── video_001.mp4
├── video_002.mp4
├── interview_01.mp4
└── ...
```

支持的格式: `.mp4`, `.avi`, `.mov` 等常见视频格式。

### 步骤 2: 安装依赖

确保已安装所需的 Python 库:

```bash
pip install mediapipe opencv-python numpy torchaudio
```

安装 ffmpeg（用于音频提取）:

```bash
# macOS
brew install ffmpeg

# Ubuntu/Debian
sudo apt-get install ffmpeg

# Windows
# 从 https://ffmpeg.org/download.html 下载并安装
```

### 步骤 3: 运行数据标注脚本

使用 `tools/prepare_data.py` 从视频中提取音频和 MotionVector 标签:

```bash
# 处理单个视频
python tools/prepare_data.py path/to/video.mp4 --output_dir data/processed

# 批量处理目录下所有视频
python tools/prepare_data.py raw_videos/ --output_dir data/processed
```

**参数说明**:
- `video_path`: 视频文件路径或包含视频的目录
- `--output_dir`: 输出目录（默认: `./data/processed`）
- `--no_audio`: 如果已有音频文件，可以跳过音频提取

### 步骤 4: 检查处理结果

成功处理后，`data/processed` 目录下会生成以下文件:

```
data/processed/
├── video_001_audio.wav      # 提取的音频（16kHz 单声道）
├── video_001_motion.npy     # MotionVector 序列（NumPy 数组）
├── video_001_meta.json      # 元数据（帧数、FPS 等）
├── video_002_audio.wav
├── video_002_motion.npy
├── video_002_meta.json
└── ...
```

**验证数据质量**:

```python
import numpy as np
import json

# 加载 MotionVector
motion = np.load('data/processed/video_001_motion.npy')
print(f"Motion shape: {motion.shape}")  # 应该是 (num_frames, 7)

# 加载元数据
with open('data/processed/video_001_meta.json', 'r') as f:
    meta = json.load(f)
    print(f"视频时长: {meta['duration_seconds']:.2f} 秒")
    print(f"帧数: {meta['num_frames']}")
    print(f"FPS: {meta['fps']}")

# 检查 MotionVector 的值域
print(f"jaw_open 范围: [{motion[:, 0].min():.2f}, {motion[:, 0].max():.2f}]")
print(f"mouth_wide 范围: [{motion[:, 1].min():.2f}, {motion[:, 1].max():.2f}]")
```

**健康的数据应该**:
- `motion.shape[0]` 与 `meta['num_frames']` 一致
- 各维度的值域合理（大部分在 [-1, 1] 或 [0, 1] 范围内）
- 有明显的动态变化（不是全零或常量）

## 训练模型

### 步骤 5: 开始训练

使用 `tools/train.py` 训练 Audio2Motion 模型:

```bash
python tools/train.py \
    --data_dir data/processed \
    --output_dir checkpoints \
    --model_type tcn \
    --feature_type mel \
    --hidden_dim 128 \
    --num_layers 4 \
    --batch_size 16 \
    --num_epochs 100 \
    --lr 1e-3 \
    --sequence_length 100
```

**参数说明**:

- `--data_dir`: 处理后的数据目录
- `--output_dir`: 模型保存目录
- `--model_type`: 模型类型 (`tcn` 或 `lstm`)
- `--feature_type`: 音频特征类型 (`mel` 或 `mfcc`)
- `--hidden_dim`: 隐藏层维度（越大模型越强但也越慢）
- `--num_layers`: 模型层数
- `--batch_size`: 批大小（根据 GPU 内存调整）
- `--num_epochs`: 训练轮数
- `--lr`: 学习率
- `--sequence_length`: 序列长度（帧数，100 帧 ≈ 4 秒）

### 步骤 6: 监控训练

训练过程中会显示:

```
Epoch 1/100
训练: 100%|████████| 245/245 [02:15<00:00,  1.81it/s, loss=0.0234]
训练损失: 0.025431
验证损失: 0.023876
✅ 保存最佳模型 (val_loss: 0.023876)

Epoch 2/100
...
```

**关键指标**:
- **训练损失 (train loss)**: 应该逐渐下降
- **验证损失 (val loss)**: 应该逐渐下降，与训练损失差距不要太大
- 如果验证损失不再下降或开始上升，可能出现过拟合

### 步骤 7: 使用训练好的模型

训练完成后，最佳模型保存在 `checkpoints/best_model.pt`。

在 demo 中使用:

```python
from engine.audio2motion import Audio2Motion
from engine.config import Audio2MotionConfig

# 加载模型
config = Audio2MotionConfig()
audio2motion = Audio2Motion(
    model_path="checkpoints/best_model.pt",
    config=config
)

# 启动推理
audio2motion.start()

# 推送音频
import numpy as np
audio_chunk = np.random.randn(1600)  # 0.1秒 @ 16kHz
audio2motion.push_audio(audio_chunk)

# 获取 motion
motion = audio2motion.get_motion(timestamp=0.5)
print(motion)

# 停止
audio2motion.stop()
```

## 数据标注细节

`tools/prepare_data.py` 使用 **MediaPipe Face Mesh** 提取面部关键点，然后转换为 7 维 MotionVector:

| 维度 | 参数名 | 说明 | 计算方法 |
|------|--------|------|----------|
| 0 | `jaw_open` | 张嘴程度 | 上下嘴唇距离归一化 |
| 1 | `mouth_wide` | 嘴横向宽度 | 左右嘴角距离归一化 |
| 2 | `mouth_narrow` | 嘴纵向收缩 | 预留（当前为0） |
| 3 | `head_yaw` | 左右转头 | 鼻尖 x 坐标变化 |
| 4 | `head_pitch` | 抬头低头 | 鼻尖 y 坐标变化 |
| 5 | `head_roll` | 侧倾 | 左右眼相对位置 |
| 6 | `eye_blink` | 眨眼 | 眼睛开合程度 |

这些参数的提取是基于启发式规则，可能需要根据实际效果调整 `tools/prepare_data.py` 中的归一化系数。

## 常见问题与解决方案

### 1. MediaPipe 检测不到人脸

**症状**: 处理视频时输出大量零值或重复帧

**原因**:
- 人脸太小、太侧、或有遮挡
- 光照太暗或有强烈阴影
- 视频质量太低

**解决方案**:
- 使用更高质量的视频
- 调整 MediaPipe 参数:
  ```python
  # 在 prepare_data.py 中修改
  with mp_face_mesh.FaceMesh(
      min_detection_confidence=0.3,  # 降低阈值（默认 0.5）
      min_tracking_confidence=0.3,   # 降低阈值（默认 0.5）
  ) as face_mesh:
  ```

### 2. 音频提取失败

**症状**: `ffmpeg` 报错或音频文件未生成

**原因**:
- ffmpeg 未安装
- 视频文件无音轨
- 视频编码格式不支持

**解决方案**:
- 确保 ffmpeg 已正确安装: `ffmpeg -version`
- 检查视频是否有音频: `ffplay video.mp4`
- 转换视频格式:
  ```bash
  ffmpeg -i input.avi -c:v libx264 -c:a aac output.mp4
  ```

### 3. 训练损失不下降

**症状**: loss 一直在高位震荡，不收敛

**可能原因**:
- 数据量太少
- 数据质量差（MotionVector 全零或噪声大）
- 学习率太高或太低
- 模型容量不足

**解决方案**:
- 增加更多训练数据
- 检查数据质量（参考步骤 4）
- 调整学习率: `--lr 5e-4` 或 `--lr 1e-4`
- 增大模型: `--hidden_dim 256 --num_layers 6`

### 4. 过拟合（训练损失低但验证损失高）

**症状**: 训练损失很低，但验证损失不降或上升

**解决方案**:
- 增加更多训练数据
- 增大 dropout:
  ```python
  # 在 audio2motion_model.py 中修改
  model = Audio2MotionTCN(..., dropout=0.3)  # 默认 0.2
  ```
- 减小模型容量: `--hidden_dim 64`
- 提前停止训练

### 5. 推理时嘴型不准确

**症状**: 模型训练完成，但实时推理时嘴型对不上

**可能原因**:
- 训练数据的说话风格与测试音频差异大
- 音频特征提取参数不一致
- 模型预测的延迟过大

**解决方案**:
- 使用更多样化的训练数据
- 确保推理时的音频特征参数与训练时一致
- 检查 `Audio2Motion` 的缓冲区和时间戳对齐逻辑
- 尝试不同的特征类型 (`mel` vs `mfcc`)

### 6. GPU 内存不足

**症状**: 训练时 CUDA out of memory

**解决方案**:
- 减小 batch size: `--batch_size 8` 或 `--batch_size 4`
- 减小 sequence length: `--sequence_length 50`
- 减小模型大小: `--hidden_dim 64`
- 使用 CPU 训练（会很慢）

## 高级技巧

### 数据增强

在 `Audio2MotionDataset.__getitem__()` 中添加数据增强:

```python
# 音频特征增强
if self.augment:
    # 添加高斯噪声
    audio_seq += np.random.randn(*audio_seq.shape) * 0.01

    # 时间拉伸（轻微）
    # ...
```

### 使用预训练模型

如果有相关领域的预训练模型（如语音识别模型），可以迁移学习:

```python
# 加载预训练的 HuBERT 特征
feature_extractor = AudioFeatureExtractor(feature_type="hubert")
feature_extractor.load_hubert("facebook/hubert-base-ls960")
```

### 多任务学习

同时预测 MotionVector 和其他任务（如情感分类）可以提升泛化能力。

## 总结

完整的数据准备和训练流程:

1. 收集 30-60 分钟的高质量说话人脸视频
2. 使用 `tools/prepare_data.py` 提取音频和 MotionVector
3. 验证数据质量
4. 使用 `tools/train.py` 训练模型（建议 50-100 epochs）
5. 使用最佳模型替换 `engine/audio2motion_stub.py`
6. 在实时 demo 中测试效果
7. 根据效果调整数据或模型参数，迭代优化

祝训练顺利！如有问题，请查阅代码注释或提交 Issue。
