# 使用训练好的 Audio2Motion 模型

本指南介绍如何使用训练好的 Audio2Motion 模型驱动实时数字人。

## 快速开始

### 1. 使用占位符模式（无需训练模型）

如果你还没有训练模型，可以先使用占位符模式测试系统：

```bash
# 使用正弦波占位符
python -m engine.demo_realtime_stub
```

按 `ESC` 退出。

### 2. 使用训练好的模型 + 音频文件

训练好模型后（参考 `docs/audio2motion_data_preparation.md`），使用音频文件驱动：

```bash
python -m engine.demo_realtime \
    --model checkpoints/best_model.pt \
    --audio your_audio.wav
```

**支持的音频格式**: `.wav`, `.mp3`, `.flac`, `.ogg` 等常见格式

### 3. 使用训练好的模型 + 麦克风

实时从麦克风采集音频驱动数字人：

```bash
# 需要先安装 pyaudio
pip install pyaudio

python -m engine.demo_realtime \
    --model checkpoints/best_model.pt \
    --microphone
```

**注意**:
- macOS 可能需要在系统设置中授予麦克风权限
- Linux 需要确保 ALSA 或 PulseAudio 正确配置

## 命令行参数

### demo_realtime.py 参数

```bash
python -m engine.demo_realtime [参数]
```

| 参数 | 说明 | 默认值 |
|------|------|--------|
| `--avatar` | Avatar 资源目录路径 | `engine/avatars/demo` |
| `--model` | Audio2Motion 模型路径 | `None`（使用占位符） |
| `--audio` | 音频文件路径 | `None` |
| `--microphone` | 使用麦克风输入 | `False` |
| `--fps` | 目标帧率 | `25.0` |
| `--config` | 配置文件路径 | `None` |

**注意**: `--audio` 和 `--microphone` 不能同时使用。

## 模型降级逻辑

系统会自动检测模型是否可用：

1. **如果提供了 `--model` 且文件存在**:
   - 尝试加载训练好的模型
   - 如果加载成功，使用真实模型推理
   - 如果加载失败，降级到占位符模式

2. **如果没有提供 `--model`**:
   - 直接使用占位符模式（正弦波）

3. **如果模型加载成功但音频输入失败**:
   - 继续使用占位符模式

降级时会在终端输出警告信息，方便调试。

## 使用自定义配置

创建自定义配置文件：

```bash
cd engine
cp config.example.json my_config.json
# 编辑 my_config.json
```

使用自定义配置运行：

```bash
python -m engine.demo_realtime \
    --model checkpoints/best_model.pt \
    --audio speech.wav \
    --config engine/my_config.json
```

## 性能监控

运行时，窗口左上角会显示：

- **FPS**: 当前帧率（绿色）
- **Drift**: 音视频漂移（黄色，单位毫秒）
- **Model**: 使用的模型类型（Trained 或 Stub）
- **Audio**: 音频输入模式（microphone 或 file）

退出时会打印统计信息：

```
==================================================
运行统计:
  总时长: 10.50 秒
  总帧数: 263
  平均 FPS: 25.05
  丢帧数: 2
  重复帧数: 1
==================================================
```

## 常见问题

### 1. 麦克风无法使用

**症状**: `--microphone` 报错 "麦克风输入需要 pyaudio"

**解决方案**:

```bash
# macOS
brew install portaudio
pip install pyaudio

# Ubuntu/Debian
sudo apt-get install portaudio19-dev python3-pyaudio
pip install pyaudio

# Windows
pip install pyaudio
```

### 2. 音频文件加载失败

**症状**: `--audio` 报错 "音频文件输入需要 torchaudio"

**解决方案**:

```bash
pip install torchaudio
```

### 3. 模型加载失败

**症状**: "加载模型失败" 错误

**可能原因**:
- 模型文件损坏
- PyTorch 版本不兼容
- 模型训练时的配置与当前不匹配

**解决方案**:
1. 检查模型文件是否存在且完整
2. 确保 PyTorch 版本 >= 1.12
3. 检查模型训练时的 `feature_type` 和 `input_dim` 是否与当前一致

### 4. 音视频不同步

**症状**: 嘴型滞后或超前于音频

**可能原因**:
- 模型推理延迟过高
- PTS 同步配置不当

**解决方案**:
1. 启用 FP16 加速（修改配置文件 `render.enable_fp16: true`）
2. 调整 PTS 同步阈值（修改配置文件 `sync.drift_threshold`）
3. 降低分辨率或帧率

### 5. 嘴型不准确

**症状**: 嘴型对不上音频内容

**可能原因**:
- 训练数据不足或质量差
- 训练时的音频特征与推理时不匹配
- 模型欠拟合

**解决方案**:
1. 增加训练数据（至少 30 分钟高质量视频）
2. 确保训练和推理使用相同的 `feature_type`
3. 增大模型容量（`hidden_dim`, `num_layers`）
4. 训练更多 epoch

## 端到端示例

### 从零开始的完整流程

```bash
# 1. 准备训练视频
mkdir -p raw_videos
# 将说话视频放入 raw_videos/

# 2. 数据标注
python tools/prepare_data.py raw_videos/ --output_dir data/processed

# 3. 训练模型
python tools/train.py \
    --data_dir data/processed \
    --output_dir checkpoints \
    --model_type tcn \
    --feature_type mel \
    --num_epochs 100

# 4. 测试模型（使用音频文件）
python -m engine.demo_realtime \
    --model checkpoints/best_model.pt \
    --audio test_audio.wav

# 5. 实时测试（使用麦克风）
python -m engine.demo_realtime \
    --model checkpoints/best_model.pt \
    --microphone
```

## 下一步

- 阅读 [`docs/audio2motion_data_preparation.md`](audio2motion_data_preparation.md) 了解如何准备训练数据
- 查看 [`../engine/README.md`](../engine/README.md) 了解引擎架构
- 调整配置文件优化性能

如有问题，请检查终端输出的错误信息，或提交 Issue。
