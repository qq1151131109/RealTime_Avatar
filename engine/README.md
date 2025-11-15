# PC 端 2D 数字人引擎（原型）

本目录用于实现基于 LivePortrait 思路的 PC 端 2D 照片级数字人引擎原型，包括：

- Avatar 资源与配置格式
- 渲染内核（LivePortrait 推理）
- Audio2Motion 接口与实时渲染 demo
- 配置系统、性能优化、眨眼和 PTS 同步

## 快速开始

### 1. 安装依赖

```bash
cd engine
pip install -r requirements.txt
```

### 2. 配置引擎

复制示例配置文件并根据需要修改：

```bash
cp config.example.json config.json
```

配置项说明：

- `render.enable_fp16`: 启用 FP16 半精度推理（提升性能，轻微降低质量）
- `render.enable_tensorrt`: 启用 TensorRT 优化（需要安装 TensorRT）
- `blink.enabled`: 启用自动眨眼
- `blink.min_interval` / `max_interval`: 眨眼间隔范围（秒）
- `sync.use_pts`: 启用 PTS 音视频同步
- `show_fps`: 在窗口显示 FPS 和同步信息

### 3. 运行 Demo

```bash
python -m engine.demo_realtime_stub
```

按 ESC 退出。

## 主要功能

### 1. 配置系统

所有硬编码参数已移至配置文件，支持：

- 从 JSON 文件加载配置
- 运行时修改配置
- 全局默认配置

使用示例：

```python
from engine.config import EngineConfig

# 从文件加载
config = EngineConfig.load_from_file("config.json")

# 修改配置
config.render.enable_fp16 = True
config.blink.enabled = False

# 保存配置
config.save_to_file("config_custom.json")
```

### 2. 性能优化

- **FP16 半精度推理**: 设置 `render.enable_fp16 = true`，可提升 1.5-2x 性能
- **TensorRT 优化**: 设置 `render.enable_tensorrt = true`（需要额外配置）
- **批处理**: 调整 `render.batch_size`（实验性功能）

### 3. 自动眨眼

基于规则的随机眨眼，模拟自然眨眼行为：

- 自动随机间隔（2-5秒可配置）
- 平滑的眨眼动画（快速闭眼-停留-快速睁眼）
- 支持手动触发眨眼

配置示例：

```json
{
  "blink": {
    "enabled": true,
    "min_interval": 2.0,
    "max_interval": 5.0,
    "duration": 0.15,
    "eye_close_ratio": 0.8
  }
}
```

### 4. PTS 音视频同步

高精度音视频同步机制，避免音画不同步：

- 基于 Presentation Timestamp
- 自动检测和校正时间漂移
- 支持丢帧/重复帧策略

配置示例：

```json
{
  "sync": {
    "use_pts": true,
    "pts_tolerance_ms": 50,
    "drift_correction_threshold_ms": 200
  }
}
```

统计信息：

运行结束后会输出：
- 总帧数
- 平均 FPS
- 丢帧数 / 重复帧数
- 最终漂移

## 架构说明

### 模块划分

```
engine/
├── config.py              # 配置管理
├── motion.py              # MotionVector 定义（7维）
├── avatar_loader.py       # Avatar 资源加载
├── renderer.py            # 渲染器封装
├── liveportrait_backend.py # LivePortrait 后端
├── audio2motion_stub.py   # Audio2Motion 占位实现
├── blink_controller.py    # 眨眼控制器
├── pts_sync.py            # PTS 同步器
└── demo_realtime_stub.py  # 实时渲染 demo
```

### MotionVector 格式

当前版本使用 7 维向量（所有维度 [-1, 1]，眨眼除外）：

- 0: jaw_open (下巴张开)
- 1: mouth_wide (嘴横向张开)
- 2: mouth_narrow (嘴纵向收缩)
- 3: head_yaw (水平转头)
- 4: head_pitch (抬头低头)
- 5: head_roll (侧倾)
- 6: eye_blink (眨眼，[0, 1])

## 后续计划

- [x] 接入真实 Audio2Motion 模型
  - [x] 音频特征提取（Mel/MFCC/HuBERT）
  - [x] MediaPipe 数据标注脚本
  - [x] TCN/LSTM 模型架构
  - [x] 训练脚本和数据集
  - [x] 实时推理引擎
- [ ] 支持 TensorRT 加速
- [ ] 添加更多表情控制（微笑、皱眉等）
- [ ] 优化嘴型映射精度
- [ ] 支持更高分辨率（512x512）

## Audio2Motion 使用指南

### 训练自己的模型

详细步骤请参考：`docs/audio2motion_data_preparation.md`

**快速开始**:

1. 准备训练数据（说话视频）:
   ```bash
   python tools/prepare_data.py raw_videos/ --output_dir data/processed
   ```

2. 训练模型:
   ```bash
   python tools/train.py \
       --data_dir data/processed \
       --output_dir checkpoints \
       --model_type tcn \
       --feature_type mel \
       --num_epochs 100
   ```

3. 在 demo 中使用训练好的模型:
   ```python
   # 修改 demo_realtime_stub.py
   from engine.audio2motion import Audio2Motion

   audio2motion = Audio2Motion(
       model_path="checkpoints/best_model.pt",
       config=config.audio2motion
   )
   ```

### 模型架构选择

- **TCN**（推荐）: 速度快，实时性好，参数量小（~1-2M）
- **LSTM**: 精度可能更高，但推理速度较慢

### 音频特征选择

- **Mel-spectrogram**（推荐）: 80 维，包含丰富的频率信息，适合语音
- **MFCC**: 13 维，更紧凑，计算快，但信息量较少
- **HuBERT**: 768 维，预训练模型，需要额外安装 `transformers`

当前阶段包含完整的训练和推理系统，可以训练自定义模型。


