# CLAUDE.md

本文件为 Claude Code (claude.ai/code) 提供在此代码库中工作的指导。

## 项目概述

这是一个基于 PC 端的 2D 照片级数字人引擎，使用 LivePortrait 作为渲染内核，配合自研 Audio2Motion 模块从音频输入驱动面部动画。系统以 ≥25 FPS 运行，延迟 100-200ms，用于实时对话应用。

**技术栈**: LivePortrait (Motion→Image) + Audio2Motion TCN/LSTM (Audio→Motion) + MediaPipe Face Mesh (数据标注) + 自定义 Avatar 格式 + 实时调度框架

**Audio2Motion**: 完整的训练和推理系统已实现，支持从原始视频数据到可用模型的全流程。

## 运行 Demo

```bash
# 安装依赖
cd engine
pip install -r requirements.txt

# 运行实时 demo（按 ESC 退出）
python -m engine.demo_realtime_stub
```

## 配置系统

所有参数通过 JSON 配置文件管理：

```bash
# 复制并自定义配置
cd engine
cp config.example.json config.json
```

编程方式加载和修改配置：

```python
from engine.config import EngineConfig

config = EngineConfig.load_from_file("config.json")
config.render.enable_fp16 = True  # 启用 FP16 获得 1.5-2x 性能提升
config.blink.enabled = False
config.save_to_file("config_custom.json")
```

## 核心架构

### 数据流管道

**推理流程**:
```
音频输入 → Audio2Motion 推理 → MotionVector (7维) → LivePortrait 后端 → 渲染帧
              ↓                        ↓
        特征提取 (Mel/MFCC)      眨眼控制器 → eye_blink 参数
              ↓                        ↓
        TCN/LSTM 模型              PTS 同步 → 帧时序控制
```

**训练流程**:
```
说话视频 → MediaPipe Face Mesh → MotionVector 标签
   ↓                                    ↓
音频提取 (ffmpeg)                       |
   ↓                                    ↓
特征提取 (Mel/MFCC)  ←──────  Audio2Motion 模型训练
                                         ↓
                                   训练好的模型
```

### 关键组件

**MotionVector (motion.py)**: 7 维向量控制所有面部运动
- 维度 0-2: jaw_open, mouth_wide, mouth_narrow (嘴唇控制)
- 维度 3-5: head_yaw, head_pitch, head_roll (头部姿态)
- 维度 6: eye_blink (眨眼，[0,1] 范围)
- 其他维度使用 [-1, 1] 范围

**LivePortraitBackend (liveportrait_backend.py)**: 封装 KlingTeam/LivePortrait
- 导入时动态添加 `third_party/LivePortrait` 到 sys.path
- 使用 `retarget_lip()` 进行精确嘴型控制
- 使用 `retarget_eye()` 控制眨眼
- 缓存源图像特征以提升性能
- 临时修改 sys.path 并在导入后清理，避免污染

**配置系统 (config.py)**: 集中式配置管理
- `RenderConfig`: LivePortrait 渲染参数（FP16、头部旋转范围、嘴型映射）
- `Audio2MotionConfig`: 音频处理（FPS、队列大小、前瞻时长）
- `BlinkConfig`: 眨眼行为（间隔、持续时间、强度）
- `SyncConfig`: PTS 音视频同步

**PTS 同步 (pts_sync.py)**: 高精度音视频同步
- 基于 Presentation Timestamp 的同步
- 自动检测和校正漂移
- 同步偏离时的丢帧/重复帧策略

**眨眼控制器 (blink_controller.py)**: 自然眨眼模拟
- 随机间隔（可配置 2-5 秒）
- 三阶段动画：30% 闭眼，40% 保持，30% 睁眼

**Audio2Motion 系统**: 完整的训练和推理管道

- **audio_features.py**: 音频特征提取
  - 支持 Mel-spectrogram (80维)、MFCC (13维)、HuBERT (768维)
  - 可配置采样率、帧移、FFT 参数
  - 自动归一化和重采样

- **audio2motion.py**: 实时推理引擎
  - 加载训练好的 TCN/LSTM 模型
  - 异步推理线程，处理流式音频
  - 音频帧率下采样（~100fps → 25fps）
  - MotionVector 时间戳对齐和线性插值
  - 自动缓冲区管理，防止内存泄漏

- **audio2motion_model.py**: 神经网络架构
  - `Audio2MotionTCN`: Temporal Convolutional Network（推荐，速度快）
  - `Audio2MotionLSTM`: 双向 LSTM（精度可能更高）
  - 输入: (batch, time, feature_dim)，输出: (batch, time, 7)

- **tools/prepare_data.py**: 数据标注脚本
  - 使用 MediaPipe Face Mesh 提取 468 个 3D 面部关键点
  - 自动转换为 7 维 MotionVector 格式
  - 使用 ffmpeg 提取同步音频 (16kHz 单声道)
  - 输出: `{name}_audio.wav`, `{name}_motion.npy`, `{name}_meta.json`

- **tools/train.py**: 训练脚本
  - `Audio2MotionDataset`: 加载音频-motion 对，支持序列分片
  - MSE 损失 + Adam 优化器 + 学习率调度
  - 自动划分训练/验证集 (90%/10%)
  - 保存最佳模型和定期检查点

### Avatar 资源格式

```
avatars/demo/
├── avatar.json          # 配置：分辨率、渲染器模型、背景
└── source.png          # 对齐的人脸图像（训练分辨率）
```

Avatar 路径必须相对于 avatar 目录指定。`avatars/demo/avatar.json` 示例：
```json
{
  "source_image": "../../../Duix-Mobile/res/avatar/Leo.jpg"
}
```

## 关键实现细节

### LivePortrait 集成

系统封装了 LivePortrait 的推理管道：

1. **模型初始化** (`liveportrait_backend.py:_init_model`):
   - 通过临时修改 sys.path 进行导入
   - 在 finally 块中清理路径以避免全局污染
   - 使用自定义标志的 `InferenceConfig`（无裁剪、无贴回、无旋转）

2. **源图缓存**（首次渲染）:
   - 提取并缓存 `f_s`（特征体积）、`x_s`（标准关键点）、`source_kp_info`
   - 这些被所有后续帧重用

3. **逐帧渲染**:
   - 对 `source_kp_info` 应用头部姿态增量
   - 调用 `transform_keypoint()` 获取驱动关键点 `x_d`
   - 使用 `retarget_lip()` 应用精确的嘴部运动
   - 使用 `retarget_eye()` 应用眨眼
   - 调用 `warp_decode()` 生成最终帧

### MotionVector 映射

LivePortrait 兼容性的关键范围：
- `jaw_open`: [-1, 1] 映射到 lip_close_ratio [0, lip_ratio_multiplier]（默认 2.0）
- `head_yaw/pitch/roll`: [-1, 1] 映射到 ±max_deg（默认 20°/15°/10°）
- 仅在 `jaw_open > lip_min_threshold`（默认 -0.8）时应用嘴唇重定向
- 眼睛比例是反向的：`eye_blink=0` → 比例高（睁开），`eye_blink=1` → 比例低（闭合）

### Audio2Motion 训练和推理

**完整的训练流程**：

1. **准备训练数据**（详见 `docs/audio2motion_data_preparation.md`）:
   ```bash
   # 处理说话视频，提取音频和 MotionVector 标签
   python tools/prepare_data.py raw_videos/ --output_dir data/processed
   ```

2. **训练模型**:
   ```bash
   python tools/train.py \
       --data_dir data/processed \
       --output_dir checkpoints \
       --model_type tcn \
       --feature_type mel \
       --hidden_dim 128 \
       --num_layers 4 \
       --batch_size 16 \
       --num_epochs 100
   ```

3. **使用训练好的模型**:
   ```python
   from engine.audio2motion import Audio2Motion
   from engine.config import Audio2MotionConfig

   # 替换 demo 中的 audio2motion_stub
   audio2motion = Audio2Motion(
       model_path="checkpoints/best_model.pt",
       config=Audio2MotionConfig()
   )
   audio2motion.start()

   # 推送音频
   audio2motion.push_audio(audio_chunk)  # shape: (num_samples,)

   # 获取 MotionVector
   motion = audio2motion.get_motion(timestamp=current_time)
   ```

**关键技术细节**:
- 音频特征以 ~100 fps 提取，下采样到 25 fps 与 motion 对齐
- 异步推理线程避免阻塞主渲染循环
- 使用时间戳和线性插值确保音视频同步
- MotionVector 缓冲区限制为 5 秒，防止内存泄漏
- 支持 Mel-spectrogram (推荐) 和 MFCC 两种特征

## 开发阶段（参考）

**阶段 1（已完成）**: 使用手动驱动代码的 LivePortrait 渲染
**阶段 2（已完成）**: Avatar 资源格式和加载
**阶段 3（已完成）**: 带 FPS 监控的实时渲染循环
**阶段 4（代码完成，待端到端验证）**: Audio2Motion 模型训练和集成
  - ✅ 音频特征提取（Mel/MFCC/HuBERT）
  - ✅ MediaPipe 数据标注脚本
  - ✅ TCN/LSTM 模型架构
  - ✅ 完整的训练流程
  - ✅ 实时推理引擎
  - ✅ 音频输入模块（麦克风/文件）
  - ✅ 完整版 demo (demo_realtime.py)
  - ⏳ 端到端验证（需要实际训练和测试）
**阶段 5（进行中）**: 端到端音频驱动管道
  - ⏳ 准备示例训练数据
  - ⏳ 验证完整训练流程
  - ⏳ 测试实时音视频同步效果
**阶段 6（计划中）**: ONNX 导出和 C++ SDK

### 当前状态说明

**已实现的功能**：
- ✅ 所有 Audio2Motion 相关代码已编写完成
- ✅ 支持从麦克风或音频文件获取输入
- ✅ 模型加载失败时自动降级到占位符
- ✅ 完整的错误处理和用户提示

**待验证的工作**：
- ⏳ 需要实际训练一个模型来验证完整流程
- ⏳ 需要测试音视频同步质量
- ⏳ 需要评估模型性能和嘴型准确度

**如何使用**：
- 参考 `docs/usage_guide.md` 使用新的 demo_realtime.py
- 参考 `docs/audio2motion_data_preparation.md` 训练自己的模型

## 性能优化

启用 FP16 以加快推理：
```json
{
  "render": {
    "enable_fp16": true
  }
}
```

监控性能：
- FPS 以绿色显示在窗口左上角
- 漂移（毫秒，黄色）显示音视频同步状态
- 退出时统计信息显示总帧数、丢帧、重复帧

## 常见陷阱

1. **sys.path 污染**: LivePortrait 后端临时修改 sys.path。确保在 LivePortrait 初始化之前不要导入 engine 模块，否则清理可能失败。

2. **MotionVector 维度不匹配**: 始终使用 `MotionVector.dim()` 获取当前维度。添加眨眼功能时，向量从 6 维扩展到 7 维。

3. **Avatar 路径解析**: Avatar JSON 路径相对于 avatar 目录本身，而不是项目根目录。使用 `../../../` 从 `engine/avatars/demo/` 向上到项目根目录。

4. **GPU 内存**: LivePortrait 在 GPU 内存中保留源特征。多次切换 avatar 而不清理可能导致 OOM。

5. **Audio2Motion 模型加载**: 确保模型检查点中的 `config` 字段与当前代码版本兼容。如果训练时使用了不同的 `input_dim` 或 `feature_type`，推理时必须匹配。

6. **音频特征不匹配**: 训练和推理时必须使用相同的音频特征参数（`feature_type`、`n_mels`、`hop_length` 等），否则预测结果会错误。

7. **MediaPipe 人脸检测失败**: 如果视频中人脸太小、太侧或有遮挡，MediaPipe 可能检测失败，导致大量零值或重复帧。建议使用正面清晰的高质量视频。

8. **训练数据不足**: 至少需要 10-30 分钟的说话视频才能训练出基本可用的模型。数据量太少会导致过拟合。

9. **无音频的 PTS 同步**: 当前占位符不提供真实的音频 PTS。集成真实音频时，使用实际音频时间戳调用 `pts_sync.update_audio_pts()`。

## 文件位置参考

- 核心引擎代码: `engine/` 目录
- LivePortrait 第三方库: `engine/third_party/LivePortrait/`
- Avatar 资源: `engine/avatars/`
- 配置示例: `engine/config.example.json`
- 实现路线图: `数字人引擎落地清单.md`

**Audio2Motion 相关文件**:
- 音频特征提取: `engine/audio_features.py`
- 推理引擎: `engine/audio2motion.py`
- 模型架构: `engine/audio2motion_model.py`
- 数据标注脚本: `tools/prepare_data.py`
- 训练脚本: `tools/train.py`
- 数据准备指南: `docs/audio2motion_data_preparation.md`
- 占位符实现（参考）: `engine/audio2motion_stub.py`
