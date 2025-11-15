"""LivePortrait 后端封装与接入。

当前实现基于 KlingTeam/LivePortrait 仓库的人类模型，做了一个最小可用的
单帧推理封装：

- 在初始化时通过 InferenceConfig + LivePortraitWrapper 加载模型和权重；
- 在首次渲染时对源头像做一次特征提取与 canonical keypoints 计算并缓存；
- 每帧调用 warping + decode 得到一帧结果。

改进：
- 使用 LivePortrait 内置的 retarget_lip 功能进行精确的嘴型控制；
- 将 MotionVector 的 jaw_open 等参数映射到 LivePortrait 的 lip/eye ratio。
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch

from .motion import MotionVector
from .config import RenderConfig


@dataclass
class LivePortraitConfig:
    model_root: Path
    device: str = "cuda"
    render_config: Optional[RenderConfig] = None


class LivePortraitBackend:
    """
    LivePortrait 推理后端封装。

    当前版本会：
    - 将 KlingTeam/LivePortrait 仓库加入 sys.path；
    - 使用 InferenceConfig + LivePortraitWrapper 加载人类模型；
    - 对单张源图进行一次特征/关键点计算并缓存；
    - 每帧调用 warping + decode 生成一帧结果。
    - 使用 LivePortrait 的 retarget_lip 功能进行精确的嘴型映射。
    """

    def __init__(self, config: LivePortraitConfig) -> None:
        self._config = config
        self._model_root = config.model_root.resolve()
        if not self._model_root.exists():
            raise FileNotFoundError(
                f"LivePortrait model_root 不存在，请确认路径：{self._model_root}"
            )
        self._device = config.device
        self._render_config = config.render_config
        if self._render_config is None:
            from .config import RenderConfig
            self._render_config = RenderConfig()

        self._initialized = False
        self._wrapper = None
        self._source_cache: Optional[np.ndarray] = None
        self._source_feature_3d = None
        self._source_x_s = None
        self._init_model()

    def _init_model(self) -> None:
        """初始化 LivePortrait 模型（人类模式）。"""
        try:
            import sys
            from importlib import import_module

            # 临时将 LivePortrait 仓库根目录加入 sys.path，导入后移除
            model_root_str = str(self._model_root)
            path_added = False
            if model_root_str not in sys.path:
                sys.path.insert(0, model_root_str)
                path_added = True

            try:
                InferenceConfig = import_module(
                    "src.config.inference_config"
                ).InferenceConfig
                LivePortraitWrapper = import_module(
                    "src.live_portrait_wrapper"
                ).LivePortraitWrapper

                # 创建默认推理配置，并根据我们的使用场景做少量调整
                inf_cfg = InferenceConfig()
                # 我们直接给 256x256 的已裁剪头像，不做 pasteback/crop/旋转
                inf_cfg.flag_pasteback = False
                inf_cfg.flag_do_crop = False
                inf_cfg.flag_do_rot = False

                # 性能优化：启用 FP16
                if self._render_config.enable_fp16:
                    inf_cfg.flag_use_half_precision = True
                    print(f"[LivePortraitBackend] 启用 FP16 半精度推理")

                # 设备选择：InferenceConfig 内已有 device_id/flag_force_cpu 等配置
                self._wrapper = LivePortraitWrapper(inference_cfg=inf_cfg)
                self._initialized = True
                print(f"[LivePortraitBackend] 模型加载成功，设备: {self._device}")
            finally:
                # 清理 sys.path，避免污染全局环境
                if path_added and model_root_str in sys.path:
                    sys.path.remove(model_root_str)

        except Exception as e:
            # 初始化失败时保持占位模式
            import traceback
            print(f"[LivePortraitBackend] 初始化失败，将使用占位渲染")
            print(f"  错误: {e}")
            print(f"  详细堆栈:")
            traceback.print_exc()
            self._wrapper = None
            self._initialized = False

    def is_ready(self) -> bool:
        return self._initialized

    def render(self, source_bgr: np.ndarray, motion: Optional[MotionVector]) -> np.ndarray:
        """
        使用 LivePortrait 根据 source_bgr + motion 渲染一帧图像（BGR）。

        - 如果后端未就绪，则直接返回 source_bgr 的拷贝；
        - 如果后端已就绪，则走 LivePortrait 单帧推理。
        """
        if not self._initialized or self._wrapper is None:
            return source_bgr.copy()

        return self._forward(source_bgr, motion)

    def _forward(self, source_bgr: np.ndarray, motion: Optional[MotionVector]) -> np.ndarray:
        """单帧前向推理逻辑（当前版本尚未使用 MotionVector）。"""
        assert self._wrapper is not None

        # 若源图变化，则重新计算并缓存其特征与 canonical keypoints
        if (
            self._source_cache is None
            or self._source_cache.shape != source_bgr.shape
            or not np.array_equal(self._source_cache, source_bgr)
        ):
            self._source_cache = source_bgr.copy()
            # LivePortrait 默认以 RGB 输入，这里从 BGR 转 RGB
            x_s_tensor = self._wrapper.prepare_source(source_bgr[:, :, ::-1])
            source_kp_info = self._wrapper.get_kp_info(x_s_tensor, flag_refine_info=True)
            x_s = self._wrapper.transform_keypoint(source_kp_info)
            f_s = self._wrapper.extract_feature_3d(x_s_tensor)

            # 缓存 canonical 特征与关键点信息
            self._source_feature_3d = f_s
            self._source_x_s = x_s
            self._source_kp_info = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in source_kp_info.items()
            }

        f_s = self._source_feature_3d
        x_s = self._source_x_s

        # === 将 MotionVector 映射到简单的 head pose / jaw 动作（初版近似实现） ===
        if motion is not None and hasattr(self, "_source_kp_info"):
            mv = motion.as_numpy()
            jaw_open = float(mv[0])  # [-1, 1]，-1=闭合，1=最大张开
            head_yaw = float(mv[3])  # [-1, 1]
            head_pitch = float(mv[4])
            head_roll = float(mv[5])
            eye_blink = float(mv[6]) if len(mv) > 6 else 0.0  # [0, 1]，0=睁眼，1=闭眼

            kp_info = {
                k: v.clone() if isinstance(v, torch.Tensor) else v
                for k, v in self._source_kp_info.items()
            }

            # 基于 MotionVector 调整 pitch/yaw/roll
            # 将 [-1,1] 映射到若干度数范围（从配置读取）
            max_yaw_deg = self._render_config.max_yaw_deg
            max_pitch_deg = self._render_config.max_pitch_deg
            max_roll_deg = self._render_config.max_roll_deg

            device = kp_info["pitch"].device
            dtype = kp_info["pitch"].dtype

            yaw_delta = head_yaw * max_yaw_deg
            pitch_delta = head_pitch * max_pitch_deg
            roll_delta = head_roll * max_roll_deg

            kp_info["yaw"] = kp_info["yaw"] + torch.full_like(
                kp_info["yaw"], yaw_delta, dtype=dtype, device=device
            )
            kp_info["pitch"] = kp_info["pitch"] + torch.full_like(
                kp_info["pitch"], pitch_delta, dtype=dtype, device=device
            )
            kp_info["roll"] = kp_info["roll"] + torch.full_like(
                kp_info["roll"], roll_delta, dtype=dtype, device=device
            )

            # 根据新的姿态重新计算 driving keypoints
            x_d = self._wrapper.transform_keypoint(kp_info)

            # === 精确嘴型控制：使用 LivePortrait 的 retarget_lip ===
            # jaw_open 范围 [-1, 1]，映射到 lip_close_ratio
            # lip_close_ratio 格式: [source_lip_ratio, driving_lip_ratio]
            # 较小的值表示嘴闭合，较大的值表示嘴张开
            lip_min_threshold = self._render_config.lip_min_threshold
            if jaw_open > lip_min_threshold:  # 只在嘴不完全闭合时应用
                # 将 jaw_open [lip_min_threshold, 1] 映射到 driving lip ratio [0, lip_ratio_multiplier]
                # 0 = 闭合，1 = 中性，2 = 最大张开
                jaw_range = 1.0 - lip_min_threshold
                jaw_normalized = (jaw_open - lip_min_threshold) / jaw_range
                driving_lip_ratio = jaw_normalized * self._render_config.lip_ratio_multiplier

                # 构造 lip_close_ratio tensor (1x2)
                # 第一个值是 source 的 lip ratio（从配置读取）
                # 第二个值是 driving 的 lip ratio
                source_lip_ratio = self._render_config.lip_source_ratio
                lip_close_ratio = torch.tensor(
                    [[source_lip_ratio, driving_lip_ratio]],
                    dtype=x_d.dtype,
                    device=x_d.device,
                )

                # 使用 LivePortrait 的 retarget_lip 生成嘴型 delta
                try:
                    lip_delta = self._wrapper.retarget_lip(x_d, lip_close_ratio)
                    # 将 delta 应用到 x_d
                    x_d = x_d + lip_delta
                except Exception as e:
                    # 如果 retarget_lip 失败，使用简单的关键点偏移方案
                    print(f"[Warning] retarget_lip 失败: {e}，使用简单方案")
                    y = x_d[..., 1]
                    y_center = y.mean()
                    mask_lower = y > y_center
                    jaw_amp = self._render_config.jaw_amp_factor * jaw_normalized
                    y = torch.where(mask_lower, y + jaw_amp, y)
                    x_d[..., 1] = y

            # === 眨眼控制：使用 LivePortrait 的 retarget_eye ===
            if eye_blink > 0.01:  # 眨眼阈值
                # eye_close_ratio 格式: [left_eye, right_eye, target_eye(可选)]
                # 这里简化处理，左右眼使用相同的闭眼比例
                eye_ratio = eye_blink  # [0, 1]

                # 构造 eye_close_ratio tensor (1x2)
                # 较小的值表示睁眼，较大的值表示闭眼
                # LivePortrait 的 eye ratio 是睁眼与闭眼时眼睛高度的比值
                # 需要反转：eye_blink=0 -> ratio高，eye_blink=1 -> ratio低
                left_eye_ratio = 1.0 - eye_ratio * 0.9  # 保留0.1避免完全闭合
                right_eye_ratio = left_eye_ratio

                eye_close_ratio = torch.tensor(
                    [[left_eye_ratio, right_eye_ratio]],
                    dtype=x_d.dtype,
                    device=x_d.device,
                )

                # 使用 LivePortrait 的 retarget_eye 生成眨眼 delta
                try:
                    eye_delta = self._wrapper.retarget_eye(x_d, eye_close_ratio)
                    # 将 delta 应用到 x_d
                    x_d = x_d + eye_delta
                except Exception as e:
                    # retarget_eye 失败时打印警告
                    if self._render_config.verbose if hasattr(self._render_config, 'verbose') else False:
                        print(f"[Warning] retarget_eye 失败: {e}")
        else:
            # 无 MotionVector 时，使用 canonical keypoints 自驱动
            x_d = x_s

        # driving_multiplier 目前固定为 1，可根据需要做缩放
        out_dct = self._wrapper.warp_decode(f_s, x_s, x_d)
        I_p = self._wrapper.parse_output(out_dct["out"])[0]  # HxWx3, uint8, RGB

        # 转回 BGR 以便与 OpenCV 显示一致
        return I_p[:, :, ::-1].copy()
