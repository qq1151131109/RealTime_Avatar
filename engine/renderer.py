from typing import Optional
from pathlib import Path

import numpy as np

from .avatar_loader import AvatarConfig
from .motion import MotionVector
from .liveportrait_backend import LivePortraitBackend, LivePortraitConfig
from .config import RenderConfig


class LivePortraitRenderer:
    """
    LivePortrait 渲染器占位实现。

    当前实现仅将源图像简单复制为输出帧，DrivingCode 未被真正使用。
    后续接入真实 LivePortrait 模型时，只需要替换 render_frame 内部逻辑。
    """

    def __init__(
        self,
        config: AvatarConfig,
        source_bgr: np.ndarray,
        render_config: Optional[RenderConfig] = None,
    ) -> None:
        self._config = config
        self._source_bgr = source_bgr
        self._render_config = render_config
        if self._render_config is None:
            from .config import get_default_config
            self._render_config = get_default_config().render

        # 尝试构建 LivePortrait 后端；失败时保持占位渲染
        self._backend: Optional[LivePortraitBackend] = None
        model_root = Path(__file__).resolve().parent / "third_party" / "LivePortrait"
        if model_root.exists():
            try:
                backend_cfg = LivePortraitConfig(
                    model_root=model_root, render_config=self._render_config
                )
                backend = LivePortraitBackend(backend_cfg)
                if backend.is_ready():
                    self._backend = backend
                    print(f"[LivePortraitRenderer] LivePortrait 后端初始化成功")
                else:
                    print(f"[LivePortraitRenderer] LivePortrait 后端未就绪，使用占位渲染")
            except Exception as e:
                # 后端初始化失败时静默降级为占位渲染
                import traceback
                print(f"[LivePortraitRenderer] LivePortrait 后端初始化失败，使用占位渲染")
                print(f"  错误: {e}")
                print(f"  详细信息:")
                traceback.print_exc()
                self._backend = None
        else:
            print(f"[LivePortraitRenderer] LivePortrait 目录不存在: {model_root}，使用占位渲染")

    @property
    def config(self) -> AvatarConfig:
        return self._config

    def render_frame(self, motion: Optional[MotionVector] = None) -> np.ndarray:
        """
        根据 driving_code 渲染一帧图像。

        - 如 LivePortrait 后端可用，则调用后端渲染；
        - 否则使用占位实现：直接返回源图像拷贝。
        """
        if self._backend is not None and self._backend.is_ready():
            return self._backend.render(self._source_bgr, motion)
        return self._source_bgr.copy()
