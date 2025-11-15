import json
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import cv2
import numpy as np


@dataclass
class AvatarConfig:
    name: str
    resolution: int
    renderer_model: str
    source_image_path: Path
    background_color: Tuple[int, int, int]


def load_avatar(avatar_root: Path) -> Tuple[AvatarConfig, np.ndarray]:
    """
    从 avatar_root 目录加载配置和源图像。
    """
    avatar_root = avatar_root.resolve()
    config_path = avatar_root / "avatar.json"
    if not config_path.is_file():
        raise FileNotFoundError(f"avatar.json not found in {avatar_root}")

    with config_path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    resolution = int(data.get("resolution", 256))
    source_image_rel = data.get("source_image")
    if not source_image_rel:
        raise ValueError("source_image is not set in avatar.json")

    source_image_path = (avatar_root / source_image_rel).resolve()
    if not source_image_path.is_file():
        raise FileNotFoundError(f"source image not found: {source_image_path}")

    bg_cfg = data.get("background", {})
    bg_color = tuple(bg_cfg.get("color", [0, 0, 0]))
    if len(bg_color) != 3:
        raise ValueError("background.color must be length 3")

    config = AvatarConfig(
        name=data.get("name", "avatar"),
        resolution=resolution,
        renderer_model=data.get("renderer_model", "liveportrait_stub_v1"),
        source_image_path=source_image_path,
        background_color=(int(bg_color[0]), int(bg_color[1]), int(bg_color[2])),
    )

    image_bgr = cv2.imread(str(source_image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise RuntimeError(f"failed to read source image: {source_image_path}")

    image_bgr = cv2.resize(image_bgr, (resolution, resolution), interpolation=cv2.INTER_AREA)
    return config, image_bgr

