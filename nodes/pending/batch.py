"""Dynamic image-to-batch utility node."""

from __future__ import annotations

import re

import torch
import torch.nn.functional as torch_functional

from ...categories import TOOLS_BATCH


_IMAGE_INPUT = re.compile(r"^image_(\d+)$")
MAX_IMAGE_INPUTS = 10


def _validate_image(image, name: str):
    if not isinstance(image, torch.Tensor) or image.ndim != 4 or image.shape[0] < 1:
        raise ValueError(f"{name} 必须是非空 IMAGE 批次 [B, H, W, C]")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"{name} 的通道数必须为 1、3 或 4，当前为 {image.shape[-1]}")


class TUT_ImageToBatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {},
            "optional": {
                "image_1": ("IMAGE", {"tooltip": "连接后会自动增加下一个图像接口，最多 10 个。"}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("images",)
    FUNCTION = "make_batch"
    CATEGORY = TOOLS_BATCH
    DESCRIPTION = "按接口顺序将最多十个 IMAGE 输入合并为一个批次；自动统一尺寸及 RGB/RGBA 通道，连接或断开时自动增减输入接口。"

    def make_batch(self, **kwargs):
        indexed = []
        for name, image in kwargs.items():
            match = _IMAGE_INPUT.fullmatch(str(name))
            if not match or image is None:
                continue
            index = int(match.group(1))
            if index > MAX_IMAGE_INPUTS:
                raise ValueError(f"图像输入最多只能有 {MAX_IMAGE_INPUTS} 个")
            _validate_image(image, f"图像 {index}")
            indexed.append((index, image))
        if not indexed:
            raise ValueError("至少需要连接一个图像输入")
        indexed.sort(key=lambda item: item[0])

        first = indexed[0][1]
        target_height, target_width = first.shape[1:3]
        target_channels = 4 if any(image.shape[-1] == 4 for _, image in indexed) else 3
        target_device, target_dtype = first.device, first.dtype
        normalized = []
        for index, image in indexed:
            current = image.to(device=target_device, dtype=target_dtype)
            if current.shape[-1] == 1:
                current = current.repeat(1, 1, 1, 3)
            if target_channels == 4 and current.shape[-1] == 3:
                current = torch.cat((current, torch.ones_like(current[..., :1])), dim=-1)
            if current.shape[1:3] != (target_height, target_width):
                current = torch_functional.interpolate(
                    current.movedim(-1, 1), size=(target_height, target_width),
                    mode="bilinear", align_corners=False,
                ).movedim(1, -1)
            normalized.append(current)
        return (torch.cat(normalized, dim=0),)


NODE_CLASS_MAPPINGS = {"TUT_ImageToBatch": TUT_ImageToBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"TUT_ImageToBatch": "TUT_图像到批次"}
