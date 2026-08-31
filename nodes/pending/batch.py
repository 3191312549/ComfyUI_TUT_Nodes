"""Dynamic image-to-batch utility node."""

from __future__ import annotations

import re

import torch

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
    DESCRIPTION = "按接口顺序将最多十个 IMAGE 输入合并为一个批次；图片保持原始比例与像素尺寸，不同画幅会居中补透明边到最大画布。"

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
        target_height = max(int(image.shape[1]) for _, image in indexed)
        target_width = max(int(image.shape[2]) for _, image in indexed)
        has_size_mismatch = any(
            image.shape[1:3] != (target_height, target_width) for _, image in indexed
        )
        target_channels = 4 if has_size_mismatch or any(
            image.shape[-1] == 4 for _, image in indexed
        ) else 3
        target_device, target_dtype = first.device, first.dtype
        normalized = []
        for index, image in indexed:
            current = image.to(device=target_device, dtype=target_dtype)
            if current.shape[-1] == 1:
                current = current.repeat(1, 1, 1, 3)
            if target_channels == 4 and current.shape[-1] == 3:
                current = torch.cat((current, torch.ones_like(current[..., :1])), dim=-1)
            height, width = int(current.shape[1]), int(current.shape[2])
            if (height, width) == (target_height, target_width):
                normalized.append(current)
                continue
            top = (target_height - height) // 2
            left = (target_width - width) // 2
            canvas = torch.zeros(
                (current.shape[0], target_height, target_width, target_channels),
                dtype=target_dtype,
                device=target_device,
            )
            canvas[:, top:top + height, left:left + width, :] = current
            normalized.append(canvas)
        return (torch.cat(normalized, dim=0),)


NODE_CLASS_MAPPINGS = {"TUT_ImageToBatch": TUT_ImageToBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"TUT_ImageToBatch": "TUT_图像到批次"}
