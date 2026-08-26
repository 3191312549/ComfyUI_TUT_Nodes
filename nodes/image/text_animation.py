"""Deterministic animated text sequence node."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageChops

from ...categories import IMAGE_TEXT
from ...core.imaging import (
    COLOR_NAMES,
    compose_text_effect,
    image_tensor_to_pil_batch,
    pil_batch_to_image_tensor,
    pil_batch_to_mask_tensor,
)
from ...core.text_layout import measure_text_layout, render_unit_masks
from .text import _effect_inputs, _font_input


ANIMATIONS = ("打字机", "淡入", "滑入", "缩放弹入", "扫光")
UNITS = ("字符", "词", "行")
DIRECTIONS = ("从左", "从右", "从上", "从下")
ALIGNMENTS = ("center", "top", "bottom")
JUSTIFICATIONS = ("center", "left", "right")
_RESAMPLING = getattr(Image, "Resampling", Image)


def _help(node_name: str) -> str:
    return f"TUT_Nodes/图片/文本/{node_name}"


def _scaled_alpha(mask: Image.Image, opacity: float) -> Image.Image:
    value = max(0.0, min(1.0, float(opacity)))
    if value <= 0.0:
        return Image.new("L", mask.size, 0)
    if value >= 1.0:
        return mask.copy()
    return mask.point(lambda pixel: round(pixel * value))


def _combine_masks(size: tuple[int, int], masks: list[Image.Image]) -> Image.Image:
    combined = Image.new("L", size, 0)
    for mask in masks:
        combined = ImageChops.lighter(combined, mask)
    return combined


def _offset_mask(mask: Image.Image, x: int, y: int) -> Image.Image:
    shifted = Image.new("L", mask.size, 0)
    shifted.paste(mask, (int(x), int(y)))
    return shifted


def _scale_mask(mask: Image.Image, scale: float) -> Image.Image:
    bbox = mask.getbbox()
    if bbox is None:
        return mask.copy()
    crop = mask.crop(bbox)
    factor = max(0.01, float(scale))
    width = max(1, round(crop.width * factor))
    height = max(1, round(crop.height * factor))
    resized = crop.resize((width, height), _RESAMPLING.BICUBIC)
    center_x = (bbox[0] + bbox[2]) / 2.0
    center_y = (bbox[1] + bbox[3]) / 2.0
    output = Image.new("L", mask.size, 0)
    output.paste(resized, (round(center_x - width / 2), round(center_y - height / 2)))
    return output


def _ease_out_cubic(progress: float) -> float:
    p = max(0.0, min(1.0, float(progress)))
    return 1.0 - (1.0 - p) ** 3


def _ease_out_back(progress: float) -> float:
    p = max(0.0, min(1.0, float(progress)))
    c1 = 1.70158
    c3 = c1 + 1.0
    return 1.0 + c3 * (p - 1.0) ** 3 + c1 * (p - 1.0) ** 2


def _unit_progresses(
    frame_index: int,
    unit_count: int,
    frame_count: int,
    stagger_frames: int,
    hold_frames: int,
) -> list[float]:
    animation_frames = frame_count - hold_frames
    if frame_index >= animation_frames:
        return [1.0] * unit_count
    last_start = max(0, unit_count - 1) * stagger_frames
    duration = max(1, animation_frames - last_start)
    denominator = max(1, duration - 1)
    return [
        max(0.0, min(1.0, (frame_index - index * stagger_frames) / denominator))
        for index in range(unit_count)
    ]


def _slide_mask(mask: Image.Image, progress: float, direction: str) -> Image.Image:
    eased = _ease_out_cubic(progress)
    distance_x = max(12, round(mask.width * 0.18))
    distance_y = max(12, round(mask.height * 0.18))
    remaining = 1.0 - eased
    offsets = {
        "从左": (-round(distance_x * remaining), 0),
        "从右": (round(distance_x * remaining), 0),
        "从上": (0, -round(distance_y * remaining)),
        "从下": (0, round(distance_y * remaining)),
    }
    return _scaled_alpha(_offset_mask(mask, *offsets[direction]), eased)


def _shine_mask(mask: Image.Image, progress: float, direction: str) -> Image.Image:
    bbox = mask.getbbox()
    if bbox is None:
        return Image.new("L", mask.size, 0)
    array = np.asarray(mask, dtype=np.float32) / 255.0
    height, width = array.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    if direction in ("从左", "从右"):
        low, high = float(bbox[0]), float(max(bbox[0] + 1, bbox[2] - 1))
        coordinate = xx
    else:
        low, high = float(bbox[1]), float(max(bbox[1] + 1, bbox[3] - 1))
        coordinate = yy
    if direction in ("从右", "从下"):
        low, high = high, low
    span = abs(high - low)
    center = low + (high - low) * (-0.2 + 1.4 * max(0.0, min(1.0, progress)))
    band = max(2.0, span * 0.14)
    shine = np.clip(1.0 - np.abs(coordinate - center) / band, 0.0, 1.0)
    return Image.fromarray(np.clip(array * shine * 255.0, 0, 255).astype(np.uint8), "L")


def _animated_mask(
    size: tuple[int, int],
    unit_masks: list[Image.Image],
    progresses: list[float],
    animation: str,
    direction: str,
) -> Image.Image:
    if animation == "打字机":
        masks = [mask for mask, progress in zip(unit_masks, progresses) if progress > 0.0]
    elif animation == "淡入":
        masks = [_scaled_alpha(mask, _ease_out_cubic(progress)) for mask, progress in zip(unit_masks, progresses)]
    elif animation == "滑入":
        masks = [_slide_mask(mask, progress, direction) for mask, progress in zip(unit_masks, progresses)]
    elif animation == "缩放弹入":
        masks = [
            _scaled_alpha(_scale_mask(mask, 0.2 + 0.8 * _ease_out_back(progress)), _ease_out_cubic(progress))
            for mask, progress in zip(unit_masks, progresses)
        ]
    elif animation == "扫光":
        return _combine_masks(size, unit_masks)
    else:
        raise ValueError(f"未知动态文字模式：{animation}")
    return _combine_masks(size, masks)


class TUT_动态文字序列:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "text": ("STRING", {"multiline": True, "default": "动态文字"}),
            "font_name": _font_input(),
            "font_size": ("INT", {"default": 72, "min": 1, "max": 2048}),
            "animation": (list(ANIMATIONS),),
            "unit": (list(UNITS),),
            "frame_count": ("INT", {"default": 24, "min": 2, "max": 1000}),
            "stagger_frames": ("INT", {"default": 2, "min": 0, "max": 1000}),
            "hold_frames": ("INT", {"default": 4, "min": 0, "max": 999}),
            "direction": (list(DIRECTIONS),),
            "align": (list(ALIGNMENTS),),
            "justify": (list(JUSTIFICATIONS),),
            "margins": ("INT", {"default": 16, "min": -1024, "max": 2048}),
            "line_spacing": ("INT", {"default": 0, "min": -64, "max": 512}),
            "letter_spacing": ("INT", {"default": 0, "min": -32, "max": 256}),
            "font_color": (list(COLOR_NAMES),),
        }
        return {"required": required, "optional": _effect_inputs()}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "text_mask")
    FUNCTION = "animate_text"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "生成打字机、淡入、滑入、缩放弹入或扫光文字动画帧，不管理 FPS。"

    def animate_text(
        self,
        image,
        text,
        font_name,
        font_size,
        animation,
        unit,
        frame_count,
        stagger_frames,
        hold_frames,
        direction,
        align,
        justify,
        margins,
        line_spacing,
        letter_spacing,
        font_color,
        **effects,
    ):
        total_frames = int(frame_count)
        hold = int(hold_frames)
        stagger = int(stagger_frames)
        if hold >= total_frames:
            raise ValueError("hold_frames 必须小于 frame_count，至少保留一帧用于动画")
        if animation not in ANIMATIONS:
            raise ValueError(f"未知动态文字模式：{animation}")
        if unit not in UNITS:
            raise ValueError(f"未知文字拆分模式：{unit}")
        if direction not in DIRECTIONS:
            raise ValueError(f"未知动画方向：{direction}")

        source_frames = image_tensor_to_pil_batch(image)
        outputs: list[Image.Image] = []
        output_masks: list[Image.Image] = []
        for source in source_frames:
            margin = int(margins)
            region = (margin, margin, source.width - margin, source.height - margin)
            if region[2] <= region[0] or region[3] <= region[1]:
                raise ValueError("margins 过大，动态文字排版区域为空")
            layout = measure_text_layout(
                str(text),
                font_name,
                int(font_size),
                source.size,
                region=region,
                line_spacing=int(line_spacing),
                letter_spacing=int(letter_spacing),
                justify=justify,
                vertical_align=align,
            )
            unit_masks = render_unit_masks(layout, unit, include_whitespace=False)
            if not unit_masks:
                unit_masks = [Image.new("L", source.size, 0)]

            for frame_index in range(total_frames):
                progresses = _unit_progresses(frame_index, len(unit_masks), total_frames, stagger, hold)
                text_mask = _animated_mask(source.size, unit_masks, progresses, animation, direction)
                output = compose_text_effect(source, text_mask, font_color=font_color, **effects)
                if animation == "扫光":
                    shine_parts = [
                        _shine_mask(mask, progress, direction)
                        for mask, progress in zip(unit_masks, progresses)
                    ]
                    shine_mask = _combine_masks(source.size, shine_parts)
                    output = compose_text_effect(
                        output,
                        shine_mask,
                        effect_preset="none",
                        fill_mode="solid",
                        font_color="white",
                        text_opacity=0.75,
                        blend_mode="screen",
                    )
                outputs.append(output)
                output_masks.append(text_mask)

        return (
            pil_batch_to_image_tensor(outputs),
            _help("TUT_动态文字序列"),
            pil_batch_to_mask_tensor(output_masks),
        )


NODE_CLASS_MAPPINGS = {
    "TUT_AnimatedTextSequence": TUT_动态文字序列,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_AnimatedTextSequence": "TUT_动态文字序列",
}
