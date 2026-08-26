"""Image, mask, text-layout, and static text-effect helpers."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .fonts import load_font


COLOR_NAMES = (
    "custom",
    "white",
    "black",
    "red",
    "green",
    "blue",
    "yellow",
    "cyan",
    "magenta",
    "orange",
    "purple",
    "pink",
    "gray",
    "gold",
    "silver",
    "navy",
    "teal",
)

COLOR_MAP = {
    "white": (255, 255, 255),
    "black": (0, 0, 0),
    "red": (255, 0, 0),
    "green": (0, 128, 0),
    "blue": (0, 0, 255),
    "yellow": (255, 255, 0),
    "cyan": (0, 255, 255),
    "magenta": (255, 0, 255),
    "orange": (255, 165, 0),
    "purple": (128, 0, 128),
    "pink": (255, 192, 203),
    "gray": (128, 128, 128),
    "gold": (255, 215, 0),
    "silver": (192, 192, 192),
    "navy": (0, 0, 128),
    "teal": (0, 128, 128),
}


def parse_color(name: str, custom_hex: str, default=(0, 0, 0)) -> tuple[int, int, int]:
    if name != "custom":
        return COLOR_MAP.get(name, default)

    value = str(custom_hex).strip().lstrip("#")
    if len(value) == 3:
        value = "".join(char * 2 for char in value)
    if len(value) != 6:
        raise ValueError(f"颜色必须是 #RGB 或 #RRGGBB：{custom_hex!r}")
    try:
        return tuple(int(value[index : index + 2], 16) for index in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"颜色包含无效字符：{custom_hex!r}") from exc


def image_tensor_to_pil_batch(images) -> list[Image.Image]:
    array = images.detach().cpu().float().numpy() if hasattr(images, "detach") else np.asarray(images)
    if array.ndim == 3:
        array = array[None, ...]
    if array.ndim != 4 or array.shape[0] == 0:
        raise ValueError("IMAGE 必须是非空批次 [B, H, W, C]")

    array = np.clip(array * 255.0, 0, 255).round().astype(np.uint8)
    result = []
    for frame in array:
        channels = frame.shape[-1]
        if channels == 1:
            image = Image.fromarray(frame[..., 0], "L").convert("RGB")
        elif channels == 3:
            image = Image.fromarray(frame, "RGB")
        elif channels == 4:
            rgba = Image.fromarray(frame, "RGBA")
            background = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
            image = Image.alpha_composite(background, rgba).convert("RGB")
        else:
            raise ValueError(f"不支持 {channels} 通道 IMAGE")
        result.append(image)
    return result


def pil_batch_to_image_tensor(images: Iterable[Image.Image]):
    arrays = [np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0 for image in images]
    if not arrays:
        raise ValueError("不能输出空 IMAGE 批次")
    return torch.from_numpy(np.stack(arrays, axis=0))


def mask_tensor_to_pil_batch(masks) -> list[Image.Image]:
    array = masks.detach().cpu().float().numpy() if hasattr(masks, "detach") else np.asarray(masks)
    if array.ndim == 2:
        array = array[None, ...]
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3 or array.shape[0] == 0:
        raise ValueError("MASK 必须是非空批次 [B, H, W]")
    array = np.clip(array * 255.0, 0, 255).round().astype(np.uint8)
    return [Image.fromarray(mask, "L") for mask in array]


def pil_batch_to_mask_tensor(masks: Iterable[Image.Image]):
    arrays = [np.asarray(mask.convert("L"), dtype=np.float32) / 255.0 for mask in masks]
    if not arrays:
        raise ValueError("不能输出空 MASK 批次")
    return torch.from_numpy(np.stack(arrays, axis=0))


def broadcast_batches(*batches):
    lengths = [len(batch) for batch in batches]
    target = max(lengths)
    if any(length not in (1, target) for length in lengths):
        raise ValueError(f"批次数量无法匹配：{lengths}；只允许长度相同或单帧广播")
    return tuple(batch * target if len(batch) == 1 else batch for batch in batches)


def _text_width(draw: ImageDraw.ImageDraw, text: str, font, letter_spacing: int) -> float:
    if not text:
        return 0.0
    widths = [float(draw.textlength(char, font=font)) for char in text]
    return sum(widths) + max(0, len(text) - 1) * letter_spacing


def _draw_spaced_text(draw, position, text, font, fill, letter_spacing):
    x, y = position
    if not text:
        return
    if letter_spacing == 0:
        draw.text((x, y), text, fill=fill, font=font, anchor="lt")
        return
    for char in text:
        draw.text((x, y), char, fill=fill, font=font, anchor="lt")
        x += float(draw.textlength(char, font=font)) + letter_spacing


def render_text_mask(
    size: tuple[int, int],
    text: str,
    font_name: str,
    font_size: int,
    align: str = "center",
    justify: str = "center",
    margins: int = 0,
    line_spacing: int = 0,
    letter_spacing: int = 0,
    position_x: int = 0,
    position_y: int = 0,
    rotation_angle: float = 0.0,
    rotation_options: str = "text center",
) -> Image.Image:
    width, height = size
    mask = Image.new("L", size, 0)
    if not str(text):
        return mask

    font = load_font(font_name, font_size)
    draw = ImageDraw.Draw(mask)
    lines = str(text).split("\n")

    sample_box = draw.textbbox((0, 0), "Ag", font=font, anchor="lt")
    base_line_height = max(1, sample_box[3] - sample_box[1])
    line_heights = []
    for line in lines:
        line_box = draw.textbbox((0, 0), line or "Ag", font=font, anchor="lt")
        line_heights.append(max(base_line_height, line_box[3] - line_box[1]))
    total_height = sum(line_heights) + int(line_spacing) * max(0, len(lines) - 1)

    if align == "top":
        start_y = margins + position_y
    elif align == "bottom":
        start_y = height - margins - total_height + position_y
    else:
        start_y = (height - total_height) / 2 + position_y

    plotted_boxes = []
    current_y = start_y
    for index, line in enumerate(lines):
        line_width = _text_width(draw, line, font, int(letter_spacing))
        if justify == "left":
            x = margins + position_x
        elif justify == "right":
            x = width - margins - line_width + position_x
        else:
            x = (width - line_width) / 2 + position_x
        y = current_y
        _draw_spaced_text(draw, (x, y), line, font, 255, int(letter_spacing))
        plotted_boxes.append((x, y, x + line_width, y + line_heights[index]))
        current_y += line_heights[index] + int(line_spacing)

    if float(rotation_angle) != 0.0:
        if rotation_options == "image center":
            center = (width / 2, height / 2)
        else:
            left = min(box[0] for box in plotted_boxes)
            top = min(box[1] for box in plotted_boxes)
            right = max(box[2] for box in plotted_boxes)
            bottom = max(box[3] for box in plotted_boxes)
            center = ((left + right) / 2, (top + bottom) / 2)
        resampling = getattr(Image, "Resampling", Image).BICUBIC
        mask = mask.rotate(float(rotation_angle), resample=resampling, center=center, expand=False, fillcolor=0)
    return mask


def _scaled_mask(mask: Image.Image, factor: float) -> Image.Image:
    return mask.point(lambda value: max(0, min(255, round(value * factor))))


def _offset_mask(mask: Image.Image, offset_x: int, offset_y: int) -> Image.Image:
    shifted = Image.new("L", mask.size, 0)
    shifted.paste(mask, (int(offset_x), int(offset_y)))
    return shifted


def _gradient(size, color_1, color_2, mode, angle):
    width, height = size
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    if mode == "radial_gradient":
        cx, cy = (width - 1) / 2, (height - 1) / 2
        maximum = max(1.0, math.hypot(max(cx, 1), max(cy, 1)))
        ratio = np.clip(np.hypot(xx - cx, yy - cy) / maximum, 0.0, 1.0)
    else:
        radians = math.radians(float(angle))
        projection = (xx - (width - 1) / 2) * math.cos(radians) + (yy - (height - 1) / 2) * math.sin(radians)
        minimum, maximum = float(projection.min()), float(projection.max())
        ratio = (projection - minimum) / max(1e-6, maximum - minimum)
    first = np.asarray(color_1, dtype=np.float32)
    second = np.asarray(color_2, dtype=np.float32)
    pixels = first[None, None, :] * (1.0 - ratio[..., None]) + second[None, None, :] * ratio[..., None]
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8), "RGB")


def _solid(size, color):
    return Image.new("RGB", size, color)


def _blend(background: Image.Image, foreground: Image.Image, mode: str) -> Image.Image:
    background = background.convert("RGB")
    foreground = foreground.convert("RGB")
    if mode == "multiply":
        return ImageChops.multiply(background, foreground)
    if mode == "screen":
        return ImageChops.screen(background, foreground)
    if mode == "overlay":
        base = np.asarray(background, dtype=np.float32) / 255.0
        top = np.asarray(foreground, dtype=np.float32) / 255.0
        result = np.where(base <= 0.5, 2 * base * top, 1 - 2 * (1 - base) * (1 - top))
        return Image.fromarray(np.clip(result * 255.0, 0, 255).astype(np.uint8), "RGB")
    return foreground


def _composite_layer(background, foreground, alpha, blend_mode="normal"):
    blended = _blend(background, foreground, blend_mode)
    return Image.composite(blended, background, alpha.convert("L"))


def _preset_values(effect_preset: str, values: dict) -> dict:
    values = dict(values)
    preset = str(effect_preset)
    if preset == "none":
        values.update(
            outline_width=0,
            outer_outline2_width=0,
            inner_outline_width=0,
            highlight_strength=0.0,
            glow_radius=0,
            shadow_opacity=0.0,
            emboss_depth=0,
            extrude_depth=0,
        )
    elif preset == "outline":
        values["outline_width"] = max(2, int(values["outline_width"]))
    elif preset == "shadow":
        values["shadow_opacity"] = max(0.65, float(values["shadow_opacity"]))
        values["shadow_offset_x"] = int(values["shadow_offset_x"]) or 8
        values["shadow_offset_y"] = int(values["shadow_offset_y"]) or 8
        values["shadow_blur"] = max(3, int(values["shadow_blur"]))
    elif preset == "glow":
        values["glow_radius"] = max(8, int(values["glow_radius"]))
        values["glow_strength"] = max(1.0, float(values["glow_strength"]))
    elif preset == "neon":
        values["outline_width"] = max(2, int(values["outline_width"]))
        values["glow_radius"] = max(12, int(values["glow_radius"]))
        values["glow_strength"] = max(1.4, float(values["glow_strength"]))
    elif preset == "gradient":
        values["fill_mode"] = "linear_gradient"
    elif preset == "emboss":
        values["emboss_depth"] = max(2, int(values["emboss_depth"]))
    elif preset == "extrude":
        values["extrude_depth"] = max(6, int(values["extrude_depth"]))
        values["extrude_offset_x"] = int(values["extrude_offset_x"]) or 1
        values["extrude_offset_y"] = int(values["extrude_offset_y"]) or 1
    return values


def compose_text_effect(
    background: Image.Image,
    text_mask: Image.Image,
    *,
    fill_image: Image.Image | None = None,
    effect_preset="none",
    fill_mode="solid",
    font_color="white",
    font_color_hex="#ffffff",
    gradient_color="blue",
    gradient_color_hex="#0066ff",
    gradient_angle=0.0,
    text_opacity=1.0,
    outline_width=0,
    outline_color="black",
    outline_color_hex="#000000",
    outer_outline2_width=0,
    outer_outline2_color="white",
    outer_outline2_color_hex="#ffffff",
    outline_gap=0,
    inner_outline_width=0,
    inner_outline_color="black",
    inner_outline_color_hex="#000000",
    highlight_offset_x=-2,
    highlight_offset_y=-2,
    highlight_strength=0.0,
    glow_radius=0,
    glow_strength=1.0,
    glow_color="cyan",
    glow_color_hex="#00ffff",
    shadow_offset_x=0,
    shadow_offset_y=0,
    shadow_blur=0,
    shadow_opacity=0.0,
    shadow_color="black",
    shadow_color_hex="#000000",
    emboss_depth=0,
    extrude_depth=0,
    extrude_offset_x=1,
    extrude_offset_y=1,
    extrude_color="gray",
    extrude_color_hex="#808080",
    blend_mode="normal",
    layer_opacity=1.0,
    **_,
) -> Image.Image:
    background = background.convert("RGB")
    if text_mask.size != background.size:
        text_mask = text_mask.resize(background.size, getattr(Image, "Resampling", Image).BILINEAR)
    text_mask = text_mask.convert("L")

    values = _preset_values(effect_preset, locals())
    fill_mode = values["fill_mode"]
    outline_width = int(values["outline_width"])
    outer_outline2_width = int(values["outer_outline2_width"])
    outline_gap = int(values["outline_gap"])
    inner_outline_width = int(values["inner_outline_width"])
    highlight_offset_x = int(values["highlight_offset_x"])
    highlight_offset_y = int(values["highlight_offset_y"])
    highlight_strength = float(values["highlight_strength"])
    glow_radius = int(values["glow_radius"])
    glow_strength = float(values["glow_strength"])
    shadow_offset_x = int(values["shadow_offset_x"])
    shadow_offset_y = int(values["shadow_offset_y"])
    shadow_blur = int(values["shadow_blur"])
    shadow_opacity = float(values["shadow_opacity"])
    emboss_depth = int(values["emboss_depth"])
    extrude_depth = int(values["extrude_depth"])
    extrude_offset_x = int(values["extrude_offset_x"])
    extrude_offset_y = int(values["extrude_offset_y"])

    original_background = background.copy()
    result = background
    font_rgb = parse_color(font_color, font_color_hex, (255, 255, 255))
    gradient_rgb = parse_color(gradient_color, gradient_color_hex, (0, 102, 255))
    outline_rgb = parse_color(outline_color, outline_color_hex, (0, 0, 0))
    outer_outline2_rgb = parse_color(outer_outline2_color, outer_outline2_color_hex, (255, 255, 255))
    inner_outline_rgb = parse_color(inner_outline_color, inner_outline_color_hex, (0, 0, 0))
    glow_rgb = parse_color(glow_color, glow_color_hex, (0, 255, 255))
    shadow_rgb = parse_color(shadow_color, shadow_color_hex, (0, 0, 0))
    extrude_rgb = parse_color(extrude_color, extrude_color_hex, (128, 128, 128))

    if shadow_opacity > 0:
        alpha = _offset_mask(text_mask, shadow_offset_x, shadow_offset_y)
        if shadow_blur > 0:
            alpha = alpha.filter(ImageFilter.GaussianBlur(shadow_blur))
        alpha = _scaled_mask(alpha, max(0.0, min(1.0, shadow_opacity)))
        result = _composite_layer(result, _solid(result.size, shadow_rgb), alpha)

    if glow_radius > 0:
        alpha = text_mask.filter(ImageFilter.GaussianBlur(glow_radius))
        alpha = _scaled_mask(alpha, max(0.0, glow_strength))
        result = _composite_layer(result, _solid(result.size, glow_rgb), alpha, "screen")

    if extrude_depth > 0:
        extrusion = Image.new("L", text_mask.size, 0)
        for depth in range(extrude_depth, 0, -1):
            shifted = _offset_mask(text_mask, extrude_offset_x * depth, extrude_offset_y * depth)
            extrusion = ImageChops.lighter(extrusion, shifted)
        extrusion = ImageChops.subtract(extrusion, text_mask)
        result = _composite_layer(result, _solid(result.size, extrude_rgb), extrusion)

    if outer_outline2_width > 0:
        inner_radius = max(0, outline_width + outline_gap)
        outer_radius = inner_radius + outer_outline2_width
        outer_expanded = text_mask.filter(ImageFilter.MaxFilter(outer_radius * 2 + 1))
        if inner_radius > 0:
            inner_expanded = text_mask.filter(ImageFilter.MaxFilter(inner_radius * 2 + 1))
        else:
            inner_expanded = text_mask
        outer_outline2 = ImageChops.subtract(outer_expanded, inner_expanded)
        result = _composite_layer(result, _solid(result.size, outer_outline2_rgb), outer_outline2)

    if outline_width > 0:
        filter_size = max(3, outline_width * 2 + 1)
        expanded = text_mask.filter(ImageFilter.MaxFilter(filter_size))
        outline = ImageChops.subtract(expanded, text_mask)
        result = _composite_layer(result, _solid(result.size, outline_rgb), outline)

    if inner_outline_width > 0:
        eroded = text_mask.filter(ImageFilter.MinFilter(inner_outline_width * 2 + 1))
        inner_outline = ImageChops.subtract(text_mask, eroded)
        result = _composite_layer(result, _solid(result.size, inner_outline_rgb), inner_outline)
        fill_mask = eroded
    else:
        fill_mask = text_mask

    if fill_image is not None and fill_mode == "image":
        fill = fill_image.convert("RGB")
        if fill.size != result.size:
            fill = fill.resize(result.size, getattr(Image, "Resampling", Image).LANCZOS)
    elif fill_mode in ("linear_gradient", "radial_gradient"):
        fill = _gradient(result.size, font_rgb, gradient_rgb, fill_mode, gradient_angle)
    else:
        fill = _solid(result.size, font_rgb)

    alpha = _scaled_mask(fill_mask, max(0.0, min(1.0, float(text_opacity))))
    result = _composite_layer(result, fill, alpha, blend_mode)

    if highlight_strength > 0:
        shifted = _offset_mask(text_mask, highlight_offset_x, highlight_offset_y)
        highlight = ImageChops.subtract(text_mask, shifted)
        highlight = _scaled_mask(highlight, max(0.0, min(1.0, highlight_strength)))
        result = _composite_layer(result, _solid(result.size, (255, 255, 255)), highlight, "screen")

    if emboss_depth > 0:
        highlight = ImageChops.subtract(text_mask, _offset_mask(text_mask, emboss_depth, emboss_depth))
        shade = ImageChops.subtract(text_mask, _offset_mask(text_mask, -emboss_depth, -emboss_depth))
        result = _composite_layer(result, _solid(result.size, (255, 255, 255)), _scaled_mask(highlight, 0.55), "screen")
        result = _composite_layer(result, _solid(result.size, (0, 0, 0)), _scaled_mask(shade, 0.5), "multiply")

    layer_opacity = max(0.0, min(1.0, float(layer_opacity)))
    if layer_opacity < 1.0:
        result = Image.blend(original_background, result, layer_opacity)
    return result
