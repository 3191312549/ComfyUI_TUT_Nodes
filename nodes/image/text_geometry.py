"""Geometric text nodes: tight-mask warping and glyphs laid on a path."""

from __future__ import annotations

import math

from PIL import Image, ImageChops, ImageDraw

from ...categories import IMAGE_TEXT
from ...core.fonts import load_font
from ...core.imaging import (
    COLOR_NAMES,
    broadcast_batches,
    compose_text_effect,
    image_tensor_to_pil_batch,
    mask_tensor_to_pil_batch,
    pil_batch_to_image_tensor,
    pil_batch_to_mask_tensor,
    render_text_mask,
)
from ...core.text_layout import (
    path_lengths,
    point_on_path,
    sample_mask_path,
    segment_text,
    transform_text_mask,
)
from .text import _effect_inputs, _font_input


TRANSFORMS = ("上拱", "下拱", "波浪", "斜切", "梯形", "透视")
DIRECTIONS = ("水平", "垂直")
ALIGNMENTS = ("center", "top", "bottom")
JUSTIFY = ("center", "left", "right")
ORIENTATIONS = ("跟随切线", "保持直立")
OVERFLOW_MODES = ("截断", "压缩间距", "报错")


def _help(node_name: str) -> str:
    return f"TUT_Nodes/图片/文本/{node_name}"


def _resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    if mask.size == size:
        return mask.convert("L")
    return mask.convert("L").resize(size, getattr(Image, "Resampling", Image).BILINEAR)


def _glyph_mask(text: str, font) -> tuple[Image.Image, float]:
    probe = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(probe)
    advance = float(draw.textlength(text, font=font))
    box = draw.textbbox((0, 0), text or " ", font=font, anchor="lt")
    pad = 4
    width = max(1, int(math.ceil(box[2] - box[0])) + pad * 2)
    height = max(1, int(math.ceil(box[3] - box[1])) + pad * 2)
    glyph = Image.new("L", (width, height), 0)
    glyph_draw = ImageDraw.Draw(glyph)
    glyph_draw.text((pad - box[0], pad - box[1]), text, font=font, fill=255, anchor="lt")
    return glyph, advance


def _path_text_mask(
    size: tuple[int, int],
    path_mask: Image.Image,
    text: str,
    font_name: str,
    font_size: int,
    start_offset: float,
    path_offset: float,
    letter_spacing: float,
    reverse: bool,
    orientation: str,
    overflow: str,
) -> Image.Image:
    output = Image.new("L", size, 0)
    if not str(text):
        return output

    path = sample_mask_path(path_mask)
    if reverse:
        path = list(reversed(path))
    lengths = path_lengths(path)
    path_length = float(lengths[-1])
    start = max(0.0, float(start_offset))
    available = path_length - start
    if available <= 0:
        raise ValueError(f"路径起点偏移 {start:g} 已超出路径长度 {path_length:.1f}")

    font = load_font(font_name, int(font_size))
    units = segment_text(str(text), "字符", include_whitespace=True)
    glyphs = [_glyph_mask(unit, font) for unit in units]
    advances = [advance for _, advance in glyphs]
    spacing = float(letter_spacing)

    def required(count: int, gap: float = spacing) -> float:
        return sum(advances[:count]) + max(0, count - 1) * gap

    needed = required(len(units))
    if needed > available + 1e-6:
        if overflow == "报错":
            raise ValueError(f"路径过短：文字需要 {needed:.1f}px，可用路径仅 {available:.1f}px")
        if overflow == "截断":
            count = 0
            while count < len(units) and required(count + 1) <= available + 1e-6:
                count += 1
            units, glyphs, advances = units[:count], glyphs[:count], advances[:count]
        elif overflow == "压缩间距":
            if len(units) < 2:
                raise ValueError(f"路径过短：单个字形需要 {needed:.1f}px，可用路径仅 {available:.1f}px")
            spacing = (available - sum(advances)) / (len(units) - 1)
        else:
            raise ValueError(f"未知路径溢出模式：{overflow}")

    cursor = start
    for index, (unit, (glyph, advance)) in enumerate(zip(units, glyphs)):
        center_distance = cursor + advance / 2.0
        if center_distance > path_length + 1e-6:
            break
        x, y, angle = point_on_path(path, lengths, center_distance)
        radians = math.radians(angle)
        x += -math.sin(radians) * float(path_offset)
        y += math.cos(radians) * float(path_offset)
        if unit and not unit.isspace():
            placed = glyph
            if orientation == "跟随切线":
                placed = glyph.rotate(
                    -angle,
                    resample=getattr(Image, "Resampling", Image).BICUBIC,
                    expand=True,
                    fillcolor=0,
                )
            elif orientation != "保持直立":
                raise ValueError(f"未知文字方向模式：{orientation}")
            layer = Image.new("L", size, 0)
            layer.paste(placed, (int(round(x - placed.width / 2)), int(round(y - placed.height / 2))))
            output = ImageChops.lighter(output, layer)
        cursor += advance
        if index < len(units) - 1:
            cursor += spacing
    return output


class TUT_文字变形:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "text": ("STRING", {"multiline": True, "default": "几何文字"}),
            "font_name": _font_input(),
            "font_size": ("INT", {"default": 96, "min": 1, "max": 2048}),
            "transform": (list(TRANSFORMS),),
            "strength": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 1.0, "step": 0.01}),
            "frequency": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 12.0, "step": 0.1}),
            "direction": (list(DIRECTIONS),),
            "align": (list(ALIGNMENTS),),
            "justify": (list(JUSTIFY),),
            "margins": ("INT", {"default": 16, "min": -1024, "max": 2048}),
            "line_spacing": ("INT", {"default": 0, "min": -64, "max": 512}),
            "letter_spacing": ("INT", {"default": 0, "min": -32, "max": 256}),
            "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
            "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
            "font_color": (list(COLOR_NAMES),),
        }
        return {"required": required, "optional": _effect_inputs()}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "text_mask")
    FUNCTION = "warp_text"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "仅对文字紧边界执行拱形、波浪、斜切、梯形或透视变形。"

    def warp_text(
        self, image, text, font_name, font_size, transform, strength, frequency,
        direction, align, justify, margins, line_spacing, letter_spacing,
        position_x, position_y, font_color, **effects,
    ):
        frames = image_tensor_to_pil_batch(image)
        outputs, masks = [], []
        for frame in frames:
            base_mask = render_text_mask(
                frame.size, str(text), font_name, int(font_size), align=align,
                justify=justify, margins=int(margins), line_spacing=int(line_spacing),
                letter_spacing=int(letter_spacing), position_x=int(position_x),
                position_y=int(position_y),
            )
            text_mask = transform_text_mask(base_mask, transform, float(strength), float(frequency), direction)
            outputs.append(compose_text_effect(frame, text_mask, font_color=font_color, **effects))
            masks.append(text_mask)
        return (
            pil_batch_to_image_tensor(outputs),
            _help("TUT_文字变形"),
            pil_batch_to_mask_tensor(masks),
        )


class TUT_文字沿路径:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "path_mask": ("MASK",),
            "text": ("STRING", {"multiline": False, "default": "沿路径排列文字"}),
            "font_name": _font_input(),
            "font_size": ("INT", {"default": 48, "min": 1, "max": 1024}),
            "start_offset": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 8192.0, "step": 1.0}),
            "path_offset": ("FLOAT", {"default": 0.0, "min": -2048.0, "max": 2048.0, "step": 1.0}),
            "letter_spacing": ("FLOAT", {"default": 0.0, "min": -256.0, "max": 512.0, "step": 1.0}),
            "reverse": ("BOOLEAN", {"default": False}),
            "orientation": (list(ORIENTATIONS),),
            "overflow": (list(OVERFLOW_MODES),),
            "font_color": (list(COLOR_NAMES),),
        }
        return {"required": required, "optional": _effect_inputs()}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "text_mask")
    FUNCTION = "text_on_path"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "沿路径 MASK 的最长轮廓排列文字，支持方向、偏移与溢出策略。"

    def text_on_path(
        self, image, path_mask, text, font_name, font_size, start_offset,
        path_offset, letter_spacing, reverse, orientation, overflow,
        font_color, **effects,
    ):
        frames = image_tensor_to_pil_batch(image)
        paths = mask_tensor_to_pil_batch(path_mask)
        frames, paths = broadcast_batches(frames, paths)
        outputs, masks = [], []
        for frame, path in zip(frames, paths):
            resized_path = _resize_mask(path, frame.size)
            text_mask = _path_text_mask(
                frame.size, resized_path, str(text), font_name, int(font_size),
                float(start_offset), float(path_offset), float(letter_spacing),
                bool(reverse), orientation, overflow,
            )
            outputs.append(compose_text_effect(frame, text_mask, font_color=font_color, **effects))
            masks.append(text_mask)
        return (
            pil_batch_to_image_tensor(outputs),
            _help("TUT_文字沿路径"),
            pil_batch_to_mask_tensor(masks),
        )


NODE_CLASS_MAPPINGS = {
    "TUT_WarpText": TUT_文字变形,
    "TUT_TextOnPath": TUT_文字沿路径,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_WarpText": "TUT_文字变形",
    "TUT_TextOnPath": "TUT_文字沿路径",
}
