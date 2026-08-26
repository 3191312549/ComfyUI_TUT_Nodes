"""Adaptive inverse-color text watermark helpers."""

from __future__ import annotations

from PIL import Image, ImageDraw, ImageOps

from .fonts import load_font
from .imaging import render_text_mask


CORNER_POSITIONS = ("左上角", "右上角", "左下角", "右下角")


def _text_extent(text: str, font_name: str, font_size: int) -> tuple[int, int]:
    canvas = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(canvas)
    font = load_font(font_name, font_size)
    lines = str(text).split("\n")
    widths: list[int] = []
    heights: list[int] = []
    for line in lines:
        box = draw.textbbox((0, 0), line or "Ag", font=font, anchor="lt")
        widths.append(max(0, int(box[2] - box[0])))
        heights.append(max(1, int(box[3] - box[1])))
    return max(widths, default=0), sum(heights)


def adaptive_font_size(
    image_size: tuple[int, int],
    text: str,
    font_name: str,
    size_percent: float,
    max_width_percent: float,
) -> int:
    """Scale from the short edge, then shrink long text to the allowed width."""

    width, height = image_size
    upper = max(1, int(round(min(width, height) * float(size_percent) / 100.0)))
    allowed_width = max(1, int(round(width * float(max_width_percent) / 100.0)))
    if not str(text):
        return upper

    low, high, best = 1, upper, 1
    while low <= high:
        middle = (low + high) // 2
        text_width, text_height = _text_extent(text, font_name, middle)
        if text_width <= allowed_width and text_height <= height:
            best = middle
            low = middle + 1
        else:
            high = middle - 1
    return best


def render_adaptive_watermark(
    frame: Image.Image,
    text: str,
    font_name: str,
    position: str,
    size_percent: float,
    max_width_percent: float,
    x_margin_percent: float,
    y_margin_percent: float,
    opacity: float,
) -> tuple[Image.Image, Image.Image]:
    """Render a watermark by revealing a pixel-wise inverted copy through its mask."""

    frame = frame.convert("RGB")
    if position not in CORNER_POSITIONS:
        raise ValueError(f"不支持的水印位置：{position}")
    if not str(text) or float(opacity) <= 0.0:
        return frame.copy(), Image.new("L", frame.size, 0)

    width, height = frame.size
    font_size = adaptive_font_size(frame.size, text, font_name, size_percent, max_width_percent)
    short_edge = min(width, height)
    x_margin = max(0, int(round(short_edge * float(x_margin_percent) / 100.0)))
    y_margin = max(0, int(round(short_edge * float(y_margin_percent) / 100.0)))
    is_left = position in {"左上角", "左下角"}
    is_top = position in {"左上角", "右上角"}
    mask = render_text_mask(
        frame.size,
        text,
        font_name,
        font_size,
        align="top" if is_top else "bottom",
        justify="left" if is_left else "right",
        position_x=x_margin if is_left else -x_margin,
        position_y=y_margin if is_top else -y_margin,
    )

    if mask.getbbox() is None:
        return frame.copy(), mask
    opacity = max(0.0, min(1.0, float(opacity)))
    alpha = mask if opacity >= 1.0 else mask.point(lambda value: round(value * opacity))
    output = Image.composite(ImageOps.invert(frame), frame, alpha)
    return output, mask
