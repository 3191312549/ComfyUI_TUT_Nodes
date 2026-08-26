"""Reusable text layout, Unicode segmentation, geometry, and path helpers."""

from __future__ import annotations

import math
import unicodedata
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from .fonts import load_font
from .shaping import render_shaped_line, shape_pillow_font


@lru_cache(maxsize=1)
def _regex_module():
    try:
        import regex
    except ImportError:
        return None
    return regex


@lru_cache(maxsize=1)
def _cv2_module():
    try:
        import cv2
    except ImportError:
        return None
    return cv2


@dataclass(frozen=True)
class GlyphPlacement:
    text: str
    source_start: int
    source_end: int
    x: float
    y: float
    width: float
    height: float
    line_index: int
    angle: float = 0.0


@dataclass(frozen=True)
class TextLayout:
    size: tuple[int, int]
    font_name: str
    font_size: int
    lines: tuple[str, ...]
    placements: tuple[GlyphPlacement, ...]
    bbox: tuple[int, int, int, int]
    line_height: int
    overflowed: bool = False
    line_origins: tuple[tuple[float, float], ...] = ()
    letter_spacing: int = 0


def _is_variation_selector(char: str) -> bool:
    code = ord(char)
    return 0xFE00 <= code <= 0xFE0F or 0xE0100 <= code <= 0xE01EF


def _is_emoji_modifier(char: str) -> bool:
    return 0x1F3FB <= ord(char) <= 0x1F3FF


def _graphemes(text: str) -> list[str]:
    """Split extended grapheme clusters, with a dependency-free fallback."""

    regex = _regex_module()
    if regex is not None:
        return regex.findall(r"\X", str(text))

    result: list[str] = []
    join_next = False
    for char in str(text):
        combining = bool(unicodedata.combining(char))
        if char == "\u200d":
            if result:
                result[-1] += char
            else:
                result.append(char)
            join_next = True
        elif result and (join_next or combining or _is_variation_selector(char) or _is_emoji_modifier(char)):
            result[-1] += char
            join_next = False
        else:
            result.append(char)
            join_next = False
    return result


def _is_cjk(unit: str) -> bool:
    if not unit:
        return False
    code = ord(unit[0])
    return (
        0x3400 <= code <= 0x9FFF
        or 0xF900 <= code <= 0xFAFF
        or 0x3040 <= code <= 0x30FF
        or 0xAC00 <= code <= 0xD7AF
    )


def segment_text(text: str, mode: str = "字符", include_whitespace: bool = False) -> list[str]:
    """Split text into character, word, or line units without third-party Unicode libs."""

    value = str(text)
    if mode == "行":
        units = value.splitlines() or ([""] if value == "" else [value])
    elif mode == "字符":
        units = _graphemes(value)
    elif mode == "词":
        units = []
        current = ""
        current_kind = None
        for unit in _graphemes(value):
            if unit.isspace():
                kind = "space"
            elif _is_cjk(unit):
                kind = "cjk"
            elif unit[0].isalnum() or unit[0] == "_":
                kind = "word"
            else:
                kind = "punct"
            if kind in ("cjk", "punct"):
                if current:
                    units.append(current)
                    current = ""
                units.append(unit)
                current_kind = None
            elif current and kind == current_kind:
                current += unit
            else:
                if current:
                    units.append(current)
                current = unit
                current_kind = kind
        if current:
            units.append(current)
    else:
        raise ValueError(f"未知文字拆分模式：{mode}")
    return units if include_whitespace else [unit for unit in units if unit and not unit.isspace()]


def _advance(draw: ImageDraw.ImageDraw, text: str, font, letter_spacing: int = 0) -> float:
    units = _graphemes(text)
    if not units:
        return 0.0
    shaped = shape_pillow_font(text, font)
    width = shaped.advance if shaped is not None else sum(float(draw.textlength(unit, font=font)) for unit in units)
    return width + max(0, len(units) - 1) * int(letter_spacing)


def _wrap_paragraph(draw, paragraph, font, max_width, letter_spacing) -> list[str]:
    if paragraph == "":
        return [""]
    if max_width is None or max_width <= 0:
        return [paragraph]

    chunks = segment_text(paragraph, "词", include_whitespace=True)
    lines: list[str] = []
    current = ""
    for chunk in chunks:
        candidate = current + chunk
        if current and _advance(draw, candidate, font, letter_spacing) > max_width:
            lines.append(current.rstrip())
            current = chunk.lstrip()
        else:
            current = candidate
        if current and _advance(draw, current, font, letter_spacing) > max_width:
            pieces = _graphemes(current)
            current = ""
            for piece in pieces:
                candidate = current + piece
                if current and _advance(draw, candidate, font, letter_spacing) > max_width:
                    lines.append(current)
                    current = piece
                else:
                    current = candidate
    lines.append(current.rstrip())
    return lines


def measure_text_layout(
    text: str,
    font_name: str,
    font_size: int,
    size: tuple[int, int],
    *,
    region: tuple[int, int, int, int] | None = None,
    max_width: int | None = None,
    max_lines: int = 0,
    line_spacing: int = 0,
    letter_spacing: int = 0,
    justify: str = "center",
    vertical_align: str = "center",
) -> TextLayout:
    """Measure and place wrapped text on a fixed canvas using real font metrics."""

    canvas_width, canvas_height = map(int, size)
    if canvas_width < 1 or canvas_height < 1:
        raise ValueError("文字画布尺寸必须大于 0")
    left, top, right, bottom = region or (0, 0, canvas_width, canvas_height)
    left, top = max(0, int(left)), max(0, int(top))
    right, bottom = min(canvas_width, int(right)), min(canvas_height, int(bottom))
    if right <= left or bottom <= top:
        raise ValueError("文字排版区域为空")

    font = load_font(font_name, int(font_size))
    probe = Image.new("L", (1, 1), 0)
    draw = ImageDraw.Draw(probe)
    sample = draw.textbbox((0, 0), "Ag", font=font, anchor="lt")
    line_height = max(1, int(math.ceil(sample[3] - sample[1])))
    available_width = int(max_width) if max_width is not None else right - left

    wrapped: list[str] = []
    for paragraph in str(text).split("\n"):
        wrapped.extend(_wrap_paragraph(draw, paragraph, font, available_width, int(letter_spacing)))
    if not wrapped:
        wrapped = [""]
    overflowed = bool(max_lines > 0 and len(wrapped) > int(max_lines))
    if max_lines > 0:
        wrapped = wrapped[: int(max_lines)]

    total_height = len(wrapped) * line_height + max(0, len(wrapped) - 1) * int(line_spacing)
    if vertical_align == "top":
        y0 = float(top)
    elif vertical_align == "bottom":
        y0 = float(bottom - total_height)
    else:
        y0 = float(top + (bottom - top - total_height) / 2)

    placements: list[GlyphPlacement] = []
    line_origins: list[tuple[float, float]] = []
    source_cursor = 0
    for line_index, line in enumerate(wrapped):
        line_width = _advance(draw, line, font, int(letter_spacing))
        if justify == "left":
            x = float(left)
        elif justify == "right":
            x = float(right - line_width)
        else:
            x = float(left + (right - left - line_width) / 2)
        y = y0 + line_index * (line_height + int(line_spacing))
        line_origins.append((x, y))
        for unit in _graphemes(line):
            unit_width = float(draw.textlength(unit, font=font))
            box = draw.textbbox((0, 0), unit or " ", font=font, anchor="lt")
            unit_height = max(1.0, float(box[3] - box[1]))
            placements.append(
                GlyphPlacement(unit, source_cursor, source_cursor + len(unit), x, y, unit_width, unit_height, line_index)
            )
            source_cursor += len(unit)
            x += unit_width + int(letter_spacing)
        source_cursor += 1

    ink = [placement for placement in placements if not placement.text.isspace()]
    if ink:
        bbox = (
            max(0, int(math.floor(min(item.x for item in ink)))),
            max(0, int(math.floor(min(item.y for item in ink)))),
            min(canvas_width, int(math.ceil(max(item.x + item.width for item in ink)))),
            min(canvas_height, int(math.ceil(max(item.y + item.height for item in ink)))),
        )
    else:
        bbox = (left, top, left, top)
    return TextLayout(
        (canvas_width, canvas_height), font_name, int(font_size), tuple(wrapped), tuple(placements), bbox,
        line_height, overflowed, tuple(line_origins), int(letter_spacing)
    )


def render_layout_mask(layout: TextLayout, placements: list[GlyphPlacement] | tuple[GlyphPlacement, ...] | None = None) -> Image.Image:
    mask = Image.new("L", layout.size, 0)
    if not layout.placements:
        return mask
    font = load_font(layout.font_name, layout.font_size)
    if placements is None and layout.line_origins and getattr(font, "path", None):
        try:
            ascent = float(font.getmetrics()[0])
        except (AttributeError, TypeError):
            ascent = float(layout.line_height)
        draw = ImageDraw.Draw(mask)
        for line, origin in zip(layout.lines, layout.line_origins):
            shaped_mask = render_shaped_line(
                line, font.path, layout.font_size, layout.size, origin, ascent, layout.letter_spacing
            )
            if shaped_mask is not None:
                mask = Image.fromarray(np.maximum(np.asarray(mask), np.asarray(shaped_mask)).astype(np.uint8), "L")
                draw = ImageDraw.Draw(mask)
            elif line:
                box = draw.textbbox((0, 0), line, font=font, anchor="lt")
                draw.text((origin[0] - box[0], origin[1] - box[1]), line, font=font, fill=255, anchor="lt")
        return mask
    draw = ImageDraw.Draw(mask)
    for placement in placements if placements is not None else layout.placements:
        if placement.text.isspace():
            continue
        box = draw.textbbox((0, 0), placement.text, font=font, anchor="lt")
        draw.text(
            (placement.x - box[0], placement.y - box[1]),
            placement.text,
            font=font,
            fill=255,
            anchor="lt",
        )
    return mask


def _placement_groups(layout: TextLayout, mode: str, include_whitespace: bool) -> list[list[GlyphPlacement]]:
    items = list(layout.placements)
    if mode == "字符":
        return [[item] for item in items if include_whitespace or not item.text.isspace()]
    if mode == "行":
        groups = []
        for index in range(len(layout.lines)):
            group = [item for item in items if item.line_index == index]
            if group and (include_whitespace or any(not item.text.isspace() for item in group)):
                groups.append(group)
        return groups
    if mode != "词":
        raise ValueError(f"未知文字拆分模式：{mode}")

    groups: list[list[GlyphPlacement]] = []
    current: list[GlyphPlacement] = []
    current_kind = None
    for item in items:
        if item.text.isspace():
            kind = "space"
        elif _is_cjk(item.text):
            kind = "cjk"
        elif item.text[0].isalnum() or item.text[0] == "_":
            kind = "word"
        else:
            kind = "punct"
        if kind in ("cjk", "punct"):
            if current:
                groups.append(current)
                current = []
            groups.append([item])
            current_kind = None
        elif current and current_kind == kind:
            current.append(item)
        else:
            if current:
                groups.append(current)
            current = [item]
            current_kind = kind
    if current:
        groups.append(current)
    return [group for group in groups if include_whitespace or any(not item.text.isspace() for item in group)]


def render_unit_masks(layout: TextLayout, mode: str = "字符", include_whitespace: bool = False) -> list[Image.Image]:
    return [render_layout_mask(layout, group) for group in _placement_groups(layout, mode, include_whitespace)]


def fit_text_layout(
    text: str,
    font_name: str,
    min_font_size: int,
    max_font_size: int,
    size: tuple[int, int],
    *,
    region: tuple[int, int, int, int],
    max_lines: int = 0,
    line_spacing: int = 0,
    letter_spacing: int = 0,
    justify: str = "center",
    vertical_align: str = "center",
    overflow: str = "缩小字号",
) -> TextLayout:
    minimum, maximum = int(min_font_size), int(max_font_size)
    if minimum < 1 or maximum < minimum:
        raise ValueError("字号范围无效：最大字号必须不小于最小字号")
    available_height = region[3] - region[1]
    best = None
    low, high = minimum, maximum
    while low <= high:
        middle = (low + high) // 2
        layout = measure_text_layout(
            text, font_name, middle, size, region=region, max_lines=max_lines,
            line_spacing=line_spacing, letter_spacing=letter_spacing,
            justify=justify, vertical_align=vertical_align,
        )
        used_height = len(layout.lines) * layout.line_height + max(0, len(layout.lines) - 1) * int(line_spacing)
        fits = not layout.overflowed and used_height <= available_height
        if fits:
            best = layout
            low = middle + 1
        else:
            high = middle - 1
    if best is not None:
        return best
    if overflow == "截断":
        return measure_text_layout(
            text, font_name, minimum, size, region=region, max_lines=max_lines,
            line_spacing=line_spacing, letter_spacing=letter_spacing,
            justify=justify, vertical_align=vertical_align,
        )
    raise ValueError(f"文字无法放入区域：最小字号 {minimum} 仍超出可用范围")


def _bilinear_sample(array: np.ndarray, source_x: np.ndarray, source_y: np.ndarray) -> np.ndarray:
    height, width = array.shape
    x0 = np.floor(source_x).astype(np.int32)
    y0 = np.floor(source_y).astype(np.int32)
    x1, y1 = x0 + 1, y0 + 1
    valid = (source_x >= 0) & (source_x <= width - 1) & (source_y >= 0) & (source_y <= height - 1)
    x0c, x1c = np.clip(x0, 0, width - 1), np.clip(x1, 0, width - 1)
    y0c, y1c = np.clip(y0, 0, height - 1), np.clip(y1, 0, height - 1)
    wx, wy = source_x - x0, source_y - y0
    sampled = (
        array[y0c, x0c] * (1 - wx) * (1 - wy)
        + array[y0c, x1c] * wx * (1 - wy)
        + array[y1c, x0c] * (1 - wx) * wy
        + array[y1c, x1c] * wx * wy
    )
    return np.where(valid, sampled, 0.0)


def transform_text_mask(mask: Image.Image, transform: str, strength: float, frequency: float = 1.0, direction: str = "水平") -> Image.Image:
    """Transform only the tight text area and return it on the original canvas."""

    source_mask = mask.convert("L")
    bbox = source_mask.getbbox()
    if bbox is None or abs(float(strength)) < 1e-6:
        return source_mask.copy()
    crop = source_mask.crop(bbox)
    pad = max(4, int(max(crop.size) * min(1.0, abs(float(strength))) * 0.35))
    padded = Image.new("L", (crop.width + pad * 2, crop.height + pad * 2), 0)
    padded.paste(crop, (pad, pad))
    array = np.asarray(padded, dtype=np.float32)
    height, width = array.shape
    yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
    normalized_x = (xx - (width - 1) / 2) / max(1.0, (width - 1) / 2)
    normalized_y = (yy - (height - 1) / 2) / max(1.0, (height - 1) / 2)
    amount = float(strength)
    source_x, source_y = xx.copy(), yy.copy()

    if transform in ("上拱", "下拱"):
        sign = -1.0 if transform == "上拱" else 1.0
        shift = sign * amount * height * 0.35 * (1.0 - normalized_x ** 2)
        source_y = yy - shift
    elif transform == "波浪":
        phase = 2.0 * math.pi * max(0.1, float(frequency))
        if direction == "垂直":
            source_x = xx - amount * width * 0.18 * np.sin(phase * normalized_y)
        else:
            source_y = yy - amount * height * 0.28 * np.sin(phase * (normalized_x + 1.0) / 2.0)
    elif transform == "斜切":
        if direction == "垂直":
            source_y = yy - amount * 0.65 * (xx - width / 2)
        else:
            source_x = xx - amount * 0.65 * (yy - height / 2)
    elif transform == "梯形":
        scale = np.clip(1.0 + amount * 0.55 * normalized_y, 0.2, 3.0)
        source_x = (xx - width / 2) / scale + width / 2
    elif transform == "透视":
        scale = np.clip(1.0 + amount * 0.5 * normalized_y, 0.25, 3.0)
        source_x = (xx - width / 2) / scale + width / 2 - amount * width * 0.12 * normalized_y
    else:
        raise ValueError(f"未知文字变形模式：{transform}")

    cv2 = _cv2_module()
    if cv2 is not None:
        sampled = cv2.remap(
            array,
            source_x.astype(np.float32),
            source_y.astype(np.float32),
            interpolation=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    else:
        sampled = _bilinear_sample(array, source_x, source_y)
    transformed = Image.fromarray(np.clip(sampled, 0, 255).astype(np.uint8), "L")
    output = Image.new("L", source_mask.size, 0)
    center_x = (bbox[0] + bbox[2]) // 2
    center_y = (bbox[1] + bbox[3]) // 2
    output.paste(transformed, (center_x - transformed.width // 2, center_y - transformed.height // 2))
    return output


def sample_mask_path(mask: Image.Image) -> list[tuple[float, float]]:
    """Extract and order the largest 8-connected boundary from a path mask."""

    binary = np.asarray(mask.convert("L"), dtype=np.uint8) > 0
    if not binary.any():
        raise ValueError("路径 MASK 为空，无法排列文字")
    cv2 = _cv2_module()
    if cv2 is not None:
        contours, _ = cv2.findContours(
            (binary.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE
        )
        usable = [contour.reshape(-1, 2) for contour in contours if len(contour) >= 2]
        if usable:
            contour = max(usable, key=len)
            start = min(range(len(contour)), key=lambda index: (int(contour[index, 0]), int(contour[index, 1])))
            contour = np.concatenate((contour[start:], contour[:start]), axis=0)
            return [(float(x), float(y)) for x, y in contour]
    eroded = np.asarray(mask.convert("L").filter(ImageFilter.MinFilter(3)), dtype=np.uint8) > 0
    boundary = binary & ~eroded
    points = set(map(tuple, np.argwhere(boundary).tolist()))
    if len(points) < 2:
        raise ValueError("路径 MASK 太短，至少需要两个轮廓点")

    components: list[list[tuple[int, int]]] = []
    remaining = set(points)
    neighbors = tuple((dy, dx) for dy in (-1, 0, 1) for dx in (-1, 0, 1) if dx or dy)
    while remaining:
        start = remaining.pop()
        stack = [start]
        component = [start]
        while stack:
            y, x = stack.pop()
            for dy, dx in neighbors:
                candidate = (y + dy, x + dx)
                if candidate in remaining:
                    remaining.remove(candidate)
                    stack.append(candidate)
                    component.append(candidate)
        components.append(component)
    component = max(components, key=len)
    if len(component) < 2:
        raise ValueError("未找到可用的连续路径轮廓")

    available = set(component)
    endpoint_candidates = []
    for point in component:
        y, x = point
        degree = sum((y + dy, x + dx) in available for dy, dx in neighbors)
        if degree <= 2:
            endpoint_candidates.append(point)
    current = min(endpoint_candidates or component, key=lambda item: (item[1], item[0]))
    ordered = [current]
    available.remove(current)
    previous = None
    while available:
        y, x = current
        local = [(y + dy, x + dx) for dy, dx in neighbors if (y + dy, x + dx) in available]
        if local:
            if previous is None:
                next_point = min(local)
            else:
                vy, vx = y - previous[0], x - previous[1]
                next_point = max(local, key=lambda item: vy * (item[0] - y) + vx * (item[1] - x))
        else:
            next_point = min(available, key=lambda item: (item[0] - y) ** 2 + (item[1] - x) ** 2)
            if (next_point[0] - y) ** 2 + (next_point[1] - x) ** 2 > 8:
                break
        previous, current = current, next_point
        ordered.append(current)
        available.remove(current)
    if len(ordered) < 2:
        raise ValueError("最长轮廓不足以排列文字")
    return [(float(x), float(y)) for y, x in ordered]


def path_lengths(path: list[tuple[float, float]]) -> np.ndarray:
    points = np.asarray(path, dtype=np.float32)
    distances = np.hypot(np.diff(points[:, 0]), np.diff(points[:, 1]))
    return np.concatenate(([0.0], np.cumsum(distances)))


def point_on_path(path: list[tuple[float, float]], lengths: np.ndarray, distance: float) -> tuple[float, float, float]:
    distance = float(np.clip(distance, 0.0, float(lengths[-1])))
    index = min(len(path) - 2, max(0, int(np.searchsorted(lengths, distance, side="right") - 1)))
    span = max(1e-6, float(lengths[index + 1] - lengths[index]))
    ratio = (distance - float(lengths[index])) / span
    x0, y0 = path[index]
    x1, y1 = path[index + 1]
    x, y = x0 + (x1 - x0) * ratio, y0 + (y1 - y0) * ratio
    return x, y, math.degrees(math.atan2(y1 - y0, x1 - x0))
