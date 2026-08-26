"""Optional HarfBuzz shaping and outline rasterization."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from PIL import Image


@dataclass(frozen=True)
class ShapedGlyph:
    glyph_id: int
    cluster: int
    x_advance: float
    y_advance: float
    x_offset: float
    y_offset: float


@dataclass(frozen=True)
class ShapedText:
    glyphs: tuple[ShapedGlyph, ...]
    direction: str
    advance: float


@lru_cache(maxsize=1)
def _harfbuzz():
    try:
        import uharfbuzz
    except ImportError:
        return None
    return uharfbuzz


def harfbuzz_available() -> bool:
    return _harfbuzz() is not None


@lru_cache(maxsize=64)
def _font_bytes(path: str) -> bytes:
    return Path(path).read_bytes()


def shape_text(text: str, font_path: str | Path, font_size: int) -> ShapedText | None:
    """Shape one line; return ``None`` when HarfBuzz or a real font is unavailable."""

    hb = _harfbuzz()
    if hb is None or not text or not font_path:
        return None
    try:
        face = hb.Face(_font_bytes(str(font_path)))
        font = hb.Font(face)
        scale = max(1, int(font_size)) * 64
        font.scale = (scale, scale)
        buffer = hb.Buffer()
        buffer.add_str(str(text))
        buffer.guess_segment_properties()
        hb.shape(font, buffer)
    except (OSError, RuntimeError, ValueError):
        return None
    glyphs = tuple(
        ShapedGlyph(
            int(info.codepoint), int(info.cluster),
            float(position.x_advance) / 64.0,
            float(position.y_advance) / 64.0,
            float(position.x_offset) / 64.0,
            float(position.y_offset) / 64.0,
        )
        for info, position in zip(buffer.glyph_infos, buffer.glyph_positions)
    )
    return ShapedText(glyphs, str(buffer.direction), sum(item.x_advance for item in glyphs))


def shape_pillow_font(text: str, font) -> ShapedText | None:
    path = getattr(font, "path", None)
    size = getattr(font, "size", None)
    if not path or not size:
        return None
    return shape_text(text, str(path), int(size))


def render_shaped_line(
    text: str,
    font_path: str | Path,
    font_size: int,
    size: tuple[int, int],
    origin: tuple[float, float],
    ascent: float,
    letter_spacing: int = 0,
) -> Image.Image | None:
    """Rasterize a shaped line into a fixed-size L mask."""

    hb = _harfbuzz()
    shaped = shape_text(text, font_path, font_size)
    if hb is None or shaped is None or not shaped.glyphs:
        return None
    width, height = map(int, size)
    try:
        face = hb.Face(_font_bytes(str(font_path)))
        font = hb.Font(face)
        scale = max(1, int(font_size)) * 64
        font.scale = (scale, scale)
        raster = hb.RasterDraw()
        raster.extents = hb.RasterExtents(0, 0, width, height, width)
        cursor_x, cursor_y = float(origin[0]), float(origin[1]) + float(ascent)
        drew_outline = False
        for index, glyph in enumerate(shaped.glyphs):
            raster.transform = (
                1.0 / 64.0, 0.0, 0.0, -1.0 / 64.0,
                cursor_x + glyph.x_offset,
                cursor_y - glyph.y_offset,
            )
            drew_outline = raster.draw_glyph_or_fail(font, glyph.glyph_id) or drew_outline
            cursor_x += glyph.x_advance
            cursor_y -= glyph.y_advance
            if index + 1 < len(shaped.glyphs) and shaped.glyphs[index + 1].cluster != glyph.cluster:
                cursor_x += int(letter_spacing)
        if not drew_outline:
            return None
        rendered = raster.render()
        if rendered is None or len(rendered.buffer) != width * height:
            return None
        return Image.frombytes("L", (width, height), rendered.buffer)
    except (OSError, RuntimeError, TypeError, ValueError):
        return None
