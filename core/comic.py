"""Comic panel layout and speech-bubble rendering helpers."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .fonts import DEFAULT_FONT_TOKEN, font_for_text, load_font
from .imaging import (
    image_tensor_to_pil_batch,
    pil_batch_to_image_tensor,
    pil_batch_to_mask_tensor,
)
from .text_layout import measure_text_layout, render_layout_mask


PANEL_SCHEMA_VERSION = 1
MAX_CUSTOM_PANELS = 20
AUTO_PANELS_PER_PAGE = 6
BUBBLE_SCHEMA_VERSION = 1
BUBBLE_TEXT_DIRECTIONS = ("horizontal", "vertical_ltr", "vertical_rtl")

AUTO_LAYOUT = "自动匹配数量"
CUSTOM_LAYOUT = "自由画框"


@lru_cache(maxsize=1)
def _cv2():
    try:
        import cv2
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "漫画自由四边形抗锯齿需要 OpenCV，请安装 opencv-python-headless>=4.10,<6"
        ) from exc
    return cv2
PANEL_LAYOUTS: dict[str, tuple[tuple[float, float, float, float], ...]] = {
    "整页单格": ((0.0, 0.0, 1.0, 1.0),),
    "左右双格": ((0.0, 0.0, 0.5, 1.0), (0.5, 0.0, 1.0, 1.0)),
    "上下双格": ((0.0, 0.0, 1.0, 0.5), (0.0, 0.5, 1.0, 1.0)),
    "上大下二": ((0.0, 0.0, 1.0, 0.62), (0.0, 0.62, 0.5, 1.0), (0.5, 0.62, 1.0, 1.0)),
    "左大右二": ((0.0, 0.0, 0.62, 1.0), (0.62, 0.0, 1.0, 0.5), (0.62, 0.5, 1.0, 1.0)),
    "四宫格": (
        (0.0, 0.0, 0.5, 0.5), (0.5, 0.0, 1.0, 0.5),
        (0.0, 0.5, 0.5, 1.0), (0.5, 0.5, 1.0, 1.0),
    ),
    "主格加三小格": (
        (0.0, 0.0, 0.64, 1.0), (0.64, 0.0, 1.0, 1 / 3),
        (0.64, 1 / 3, 1.0, 2 / 3), (0.64, 2 / 3, 1.0, 1.0),
    ),
    "五格错落": (
        (0.0, 0.0, 0.5, 0.54), (0.5, 0.0, 1.0, 0.54),
        (0.0, 0.54, 1 / 3, 1.0), (1 / 3, 0.54, 2 / 3, 1.0), (2 / 3, 0.54, 1.0, 1.0),
    ),
    "六宫格": tuple(
        (column / 3, row / 2, (column + 1) / 3, (row + 1) / 2)
        for row in range(2) for column in range(3)
    ),
    CUSTOM_LAYOUT: ((0.08, 0.08, 0.92, 0.92),),
}

BUBBLE_SHAPES = (
    "椭圆对白框", "圆角矩形", "云朵思考框", "爆炸喊话框",
    "爆炸对话框", "闪光对话框", "方形旁白框", "无边框文字",
)
BUBBLE_SPIKE_COUNT_DEFAULT = 16
BUBBLE_SPIKE_DEPTH_DEFAULT = 0.22
BUBBLE_CLOUD_LOBES_DEFAULT = 10
BUBBLE_CLOUD_DEPTH_DEFAULT = 0.14


def _parse_hex(value: str) -> tuple[int, int, int]:
    text = str(value).strip().lstrip("#")
    if len(text) == 3:
        text = "".join(character * 2 for character in text)
    if len(text) != 6:
        raise ValueError(f"颜色必须是 #RGB 或 #RRGGBB：{value!r}")
    try:
        return tuple(int(text[index:index + 2], 16) for index in (0, 2, 4))
    except ValueError as exc:
        raise ValueError(f"颜色包含无效字符：{value!r}") from exc


def _finite_number(value: Any, name: str, minimum: float, maximum: float) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{name} 必须是数值")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} 必须是数值") from exc
    if not math.isfinite(number):
        raise ValueError(f"{name} 必须是有限数值")
    if not minimum <= number <= maximum:
        raise ValueError(f"{name} 必须在 {minimum} 到 {maximum} 之间")
    return number


def _decode_object(value: str, label: str) -> dict[str, Any]:
    text = str(value or "").strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{label}不是有效 JSON：第 {exc.lineno} 行第 {exc.colno} 列") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"{label}必须是 JSON 对象")
    return parsed


def _segments_intersect(a, b, c, d, epsilon: float = 1e-9) -> bool:
    def cross(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    def on_segment(p, q, r):
        return (
            min(p[0], r[0]) - epsilon <= q[0] <= max(p[0], r[0]) + epsilon
            and min(p[1], r[1]) - epsilon <= q[1] <= max(p[1], r[1]) + epsilon
        )

    values = (cross(a, b, c), cross(a, b, d), cross(c, d, a), cross(c, d, b))
    if values[0] * values[1] < -epsilon and values[2] * values[3] < -epsilon:
        return True
    return any(
        abs(value) <= epsilon and on_segment(*points)
        for value, points in zip(values, ((a, c, b), (a, d, b), (c, a, d), (c, b, d)))
    )


def _quad_area(points) -> float:
    return abs(sum(
        points[index][0] * points[(index + 1) % 4][1]
        - points[(index + 1) % 4][0] * points[index][1]
        for index in range(4)
    )) / 2.0


def _parse_quad(raw_quad, label: str) -> tuple[tuple[float, float], ...]:
    if not isinstance(raw_quad, list) or len(raw_quad) != 4:
        raise ValueError(f"{label}必须包含 4 个顶点")
    points = []
    for point_index, raw_point in enumerate(raw_quad):
        if not isinstance(raw_point, list) or len(raw_point) != 2:
            raise ValueError(f"{label}第 {point_index + 1} 个顶点必须是 [X,Y]")
        points.append((
            _finite_number(raw_point[0], f"{label}第 {point_index + 1} 个顶点 X", 0.0, 1.0),
            _finite_number(raw_point[1], f"{label}第 {point_index + 1} 个顶点 Y", 0.0, 1.0),
        ))
    for index in range(4):
        next_index = (index + 1) % 4
        if math.dist(points[index], points[next_index]) < 0.005:
            raise ValueError(f"{label}第 {index + 1} 条边过短")
    if _segments_intersect(points[0], points[1], points[2], points[3]) or _segments_intersect(
        points[1], points[2], points[3], points[0]
    ):
        raise ValueError(f"{label}边线不能自交")
    if _quad_area(points) < 0.0005:
        raise ValueError(f"{label}面积过小")
    return tuple(points)


def _parse_panel_config(value: str) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[tuple[float, float, float, float], ...]],
    dict[str, tuple[tuple[tuple[float, float], ...], ...]],
    dict[str, tuple[int, ...]],
    str,
]:
    parsed = _decode_object(value, "分镜配置")
    if parsed and parsed.get("version") != PANEL_SCHEMA_VERSION:
        raise ValueError(f"分镜配置 version 必须为 {PANEL_SCHEMA_VERSION}")
    raw_panels = parsed.get("panels", [])
    if not isinstance(raw_panels, list):
        raise ValueError("分镜配置 panels 必须是数组")
    if len(raw_panels) > MAX_CUSTOM_PANELS:
        raise ValueError(f"每页最多只能配置 {MAX_CUSTOM_PANELS} 个画格")
    panels: list[dict[str, Any]] = []
    for index, raw in enumerate(raw_panels):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {index + 1} 个画格配置必须是对象")
        flip = raw.get("flip", False)
        if not isinstance(flip, bool):
            raise ValueError(f"第 {index + 1} 个画格的水平翻转必须是布尔值")
        overflow = {}
        for key, label in (
            ("overflow_top", "上边缘开放"), ("overflow_bottom", "下边缘开放"),
            ("overflow_left", "左边缘开放"), ("overflow_right", "右边缘开放"),
        ):
            overflow[key] = raw.get(key, False)
            if not isinstance(overflow[key], bool):
                raise ValueError(f"第 {index + 1} 个画格的{label}必须是布尔值")
        raw_open_edges = raw.get("open_edges")
        if raw_open_edges is None:
            open_edges = [
                overflow["overflow_top"], overflow["overflow_right"],
                overflow["overflow_bottom"], overflow["overflow_left"],
            ]
        else:
            if not isinstance(raw_open_edges, list) or len(raw_open_edges) != 4:
                raise ValueError(f"第 {index + 1} 个画格的 open_edges 必须包含 4 个布尔值")
            if any(not isinstance(item, bool) for item in raw_open_edges):
                raise ValueError(f"第 {index + 1} 个画格的 open_edges 必须使用布尔值")
            open_edges = list(raw_open_edges)
            overflow = {
                "overflow_top": open_edges[0], "overflow_right": open_edges[1],
                "overflow_bottom": open_edges[2], "overflow_left": open_edges[3],
            }
        panels.append({
            "focus_x": _finite_number(raw.get("focus_x", 0.5), f"第 {index + 1} 格焦点 X", 0.0, 1.0),
            "focus_y": _finite_number(raw.get("focus_y", 0.5), f"第 {index + 1} 格焦点 Y", 0.0, 1.0),
            "zoom": _finite_number(raw.get("zoom", 1.0), f"第 {index + 1} 格缩放", 0.25, 4.0),
            "flip": flip,
            "open_edges": open_edges,
            **overflow,
        })
    while len(panels) < MAX_CUSTOM_PANELS:
        panels.append({
            "focus_x": 0.5, "focus_y": 0.5, "zoom": 1.0, "flip": False,
            "open_edges": [False, False, False, False],
            "overflow_top": False, "overflow_bottom": False,
            "overflow_left": False, "overflow_right": False,
        })

    raw_overrides = parsed.get("layout_overrides", {})
    if not isinstance(raw_overrides, dict):
        raise ValueError("分镜配置 layout_overrides 必须是对象")
    layout_overrides: dict[str, tuple[tuple[float, float, float, float], ...]] = {}
    for layout_name, raw_rectangles in raw_overrides.items():
        if layout_name not in PANEL_LAYOUTS:
            raise ValueError(f"分镜配置包含未知模板：{layout_name}")
        if layout_name == CUSTOM_LAYOUT:
            if not isinstance(raw_rectangles, list) or not 1 <= len(raw_rectangles) <= MAX_CUSTOM_PANELS:
                raise ValueError(f"自由画框的数量必须为 1 到 {MAX_CUSTOM_PANELS}")
        elif not isinstance(raw_rectangles, list) or len(raw_rectangles) != len(PANEL_LAYOUTS[layout_name]):
            raise ValueError(f"{layout_name} 的自定义画框数量必须为 {len(PANEL_LAYOUTS[layout_name])}")
        rectangles = []
        for index, raw_rect in enumerate(raw_rectangles):
            if not isinstance(raw_rect, list) or len(raw_rect) != 4:
                raise ValueError(f"{layout_name} 第 {index + 1} 个画框必须是四项坐标数组")
            x0, y0, x1, y1 = (
                _finite_number(raw_rect[0], f"{layout_name} 第 {index + 1} 格左边界", 0.0, 1.0),
                _finite_number(raw_rect[1], f"{layout_name} 第 {index + 1} 格上边界", 0.0, 1.0),
                _finite_number(raw_rect[2], f"{layout_name} 第 {index + 1} 格右边界", 0.0, 1.0),
                _finite_number(raw_rect[3], f"{layout_name} 第 {index + 1} 格下边界", 0.0, 1.0),
            )
            if x1 <= x0 or y1 <= y0:
                raise ValueError(f"{layout_name} 第 {index + 1} 个画框的右/下边界必须大于左/上边界")
            rectangles.append((x0, y0, x1, y1))
        layout_overrides[layout_name] = tuple(rectangles)

    raw_quads = parsed.get("quad_overrides", {})
    if not isinstance(raw_quads, dict):
        raise ValueError("分镜配置 quad_overrides 必须是对象")
    quad_overrides: dict[str, tuple[tuple[tuple[float, float], ...], ...]] = {}
    for layout_name, raw_layout_quads in raw_quads.items():
        if layout_name not in PANEL_LAYOUTS:
            raise ValueError(f"四边形配置包含未知模板：{layout_name}")
        if layout_name == CUSTOM_LAYOUT:
            if not isinstance(raw_layout_quads, list) or not 1 <= len(raw_layout_quads) <= MAX_CUSTOM_PANELS:
                raise ValueError(f"自由画框的四边形数量必须为 1 到 {MAX_CUSTOM_PANELS}")
        elif not isinstance(raw_layout_quads, list) or len(raw_layout_quads) != len(PANEL_LAYOUTS[layout_name]):
            raise ValueError(f"{layout_name} 的四边形数量必须为 {len(PANEL_LAYOUTS[layout_name])}")
        quad_overrides[layout_name] = tuple(
            _parse_quad(raw_quad, f"{layout_name} 第 {index + 1} 个四边形")
            for index, raw_quad in enumerate(raw_layout_quads)
        )

    raw_layer_orders = parsed.get("layer_orders", {})
    if not isinstance(raw_layer_orders, dict):
        raise ValueError("分镜配置 layer_orders 必须是对象")
    layer_orders: dict[str, tuple[int, ...]] = {}
    for layout_name, raw_order in raw_layer_orders.items():
        if layout_name not in PANEL_LAYOUTS:
            raise ValueError(f"分镜图层包含未知模板：{layout_name}")
        count = len(quad_overrides.get(layout_name, layout_overrides.get(layout_name, PANEL_LAYOUTS[layout_name])))
        if not isinstance(raw_order, list) or len(raw_order) != count:
            raise ValueError(f"{layout_name} 的图层顺序必须包含 {count} 个画格")
        if any(isinstance(value, bool) or not isinstance(value, int) for value in raw_order):
            raise ValueError(f"{layout_name} 的图层顺序必须使用整数索引")
        if sorted(raw_order) != list(range(count)):
            raise ValueError(f"{layout_name} 的图层顺序必须是 0 到 {count - 1} 的完整排列")
        layer_orders[layout_name] = tuple(raw_order)

    canonical_data: dict[str, Any] = {"version": PANEL_SCHEMA_VERSION, "panels": panels}
    if layout_overrides:
        canonical_data["layout_overrides"] = {
            name: [list(rect) for rect in rectangles]
            for name, rectangles in layout_overrides.items()
        }
    if quad_overrides:
        canonical_data["quad_overrides"] = {
            name: [[list(point) for point in quad] for quad in quads]
            for name, quads in quad_overrides.items()
        }
    if layer_orders:
        canonical_data["layer_orders"] = {
            name: list(order) for name, order in layer_orders.items()
        }
    canonical = json.dumps(
        canonical_data, ensure_ascii=False, separators=(",", ":")
    )
    return panels, layout_overrides, quad_overrides, layer_orders, canonical


def parse_panel_data(value: str) -> tuple[list[dict[str, Any]], str]:
    panels, _layout_overrides, _quad_overrides, _layer_orders, canonical = _parse_panel_config(value)
    return panels, canonical


def _auto_layout(count: int) -> str:
    return {
        1: "整页单格", 2: "左右双格", 3: "上大下二",
        4: "四宫格", 5: "五格错落", 6: "六宫格",
    }[max(1, min(6, int(count)))]


def _pixel_rect(rect, width: int, height: int, margin: int, gutter: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    content_w, content_h = width - 2 * margin, height - 2 * margin
    left = margin + round(x0 * content_w) + (round(gutter / 2) if x0 > 0 else 0)
    top = margin + round(y0 * content_h) + (round(gutter / 2) if y0 > 0 else 0)
    right = margin + round(x1 * content_w) - (gutter // 2 if x1 < 1 else 0)
    bottom = margin + round(y1 * content_h) - (gutter // 2 if y1 < 1 else 0)
    if right <= left or bottom <= top:
        raise ValueError("画布尺寸不足以容纳当前页边距和画格间距")
    return left, top, right, bottom


def _absolute_pixel_rect(rect, width: int, height: int, margin: int) -> tuple[int, int, int, int]:
    x0, y0, x1, y1 = rect
    left, top = max(margin, round(x0 * width)), max(margin, round(y0 * height))
    right, bottom = min(width - margin, round(x1 * width)), min(height - margin, round(y1 * height))
    if right <= left or bottom <= top:
        raise ValueError("自定义画框超出强制页边距或尺寸过小")
    return left, top, right, bottom


def _position_image(source: Image.Image, size: tuple[int, int], mode: str, settings: dict[str, Any]) -> tuple[Image.Image, int, int]:
    target_w, target_h = size
    if settings["flip"]:
        source = source.transpose(Image.Transpose.FLIP_LEFT_RIGHT)
    if mode == "完整显示":
        scale = min(target_w / source.width, target_h / source.height) * settings["zoom"]
    else:
        scale = max(target_w / source.width, target_h / source.height) * settings["zoom"]
    resized = source.resize(
        (max(1, round(source.width * scale)), max(1, round(source.height * scale))), Image.Resampling.LANCZOS
    )
    paste_x = round((target_w - resized.width) * settings["focus_x"])
    paste_y = round((target_h - resized.height) * settings["focus_y"])
    return resized, paste_x, paste_y


def _fit_image(source: Image.Image, size: tuple[int, int], mode: str, settings: dict[str, Any], background) -> Image.Image:
    resized, paste_x, paste_y = _position_image(source, size, mode, settings)
    result = Image.new("RGB", size, background)
    alpha = resized.getchannel("A") if resized.mode == "RGBA" else None
    result.paste(resized, (paste_x, paste_y), alpha)
    return result


def _placed_source_alpha(source: Image.Image, canvas_size, position) -> Image.Image:
    alpha = source.getchannel("A") if source.mode == "RGBA" else Image.new("L", source.size, 255)
    placed = Image.new("L", canvas_size, 0)
    placed.paste(alpha, position)
    return placed


def _quad_pixel_points(quad, width: int, height: int, margin: int, label: str):
    min_x, max_x = margin / width, (width - margin) / width
    min_y, max_y = margin / height, (height - margin) / height
    for index, (x, y) in enumerate(quad):
        if not min_x <= x <= max_x or not min_y <= y <= max_y:
            raise ValueError(f"{label}第 {index + 1} 个顶点超出强制页边距")
    return tuple(
        (
            max(0.0, min(width - 1.0, x * width)),
            max(0.0, min(height - 1.0, y * height)),
        )
        for x, y in quad
    )


def _polygon_signed_area(points) -> float:
    return sum(
        points[index][0] * points[(index + 1) % len(points)][1]
        - points[(index + 1) % len(points)][0] * points[index][1]
        for index in range(len(points))
    ) / 2.0


def _aa_polygon_mask(size: tuple[int, int], points) -> Image.Image:
    """Rasterize normalized geometry with sub-pixel coordinates and antialiased edges."""
    cv2 = _cv2()
    shift = 8
    fixed_points = np.rint(np.asarray(points, dtype=np.float64) * (1 << shift)).astype(np.int32)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.fillPoly(mask, [fixed_points], 255, lineType=cv2.LINE_AA, shift=shift)
    return Image.fromarray(mask, "L")


def _aa_line_mask(size: tuple[int, int], start, end, width: int) -> Image.Image:
    cv2 = _cv2()
    shift = 8
    fixed_points = np.rint(np.asarray((start, end), dtype=np.float64) * (1 << shift)).astype(np.int32)
    mask = np.zeros((size[1], size[0]), dtype=np.uint8)
    cv2.polylines(
        mask, [fixed_points], False, 255,
        thickness=max(1, int(width)), lineType=cv2.LINE_AA, shift=shift,
    )
    return Image.fromarray(mask, "L")


def _edge_extrusion_mask(size: tuple[int, int], points, edge_index: int) -> Image.Image:
    width, height = size
    start, end = points[edge_index], points[(edge_index + 1) % 4]
    dx, dy = end[0] - start[0], end[1] - start[1]
    length = max(1e-6, math.hypot(dx, dy))
    if _polygon_signed_area(points) >= 0:
        normal = (dy / length, -dx / length)
    else:
        normal = (-dy / length, dx / length)
    distance = math.hypot(width, height) * 3.0
    far_start = (start[0] + normal[0] * distance, start[1] + normal[1] * distance)
    far_end = (end[0] + normal[0] * distance, end[1] + normal[1] * distance)
    return _aa_polygon_mask(size, (start, end, far_end, far_start))


def _quad_geometry(quad, size: tuple[int, int], margin: int, stroke: int, label: str):
    width, height = size
    points = _quad_pixel_points(quad, width, height, margin, label)
    left, top = math.floor(min(point[0] for point in points)), math.floor(min(point[1] for point in points))
    right = math.ceil(max(point[0] for point in points)) + 1
    bottom = math.ceil(max(point[1] for point in points)) + 1
    if right - left <= stroke * 2 or bottom - top <= stroke * 2:
        raise ValueError(f"{label}边框过宽，已经挤占全部画面区域")
    outer = _aa_polygon_mask(size, points)
    edge_masks = []
    border = Image.new("L", size, 0)
    if stroke:
        for edge_index in range(4):
            edge = _aa_line_mask(
                size, points[edge_index], points[(edge_index + 1) % 4], max(1, stroke * 2)
            )
            edge = ImageChops.multiply(edge, outer)
            edge_masks.append(edge)
            border = ImageChops.lighter(border, edge)
    else:
        edge_masks = [Image.new("L", size, 0) for _ in range(4)]
    inner = ImageChops.subtract(outer, border)
    if inner.getbbox() is None:
        raise ValueError(f"{label}边框过宽，已经挤占全部画面区域")
    return {
        "points": points, "bbox": (left, top, right, bottom),
        "inner": inner, "border": border, "edge_masks": edge_masks,
    }


def _render_quad_source(canvas, panel_mask, border_mask, source, geometry, settings, mode, background):
    width, height = canvas.size
    left, top, right, bottom = geometry["bbox"]
    target_size = (right - left, bottom - top)
    resized, offset_x, offset_y = _position_image(source, target_size, mode, settings)
    paste_x, paste_y = left + offset_x, top + offset_y
    image_layer = Image.new("RGB", canvas.size, background)
    source_alpha = resized.getchannel("A") if resized.mode == "RGBA" else None
    image_layer.paste(resized, (paste_x, paste_y), source_alpha)
    placed_alpha = _placed_source_alpha(resized, canvas.size, (paste_x, paste_y))

    display_mask = geometry["inner"]
    open_edges = settings.get("open_edges", (False, False, False, False))
    if any(open_edges):
        presence = Image.new("L", canvas.size, 0)
        ImageDraw.Draw(presence).rectangle(
            (paste_x, paste_y, paste_x + resized.width - 1, paste_y + resized.height - 1), fill=255
        )
        overflow = Image.new("L", canvas.size, 0)
        for edge_index, is_open in enumerate(open_edges):
            if not is_open:
                continue
            extension = _edge_extrusion_mask(canvas.size, geometry["points"], edge_index)
            extension = ImageChops.lighter(extension, geometry["edge_masks"][edge_index])
            overflow = ImageChops.lighter(overflow, extension)
        overflow = ImageChops.multiply(overflow, presence)
        display_mask = ImageChops.lighter(display_mask, overflow)

    effective_mask = ImageChops.multiply(display_mask, placed_alpha)
    canvas.paste(image_layer, (0, 0), effective_mask)
    panel_mask.paste(255, (0, 0), ImageChops.lighter(panel_mask, display_mask))
    border_mask.paste(ImageChops.subtract(border_mask, effective_mask))


def render_comic_panels(
    image, layout: str, canvas_width: int, canvas_height: int, page_margin: int,
    gutter: int, border_width: int, border_color: str, background_color: str,
    fit_mode: str, empty_fill: str, panel_data: str,
):
    frames = image_tensor_to_pil_batch(image, preserve_alpha=True)
    width, height = int(canvas_width), int(canvas_height)
    margin, gap, stroke = int(page_margin), int(gutter), int(border_width)
    if width < 64 or height < 64:
        raise ValueError("漫画画布宽高不能小于 64")
    if margin < 0 or gap < 0 or stroke < 0:
        raise ValueError("页边距、画格间距和边框宽度不能为负数")
    if width - 2 * margin < 1 or height - 2 * margin < 1:
        raise ValueError("页边距不能占满漫画画布")
    if layout != AUTO_LAYOUT and layout not in PANEL_LAYOUTS:
        raise ValueError(f"未知分镜模板：{layout}")
    if fit_mode not in ("裁切填充", "完整显示"):
        raise ValueError(f"未知图片适配方式：{fit_mode}")
    if empty_fill not in ("留空", "循环填充", "复制最后一张"):
        raise ValueError(f"未知空格填充方式：{empty_fill}")
    panels, layout_overrides, quad_overrides, layer_orders, canonical = _parse_panel_config(panel_data)
    bg, line = _parse_hex(background_color), _parse_hex(border_color)

    pages: list[tuple[str, list[Image.Image | None]]] = []
    if layout == AUTO_LAYOUT:
        for start in range(0, len(frames), AUTO_PANELS_PER_PAGE):
            chunk = frames[start:start + AUTO_PANELS_PER_PAGE]
            pages.append((_auto_layout(len(chunk)), list(chunk)))
    else:
        capacity = len(quad_overrides.get(layout, layout_overrides.get(layout, PANEL_LAYOUTS[layout])))
        for start in range(0, len(frames), capacity):
            chunk: list[Image.Image | None] = list(frames[start:start + capacity])
            if len(chunk) < capacity and empty_fill != "留空":
                while len(chunk) < capacity:
                    if empty_fill == "循环填充":
                        chunk.append(frames[(start + len(chunk)) % len(frames)])
                    else:
                        chunk.append(chunk[-1])
            chunk.extend([None] * (capacity - len(chunk)))
            pages.append((layout, chunk))

    outputs, panel_masks, border_masks = [], [], []
    for page_layout, sources in pages:
        canvas = Image.new("RGB", (width, height), bg)
        panel_mask = Image.new("L", (width, height), 0)
        border_mask = Image.new("L", (width, height), 0)
        canvas_draw, panel_draw, border_draw = ImageDraw.Draw(canvas), ImageDraw.Draw(panel_mask), ImageDraw.Draw(border_mask)
        custom_quads = quad_overrides.get(page_layout)
        if custom_quads is not None:
            geometry = [
                _quad_geometry(
                    quad, (width, height), margin, stroke,
                    f"{page_layout} 第 {index + 1} 个四边形",
                )
                for index, quad in enumerate(custom_quads)
            ]
            border_layer = Image.new("RGB", (width, height), line)
            for item in geometry:
                canvas.paste(border_layer, (0, 0), item["border"])
                panel_mask.paste(255, (0, 0), item["inner"])
                border_mask.paste(255, (0, 0), item["border"])
            order = layer_orders.get(page_layout, tuple(range(len(custom_quads))))
            for index in order:
                source = sources[index] if index < len(sources) else None
                if source is not None:
                    _render_quad_source(
                        canvas, panel_mask, border_mask, source, geometry[index],
                        panels[index], fit_mode, bg,
                    )
            outputs.append(canvas)
            panel_masks.append(panel_mask)
            border_masks.append(border_mask)
            continue
        custom_rectangles = layout_overrides.get(page_layout)
        rectangles = custom_rectangles or PANEL_LAYOUTS[page_layout]
        geometry = []
        for index, normalized in enumerate(rectangles):
            if custom_rectangles is not None:
                x0, y0, x1, y1 = _absolute_pixel_rect(normalized, width, height, margin)
            else:
                x0, y0, x1, y1 = _pixel_rect(normalized, width, height, margin, gap)
            if stroke:
                canvas_draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=line)
                border_draw.rectangle((x0, y0, x1 - 1, y1 - 1), fill=255, width=stroke)
            inner = (x0 + stroke, y0 + stroke, x1 - stroke, y1 - stroke)
            if inner[2] <= inner[0] or inner[3] <= inner[1]:
                raise ValueError("画格边框过宽，已经挤占全部画面区域")
            panel_draw.rectangle((inner[0], inner[1], inner[2] - 1, inner[3] - 1), fill=255)
            geometry.append(inner)

        order = layer_orders.get(page_layout, tuple(range(len(rectangles))))
        for index in order:
            inner = geometry[index]
            source = sources[index] if index < len(sources) else None
            if source is not None:
                settings = panels[index]
                opened = any(settings[key] for key in (
                    "overflow_top", "overflow_bottom", "overflow_left", "overflow_right"
                ))
                if not opened:
                    fitted = _fit_image(
                        source, (inner[2] - inner[0], inner[3] - inner[1]), fit_mode, settings, bg
                    )
                    canvas.paste(fitted, (inner[0], inner[1]))
                else:
                    resized, offset_x, offset_y = _position_image(
                        source, (inner[2] - inner[0], inner[3] - inner[1]), fit_mode, settings
                    )
                    paste_x, paste_y = inner[0] + offset_x, inner[1] + offset_y
                    clip_left = 0 if settings["overflow_left"] else inner[0]
                    clip_top = 0 if settings["overflow_top"] else inner[1]
                    clip_right = width if settings["overflow_right"] else inner[2]
                    clip_bottom = height if settings["overflow_bottom"] else inner[3]
                    visible_left = max(0, clip_left, paste_x)
                    visible_top = max(0, clip_top, paste_y)
                    visible_right = min(width, clip_right, paste_x + resized.width)
                    visible_bottom = min(height, clip_bottom, paste_y + resized.height)
                    if visible_right > visible_left and visible_bottom > visible_top:
                        crop = resized.crop((
                            visible_left - paste_x, visible_top - paste_y,
                            visible_right - paste_x, visible_bottom - paste_y,
                        ))
                        crop_alpha = crop.getchannel("A") if crop.mode == "RGBA" else None
                        canvas.paste(crop, (visible_left, visible_top), crop_alpha)
                        if crop_alpha is None:
                            panel_draw.rectangle(
                                (visible_left, visible_top, visible_right - 1, visible_bottom - 1), fill=255
                            )
                            border_draw.rectangle(
                                (visible_left, visible_top, visible_right - 1, visible_bottom - 1), fill=0
                            )
                        else:
                            panel_mask.paste(255, (visible_left, visible_top), crop_alpha)
                            current_border = border_mask.crop(
                                (visible_left, visible_top, visible_right, visible_bottom)
                            )
                            border_mask.paste(
                                ImageChops.subtract(current_border, crop_alpha),
                                (visible_left, visible_top),
                            )
        outputs.append(canvas)
        panel_masks.append(panel_mask)
        border_masks.append(border_mask)
    return (
        pil_batch_to_image_tensor(outputs), pil_batch_to_mask_tensor(panel_masks),
        pil_batch_to_mask_tensor(border_masks), canonical,
    )


def _parse_bubble_document(
    value: str, default_font: str = DEFAULT_FONT_TOKEN,
) -> tuple[list[dict[str, Any]], bool, str]:
    parsed = _decode_object(value, "对话框配置")
    if parsed and parsed.get("version") != BUBBLE_SCHEMA_VERSION:
        raise ValueError(f"对话框配置 version 必须为 {BUBBLE_SCHEMA_VERSION}")
    merge_overlaps = parsed.get("merge_overlaps", False)
    if not isinstance(merge_overlaps, bool):
        raise ValueError("对话框配置 merge_overlaps 必须是布尔值")
    raw_bubbles = parsed.get("bubbles", [])
    if not isinstance(raw_bubbles, list):
        raise ValueError("对话框配置 bubbles 必须是数组")
    if len(raw_bubbles) > 32:
        raise ValueError("每张漫画最多只能添加 32 个对话框")
    bubbles = []
    for index, raw in enumerate(raw_bubbles):
        if not isinstance(raw, dict):
            raise ValueError(f"第 {index + 1} 个对话框配置必须是对象")
        shape = str(raw.get("shape", "椭圆对白框"))
        if shape not in BUBBLE_SHAPES:
            raise ValueError(f"第 {index + 1} 个对话框形状无效：{shape}")
        text_direction = str(raw.get("text_direction", "horizontal"))
        if text_direction not in BUBBLE_TEXT_DIRECTIONS:
            raise ValueError(f"第 {index + 1} 个对话框文字方向无效：{text_direction}")
        font_size = _finite_number(raw.get("font_size", 36), f"第 {index + 1} 个对话框字号", 8, 512)
        border_width = _finite_number(raw.get("border_width", 4), f"第 {index + 1} 个对话框边框", 0, 64)
        spike_count_value = _finite_number(
            raw.get("spike_count", BUBBLE_SPIKE_COUNT_DEFAULT),
            f"第 {index + 1} 个爆炸框尖角数量", 6, 32,
        )
        if not spike_count_value.is_integer():
            raise ValueError(f"第 {index + 1} 个爆炸框尖角数量必须是整数")
        cloud_lobes_value = _finite_number(
            raw.get("cloud_lobes", BUBBLE_CLOUD_LOBES_DEFAULT),
            f"第 {index + 1} 个云朵框云瓣数量", 6, 16,
        )
        if not cloud_lobes_value.is_integer():
            raise ValueError(f"第 {index + 1} 个云朵框云瓣数量必须是整数")
        bubble = {
            "id": str(raw.get("id", f"bubble-{index + 1}"))[:64],
            "shape": shape,
            "x": _finite_number(raw.get("x", 0.5), f"第 {index + 1} 个对话框 X", 0.0, 1.0),
            "y": _finite_number(raw.get("y", 0.3), f"第 {index + 1} 个对话框 Y", 0.0, 1.0),
            "w": _finite_number(raw.get("w", 0.3), f"第 {index + 1} 个对话框宽度", 0.03, 1.0),
            "h": _finite_number(raw.get("h", 0.2), f"第 {index + 1} 个对话框高度", 0.03, 1.0),
            "text": str(raw.get("text", "")),
            "text_direction": text_direction,
            "font_name": str(raw.get("font_name", default_font)),
            "font_size": int(round(font_size)),
            "text_color": str(raw.get("text_color", "#111111")),
            "fill_color": str(raw.get("fill_color", "#ffffff")),
            "border_color": str(raw.get("border_color", "#111111")),
            "border_width": int(round(border_width)),
            "opacity": _finite_number(raw.get("opacity", 1.0), f"第 {index + 1} 个对话框透明度", 0.0, 1.0),
            "spike_count": int(spike_count_value),
            "spike_depth": _finite_number(
                raw.get("spike_depth", BUBBLE_SPIKE_DEPTH_DEFAULT),
                f"第 {index + 1} 个爆炸框尖角深度", 0.05, 0.70,
            ),
            "cloud_lobes": int(cloud_lobes_value),
            "cloud_depth": _finite_number(
                raw.get("cloud_depth", BUBBLE_CLOUD_DEPTH_DEFAULT),
                f"第 {index + 1} 个云朵框云瓣起伏", 0.05, 0.30,
            ),
        }
        _parse_hex(bubble["text_color"]); _parse_hex(bubble["fill_color"]); _parse_hex(bubble["border_color"])
        bubbles.append(bubble)
    canonical = json.dumps(
        {"version": BUBBLE_SCHEMA_VERSION, "merge_overlaps": merge_overlaps, "bubbles": bubbles},
        ensure_ascii=False, separators=(",", ":"),
    )
    return bubbles, merge_overlaps, canonical


def parse_bubble_data(value: str, default_font: str = DEFAULT_FONT_TOKEN) -> tuple[list[dict[str, Any]], str]:
    bubbles, _, canonical = _parse_bubble_document(value, default_font)
    return bubbles, canonical


def _bubble_shape(size: tuple[int, int], bubble: dict[str, Any]) -> Image.Image:
    width, height = size
    cx, cy = round(bubble["x"] * width), round(bubble["y"] * height)
    box_w, box_h = max(2, round(bubble["w"] * width)), max(2, round(bubble["h"] * height))
    box = (cx - box_w // 2, cy - box_h // 2, cx + (box_w + 1) // 2, cy + (box_h + 1) // 2)
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    shape = bubble["shape"]
    if shape == "无边框文字":
        return mask
    if shape == "椭圆对白框":
        draw.ellipse(box, fill=255)
    elif shape == "圆角矩形":
        draw.rounded_rectangle(box, radius=max(2, min(box_w, box_h) // 6), fill=255)
    elif shape == "方形旁白框":
        draw.rectangle(box, fill=255)
    elif shape == "闪光对话框":
        left, top = max(0, math.floor(cx - box_w * .5)), max(0, math.floor(cy - box_h * .5))
        right, bottom = min(width, math.ceil(cx + box_w * .5) + 1), min(height, math.ceil(cy + box_h * .5) + 1)
        if right > left and bottom > top:
            xs = (np.arange(left, right, dtype=np.float32) - cx) / box_w
            ys = (np.arange(top, bottom, dtype=np.float32) - cy) / box_h
            radius = np.sqrt(ys[:, None] ** 2 + xs[None, :] ** 2)
            fade = np.clip((.50 - radius) / (.50 - .34), 0.0, 1.0)
            fade = fade * fade * (3.0 - 2.0 * fade)
            mask.paste(Image.fromarray(np.round(fade * 255).astype(np.uint8), "L"), (left, top))
    elif shape == "爆炸喊话框":
        points = []
        point_count = bubble["spike_count"] * 2
        inner_radius = 0.5 * (1.0 - bubble["spike_depth"])
        for index in range(point_count):
            angle = -math.pi / 2 + index * math.tau / point_count
            radius = 0.5 if index % 2 == 0 else inner_radius
            points.append((cx + math.cos(angle) * box_w * radius, cy + math.sin(angle) * box_h * radius))
        draw.polygon(points, fill=255)
    elif shape == "爆炸对话框":
        points = []

        def cubic(start, control_a, control_b, end, samples=12):
            for sample in range(samples):
                t = sample / samples
                inverse = 1.0 - t
                points.append((
                    inverse ** 3 * start[0] + 3 * inverse * inverse * t * control_a[0] + 3 * inverse * t * t * control_b[0] + t ** 3 * end[0],
                    inverse ** 3 * start[1] + 3 * inverse * inverse * t * control_a[1] + 3 * inverse * t * t * control_b[1] + t ** 3 * end[1],
                ))

        def point(nx, ny):
            return cx + nx * box_w, cy + ny * box_h

        top_left = point(-.48, -.46)
        cubic(top_left, point(-.39, -.39), point(-.30, -.34), point(-.22, -.36))
        points.extend((point(-.17, -.35), point(-.23, -.40), point(-.13, -.36)))
        cubic(point(-.13, -.36), point(.13, -.35), point(.34, -.42), point(.48, -.50))
        cubic(point(.48, -.50), point(.45, -.40), point(.42, -.31), point(.43, -.25))
        points.extend((point(.54, -.32), point(.45, -.18)))
        cubic(point(.45, -.18), point(.41, -.02), point(.41, .13), point(.44, .22))
        points.extend((point(.50, .16), point(.44, .31), point(.48, .48)))
        cubic(point(.48, .48), point(.20, .37), point(-.18, .35), point(-.43, .49))
        points.extend((point(-.35, .31), point(-.48, .36), point(-.41, .19)))
        cubic(point(-.41, .19), point(-.38, .02), point(-.40, -.28), top_left)
        draw.polygon(points, fill=255)
    else:
        points = []
        lobe_count = bubble["cloud_lobes"]
        cloud_depth = bubble["cloud_depth"]
        base_valley_radius = 0.48 - 0.35 * cloud_depth
        base_control_radius = 0.54 + 0.15 * cloud_depth
        samples_per_lobe = 12
        weights = [1.0 + .20 * math.sin(index * 2.17 + .3) + .10 * math.sin(index * .91 + 1.4) for index in range(lobe_count)]
        angle_steps = [math.tau * weight / sum(weights) for weight in weights]
        start_angle = -math.pi / 2 - angle_steps[0] / 2
        current_angle = start_angle
        for lobe in range(lobe_count):
            angle_step = angle_steps[lobe]
            start, middle, end = current_angle, current_angle + angle_step / 2, current_angle + angle_step
            start_radius = base_valley_radius * (1.0 + .045 * math.sin(lobe * 1.73 + .5))
            end_radius = base_valley_radius * (1.0 + .045 * math.sin((lobe + 1) * 1.73 + .5))
            control_radius = base_control_radius * (1.0 + .12 * math.sin(lobe * 2.39 + .8))
            p0 = (cx + math.cos(start) * box_w * start_radius, cy + math.sin(start) * box_h * start_radius)
            control = (cx + math.cos(middle) * box_w * control_radius, cy + math.sin(middle) * box_h * control_radius)
            p1 = (cx + math.cos(end) * box_w * end_radius, cy + math.sin(end) * box_h * end_radius)
            for sample in range(samples_per_lobe):
                t = sample / samples_per_lobe
                inverse = 1.0 - t
                points.append((
                    inverse * inverse * p0[0] + 2 * inverse * t * control[0] + t * t * p1[0],
                    inverse * inverse * p0[1] + 2 * inverse * t * control[1] + t * t * p1[1],
                ))
            current_angle = end
        draw.polygon(points, fill=255)
    return mask


def _flash_border_mask(size: tuple[int, int], bubble: dict[str, Any]) -> Image.Image:
    width, height = size
    cx, cy = round(bubble["x"] * width), round(bubble["y"] * height)
    box_w, box_h = max(2, round(bubble["w"] * width)), max(2, round(bubble["h"] * height))
    mask = Image.new("L", size, 0)
    if bubble["border_width"] <= 0:
        return mask
    draw = ImageDraw.Draw(mask)
    ray_count = 96
    angle_step = math.tau / ray_count
    half_angle = angle_step * min(.32, .16 + bubble["border_width"] / 80)
    for index in range(ray_count):
        angle = -math.pi / 2 + index * angle_step
        outer_radius = .47 + .03 * (.5 + .5 * math.sin(index * 2.07 + .4))
        draw.polygon((
            (cx + math.cos(angle) * box_w * .31, cy + math.sin(angle) * box_h * .31),
            (cx + math.cos(angle - half_angle) * box_w * .40, cy + math.sin(angle - half_angle) * box_h * .40),
            (cx + math.cos(angle) * box_w * outer_radius, cy + math.sin(angle) * box_h * outer_radius),
            (cx + math.cos(angle + half_angle) * box_w * .40, cy + math.sin(angle + half_angle) * box_h * .40),
        ), fill=255)
    return mask


def _scaled_mask(mask: Image.Image, opacity: float) -> Image.Image:
    return mask.point(lambda value: max(0, min(255, round(value * opacity))))


def _text_mask(size: tuple[int, int], bubble: dict[str, Any]) -> Image.Image:
    if not bubble["text"]:
        return Image.new("L", size, 0)
    width, height = size
    render_font_name = font_for_text(bubble["font_name"], bubble["text"])
    cx, cy = round(bubble["x"] * width), round(bubble["y"] * height)
    box_w, box_h = max(2, round(bubble["w"] * width)), max(2, round(bubble["h"] * height))
    padding = max(3, round(min(box_w, box_h) * (0.16 if bubble["shape"] in ("爆炸喊话框", "爆炸对话框", "闪光对话框") else 0.1)))
    region = (
        max(0, cx - box_w // 2 + padding), max(0, cy - box_h // 2 + padding),
        min(width, cx + (box_w + 1) // 2 - padding), min(height, cy + (box_h + 1) // 2 - padding),
    )
    if region[2] <= region[0] or region[3] <= region[1]:
        return Image.new("L", size, 0)
    if bubble["text_direction"] != "horizontal":
        region_width, region_height = region[2] - region[0], region[3] - region[1]
        font_size = bubble["font_size"]

        def vertical_columns(capacity: int) -> list[list[str]]:
            columns = []
            for paragraph in bubble["text"].split("\n"):
                characters = list(paragraph)
                if not characters:
                    columns.append([])
                else:
                    columns.extend(characters[offset:offset + capacity] for offset in range(0, len(characters), capacity))
            return columns

        while True:
            line_step = max(1, round(font_size * 1.15))
            column_step = max(1, round(font_size * 1.15))
            capacity = max(1, region_height // line_step)
            columns = vertical_columns(capacity)
            if len(columns) * column_step <= region_width or font_size <= 8:
                break
            font_size = max(8, font_size - 2)
        max_columns = max(1, region_width // column_step)
        columns = columns[:max_columns]
        local = Image.new("L", (region_width, region_height), 0)
        draw = ImageDraw.Draw(local)
        font = load_font(render_font_name, font_size)
        total_width = len(columns) * column_step
        left = (region_width - total_width) / 2
        for logical_index, characters in enumerate(columns):
            visual_index = logical_index if bubble["text_direction"] == "vertical_ltr" else len(columns) - 1 - logical_index
            x = left + (visual_index + .5) * column_step
            column_height = len(characters) * line_step
            top = (region_height - column_height) / 2
            for row, character in enumerate(characters):
                y = top + (row + .5) * line_step
                bounds = draw.textbbox((0, 0), character, font=font)
                draw.text((x - (bounds[0] + bounds[2]) / 2, y - (bounds[1] + bounds[3]) / 2), character, font=font, fill=255)
        mask = Image.new("L", size, 0)
        mask.paste(local, (region[0], region[1]))
        return mask
    font_size = bubble["font_size"]
    while True:
        layout = measure_text_layout(
            bubble["text"], render_font_name, font_size, size, region=region,
            max_width=region[2] - region[0], line_spacing=max(0, round(font_size * 0.15)),
            justify="center", vertical_align="center",
        )
        total_height = len(layout.lines) * layout.line_height + max(0, len(layout.lines) - 1) * max(0, round(font_size * 0.15))
        if total_height <= region[3] - region[1] or font_size <= 8:
            break
        font_size = max(8, font_size - 2)
    return render_layout_mask(layout)


def render_speech_bubbles(image, enabled: bool, default_font: str, bubble_data: str):
    bubbles, merge_overlaps, canonical = _parse_bubble_document(bubble_data, default_font)
    if not enabled:
        batch = int(image.shape[0]) if getattr(image, "ndim", 0) == 4 else 1
        height, width = int(image.shape[-3]), int(image.shape[-2])
        zero = torch.zeros((batch, height, width), dtype=torch.float32)
        return image, zero, zero.clone(), canonical
    frames = image_tensor_to_pil_batch(image)
    outputs, bubble_masks, text_masks = [], [], []
    for source in frames:
        canvas = source.copy()
        bubble_total = Image.new("L", source.size, 0)
        text_total = Image.new("L", source.size, 0)
        layers = []
        for bubble in bubbles:
            opacity = bubble["opacity"]
            shape = _bubble_shape(source.size, bubble)
            border_width = bubble["border_width"]
            if bubble["shape"] == "闪光对话框":
                border = _flash_border_mask(source.size, bubble)
                outer = ImageChops.lighter(shape, border)
            else:
                outer = shape.filter(ImageFilter.MaxFilter(border_width * 2 + 1)) if border_width else shape
                border = ImageChops.subtract(outer, shape) if border_width else Image.new("L", source.size, 0)
            text = _text_mask(source.size, bubble)
            layers.append((bubble, shape, outer, border, text))

        def composite(mask: Image.Image, color: str, opacity: float) -> None:
            nonlocal canvas
            if mask.getbbox():
                canvas = Image.composite(
                    Image.new("RGB", source.size, _parse_hex(color)), canvas, _scaled_mask(mask, opacity),
                )

        if merge_overlaps:
            parents = list(range(len(layers)))

            def find(index: int) -> int:
                while parents[index] != index:
                    parents[index] = parents[parents[index]]
                    index = parents[index]
                return index

            def union(left: int, right: int) -> None:
                left_root, right_root = find(left), find(right)
                if left_root != right_root:
                    parents[right_root] = left_root

            for left in range(len(layers)):
                left_box = layers[left][1].getbbox()
                if not left_box or layers[left][0]["shape"] == "闪光对话框":
                    continue
                for right in range(left + 1, len(layers)):
                    right_box = layers[right][1].getbbox()
                    if not right_box or layers[right][0]["shape"] == "闪光对话框" or left_box[2] <= right_box[0] or right_box[2] <= left_box[0] or left_box[3] <= right_box[1] or right_box[3] <= left_box[1]:
                        continue
                    if ImageChops.multiply(layers[left][1], layers[right][1]).getbbox():
                        union(left, right)

            groups: dict[int, list[int]] = {}
            for index in range(len(layers)):
                groups.setdefault(find(index), []).append(index)
            for indices in groups.values():
                merged_shape = Image.new("L", source.size, 0)
                for index in indices:
                    merged_shape = ImageChops.lighter(merged_shape, layers[index][1])
                style = layers[max(indices)][0]
                if len(indices) == 1 and style["shape"] == "闪光对话框":
                    merged_outer, border = layers[indices[0]][2], layers[indices[0]][3]
                    composite(merged_shape, style["fill_color"], style["opacity"])
                    composite(border, style["border_color"], style["opacity"])
                else:
                    border_width = style["border_width"]
                    merged_outer = merged_shape.filter(ImageFilter.MaxFilter(border_width * 2 + 1)) if border_width else merged_shape
                    border = ImageChops.subtract(merged_outer, merged_shape) if border_width else Image.new("L", source.size, 0)
                    composite(border, style["border_color"], style["opacity"])
                    composite(merged_shape, style["fill_color"], style["opacity"])
                bubble_total = ImageChops.lighter(bubble_total, _scaled_mask(merged_outer, style["opacity"]))
            for bubble, _, _, _, text in layers:
                composite(text, bubble["text_color"], bubble["opacity"])
                text_total = ImageChops.lighter(text_total, _scaled_mask(text, bubble["opacity"]))
        else:
            for bubble, shape, outer, border, text in layers:
                if bubble["shape"] == "闪光对话框":
                    composite(shape, bubble["fill_color"], bubble["opacity"])
                    composite(border, bubble["border_color"], bubble["opacity"])
                else:
                    composite(border, bubble["border_color"], bubble["opacity"])
                    composite(shape, bubble["fill_color"], bubble["opacity"])
                composite(text, bubble["text_color"], bubble["opacity"])
                bubble_total = ImageChops.lighter(bubble_total, _scaled_mask(outer, bubble["opacity"]))
                text_total = ImageChops.lighter(text_total, _scaled_mask(text, bubble["opacity"]))
        outputs.append(canvas); bubble_masks.append(bubble_total); text_masks.append(text_total)
    return (
        pil_batch_to_image_tensor(outputs), pil_batch_to_mask_tensor(bubble_masks),
        pil_batch_to_mask_tensor(text_masks), canonical,
    )
