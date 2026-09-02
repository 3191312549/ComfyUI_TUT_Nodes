"""Comic panel layout and speech-bubble rendering helpers."""

from __future__ import annotations

import json
import math
from functools import lru_cache
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageChops, ImageDraw, ImageFilter

from .fonts import DEFAULT_FONT_TOKEN, load_font
from .imaging import (
    image_tensor_to_pil_batch,
    pil_batch_to_image_tensor,
    pil_batch_to_mask_tensor,
)
from .text_layout import measure_text_layout, render_layout_mask


PANEL_SCHEMA_VERSION = 1
BUBBLE_SCHEMA_VERSION = 1

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

BUBBLE_SHAPES = ("椭圆对白框", "圆角矩形", "云朵思考框", "爆炸喊话框", "方形旁白框", "无边框文字")


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
    if len(raw_panels) > 6:
        raise ValueError("每页最多只能配置 6 个画格")
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
    while len(panels) < 6:
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
            if not isinstance(raw_rectangles, list) or not 1 <= len(raw_rectangles) <= 6:
                raise ValueError("自由画框的数量必须为 1 到 6")
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
            if not isinstance(raw_layout_quads, list) or not 1 <= len(raw_layout_quads) <= 6:
                raise ValueError("自由画框的四边形数量必须为 1 到 6")
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
    result.paste(resized, (paste_x, paste_y))
    return result


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
    image_layer.paste(resized, (paste_x, paste_y))

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

    canvas.paste(image_layer, (0, 0), display_mask)
    panel_mask.paste(255, (0, 0), ImageChops.lighter(panel_mask, display_mask))
    border_mask.paste(ImageChops.subtract(border_mask, display_mask))


def render_comic_panels(
    image, layout: str, canvas_width: int, canvas_height: int, page_margin: int,
    gutter: int, border_width: int, border_color: str, background_color: str,
    fit_mode: str, empty_fill: str, panel_data: str,
):
    frames = image_tensor_to_pil_batch(image)
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
        for start in range(0, len(frames), 6):
            chunk = frames[start:start + 6]
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
                        canvas.paste(crop, (visible_left, visible_top))
                        panel_draw.rectangle(
                            (visible_left, visible_top, visible_right - 1, visible_bottom - 1), fill=255
                        )
                        border_draw.rectangle(
                            (visible_left, visible_top, visible_right - 1, visible_bottom - 1), fill=0
                        )
        outputs.append(canvas)
        panel_masks.append(panel_mask)
        border_masks.append(border_mask)
    return (
        pil_batch_to_image_tensor(outputs), pil_batch_to_mask_tensor(panel_masks),
        pil_batch_to_mask_tensor(border_masks), canonical,
    )


def parse_bubble_data(value: str, default_font: str = DEFAULT_FONT_TOKEN) -> tuple[list[dict[str, Any]], str]:
    parsed = _decode_object(value, "对话框配置")
    if parsed and parsed.get("version") != BUBBLE_SCHEMA_VERSION:
        raise ValueError(f"对话框配置 version 必须为 {BUBBLE_SCHEMA_VERSION}")
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
        font_size = _finite_number(raw.get("font_size", 36), f"第 {index + 1} 个对话框字号", 8, 512)
        border_width = _finite_number(raw.get("border_width", 4), f"第 {index + 1} 个对话框边框", 0, 64)
        bubble = {
            "id": str(raw.get("id", f"bubble-{index + 1}"))[:64],
            "shape": shape,
            "x": _finite_number(raw.get("x", 0.5), f"第 {index + 1} 个对话框 X", 0.0, 1.0),
            "y": _finite_number(raw.get("y", 0.3), f"第 {index + 1} 个对话框 Y", 0.0, 1.0),
            "w": _finite_number(raw.get("w", 0.3), f"第 {index + 1} 个对话框宽度", 0.03, 1.0),
            "h": _finite_number(raw.get("h", 0.2), f"第 {index + 1} 个对话框高度", 0.03, 1.0),
            "text": str(raw.get("text", "")),
            "font_name": str(raw.get("font_name", default_font)),
            "font_size": int(round(font_size)),
            "text_color": str(raw.get("text_color", "#111111")),
            "fill_color": str(raw.get("fill_color", "#ffffff")),
            "border_color": str(raw.get("border_color", "#111111")),
            "border_width": int(round(border_width)),
            "opacity": _finite_number(raw.get("opacity", 1.0), f"第 {index + 1} 个对话框透明度", 0.0, 1.0),
        }
        _parse_hex(bubble["text_color"]); _parse_hex(bubble["fill_color"]); _parse_hex(bubble["border_color"])
        bubbles.append(bubble)
    canonical = json.dumps(
        {"version": BUBBLE_SCHEMA_VERSION, "bubbles": bubbles}, ensure_ascii=False, separators=(",", ":")
    )
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
    elif shape == "爆炸喊话框":
        points = []
        for index in range(32):
            angle = -math.pi / 2 + index * math.pi / 16
            radius = 0.5 if index % 2 == 0 else 0.39
            points.append((cx + math.cos(angle) * box_w * radius, cy + math.sin(angle) * box_h * radius))
        draw.polygon(points, fill=255)
    else:
        for index in range(12):
            angle = index * math.pi / 6
            px = cx + math.cos(angle) * box_w * 0.42
            py = cy + math.sin(angle) * box_h * 0.38
            radius = max(4, min(box_w, box_h) // 6)
            draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=255)
        draw.ellipse(box, fill=255)
    return mask


def _scaled_mask(mask: Image.Image, opacity: float) -> Image.Image:
    return mask.point(lambda value: max(0, min(255, round(value * opacity))))


def _text_mask(size: tuple[int, int], bubble: dict[str, Any]) -> Image.Image:
    if not bubble["text"]:
        return Image.new("L", size, 0)
    width, height = size
    cx, cy = round(bubble["x"] * width), round(bubble["y"] * height)
    box_w, box_h = max(2, round(bubble["w"] * width)), max(2, round(bubble["h"] * height))
    padding = max(3, round(min(box_w, box_h) * (0.15 if bubble["shape"] == "爆炸喊话框" else 0.1)))
    region = (
        max(0, cx - box_w // 2 + padding), max(0, cy - box_h // 2 + padding),
        min(width, cx + (box_w + 1) // 2 - padding), min(height, cy + (box_h + 1) // 2 - padding),
    )
    if region[2] <= region[0] or region[3] <= region[1]:
        return Image.new("L", size, 0)
    font_size = bubble["font_size"]
    while True:
        layout = measure_text_layout(
            bubble["text"], bubble["font_name"], font_size, size, region=region,
            max_width=region[2] - region[0], line_spacing=max(0, round(font_size * 0.15)),
            justify="center", vertical_align="center",
        )
        total_height = len(layout.lines) * layout.line_height + max(0, len(layout.lines) - 1) * max(0, round(font_size * 0.15))
        if total_height <= region[3] - region[1] or font_size <= 8:
            break
        font_size = max(8, font_size - 2)
    return render_layout_mask(layout)


def render_speech_bubbles(image, enabled: bool, default_font: str, bubble_data: str):
    bubbles, canonical = parse_bubble_data(bubble_data, default_font)
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
        for bubble in bubbles:
            opacity = bubble["opacity"]
            shape = _bubble_shape(source.size, bubble)
            border_width = bubble["border_width"]
            outer = shape.filter(ImageFilter.MaxFilter(border_width * 2 + 1)) if border_width else shape
            border = ImageChops.subtract(outer, shape) if border_width else Image.new("L", source.size, 0)
            border_alpha, fill_alpha = _scaled_mask(border, opacity), _scaled_mask(shape, opacity)
            if border.getbbox():
                canvas = Image.composite(Image.new("RGB", source.size, _parse_hex(bubble["border_color"])), canvas, border_alpha)
            if shape.getbbox():
                canvas = Image.composite(Image.new("RGB", source.size, _parse_hex(bubble["fill_color"])), canvas, fill_alpha)
            text = _text_mask(source.size, bubble)
            text_alpha = _scaled_mask(text, opacity)
            if text.getbbox():
                canvas = Image.composite(Image.new("RGB", source.size, _parse_hex(bubble["text_color"])), canvas, text_alpha)
            bubble_total = Image.fromarray(np.maximum(np.asarray(bubble_total), np.asarray(_scaled_mask(outer, opacity))).astype(np.uint8), "L")
            text_total = Image.fromarray(np.maximum(np.asarray(text_total), np.asarray(text_alpha)).astype(np.uint8), "L")
        outputs.append(canvas); bubble_masks.append(bubble_total); text_masks.append(text_total)
    return (
        pil_batch_to_image_tensor(outputs), pil_batch_to_mask_tensor(bubble_masks),
        pil_batch_to_mask_tensor(text_masks), canonical,
    )
