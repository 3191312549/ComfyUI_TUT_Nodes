"""Comic panel layout and speech-bubble rendering helpers."""

from __future__ import annotations

import json
import math
from typing import Any

import numpy as np
import torch
from PIL import Image, ImageDraw

from .imaging import (
    image_tensor_to_pil_batch,
    pil_batch_to_image_tensor,
    pil_batch_to_mask_tensor,
)
PANEL_SCHEMA_VERSION = 1

AUTO_LAYOUT = "自动匹配数量"
CUSTOM_LAYOUT = "自由画框"
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


def _parse_panel_config(value: str) -> tuple[
    list[dict[str, Any]],
    dict[str, tuple[tuple[float, float, float, float], ...]],
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
        panels.append({
            "focus_x": _finite_number(raw.get("focus_x", 0.5), f"第 {index + 1} 格焦点 X", 0.0, 1.0),
            "focus_y": _finite_number(raw.get("focus_y", 0.5), f"第 {index + 1} 格焦点 Y", 0.0, 1.0),
            "zoom": _finite_number(raw.get("zoom", 1.0), f"第 {index + 1} 格缩放", 0.25, 4.0),
            "flip": flip,
            **overflow,
        })
    while len(panels) < 6:
        panels.append({
            "focus_x": 0.5, "focus_y": 0.5, "zoom": 1.0, "flip": False,
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

    raw_layer_orders = parsed.get("layer_orders", {})
    if not isinstance(raw_layer_orders, dict):
        raise ValueError("分镜配置 layer_orders 必须是对象")
    layer_orders: dict[str, tuple[int, ...]] = {}
    for layout_name, raw_order in raw_layer_orders.items():
        if layout_name not in PANEL_LAYOUTS:
            raise ValueError(f"分镜图层包含未知模板：{layout_name}")
        count = len(layout_overrides.get(layout_name, PANEL_LAYOUTS[layout_name]))
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
    if layer_orders:
        canonical_data["layer_orders"] = {
            name: list(order) for name, order in layer_orders.items()
        }
    canonical = json.dumps(
        canonical_data, ensure_ascii=False, separators=(",", ":")
    )
    return panels, layout_overrides, layer_orders, canonical


def parse_panel_data(value: str) -> tuple[list[dict[str, Any]], str]:
    panels, _layout_overrides, _layer_orders, canonical = _parse_panel_config(value)
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
    panels, layout_overrides, layer_orders, canonical = _parse_panel_config(panel_data)
    bg, line = _parse_hex(background_color), _parse_hex(border_color)

    pages: list[tuple[str, list[Image.Image | None]]] = []
    if layout == AUTO_LAYOUT:
        for start in range(0, len(frames), 6):
            chunk = frames[start:start + 6]
            pages.append((_auto_layout(len(chunk)), list(chunk)))
    else:
        capacity = len(layout_overrides.get(layout, PANEL_LAYOUTS[layout]))
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
