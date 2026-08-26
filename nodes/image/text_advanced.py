"""Advanced layout and production text nodes."""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from ...categories import IMAGE_TEXT
from ...core.fonts import DEFAULT_FONT_TOKEN, font_display_name, font_options, load_font
from ...core.imaging import (
    COLOR_NAMES,
    broadcast_batches,
    compose_text_effect,
    image_tensor_to_pil_batch,
    mask_tensor_to_pil_batch,
    parse_color,
    pil_batch_to_image_tensor,
    pil_batch_to_mask_tensor,
)
from ...core.text_layout import (
    fit_text_layout,
    measure_text_layout,
    render_layout_mask,
    render_unit_masks,
)
from .text import _effect_inputs, _font_input


JUSTIFY_OPTIONS = ("center", "left", "right")
VERTICAL_OPTIONS = ("center", "top", "bottom")
SPLIT_OPTIONS = ("字符", "词", "行")


def _help(node_name: str) -> str:
    return f"TUT_Nodes/图片/文本/{node_name}"


def _resize_mask(mask: Image.Image, size: tuple[int, int]) -> Image.Image:
    if mask.size == size:
        return mask.convert("L")
    return mask.convert("L").resize(size, getattr(Image, "Resampling", Image).BILINEAR)


class TUT_区域自适应文字:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "region_mask": ("MASK",),
            "text": ("STRING", {"multiline": True, "default": "自动适应区域的文字"}),
            "font_name": _font_input(),
            "min_font_size": ("INT", {"default": 12, "min": 1, "max": 2048}),
            "max_font_size": ("INT", {"default": 160, "min": 1, "max": 2048}),
            "padding": ("INT", {"default": 12, "min": 0, "max": 1024}),
            "max_lines": ("INT", {"default": 4, "min": 0, "max": 100}),
            "justify": (list(JUSTIFY_OPTIONS),),
            "vertical_align": (list(VERTICAL_OPTIONS),),
            "overflow": (["缩小字号", "截断", "报错"],),
            "font_color": (list(COLOR_NAMES),),
        }
        optional = {
            "line_spacing": ("INT", {"default": 0, "min": -64, "max": 512}),
            "letter_spacing": ("INT", {"default": 0, "min": -32, "max": 256}),
            **_effect_inputs(),
        }
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "text_mask")
    FUNCTION = "fit_text"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "按区域 MASK 的有效边界自动换行并寻找最大可用字号。"

    def fit_text(
        self, image, region_mask, text, font_name, min_font_size, max_font_size,
        padding, max_lines, justify, vertical_align, overflow, font_color,
        line_spacing=0, letter_spacing=0, **effects,
    ):
        frames = image_tensor_to_pil_batch(image)
        regions = mask_tensor_to_pil_batch(region_mask)
        frames, regions = broadcast_batches(frames, regions)
        outputs, text_masks = [], []
        for frame, region_mask_pil in zip(frames, regions):
            region_mask_pil = _resize_mask(region_mask_pil, frame.size)
            if not str(text):
                text_mask = Image.new("L", frame.size, 0)
                outputs.append(frame.copy())
                text_masks.append(text_mask)
                continue
            bbox = region_mask_pil.getbbox()
            if bbox is None:
                raise ValueError("区域 MASK 为空，无法放置文字")
            inset = int(padding)
            region = (bbox[0] + inset, bbox[1] + inset, bbox[2] - inset, bbox[3] - inset)
            if region[2] <= region[0] or region[3] <= region[1]:
                raise ValueError(f"区域 MASK 扣除 padding={inset} 后没有可用空间")
            layout = fit_text_layout(
                str(text), font_name, int(min_font_size), int(max_font_size), frame.size,
                region=region, max_lines=int(max_lines), line_spacing=int(line_spacing),
                letter_spacing=int(letter_spacing), justify=justify,
                vertical_align=vertical_align, overflow=overflow,
            )
            text_mask = render_layout_mask(layout)
            outputs.append(compose_text_effect(frame, text_mask, font_color=font_color, **effects))
            text_masks.append(text_mask)
        return (
            pil_batch_to_image_tensor(outputs),
            _help("TUT_区域自适应文字"),
            pil_batch_to_mask_tensor(text_masks),
        )


class TUT_逐字逐词遮罩:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "text": ("STRING", {"multiline": True, "default": "逐字逐词"}),
            "font_name": _font_input(),
            "font_size": ("INT", {"default": 72, "min": 1, "max": 2048}),
            "split_mode": (list(SPLIT_OPTIONS),),
            "include_whitespace": ("BOOLEAN", {"default": False}),
            "align": (list(VERTICAL_OPTIONS),),
            "justify": (list(JUSTIFY_OPTIONS),),
            "margins": ("INT", {"default": 16, "min": -1024, "max": 2048}),
            "line_spacing": ("INT", {"default": 0, "min": -64, "max": 512}),
            "letter_spacing": ("INT", {"default": 0, "min": -32, "max": 256}),
            "font_color": (list(COLOR_NAMES),),
        }
        return {"required": required, "optional": _effect_inputs()}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "unit_mask")
    FUNCTION = "split_masks"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "按字符、词或行输出逐单位 IMAGE/MASK 批次。"

    def split_masks(
        self, image, text, font_name, font_size, split_mode, include_whitespace,
        align, justify, margins, line_spacing, letter_spacing, font_color, **effects,
    ):
        frames = image_tensor_to_pil_batch(image)
        outputs, masks = [], []
        for frame in frames:
            margin = int(margins)
            region = (margin, margin, frame.width - margin, frame.height - margin)
            if region[2] <= region[0] or region[3] <= region[1]:
                raise ValueError("margins 过大，文字排版区域为空")
            layout = measure_text_layout(
                str(text), font_name, int(font_size), frame.size, region=region,
                line_spacing=int(line_spacing), letter_spacing=int(letter_spacing),
                justify=justify, vertical_align=align,
            )
            unit_masks = render_unit_masks(layout, split_mode, bool(include_whitespace))
            if not unit_masks:
                unit_masks = [Image.new("L", frame.size, 0)]
            for unit_mask in unit_masks:
                outputs.append(compose_text_effect(frame, unit_mask, font_color=font_color, **effects))
                masks.append(unit_mask)
        return (
            pil_batch_to_image_tensor(outputs),
            _help("TUT_逐字逐词遮罩"),
            pil_batch_to_mask_tensor(masks),
        )


class TUT_字体预览墙:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sample_text": ("STRING", {"multiline": False, "default": "字体 Font 123"}),
                "font_scope": (["全部", "插件字体", "系统字体"],),
                "page": ("INT", {"default": 1, "min": 1, "max": 10000}),
                "page_size": ("INT", {"default": 12, "min": 1, "max": 100}),
                "columns": ("INT", {"default": 3, "min": 1, "max": 12}),
                "cell_width": ("INT", {"default": 420, "min": 128, "max": 2048}),
                "cell_height": ("INT", {"default": 150, "min": 80, "max": 1024}),
                "sample_font_size": ("INT", {"default": 44, "min": 8, "max": 256}),
                "background_color": (list(COLOR_NAMES),),
                "text_color": (list(COLOR_NAMES),),
            },
            "optional": {
                "background_color_hex": ("STRING", {"default": "#202020"}),
                "text_color_hex": ("STRING", {"default": "#ffffff"}),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "text_mask")
    FUNCTION = "preview_fonts"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "分页生成插件字体或系统字体的预览网格。"

    def preview_fonts(
        self, sample_text, font_scope, page, page_size, columns, cell_width,
        cell_height, sample_font_size, background_color, text_color,
        background_color_hex="#202020", text_color_hex="#ffffff",
    ):
        options = list(font_options())
        if font_scope == "插件字体":
            options = [token for token in options if token.startswith("builtin/")]
        elif font_scope == "系统字体":
            options = [token for token in options if token.startswith("system/")]
        if not options:
            options = [DEFAULT_FONT_TOKEN]
        start = (int(page) - 1) * int(page_size)
        selected = options[start : start + int(page_size)]
        if not selected:
            total_pages = max(1, math.ceil(len(options) / int(page_size)))
            raise ValueError(f"字体预览页超出范围：当前共 {total_pages} 页")

        cols = int(columns)
        rows = math.ceil(len(selected) / cols)
        cell_w, cell_h = int(cell_width), int(cell_height)
        background_rgb = parse_color(background_color, background_color_hex, (32, 32, 32))
        text_rgb = parse_color(text_color, text_color_hex, (255, 255, 255))
        canvas = Image.new("RGB", (cols * cell_w, rows * cell_h), background_rgb)
        mask = Image.new("L", canvas.size, 0)
        draw, mask_draw = ImageDraw.Draw(canvas), ImageDraw.Draw(mask)
        label_font = load_font(DEFAULT_FONT_TOKEN, max(10, min(18, cell_h // 7)))
        for index, token in enumerate(selected):
            left = (index % cols) * cell_w
            top = (index // cols) * cell_h
            try:
                sample_font = load_font(token, int(sample_font_size))
                draw.text((left + 12, top + 10), str(sample_text), font=sample_font, fill=text_rgb, anchor="lt")
                mask_draw.text((left + 12, top + 10), str(sample_text), font=sample_font, fill=255, anchor="lt")
            except (OSError, ValueError):
                draw.text((left + 12, top + 10), "[font unavailable]", font=label_font, fill=(255, 96, 96), anchor="lt")
                mask_draw.text((left + 12, top + 10), "[font unavailable]", font=label_font, fill=255, anchor="lt")
            full_label = f"{font_display_name(token)} · {token}"
            label = full_label if len(full_label) <= 55 else full_label[:52] + "..."
            label_y = top + cell_h - max(26, cell_h // 5)
            draw.text((left + 12, label_y), label, font=label_font, fill=text_rgb, anchor="lt")
            mask_draw.text((left + 12, label_y), label, font=label_font, fill=255, anchor="lt")
        return (
            pil_batch_to_image_tensor([canvas]),
            _help("TUT_字体预览墙"),
            pil_batch_to_mask_tensor([mask]),
        )


NODE_CLASS_MAPPINGS = {
    "TUT_FitTextToRegion": TUT_区域自适应文字,
    "TUT_SplitTextMasks": TUT_逐字逐词遮罩,
    "TUT_FontPreviewWall": TUT_字体预览墙,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_FitTextToRegion": "TUT_区域自适应文字",
    "TUT_SplitTextMasks": "TUT_逐字逐词遮罩",
    "TUT_FontPreviewWall": "TUT_字体预览墙",
}
