"""Image/text nodes for the ``TUT_Nodes/图片/文本`` menu."""

from __future__ import annotations

import torch
from PIL import Image

from ...categories import IMAGE_TEXT
from ...core.fonts import font_options
from ...core.imaging import (
    COLOR_NAMES,
    broadcast_batches,
    compose_text_effect,
    image_tensor_to_pil_batch,
    mask_tensor_to_pil_batch,
    parse_color,
    pil_batch_to_image_tensor,
    pil_batch_to_mask_tensor,
    render_text_mask,
)
from ...core.watermark import CORNER_POSITIONS, render_adaptive_watermark


ALIGN_OPTIONS = ("center", "top", "bottom")
JUSTIFY_OPTIONS = ("center", "left", "right")
ROTATION_OPTIONS = ("text center", "image center")
WATERMARK_ALIGNMENTS = (
    "center",
    "top left",
    "top center",
    "top right",
    "bottom left",
    "bottom center",
    "bottom right",
)
EFFECT_PRESETS = ("none", "custom", "outline", "shadow", "glow", "neon", "gradient", "emboss", "extrude")
FILL_MODES = ("solid", "linear_gradient", "radial_gradient")
BLEND_MODES = ("normal", "multiply", "screen", "overlay")


def _font_input():
    return (list(font_options()),)


def _layout_inputs():
    return {
        "align": (list(ALIGN_OPTIONS),),
        "justify": (list(JUSTIFY_OPTIONS),),
        "margins": ("INT", {"default": 0, "min": -1024, "max": 1024}),
        "line_spacing": ("INT", {"default": 0, "min": -256, "max": 1024}),
        "letter_spacing": ("INT", {"default": 0, "min": -32, "max": 256}),
        "position_x": ("INT", {"default": 0, "min": -4096, "max": 4096}),
        "position_y": ("INT", {"default": 0, "min": -4096, "max": 4096}),
        "rotation_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
        "rotation_options": (list(ROTATION_OPTIONS),),
    }


def _effect_inputs(default_fill_mode="solid", include_image=False):
    fill_modes = list(FILL_MODES)
    if include_image:
        fill_modes.append("image")
    fill_modes.remove(default_fill_mode)
    fill_modes.insert(0, default_fill_mode)
    return {
        "font_color_hex": ("STRING", {"default": "#ffffff"}),
        "effect_preset": (list(EFFECT_PRESETS),),
        "fill_mode": (fill_modes,),
        "gradient_color": (list(COLOR_NAMES),),
        "gradient_color_hex": ("STRING", {"default": "#0066ff"}),
        "gradient_angle": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 1.0}),
        "text_opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "outline_width": ("INT", {"default": 0, "min": 0, "max": 64}),
        "outline_color": (list(COLOR_NAMES),),
        "outline_color_hex": ("STRING", {"default": "#000000"}),
        "outer_outline2_width": ("INT", {"default": 0, "min": 0, "max": 64}),
        "outer_outline2_color": (list(COLOR_NAMES),),
        "outer_outline2_color_hex": ("STRING", {"default": "#ffffff"}),
        "outline_gap": ("INT", {"default": 0, "min": 0, "max": 64}),
        "inner_outline_width": ("INT", {"default": 0, "min": 0, "max": 64}),
        "inner_outline_color": (list(COLOR_NAMES),),
        "inner_outline_color_hex": ("STRING", {"default": "#000000"}),
        "highlight_offset_x": ("INT", {"default": -2, "min": -64, "max": 64}),
        "highlight_offset_y": ("INT", {"default": -2, "min": -64, "max": 64}),
        "highlight_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "glow_radius": ("INT", {"default": 0, "min": 0, "max": 128}),
        "glow_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
        "glow_color": (list(COLOR_NAMES),),
        "glow_color_hex": ("STRING", {"default": "#00ffff"}),
        "shadow_offset_x": ("INT", {"default": 0, "min": -256, "max": 256}),
        "shadow_offset_y": ("INT", {"default": 0, "min": -256, "max": 256}),
        "shadow_blur": ("INT", {"default": 0, "min": 0, "max": 128}),
        "shadow_opacity": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
        "shadow_color": (list(COLOR_NAMES),),
        "shadow_color_hex": ("STRING", {"default": "#000000"}),
        "emboss_depth": ("INT", {"default": 0, "min": 0, "max": 32}),
        "extrude_depth": ("INT", {"default": 0, "min": 0, "max": 128}),
        "extrude_offset_x": ("INT", {"default": 1, "min": -16, "max": 16}),
        "extrude_offset_y": ("INT", {"default": 1, "min": -16, "max": 16}),
        "extrude_color": (list(COLOR_NAMES),),
        "extrude_color_hex": ("STRING", {"default": "#808080"}),
        "blend_mode": (list(BLEND_MODES),),
    }


def _render_mask(size, text, font_name, font_size, layout):
    return render_text_mask(
        size,
        text,
        font_name,
        font_size,
        align=layout["align"],
        justify=layout["justify"],
        margins=layout["margins"],
        line_spacing=layout["line_spacing"],
        letter_spacing=layout["letter_spacing"],
        position_x=layout["position_x"],
        position_y=layout["position_y"],
        rotation_angle=layout["rotation_angle"],
        rotation_options=layout["rotation_options"],
    )


def _layout_values(align, justify, margins, line_spacing, letter_spacing, position_x, position_y, rotation_angle, rotation_options):
    return locals()


def _help(node_name):
    return f"TUT_Nodes/图片/文本/{node_name}"


class TUT_OverlayText:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "text": ("STRING", {"multiline": True, "default": "text"}),
            "font_name": _font_input(),
            "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
            "font_color": (list(COLOR_NAMES),),
            **_layout_inputs(),
        }
        return {"required": required, "optional": _effect_inputs()}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "text_mask")
    FUNCTION = "overlay_text"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "在 IMAGE 批次上绘制文字，并输出可复用的文字遮罩。"

    def overlay_text(
        self, image, text, font_name, font_size, font_color, align, justify, margins,
        line_spacing, letter_spacing, position_x, position_y, rotation_angle,
        rotation_options, **effects,
    ):
        frames = image_tensor_to_pil_batch(image)
        layout = _layout_values(align, justify, margins, line_spacing, letter_spacing, position_x, position_y, rotation_angle, rotation_options)
        masks = [_render_mask(frame.size, text, font_name, font_size, layout) for frame in frames]
        outputs = [compose_text_effect(frame, mask, font_color=font_color, **effects) for frame, mask in zip(frames, masks)]
        return (pil_batch_to_image_tensor(outputs), _help("TUT_OverlayText"), pil_batch_to_mask_tensor(masks))


class TUT_DrawText:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image_width": ("INT", {"default": 512, "min": 64, "max": 8192}),
            "image_height": ("INT", {"default": 512, "min": 64, "max": 8192}),
            "text": ("STRING", {"multiline": True, "default": "text"}),
            "font_name": _font_input(),
            "font_size": ("INT", {"default": 50, "min": 1, "max": 2048}),
            "font_color": (list(COLOR_NAMES),),
            "background_color": (list(COLOR_NAMES),),
            **_layout_inputs(),
        }
        optional = _effect_inputs()
        optional["background_color_hex"] = ("STRING", {"default": "#000000"})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "text_mask")
    FUNCTION = "draw_text"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "在纯色画布上绘制带静态特效的文字。"

    def draw_text(
        self, image_width, image_height, text, font_name, font_size, font_color,
        background_color, align, justify, margins, line_spacing, letter_spacing,
        position_x, position_y, rotation_angle, rotation_options,
        background_color_hex="#000000", **effects,
    ):
        size = (int(image_width), int(image_height))
        background = Image.new("RGB", size, parse_color(background_color, background_color_hex))
        layout = _layout_values(align, justify, margins, line_spacing, letter_spacing, position_x, position_y, rotation_angle, rotation_options)
        mask = _render_mask(size, text, font_name, font_size, layout)
        output = compose_text_effect(background, mask, font_color=font_color, **effects)
        return (pil_batch_to_image_tensor([output]), _help("TUT_DrawText"), pil_batch_to_mask_tensor([mask]))


class TUT_MaskText:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "text": ("STRING", {"multiline": True, "default": "text"}),
            "font_name": _font_input(),
            "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
            "background_color": (list(COLOR_NAMES),),
            **_layout_inputs(),
        }
        optional = _effect_inputs("image", include_image=True)
        optional["background_color_hex"] = ("STRING", {"default": "#000000"})
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "text_mask")
    FUNCTION = "mask_text"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "以输入图像作为文字纹理，背景使用指定颜色。"

    def mask_text(
        self, image, text, font_name, font_size, background_color, align, justify,
        margins, line_spacing, letter_spacing, position_x, position_y,
        rotation_angle, rotation_options, background_color_hex="#000000", **effects,
    ):
        frames = image_tensor_to_pil_batch(image)
        layout = _layout_values(align, justify, margins, line_spacing, letter_spacing, position_x, position_y, rotation_angle, rotation_options)
        background_rgb = parse_color(background_color, background_color_hex)
        effects.setdefault("fill_mode", "image")
        masks = [_render_mask(frame.size, text, font_name, font_size, layout) for frame in frames]
        outputs = [
            compose_text_effect(Image.new("RGB", frame.size, background_rgb), mask, fill_image=frame, **effects)
            for frame, mask in zip(frames, masks)
        ]
        return (pil_batch_to_image_tensor(outputs), _help("TUT_MaskText"), pil_batch_to_mask_tensor(masks))


class TUT_CompositeText:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image_text": ("IMAGE",),
            "image_background": ("IMAGE",),
            "text": ("STRING", {"multiline": True, "default": "text"}),
            "font_name": _font_input(),
            "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
            **_layout_inputs(),
        }
        return {"required": required, "optional": _effect_inputs("image", include_image=True)}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "text_mask")
    FUNCTION = "composite_text"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "用文字遮罩将前景图像合成到背景图像。"

    def composite_text(
        self, image_text, image_background, text, font_name, font_size, align,
        justify, margins, line_spacing, letter_spacing, position_x, position_y,
        rotation_angle, rotation_options, **effects,
    ):
        foregrounds = image_tensor_to_pil_batch(image_text)
        backgrounds = image_tensor_to_pil_batch(image_background)
        effects.setdefault("fill_mode", "image")
        foregrounds, backgrounds = broadcast_batches(foregrounds, backgrounds)
        layout = _layout_values(align, justify, margins, line_spacing, letter_spacing, position_x, position_y, rotation_angle, rotation_options)
        masks = [_render_mask(background.size, text, font_name, font_size, layout) for background in backgrounds]
        outputs = [
            compose_text_effect(background, mask, fill_image=foreground, **effects)
            for foreground, background, mask in zip(foregrounds, backgrounds, masks)
        ]
        return (pil_batch_to_image_tensor(outputs), _help("TUT_CompositeText"), pil_batch_to_mask_tensor(masks))


class TUT_SimpleTextWatermark:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "text": ("STRING", {"multiline": False, "default": "@ your name"}),
            "align": (list(WATERMARK_ALIGNMENTS),),
            "opacity": ("FLOAT", {"default": 0.30, "min": 0.0, "max": 1.0, "step": 0.01}),
            "font_name": _font_input(),
            "font_size": ("INT", {"default": 50, "min": 1, "max": 1024}),
            "font_color": (list(COLOR_NAMES),),
            "x_margin": ("INT", {"default": 20, "min": -1024, "max": 1024}),
            "y_margin": ("INT", {"default": 20, "min": -1024, "max": 1024}),
        }
        optional = _effect_inputs()
        optional.pop("text_opacity")
        return {"required": required, "optional": optional}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "text_mask")
    FUNCTION = "watermark"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "为 IMAGE 批次添加可定位、可调透明度和特效的文字水印。"

    def watermark(self, image, text, align, opacity, font_name, font_size, font_color, x_margin, y_margin, **effects):
        frames = image_tensor_to_pil_batch(image)
        vertical = "center" if align == "center" else ("top" if align.startswith("top") else "bottom")
        horizontal = "center" if align == "center" or align.endswith("center") else ("left" if align.endswith("left") else "right")
        position_x = int(x_margin) if horizontal == "left" else (-int(x_margin) if horizontal == "right" else 0)
        position_y = int(y_margin) if vertical == "top" else (-int(y_margin) if vertical == "bottom" else 0)
        layout = _layout_values(vertical, horizontal, 0, 0, 0, position_x, position_y, 0.0, "text center")
        masks = [_render_mask(frame.size, text, font_name, font_size, layout) for frame in frames]
        effects["layer_opacity"] = opacity
        outputs = [compose_text_effect(frame, mask, font_color=font_color, **effects) for frame, mask in zip(frames, masks)]
        return (pil_batch_to_image_tensor(outputs), _help("TUT_SimpleTextWatermark"), pil_batch_to_mask_tensor(masks))


class TUT_AutoContrastWatermark:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "text": ("STRING", {"multiline": False, "default": "@ your name"}),
                "position": (list(CORNER_POSITIONS),),
                "font_name": _font_input(),
                "size_percent": ("FLOAT", {"default": 4.0, "min": 0.5, "max": 20.0, "step": 0.1}),
                "max_width_percent": ("FLOAT", {"default": 40.0, "min": 5.0, "max": 95.0, "step": 1.0}),
                "x_margin_percent": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "y_margin_percent": ("FLOAT", {"default": 2.0, "min": 0.0, "max": 20.0, "step": 0.1}),
                "opacity": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "text_mask")
    FUNCTION = "watermark"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "在四角添加随图片缩放的文字水印，并用文字遮罩显示底图的逐像素反色。"

    def watermark(
        self, image, text, position, font_name, size_percent,
        max_width_percent, x_margin_percent, y_margin_percent, opacity,
    ):
        if not str(text) or float(opacity) <= 0.0:
            if not isinstance(image, torch.Tensor) or image.ndim != 4:
                raise ValueError("IMAGE 必须是批次 [B, H, W, C]")
            unchanged = image if image.dtype == torch.float32 else image.float()
            empty_mask = torch.zeros(
                (image.shape[0], image.shape[1], image.shape[2]),
                dtype=torch.float32,
                device=image.device,
            )
            return (unchanged, _help("TUT_AutoContrastWatermark"), empty_mask)
        frames = image_tensor_to_pil_batch(image)
        rendered = [
            render_adaptive_watermark(
                frame, text, font_name, position, size_percent,
                max_width_percent, x_margin_percent, y_margin_percent, opacity,
            )
            for frame in frames
        ]
        outputs = [item[0] for item in rendered]
        masks = [item[1] for item in rendered]
        return (
            pil_batch_to_image_tensor(outputs),
            _help("TUT_AutoContrastWatermark"),
            pil_batch_to_mask_tensor(masks),
        )


class TUT_SelectFont:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"font_name": _font_input()}}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("font_name", "show_help")
    FUNCTION = "select_font"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "从插件字体和系统字体中选择字体。"

    def select_font(self, font_name):
        return (font_name, _help("TUT_SelectFont"))


class TUT_TextEffect:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "background": ("IMAGE",),
            "text_mask": ("MASK",),
            "font_color": (list(COLOR_NAMES),),
        }
        return {"required": required, "optional": _effect_inputs()}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "text_mask")
    FUNCTION = "apply_effect"
    CATEGORY = IMAGE_TEXT
    DESCRIPTION = "根据文字 MASK 独立应用渐变、描边、发光、阴影、浮雕和立体效果。"

    def apply_effect(self, background, text_mask, font_color, **effects):
        backgrounds = image_tensor_to_pil_batch(background)
        masks = mask_tensor_to_pil_batch(text_mask)
        backgrounds, masks = broadcast_batches(backgrounds, masks)
        resized_masks = [
            mask if mask.size == frame.size else mask.resize(frame.size, getattr(Image, "Resampling", Image).BILINEAR)
            for frame, mask in zip(backgrounds, masks)
        ]
        outputs = [compose_text_effect(frame, mask, font_color=font_color, **effects) for frame, mask in zip(backgrounds, resized_masks)]
        return (pil_batch_to_image_tensor(outputs), _help("TUT_TextEffect"), pil_batch_to_mask_tensor(resized_masks))


NODE_CLASS_MAPPINGS = {
    "TUT_OverlayText": TUT_OverlayText,
    "TUT_DrawText": TUT_DrawText,
    "TUT_MaskText": TUT_MaskText,
    "TUT_CompositeText": TUT_CompositeText,
    "TUT_SimpleTextWatermark": TUT_SimpleTextWatermark,
    "TUT_AutoContrastWatermark": TUT_AutoContrastWatermark,
    "TUT_SelectFont": TUT_SelectFont,
    "TUT_TextEffect": TUT_TextEffect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_OverlayText": "TUT_叠加文字",
    "TUT_DrawText": "TUT_绘制文字",
    "TUT_MaskText": "TUT_文字遮罩填充",
    "TUT_CompositeText": "TUT_文字图像合成",
    "TUT_SimpleTextWatermark": "TUT_文字水印",
    "TUT_AutoContrastWatermark": "TUT_自适应反色水印",
    "TUT_SelectFont": "TUT_选择字体",
    "TUT_TextEffect": "TUT_文字特效",
}
