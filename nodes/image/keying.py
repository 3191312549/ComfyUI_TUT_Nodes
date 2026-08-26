"""Color, rembg AI, and external-SAM-mask keying nodes."""

from __future__ import annotations

from ...categories import IMAGE_KEYING
from ...core.imaging import parse_color
from ...core.keying import (
    KEY_COLOR_MAP,
    REMBG_MODELS,
    color_key_mask,
    foreground_outputs,
    image_tensor,
    refine_mask,
    rembg_masks,
)


OUTPUT_TYPES = ("IMAGE", "STRING", "MASK", "MASK")
OUTPUT_NAMES = ("foreground", "show_help", "foreground_mask", "background_mask")


def _refine_inputs():
    return {
        "grow_shrink": ("INT", {"default": 0, "min": -32, "max": 32, "step": 1}),
        "edge_feather": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.1}),
        "invert_mask": ("BOOLEAN", {"default": False}),
    }


class TUT_ColorKeying:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "key_color": (["green", "blue", "white", "black", "custom"], {"default": "green"}),
            "custom_color": ("STRING", {"default": "#00ff00"}),
            "color_tolerance": ("FLOAT", {"default": 20.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "color_softness": ("FLOAT", {"default": 15.0, "min": 0.0, "max": 100.0, "step": 0.1}),
            "spill_suppression": ("FLOAT", {"default": 0.6, "min": 0.0, "max": 1.0, "step": 0.01}),
        }
        required.update(_refine_inputs())
        return {"required": required}

    RETURN_TYPES = OUTPUT_TYPES
    RETURN_NAMES = OUTPUT_NAMES
    FUNCTION = "key"
    CATEGORY = IMAGE_KEYING
    DESCRIPTION = "按 Lab 感知色差移除绿幕、蓝幕、白底、黑底或自定义颜色。"

    def key(self, image, key_color, custom_color, color_tolerance, color_softness,
            spill_suppression, grow_shrink, edge_feather, invert_mask):
        key_rgb = KEY_COLOR_MAP.get(str(key_color))
        if key_rgb is None:
            key_rgb = parse_color("custom", custom_color)
        images = image_tensor(image)
        masks = color_key_mask(images, key_rgb, color_tolerance, color_softness)
        masks = refine_mask(masks, grow_shrink, edge_feather, invert_mask)
        foreground, foreground_mask, background_mask = foreground_outputs(
            images, masks, spill_color=key_rgb, spill_strength=spill_suppression
        )
        return foreground, "TUT_ColorKeying：Lab 色差颜色抠像", foreground_mask, background_mask


class TUT_AIKeying:
    @classmethod
    def INPUT_TYPES(cls):
        required = {
            "image": ("IMAGE",),
            "model": (list(REMBG_MODELS), {"default": "birefnet-general-lite"}),
            "provider": (["auto", "cuda", "cpu"], {"default": "auto"}),
        }
        required.update(_refine_inputs())
        return {"required": required}

    RETURN_TYPES = OUTPUT_TYPES
    RETURN_NAMES = OUTPUT_NAMES
    FUNCTION = "key"
    CATEGORY = IMAGE_KEYING
    DESCRIPTION = "使用 rembg 按需加载 AI 模型自动生成前景遮罩。首次使用模型时可能需要下载。"

    def key(self, image, model, provider, grow_shrink, edge_feather, invert_mask):
        images = image_tensor(image)
        masks = rembg_masks(images, model, provider)
        masks = refine_mask(masks, grow_shrink, edge_feather, invert_mask)
        foreground, foreground_mask, background_mask = foreground_outputs(images, masks)
        return foreground, f"TUT_AIKeying：rembg / {model} / {provider}", foreground_mask, background_mask


class TUT_SAMMaskKeying:
    @classmethod
    def INPUT_TYPES(cls):
        required = {"image": ("IMAGE",), "sam_mask": ("MASK",)}
        required.update(_refine_inputs())
        return {"required": required}

    RETURN_TYPES = OUTPUT_TYPES
    RETURN_NAMES = OUTPUT_NAMES
    FUNCTION = "key"
    CATEGORY = IMAGE_KEYING
    DESCRIPTION = "接收任意 SAM1、SAM2 或 SAM3 节点输出的 MASK，并生成抠像前景。"

    def key(self, image, sam_mask, grow_shrink, edge_feather, invert_mask):
        masks = refine_mask(sam_mask, grow_shrink, edge_feather, invert_mask)
        foreground, foreground_mask, background_mask = foreground_outputs(image, masks)
        return foreground, "TUT_SAMMaskKeying：外部 SAM MASK 抠像", foreground_mask, background_mask


NODE_CLASS_MAPPINGS = {
    "TUT_ColorKeying": TUT_ColorKeying,
    "TUT_AIKeying": TUT_AIKeying,
    "TUT_SAMMaskKeying": TUT_SAMMaskKeying,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_ColorKeying": "TUT_颜色抠像",
    "TUT_AIKeying": "TUT_AI智能抠像",
    "TUT_SAMMaskKeying": "TUT_SAM遮罩抠像",
}
