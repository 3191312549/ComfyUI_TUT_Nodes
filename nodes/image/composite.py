"""Layer-style two-image compositing node."""

from __future__ import annotations

import numpy as np

from ...categories import IMAGE_COMPOSITE
from ...core.compositing import composite_soft_layer
from ...core.imaging import (
    COLOR_NAMES,
    broadcast_batches,
    image_tensor_to_pil_batch,
    mask_tensor_to_pil_batch,
    parse_color,
    pil_batch_to_image_tensor,
)


PRESETS = (
    "自定义", "自然悬浮", "柔边照片", "海报卡片", "白边贴纸", "撕纸拼贴",
    "烧焦照片", "玻璃面板", "霓虹窗口", "墨水渗透", "像素崩解",
    "噪声消散", "厚卡片",
)


class TUT_柔边图层合成:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_layer": ("IMAGE",),
                "image_background": ("IMAGE",),
                "preset": (list(PRESETS), {"default": "自然悬浮"}),
                "size_mode": (["适应背景", "原始尺寸"], {"default": "适应背景"}),
                "scale": ("FLOAT", {"default": 0.65, "min": 0.01, "max": 4.0, "step": 0.01}),
                "position_x": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 2.0, "step": 0.01}),
                "position_y": ("FLOAT", {"default": 0.5, "min": -1.0, "max": 2.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "tilt_x": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 0.5}),
                "tilt_y": ("FLOAT", {"default": 0.0, "min": -60.0, "max": 60.0, "step": 0.5}),
                "opacity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "blend_mode": (["normal", "multiply", "screen", "overlay"], {"default": "normal"}),
                "shape_mode": (["关闭", "圆角", "切角", "波浪", "撕裂"], {"default": "圆角"}),
                "shape_amount": ("INT", {"default": 32, "min": 0, "max": 1024}),
                "shape_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "transition_mode": (["关闭", "羽化", "噪声溶解", "像素崩解", "墨水扩散"], {"default": "羽化"}),
                "transition_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "material_mode": (["关闭", "白边贴纸", "纸张纤维", "烧焦", "玻璃切边"], {"default": "关闭"}),
                "material_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "depth_mode": (["关闭", "柔投影", "斜面浮雕", "霓虹边光", "伪厚度"], {"default": "柔投影"}),
                "depth_strength": ("FLOAT", {"default": 0.55, "min": 0.0, "max": 1.0, "step": 0.01}),
                "edge_width": ("INT", {"default": 12, "min": 1, "max": 128}),
                "edge_color": (list(COLOR_NAMES), {"default": "white"}),
                "edge_color_hex": ("STRING", {"default": "#FFFFFF"}),
                "detail_scale": ("FLOAT", {"default": 16.0, "min": 2.0, "max": 256.0, "step": 1.0}),
                "irregularity": ("FLOAT", {"default": 0.65, "min": 0.0, "max": 1.0, "step": 0.01}),
                "background_wrap": ("FLOAT", {"default": 0.15, "min": 0.0, "max": 1.0, "step": 0.01}),
                "background_blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 64.0, "step": 0.5}),
                "depth_offset_x": ("INT", {"default": 12, "min": -256, "max": 256}),
                "depth_offset_y": ("INT", {"default": 12, "min": -256, "max": 256}),
                "shadow_blur": ("FLOAT", {"default": 14.0, "min": 0.0, "max": 128.0, "step": 0.5}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
            },
            "optional": {"layer_mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "composite_layer"
    CATEGORY = IMAGE_COMPOSITE
    DESCRIPTION = "将图一作为可变换图层悬浮合成到图二，并组合形状、消散、材质和光影边缘。"

    def composite_layer(
        self, image_layer, image_background, preset, size_mode, scale,
        position_x, position_y, rotation, tilt_x, tilt_y, opacity, blend_mode,
        shape_mode, shape_amount, shape_strength, transition_mode,
        transition_strength, material_mode, material_strength, depth_mode,
        depth_strength, edge_width, edge_color, edge_color_hex, detail_scale,
        irregularity, background_wrap, background_blur, depth_offset_x,
        depth_offset_y, shadow_blur, seed, layer_mask=None,
    ):
        layers = image_tensor_to_pil_batch(image_layer)
        backgrounds = image_tensor_to_pil_batch(image_background)
        masks = mask_tensor_to_pil_batch(layer_mask) if layer_mask is not None else [None]
        layers, backgrounds, masks = broadcast_batches(layers, backgrounds, masks)
        color = parse_color(edge_color, edge_color_hex, default=(255, 255, 255))

        rendered, active = [], []
        for batch_index, (layer, background, mask) in enumerate(zip(layers, backgrounds, masks)):
            output, is_active = composite_soft_layer(
                layer, background, mask,
                size_mode=size_mode, scale=scale, position_x=position_x,
                position_y=position_y, rotation=rotation, tilt_x=tilt_x,
                tilt_y=tilt_y, opacity=opacity, blend_mode=blend_mode,
                shape_mode=shape_mode, shape_amount=shape_amount,
                shape_strength=shape_strength, transition_mode=transition_mode,
                transition_strength=transition_strength, material_mode=material_mode,
                material_strength=material_strength, depth_mode=depth_mode,
                depth_strength=depth_strength, edge_width=edge_width,
                edge_color=color, detail_scale=detail_scale,
                irregularity=irregularity, background_wrap=background_wrap,
                background_blur=background_blur, depth_offset_x=depth_offset_x,
                depth_offset_y=depth_offset_y, shadow_blur=shadow_blur,
                rng=np.random.default_rng((int(seed) + batch_index) & 0xFFFFFFFFFFFFFFFF),
            )
            rendered.append(output)
            active.append(is_active)

        tensor = pil_batch_to_image_tensor(rendered)
        background_count = int(image_background.shape[0])
        for index, is_active in enumerate(active):
            if not is_active:
                source_index = index if background_count == len(rendered) else 0
                tensor[index] = image_background[source_index]
        return (tensor,)


NODE_CLASS_MAPPINGS = {"TUT_SoftLayerComposite": TUT_柔边图层合成}
NODE_DISPLAY_NAME_MAPPINGS = {"TUT_SoftLayerComposite": "TUT_柔边图层合成"}
