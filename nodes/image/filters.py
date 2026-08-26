"""Static image-filter node wrappers for TUT_Nodes."""

from __future__ import annotations

from ...categories import IMAGE_FILTER
from ...core.filters import (
    comic_filter,
    composite_effect,
    effect_mask_tensor,
    glass_refraction,
    glitch_art,
    image_batch_tensor,
    kaleidoscope,
    palette_preview,
    pixel_art,
    prepare_filter_batches,
    retro_print,
    rng_for,
)


HELP = "TUT_Nodes/图片/滤镜"


def _optional_mask():
    return {"mask": ("MASK",)}


def _output_tensor(original, rendered, strength, masks=None):
    """Preserve exact input values when an effect is disabled."""
    batch_size = len(rendered)
    if float(strength) != 0.0:
        result = image_batch_tensor(rendered)
        if masks is not None:
            for index, region in enumerate(masks):
                if region.getbbox() is None:
                    source_index = index if int(original.shape[0]) == batch_size else 0
                    result[index] = original[source_index]
        return result
    if int(original.shape[0]) == batch_size:
        return original
    if int(original.shape[0]) == 1:
        return original.repeat(batch_size, 1, 1, 1)
    raise ValueError("IMAGE 批次无法与滤镜输出批次对应")


class TUT_RetroPrintFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["自定义", "报纸", "双色海报", "Risograph", "CMYK旧印刷"],),
                "color_mode": (["单色", "双色", "三色", "CMYK"], {"default": "三色"}),
                "palette_text": ("STRING", {"default": "#19181C,#E2314D,#1D91C0,#F4D64C"}),
                "dot_size": ("INT", {"default": 6, "min": 2, "max": 128}),
                "screen_angle": ("FLOAT", {"default": 15.0, "min": -180.0, "max": 180.0}),
                "registration_shift": ("FLOAT", {"default": 1.5, "min": -32.0, "max": 32.0, "step": 0.1}),
                "ink_bleed": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.5}),
                "paper_grain": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "MASK")
    RETURN_NAMES = ("image", "show_help", "effect_mask", "ink_texture")
    FUNCTION = "apply_filter"
    CATEGORY = IMAGE_FILTER
    DESCRIPTION = "模拟分色网点、套印偏移、油墨扩散和纸张纹理。"

    def apply_filter(self, image, preset, color_mode, palette_text, dot_size, screen_angle,
                     registration_shift, ink_bleed, paper_grain, seed, strength, mask=None):
        frames, masks = prepare_filter_batches(image, mask)
        output, textures = [], []
        for index, (frame, region) in enumerate(zip(frames, masks)):
            effect, texture = retro_print(
                frame, color_mode, palette_text, dot_size, screen_angle,
                registration_shift, ink_bleed, paper_grain, rng_for(seed, index),
            )
            output.append(composite_effect(frame, effect, region, strength))
            textures.append(texture)
        return _output_tensor(image, output, strength, masks), HELP, effect_mask_tensor(masks), effect_mask_tensor(textures)


class TUT_ComicFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["自定义", "日漫黑白", "彩色漫画", "美漫", "波普"],),
                "color_levels": ("INT", {"default": 5, "min": 2, "max": 32}),
                "line_width": ("INT", {"default": 2, "min": 1, "max": 16}),
                "line_strength": ("FLOAT", {"default": 1.5, "min": 0.0, "max": 8.0, "step": 0.05}),
                "line_threshold": ("FLOAT", {"default": 0.18, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_threshold": ("FLOAT", {"default": 0.42, "min": 0.0, "max": 1.0, "step": 0.01}),
                "shadow_halftone": ("BOOLEAN", {"default": True}),
                "dot_size": ("INT", {"default": 6, "min": 2, "max": 64}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("image", "show_help", "effect_mask", "line_mask", "shadow_mask")
    FUNCTION = "apply_filter"
    CATEGORY = IMAGE_FILTER
    DESCRIPTION = "组合色彩分层、线稿和阴影网点，并输出可复用 MASK。"

    def apply_filter(self, image, preset, color_levels, line_width, line_strength, line_threshold,
                     shadow_threshold, shadow_halftone, dot_size, strength, mask=None):
        frames, masks = prepare_filter_batches(image, mask)
        output, lines, shadows = [], [], []
        for frame, region in zip(frames, masks):
            effect, line, shadow = comic_filter(
                frame, color_levels, line_width, line_strength, line_threshold,
                shadow_threshold, shadow_halftone, dot_size,
            )
            output.append(composite_effect(frame, effect, region, strength))
            lines.append(line)
            shadows.append(shadow)
        return (
            _output_tensor(image, output, strength, masks), HELP, effect_mask_tensor(masks),
            effect_mask_tensor(lines), effect_mask_tensor(shadows),
        )


class TUT_KaleidoscopeFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "segments": ("INT", {"default": 8, "min": 2, "max": 24}),
                "center_x": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "center_y": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation": ("FLOAT", {"default": 0.0, "min": -360.0, "max": 360.0, "step": 0.1}),
                "zoom": ("FLOAT", {"default": 1.0, "min": 0.05, "max": 8.0, "step": 0.05}),
                "mirror": ("BOOLEAN", {"default": True}),
                "seam_softness": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 3.0, "step": 0.1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "effect_mask")
    FUNCTION = "apply_filter"
    CATEGORY = IMAGE_FILTER
    DESCRIPTION = "使用极坐标扇区镜像生成可调中心的万花筒效果。"

    def apply_filter(self, image, segments, center_x, center_y, rotation, zoom,
                     mirror, seam_softness, strength, mask=None):
        frames, masks = prepare_filter_batches(image, mask)
        output = []
        for frame, region in zip(frames, masks):
            effect = kaleidoscope(frame, segments, center_x, center_y, rotation, zoom, mirror, seam_softness)
            output.append(composite_effect(frame, effect, region, strength))
        return _output_tensor(image, output, strength, masks), HELP, effect_mask_tensor(masks)


class TUT_PixelArtFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["自定义", "Game Boy", "NES风格", "16-bit", "街机"],),
                "pixel_size": ("INT", {"default": 8, "min": 1, "max": 256}),
                "max_colors": ("INT", {"default": 16, "min": 2, "max": 256}),
                "palette_mode": (["自动", "Game Boy", "NES风格", "16-bit", "街机", "自定义"], {"default": "自动"}),
                "custom_palette": ("STRING", {"default": "#0F0F1C,#306082,#E6413A,#F7D65A"}),
                "dither": (["无", "Floyd-Steinberg", "Bayer 2x2", "Bayer 4x4"], {"default": "无"}),
                "outline_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "outline_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "show_help", "effect_mask", "palette_image")
    FUNCTION = "apply_filter"
    CATEGORY = IMAGE_FILTER
    DESCRIPTION = "生成有限色板、抖动和像素描边的完整像素艺术效果。"

    def apply_filter(self, image, preset, pixel_size, max_colors, palette_mode, custom_palette,
                     dither, outline_strength, outline_threshold, strength, mask=None):
        frames, masks = prepare_filter_batches(image, mask)
        output, previews = [], []
        for frame, region in zip(frames, masks):
            effect, colors = pixel_art(
                frame, pixel_size, max_colors, palette_mode, custom_palette,
                dither, outline_strength, outline_threshold,
            )
            output.append(composite_effect(frame, effect, region, strength))
            previews.append(palette_preview(colors))
        return _output_tensor(image, output, strength, masks), HELP, effect_mask_tensor(masks), image_batch_tensor(previews)


class TUT_GlassRefractionFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["自定义", "波纹玻璃", "条纹玻璃", "磨砂玻璃", "液态玻璃", "水滴透镜"],),
                "mode": (["波纹玻璃", "条纹玻璃", "磨砂玻璃", "液态玻璃", "水滴透镜"], {"default": "波纹玻璃"}),
                "amount": ("FLOAT", {"default": 12.0, "min": -128.0, "max": 128.0, "step": 0.5}),
                "scale": ("FLOAT", {"default": 32.0, "min": 2.0, "max": 512.0, "step": 1.0}),
                "angle": ("FLOAT", {"default": 0.0, "min": -180.0, "max": 180.0, "step": 0.1}),
                "blur": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 32.0, "step": 0.1}),
                "chromatic_aberration": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 32.0, "step": 0.1}),
                "roughness": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "show_help", "effect_mask", "displacement_map")
    FUNCTION = "apply_filter"
    CATEGORY = IMAGE_FILTER
    DESCRIPTION = "使用程序化位移场生成玻璃折射、模糊和色差。"

    def apply_filter(self, image, preset, mode, amount, scale, angle, blur, chromatic_aberration,
                     roughness, seed, strength, mask=None):
        frames, masks = prepare_filter_batches(image, mask)
        output, displacement = [], []
        for index, (frame, region) in enumerate(zip(frames, masks)):
            effect, disp = glass_refraction(
                frame, mode, amount, scale, angle, blur, chromatic_aberration,
                roughness, rng_for(seed, index),
            )
            output.append(composite_effect(frame, effect, region, strength))
            displacement.append(disp)
        return _output_tensor(image, output, strength, masks), HELP, effect_mask_tensor(masks), image_batch_tensor(displacement)


class TUT_GlitchArtFilter:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["自定义", "RGB故障", "VHS", "数据损坏", "像素排序"],),
                "mode": (["RGB故障", "VHS", "数据损坏", "像素排序"], {"default": "数据损坏"}),
                "rgb_shift": ("INT", {"default": 8, "min": 0, "max": 256}),
                "block_count": ("INT", {"default": 8, "min": 0, "max": 256}),
                "block_height": ("INT", {"default": 12, "min": 1, "max": 512}),
                "scanline_strength": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "sort_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "noise_strength": ("FLOAT", {"default": 0.08, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "MASK")
    RETURN_NAMES = ("image", "show_help", "effect_mask", "glitch_mask")
    FUNCTION = "apply_filter"
    CATEGORY = IMAGE_FILTER
    DESCRIPTION = "生成确定性数据块、扫描线、像素排序和通道故障。"

    def apply_filter(self, image, preset, mode, rgb_shift, block_count, block_height,
                     scanline_strength, sort_threshold, noise_strength, seed,
                     strength, mask=None):
        frames, masks = prepare_filter_batches(image, mask)
        output, affected = [], []
        for index, (frame, region) in enumerate(zip(frames, masks)):
            effect, glitch_mask = glitch_art(
                frame, mode, rgb_shift, block_count, block_height, scanline_strength,
                sort_threshold, noise_strength, rng_for(seed, index),
            )
            output.append(composite_effect(frame, effect, region, strength))
            affected.append(glitch_mask)
        return _output_tensor(image, output, strength, masks), HELP, effect_mask_tensor(masks), effect_mask_tensor(affected)


NODE_CLASS_MAPPINGS = {
    "TUT_RetroPrintFilter": TUT_RetroPrintFilter,
    "TUT_ComicFilter": TUT_ComicFilter,
    "TUT_KaleidoscopeFilter": TUT_KaleidoscopeFilter,
    "TUT_PixelArtFilter": TUT_PixelArtFilter,
    "TUT_GlassRefractionFilter": TUT_GlassRefractionFilter,
    "TUT_GlitchArtFilter": TUT_GlitchArtFilter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_RetroPrintFilter": "TUT_复古印刷滤镜",
    "TUT_ComicFilter": "TUT_漫画化滤镜",
    "TUT_KaleidoscopeFilter": "TUT_万花筒滤镜",
    "TUT_PixelArtFilter": "TUT_像素艺术滤镜",
    "TUT_GlassRefractionFilter": "TUT_玻璃折射滤镜",
    "TUT_GlitchArtFilter": "TUT_故障艺术滤镜",
}
