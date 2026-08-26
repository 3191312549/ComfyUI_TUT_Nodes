"""Professional color matching and film-look nodes."""

from __future__ import annotations

import numpy as np
import torch

from ...categories import IMAGE_COLOR
from ...core.color import (
    ADVANCED_WHITE_BALANCE_METHODS,
    advanced_auto_color_correct,
    auto_color_correct,
    basic_color,
    basic_tone,
    color_compress,
    color_match,
    detail_enhance,
    exact_or_composite,
    film_tone,
    halation,
    lens_diffusion,
    palette_preview,
    prepare_analysis_effect,
    parse_hex,
    prepare_pair,
    prepare_single,
    selective_hsl,
    tensors,
)


HELP = "TUT_Nodes/图片/调色"


def _optional_mask(name="mask"):
    return {name: ("MASK",)}


def _slider(default, minimum, maximum, step):
    return {
        "default": default,
        "min": minimum,
        "max": maximum,
        "step": step,
        "display": "slider",
    }


def _effect_output(original, rendered, regions, strength):
    return exact_or_composite(original, np.stack(rendered), np.stack(regions), strength)


def _graded_result(original, frames, rendered, regions, strength):
    output = _effect_output(original, rendered, regions, strength)
    output_np = output.detach().cpu().numpy()
    effect = np.max(np.abs(output_np - np.asarray(frames, dtype=np.float32)), axis=-1)
    _, effect_mask = tensors(output_np, (effect,))
    return output, HELP, effect_mask


class TUT_AutoColorCorrect:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "auto_white_balance": ("BOOLEAN", {"default": True}),
                "auto_exposure": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", _slider(1.0, 0.0, 1.0, 0.01)),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "effect_mask")
    FUNCTION = "correct"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "自动分析有效区域，稳健修正白平衡和曝光，并保护极暗与过曝像素。"

    def correct(self, image, auto_white_balance, auto_exposure, strength, mask=None):
        frames, masks = prepare_single(image, mask)
        rendered = [auto_color_correct(frame, region, auto_white_balance, auto_exposure)
                    for frame, region in zip(frames, masks)]
        return _graded_result(image, frames, rendered, masks, strength)


def _advanced_diagnostic_report(index, diagnostics, auto_white_balance, auto_exposure):
    if diagnostics["status"] != "已完成":
        return f"第 {index + 1} 张：{diagnostics['status']}，已保持原图。"
    raw = diagnostics["raw_gains"]
    effective = diagnostics["effective_gains"]
    confidence = diagnostics["confidence"]
    factor = diagnostics["confidence_factor"]
    if not auto_white_balance:
        confidence_note = "白平衡已关闭"
    elif factor < 1e-6:
        confidence_note = "置信度不足，白平衡已跳过"
    elif factor < 0.999:
        confidence_note = f"低置信度自动减弱至 {factor:.1%}"
    else:
        confidence_note = "白平衡完整应用"
    exposure_note = (
        f"目标/实际曝光 {diagnostics['target_exposure_ev']:+.2f}/"
        f"{diagnostics['effective_exposure_ev']:+.2f} EV"
        if auto_exposure else "自动曝光已关闭"
    )
    return (
        f"第 {index + 1} 张：置信度 {confidence:.1%}（{confidence_note}）；"
        f"有效样本 {diagnostics['valid_ratio']:.1%}，中性样本 {diagnostics['neutral_ratio']:.1%}；"
        f"原始 RGB 增益 {raw[0]:.3f}/{raw[1]:.3f}/{raw[2]:.3f}，"
        f"实际增益 {effective[0]:.3f}/{effective[1]:.3f}/{effective[2]:.3f}；"
        f"{exposure_note}。"
    )


class TUT_AutoColorCorrectAdvanced:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "auto_white_balance": ("BOOLEAN", {"default": True}),
                "white_balance_method": (list(ADVANCED_WHITE_BALANCE_METHODS), {"default": "自适应融合"}),
                "neutral_strictness": ("FLOAT", _slider(0.65, 0.0, 1.0, 0.01)),
                "white_balance_strength": ("FLOAT", _slider(1.0, 0.0, 1.0, 0.01)),
                "max_white_balance_ev": ("FLOAT", _slider(0.75, 0.0, 2.0, 0.05)),
                "auto_exposure": ("BOOLEAN", {"default": True}),
                "target_midtone": ("FLOAT", _slider(0.18, 0.08, 0.30, 0.01)),
                "max_exposure_ev": ("FLOAT", _slider(1.5, 0.0, 3.0, 0.05)),
                "highlight_protection": ("FLOAT", _slider(0.85, 0.0, 1.0, 0.01)),
                "confidence_threshold": ("FLOAT", _slider(0.45, 0.0, 1.0, 0.01)),
                "strength": ("FLOAT", _slider(1.0, 0.0, 1.0, 0.01)),
            },
            "optional": {
                "analysis_mask": ("MASK",),
                "effect_mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "show_help", "effect_mask", "analysis_mask", "diagnostic_report")
    FUNCTION = "correct_advanced"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "通过中性像素筛选、多估计器融合和置信度衰减执行更稳健的全局白平衡与曝光校正。"

    def correct_advanced(
        self, image, auto_white_balance, white_balance_method, neutral_strictness,
        white_balance_strength, max_white_balance_ev, auto_exposure, target_midtone,
        max_exposure_ev, highlight_protection, confidence_threshold, strength,
        analysis_mask=None, effect_mask=None,
    ):
        frames, analysis_regions, effect_regions = prepare_analysis_effect(
            image, analysis_mask, effect_mask
        )
        corrected, samples, diagnostics = [], [], []
        for frame, region in zip(frames, analysis_regions):
            rendered, sample, diagnostic = advanced_auto_color_correct(
                frame, region, auto_white_balance, white_balance_method,
                neutral_strictness, white_balance_strength, max_white_balance_ev,
                auto_exposure, target_midtone, max_exposure_ev,
                highlight_protection, confidence_threshold,
            )
            corrected.append(rendered)
            samples.append(sample)
            diagnostics.append(diagnostic)

        rendered = np.stack(corrected)
        output = exact_or_composite(image, rendered, effect_regions, strength)
        output_np = output.detach().cpu().numpy()
        effect = np.max(np.abs(output_np - frames), axis=-1)
        _, effect_tensor, analysis_tensor = tensors(output_np, (effect, np.stack(samples)))
        report = "\n".join(
            _advanced_diagnostic_report(index, diagnostic, auto_white_balance, auto_exposure)
            for index, diagnostic in enumerate(diagnostics)
        )
        return output, HELP, effect_tensor, analysis_tensor, report


class TUT_BasicTone:
    @classmethod
    def INPUT_TYPES(cls):
        control = _slider(0.0, -1.0, 1.0, 0.01)
        return {
            "required": {
                "image": ("IMAGE",),
                "exposure": ("FLOAT", _slider(0.0, -5.0, 5.0, 0.05)),
                "brightness": ("FLOAT", dict(control)),
                "contrast": ("FLOAT", dict(control)),
                "highlights": ("FLOAT", dict(control)),
                "shadows": ("FLOAT", dict(control)),
                "whites": ("FLOAT", dict(control)),
                "blacks": ("FLOAT", dict(control)),
                "light_sensation": ("FLOAT", dict(control)),
                "strength": ("FLOAT", _slider(1.0, 0.0, 1.0, 0.01)),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "effect_mask")
    FUNCTION = "adjust_tone"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "在线性光中调整曝光，再以单调曲线控制亮度、对比、高光、阴影、黑白场和光感。"

    def adjust_tone(self, image, exposure, brightness, contrast, highlights, shadows,
                    whites, blacks, light_sensation, strength, mask=None):
        frames, masks = prepare_single(image, mask)
        rendered = [basic_tone(frame, exposure, brightness, contrast, highlights,
                               shadows, whites, blacks, light_sensation) for frame in frames]
        return _graded_result(image, frames, rendered, masks, strength)


class TUT_BasicColor:
    @classmethod
    def INPUT_TYPES(cls):
        control = _slider(0.0, -1.0, 1.0, 0.01)
        return {
            "required": {
                "image": ("IMAGE",),
                "temperature": ("FLOAT", dict(control)),
                "tint": ("FLOAT", dict(control)),
                "saturation": ("FLOAT", dict(control)),
                "vibrance": ("FLOAT", dict(control)),
                "strength": ("FLOAT", _slider(1.0, 0.0, 1.0, 0.01)),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "effect_mask")
    FUNCTION = "adjust_color"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "调整色温、色调、饱和度与自然饱和度，并降低高饱和颜色和肤色过度增强。"

    def adjust_color(self, image, temperature, tint, saturation, vibrance, strength, mask=None):
        frames, masks = prepare_single(image, mask)
        rendered = [basic_color(frame, temperature, tint, saturation, vibrance) for frame in frames]
        return _graded_result(image, frames, rendered, masks, strength)


class TUT_DetailEnhance:
    @classmethod
    def INPUT_TYPES(cls):
        signed = _slider(0.0, -1.0, 1.0, 0.01)
        unit = _slider(0.0, 0.0, 1.0, 0.01)
        return {
            "required": {
                "image": ("IMAGE",),
                "clarity": ("FLOAT", dict(signed)),
                "texture": ("FLOAT", dict(signed)),
                "dehaze": ("FLOAT", dict(signed)),
                "sharpen": ("FLOAT", _slider(0.0, 0.0, 2.0, 0.01)),
                "luminance_denoise": ("FLOAT", dict(unit)),
                "color_denoise": ("FLOAT", dict(unit)),
                "grain": ("FLOAT", dict(unit)),
                "fade": ("FLOAT", dict(unit)),
                "vignette": ("FLOAT", dict(signed)),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0x7FFFFFFFFFFFFFFF}),
                "strength": ("FLOAT", _slider(1.0, 0.0, 1.0, 0.01)),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "effect_mask")
    FUNCTION = "enhance_detail"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "依次执行降噪、去雾、清晰度、纹理、锐化，并可加入颗粒、褪色与暗角。"

    def enhance_detail(self, image, clarity, texture, dehaze, sharpen,
                       luminance_denoise, color_denoise, grain, fade, vignette,
                       seed, strength, mask=None):
        frames, masks = prepare_single(image, mask)
        rendered = [detail_enhance(frame, clarity, texture, dehaze, sharpen,
                                   luminance_denoise, color_denoise, grain, fade,
                                   vignette, int(seed) + index)
                    for index, frame in enumerate(frames)]
        return _graded_result(image, frames, rendered, masks, strength)


class TUT_HSLBasic:
    @classmethod
    def INPUT_TYPES(cls):
        control = _slider(0.0, -1.0, 1.0, 0.01)
        return {
            "required": {
                "image": ("IMAGE",),
                "color_range": (["全局", "红色", "橙色", "黄色", "绿色", "青色", "蓝色", "紫色", "洋红"],),
                "hue_shift": ("FLOAT", dict(control)),
                "saturation": ("FLOAT", dict(control)),
                "lightness": ("FLOAT", dict(control)),
                "strength": ("FLOAT", _slider(1.0, 0.0, 1.0, 0.01)),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "effect_mask")
    FUNCTION = "adjust_hsl"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "按全局或八个平滑颜色范围调整色相、饱和度和亮度。"

    def adjust_hsl(self, image, color_range, hue_shift, saturation, lightness, strength, mask=None):
        frames, masks = prepare_single(image, mask)
        rendered = [selective_hsl(frame, color_range, hue_shift, saturation, lightness)
                    for frame in frames]
        return _graded_result(image, frames, rendered, masks, strength)


class TUT_ColorMatch:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "source_image": ("IMAGE",),
                "reference_image": ("IMAGE",),
                "method": (["Lab均值方差", "直方图匹配", "分区分布"],),
                "channel_mode": (["颜色和亮度", "仅颜色", "仅亮度"],),
                "protect_luminance": ("BOOLEAN", {"default": False}),
                "strength": ("FLOAT", _slider(1.0, 0.0, 1.0, 0.01)),
            },
            "optional": {"source_mask": ("MASK",), "reference_mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "show_help", "correction_mask", "difference_image")
    FUNCTION = "match_color"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "将源图的 Lab 统计、直方图或分区色彩分布匹配到参考图。"

    def match_color(self, source_image, reference_image, method, channel_mode,
                    protect_luminance, strength, source_mask=None, reference_mask=None):
        sources, references, source_masks, reference_masks = prepare_pair(
            source_image, reference_image, source_mask, reference_mask
        )
        rendered = [color_match(src, ref, sm, rm, method, channel_mode, protect_luminance)
                    for src, ref, sm, rm in zip(sources, references, source_masks, reference_masks)]
        output = _effect_output(source_image, rendered, source_masks, strength)
        output_np = output.detach().cpu().numpy()
        difference = np.abs(output_np - sources)
        correction = np.max(difference, axis=-1)
        _, correction_mask = tensors(np.stack(rendered), (correction,))
        difference_tensor, = tensors(difference)
        return output, HELP, correction_mask, difference_tensor


class TUT_FilmTone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["自定义", "中性电影", "暖调印片", "冷调惊悚", "低饱和剧情", "高反差银幕"],),
                "toe": ("FLOAT", _slider(0.18, 0.0, 1.0, 0.01)),
                "shoulder": ("FLOAT", _slider(0.22, 0.0, 1.0, 0.01)),
                "density": ("FLOAT", _slider(1.0, 0.25, 2.0, 0.01)),
                "saturation_compression": ("FLOAT", _slider(0.2, 0.0, 1.0, 0.01)),
                "temperature": ("FLOAT", _slider(0.0, -1.0, 1.0, 0.01)),
                "highlight_tint": ("FLOAT", _slider(0.0, -1.0, 1.0, 0.01)),
                "strength": ("FLOAT", _slider(1.0, 0.0, 1.0, 0.01)),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "effect_mask")
    FUNCTION = "apply_tone"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "塑造胶片式 Toe、Shoulder、密度、饱和度和冷暖偏色。"

    def apply_tone(self, image, preset, toe, shoulder, density, saturation_compression,
                   temperature, highlight_tint, strength, mask=None):
        frames, masks = prepare_single(image, mask)
        rendered = [film_tone(frame, toe, shoulder, density, saturation_compression,
                              temperature, highlight_tint) for frame in frames]
        output = _effect_output(image, rendered, masks, strength)
        _, effect_mask = tensors(np.stack(rendered), (masks * float(strength),))
        return output, HELP, effect_mask


class TUT_Halation:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["自定义", "细腻胶片", "经典35mm", "强烈红晕", "柔和暖晕"],),
                "highlight_threshold": ("FLOAT", _slider(0.72, 0.0, 1.0, 0.01)),
                "softness": ("FLOAT", _slider(0.18, 0.01, 1.0, 0.01)),
                "radius": ("FLOAT", _slider(8.0, 0.1, 128.0, 0.1)),
                "spread": ("FLOAT", _slider(0.6, 0.0, 2.0, 0.01)),
                "red_orange_ratio": ("FLOAT", _slider(0.55, 0.0, 1.0, 0.01)),
                "strength": ("FLOAT", _slider(0.6, 0.0, 1.0, 0.01)),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "halation_mask")
    FUNCTION = "apply_halation"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "从高光边缘生成红橙色胶片卤化光晕。"

    def apply_halation(self, image, preset, highlight_threshold, softness, radius,
                       spread, red_orange_ratio, strength, mask=None):
        frames, masks = prepare_single(image, mask)
        pairs = [halation(frame, highlight_threshold, softness, radius, spread, red_orange_ratio)
                 for frame in frames]
        rendered = [item[0] for item in pairs]
        halos = np.stack([item[1] * region * float(strength) for item, region in zip(pairs, masks)])
        output = _effect_output(image, rendered, masks, strength)
        _, halo_tensor = tensors(np.stack(rendered), (halos,))
        return output, HELP, halo_tensor


class TUT_LensDiffusion:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["自定义", "柔光镜", "黑柔", "薄雾", "梦幻扩散"],),
                "mode": (["柔光镜", "黑柔", "薄雾", "梦幻扩散"],),
                "radius": ("FLOAT", _slider(5.0, 0.1, 128.0, 0.1)),
                "highlight_threshold": ("FLOAT", _slider(0.62, 0.0, 1.0, 0.01)),
                "contrast_softening": ("FLOAT", _slider(0.45, 0.0, 1.0, 0.01)),
                "strength": ("FLOAT", _slider(0.5, 0.0, 1.0, 0.01)),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "diffusion_mask")
    FUNCTION = "apply_diffusion"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "模拟柔光镜、黑柔、薄雾与梦幻镜头扩散。"

    def apply_diffusion(self, image, preset, mode, radius, highlight_threshold,
                        contrast_softening, strength, mask=None):
        frames, masks = prepare_single(image, mask)
        pairs = [lens_diffusion(frame, mode, radius, highlight_threshold, contrast_softening)
                 for frame in frames]
        rendered = [item[0] for item in pairs]
        diffusion = np.stack([item[1] * region * float(strength) for item, region in zip(pairs, masks)])
        output = _effect_output(image, rendered, masks, strength)
        _, diffusion_tensor = tensors(np.stack(rendered), (diffusion,))
        return output, HELP, diffusion_tensor


class TUT_ColorCompressor:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "preset": (["自定义", "青橙聚合", "暖棕电影", "冷蓝夜景", "单色海报", "柔和肤色"],),
                "target_color": ("STRING", {"default": "#2B8C8C"}),
                "hue_range": ("FLOAT", _slider(90.0, 1.0, 180.0, 1.0)),
                "saturation_limit": ("FLOAT", _slider(0.72, 0.0, 1.0, 0.01)),
                "preserve_luminance": ("BOOLEAN", {"default": True}),
                "protect_skin": ("BOOLEAN", {"default": True}),
                "strength": ("FLOAT", _slider(0.65, 0.0, 1.0, 0.01)),
            },
            "optional": _optional_mask(),
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "show_help", "compressed_mask", "palette_image")
    FUNCTION = "compress_color"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "将颜色向目标色域聚合，并可保护亮度和肤色。"

    def compress_color(self, image, preset, target_color, hue_range, saturation_limit,
                       preserve_luminance, protect_skin, strength, mask=None):
        frames, masks = prepare_single(image, mask)
        pairs = [color_compress(frame, target_color, hue_range, saturation_limit,
                                preserve_luminance, protect_skin) for frame in frames]
        rendered = [item[0] for item in pairs]
        compressed = np.stack([item[1] * region * float(strength) for item, region in zip(pairs, masks)])
        output = _effect_output(image, rendered, masks, strength)
        target = parse_hex(target_color)
        colors = [np.clip(target * factor, 0.0, 1.0) for factor in (0.35, 0.65, 1.0, 1.25)]
        previews = np.stack([palette_preview(colors) for _ in frames])
        _, compressed_tensor = tensors(np.stack(rendered), (compressed,))
        preview_tensor, = tensors(previews)
        return output, HELP, compressed_tensor, preview_tensor


NODE_CLASS_MAPPINGS = {
    "TUT_AutoColorCorrect": TUT_AutoColorCorrect,
    "TUT_AutoColorCorrectAdvanced": TUT_AutoColorCorrectAdvanced,
    "TUT_BasicTone": TUT_BasicTone,
    "TUT_BasicColor": TUT_BasicColor,
    "TUT_DetailEnhance": TUT_DetailEnhance,
    "TUT_HSLBasic": TUT_HSLBasic,
    "TUT_ColorMatch": TUT_ColorMatch,
    "TUT_FilmTone": TUT_FilmTone,
    "TUT_Halation": TUT_Halation,
    "TUT_LensDiffusion": TUT_LensDiffusion,
    "TUT_ColorCompressor": TUT_ColorCompressor,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_AutoColorCorrect": "TUT_自动基础校色",
    "TUT_AutoColorCorrectAdvanced": "TUT_自动基础校色（高级）",
    "TUT_BasicTone": "TUT_基础明暗调整",
    "TUT_BasicColor": "TUT_基础色彩调整",
    "TUT_DetailEnhance": "TUT_图像细节增强",
    "TUT_HSLBasic": "TUT_HSL基础调整",
    "TUT_ColorMatch": "TUT_双图颜色匹配",
    "TUT_FilmTone": "TUT_电影色调塑形",
    "TUT_Halation": "TUT_卤化光晕",
    "TUT_LensDiffusion": "TUT_镜头扩散",
    "TUT_ColorCompressor": "TUT_色彩压缩器",
}
