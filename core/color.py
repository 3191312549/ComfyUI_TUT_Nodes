"""Color grading primitives shared by TUT color nodes."""

from __future__ import annotations

import colorsys
from functools import lru_cache

import numpy as np
import torch
from PIL import Image


@lru_cache(maxsize=1)
def _cv2():
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError(
            "电影调色节点需要 OpenCV，请安装 opencv-python-headless>=4.10,<6"
        ) from exc
    return cv2


def _images(value, name="IMAGE"):
    array = value.detach().cpu().float().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 3:
        array = array[None]
    if array.ndim != 4 or array.shape[0] == 0 or array.shape[-1] not in (1, 3, 4):
        raise ValueError(f"{name} 必须是非空批次 [B,H,W,C]")
    if array.shape[-1] == 1:
        array = np.repeat(array, 3, axis=-1)
    return np.clip(array[..., :3], 0.0, 1.0).astype(np.float32, copy=False)


def _masks(value, name="MASK"):
    array = value.detach().cpu().float().numpy() if hasattr(value, "detach") else np.asarray(value)
    if array.ndim == 2:
        array = array[None]
    if array.ndim == 4 and array.shape[-1] == 1:
        array = array[..., 0]
    if array.ndim != 3 or array.shape[0] == 0:
        raise ValueError(f"{name} 必须是非空批次 [B,H,W]")
    return np.clip(array, 0.0, 1.0).astype(np.float32, copy=False)


def _broadcast(*batches):
    lengths = [len(item) for item in batches]
    target = max(lengths)
    if any(length not in (1, target) for length in lengths):
        raise ValueError(f"批次数量无法匹配：{lengths}；只允许长度相同或单帧广播")
    return tuple(np.repeat(item, target, axis=0) if len(item) == 1 else item for item in batches)


def _resize_image(image, size):
    cv2 = _cv2()
    return cv2.resize(image, size, interpolation=cv2.INTER_AREA if image.shape[1] > size[0] else cv2.INTER_CUBIC)


def _resize_mask(mask, size):
    return np.clip(_cv2().resize(mask, size, interpolation=_cv2().INTER_LINEAR), 0.0, 1.0)


def prepare_single(image, mask=None):
    images = _images(image)
    if mask is None:
        masks = np.ones(images.shape[:3], dtype=np.float32)
    else:
        masks = _masks(mask)
        images, masks = _broadcast(images, masks)
        masks = np.stack([_resize_mask(item, (frame.shape[1], frame.shape[0]))
                          if item.shape != frame.shape[:2] else item
                          for frame, item in zip(images, masks)])
    return images, masks


def prepare_analysis_effect(image, analysis_mask=None, effect_mask=None):
    images = _images(image)
    analysis = (np.ones(images.shape[:3], dtype=np.float32)
                if analysis_mask is None else _masks(analysis_mask, "分析 MASK"))
    effects = (np.ones(images.shape[:3], dtype=np.float32)
               if effect_mask is None else _masks(effect_mask, "效果 MASK"))
    images, analysis, effects = _broadcast(images, analysis, effects)

    def resized(masks):
        return np.stack([
            _resize_mask(item, (frame.shape[1], frame.shape[0]))
            if item.shape != frame.shape[:2] else item
            for frame, item in zip(images, masks)
        ])

    return images, resized(analysis), resized(effects)


def prepare_pair(first, second, first_mask=None, second_mask=None):
    a, b = _images(first, "图一"), _images(second, "图二")
    ma = np.ones(a.shape[:3], np.float32) if first_mask is None else _masks(first_mask, "图一 MASK")
    mb = np.ones(b.shape[:3], np.float32) if second_mask is None else _masks(second_mask, "图二 MASK")
    a, b, ma, mb = _broadcast(a, b, ma, mb)
    out_b, out_ma, out_mb = [], [], []
    for source, reference, sm, rm in zip(a, b, ma, mb):
        size = (source.shape[1], source.shape[0])
        out_b.append(_resize_image(reference, size) if reference.shape[:2] != source.shape[:2] else reference)
        out_ma.append(_resize_mask(sm, size) if sm.shape != source.shape[:2] else sm)
        out_mb.append(_resize_mask(rm, size) if rm.shape != source.shape[:2] else rm)
    return a, np.stack(out_b), np.stack(out_ma), np.stack(out_mb)


def tensors(images, masks=()):
    image_tensor = torch.from_numpy(np.ascontiguousarray(np.clip(images, 0.0, 1.0))).float()
    mask_tensors = tuple(torch.from_numpy(np.ascontiguousarray(np.clip(mask, 0.0, 1.0))).float() for mask in masks)
    return (image_tensor, *mask_tensors)


def exact_or_composite(original, rendered, masks, strength):
    strength = float(strength)
    if not 0.0 <= strength <= 1.0:
        raise ValueError("strength 必须在 0 到 1 之间")
    target = len(rendered)
    if strength == 0.0:
        return original if int(original.shape[0]) == target else original.repeat(target, 1, 1, 1)
    original_np = _images(original)
    if len(original_np) == 1 and target > 1:
        original_np = np.repeat(original_np, target, axis=0)
    result = original_np.copy()
    for index, (effect, mask) in enumerate(zip(rendered, masks)):
        if np.max(mask) <= 0.0:
            continue
        alpha = np.clip(mask * strength, 0.0, 1.0)[..., None]
        result[index] = original_np[index] * (1.0 - alpha) + effect * alpha
    return torch.from_numpy(np.ascontiguousarray(np.clip(result, 0.0, 1.0))).float()


def _srgb_to_linear(image):
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.where(
        image <= 0.04045,
        image / 12.92,
        np.power((image + 0.055) / 1.055, 2.4),
    ).astype(np.float32)


def _linear_to_srgb(image):
    image = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    return np.where(
        image <= 0.0031308,
        image * 12.92,
        1.055 * np.power(image, 1.0 / 2.4) - 0.055,
    ).astype(np.float32)


def _luminance(image):
    return np.asarray(image, dtype=np.float32) @ np.array([0.2126, 0.7152, 0.0722], np.float32)


def auto_color_correct(image, region, auto_white_balance=True, auto_exposure=True):
    """Deterministic, masked white-balance and exposure correction."""
    source = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    region = np.clip(np.asarray(region, dtype=np.float32), 0.0, 1.0)
    if (not bool(auto_white_balance) and not bool(auto_exposure)) or np.max(region) <= 1e-5:
        return source.copy()

    linear = _srgb_to_linear(source)
    display_luma = _luminance(source)
    valid = (region > 1e-4) & (display_luma > 0.03) & (display_luma < 0.97)
    if not np.any(valid):
        valid = region > 1e-4
    weights = np.where(valid, region, 0.0).astype(np.float32)
    weight_sum = float(weights.sum())
    if weight_sum <= 1e-6:
        return source.copy()

    corrected = linear.copy()
    if bool(auto_white_balance):
        means = (linear * weights[..., None]).sum(axis=(0, 1)) / weight_sum
        target = float(np.exp(np.mean(np.log(np.maximum(means, 1e-6)))))
        gains = np.clip(target / np.maximum(means, 1e-6), 0.5, 2.0)
        # Keep white balance from unintentionally acting as a second exposure control.
        gain_luma = float(gains @ np.array([0.2126, 0.7152, 0.0722], np.float32))
        gains /= max(gain_luma, 1e-6)
        corrected = np.clip(corrected * gains.reshape(1, 1, 3), 0.0, 1.0)

    if bool(auto_exposure):
        selected_luma = _luminance(corrected)[valid]
        if selected_luma.size:
            median = float(np.median(selected_luma))
            exposure_ev = float(np.clip(np.log2(0.18 / max(median, 1e-6)), -2.0, 2.0))
            if exposure_ev > 0.0:
                high = float(np.percentile(selected_luma, 99.0))
                highlight_limit = max(0.0, float(np.log2(0.98 / max(high, 1e-6))))
                exposure_ev = min(exposure_ev, highlight_limit)
            corrected = np.clip(corrected * (2.0 ** exposure_ev), 0.0, 1.0)

    return _linear_to_srgb(corrected)


ADVANCED_WHITE_BALANCE_METHODS = ("自适应融合", "中性像素", "灰世界", "灰度幂均值")


def _advanced_channel_gains(illuminant):
    illuminant = np.maximum(np.asarray(illuminant, dtype=np.float64), 1e-8)
    target = float(np.exp(np.mean(np.log(illuminant))))
    gains = target / illuminant
    gain_luma = float(gains @ np.array([0.2126, 0.7152, 0.0722], np.float64))
    return gains / max(gain_luma, 1e-8)


def _advanced_weighted_mean(values, weights):
    total = float(np.sum(weights))
    if total <= 1e-8:
        return np.mean(values, axis=0, dtype=np.float64)
    return np.sum(values * weights[:, None], axis=0, dtype=np.float64) / total


def _smoothstep_ratio(value, threshold):
    if threshold <= 0.0:
        return 1.0
    ratio = float(np.clip(value / threshold, 0.0, 1.0))
    return ratio * ratio * (3.0 - 2.0 * ratio)


def advanced_auto_color_correct(
    image,
    analysis_region,
    auto_white_balance=True,
    white_balance_method="自适应融合",
    neutral_strictness=0.65,
    white_balance_strength=1.0,
    max_white_balance_ev=0.75,
    auto_exposure=True,
    target_midtone=0.18,
    max_exposure_ev=1.5,
    highlight_protection=0.85,
    confidence_threshold=0.45,
):
    """Robust global correction with neutral sampling and confidence attenuation."""
    if white_balance_method not in ADVANCED_WHITE_BALANCE_METHODS:
        raise ValueError(f"不支持的高级白平衡方法：{white_balance_method!r}")
    ranges = {
        "中性取样严格度": (neutral_strictness, 0.0, 1.0),
        "白平衡强度": (white_balance_strength, 0.0, 1.0),
        "白平衡最大档位": (max_white_balance_ev, 0.0, 2.0),
        "中灰目标": (target_midtone, 0.08, 0.30),
        "曝光最大档位": (max_exposure_ev, 0.0, 3.0),
        "高光保护": (highlight_protection, 0.0, 1.0),
        "置信度阈值": (confidence_threshold, 0.0, 1.0),
    }
    for label, (value, lower, upper) in ranges.items():
        value = float(value)
        if not np.isfinite(value) or not lower <= value <= upper:
            raise ValueError(f"{label} 必须在 {lower:g} 到 {upper:g} 之间")

    source = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    region = np.clip(np.asarray(analysis_region, dtype=np.float32), 0.0, 1.0)
    empty_sample = np.zeros(source.shape[:2], dtype=np.float32)
    diagnostics = {
        "status": "没有有效分析像素",
        "confidence": 0.0,
        "confidence_factor": 0.0,
        "valid_ratio": 0.0,
        "neutral_ratio": 0.0,
        "raw_gains": np.ones(3, dtype=np.float64),
        "effective_gains": np.ones(3, dtype=np.float64),
        "target_exposure_ev": 0.0,
        "effective_exposure_ev": 0.0,
    }
    region_pixels = region > 1e-4
    if not np.any(region_pixels):
        return source.copy(), empty_sample, diagnostics

    display_luma = _luminance(source)
    luma_values = display_luma[region_pixels]
    lower = max(0.01, float(np.percentile(luma_values, 2.0)))
    upper = min(0.99, float(np.percentile(luma_values, 98.0)))
    valid = region_pixels & (display_luma >= lower) & (display_luma <= upper)
    if lower >= upper or int(np.count_nonzero(valid)) < 64:
        valid = region_pixels & (display_luma > 0.01) & (display_luma < 0.99)
    valid_count = int(np.count_nonzero(valid))
    if valid_count < 16:
        return source.copy(), empty_sample, diagnostics

    linear = _srgb_to_linear(source)
    valid_values = linear[valid].astype(np.float64)
    valid_weights = region[valid].astype(np.float64)
    channel_max = np.max(valid_values, axis=1)
    channel_min = np.min(valid_values, axis=1)
    chroma = (channel_max - channel_min) / np.maximum(channel_max + channel_min, 1e-6)

    retain_fraction = 0.50 - 0.45 * float(neutral_strictness)
    neutral_count = max(1, int(np.ceil(valid_count * retain_fraction)))
    neutral_local = np.argsort(chroma, kind="stable")[:neutral_count]
    neutral_values = valid_values[neutral_local]
    neutral_weights = valid_weights[neutral_local] * (
        np.square(1.0 - np.clip(chroma[neutral_local], 0.0, 1.0)) + 1e-3
    )

    neutral_illuminant = _advanced_weighted_mean(neutral_values, neutral_weights)
    gray_illuminant = _advanced_weighted_mean(valid_values, valid_weights)
    power_illuminant = np.power(
        _advanced_weighted_mean(np.power(np.maximum(valid_values, 0.0), 6.0), valid_weights),
        1.0 / 6.0,
    )
    estimator_gains = np.stack([
        _advanced_channel_gains(neutral_illuminant),
        _advanced_channel_gains(power_illuminant),
        _advanced_channel_gains(gray_illuminant),
    ])
    if white_balance_method == "自适应融合":
        raw_log_gains = np.sum(
            np.log2(np.maximum(estimator_gains, 1e-8))
            * np.array([0.80, 0.15, 0.05], dtype=np.float64)[:, None],
            axis=0,
        )
        raw_gains = np.exp2(raw_log_gains)
    elif white_balance_method == "中性像素":
        raw_gains = estimator_gains[0]
    elif white_balance_method == "灰度幂均值":
        raw_gains = estimator_gains[1]
    else:
        raw_gains = estimator_gains[2]

    neutral_quality = float(np.clip(1.0 - np.median(chroma[neutral_local]) / 0.90, 0.0, 1.0))
    log_estimators = np.log2(np.maximum(estimator_gains, 1e-8))
    agreement_spread = float(np.max(np.ptp(log_estimators, axis=0)))
    agreement = float(np.clip(1.0 - agreement_spread, 0.0, 1.0))
    support = float(np.clip(neutral_count / max(64.0, 0.02 * valid_count), 0.0, 1.0))
    confidence = float(np.sqrt(neutral_quality * (0.75 * agreement + 0.25 * support)))
    confidence_factor = _smoothstep_ratio(confidence, float(confidence_threshold))

    corrected = linear.copy()
    effective_gains = np.ones(3, dtype=np.float64)
    if bool(auto_white_balance):
        limited_log_gains = np.clip(
            np.log2(np.maximum(raw_gains, 1e-8)),
            -float(max_white_balance_ev),
            float(max_white_balance_ev),
        )
        effective_gains = np.exp2(
            limited_log_gains * float(white_balance_strength) * confidence_factor
        )
        corrected = np.clip(corrected * effective_gains.reshape(1, 1, 3), 0.0, 1.0)

    target_ev = 0.0
    effective_ev = 0.0
    if bool(auto_exposure):
        selected_luma = _luminance(corrected)[valid]
        if selected_luma.size:
            median = float(np.median(selected_luma))
            target_ev = float(np.clip(
                np.log2(float(target_midtone) / max(median, 1e-8)),
                -float(max_exposure_ev),
                float(max_exposure_ev),
            ))
            effective_ev = target_ev
            if target_ev > 0.0:
                high = float(np.percentile(_luminance(corrected)[region_pixels], 99.5))
                safe_ev = max(0.0, float(np.log2(0.98 / max(high, 1e-8))))
                protected_ev = min(target_ev, safe_ev)
                protection = float(highlight_protection)
                effective_ev = target_ev * (1.0 - protection) + protected_ev * protection
            corrected = np.clip(corrected * (2.0 ** effective_ev), 0.0, 1.0)

    sample_mask = np.zeros(source.shape[:2], dtype=np.float32)
    valid_flat = np.flatnonzero(valid)
    chosen_flat = valid_flat[neutral_local] if bool(auto_white_balance) else valid_flat
    sample_mask.flat[chosen_flat] = region.flat[chosen_flat]
    diagnostics.update({
        "status": "已完成",
        "confidence": confidence,
        "confidence_factor": confidence_factor if bool(auto_white_balance) else 1.0,
        "valid_ratio": valid_count / max(1, int(np.count_nonzero(region_pixels))),
        "neutral_ratio": neutral_count / max(1, valid_count),
        "raw_gains": raw_gains,
        "effective_gains": effective_gains,
        "target_exposure_ev": target_ev,
        "effective_exposure_ev": effective_ev,
    })
    rendered = (source.copy() if not bool(auto_white_balance) and not bool(auto_exposure)
                else _linear_to_srgb(corrected))
    return rendered, sample_mask, diagnostics


def basic_tone(image, exposure, brightness, contrast, highlights, shadows, whites, blacks,
               light_sensation=0.0):
    """Apply photographic exposure followed by a monotonic display-referred tone curve."""
    source = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    values = [exposure, brightness, contrast, highlights, shadows, whites, blacks, light_sensation]
    if all(abs(float(value)) <= 1e-12 for value in values):
        return source.copy()

    exposure = float(np.clip(exposure, -5.0, 5.0))
    exposed = _linear_to_srgb(np.clip(_srgb_to_linear(source) * (2.0 ** exposure), 0.0, 1.0))

    brightness, contrast, highlights, shadows, whites, blacks, light_sensation = (
        float(np.clip(value, -1.0, 1.0))
        for value in (brightness, contrast, highlights, shadows, whites, blacks, light_sensation)
    )
    grid = np.linspace(0.0, 1.0, 4097, dtype=np.float32)
    middle = 4.0 * grid * (1.0 - grid)
    curve = grid.copy()
    curve += brightness * 0.28 * middle
    curve += contrast * 0.22 * (2.0 * grid - 1.0) * middle
    curve += shadows * 0.24 * (1.0 - grid) * middle
    curve += highlights * 0.24 * grid * middle
    curve += blacks * 0.12 * np.power(1.0 - grid, 3.0)
    curve += whites * 0.12 * np.power(grid, 3.0)
    curve += light_sensation * 0.18 * np.sqrt(grid) * (1.0 - grid)
    curve = np.maximum.accumulate(np.clip(curve, 0.0, 1.0))
    if curve[-1] > curve[0] + 1e-6:
        # Preserve the adjusted endpoints while keeping the full curve monotonic.
        curve = np.clip(curve, 0.0, 1.0)
    result = np.empty_like(exposed)
    for channel in range(3):
        result[..., channel] = np.interp(exposed[..., channel], grid, curve)
    return np.clip(result, 0.0, 1.0)


def basic_color(image, temperature, tint, saturation, vibrance):
    """Adjust blue-yellow, green-magenta, saturation and protected vibrance."""
    source = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    temperature, tint, saturation, vibrance = (
        float(np.clip(value, -1.0, 1.0))
        for value in (temperature, tint, saturation, vibrance)
    )
    if max(abs(temperature), abs(tint), abs(saturation), abs(vibrance)) <= 1e-12:
        return source.copy()

    linear = _srgb_to_linear(source)
    gains = np.array([
        2.0 ** (temperature * 0.35 + tint * 0.12),
        2.0 ** (-tint * 0.24),
        2.0 ** (-temperature * 0.35 + tint * 0.12),
    ], dtype=np.float32)
    gain_luma = float(gains @ np.array([0.2126, 0.7152, 0.0722], np.float32))
    balanced = _linear_to_srgb(np.clip(linear * (gains / max(gain_luma, 1e-6)), 0.0, 1.0))

    luma = _luminance(balanced)
    channel_max = balanced.max(axis=-1)
    channel_min = balanced.min(axis=-1)
    chroma = channel_max - channel_min
    saturation_factor = 1.0 + saturation

    skin_protection = np.zeros_like(chroma)
    if vibrance > 0.0:
        hsv = _cv2().cvtColor(balanced, _cv2().COLOR_RGB2HSV)
        hue_distance = np.minimum(np.abs(hsv[..., 0] - 25.0), 360.0 - np.abs(hsv[..., 0] - 25.0))
        skin_protection = np.clip(1.0 - hue_distance / 28.0, 0.0, 1.0)
        skin_protection *= np.clip((hsv[..., 1] - 0.08) / 0.42, 0.0, 1.0)
    vibrance_factor = 1.0 + vibrance * np.power(1.0 - chroma, 2.0) * (1.0 - 0.55 * skin_protection)
    factor = np.maximum(0.0, saturation_factor * vibrance_factor)
    result = luma[..., None] + (balanced - luma[..., None]) * factor[..., None]
    return np.clip(result, 0.0, 1.0)


def detail_enhance(image, clarity, texture, dehaze, sharpen,
                   luminance_denoise, color_denoise, grain=0.0,
                   fade=0.0, vignette=0.0, seed=0):
    """Resolution-aware denoise, dehaze, local contrast, texture and sharpening."""
    source = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    clarity, texture, dehaze = (
        float(np.clip(value, -1.0, 1.0)) for value in (clarity, texture, dehaze)
    )
    sharpen = float(np.clip(sharpen, 0.0, 2.0))
    luminance_denoise = float(np.clip(luminance_denoise, 0.0, 1.0))
    color_denoise = float(np.clip(color_denoise, 0.0, 1.0))
    grain = float(np.clip(grain, 0.0, 1.0))
    fade = float(np.clip(fade, 0.0, 1.0))
    vignette = float(np.clip(vignette, -1.0, 1.0))
    if max(abs(clarity), abs(texture), abs(dehaze), sharpen,
           luminance_denoise, color_denoise, grain, fade, abs(vignette)) <= 1e-12:
        return source.copy()

    cv2 = _cv2()
    scale = max(0.5, min(source.shape[:2]) / 512.0)
    luma = _luminance(source)
    chroma = source - luma[..., None]

    if luminance_denoise > 0.0:
        blurred_luma = cv2.GaussianBlur(luma, (0, 0), sigmaX=max(0.5, 1.2 * scale))
        luma = luma * (1.0 - luminance_denoise) + blurred_luma * luminance_denoise
    if color_denoise > 0.0:
        blurred_chroma = cv2.GaussianBlur(chroma, (0, 0), sigmaX=max(0.5, 2.0 * scale))
        chroma = chroma * (1.0 - color_denoise) + blurred_chroma * color_denoise
    working = np.clip(luma[..., None] + chroma, 0.0, 1.0)

    if dehaze != 0.0:
        luma = _luminance(working)
        broad = cv2.GaussianBlur(luma, (0, 0), sigmaX=max(0.5, 24.0 * scale))
        delta = dehaze * ((luma - broad) * 0.85 + (luma - 0.5) * 0.10)
        working = np.clip(working + delta[..., None], 0.0, 1.0)
    if clarity != 0.0:
        luma = _luminance(working)
        local = luma - cv2.GaussianBlur(luma, (0, 0), sigmaX=max(0.5, 7.0 * scale))
        midtone_weight = 4.0 * luma * (1.0 - luma)
        working = np.clip(working + (clarity * 1.25 * local * midtone_weight)[..., None], 0.0, 1.0)
    if texture != 0.0:
        luma = _luminance(working)
        fine = luma - cv2.GaussianBlur(luma, (0, 0), sigmaX=max(0.5, 1.5 * scale))
        working = np.clip(working + (texture * fine)[..., None], 0.0, 1.0)
    if sharpen > 0.0:
        luma = _luminance(working)
        edge = luma - cv2.GaussianBlur(luma, (0, 0), sigmaX=max(0.5, 0.8 * scale))
        working = np.clip(working + (sharpen * 0.8 * edge)[..., None], 0.0, 1.0)
    if grain > 0.0:
        generator = np.random.default_rng(int(seed) & 0xFFFFFFFFFFFFFFFF)
        noise = generator.normal(0.0, 0.035 * grain, size=working.shape[:2]).astype(np.float32)
        grain_weight = 0.35 + 0.65 * (1.0 - np.abs(2.0 * _luminance(working) - 1.0))
        working = np.clip(working + (noise * grain_weight)[..., None], 0.0, 1.0)
    if fade > 0.0:
        working = np.clip(working * (1.0 - 0.18 * fade) + 0.10 * fade, 0.0, 1.0)
    if vignette != 0.0:
        height, width = working.shape[:2]
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        xx = (xx + 0.5 - width * 0.5) / max(width * 0.5, 1.0)
        yy = (yy + 0.5 - height * 0.5) / max(height * 0.5, 1.0)
        radial = np.clip((np.sqrt(xx * xx + yy * yy) - 0.25) / 0.75, 0.0, 1.0)
        radial = radial * radial * (3.0 - 2.0 * radial)
        factor = 1.0 + vignette * 0.65 * radial
        working = np.clip(working * factor[..., None], 0.0, 1.0)
    return working.astype(np.float32, copy=False)


HSL_RANGES = {
    "红色": (0.0, 35.0),
    "橙色": (30.0, 30.0),
    "黄色": (60.0, 35.0),
    "绿色": (120.0, 55.0),
    "青色": (180.0, 45.0),
    "蓝色": (230.0, 45.0),
    "紫色": (275.0, 40.0),
    "洋红": (315.0, 40.0),
}


def selective_hsl(image, color_range, hue_shift, saturation, lightness):
    """Adjust one smooth HSL hue range, or the complete image."""
    source = np.clip(np.asarray(image, dtype=np.float32), 0.0, 1.0)
    hue_shift, saturation, lightness = (
        float(np.clip(value, -1.0, 1.0)) for value in (hue_shift, saturation, lightness)
    )
    if max(abs(hue_shift), abs(saturation), abs(lightness)) <= 1e-12:
        return source.copy()
    if color_range != "全局" and color_range not in HSL_RANGES:
        raise ValueError(f"不支持的 HSL 颜色范围：{color_range!r}")

    cv2 = _cv2()
    hls = cv2.cvtColor(source, cv2.COLOR_RGB2HLS)
    if color_range == "全局":
        weight = np.ones(source.shape[:2], dtype=np.float32)
    else:
        center, width = HSL_RANGES[color_range]
        distance = np.abs(hls[..., 0] - center)
        distance = np.minimum(distance, 360.0 - distance)
        weight = np.clip(1.0 - distance / width, 0.0, 1.0)
        weight = weight * weight * (3.0 - 2.0 * weight)

    adjusted = hls.copy()
    adjusted[..., 0] = (hls[..., 0] + hue_shift * 60.0 * weight) % 360.0
    if saturation >= 0.0:
        target_saturation = hls[..., 2] + (1.0 - hls[..., 2]) * saturation
    else:
        target_saturation = hls[..., 2] * (1.0 + saturation)
    if lightness >= 0.0:
        target_lightness = hls[..., 1] + (1.0 - hls[..., 1]) * lightness
    else:
        target_lightness = hls[..., 1] * (1.0 + lightness)
    adjusted[..., 2] = hls[..., 2] * (1.0 - weight) + target_saturation * weight
    adjusted[..., 1] = hls[..., 1] * (1.0 - weight) + target_lightness * weight
    return np.clip(cv2.cvtColor(adjusted.astype(np.float32), cv2.COLOR_HLS2RGB), 0.0, 1.0)


def _weighted_stats(values, mask):
    selected = mask.reshape(-1) > 1e-5
    flat = values.reshape(-1, values.shape[-1])
    if not np.any(selected):
        return np.zeros(values.shape[-1], np.float32), np.ones(values.shape[-1], np.float32)
    sample = flat[selected]
    return sample.mean(axis=0), np.maximum(sample.std(axis=0), 1e-4)


def _histogram_match(source, reference, source_mask, reference_mask):
    result = source.copy()
    source_selected = source_mask.reshape(-1) > 1e-5
    reference_selected = reference_mask.reshape(-1) > 1e-5
    if not np.any(source_selected) or not np.any(reference_selected):
        return result
    flat = result.reshape(-1, 3)
    source_flat, reference_flat = source.reshape(-1, 3), reference.reshape(-1, 3)
    for channel in range(3):
        values, inverse, counts = np.unique(source_flat[source_selected, channel], return_inverse=True, return_counts=True)
        ref_values, ref_counts = np.unique(reference_flat[reference_selected, channel], return_counts=True)
        source_q = np.cumsum(counts).astype(np.float64); source_q /= source_q[-1]
        ref_q = np.cumsum(ref_counts).astype(np.float64); ref_q /= ref_q[-1]
        mapped = np.interp(source_q, ref_q, ref_values)
        flat[source_selected, channel] = mapped[inverse]
    return result


def color_match(source, reference, source_mask, reference_mask, method, channel_mode, protect_luminance):
    cv2 = _cv2()
    if not np.any(source_mask > 1e-5) or not np.any(reference_mask > 1e-5):
        return source.copy()
    src_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
    ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)
    if method == "直方图匹配":
        matched = _histogram_match(src_lab, ref_lab, source_mask, reference_mask)
    elif method == "分区分布":
        matched = src_lab.copy()
        src_l, ref_l = src_lab[..., 0], ref_lab[..., 0]
        for low, high in ((0.0, 33.0), (33.0, 67.0), (67.0, 101.0)):
            sm = source_mask * ((src_l >= low) & (src_l < high))
            rm = reference_mask * ((ref_l >= low) & (ref_l < high))
            smean, sstd = _weighted_stats(src_lab, sm)
            rmean, rstd = _weighted_stats(ref_lab, rm)
            region = sm > 0
            matched[region] = (src_lab[region] - smean) * (rstd / sstd) + rmean
    else:
        smean, sstd = _weighted_stats(src_lab, source_mask)
        rmean, rstd = _weighted_stats(ref_lab, reference_mask)
        matched = (src_lab - smean) * (rstd / sstd) + rmean
    if channel_mode == "仅颜色" or bool(protect_luminance):
        matched[..., 0] = src_lab[..., 0]
    elif channel_mode == "仅亮度":
        matched[..., 1:] = src_lab[..., 1:]
    return np.clip(cv2.cvtColor(matched.astype(np.float32), cv2.COLOR_LAB2RGB), 0.0, 1.0)


def film_tone(image, toe, shoulder, density, saturation_compression, temperature, highlight_tint):
    x = np.clip(image, 0.0, 1.0)
    toe = max(0.0, float(toe)); shoulder = max(0.0, float(shoulder))
    shaped = np.power(x, 1.0 + toe * 1.5)
    shaped = shaped * (1.0 + shoulder) / (1.0 + shoulder * shaped)
    shaped = np.clip(shaped * max(0.0, float(density)), 0.0, 1.0)
    luma = shaped @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    saturation = 1.0 - np.clip(float(saturation_compression), 0.0, 1.0) * np.clip((luma - 0.55) / 0.45, 0.0, 1.0)
    shaped = luma[..., None] + (shaped - luma[..., None]) * saturation[..., None]
    temp = float(temperature) * 0.12
    shaped[..., 0] += temp; shaped[..., 2] -= temp
    tint = float(highlight_tint) * np.clip((luma - 0.5) / 0.5, 0.0, 1.0)
    shaped[..., 0] += tint * 0.08; shaped[..., 1] += tint * 0.025; shaped[..., 2] -= tint * 0.06
    return np.clip(shaped, 0.0, 1.0)


def halation(image, threshold, softness, radius, spread, red_orange):
    cv2 = _cv2()
    luma = image @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    soft = max(1e-4, float(softness))
    highlights = np.clip((luma - float(threshold)) / soft, 0.0, 1.0)
    sigma = max(0.1, float(radius))
    broad = cv2.GaussianBlur(highlights, (0, 0), sigmaX=sigma)
    core = cv2.GaussianBlur(highlights, (0, 0), sigmaX=max(0.1, sigma * 0.25))
    ring = np.clip(broad * (1.0 + float(spread)) - core * float(spread), 0.0, 1.0)
    orange = np.array([1.0, 0.24 + 0.36 * float(red_orange), 0.04], np.float32)
    effect = 1.0 - (1.0 - image) * (1.0 - ring[..., None] * orange)
    return np.clip(effect, 0.0, 1.0), ring


def lens_diffusion(image, mode, radius, highlight_threshold, contrast_softening):
    cv2 = _cv2(); sigma = max(0.1, float(radius))
    blurred = cv2.GaussianBlur(image, (0, 0), sigmaX=sigma)
    luma = image @ np.array([0.2126, 0.7152, 0.0722], np.float32)
    highlight = np.clip((luma - float(highlight_threshold)) / max(1e-4, 1.0 - float(highlight_threshold)), 0.0, 1.0)
    if mode == "黑柔":
        effect = image * (1.0 - float(contrast_softening) * 0.22) + blurred * float(contrast_softening) * 0.22
        effect += blurred * highlight[..., None] * 0.18
        mask = np.maximum(highlight, float(contrast_softening) * 0.25)
    elif mode == "薄雾":
        effect = image * (1.0 - float(contrast_softening) * 0.3) + blurred * float(contrast_softening) * 0.3 + 0.07 * float(contrast_softening)
        mask = np.clip(highlight + float(contrast_softening) * 0.35, 0.0, 1.0)
    elif mode == "梦幻扩散":
        screen = 1.0 - (1.0 - image) * (1.0 - blurred)
        effect = image * (1.0 - 0.55 * float(contrast_softening)) + screen * 0.55 * float(contrast_softening)
        mask = np.clip(highlight + float(contrast_softening) * 0.45, 0.0, 1.0)
    else:
        effect = image + blurred * highlight[..., None] * (0.45 + float(contrast_softening) * 0.35)
        mask = highlight
    return np.clip(effect, 0.0, 1.0), np.clip(mask, 0.0, 1.0)


def parse_hex(value):
    token = str(value).strip().lstrip("#")
    if len(token) == 3:
        token = "".join(char * 2 for char in token)
    if len(token) != 6:
        raise ValueError(f"颜色必须是 #RGB 或 #RRGGBB：{value!r}")
    try:
        return np.array([int(token[i:i + 2], 16) / 255.0 for i in (0, 2, 4)], np.float32)
    except ValueError as exc:
        raise ValueError(f"颜色包含无效字符：{value!r}") from exc


def color_compress(image, target_hex, hue_range, saturation_limit, preserve_luminance, protect_skin):
    cv2 = _cv2()
    target = parse_hex(target_hex)
    target_hsv = cv2.cvtColor(target.reshape(1, 1, 3), cv2.COLOR_RGB2HSV)[0, 0]
    hsv = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    delta = np.abs(hsv[..., 0] - target_hsv[0]); delta = np.minimum(delta, 360.0 - delta)
    range_value = max(1.0, float(hue_range))
    mask = np.clip(1.0 - delta / range_value, 0.0, 1.0)
    if bool(protect_skin):
        skin_delta = np.minimum(np.abs(hsv[..., 0] - 25.0), 360.0 - np.abs(hsv[..., 0] - 25.0))
        mask *= 1.0 - np.clip(1.0 - skin_delta / 22.0, 0.0, 1.0) * np.clip(hsv[..., 1] / 0.25, 0.0, 1.0)
    signed_hue_delta = (target_hsv[0] - hsv[..., 0] + 180.0) % 360.0 - 180.0
    hsv[..., 0] = (hsv[..., 0] + signed_hue_delta * mask) % 360.0
    hsv[..., 1] = np.minimum(hsv[..., 1], float(saturation_limit))
    result = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
    if bool(preserve_luminance):
        old_luma = image @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        new_luma = result @ np.array([0.2126, 0.7152, 0.0722], np.float32)
        result *= (old_luma / np.maximum(new_luma, 1e-4))[..., None]
    return np.clip(result, 0.0, 1.0), mask


def palette_preview(colors, width=512, height=64):
    canvas = np.zeros((height, width, 3), np.float32)
    for index, color in enumerate(colors):
        left, right = round(index * width / len(colors)), round((index + 1) * width / len(colors))
        canvas[:, left:right] = color
    return canvas
