"""Reusable image-filter primitives for TUT_Nodes.

Optional Kornia acceleration is used when available, with a Pillow/NumPy/Torch
fallback. Node wrappers live in ``nodes/image``; keeping the algorithms here
makes static and animated filters share the same deterministic behavior.
"""

from __future__ import annotations

import colorsys
import math
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageChops, ImageFilter, ImageOps

from .imaging import (
    broadcast_batches,
    image_tensor_to_pil_batch,
    mask_tensor_to_pil_batch,
    pil_batch_to_image_tensor,
    pil_batch_to_mask_tensor,
)


RESAMPLING = getattr(Image, "Resampling", Image)
DITHER = getattr(Image, "Dither", Image)
QUANTIZE = getattr(Image, "Quantize", Image)


@lru_cache(maxsize=1)
def _kornia_filters():
    try:
        from kornia import filters
    except ImportError:
        return None
    return filters


def clamp_strength(value) -> float:
    try:
        value = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"strength 必须是 0 到 1 的数字：{value!r}") from exc
    if not 0.0 <= value <= 1.0:
        raise ValueError(f"strength 必须在 0 到 1 之间：{value}")
    return value


def prepare_filter_batches(images, mask=None):
    frames = image_tensor_to_pil_batch(images)
    if mask is None:
        masks = [Image.new("L", frame.size, 255) for frame in frames]
    else:
        masks = mask_tensor_to_pil_batch(mask)
        frames, masks = broadcast_batches(frames, masks)
        masks = [item.resize(frame.size, RESAMPLING.BILINEAR) if item.size != frame.size else item
                 for frame, item in zip(frames, masks)]
    return frames, masks


def effect_mask_tensor(masks):
    return pil_batch_to_mask_tensor(masks)


def image_batch_tensor(images):
    return pil_batch_to_image_tensor(images)


def composite_effect(original: Image.Image, effect: Image.Image, mask: Image.Image, strength: float):
    strength = clamp_strength(strength)
    if strength == 0.0:
        return original.copy()
    alpha = np.asarray(mask.convert("L"), dtype=np.float32) * strength
    alpha_image = Image.fromarray(np.clip(alpha, 0, 255).astype(np.uint8), "L")
    return Image.composite(effect.convert("RGB"), original.convert("RGB"), alpha_image)


def rng_for(seed, batch_index=0):
    try:
        seed = int(seed)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"seed 必须是整数：{seed!r}") from exc
    return np.random.default_rng((seed + int(batch_index)) & 0xFFFFFFFFFFFFFFFF)


def parse_palette(value: str, fallback=None):
    colors = []
    for token in str(value or "").replace(";", ",").split(","):
        token = token.strip().lstrip("#")
        if not token:
            continue
        if len(token) == 3:
            token = "".join(char * 2 for char in token)
        if len(token) != 6:
            raise ValueError(f"调色板颜色必须是 #RGB 或 #RRGGBB：{token!r}")
        try:
            colors.append(tuple(int(token[index:index + 2], 16) for index in (0, 2, 4)))
        except ValueError as exc:
            raise ValueError(f"调色板包含无效颜色：{token!r}") from exc
    if not colors and fallback:
        colors = list(fallback)
    if not colors:
        raise ValueError("调色板至少需要一种有效颜色")
    return colors[:256]


def _nearest_palette(array, colors):
    palette = np.asarray(colors, dtype=np.float32)
    flat = array.reshape(-1, 3).astype(np.float32)
    indexes = np.argmin(np.sum((flat[:, None, :] - palette[None, :, :]) ** 2, axis=2), axis=1)
    return palette[indexes].reshape(array.shape).astype(np.uint8)


def _bayer_quantize(array, colors, size=4):
    matrices = {
        2: np.array([[0, 2], [3, 1]], dtype=np.float32),
        4: np.array([[0, 8, 2, 10], [12, 4, 14, 6], [3, 11, 1, 9], [15, 7, 13, 5]], dtype=np.float32),
    }
    matrix = matrices[size]
    h, w = array.shape[:2]
    tiled = np.tile(matrix, (math.ceil(h / size), math.ceil(w / size)))[:h, :w]
    offset = ((tiled + 0.5) / (size * size) - 0.5) * 42.0
    adjusted = np.clip(array.astype(np.float32) + offset[..., None], 0, 255)
    return _nearest_palette(adjusted, colors)


def quantize_image(image, max_colors=16, palette=None, dither="无"):
    max_colors = max(2, min(256, int(max_colors)))
    if palette:
        colors = list(palette)[:max_colors]
        array = np.asarray(image.convert("RGB"), dtype=np.uint8)
        if dither == "Bayer 2x2":
            output = _bayer_quantize(array, colors, 2)
        elif dither == "Bayer 4x4":
            output = _bayer_quantize(array, colors, 4)
        elif dither == "Floyd-Steinberg":
            pal = Image.new("P", (1, 1))
            flat = [channel for color in colors for channel in color] + [0] * (768 - len(colors) * 3)
            pal.putpalette(flat)
            output = np.asarray(image.convert("RGB").quantize(palette=pal, dither=DITHER.FLOYDSTEINBERG).convert("RGB"))
        else:
            output = _nearest_palette(array, colors)
        return Image.fromarray(output, "RGB"), colors

    dither_mode = DITHER.FLOYDSTEINBERG if dither == "Floyd-Steinberg" else DITHER.NONE
    quantized = image.convert("RGB").quantize(colors=max_colors, method=QUANTIZE.MEDIANCUT, dither=dither_mode)
    raw_palette = quantized.getpalette() or []
    used = sorted(set(np.asarray(quantized, dtype=np.uint8).reshape(-1).tolist()))
    colors = [tuple(raw_palette[index * 3:index * 3 + 3]) for index in used if index * 3 + 2 < len(raw_palette)]
    result = quantized.convert("RGB")
    if dither in ("Bayer 2x2", "Bayer 4x4"):
        result = Image.fromarray(_bayer_quantize(np.asarray(image.convert("RGB")), colors, 2 if "2x2" in dither else 4), "RGB")
    return result, colors


def palette_preview(colors, width=512, height=64):
    colors = list(colors) or [(0, 0, 0)]
    array = np.zeros((height, width, 3), dtype=np.uint8)
    for index, color in enumerate(colors):
        left = round(index * width / len(colors))
        right = round((index + 1) * width / len(colors))
        array[:, left:right] = color
    return Image.fromarray(array, "RGB")


def retro_print(image, color_mode, palette_text, dot_size, screen_angle, registration_shift,
                ink_bleed, paper_grain, rng):
    source = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    h, w = source.shape[:2]
    fallback = [(25, 24, 28), (226, 49, 77), (29, 145, 192), (244, 214, 76)]
    colors = parse_palette(palette_text, fallback)
    count_map = {"单色": 2, "双色": 3, "三色": 4, "CMYK": 5}
    count = min(len(colors), count_map.get(color_mode, 4) - 1)
    colors = colors[:max(1, count)]
    luminance = source @ np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    dot_size = max(2, int(dot_size))
    result = np.ones((h, w, 3), dtype=np.float32) * np.array([0.94, 0.91, 0.82], dtype=np.float32)
    total_ink = np.zeros((h, w), dtype=np.float32)
    for channel, color in enumerate(colors):
        angle = math.radians(float(screen_angle) + channel * (30.0 if color_mode == "CMYK" else 17.0))
        shift = float(registration_shift) * (channel - (len(colors) - 1) / 2)
        u = (xx * math.cos(angle) + yy * math.sin(angle) + shift) / dot_size
        v = (-xx * math.sin(angle) + yy * math.cos(angle) - shift) / dot_size
        dots = (np.sin(u * math.pi) * np.sin(v * math.pi) + 1.0) * 0.5
        if color_mode == "CMYK":
            coverage = 1.0 - source[..., channel] if channel < 3 else 1.0 - np.max(source, axis=2)
        else:
            target = np.asarray(color, dtype=np.float32) / 255.0
            chroma = 1.0 - np.mean(np.abs(source - target), axis=2)
            coverage = np.clip((1.0 - luminance) * 0.55 + chroma * 0.65 - channel * 0.08, 0.0, 1.0)
        plate = (coverage > dots).astype(np.float32)
        if float(ink_bleed) > 0:
            plate_img = Image.fromarray((plate * 255).astype(np.uint8), "L").filter(ImageFilter.MaxFilter(max(3, int(ink_bleed) * 2 + 1)))
            plate = np.asarray(plate_img, dtype=np.float32) / 255.0
        ink = np.asarray(color, dtype=np.float32) / 255.0
        result = result * (1.0 - plate[..., None] * 0.78) + ink * plate[..., None] * 0.78
        total_ink = np.maximum(total_ink, plate)
    grain = rng.normal(0.0, max(0.0, float(paper_grain)) * 0.08, (h, w)).astype(np.float32)
    result = np.clip(result + grain[..., None] * (1.0 - total_ink[..., None] * 0.5), 0.0, 1.0)
    texture = np.clip(total_ink * (1.0 - np.maximum(0.0, grain)), 0.0, 1.0)
    return Image.fromarray((result * 255).astype(np.uint8), "RGB"), Image.fromarray((texture * 255).astype(np.uint8), "L")


def comic_filter(image, color_levels, line_width, line_strength, line_threshold,
                 shadow_threshold, shadow_halftone, dot_size):
    rgb = image.convert("RGB")
    levels = max(2, int(color_levels))
    poster = np.asarray(rgb, dtype=np.float32)
    poster = np.round(poster / 255.0 * (levels - 1)) / (levels - 1) * 255.0
    gray = ImageOps.grayscale(rgb)
    edges = gray.filter(ImageFilter.FIND_EDGES).filter(ImageFilter.GaussianBlur(0.6))
    edge_array = np.asarray(edges, dtype=np.float32) / 255.0
    line = (edge_array * max(0.0, float(line_strength)) >= float(line_threshold)).astype(np.uint8) * 255
    line_mask = Image.fromarray(line, "L")
    if int(line_width) > 1:
        line_mask = line_mask.filter(ImageFilter.MaxFilter(max(3, int(line_width) * 2 - 1)))
    lum = np.asarray(gray, dtype=np.float32) / 255.0
    shadow = lum < float(shadow_threshold)
    if bool(shadow_halftone):
        yy, xx = np.mgrid[0:lum.shape[0], 0:lum.shape[1]]
        size = max(2, int(dot_size))
        pattern = ((xx % size - size / 2) ** 2 + (yy % size - size / 2) ** 2) < (size * 0.28) ** 2
        shadow = shadow & pattern
    shadow_mask = Image.fromarray(shadow.astype(np.uint8) * 255, "L")
    output = Image.fromarray(np.clip(poster, 0, 255).astype(np.uint8), "RGB")
    output = Image.composite(Image.new("RGB", rgb.size, (20, 20, 24)), output, shadow_mask.point(lambda p: int(p * 0.35)))
    output = Image.composite(Image.new("RGB", rgb.size, (8, 8, 10)), output, line_mask)
    return output, line_mask, shadow_mask


def _sample_with_grid(image, grid_x, grid_y, channel_offsets=None):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    tensor = torch.from_numpy(array).permute(2, 0, 1).unsqueeze(0)
    h, w = array.shape[:2]
    gx = torch.from_numpy((grid_x / max(1, w - 1) * 2.0 - 1.0).astype(np.float32))
    gy = torch.from_numpy((grid_y / max(1, h - 1) * 2.0 - 1.0).astype(np.float32))
    if channel_offsets is None:
        grid = torch.stack((gx, gy), dim=-1).unsqueeze(0)
        sampled = F.grid_sample(tensor, grid, mode="bilinear", padding_mode="reflection", align_corners=True)[0]
    else:
        channels = []
        for channel, offset in enumerate(channel_offsets):
            grid = torch.stack((gx + float(offset) / max(1, w - 1) * 2.0, gy), dim=-1).unsqueeze(0)
            channels.append(F.grid_sample(tensor[:, channel:channel + 1], grid, mode="bilinear", padding_mode="reflection", align_corners=True)[0, 0])
        sampled = torch.stack(channels, dim=0)
    result = sampled.permute(1, 2, 0).clamp(0, 1).numpy()
    return Image.fromarray((result * 255).astype(np.uint8), "RGB")


def kaleidoscope(image, segments, center_x, center_y, rotation, zoom, mirror=True, seam_softness=0.0):
    w, h = image.size
    segments = max(2, min(24, int(segments)))
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    cx, cy = float(center_x) * (w - 1), float(center_y) * (h - 1)
    dx, dy = xx - cx, yy - cy
    radius = np.hypot(dx, dy) / max(0.05, float(zoom))
    angle = np.arctan2(dy, dx) - math.radians(float(rotation))
    sector = 2.0 * math.pi / segments
    folded = np.mod(angle, sector)
    if bool(mirror):
        folded = np.where(folded > sector / 2.0, sector - folded, folded)
    source_angle = folded + math.radians(float(rotation))
    grid_x = cx + radius * np.cos(source_angle)
    grid_y = cy + radius * np.sin(source_angle)
    result = _sample_with_grid(image, grid_x, grid_y)
    if float(seam_softness) > 0:
        result = result.filter(ImageFilter.GaussianBlur(min(3.0, float(seam_softness))))
    return result


def pixel_art(image, pixel_size, max_colors, palette_mode, custom_palette, dither,
              outline_strength, outline_threshold):
    w, h = image.size
    pixel_size = max(1, int(pixel_size))
    small_size = (max(1, math.ceil(w / pixel_size)), max(1, math.ceil(h / pixel_size)))
    small = image.convert("RGB").resize(small_size, RESAMPLING.BOX)
    palettes = {
        "Game Boy": [(15, 56, 15), (48, 98, 48), (139, 172, 15), (155, 188, 15)],
        "NES风格": [(15, 15, 28), (48, 96, 130), (230, 65, 58), (247, 214, 90), (245, 245, 235)],
        "16-bit": [(24, 20, 37), (63, 63, 116), (87, 155, 180), (151, 214, 173), (247, 226, 107), (218, 124, 48), (178, 62, 78), (255, 245, 232)],
        "街机": [(12, 12, 18), (255, 0, 85), (0, 229, 255), (255, 226, 0), (139, 57, 255), (250, 250, 250)],
    }
    palette = None
    if palette_mode == "自定义":
        palette = parse_palette(custom_palette)
    elif palette_mode in palettes:
        palette = palettes[palette_mode]
    quantized, colors = quantize_image(small, max_colors, palette, dither)
    if float(outline_strength) > 0:
        gray = ImageOps.grayscale(quantized)
        edge = np.asarray(gray.filter(ImageFilter.FIND_EDGES), dtype=np.float32) / 255.0
        edge = edge >= float(outline_threshold)
        arr = np.asarray(quantized).copy()
        factor = max(0.0, min(1.0, float(outline_strength)))
        arr[edge] = (arr[edge].astype(np.float32) * (1.0 - factor)).astype(np.uint8)
        quantized = Image.fromarray(arr, "RGB")
    return quantized.resize((w, h), RESAMPLING.NEAREST), colors


def _smooth_noise(rng, size, roughness):
    w, h = size
    coarse_w = max(2, min(w, round(w / max(4.0, 32.0 * (1.1 - roughness)))))
    coarse_h = max(2, min(h, round(h / max(4.0, 32.0 * (1.1 - roughness)))))
    coarse = rng.random((coarse_h, coarse_w), dtype=np.float32)
    kornia_filters = _kornia_filters()
    if kornia_filters is not None:
        tensor = torch.from_numpy(coarse)[None, None]
        tensor = F.interpolate(tensor, size=(h, w), mode="bicubic", align_corners=False)
        sigma = max(0.5, (1.0 - float(roughness)) * 3.0)
        radius = max(1, min(7, int(math.ceil(sigma * 2.0))))
        kernel = radius * 2 + 1
        tensor = kornia_filters.gaussian_blur2d(
            tensor, (kernel, kernel), (sigma, sigma), border_type="replicate"
        )
        return tensor[0, 0].clamp(0.0, 1.0).numpy()
    image = Image.fromarray((coarse * 255).astype(np.uint8), "L").resize((w, h), RESAMPLING.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def glass_refraction(image, mode, amount, scale, angle, blur, chromatic_aberration,
                     roughness, rng):
    w, h = image.size
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    radians = math.radians(float(angle))
    scale = max(2.0, float(scale))
    if mode == "条纹玻璃":
        height_map = np.sin((xx * math.cos(radians) + yy * math.sin(radians)) * 2 * math.pi / scale)
    elif mode == "水滴透镜":
        dx, dy = xx - (w - 1) / 2, yy - (h - 1) / 2
        radius = np.hypot(dx, dy) / max(1.0, min(w, h) * 0.48)
        height_map = np.clip(1.0 - radius * radius, 0.0, 1.0)
    elif mode == "磨砂玻璃":
        height_map = _smooth_noise(rng, (w, h), max(0.0, min(1.0, float(roughness)))) * 2.0 - 1.0
    elif mode == "液态玻璃":
        noise = _smooth_noise(rng, (w, h), max(0.1, min(1.0, float(roughness))))
        height_map = np.sin(xx * 2 * math.pi / scale + noise * math.pi) + np.cos(yy * 2 * math.pi / (scale * 1.3) - noise * math.pi)
    else:
        height_map = np.sin(xx * 2 * math.pi / scale) + np.sin(yy * 2 * math.pi / (scale * 1.2))
    grad_y, grad_x = np.gradient(height_map.astype(np.float32))
    dx = grad_x * float(amount)
    dy = grad_y * float(amount)
    source = image.filter(ImageFilter.GaussianBlur(max(0.0, float(blur)))) if float(blur) > 0 else image
    output = _sample_with_grid(source, xx + dx, yy + dy, (-float(chromatic_aberration), 0.0, float(chromatic_aberration)))
    max_disp = max(1.0, np.max(np.abs([dx, dy])))
    disp = np.zeros((h, w, 3), dtype=np.uint8)
    disp[..., 0] = np.clip((dx / max_disp * 0.5 + 0.5) * 255, 0, 255)
    disp[..., 1] = np.clip((dy / max_disp * 0.5 + 0.5) * 255, 0, 255)
    return output, Image.fromarray(disp, "RGB")


def glitch_art(image, mode, rgb_shift, block_count, block_height, scanline_strength,
               sort_threshold, noise_strength, rng, phase=0.0):
    array = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
    h, w = array.shape[:2]
    affected = np.zeros((h, w), dtype=np.uint8)
    shift = int(round(float(rgb_shift) * math.sin(float(phase) + math.pi / 2)))
    if mode in ("RGB故障", "VHS", "数据损坏") and shift:
        array[..., 0] = np.roll(array[..., 0], shift, axis=1)
        array[..., 2] = np.roll(array[..., 2], -shift, axis=1)
        affected[:, :min(w, abs(shift) * 2 + 1)] = 255
    max_height = max(1, min(h, int(block_height)))
    for _ in range(max(0, int(block_count))):
        top = int(rng.integers(0, max(1, h - max_height + 1)))
        height = int(rng.integers(1, max_height + 1))
        amount = int(rng.integers(-max(1, abs(int(rgb_shift)) * 3), max(2, abs(int(rgb_shift)) * 3 + 1)))
        amount = int(round(amount * (0.65 + 0.35 * math.sin(float(phase) + top))))
        array[top:top + height] = np.roll(array[top:top + height], amount, axis=1)
        affected[top:top + height] = 255
    if mode == "像素排序":
        luminance = array.mean(axis=2)
        for row in range(h):
            indexes = np.where(luminance[row] >= float(sort_threshold) * 255.0)[0]
            if len(indexes) > 1:
                start, end = int(indexes[0]), int(indexes[-1]) + 1
                order = np.argsort(luminance[row, start:end])
                array[row, start:end] = array[row, start:end][order]
                affected[row, start:end] = 255
    scan = max(0.0, min(1.0, float(scanline_strength)))
    if scan > 0:
        array[1::2] = (array[1::2].astype(np.float32) * (1.0 - scan * 0.65)).astype(np.uint8)
        affected[1::2] = np.maximum(affected[1::2], int(scan * 255))
    noise = max(0.0, min(1.0, float(noise_strength)))
    if noise > 0:
        field = rng.normal(0.0, noise * 32.0, array.shape[:2]).astype(np.float32)
        array = np.clip(array.astype(np.float32) + field[..., None], 0, 255).astype(np.uint8)
        affected[np.abs(field) > 1.0] = 255
    return Image.fromarray(array, "RGB"), Image.fromarray(affected, "L")


def hue_shift(image, fraction):
    array = np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0
    flat = array.reshape(-1, 3)
    out = np.empty_like(flat)
    shift = float(fraction) % 1.0
    for index, color in enumerate(flat):
        h, s, v = colorsys.rgb_to_hsv(*color)
        out[index] = colorsys.hsv_to_rgb((h + shift) % 1.0, s, v)
    return Image.fromarray(np.clip(out.reshape(array.shape) * 255, 0, 255).astype(np.uint8), "RGB")


def animated_effect(image, effect, phase, amplitude, rng, base_noise=None):
    w, h = image.size
    amount = float(amplitude)
    if effect == "色相循环":
        return hue_shift(image, phase / (2 * math.pi))
    if effect == "扫光":
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        xx = np.arange(w, dtype=np.float32)[None, :]
        center = (phase / (2 * math.pi) % 1.0) * (w * 1.4) - w * 0.2
        band = np.exp(-((xx - center) / max(2.0, w * 0.08)) ** 2) * 100.0 * amount
        return Image.fromarray(np.clip(array + band[..., None], 0, 255).astype(np.uint8), "RGB")
    if effect == "波纹":
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        dx, dy = xx - (w - 1) / 2, yy - (h - 1) / 2
        radius = np.hypot(dx, dy)
        offset = np.sin(radius / max(3.0, min(w, h) * 0.04) - phase) * amount * 8.0
        denom = np.maximum(radius, 1.0)
        return _sample_with_grid(image, xx + dx / denom * offset, yy + dy / denom * offset)
    if effect == "像素化渐变":
        size = max(1, round(1 + (0.5 - 0.5 * math.cos(phase)) * amount * 31))
        return pixel_art(image, size, 256, "自动", "", "无", 0.0, 1.0)[0]
    if effect == "万花筒旋转":
        return kaleidoscope(image, 8, 0.5, 0.5, phase / (2 * math.pi) * 360.0 * amount, 1.0, True, 0.0)
    if effect == "故障动画":
        return glitch_art(image, "数据损坏", round(12 * amount), round(8 * amount), 12, 0.2 * amount, 0.5, 0.08 * amount, rng, phase)[0]
    if effect == "胶片闪烁":
        array = np.asarray(image.convert("RGB"), dtype=np.float32)
        if base_noise is None:
            base_noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
        shifted = np.roll(base_noise, int(round(math.sin(phase) * 8)), axis=1)
        exposure = 1.0 + math.sin(phase) * 0.08 * amount
        return Image.fromarray(np.clip(array * exposure + shifted[..., None] * 18.0 * amount, 0, 255).astype(np.uint8), "RGB")
    raise ValueError(f"不支持的动态滤镜：{effect}")
