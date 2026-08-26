"""Two-image layer compositing primitives with procedural edge treatments."""

from __future__ import annotations

import math

import numpy as np
from PIL import Image, ImageChops, ImageDraw, ImageFilter


_RESAMPLING = getattr(Image, "Resampling", Image)
_TRANSFORM = getattr(Image, "Transform", Image)


def resize_layer(layer: Image.Image, background_size: tuple[int, int], size_mode: str, scale: float):
    width, height = layer.size
    factor = max(0.01, float(scale))
    if size_mode == "适应背景":
        factor *= min(background_size[0] / max(1, width), background_size[1] / max(1, height))
    elif size_mode != "原始尺寸":
        raise ValueError(f"未知图层尺寸模式：{size_mode}")
    target = (max(1, round(width * factor)), max(1, round(height * factor)))
    if target[0] * target[1] > 67_108_864 or max(target) > 8192:
        raise ValueError(f"缩放后的图层尺寸过大：{target[0]}x{target[1]}")
    return layer.resize(target, _RESAMPLING.LANCZOS)


def _smooth_noise(size: tuple[int, int], rng, detail_scale: float) -> np.ndarray:
    width, height = size
    cell = max(2, int(round(float(detail_scale))))
    small_size = (max(2, math.ceil(width / cell)), max(2, math.ceil(height / cell)))
    values = rng.random((small_size[1], small_size[0]), dtype=np.float32)
    image = Image.fromarray(np.clip(values * 255.0, 0, 255).astype(np.uint8), "L")
    image = image.resize(size, _RESAMPLING.BICUBIC)
    return np.asarray(image, dtype=np.float32) / 255.0


def _shape_mask(size, mode, amount, irregularity, detail_scale, rng):
    width, height = size
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    extent = max(0, min(round(float(amount)), min(width, height) // 2))
    if mode == "关闭":
        return Image.new("L", size, 255)
    if mode == "圆角":
        draw.rounded_rectangle((0, 0, width - 1, height - 1), radius=extent, fill=255)
    elif mode == "切角":
        cut = extent
        draw.polygon(((cut, 0), (width - cut - 1, 0), (width - 1, cut),
                      (width - 1, height - cut - 1), (width - cut - 1, height - 1),
                      (cut, height - 1), (0, height - cut - 1), (0, cut)), fill=255)
    elif mode == "波浪":
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        period = max(4.0, float(detail_scale) * 3.0)
        wave_y = extent * (0.5 + 0.5 * np.sin(xx * 2.0 * math.pi / period))
        wave_x = extent * (0.5 + 0.5 * np.sin(yy * 2.0 * math.pi / period + math.pi / 2))
        valid = (yy >= wave_y) & (yy < height - wave_y) & (xx >= wave_x) & (xx < width - wave_x)
        mask = Image.fromarray(valid.astype(np.uint8) * 255, "L")
    elif mode == "撕裂":
        noise = _smooth_noise(size, rng, detail_scale)
        edge = extent * (0.2 + 0.8 * ((1.0 - irregularity) * 0.5 + irregularity * noise))
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        valid = (xx >= edge) & (xx < width - edge) & (yy >= edge) & (yy < height - edge)
        mask = Image.fromarray(valid.astype(np.uint8) * 255, "L")
    else:
        raise ValueError(f"未知边缘形状：{mode}")
    return mask


def _erode(binary: np.ndarray) -> np.ndarray:
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    views = [padded[y:y + binary.shape[0], x:x + binary.shape[1]] for y in range(3) for x in range(3)]
    return np.logical_and.reduce(views)


def _dilate(binary: np.ndarray) -> np.ndarray:
    padded = np.pad(binary, 1, mode="constant", constant_values=False)
    views = [padded[y:y + binary.shape[0], x:x + binary.shape[1]] for y in range(3) for x in range(3)]
    return np.logical_or.reduce(views)


def signed_edge_distance(alpha: Image.Image, limit: int) -> np.ndarray:
    """Return a bounded signed Chebyshev distance field in pixels."""
    maximum = max(1, int(limit))
    binary = np.asarray(alpha, dtype=np.uint8) >= 128
    signed = np.full(binary.shape, maximum + 1, dtype=np.float32)
    current = binary.copy()
    for step in range(1, maximum + 1):
        eroded = _erode(current)
        signed[current & ~eroded] = float(step)
        current = eroded
        if not current.any():
            break
    current = binary.copy()
    for step in range(1, maximum + 1):
        expanded = _dilate(current)
        signed[expanded & ~current] = -float(step)
        current = expanded
    signed[~current] = -(maximum + 1)
    return signed


def build_local_layer(layer, source_mask, shape_mode, shape_amount, shape_strength,
                      transition_mode, transition_strength, edge_width, irregularity,
                      detail_scale, material_mode, material_strength, edge_color, rng):
    width, height = layer.size
    pad = max(8, int(edge_width) * 2 + 4)
    canvas_size = (width + pad * 2, height + pad * 2)
    base = Image.new("L", canvas_size, 0)
    base_shape = _shape_mask(
        (width, height), shape_mode, shape_amount, irregularity, detail_scale, rng,
    )
    full = Image.new("L", (width, height), 255)
    shaped = Image.blend(full, base_shape, max(0.0, min(1.0, float(shape_strength))))
    if source_mask is not None:
        resized_mask = source_mask.convert("L").resize((width, height), _RESAMPLING.BILINEAR)
        shaped = ImageChops.multiply(shaped, resized_mask)
    base.paste(shaped, (pad, pad))

    distance = signed_edge_distance(base, max(2, int(edge_width) * 2 + 2))
    width_px = max(1.0, float(edge_width))
    strength = max(0.0, min(1.0, float(transition_strength)))
    original = np.asarray(base, dtype=np.float32) / 255.0
    if transition_mode == "关闭" or strength <= 0.0:
        transitioned = original
    elif transition_mode == "羽化":
        softened = np.asarray(base.filter(ImageFilter.GaussianBlur(width_px / 2.0)), dtype=np.float32) / 255.0
        transitioned = original * (1.0 - strength) + softened * strength
    else:
        noise = _smooth_noise(canvas_size, rng, detail_scale)
        if transition_mode == "像素崩解":
            block = max(2, int(round(detail_scale)))
            noise = np.repeat(np.repeat(noise[::block, ::block], block, 0), block, 1)[:canvas_size[1], :canvas_size[0]]
        elif transition_mode == "墨水扩散":
            noise_image = Image.fromarray(np.clip(noise * 255, 0, 255).astype(np.uint8), "L")
            noise = np.asarray(noise_image.filter(ImageFilter.GaussianBlur(max(1.0, width_px / 3))), dtype=np.float32) / 255.0
        elif transition_mode != "噪声溶解":
            raise ValueError(f"未知边缘消散：{transition_mode}")
        displacement = (noise - 0.5) * 2.0 * width_px * max(0.0, float(irregularity))
        coverage = np.clip((distance + displacement) / width_px + 0.5, 0.0, 1.0)
        if transition_mode == "墨水扩散":
            coverage = np.clip(coverage * 0.85 + noise * 0.3, 0.0, 1.0)
        transitioned = original * (1.0 - strength) + coverage * strength
    alpha = Image.fromarray(np.clip(transitioned * 255, 0, 255).astype(np.uint8), "L")

    rgb = Image.new("RGB", canvas_size, (0, 0, 0))
    rgb.paste(layer.convert("RGB"), (pad, pad))
    material_strength = max(0.0, min(1.0, float(material_strength)))
    if material_mode != "关闭" and material_strength > 0.0:
        distance = signed_edge_distance(alpha, max(2, int(edge_width) + 2))
        outer = np.clip((distance + width_px) / width_px, 0.0, 1.0) * (distance <= 0)
        inner = np.clip(1.0 - distance / width_px, 0.0, 1.0) * (distance > 0)
        color = np.asarray(edge_color, dtype=np.float32)
        material_rgb = np.broadcast_to(color, (canvas_size[1], canvas_size[0], 3)).copy()
        material_alpha = outer
        if material_mode == "纸张纤维":
            fibers = _smooth_noise(canvas_size, rng, max(2.0, detail_scale / 2.0))
            material_rgb = np.clip(material_rgb * (0.82 + fibers[..., None] * 0.3), 0, 255)
            material_alpha = outer * np.clip(0.65 + fibers * 0.55, 0, 1)
        elif material_mode == "烧焦":
            burn = np.clip(inner + outer, 0, 1)
            material_rgb = np.zeros_like(material_rgb)
            material_rgb[..., 0] = 58 + 120 * outer
            material_rgb[..., 1] = 23 + 42 * outer
            material_rgb[..., 2] = 10
            material_alpha = burn
        elif material_mode == "玻璃切边":
            material_rgb[..., 0] = np.clip(color[0] * 0.45 + 150, 0, 255)
            material_rgb[..., 1] = np.clip(color[1] * 0.45 + 175, 0, 255)
            material_rgb[..., 2] = np.clip(color[2] * 0.45 + 190, 0, 255)
            material_alpha = np.clip((outer + inner * 0.55) * 0.7, 0, 1)
        elif material_mode != "白边贴纸":
            raise ValueError(f"未知边缘材质：{material_mode}")
        material_alpha = np.clip(material_alpha * material_strength, 0, 1)
        material_image = Image.fromarray(material_rgb.astype(np.uint8), "RGB")
        material_mask = Image.fromarray((material_alpha * 255).astype(np.uint8), "L")
        rgb = Image.composite(rgb, material_image, alpha)
        rgb = Image.composite(rgb, material_image, material_mask)
        alpha = ImageChops.lighter(alpha, material_mask)
    return rgb, alpha


def _perspective_coefficients(destination, source):
    matrix, values = [], []
    for (x, y), (u, v) in zip(destination, source):
        matrix.append([x, y, 1, 0, 0, 0, -u * x, -u * y])
        matrix.append([0, 0, 0, x, y, 1, -v * x, -v * y])
        values.extend([u, v])
    return np.linalg.lstsq(np.asarray(matrix, dtype=np.float64), np.asarray(values, dtype=np.float64), rcond=None)[0]


def transform_and_place(image, alpha, background_size, position_x, position_y,
                        rotation, tilt_x, tilt_y):
    width, height = image.size
    source = ((0, 0), (width - 1, 0), (width - 1, height - 1), (0, height - 1))
    yaw = abs(math.sin(math.radians(float(tilt_x)))) * width * 0.28
    pitch = abs(math.sin(math.radians(float(tilt_y)))) * height * 0.28
    left = yaw if float(tilt_x) > 0 else 0.0
    right = width - 1 - (yaw if float(tilt_x) < 0 else 0.0)
    top = pitch if float(tilt_y) > 0 else 0.0
    bottom = height - 1 - (pitch if float(tilt_y) < 0 else 0.0)
    destination = ((left, top), (right, top), (right, bottom), (left, bottom))
    if yaw > 0.01 or pitch > 0.01:
        coefficients = _perspective_coefficients(destination, source)
        image = image.transform(image.size, _TRANSFORM.PERSPECTIVE, coefficients, _RESAMPLING.BICUBIC, fillcolor=(0, 0, 0))
        alpha = alpha.transform(alpha.size, _TRANSFORM.PERSPECTIVE, coefficients, _RESAMPLING.BICUBIC, fillcolor=0)
    if float(rotation) != 0.0:
        image = image.rotate(float(rotation), _RESAMPLING.BICUBIC, expand=True, fillcolor=(0, 0, 0))
        alpha = alpha.rotate(float(rotation), _RESAMPLING.BICUBIC, expand=True, fillcolor=0)
    placed_image = Image.new("RGB", background_size, (0, 0, 0))
    placed_alpha = Image.new("L", background_size, 0)
    center = (float(position_x) * background_size[0], float(position_y) * background_size[1])
    origin = (round(center[0] - image.width / 2), round(center[1] - image.height / 2))
    placed_image.paste(image, origin)
    placed_alpha.paste(alpha, origin)
    return placed_image, placed_alpha


def _blend(background, foreground, mode):
    if mode == "normal":
        return foreground
    if mode == "multiply":
        return ImageChops.multiply(background, foreground)
    if mode == "screen":
        return ImageChops.screen(background, foreground)
    if mode == "overlay":
        base = np.asarray(background, dtype=np.float32) / 255.0
        top = np.asarray(foreground, dtype=np.float32) / 255.0
        result = np.where(base <= 0.5, 2 * base * top, 1 - 2 * (1 - base) * (1 - top))
        return Image.fromarray(np.clip(result * 255, 0, 255).astype(np.uint8), "RGB")
    raise ValueError(f"未知混合模式：{mode}")


def _color_layer(size, color):
    return Image.new("RGB", size, tuple(int(value) for value in color))


def apply_depth(background, foreground, alpha, mode, strength, edge_width,
                edge_color, offset_x, offset_y, blur):
    amount = max(0.0, min(1.0, float(strength)))
    if mode == "关闭" or amount <= 0.0:
        return background, foreground
    if mode == "柔投影":
        shifted = Image.new("L", alpha.size, 0)
        shifted.paste(alpha, (int(offset_x), int(offset_y)))
        shadow = shifted.filter(ImageFilter.GaussianBlur(max(0.0, float(blur))))
        shadow = shadow.point(lambda value: round(value * amount * 0.75))
        background = Image.composite(_color_layer(background.size, (0, 0, 0)), background, shadow)
    elif mode == "霓虹边光":
        glow = alpha.filter(ImageFilter.GaussianBlur(max(1.0, float(blur))))
        glow = ImageChops.subtract(glow, alpha).point(lambda value: round(value * amount))
        lit = ImageChops.screen(background, _color_layer(background.size, edge_color))
        background = Image.composite(lit, background, glow)
    elif mode == "伪厚度":
        thickness = Image.new("L", alpha.size, 0)
        steps = max(1, int(edge_width))
        for step in range(1, steps + 1):
            x = round(float(offset_x) * step / steps)
            y = round(float(offset_y) * step / steps)
            shifted = Image.new("L", alpha.size, 0)
            shifted.paste(alpha, (x, y))
            thickness = ImageChops.lighter(thickness, shifted)
        thickness = ImageChops.subtract(thickness, alpha).point(lambda value: round(value * amount))
        background = Image.composite(_color_layer(background.size, edge_color), background, thickness)
    elif mode == "斜面浮雕":
        array = np.asarray(foreground, dtype=np.float32)
        mask = np.asarray(alpha, dtype=np.float32) / 255.0
        gy, gx = np.gradient(mask)
        shade = np.clip((gx + gy) * max(1.0, float(edge_width)) * amount, -0.4, 0.4)
        array = np.clip(array * (1.0 + shade[..., None]), 0, 255)
        foreground = Image.fromarray(array.astype(np.uint8), "RGB")
    else:
        raise ValueError(f"未知边缘光影：{mode}")
    return background, foreground


def composite_soft_layer(layer, background, source_mask, *, size_mode, scale,
                         position_x, position_y, rotation, tilt_x, tilt_y,
                         opacity, blend_mode, shape_mode, shape_amount, shape_strength,
                         transition_mode, transition_strength, material_mode,
                         material_strength, depth_mode, depth_strength, edge_width,
                         edge_color, detail_scale, irregularity, background_wrap,
                         background_blur, depth_offset_x, depth_offset_y, shadow_blur, rng):
    resized = resize_layer(layer, background.size, size_mode, scale)
    rgb, alpha = build_local_layer(
        resized, source_mask, shape_mode, shape_amount, shape_strength,
        transition_mode, transition_strength, edge_width, irregularity,
        detail_scale, material_mode, material_strength, edge_color, rng,
    )
    foreground, placed_alpha = transform_and_place(
        rgb, alpha, background.size, position_x, position_y, rotation, tilt_x, tilt_y,
    )
    opacity_value = max(0.0, min(1.0, float(opacity)))
    if opacity_value <= 0.0 or placed_alpha.getbbox() is None:
        return background.copy(), False

    result, foreground = apply_depth(
        background.copy(), foreground, placed_alpha, depth_mode, depth_strength,
        edge_width, edge_color, depth_offset_x, depth_offset_y, shadow_blur,
    )
    if float(background_blur) > 0.0:
        blurred = result.filter(ImageFilter.GaussianBlur(float(background_blur)))
        result = Image.composite(blurred, result, placed_alpha)
    wrap = max(0.0, min(1.0, float(background_wrap)))
    if wrap > 0.0:
        distance = signed_edge_distance(placed_alpha, max(2, int(edge_width) + 2))
        band = np.clip(1.0 - distance / max(1.0, float(edge_width)), 0.0, 1.0) * (distance > 0)
        band_mask = Image.fromarray(np.clip(band * wrap * 255, 0, 255).astype(np.uint8), "L")
        wrapped = result.filter(ImageFilter.GaussianBlur(max(1.0, float(edge_width) / 3.0)))
        foreground = Image.composite(wrapped, foreground, band_mask)

    final_alpha = placed_alpha.point(lambda value: round(value * opacity_value))
    blended = _blend(result, foreground, blend_mode)
    return Image.composite(blended, result, final_alpha), True
