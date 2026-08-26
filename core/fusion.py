"""OpenCV-backed primitives for Fusion-style image compositing nodes."""

from __future__ import annotations

import math
from typing import Iterable

import numpy as np
import torch


def _cv2():
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover - depends on deployment
        raise RuntimeError(
            "此节点需要 OpenCV。请在 ComfyUI 的 Python 环境安装 "
            "opencv-python-headless>=4.10,<6 后重启 ComfyUI。"
        ) from exc
    return cv2


def image_batch(value, name="IMAGE") -> torch.Tensor:
    result = value.detach().cpu().float() if hasattr(value, "detach") else torch.as_tensor(value, dtype=torch.float32)
    if result.ndim == 3:
        result = result.unsqueeze(0)
    if result.ndim != 4 or result.shape[0] < 1:
        raise ValueError(f"{name} 必须是非空批次 [B, H, W, C]")
    channels = int(result.shape[-1])
    if channels == 1:
        result = result.repeat(1, 1, 1, 3)
    elif channels >= 3:
        result = result[..., :3]
    else:
        raise ValueError(f"{name} 不支持 {channels} 个通道")
    return result.clamp(0.0, 1.0)


def mask_batch(value, name="MASK") -> torch.Tensor:
    result = value.detach().cpu().float() if hasattr(value, "detach") else torch.as_tensor(value, dtype=torch.float32)
    if result.ndim == 2:
        result = result.unsqueeze(0)
    if result.ndim == 4 and result.shape[-1] == 1:
        result = result[..., 0]
    if result.ndim != 3 or result.shape[0] < 1:
        raise ValueError(f"{name} 必须是非空批次 [B, H, W]")
    return result.clamp(0.0, 1.0)


def broadcast_tensors(*values: torch.Tensor) -> tuple[torch.Tensor, ...]:
    counts = [int(value.shape[0]) for value in values]
    target = max(counts)
    if any(count not in (1, target) for count in counts):
        raise ValueError(f"批次数量无法匹配：{counts}；只允许等长批次或单帧广播")
    return tuple(value.repeat((target,) + (1,) * (value.ndim - 1)) if value.shape[0] == 1 and target > 1 else value for value in values)


def _resize_image(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (array.shape[1], array.shape[0]) == size:
        return array
    cv2 = _cv2()
    return cv2.resize(array, size, interpolation=cv2.INTER_CUBIC).astype(np.float32)


def _resize_mask(array: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    if (array.shape[1], array.shape[0]) == size:
        return array
    return _cv2().resize(array, size, interpolation=_cv2().INTER_LINEAR).astype(np.float32)


def _blur(array: np.ndarray, radius: float) -> np.ndarray:
    if radius <= 0:
        return array
    return _cv2().GaussianBlur(array, (0, 0), max(0.01, float(radius)))


def _morph(mask: np.ndarray, amount: int) -> np.ndarray:
    if amount == 0:
        return mask
    cv2 = _cv2()
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (abs(int(amount)) * 2 + 1,) * 2)
    operation = cv2.dilate if amount > 0 else cv2.erode
    return operation(mask, kernel)


def _to_tensor(arrays: Iterable[np.ndarray]) -> torch.Tensor:
    return torch.from_numpy(np.stack([np.clip(a, 0.0, 1.0).astype(np.float32) for a in arrays]))


def refine_alpha(mask: np.ndarray, grow_shrink=0, feather=0.0, denoise=0, fill_holes=False) -> np.ndarray:
    cv2 = _cv2()
    out = np.clip(mask.astype(np.float32), 0.0, 1.0)
    if int(denoise) > 0:
        k = int(denoise) * 2 + 1
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        out = cv2.morphologyEx(out, cv2.MORPH_OPEN, kernel)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    if fill_holes:
        binary = (out >= 0.5).astype(np.uint8)
        padded = np.pad(binary, 1, mode="constant", constant_values=0)
        flood = padded.copy()
        flood_mask = np.zeros((padded.shape[0] + 2, padded.shape[1] + 2), np.uint8)
        cv2.floodFill(flood, flood_mask, (0, 0), 1)
        holes = (1 - flood)[1:-1, 1:-1]
        out = np.maximum(out, holes.astype(np.float32))
    out = _morph(out, int(grow_shrink))
    return np.clip(_blur(out, float(feather)), 0.0, 1.0)


def difference_key(current, clean_background, color_space, threshold, softness,
                   denoise, grow_shrink, feather, invert):
    current = image_batch(current, "当前图片")
    clean_background = image_batch(clean_background, "背景图片")
    current, clean_background = broadcast_tensors(current, clean_background)
    foregrounds, masks, backgrounds, differences = [], [], [], []
    cv2 = _cv2()
    for current_t, background_t in zip(current, clean_background):
        source = current_t.numpy()
        reference = _resize_image(background_t.numpy(), (source.shape[1], source.shape[0]))
        if color_space == "RGB":
            delta = np.sqrt(np.mean((source - reference) ** 2, axis=2))
        elif color_space == "Lab":
            source_lab = cv2.cvtColor(source, cv2.COLOR_RGB2LAB)
            ref_lab = cv2.cvtColor(reference, cv2.COLOR_RGB2LAB)
            delta = np.linalg.norm(source_lab - ref_lab, axis=2) / 180.0
        elif color_space == "亮度":
            delta = np.abs(np.sum((source - reference) * np.array([0.2126, 0.7152, 0.0722]), axis=2))
        else:
            raise ValueError(f"未知差异空间：{color_space}")
        soft = max(1e-6, float(softness))
        alpha = np.clip((delta - float(threshold)) / soft, 0.0, 1.0) if softness > 0 else (delta >= threshold).astype(np.float32)
        alpha = refine_alpha(alpha, grow_shrink, feather, denoise)
        if invert:
            alpha = 1.0 - alpha
        foregrounds.append(source * alpha[..., None])
        masks.append(alpha)
        backgrounds.append(1.0 - alpha)
        differences.append(np.repeat(np.clip(delta, 0, 1)[..., None], 3, axis=2))
    return _to_tensor(foregrounds), _to_tensor(masks), _to_tensor(backgrounds), _to_tensor(differences)


def matte_finesse(images, masks, black_point, white_point, edge_contrast, fill_holes,
                  grow_shrink, feather, detail_recovery, despill_color, despill_strength):
    images = image_batch(images)
    masks = mask_batch(masks)
    images, masks = broadcast_tensors(images, masks)
    foregrounds, alphas, backgrounds, edges = [], [], [], []
    color = np.asarray(despill_color, dtype=np.float32) / 255.0
    for image_t, mask_t in zip(images, masks):
        image = image_t.numpy()
        original = _resize_mask(mask_t.numpy(), (image.shape[1], image.shape[0]))
        if white_point <= black_point:
            raise ValueError("白场必须大于黑场")
        alpha = np.clip((original - black_point) / (white_point - black_point), 0, 1)
        if edge_contrast != 1.0:
            alpha = np.clip((alpha - 0.5) * edge_contrast + 0.5, 0, 1)
        alpha = refine_alpha(alpha, grow_shrink, feather, 0, fill_holes)
        edge = np.clip(_morph(alpha, 1) - _morph(alpha, -1), 0, 1)
        if detail_recovery > 0:
            alpha = np.clip(alpha * (1.0 - edge * detail_recovery) + original * edge * detail_recovery, 0, 1)
        corrected = image.copy()
        if despill_strength > 0:
            direction = color - color.mean()
            power = float(np.dot(direction, direction))
            if power > 1e-8:
                chroma = corrected - corrected.mean(axis=2, keepdims=True)
                excess = np.maximum(np.sum(chroma * direction, axis=2) / power, 0.0)
                corrected = np.clip(corrected - excess[..., None] * direction * edge[..., None] * despill_strength, 0, 1)
        foregrounds.append(corrected * alpha[..., None])
        alphas.append(alpha); backgrounds.append(1.0 - alpha); edges.append(edge)
    return _to_tensor(foregrounds), _to_tensor(alphas), _to_tensor(backgrounds), _to_tensor(edges)


def _blend(base, top, mode):
    if mode == "normal": return top
    if mode == "multiply": return base * top
    if mode == "screen": return 1.0 - (1.0 - base) * (1.0 - top)
    if mode == "overlay": return np.where(base <= 0.5, 2 * base * top, 1 - 2 * (1 - base) * (1 - top))
    raise ValueError(f"未知混合模式：{mode}")


def light_wrap(foregrounds, backgrounds, masks, width, highlight_threshold,
               color_bleed, strength, blur, inner_ratio):
    foregrounds = image_batch(foregrounds, "前景")
    backgrounds = image_batch(backgrounds, "背景")
    masks = mask_batch(masks, "前景 MASK")
    foregrounds, backgrounds, masks = broadcast_tensors(foregrounds, backgrounds, masks)
    composites, wrapped_outputs, wrap_masks = [], [], []
    for fg_t, bg_t, mask_t in zip(foregrounds, backgrounds, masks):
        bg = bg_t.numpy(); size = (bg.shape[1], bg.shape[0])
        fg = _resize_image(fg_t.numpy(), size); alpha = _resize_mask(mask_t.numpy(), size)
        if strength <= 0 or not np.any(alpha > 0):
            composites.append(bg); wrapped_outputs.append(fg); wrap_masks.append(np.zeros_like(alpha)); continue
        outer = np.clip(_morph(alpha, max(1, int(width))) - alpha, 0, 1)
        inner = np.clip(alpha - _morph(alpha, -max(1, int(width))), 0, 1)
        band = np.clip(outer * (1.0 - inner_ratio) + inner * inner_ratio, 0, 1)
        luminance = np.sum(bg * np.array([0.2126, 0.7152, 0.0722]), axis=2)
        highlights = np.clip((luminance - highlight_threshold) / max(1e-6, 1 - highlight_threshold), 0, 1)
        wrap_mask = np.clip(_blur(band * highlights, blur) * strength, 0, 1)
        sampled = _blur(bg, max(0.1, blur))
        tint = sampled * color_bleed + np.repeat(luminance[..., None], 3, axis=2) * (1 - color_bleed)
        wrapped = fg * (1 - wrap_mask[..., None]) + tint * wrap_mask[..., None]
        composite = bg * (1 - alpha[..., None]) + wrapped * alpha[..., None]
        composites.append(composite); wrapped_outputs.append(wrapped); wrap_masks.append(wrap_mask)
    return _to_tensor(composites), _to_tensor(wrapped_outputs), _to_tensor(wrap_masks)


def depth_merge(image_a, image_b, depth_a, depth_b, depth_mode, depth_offset, edge_softness, antialias):
    image_a = image_batch(image_a, "图一"); image_b = image_batch(image_b, "图二")
    depth_a = mask_batch(depth_a, "图一深度"); depth_b = mask_batch(depth_b, "图二深度")
    image_a, image_b, depth_a, depth_b = broadcast_tensors(image_a, image_b, depth_a, depth_b)
    outputs, selections, differences = [], [], []
    for a_t, b_t, da_t, db_t in zip(image_a, image_b, depth_a, depth_b):
        b = b_t.numpy(); size = (b.shape[1], b.shape[0]); a = _resize_image(a_t.numpy(), size)
        da = _resize_mask(da_t.numpy(), size); db = _resize_mask(db_t.numpy(), size)
        if depth_mode == "黑近": da, db = 1 - da, 1 - db
        elif depth_mode != "白近": raise ValueError(f"未知深度方向：{depth_mode}")
        delta = da + depth_offset - db
        selection = np.clip(delta / max(1e-6, edge_softness) + 0.5, 0, 1) if edge_softness > 0 else (delta >= 0).astype(np.float32)
        if antialias > 0: selection = np.clip(_blur(selection, antialias), 0, 1)
        outputs.append(a * selection[..., None] + b * (1 - selection[..., None]))
        selections.append(selection); differences.append(np.repeat(np.clip(delta * 0.5 + 0.5, 0, 1)[..., None], 3, axis=2))
    return _to_tensor(outputs), _to_tensor(selections), _to_tensor(differences)


def corner_pin(foregrounds, backgrounds, masks, corners, opacity, blend_mode, feather):
    foregrounds = image_batch(foregrounds, "前景"); backgrounds = image_batch(backgrounds, "背景")
    masks = mask_batch(masks, "前景 MASK") if masks is not None else torch.ones((1, foregrounds.shape[1], foregrounds.shape[2]))
    foregrounds, backgrounds, masks = broadcast_tensors(foregrounds, backgrounds, masks)
    outputs, layers, output_masks = [], [], []
    cv2 = _cv2()
    for fg_t, bg_t, mask_t in zip(foregrounds, backgrounds, masks):
        bg = bg_t.numpy(); fg = fg_t.numpy(); h, w = bg.shape[:2]; sh, sw = fg.shape[:2]
        src = np.float32([[0, 0], [sw - 1, 0], [sw - 1, sh - 1], [0, sh - 1]])
        dst = np.float32([[x * (w - 1), y * (h - 1)] for x, y in corners])
        matrix = cv2.getPerspectiveTransform(src, dst)
        layer = cv2.warpPerspective(fg, matrix, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT)
        source_mask = _resize_mask(mask_t.numpy(), (sw, sh))
        alpha = cv2.warpPerspective(source_mask, matrix, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
        alpha = np.clip(_blur(alpha, feather) * opacity, 0, 1)
        if opacity <= 0 or not np.any(alpha > 0): outputs.append(bg); layers.append(layer); output_masks.append(alpha); continue
        blended = _blend(bg, layer, blend_mode)
        outputs.append(bg * (1 - alpha[..., None]) + blended * alpha[..., None]); layers.append(layer); output_masks.append(alpha)
    return _to_tensor(outputs), _to_tensor(layers), _to_tensor(output_masks)


def _channel(image, token, channel_index):
    luminance = np.sum(image * np.array([0.2126, 0.7152, 0.0722]), axis=2)
    direct = {"R": image[..., 0], "G": image[..., 1], "B": image[..., 2], "A": np.ones(image.shape[:2]), "亮度": luminance, "0": np.zeros(image.shape[:2]), "1": np.ones(image.shape[:2])}
    if token in direct: return direct[token]
    return image[..., channel_index] if channel_index < 3 else np.ones(image.shape[:2])


def channel_boolean(images_a, images_b, expressions):
    images_a = image_batch(images_a, "图一"); images_b = image_batch(images_b, "图二")
    images_a, images_b = broadcast_tensors(images_a, images_b)
    outputs, alpha_masks = [], []
    for a_t, b_t in zip(images_a, images_b):
        a = a_t.numpy(); b = _resize_image(b_t.numpy(), (a.shape[1], a.shape[0])); channels = []
        for index, expression in enumerate(expressions):
            if expression.startswith("A."): value = _channel(a, expression[2:], index)
            elif expression.startswith("B."): value = _channel(b, expression[2:], index)
            elif expression in ("0", "1"): value = _channel(a, expression, index)
            else:
                av = _channel(a, "对应通道", index); bv = _channel(b, "对应通道", index)
                operations = {"A+B": av + bv, "A-B": av - bv, "B-A": bv - av, "A*B": av * bv,
                              "最小": np.minimum(av, bv), "最大": np.maximum(av, bv), "差值": np.abs(av - bv)}
                if expression not in operations: raise ValueError(f"未知通道表达式：{expression}")
                value = operations[expression]
            channels.append(np.clip(value, 0, 1))
        outputs.append(np.stack(channels[:3], axis=2)); alpha_masks.append(channels[3])
    return _to_tensor(outputs), _to_tensor(alpha_masks)


def displace(images, displacement_images, masks, channel_x, channel_y, strength_x,
             strength_y, neutral, interpolation, boundary):
    images = image_batch(images, "源图"); displacement_images = image_batch(displacement_images, "位移图")
    masks = mask_batch(masks, "MASK") if masks is not None else torch.ones((1, images.shape[1], images.shape[2]))
    images, displacement_images, masks = broadcast_tensors(images, displacement_images, masks)
    outputs, maps, affected = [], [], []
    cv2 = _cv2(); interp = cv2.INTER_LINEAR if interpolation == "双线性" else cv2.INTER_CUBIC
    borders = {"裁切": cv2.BORDER_CONSTANT, "钳制": cv2.BORDER_REPLICATE, "镜像": cv2.BORDER_REFLECT_101, "循环": cv2.BORDER_WRAP}
    if boundary not in borders: raise ValueError(f"未知边界模式：{boundary}")
    names = {"红": 0, "绿": 1, "蓝": 2, "亮度": -1}
    if channel_x not in names or channel_y not in names: raise ValueError("位移通道必须是红、绿、蓝或亮度")
    for image_t, displacement_t, mask_t in zip(images, displacement_images, masks):
        image = image_t.numpy(); h, w = image.shape[:2]; size = (w, h)
        disp = _resize_image(displacement_t.numpy(), size); mask = _resize_mask(mask_t.numpy(), size)
        def select(name): return np.sum(disp * np.array([0.2126, .7152, .0722]), axis=2) if names[name] < 0 else disp[..., names[name]]
        dx = (select(channel_x) - neutral) * strength_x * mask
        dy = (select(channel_y) - neutral) * strength_y * mask
        map_image = np.stack([np.clip(dx / max(1.0, abs(strength_x) * 2) + .5, 0, 1), np.clip(dy / max(1.0, abs(strength_y) * 2) + .5, 0, 1), np.full((h, w), .5)], axis=2)
        active = np.clip(mask * ((np.abs(dx) + np.abs(dy)) > 1e-7), 0, 1).astype(np.float32)
        if (strength_x == 0 and strength_y == 0) or not np.any(active): outputs.append(image); maps.append(map_image); affected.append(active); continue
        yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
        warped = cv2.remap(image, xx + dx.astype(np.float32), yy + dy.astype(np.float32), interp, borderMode=borders[boundary], borderValue=0)
        outputs.append(image * (1 - mask[..., None]) + warped * mask[..., None]); maps.append(map_image); affected.append(active)
    return _to_tensor(outputs), _to_tensor(maps), _to_tensor(affected)
