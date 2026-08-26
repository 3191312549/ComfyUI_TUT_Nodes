"""Shared color, mask-refinement, and foreground helpers for keying nodes."""

from __future__ import annotations

import importlib
from typing import Iterable

import numpy as np
import torch
from PIL import Image, ImageFilter


KEY_COLOR_MAP = {
    "green": (0, 255, 0),
    "blue": (0, 0, 255),
    "white": (255, 255, 255),
    "black": (0, 0, 0),
}

REMBG_MODELS = (
    "birefnet-general-lite",
    "birefnet-general",
    "birefnet-portrait",
    "isnet-general-use",
    "isnet-anime",
    "u2net",
    "u2net_human_seg",
)

_REMBG_SESSIONS = {}


def image_tensor(images) -> torch.Tensor:
    """Return a validated float32 IMAGE batch without quantizing its values."""
    result = images.detach().cpu().float() if hasattr(images, "detach") else torch.as_tensor(images, dtype=torch.float32)
    if result.ndim == 3:
        result = result.unsqueeze(0)
    if result.ndim != 4 or result.shape[0] == 0:
        raise ValueError("IMAGE 必须是非空批次 [B, H, W, C]")
    channels = int(result.shape[-1])
    if channels == 1:
        result = result.repeat(1, 1, 1, 3)
    elif channels >= 3:
        result = result[..., :3]
    else:
        raise ValueError(f"不支持 {channels} 通道 IMAGE")
    return result.clamp(0.0, 1.0)


def mask_tensor(masks) -> torch.Tensor:
    result = masks.detach().cpu().float() if hasattr(masks, "detach") else torch.as_tensor(masks, dtype=torch.float32)
    if result.ndim == 2:
        result = result.unsqueeze(0)
    if result.ndim == 4 and result.shape[-1] == 1:
        result = result[..., 0]
    if result.ndim != 3 or result.shape[0] == 0:
        raise ValueError("MASK 必须是非空批次 [B, H, W]")
    return result.clamp(0.0, 1.0)


def broadcast_image_mask(images, masks) -> tuple[torch.Tensor, torch.Tensor]:
    images = image_tensor(images)
    masks = mask_tensor(masks)
    image_count, mask_count = int(images.shape[0]), int(masks.shape[0])
    target = max(image_count, mask_count)
    if image_count not in (1, target) or mask_count not in (1, target):
        raise ValueError(f"IMAGE 与 MASK 批次数量无法匹配：[{image_count}, {mask_count}]；只允许等长或单帧广播")
    if image_count == 1 and target > 1:
        images = images.repeat(target, 1, 1, 1)
    if mask_count == 1 and target > 1:
        masks = masks.repeat(target, 1, 1)
    height, width = int(images.shape[1]), int(images.shape[2])
    if tuple(masks.shape[1:]) != (height, width):
        masks = torch.nn.functional.interpolate(
            masks.unsqueeze(1), size=(height, width), mode="bilinear", align_corners=False
        ).squeeze(1)
    return images, masks.clamp(0.0, 1.0)


def _rgb_to_lab(rgb) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float32) / 255.0
    linear = np.where(values <= 0.04045, values / 12.92, ((values + 0.055) / 1.055) ** 2.4)
    xyz = np.matmul(
        linear,
        np.asarray(
            [[0.4124564, 0.3575761, 0.1804375],
             [0.2126729, 0.7151522, 0.0721750],
             [0.0193339, 0.1191920, 0.9503041]],
            dtype=np.float32,
        ).T,
    )
    xyz = xyz / np.asarray([0.95047, 1.0, 1.08883], dtype=np.float32)
    delta = 6.0 / 29.0
    converted = np.where(xyz > delta ** 3, np.cbrt(xyz), xyz / (3 * delta ** 2) + 4.0 / 29.0)
    return np.stack(
        (116.0 * converted[..., 1] - 16.0,
         500.0 * (converted[..., 0] - converted[..., 1]),
         200.0 * (converted[..., 1] - converted[..., 2])),
        axis=-1,
    )


def color_key_mask(images, key_rgb, tolerance=20.0, softness=15.0) -> torch.Tensor:
    images = image_tensor(images)
    rgb = images.numpy() * 255.0
    difference = np.linalg.norm(_rgb_to_lab(rgb) - _rgb_to_lab(np.asarray(key_rgb, dtype=np.float32)), axis=-1)
    tolerance = max(0.0, float(tolerance))
    softness = max(0.0, float(softness))
    if softness <= 1e-6:
        alpha = (difference > tolerance).astype(np.float32)
    else:
        alpha = np.clip((difference - tolerance) / softness, 0.0, 1.0)
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)
    return torch.from_numpy(alpha.astype(np.float32))


def refine_mask(masks, grow_shrink=0, edge_feather=0.0, invert_mask=False) -> torch.Tensor:
    masks = mask_tensor(masks)
    amount = max(-32, min(32, int(grow_shrink)))
    feather = max(0.0, min(64.0, float(edge_feather)))
    if amount == 0 and feather == 0.0:
        return (1.0 - masks if bool(invert_mask) else masks).float()
    output = []
    for mask in masks.numpy():
        pil_mask = Image.fromarray(np.clip(mask * 255.0, 0, 255).round().astype(np.uint8), "L")
        if amount > 0:
            pil_mask = pil_mask.filter(ImageFilter.MaxFilter(amount * 2 + 1))
        elif amount < 0:
            pil_mask = pil_mask.filter(ImageFilter.MinFilter(abs(amount) * 2 + 1))
        if feather > 0:
            pil_mask = pil_mask.filter(ImageFilter.GaussianBlur(feather))
        values = np.asarray(pil_mask, dtype=np.float32) / 255.0
        output.append(1.0 - values if bool(invert_mask) else values)
    return torch.from_numpy(np.stack(output).astype(np.float32)).clamp(0.0, 1.0)


def suppress_spill(images, masks, key_rgb, strength=0.0) -> torch.Tensor:
    images, masks = broadcast_image_mask(images, masks)
    strength = max(0.0, min(1.0, float(strength)))
    if strength == 0.0:
        return images
    key = torch.as_tensor(key_rgb, dtype=images.dtype) / 255.0
    direction = key - key.mean()
    direction_power = torch.sum(direction * direction)
    if float(direction_power) <= 1e-8:
        return images
    image_chroma = images - images.mean(dim=-1, keepdim=True)
    excess = torch.clamp(torch.sum(image_chroma * direction, dim=-1) / direction_power, min=0.0)
    edge_weight = (1.0 - masks).clamp(0.0, 1.0)
    correction = excess.unsqueeze(-1) * direction * edge_weight.unsqueeze(-1) * strength
    return torch.clamp(images - correction, 0.0, 1.0)


def foreground_outputs(images, masks, *, spill_color=None, spill_strength=0.0):
    images, masks = broadcast_image_mask(images, masks)
    if spill_color is not None:
        images = suppress_spill(images, masks, spill_color, spill_strength)
    foreground = images * masks.unsqueeze(-1)
    return foreground.float(), masks.float(), (1.0 - masks).float()


def pil_images(images) -> Iterable[Image.Image]:
    for image in image_tensor(images).numpy():
        yield Image.fromarray(np.clip(image * 255.0, 0, 255).round().astype(np.uint8), "RGB")


def _provider_candidates(provider: str):
    provider = str(provider).lower()
    if provider == "cpu":
        return [("cpu", ["CPUExecutionProvider"])]
    try:
        onnxruntime = importlib.import_module("onnxruntime")
        available = set(onnxruntime.get_available_providers())
    except Exception as exc:
        raise RuntimeError(f"无法读取 ONNX Runtime 运行设备：{exc}") from exc
    if provider == "cuda":
        if "CUDAExecutionProvider" not in available:
            raise RuntimeError("当前 ONNX Runtime 不支持 CUDA，请选择 auto 或 cpu")
        return [("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"])]
    candidates = []
    if "CUDAExecutionProvider" in available:
        candidates.append(("cuda", ["CUDAExecutionProvider", "CPUExecutionProvider"]))
    candidates.append(("cpu", ["CPUExecutionProvider"]))
    return candidates


def _rembg_session(model: str, provider: str):
    if model not in REMBG_MODELS:
        raise ValueError(f"不支持的 rembg 模型：{model}")
    try:
        rembg = importlib.import_module("rembg")
    except Exception as exc:
        raise RuntimeError("AI 抠像需要 rembg。请在 ComfyUI 的 Python 环境中安装 rembg 后重试") from exc

    errors = []
    for resolved_provider, providers in _provider_candidates(provider):
        cache_key = (model, resolved_provider)
        if cache_key in _REMBG_SESSIONS:
            return rembg, _REMBG_SESSIONS[cache_key]
        try:
            session = rembg.new_session(model, providers=providers)
        except Exception as exc:
            errors.append(f"{resolved_provider}: {exc}")
            if str(provider).lower() != "auto":
                break
            continue
        _REMBG_SESSIONS[cache_key] = session
        return rembg, session
    detail = "; ".join(errors) or "未知错误"
    raise RuntimeError(f"无法加载或下载 rembg 模型 {model}：{detail}")


def rembg_masks(images, model: str, provider: str) -> torch.Tensor:
    rembg, session = _rembg_session(str(model), str(provider))
    output = []
    for image in pil_images(images):
        try:
            result = rembg.remove(image, session=session, only_mask=True, post_process_mask=False)
        except Exception as exc:
            raise RuntimeError(f"rembg 模型 {model} 执行失败：{exc}") from exc
        if isinstance(result, bytes):
            from io import BytesIO
            result = Image.open(BytesIO(result))
        if not isinstance(result, Image.Image):
            result = Image.fromarray(np.asarray(result).astype(np.uint8))
        if result.size != image.size:
            result = result.resize(image.size, getattr(Image, "Resampling", Image).BILINEAR)
        output.append(np.asarray(result.convert("L"), dtype=np.float32) / 255.0)
    return torch.from_numpy(np.stack(output).astype(np.float32)).clamp(0.0, 1.0)


def clear_rembg_session_cache():
    """Test/support hook; normal users benefit from process-lifetime sessions."""
    _REMBG_SESSIONS.clear()
