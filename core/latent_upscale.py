"""SesquiLSR model discovery, caching, and ComfyUI LATENT handling.

Importing this module never creates directories, loads checkpoints, or uses the
network.  Checkpoints are fetched and verified only when the node executes.
"""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
import tempfile
import threading
import urllib.request

import torch

from .sesqui_lsr import (
    LatentUpscaler,
    make_flux,
    make_flux2,
    make_identity,
    make_ideogram4,
    make_sdxl,
)


UPSTREAM_COMMIT = "befae004248c403f38b76b9f65fd43b901ea3eaa"
_RAW_MODEL_BASE = (
    "https://raw.githubusercontent.com/LoganBooker/SesquiLSR/"
    f"{UPSTREAM_COMMIT}/models"
)

# SHA256 values are pinned to the exact blobs at UPSTREAM_COMMIT.  They are
# filled from the verified upstream files during integration.
MODEL_INFO = {
    "upscaler_SDXL.safetensors": {
        "size": 12_293_696,
        "sha256": "392c0e36fc0389a170cf6c1682556972d91d750c31aa392cdad1b13cad4e18e4",
    },
    "upscaler_Flux.safetensors": {
        "size": 12_374_768,
        "sha256": "dbc6362c910a179a22eca26f656fdacde17eded876ee8f05b8f5b73630e4ddbe",
    },
    "upscaler_Flux2.safetensors": {
        "size": 12_525_400,
        "sha256": "f385ff2250b3f2c72ad422768558414d617b9a006cc0b1ce37a296cc99d17177",
    },
    "upscaler_Wan21.safetensors": {
        "size": 12_373_048,
        "sha256": "d1b4c2c1d80b2af81e6bbfd38501ffb5adffb9f7c0ed2599555b837d0d2d015a",
    },
}
for _filename, _info in MODEL_INFO.items():
    _info["url"] = f"{_RAW_MODEL_BASE}/{_filename}"


FORMAT_CONFIG = {
    "SDXL": {
        "model_file": "upscaler_SDXL.safetensors",
        "in_channels": 4,
        "adaptor_fn": make_sdxl,
    },
    "Flux": {
        "model_file": "upscaler_Flux.safetensors",
        "in_channels": 16,
        "adaptor_fn": make_flux,
    },
    "Flux2": {
        "model_file": "upscaler_Flux2.safetensors",
        "in_channels": 32,
        "adaptor_fn": make_flux2,
    },
    "Ideogram 4": {
        "model_file": "upscaler_Flux2.safetensors",
        "in_channels": 32,
        "adaptor_fn": make_ideogram4,
    },
    "Wan 2.1": {
        "model_file": "upscaler_Wan21.safetensors",
        "in_channels": 16,
        # ComfyUI exposes these formats after latent_format.process_out(), so
        # a LATENT node receives raw VAE latents here.
        "adaptor_fn": lambda: make_identity(16),
    },
}

_download_lock = threading.Lock()
_active_key = None
_active_model = None
_adaptor_cache = {}


def _model_directory() -> Path:
    try:
        import folder_paths

        base = Path(folder_paths.models_dir)
    except (ImportError, AttributeError):
        base = Path(__file__).resolve().parents[2] / "models"
    return base / "TUT_Nodes" / "sesqui_lsr"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _verify_model_file(path: Path, info: dict) -> None:
    actual_size = path.stat().st_size
    if actual_size != info["size"]:
        raise RuntimeError(
            f"模型文件大小校验失败：{path}（期望 {info['size']} 字节，实际 {actual_size} 字节）"
        )
    actual_hash = _sha256(path)
    if actual_hash != info["sha256"]:
        raise RuntimeError(
            f"模型文件 SHA256 校验失败：{path}（期望 {info['sha256']}，实际 {actual_hash}）"
        )


def ensure_sesqui_model(filename: str) -> Path:
    """Return a verified checkpoint, downloading it only on first execution."""
    if filename not in MODEL_INFO:
        raise ValueError(f"未知 SesquiLSR 模型文件：{filename}")
    info = MODEL_INFO[filename]
    directory = _model_directory()
    target = directory / filename
    if target.is_file():
        _verify_model_file(target, info)
        return target

    with _download_lock:
        if target.is_file():
            _verify_model_file(target, info)
            return target
        directory.mkdir(parents=True, exist_ok=True)
        temporary = None
        try:
            handle, name = tempfile.mkstemp(
                prefix=f".{filename}.", suffix=".tmp", dir=directory
            )
            os.close(handle)
            temporary = Path(name)
            urllib.request.urlretrieve(info["url"], temporary)
            _verify_model_file(temporary, info)
            os.replace(temporary, target)
            temporary = None
            return target
        except Exception as exc:
            if temporary is not None:
                temporary.unlink(missing_ok=True)
            if isinstance(exc, RuntimeError):
                raise
            raise RuntimeError(
                f"SesquiLSR 模型下载失败，可手动下载到 {target}：{exc}"
            ) from exc


def get_torch_device() -> torch.device:
    """Use ComfyUI's selected device, with a standalone-test fallback."""
    try:
        from comfy import model_management

        return torch.device(model_management.get_torch_device())
    except (ImportError, AttributeError, TypeError):
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def resolve_dtype(half_precision: bool, device: torch.device) -> torch.dtype:
    if not half_precision or device.type != "cuda":
        return torch.float32
    return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16


def _format_channel_hint(channels: int) -> str:
    return {
        4: "SDXL",
        16: "Flux 或 Wan 2.1",
        32: "Flux2 VAE 空间",
        128: "Flux2/Ideogram 4 打包空间",
    }.get(channels, f"{channels} 通道")


def flatten_to_4d(z: torch.Tensor, expected_channels: int):
    """Flatten supported image/video layouts and return a restoration tag."""
    if z.ndim == 4:
        if z.shape[1] != expected_channels:
            raise ValueError(
                f"LATENT 应为 {expected_channels} 通道（{_format_channel_hint(expected_channels)}），"
                f"实际形状为 {tuple(z.shape)}"
            )
        return z, ("4d",)
    if z.ndim != 5:
        raise ValueError(f"LATENT 必须是 4D 或 5D，实际形状为 {tuple(z.shape)}")
    if z.shape[1] == expected_channels:
        b, c, t, h, w = z.shape
        return z.permute(0, 2, 1, 3, 4).reshape(b * t, c, h, w), (
            "bcthw", (b, c, t)
        )
    if z.shape[2] == expected_channels:
        b, t, c, h, w = z.shape
        return z.reshape(b * t, c, h, w), ("btchw", (b, t, c))
    raise ValueError(
        f"无法在 LATENT 形状 {tuple(z.shape)} 中找到 {expected_channels} 通道轴"
    )


def restore_from_4d(z: torch.Tensor, layout):
    kind = layout[0]
    if kind == "4d":
        return z
    if kind == "bcthw":
        b, c, t = layout[1]
        return z.reshape(b, t, c, *z.shape[-2:]).permute(0, 2, 1, 3, 4)
    if kind == "btchw":
        b, t, c = layout[1]
        return z.reshape(b, t, c, *z.shape[-2:])
    raise ValueError(f"未知 LATENT 布局：{kind!r}")


def resize_noise_mask(mask: torch.Tensor, target_hw):
    if mask.shape[-2:] == tuple(target_hw):
        return mask
    flat = mask.reshape(-1, 1, *mask.shape[-2:]).float()
    resized = torch.nn.functional.interpolate(flat, size=target_hw, mode="nearest")
    return resized.reshape(*mask.shape[:-2], *target_hw).to(
        device=mask.device, dtype=mask.dtype
    )


def clear_model_cache() -> None:
    global _active_key, _active_model
    _active_key = None
    _active_model = None
    _adaptor_cache.clear()


def load_model(model_format: str, dtype: torch.dtype, device: torch.device):
    global _active_key, _active_model
    if model_format not in FORMAT_CONFIG:
        raise ValueError(f"未知模型格式：{model_format}")
    config = FORMAT_CONFIG[model_format]
    path = ensure_sesqui_model(config["model_file"])
    key = (model_format, dtype, str(device), str(path.resolve()))
    if _active_key != key or _active_model is None:
        try:
            from safetensors.torch import load_file
        except ImportError as exc:
            raise RuntimeError("缺少 safetensors，无法加载 SesquiLSR 模型。") from exc
        try:
            state_dict = load_file(str(path), device="cpu")
            model = LatentUpscaler(in_channels=config["in_channels"])
            model.load_state_dict(state_dict, strict=True)
            model.to(device=device, dtype=dtype).eval().requires_grad_(False)
        except Exception as exc:
            raise RuntimeError(f"SesquiLSR 模型加载失败：{path}：{exc}") from exc
        _active_model = model
        _active_key = key
    if model_format not in _adaptor_cache:
        _adaptor_cache[model_format] = config["adaptor_fn"]()
    return _active_model, _adaptor_cache[model_format]


def upscale_latent(latent: dict, model_format: str, scale: float,
                   half_precision: bool):
    if not isinstance(latent, dict) or "samples" not in latent:
        raise ValueError("输入必须是包含 samples 的 LATENT")
    samples = latent["samples"]
    if not isinstance(samples, torch.Tensor):
        raise ValueError("LATENT.samples 必须是 Torch Tensor")
    if samples.ndim not in (4, 5):
        raise ValueError(f"LATENT 必须是 4D 或 5D，实际形状为 {tuple(samples.shape)}")
    if min(samples.shape[-2:]) < 3:
        raise ValueError("LATENT 的宽和高都必须至少为 3，才能使用反射填充")
    scale = float(scale)
    if not 1.0 <= scale <= 2.0:
        raise ValueError(f"缩放倍率必须在 1.0 到 2.0 之间，实际为 {scale}")
    if model_format not in FORMAT_CONFIG:
        raise ValueError(f"未知模型格式：{model_format}")

    device = get_torch_device()
    dtype = resolve_dtype(bool(half_precision), device)
    model, adaptor = load_model(model_format, dtype, device)
    height, width = samples.shape[-2:]
    target_hw = (round(height * scale), round(width * scale))
    samples_4d, layout = flatten_to_4d(samples, adaptor.external_channels)
    samples_fp32 = samples_4d.to(device=device, dtype=torch.float32)
    vae_latent = adaptor.to_vae_latent(samples_fp32)
    vae_target = adaptor.vae_target_size(target_hw)
    with torch.no_grad():
        upscaled = model(vae_latent.to(dtype=dtype), vae_target)
    result_4d = adaptor.from_vae_latent(upscaled.float()).to(
        device=samples.device, dtype=samples.dtype
    )
    result = restore_from_4d(result_4d, layout)
    output = dict(latent)
    output["samples"] = result
    if "noise_mask" in latent:
        output["noise_mask"] = resize_noise_mask(latent["noise_mask"], result.shape[-2:])
    return output
