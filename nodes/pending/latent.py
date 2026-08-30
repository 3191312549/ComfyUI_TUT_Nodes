"""Latent-space upscaling node."""

from __future__ import annotations

from ...categories import LATENT_UPSCALING
from ...core.latent_upscale import upscale_latent


class TUT_SesquiLatentUpscale:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "latent": ("LATENT",),
            "model_format": (
                ["SDXL", "Flux", "Flux2", "Ideogram 4", "Wan 2.1"],
                {
                    "default": "SDXL",
                    "tooltip": (
                        "选择与输入 LATENT 匹配的格式：SDXL；Flux/Z-Image/Lumina；"
                        "Flux2；Ideogram 4；或 Wan 2.x/Krea 2/Anima/Qwen Image。"
                    ),
                },
            ),
            "scale": (
                "FLOAT",
                {"default": 1.5, "min": 1.0, "max": 2.0, "step": 0.05},
            ),
            "half_precision": ("BOOLEAN", {"default": True}),
        }}

    RETURN_TYPES = ("LATENT",)
    RETURN_NAMES = ("latent",)
    FUNCTION = "upscale"
    CATEGORY = LATENT_UPSCALING
    DESCRIPTION = (
        "使用 SesquiLSR 学习式潜空间放大器，将 SDXL、Flux、Flux2、"
        "Ideogram 4 或 Wan 系列 LATENT 放大 1.0–2.0 倍。首次使用某格式会下载对应模型。"
    )

    def upscale(self, latent, model_format, scale, half_precision):
        try:
            return (upscale_latent(latent, model_format, scale, half_precision),)
        except Exception as exc:
            raise ValueError(
                f"SesquiLSR 潜空间放大失败 [{model_format}]：{exc} "
                "（SDXL=4通道；Flux/Wan=16通道；Flux2/Ideogram 4=128通道打包 LATENT）"
            ) from exc


NODE_CLASS_MAPPINGS = {
    "TUT_SesquiLatentUpscale": TUT_SesquiLatentUpscale,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_SesquiLatentUpscale": "TUT_SesquiLSR潜空间放大",
}
