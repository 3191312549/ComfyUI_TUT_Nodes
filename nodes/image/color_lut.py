"""RGB curve and LUT nodes for the public plugin build."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from ...categories import IMAGE_COLOR
from ...core.color import exact_or_composite, prepare_single, tensors
from ...core.color_lut import (
    apply_lut_data,
    apply_curves,
    curves_are_identity,
    list_lut_files,
    load_lut_data,
    parse_curve_data,
    resolve_lut_path,
)


CURVES_HELP = "TUT_Nodes/图片/调色"
LUT_HELP = "TUT_Nodes/图片/调色"
IDENTITY_POINTS = [{"x": 0.0, "y": 0.0}, {"x": 1.0, "y": 1.0}]
DEFAULT_CURVES = json.dumps(
    {"version": 1, "channels": {channel: IDENTITY_POINTS for channel in ("RGB", "R", "G", "B")}},
    ensure_ascii=False,
    separators=(",", ":"),
)
def _slider(default, minimum, maximum, step):
    return {
        "default": default,
        "min": minimum,
        "max": maximum,
        "step": step,
        "display": "slider",
    }


def _prepared_bypass(image, frames, masks):
    output = exact_or_composite(image, frames, masks, 0.0)
    _, zero_mask = tensors(frames, (np.zeros_like(masks, dtype=np.float32),))
    return output, zero_mask


def _effect_output(original, rendered, masks, strength):
    output = exact_or_composite(original, np.stack(rendered), masks, strength)
    _, effect_mask = tensors(np.stack(rendered), (masks * float(strength),))
    return output, effect_mask


class TUT_ColorCurves:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "curve_data": ("STRING", {"default": DEFAULT_CURVES, "multiline": False}),
                "interpolation": (["单调三次"],),
                "strength": ("FLOAT", _slider(1.0, 0.0, 1.0, 0.01)),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "effect_mask")
    FUNCTION = "adjust_curves"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "使用 RGB 总曲线和逐通道曲线进行高精度亮度与色彩调整。"

    def adjust_curves(self, image, curve_data, interpolation, strength, mask=None):
        if interpolation != "单调三次":
            raise ValueError(f"RGB 曲线只支持单调三次插值：{interpolation!r}")
        frames, masks = prepare_single(image, mask)
        if float(strength) == 0.0 or not np.any(masks > 0.0):
            output, effect_mask = _prepared_bypass(image, frames, masks)
            return output, CURVES_HELP, effect_mask
        curves = parse_curve_data(curve_data)
        rendered = frames if curves_are_identity(curves) else np.stack(
            [apply_curves(frame, curves, interpolation) for frame in frames]
        )
        output, effect_mask = _effect_output(image, rendered, masks, strength)
        return output, CURVES_HELP, effect_mask


class TUT_LUT:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "lut_path": ("STRING", {"default": ""}),
                "strength": ("FLOAT", _slider(1.0, 0.0, 1.0, 0.01)),
            },
            "optional": {
                "mask": ("MASK",),
                "lut_data": ("TUT_LUT_DATA",),
            },
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "effect_mask")
    FUNCTION = "apply_lut"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "应用加载节点或路径提供的 1D/3D LUT；兼容既有 lut_path 工作流。"

    def apply_lut(self, image, lut_path, strength, mask=None, lut_data=None):
        frames, masks = prepare_single(image, mask)
        if float(strength) == 0.0 or not np.any(masks > 0.0):
            output, effect_mask = _prepared_bypass(image, frames, masks)
            return output, LUT_HELP, effect_mask
        data = lut_data if lut_data is not None else load_lut_data(lut_path)
        rendered = np.stack([apply_lut_data(frame, data) for frame in frames])
        output, effect_mask = _effect_output(image, rendered, masks, strength)
        return output, LUT_HELP, effect_mask


def _lut_file_options():
    return ["未选择"] + list_lut_files()


def _preview_image_options():
    try:
        import folder_paths

        root = Path(folder_paths.get_input_directory())
        extensions = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}
        files = sorted(
            path.relative_to(root).as_posix()
            for path in root.rglob("*")
            if path.is_file() and path.suffix.lower() in extensions
        )
        return ["未选择"] + files
    except (ImportError, AttributeError, OSError):
        return ["未选择"]


class TUT_LUTLoaderPreview:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lut_file": (_lut_file_options(),),
                "preview_strength": ("FLOAT", _slider(1.0, 0.0, 1.0, 0.01)),
                "preview_image_file": (_preview_image_options(), {"image_upload": True}),
            },
            "optional": {"image": ("IMAGE",)},
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ("TUT_LUT_DATA",)
    RETURN_NAMES = ("lut_data",)
    FUNCTION = "load_and_preview"
    CATEGORY = IMAGE_COLOR
    DESCRIPTION = "加载常见 LUT 文件，并用外接或上传图片在节点内滑动对比原图与效果。"
    OUTPUT_NODE = True

    @classmethod
    def VALIDATE_INPUTS(cls, lut_file, preview_image_file):
        if not lut_file or lut_file == "未选择":
            return "请选择或上传 LUT 文件"
        try:
            resolve_lut_path(lut_file)
        except ValueError as exc:
            return str(exc)
        if preview_image_file and preview_image_file != "未选择":
            try:
                import folder_paths

                if not folder_paths.exists_annotated_filepath(preview_image_file):
                    return f"找不到预览图片：{preview_image_file}"
            except (ImportError, AttributeError):
                pass
        return True

    @classmethod
    def IS_CHANGED(cls, lut_file, preview_strength, preview_image_file,
                   image=None, prompt=None, extra_pnginfo=None):
        del preview_strength, image, prompt, extra_pnginfo
        path = resolve_lut_path(lut_file)
        lut_stat = path.stat()
        parts = [str(path), str(lut_stat.st_mtime_ns), str(lut_stat.st_size)]
        if preview_image_file and preview_image_file != "未选择":
            try:
                import folder_paths

                preview_path = Path(folder_paths.get_annotated_filepath(preview_image_file))
                preview_stat = preview_path.stat()
                parts.extend((str(preview_path), str(preview_stat.st_mtime_ns), str(preview_stat.st_size)))
            except (ImportError, AttributeError, OSError):
                parts.append(str(preview_image_file))
        return "|".join(parts)

    @staticmethod
    def _load_preview_image(preview_image_file):
        if not preview_image_file or preview_image_file == "未选择":
            return None
        try:
            from nodes import LoadImage

            loaded = LoadImage().load_image(preview_image_file)[0]
        except Exception as exc:
            raise ValueError(f"无法读取预览图片 {preview_image_file!r}：{exc}") from exc
        return loaded

    @staticmethod
    def _save_preview(image, filename_prefix, prompt, extra_pnginfo):
        from nodes import PreviewImage

        result = PreviewImage().save_images(
            image[:1],
            filename_prefix=filename_prefix,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        return result.get("ui", {}).get("images", [])

    def load_and_preview(self, lut_file, preview_strength, preview_image_file,
                         image=None, prompt=None, extra_pnginfo=None):
        if not lut_file or lut_file == "未选择":
            raise ValueError("请选择或上传 LUT 文件")
        data = load_lut_data(lut_file)
        source_image = image if image is not None else self._load_preview_image(preview_image_file)
        ui = {"original_images": [], "graded_images": [], "message": ["未选择预览图片"]}
        if source_image is not None:
            if source_image.ndim != 4 or source_image.shape[0] < 1 or source_image.shape[-1] != 3:
                raise ValueError("预览图片必须是至少包含一张 RGB 图片的 IMAGE 批次")
            frame = source_image[:1].detach().cpu().numpy().astype(np.float32)
            rendered = np.stack([apply_lut_data(frame[0], data)])
            amount = float(np.clip(preview_strength, 0.0, 1.0))
            graded = torch.from_numpy(frame * (1.0 - amount) + rendered * amount).to(torch.float32)
            ui = {
                "original_images": self._save_preview(source_image, "TUT.lut.original.", prompt, extra_pnginfo),
                "graded_images": self._save_preview(graded, "TUT.lut.graded.", prompt, extra_pnginfo),
                "message": [f"已加载 {data['format'].upper()} LUT"],
            }
        return {"ui": ui, "result": (data,)}


NODE_CLASS_MAPPINGS = {
    "TUT_ColorCurves": TUT_ColorCurves,
    "TUT_LUT": TUT_LUT,
    "TUT_LUTLoaderPreview": TUT_LUTLoaderPreview,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_ColorCurves": "TUT_RGB曲线",
    "TUT_LUT": "TUT_3D LUT调色",
    "TUT_LUTLoaderPreview": "TUT_LUT加载与预览",
}
