"""Image animation nodes for the ``TUT_Nodes/图片/动画`` menu."""

from __future__ import annotations

import json
import os
from pathlib import Path

from PIL import Image

from ...categories import IMAGE_ANIMATION
from ...core.imaging import image_tensor_to_pil_batch


GIF_COLOR_LEVELS = [f"{value} 色" for value in (2, 4, 8, 16, 32, 64, 128, 256)]

GIF_COMPRESSION_PRESETS = {
    "自定义": {
        "resize_scale": 1.0,
        "max_colors": 256,
        "frame_step": 1,
        "dither": False,
        "optimize": False,
    },
    "高画质": {
        "resize_scale": 0.85,
        "max_colors": 256,
        "frame_step": 1,
        "dither": True,
        "optimize": True,
    },
    "均衡": {
        "resize_scale": 0.75,
        "max_colors": 128,
        "frame_step": 1,
        "dither": False,
        "optimize": True,
    },
    "小体积": {
        "resize_scale": 0.5,
        "max_colors": 64,
        "frame_step": 2,
        "dither": False,
        "optimize": True,
    },
}


class TUT_SaveAnimatedGIF:
    """Save a ComfyUI IMAGE batch as an animated GIF."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE", {"tooltip": "批次中的每张图片会成为 GIF 的一帧。"}),
                "filename_prefix": (
                    "STRING",
                    {"default": "animation/ComfyUI", "tooltip": "输出文件名前缀，可包含子目录。"},
                ),
                "fps": (
                    "FLOAT",
                    {"default": 12.0, "min": 0.1, "max": 100.0, "step": 0.1},
                ),
                "loop": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 100,
                        "tooltip": "0 表示无限循环，1 表示额外循环一次。",
                    },
                ),
                "pingpong": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "正序播放后再倒序播放。"},
                ),
                "optimize": (
                    "BOOLEAN",
                    {"default": False, "tooltip": "减小文件体积，但保存速度会变慢。"},
                ),
            },
            "optional": {
                "compression_preset": (
                    list(GIF_COMPRESSION_PRESETS),
                    {
                        "default": "自定义",
                        "tooltip": "选择预设会回填下面的压缩参数，回填后仍可手动修改。",
                    },
                ),
                "resize_scale": (
                    "FLOAT",
                    {
                        "default": 1.0,
                        "min": 0.1,
                        "max": 1.0,
                        "step": 0.05,
                        "tooltip": "按比例缩小宽高；1.0 保持原尺寸。",
                    },
                ),
                "max_colors": (
                    GIF_COLOR_LEVELS,
                    {
                        "default": "256 色",
                        "tooltip": "选择每帧调色板颜色档位，越小体积通常越小。",
                    },
                ),
                "frame_step": (
                    "INT",
                    {
                        "default": 1,
                        "min": 1,
                        "max": 32,
                        "step": 1,
                        "tooltip": "每隔多少帧保留一帧；会自动延长帧时长以保持总播放时间。",
                    },
                ),
                "dither": (
                    "BOOLEAN",
                    {
                        "default": False,
                        "tooltip": "降色时使用抖动改善渐变，但可能略微增大文件。",
                    },
                ),
                "save_metadata": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "display_name": "保存元数据",
                        "label_on": "保存",
                        "label_off": "不保存",
                        "tooltip": "开启时将提示词和工作流写入 GIF；关闭时不写入 metadata。",
                    },
                ),
            },
            "hidden": {"prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"},
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("filename",)
    FUNCTION = "save_gif"
    OUTPUT_NODE = True
    CATEGORY = IMAGE_ANIMATION
    DESCRIPTION = "将 IMAGE 批次按顺序保存为动画 GIF。"

    @staticmethod
    def _prepare_frames(frames, fps, pingpong, resize_scale, max_colors, frame_step, dither):
        try:
            resize_scale = float(resize_scale)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"resize_scale 必须是 0.1 到 1.0：{resize_scale!r}") from exc
        if not 0.1 <= resize_scale <= 1.0:
            raise ValueError(f"resize_scale 必须在 0.1 到 1.0 之间：{resize_scale}")

        try:
            if isinstance(max_colors, str):
                max_colors = max_colors.removesuffix("色").strip()
            max_colors = int(max_colors)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"max_colors 必须是 2 到 256 的整数：{max_colors!r}") from exc
        if not 2 <= max_colors <= 256:
            raise ValueError(f"max_colors 必须在 2 到 256 之间：{max_colors}")

        try:
            frame_step = int(frame_step)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"frame_step 必须是大于等于 1 的整数：{frame_step!r}") from exc
        if frame_step < 1:
            raise ValueError(f"frame_step 必须大于等于 1：{frame_step}")

        try:
            fps = float(fps)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"fps 必须是大于 0 的数字：{fps!r}") from exc
        if fps <= 0:
            raise ValueError(f"fps 必须大于 0：{fps}")

        frames = list(frames)
        if pingpong and len(frames) > 2:
            # Preserve the original ping-pong sequence before sampling it.
            frames = frames + frames[-2:0:-1]

        base_duration_ms = max(10, round(1000.0 / fps))
        source_frame_count = len(frames)
        sampled_indexes = range(0, source_frame_count, frame_step)
        durations = [
            base_duration_ms * min(frame_step, source_frame_count - index)
            for index in sampled_indexes
        ]
        frames = frames[::frame_step]

        if resize_scale != 1.0:
            width, height = frames[0].size
            target_size = (
                max(1, round(width * resize_scale)),
                max(1, round(height * resize_scale)),
            )
            resampling = getattr(Image, "Resampling", Image).LANCZOS
            frames = [frame.resize(target_size, resampling) for frame in frames]

        # Keep the legacy path behavior-compatible where practical. Explicit
        # quantization is only needed when the user reduces colors or enables
        # dithering; Pillow otherwise performs its normal GIF conversion.
        if max_colors < 256 or dither:
            quantize = getattr(Image, "Quantize", Image)
            dither_modes = getattr(Image, "Dither", Image)
            dither_mode = dither_modes.FLOYDSTEINBERG if dither else dither_modes.NONE
            frames = [
                frame.quantize(colors=max_colors, method=quantize.MEDIANCUT, dither=dither_mode)
                for frame in frames
            ]

        return frames, durations

    @staticmethod
    def _next_path(filename_prefix, width, height):
        import folder_paths

        output_dir = folder_paths.get_output_directory()
        full_output_folder, filename, counter, subfolder, _ = folder_paths.get_save_image_path(
            filename_prefix, output_dir, width, height
        )
        os.makedirs(full_output_folder, exist_ok=True)

        while True:
            file_name = f"{filename}_{counter:05}_.gif"
            file_path = os.path.join(full_output_folder, file_name)
            if not os.path.exists(file_path):
                return Path(file_path), file_name, subfolder
            counter += 1

    def save_gif(
        self,
        images,
        filename_prefix="animation/ComfyUI",
        fps=12.0,
        loop=0,
        pingpong=False,
        optimize=False,
        compression_preset="自定义",
        resize_scale=1.0,
        max_colors=256,
        frame_step=1,
        dither=False,
        save_metadata=True,
        prompt=None,
        extra_pnginfo=None,
    ):
        frames = image_tensor_to_pil_batch(images)
        frames, durations = self._prepare_frames(
            frames,
            fps=fps,
            pingpong=bool(pingpong),
            resize_scale=resize_scale,
            max_colors=max_colors,
            frame_step=frame_step,
            dither=bool(dither),
        )

        width, height = frames[0].size
        file_path, file_name, subfolder = self._next_path(filename_prefix, width, height)

        # Write to a temporary file first so an interrupted save never leaves
        # a corrupt GIF with the final filename.
        temp_path = file_path.with_name(file_path.stem + ".tmp.gif")
        try:
            save_options = {}
            if save_metadata:
                metadata = {}
                if prompt is not None:
                    metadata["prompt"] = prompt
                if extra_pnginfo is not None:
                    metadata.update(extra_pnginfo)
                if metadata:
                    save_options["comment"] = json.dumps(
                        metadata,
                        ensure_ascii=False,
                        separators=(",", ":"),
                    ).encode("utf-8")

            frames[0].save(
                temp_path,
                format="GIF",
                save_all=True,
                append_images=frames[1:],
                duration=durations,
                loop=int(loop),
                optimize=bool(optimize),
                disposal=2,
                **save_options,
            )
            os.replace(temp_path, file_path)
        finally:
            if temp_path.exists():
                temp_path.unlink()

        preview = {"filename": file_name, "subfolder": subfolder, "type": "output"}
        # Use a node-specific UI slot. ComfyUI's generic ``images`` preview is
        # canvas-backed in some frontend modes and therefore shows only the
        # first GIF frame; the TUT frontend renders this value in a real <img>.
        return {"ui": {"gif_preview": [preview]}, "result": (file_name,)}


NODE_CLASS_MAPPINGS = {"TUT_SaveAnimatedGIF": TUT_SaveAnimatedGIF}
NODE_DISPLAY_NAME_MAPPINGS = {"TUT_SaveAnimatedGIF": "TUT_保存动画 GIF"}
