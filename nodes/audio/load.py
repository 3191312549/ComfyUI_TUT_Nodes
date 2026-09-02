"""Advanced audio loading, metadata, and sample-accurate trimming."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch

from ...categories import AUDIO


def _folder_paths_module():
    try:
        import folder_paths
    except ImportError as exc:  # pragma: no cover - only outside ComfyUI
        raise RuntimeError("无法访问 ComfyUI 输入目录，请确认节点正在 ComfyUI 中运行。") from exc
    return folder_paths


def _audio_input_files() -> list[str]:
    """Return input audio tokens without making package import depend on ComfyUI."""

    try:
        folder_paths = _folder_paths_module()
        input_dir = folder_paths.get_input_directory()
        os.makedirs(input_dir, exist_ok=True)
        names = folder_paths.filter_files_content_types(os.listdir(input_dir), ["audio"])
        return sorted(names) or [""]
    except (RuntimeError, OSError):
        return [""]


def _resolve_audio_path(audio_file: str) -> str:
    if not str(audio_file or "").strip():
        raise ValueError("请选择或上传一个音频文件。")
    folder_paths = _folder_paths_module()
    try:
        path = folder_paths.get_annotated_filepath(audio_file)
    except (KeyError, OSError, ValueError) as exc:
        raise ValueError(f"无法定位音频文件：{audio_file}") from exc
    if not path or not os.path.isfile(path):
        raise ValueError(f"音频文件不存在：{audio_file}")
    return path


def _pcm_to_float32(waveform: torch.Tensor) -> torch.Tensor:
    if waveform.dtype.is_floating_point:
        return waveform.to(dtype=torch.float32)
    if waveform.dtype == torch.uint8:
        return (waveform.to(torch.float32) - 128.0) / 128.0
    if waveform.dtype == torch.int16:
        return waveform.to(torch.float32) / float(2**15)
    if waveform.dtype == torch.int32:
        return waveform.to(torch.float32) / float(2**31)
    raise ValueError(f"不支持的音频采样格式：{waveform.dtype}")


def _decode_audio_file(path: str) -> tuple[torch.Tensor, int]:
    """Decode the first audio stream to ``[channels, samples]`` float32 PCM."""

    try:
        import av
    except ImportError as exc:  # pragma: no cover - supplied by ComfyUI
        raise RuntimeError("当前 ComfyUI 环境缺少 PyAV，无法解码音频。") from exc

    try:
        with av.open(path) as container:
            if not container.streams.audio:
                raise ValueError("文件中没有可用的音轨。")
            stream = container.streams.audio[0]
            sample_rate = int(stream.codec_context.sample_rate or 0)
            channel_count = int(stream.codec_context.channels or 0)
            frames: list[torch.Tensor] = []

            for frame in container.decode(streams=stream.index):
                frame_rate = int(getattr(frame, "sample_rate", 0) or 0)
                if sample_rate <= 0 and frame_rate > 0:
                    sample_rate = frame_rate
                samples = torch.from_numpy(frame.to_ndarray())
                if samples.ndim == 1:
                    samples = samples.unsqueeze(0)
                elif channel_count > 0 and samples.shape[0] != channel_count:
                    if samples.numel() % channel_count != 0:
                        raise ValueError("音频帧的声道布局无法识别。")
                    samples = samples.reshape(-1, channel_count).transpose(0, 1)
                frames.append(_pcm_to_float32(samples).contiguous())

            if not frames:
                raise ValueError("音频文件没有可解码的采样。")
            if sample_rate <= 0:
                raise ValueError("无法读取音频采样率。")
            waveform = torch.cat(frames, dim=-1)
            if waveform.ndim != 2 or waveform.shape[0] <= 0:
                raise ValueError("解码后的音频声道数据无效。")
            return waveform, sample_rate
    except (ValueError, RuntimeError):
        raise
    except Exception as exc:
        raise RuntimeError(f"音频解码失败：{exc}") from exc


def _file_fingerprint(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class TUT_AdvancedAudioLoader:
    """Load one audio file and emit a sample-aligned selected range."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "audio_file": (
                    _audio_input_files(),
                    {
                        "audio_upload": True,
                        "tooltip": "选择 ComfyUI input 目录中的音频，或上传新文件。",
                    },
                ),
                "start_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1_000_000_000.0,
                        "step": 0.01,
                        "tooltip": "裁剪开始时间，单位为秒。",
                    },
                ),
                "end_time": (
                    "FLOAT",
                    {
                        "default": 0.0,
                        "min": 0.0,
                        "max": 1_000_000_000.0,
                        "step": 0.01,
                        "tooltip": "裁剪结束时间，单位为秒；0 表示音频末尾。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("AUDIO", "FLOAT", "FLOAT", "FLOAT", "INT", "INT")
    RETURN_NAMES = ("audio", "duration", "start_time", "end_time", "sample_rate", "channels")
    FUNCTION = "load_audio"
    CATEGORY = AUDIO
    DESCRIPTION = "上传或选择音频，通过波形试听并裁剪选区，输出音频及实际时长、边界、采样率和声道数。"

    @classmethod
    def VALIDATE_INPUTS(cls, audio_file, **kwargs):
        del kwargs
        try:
            _resolve_audio_path(audio_file)
        except (RuntimeError, ValueError) as exc:
            return str(exc)
        return True

    @classmethod
    def IS_CHANGED(cls, audio_file, **kwargs):
        del kwargs
        try:
            return _file_fingerprint(_resolve_audio_path(audio_file))
        except (RuntimeError, ValueError, OSError):
            return f"missing:{audio_file}"

    def load_audio(self, audio_file, start_time=0.0, end_time=0.0):
        path = _resolve_audio_path(audio_file)
        try:
            waveform, sample_rate = _decode_audio_file(path)
        except ValueError as exc:
            raise ValueError(f"无法加载音频“{Path(path).name}”：{exc}") from exc
        except RuntimeError as exc:
            raise RuntimeError(f"无法加载音频“{Path(path).name}”：{exc}") from exc

        if not isinstance(waveform, torch.Tensor) or waveform.ndim != 2:
            raise ValueError("音频解码结果必须是“声道 × 采样点”的二维张量。")
        sample_rate = int(sample_rate)
        if sample_rate <= 0:
            raise ValueError("音频采样率必须大于 0。")
        sample_count = int(waveform.shape[-1])
        channel_count = int(waveform.shape[0])
        if channel_count <= 0 or sample_count <= 0:
            raise ValueError("音频为空，无法生成裁剪结果。")

        start_seconds = max(0.0, float(start_time))
        end_seconds = float(end_time)
        start_sample = min(sample_count, max(0, int(round(start_seconds * sample_rate))))
        if end_seconds <= 0.0:
            end_sample = sample_count
        else:
            end_sample = min(sample_count, max(0, int(round(end_seconds * sample_rate))))
        if start_sample >= end_sample:
            total_duration = sample_count / sample_rate
            raise ValueError(
                f"裁剪范围无效：开始时间必须早于结束时间，并位于 0–{total_duration:.3f} 秒内。"
            )

        selected = waveform[:, start_sample:end_sample].contiguous().unsqueeze(0)
        actual_start = start_sample / sample_rate
        actual_end = end_sample / sample_rate
        duration = (end_sample - start_sample) / sample_rate
        audio = {"waveform": selected, "sample_rate": sample_rate}
        return audio, duration, actual_start, actual_end, sample_rate, channel_count


NODE_CLASS_MAPPINGS = {"TUT_AdvancedAudioLoader": TUT_AdvancedAudioLoader}
NODE_DISPLAY_NAME_MAPPINGS = {"TUT_AdvancedAudioLoader": "TUT_高级音频加载"}
