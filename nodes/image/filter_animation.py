"""Animated image-filter sequence node for TUT_Nodes."""

from __future__ import annotations

import math

import numpy as np

from ...categories import IMAGE_FILTER
from ...core.filters import (
    animated_effect,
    composite_effect,
    effect_mask_tensor,
    image_batch_tensor,
    prepare_filter_batches,
    rng_for,
)


class TUT_AnimatedFilterSequence:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "effect": ([
                    "色相循环", "扫光", "波纹", "像素化渐变",
                    "万花筒旋转", "故障动画", "胶片闪烁",
                ], {"default": "色相循环"}),
                "frames": ("INT", {"default": 24, "min": 1, "max": 240}),
                "cycles": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 16.0, "step": 0.05}),
                "amplitude": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.01}),
                "seamless_loop": ("BOOLEAN", {"default": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xFFFFFFFFFFFFFFFF}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {"mask": ("MASK",)},
        }

    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("frames", "show_help", "effect_mask")
    FUNCTION = "generate_sequence"
    CATEGORY = IMAGE_FILTER
    DESCRIPTION = "从每张输入图片生成可循环的动态滤镜 IMAGE 批次。"

    def generate_sequence(self, image, effect, frames, cycles, amplitude,
                          seamless_loop, seed, strength, mask=None):
        source_frames, source_masks = prepare_filter_batches(image, mask)
        frame_count = max(1, int(frames))
        output, output_masks = [], []

        for batch_index, (source, region) in enumerate(zip(source_frames, source_masks)):
            height, width = source.size[1], source.size[0]
            base_rng = rng_for(seed, batch_index)
            base_noise = base_rng.normal(0.0, 1.0, (height, width)).astype(np.float32)

            for frame_index in range(frame_count):
                denominator = frame_count if bool(seamless_loop) else max(1, frame_count - 1)
                phase = 2.0 * math.pi * float(cycles) * frame_index / denominator
                # Recreate the per-image RNG for every frame. Random spatial
                # choices remain stable while ``phase`` supplies the motion.
                frame_rng = rng_for(seed, batch_index)
                filtered = animated_effect(
                    source, effect, phase, amplitude, frame_rng, base_noise=base_noise,
                )
                output.append(composite_effect(source, filtered, region, strength))
                output_masks.append(region.copy())

        if float(strength) == 0.0:
            source_tensor = image
            if int(source_tensor.shape[0]) == 1 and len(source_frames) > 1:
                source_tensor = source_tensor.repeat(len(source_frames), 1, 1, 1)
            output_tensor = source_tensor.repeat_interleave(frame_count, dim=0)
        else:
            output_tensor = image_batch_tensor(output)
            for batch_index, region in enumerate(source_masks):
                if region.getbbox() is None:
                    source_index = batch_index if int(image.shape[0]) == len(source_frames) else 0
                    start = batch_index * frame_count
                    output_tensor[start:start + frame_count] = image[source_index]

        return (
            output_tensor,
            "TUT_Nodes/图片/滤镜",
            effect_mask_tensor(output_masks),
        )


NODE_CLASS_MAPPINGS = {
    "TUT_AnimatedFilterSequence": TUT_AnimatedFilterSequence,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_AnimatedFilterSequence": "TUT_动态滤镜序列",
}
