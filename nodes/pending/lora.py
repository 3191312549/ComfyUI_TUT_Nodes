"""LoRA strength sweep node.

The node lives in its own module so promotion only needs category, display-name,
and documentation changes; its saved-workflow interface stays untouched.
"""

from __future__ import annotations

from decimal import Decimal, InvalidOperation
import math
from pathlib import Path
import re

from ...categories import MODEL_LORA


RANGE_MODE = "范围生成"
CUSTOM_MODE = "自定义列表"
CLIP_FOLLOW = "跟随模型强度"
CLIP_FIXED = "固定强度"
CLIP_DISABLED = "不应用到CLIP"
MAX_STRENGTH_COUNT = 64


def _lora_names():
    try:
        import folder_paths
    except ModuleNotFoundError:
        return []

    return folder_paths.get_filename_list("loras")


def _resolve_lora_path(lora_name: str):
    import folder_paths

    return folder_paths.get_full_path_or_raise("loras", lora_name)


def _load_torch_file(lora_path: str):
    import comfy.utils

    return comfy.utils.load_torch_file(
        lora_path,
        safe_load=True,
        return_metadata=True,
    )


def _load_lora_for_models(model, clip, lora, model_strength, clip_strength, metadata):
    import comfy.sd

    return comfy.sd.load_lora_for_models(
        model,
        clip,
        lora,
        model_strength,
        clip_strength,
        lora_metadata=metadata,
    )


def _validate_strength(value, name: str) -> float:
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"{name}必须是有限数字。")
    if number < -100.0 or number > 100.0:
        raise ValueError(f"{name}必须在 -100 到 100 之间。")
    return number


def _deduplicate(values: list[float]) -> list[float]:
    result = []
    seen = set()
    for value in values:
        key = round(float(value), 12)
        if key not in seen:
            seen.add(key)
            result.append(float(value))
    return result


def parse_custom_strengths(text: str, max_tests: int) -> list[float]:
    parts = [part for part in re.split(r"[\s,，;；]+", str(text).strip()) if part]
    if not parts:
        raise ValueError("自定义强度列表为空，请至少填写一个数值。")

    values = []
    for index, part in enumerate(parts, start=1):
        try:
            value = float(part)
        except ValueError as exc:
            raise ValueError(f"自定义强度第 {index} 项“{part}”不是有效数字。") from exc
        values.append(_validate_strength(value, f"自定义强度第 {index} 项"))

    values = _deduplicate(values)
    if len(values) > max_tests:
        raise ValueError(f"共生成 {len(values)} 个强度，超过当前最大测试数量 {max_tests}。")
    return values


def build_strength_range(start: float, end: float, step: float, max_tests: int) -> list[float]:
    start_value = _validate_strength(start, "起始强度")
    end_value = _validate_strength(end, "结束强度")
    step_value = float(step)
    if not math.isfinite(step_value) or step_value <= 0:
        raise ValueError("强度步长必须是大于 0 的有限数字。")

    try:
        current = Decimal(str(start_value))
        endpoint = Decimal(str(end_value))
        increment = Decimal(str(step_value))
    except InvalidOperation as exc:
        raise ValueError("强度范围包含无法解析的数字。") from exc

    if current == endpoint:
        return [start_value]

    direction = Decimal(1) if endpoint > current else Decimal(-1)
    increment *= direction
    values = []
    while (current < endpoint if direction > 0 else current > endpoint):
        values.append(_validate_strength(float(current), "生成强度"))
        if len(values) >= max_tests:
            raise ValueError(f"强度范围会生成超过 {max_tests} 项，请增大步长或缩小范围。")
        current += increment

    values.append(end_value)
    values = _deduplicate(values)
    if len(values) > max_tests:
        raise ValueError(f"强度范围会生成超过 {max_tests} 项，请增大步长或缩小范围。")
    return values


def build_strengths(
    generation_mode: str,
    start_strength: float,
    end_strength: float,
    strength_step: float,
    custom_strengths: str,
    max_tests: int,
) -> list[float]:
    limit = int(max_tests)
    if limit < 1 or limit > MAX_STRENGTH_COUNT:
        raise ValueError(f"最大测试数量必须在 1 到 {MAX_STRENGTH_COUNT} 之间。")
    if generation_mode == RANGE_MODE:
        return build_strength_range(start_strength, end_strength, strength_step, limit)
    if generation_mode == CUSTOM_MODE:
        return parse_custom_strengths(custom_strengths, limit)
    raise ValueError(f"不支持的生成方式：{generation_mode}")


class TUT_LoraStrengthTester:
    """Load one LoRA once and expose aligned MODEL/CLIP strength lists."""

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "连接未应用本节点 LoRA 的基础模型。"}),
                "clip": ("CLIP", {"tooltip": "连接与基础模型配套的 CLIP。"}),
                "lora_name": (
                    _lora_names(),
                    {"tooltip": "选择本次需要测试的单个 LoRA。"},
                ),
                "generation_mode": (
                    (RANGE_MODE, CUSTOM_MODE),
                    {"tooltip": "按起止范围自动生成，或手动填写强度列表。"},
                ),
                "start_strength": (
                    "FLOAT",
                    {"default": 0.0, "min": -100.0, "max": 100.0, "step": 0.05, "tooltip": "范围生成的第一个强度。"},
                ),
                "end_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.05, "tooltip": "范围生成始终包含该终点。"},
                ),
                "strength_step": (
                    "FLOAT",
                    {"default": 0.2, "min": 0.0001, "max": 100.0, "step": 0.05, "tooltip": "只填写正数；节点会根据起止值自动判断升序或降序。"},
                ),
                "custom_strengths": (
                    "STRING",
                    {"default": "0, 0.25, 0.5, 0.75, 1.0", "multiline": True, "tooltip": "支持逗号、中文逗号、分号、空格或换行分隔；重复值会自动去除。"},
                ),
                "clip_strength_mode": (
                    (CLIP_FOLLOW, CLIP_FIXED, CLIP_DISABLED),
                    {"tooltip": "控制每组测试中 LoRA 对 CLIP 的作用强度。"},
                ),
                "fixed_clip_strength": (
                    "FLOAT",
                    {"default": 1.0, "min": -100.0, "max": 100.0, "step": 0.05, "tooltip": "仅在 CLIP 强度模式为固定强度时使用。"},
                ),
                "max_tests": (
                    "INT",
                    {"default": 16, "min": 1, "max": MAX_STRENGTH_COUNT, "step": 1, "tooltip": "限制一次执行生成的测试组数，防止误操作。"},
                ),
            }
        }

    RETURN_TYPES = ("MODEL", "CLIP", "FLOAT", "STRING", "INT")
    RETURN_NAMES = ("models", "clips", "strengths", "labels", "item_count")
    OUTPUT_IS_LIST = (True, True, True, True, False)
    FUNCTION = "test_strengths"
    CATEGORY = MODEL_LORA
    DESCRIPTION = "一次加载一个 LoRA，按范围或自定义列表输出严格对齐的 MODEL、CLIP、强度与标签列表，用于固定种子批量比较不同强度。"

    def _load_lora_file(self, lora_name: str):
        lora_path = _resolve_lora_path(lora_name)
        if self.loaded_lora is not None and self.loaded_lora[0] == lora_path:
            return self.loaded_lora[1], self.loaded_lora[2]

        lora, metadata = _load_torch_file(lora_path)
        self.loaded_lora = (lora_path, lora, metadata)
        return lora, metadata

    @staticmethod
    def _clip_strength(model_strength: float, mode: str, fixed_strength: float) -> float:
        if mode == CLIP_FOLLOW:
            return model_strength
        if mode == CLIP_FIXED:
            return _validate_strength(fixed_strength, "CLIP固定强度")
        if mode == CLIP_DISABLED:
            return 0.0
        raise ValueError(f"不支持的 CLIP 强度模式：{mode}")

    @staticmethod
    def _apply_strength(model, clip, lora, metadata, model_strength: float, clip_strength: float):
        apply_model = model_strength != 0.0
        apply_clip = clip_strength != 0.0
        if not apply_model and not apply_clip:
            return model, clip

        model_lora, clip_lora = _load_lora_for_models(
            model if apply_model else None,
            clip if apply_clip else None,
            lora,
            model_strength if apply_model else 0.0,
            clip_strength if apply_clip else 0.0,
            metadata,
        )
        return model_lora if apply_model else model, clip_lora if apply_clip else clip

    def test_strengths(
        self,
        model,
        clip,
        lora_name,
        generation_mode,
        start_strength,
        end_strength,
        strength_step,
        custom_strengths,
        clip_strength_mode,
        fixed_clip_strength,
        max_tests,
    ):
        strengths = build_strengths(
            generation_mode,
            start_strength,
            end_strength,
            strength_step,
            custom_strengths,
            max_tests,
        )
        lora, metadata = self._load_lora_file(lora_name)
        lora_label = Path(str(lora_name)).stem

        models = []
        clips = []
        labels = []
        for model_strength in strengths:
            clip_strength = self._clip_strength(model_strength, clip_strength_mode, fixed_clip_strength)
            model_output, clip_output = self._apply_strength(
                model,
                clip,
                lora,
                metadata,
                model_strength,
                clip_strength,
            )
            models.append(model_output)
            clips.append(clip_output)
            labels.append(
                f"{lora_label}｜模型 {model_strength:g}｜CLIP {clip_strength:g}"
            )

        return models, clips, strengths, labels, len(strengths)


NODE_CLASS_MAPPINGS = {"TUT_LoraStrengthTester": TUT_LoraStrengthTester}
NODE_DISPLAY_NAME_MAPPINGS = {"TUT_LoraStrengthTester": "TUT_LoRA强度批量测试"}
