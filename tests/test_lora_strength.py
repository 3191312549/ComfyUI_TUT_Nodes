import unittest
from unittest.mock import patch

from ComfyUI_TUT_Nodes.categories import MODEL_LORA
from ComfyUI_TUT_Nodes.nodes.pending.lora import (
    CLIP_DISABLED,
    CLIP_FIXED,
    CLIP_FOLLOW,
    CUSTOM_MODE,
    RANGE_MODE,
    TUT_LoraStrengthTester,
    build_strength_range,
    parse_custom_strengths,
)


class TUTLoraStrengthTesterTests(unittest.TestCase):
    def test_public_interface_is_stable_list_output(self):
        required = TUT_LoraStrengthTester.INPUT_TYPES()["required"]
        self.assertEqual(
            tuple(required),
            (
                "model", "clip", "lora_name", "generation_mode", "start_strength",
                "end_strength", "strength_step", "custom_strengths", "clip_strength_mode",
                "fixed_clip_strength", "max_tests",
            ),
        )
        self.assertEqual(
            TUT_LoraStrengthTester.RETURN_TYPES,
            ("MODEL", "CLIP", "FLOAT", "STRING", "INT"),
        )
        self.assertEqual(
            TUT_LoraStrengthTester.RETURN_NAMES,
            ("models", "clips", "strengths", "labels", "item_count"),
        )
        self.assertEqual(TUT_LoraStrengthTester.OUTPUT_IS_LIST, (True, True, True, True, False))
        self.assertEqual(TUT_LoraStrengthTester.CATEGORY, MODEL_LORA)

    def test_range_uses_decimal_steps_and_always_includes_endpoint(self):
        self.assertEqual(build_strength_range(0, 1, 0.2, 16), [0, 0.2, 0.4, 0.6, 0.8, 1])
        self.assertEqual(build_strength_range(1, 0, 0.3, 16), [1, 0.7, 0.4, 0.1, 0])
        self.assertEqual(build_strength_range(0.5, 0.5, 0.2, 16), [0.5])

    def test_custom_list_supports_chinese_separators_and_deduplicates(self):
        self.assertEqual(parse_custom_strengths("0，0.5; 1\n0.5", 8), [0, 0.5, 1])
        with self.assertRaisesRegex(ValueError, "不是有效数字"):
            parse_custom_strengths("0, 错误", 8)
        with self.assertRaisesRegex(ValueError, "超过"):
            parse_custom_strengths("0,1,2", 2)

    def test_invalid_limits_and_steps_have_chinese_errors(self):
        with self.assertRaisesRegex(ValueError, "步长"):
            build_strength_range(0, 1, 0, 16)
        with self.assertRaisesRegex(ValueError, "超过"):
            build_strength_range(0, 1, 0.01, 8)

    @staticmethod
    def _inputs(**overrides):
        values = {
            "model": "base-model",
            "clip": "base-clip",
            "lora_name": "styles/test.safetensors",
            "generation_mode": CUSTOM_MODE,
            "start_strength": 0.0,
            "end_strength": 1.0,
            "strength_step": 0.2,
            "custom_strengths": "0, 0.5, 1",
            "clip_strength_mode": CLIP_FOLLOW,
            "fixed_clip_strength": 1.0,
            "max_tests": 16,
        }
        values.update(overrides)
        return values

    @staticmethod
    def _fake_apply(model, clip, _lora, model_strength, clip_strength, lora_metadata=None):
        return (
            f"model:{model_strength:g}" if model is not None else None,
            f"clip:{clip_strength:g}" if clip is not None else None,
        )

    def test_aligned_outputs_zero_bypass_and_single_file_load(self):
        node = TUT_LoraStrengthTester()
        with (
            patch(
                "ComfyUI_TUT_Nodes.nodes.pending.lora._resolve_lora_path",
                return_value="X:/loras/test.safetensors",
            ),
            patch(
                "ComfyUI_TUT_Nodes.nodes.pending.lora._load_torch_file",
                return_value=({"weight": "fake"}, {"name": "test"}),
            ) as load_file,
            patch(
                "ComfyUI_TUT_Nodes.nodes.pending.lora._load_lora_for_models",
                side_effect=self._fake_apply,
            ) as apply_lora,
        ):
            models, clips, strengths, labels, count = node.test_strengths(**self._inputs())
            node.test_strengths(**self._inputs(custom_strengths="0.25"))

        self.assertEqual(models, ["base-model", "model:0.5", "model:1"])
        self.assertEqual(clips, ["base-clip", "clip:0.5", "clip:1"])
        self.assertEqual(strengths, [0, 0.5, 1])
        self.assertEqual(count, 3)
        self.assertEqual(len(labels), count)
        self.assertIn("模型 0.5｜CLIP 0.5", labels[1])
        self.assertEqual(load_file.call_count, 1)
        self.assertEqual(apply_lora.call_count, 3)

    def test_fixed_and_disabled_clip_modes_apply_only_needed_side(self):
        node = TUT_LoraStrengthTester()
        node.loaded_lora = ("X:/loras/test.safetensors", {"weight": "fake"}, {})
        with (
            patch(
                "ComfyUI_TUT_Nodes.nodes.pending.lora._resolve_lora_path",
                return_value="X:/loras/test.safetensors",
            ),
            patch(
                "ComfyUI_TUT_Nodes.nodes.pending.lora._load_lora_for_models",
                side_effect=self._fake_apply,
            ) as apply_lora,
        ):
            fixed = node.test_strengths(
                **self._inputs(custom_strengths="0", clip_strength_mode=CLIP_FIXED, fixed_clip_strength=0.7)
            )
            disabled = node.test_strengths(
                **self._inputs(custom_strengths="0.5", clip_strength_mode=CLIP_DISABLED)
            )

        self.assertEqual(fixed[0], ["base-model"])
        self.assertEqual(fixed[1], ["clip:0.7"])
        self.assertEqual(disabled[0], ["model:0.5"])
        self.assertEqual(disabled[1], ["base-clip"])
        first_call, second_call = apply_lora.call_args_list
        self.assertIsNone(first_call.args[0])
        self.assertEqual(first_call.args[1], "base-clip")
        self.assertEqual(second_call.args[0], "base-model")
        self.assertIsNone(second_call.args[1])

    def test_range_mode_runs_through_node(self):
        node = TUT_LoraStrengthTester()
        node.loaded_lora = ("X:/loras/test.safetensors", {}, None)
        with (
            patch(
                "ComfyUI_TUT_Nodes.nodes.pending.lora._resolve_lora_path",
                return_value="X:/loras/test.safetensors",
            ),
            patch(
                "ComfyUI_TUT_Nodes.nodes.pending.lora._load_lora_for_models",
                side_effect=self._fake_apply,
            ),
        ):
            result = node.test_strengths(
                **self._inputs(
                    generation_mode=RANGE_MODE,
                    start_strength=0,
                    end_strength=0.5,
                    strength_step=0.25,
                )
            )
        self.assertEqual(result[2], [0, 0.25, 0.5])
        self.assertEqual(result[4], 3)


if __name__ == "__main__":
    unittest.main()
