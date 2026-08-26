import unittest

import numpy as np
import torch

from TUT_Nodes.categories import IMAGE_COLOR
from TUT_Nodes.core.color import advanced_auto_color_correct, film_tone
from TUT_Nodes.nodes.image.color import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    TUT_AutoColorCorrect,
    TUT_AutoColorCorrectAdvanced,
    TUT_BasicColor,
    TUT_BasicTone,
    TUT_ColorCompressor,
    TUT_ColorMatch,
    TUT_DetailEnhance,
    TUT_FilmTone,
    TUT_HSLBasic,
    TUT_Halation,
    TUT_LensDiffusion,
)


class TUTColorTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(20260820)
        self.image = torch.rand((2, 32, 40, 3), generator=generator)
        self.reference = torch.rand((1, 24, 36, 3), generator=generator) * 0.45 + 0.35
        self.mask = torch.ones((1, 20, 30), dtype=torch.float32)

    def test_public_contracts(self):
        expected = {
            "TUT_AutoColorCorrect", "TUT_AutoColorCorrectAdvanced", "TUT_BasicTone", "TUT_BasicColor",
            "TUT_DetailEnhance", "TUT_HSLBasic", "TUT_ColorMatch", "TUT_FilmTone",
            "TUT_Halation", "TUT_LensDiffusion", "TUT_ColorCompressor",
        }
        self.assertEqual(set(NODE_CLASS_MAPPINGS), expected)
        self.assertEqual(set(NODE_DISPLAY_NAME_MAPPINGS), expected)
        for node_id, node_class in NODE_CLASS_MAPPINGS.items():
            self.assertTrue(node_id.startswith("TUT_"))
            self.assertTrue(NODE_DISPLAY_NAME_MAPPINGS[node_id].startswith("TUT_"))
            self.assertEqual(node_class.CATEGORY, IMAGE_COLOR)
        self.assertEqual(TUT_ColorMatch.RETURN_NAMES, ("image", "show_help", "correction_mask", "difference_image"))
        self.assertEqual(TUT_FilmTone.RETURN_NAMES, ("image", "show_help", "effect_mask"))
        self.assertEqual(TUT_Halation.RETURN_NAMES, ("image", "show_help", "halation_mask"))
        self.assertEqual(TUT_LensDiffusion.RETURN_NAMES, ("image", "show_help", "diffusion_mask"))
        self.assertEqual(TUT_ColorCompressor.RETURN_NAMES, ("image", "show_help", "compressed_mask", "palette_image"))
        self.assertEqual(
            TUT_AutoColorCorrectAdvanced.RETURN_NAMES,
            ("image", "show_help", "effect_mask", "analysis_mask", "diagnostic_report"),
        )
        self.assertEqual(
            tuple(TUT_AutoColorCorrectAdvanced.INPUT_TYPES()["required"]),
            (
                "image", "auto_white_balance", "white_balance_method", "neutral_strictness",
                "white_balance_strength", "max_white_balance_ev", "auto_exposure",
                "target_midtone", "max_exposure_ev", "highlight_protection",
                "confidence_threshold", "strength",
            ),
        )
        self.assertEqual(
            tuple(TUT_AutoColorCorrectAdvanced.INPUT_TYPES()["optional"]),
            ("analysis_mask", "effect_mask"),
        )
        for node_class in (TUT_AutoColorCorrect, TUT_BasicTone, TUT_BasicColor,
                           TUT_DetailEnhance, TUT_HSLBasic):
            self.assertEqual(node_class.RETURN_TYPES, ("IMAGE", "STRING", "MASK"))
            self.assertEqual(node_class.RETURN_NAMES, ("image", "show_help", "effect_mask"))
        self.assertEqual(TUT_BasicTone.INPUT_TYPES()["required"]["exposure"][1]["min"], -5.0)
        self.assertEqual(TUT_BasicTone.INPUT_TYPES()["required"]["exposure"][1]["max"], 5.0)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["TUT_AutoColorCorrect"], "TUT_自动基础校色")
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["TUT_AutoColorCorrectAdvanced"],
            "TUT_自动基础校色（高级）",
        )
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["TUT_BasicTone"], "TUT_基础明暗调整")
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["TUT_BasicColor"], "TUT_基础色彩调整")
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["TUT_DetailEnhance"], "TUT_图像细节增强")
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["TUT_HSLBasic"], "TUT_HSL基础调整")

    def test_all_continuous_parameters_use_sliders(self):
        for node_id, node_class in NODE_CLASS_MAPPINGS.items():
            inputs = node_class.INPUT_TYPES()
            for section in ("required", "optional"):
                for name, definition in inputs.get(section, {}).items():
                    if definition[0] == "FLOAT":
                        self.assertEqual(
                            definition[1].get("display"), "slider",
                            f"{node_id}.{name} 应显示为滑块",
                        )
        seed_options = TUT_DetailEnhance.INPUT_TYPES()["required"]["seed"][1]
        self.assertNotEqual(seed_options.get("display"), "slider")

    def basic_calls(self, strength=1.0, mask=None):
        return [
            lambda: TUT_AutoColorCorrect().correct(self.image, True, True, strength, mask),
            lambda: TUT_BasicTone().adjust_tone(
                self.image, 0.4, 0.1, 0.15, -0.1, 0.1, 0.05, -0.05, 0.1, strength, mask,
            ),
            lambda: TUT_BasicColor().adjust_color(
                self.image, 0.15, -0.1, 0.1, 0.2, strength, mask,
            ),
            lambda: TUT_DetailEnhance().enhance_detail(
                self.image, 0.2, 0.15, 0.1, 0.2, 0.1, 0.1,
                0.08, 0.05, -0.1, 42, strength, mask,
            ),
            lambda: TUT_HSLBasic().adjust_hsl(
                self.image, "红色", 0.1, 0.15, 0.05, strength, mask,
            ),
        ]

    def calls(self, strength=1.0, mask=None):
        return [
            lambda: TUT_FilmTone().apply_tone(self.image, "自定义", 0.2, 0.2, 1.0, 0.25, 0.1, 0.1, strength, mask),
            lambda: TUT_Halation().apply_halation(self.image, "自定义", 0.65, 0.2, 5.0, 0.6, 0.5, strength, mask),
            lambda: TUT_LensDiffusion().apply_diffusion(self.image, "自定义", "柔光镜", 4.0, 0.6, 0.5, strength, mask),
            lambda: TUT_ColorCompressor().compress_color(self.image, "自定义", "#278E91", 100.0, 0.7, True, True, strength, mask),
        ]

    def test_single_image_nodes_shapes_ranges_and_mask_broadcast(self):
        lengths = (3, 3, 3, 4)
        for call, expected_length in zip(self.calls(mask=self.mask), lengths):
            result = call()
            self.assertEqual(len(result), expected_length)
            self.assertEqual(tuple(result[0].shape), (2, 32, 40, 3))
            self.assertEqual(result[0].dtype, torch.float32)
            self.assertTrue(torch.isfinite(result[0]).all())
            self.assertGreaterEqual(float(result[0].min()), 0.0)
            self.assertLessEqual(float(result[0].max()), 1.0)
            self.assertEqual(tuple(result[2].shape), (2, 32, 40))
        palette = self.calls(mask=self.mask)[3]()[3]
        self.assertEqual(tuple(palette.shape), (2, 64, 512, 3))

    def test_strength_zero_and_black_mask_are_exact_identity(self):
        for call in self.calls(strength=0.0):
            self.assertTrue(torch.equal(call()[0], self.image))
        black = torch.zeros((1, 32, 40), dtype=torch.float32)
        for call in self.calls(mask=black):
            self.assertTrue(torch.equal(call()[0], self.image))
        match = TUT_ColorMatch().match_color(
            self.image, self.reference, "Lab均值方差", "颜色和亮度", False, 0.0
        )[0]
        self.assertTrue(torch.equal(match, self.image))
        for call in self.basic_calls(strength=0.0):
            result = call()
            self.assertTrue(torch.equal(result[0], self.image))
            self.assertEqual(float(result[2].max()), 0.0)
        black = torch.zeros((1, 20, 30), dtype=torch.float32)
        for call in self.basic_calls(mask=black):
            result = call()
            self.assertTrue(torch.equal(result[0], self.image))
            self.assertEqual(float(result[2].max()), 0.0)

    def test_basic_nodes_batch_mask_and_actual_effect_mask(self):
        for call in self.basic_calls(mask=self.mask):
            output, _, effect_mask = call()
            self.assertEqual(tuple(output.shape), (2, 32, 40, 3))
            self.assertEqual(tuple(effect_mask.shape), (2, 32, 40))
            expected = torch.max(torch.abs(output - self.image), dim=-1).values
            self.assertTrue(torch.allclose(effect_mask, expected, atol=1e-7))
            self.assertTrue(torch.isfinite(output).all())
            self.assertGreaterEqual(float(output.min()), 0.0)
            self.assertLessEqual(float(output.max()), 1.0)

    def test_neutral_controls_are_exact_identity(self):
        results = [
            TUT_AutoColorCorrect().correct(self.image, False, False, 1.0),
            TUT_BasicTone().adjust_tone(self.image, 0, 0, 0, 0, 0, 0, 0, 0, 1.0),
            TUT_BasicColor().adjust_color(self.image, 0, 0, 0, 0, 1.0),
            TUT_DetailEnhance().enhance_detail(
                self.image, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1.0,
            ),
            TUT_HSLBasic().adjust_hsl(self.image, "全局", 0, 0, 0, 1.0),
        ]
        for output, _, effect_mask in results:
            self.assertTrue(torch.equal(output, self.image))
            self.assertEqual(float(effect_mask.max()), 0.0)

    def test_linear_exposure_and_monotonic_tone(self):
        gray = torch.full((1, 8, 64, 3), 0.25, dtype=torch.float32)
        exposed = TUT_BasicTone().adjust_tone(gray, 1.0, 0, 0, 0, 0, 0, 0, 0, 1.0)[0]

        def linear(value):
            return torch.where(value <= 0.04045, value / 12.92,
                               torch.pow((value + 0.055) / 1.055, 2.4))

        ratio = linear(exposed).mean() / linear(gray).mean()
        self.assertAlmostEqual(float(ratio), 2.0, places=4)
        ramp = torch.linspace(0, 1, 512).reshape(1, 1, 512, 1).repeat(1, 1, 1, 3)
        toned = TUT_BasicTone().adjust_tone(
            ramp, 0.25, 0.4, 0.6, -0.3, 0.35, -0.2, 0.15, 0.2, 1.0,
        )[0]
        self.assertTrue(torch.all(torch.diff(toned[0, 0, :, 0]) >= -1e-6))

    def test_auto_correction_color_and_exposure(self):
        cast = torch.empty((1, 32, 32, 3), dtype=torch.float32)
        cast[..., 0], cast[..., 1], cast[..., 2] = 0.42, 0.28, 0.18
        corrected = TUT_AutoColorCorrect().correct(cast, True, False, 1.0)[0]
        before_spread = torch.std(cast.mean(dim=(0, 1, 2)))
        after_spread = torch.std(corrected.mean(dim=(0, 1, 2)))
        self.assertLess(float(after_spread), float(before_spread))
        dark = torch.full((1, 32, 32, 3), 0.12, dtype=torch.float32)
        brighter = TUT_AutoColorCorrect().correct(dark, False, True, 1.0)[0]
        self.assertGreater(float(brighter.mean()), float(dark.mean()))

    def test_advanced_auto_color_reduces_dominant_color_false_correction(self):
        scene = torch.empty((1, 64, 64, 3), dtype=torch.float32)
        scene[..., 0], scene[..., 1], scene[..., 2] = 0.82, 0.08, 0.08
        scene[:, 24:40, 24:40] = 0.45
        basic = TUT_AutoColorCorrect().correct(scene, True, False, 1.0)[0]
        advanced = TUT_AutoColorCorrectAdvanced().correct_advanced(
            scene, True, "自适应融合", 0.65, 1.0, 0.75,
            False, 0.18, 1.5, 0.85, 0.45, 1.0,
        )[0]
        basic_change = torch.mean(torch.abs(basic - scene))
        advanced_change = torch.mean(torch.abs(advanced - scene))
        self.assertLess(float(advanced_change), float(basic_change) * 0.25)

    def test_advanced_auto_color_confidence_and_real_cast(self):
        cast = torch.empty((1, 32, 32, 3), dtype=torch.float32)
        cast[..., 0], cast[..., 1], cast[..., 2] = 0.52, 0.36, 0.24
        corrected, _, _, analysis, report = TUT_AutoColorCorrectAdvanced().correct_advanced(
            cast, True, "自适应融合", 0.65, 1.0, 0.75,
            False, 0.18, 1.5, 0.85, 0.45, 1.0,
        )
        self.assertLess(float(torch.std(corrected.mean(dim=(0, 1, 2)))),
                        float(torch.std(cast.mean(dim=(0, 1, 2)))))
        self.assertGreater(float(analysis.max()), 0.0)
        self.assertIn("置信度", report)
        self.assertIn("RGB 增益", report)

        red = torch.zeros((1, 32, 32, 3), dtype=torch.float32)
        red[..., 0] = 0.8
        automatic = TUT_AutoColorCorrectAdvanced().correct_advanced(
            red, True, "自适应融合", 0.65, 1.0, 0.75,
            False, 0.18, 1.5, 0.85, 0.45, 1.0,
        )[0]
        forced = TUT_AutoColorCorrectAdvanced().correct_advanced(
            red, True, "自适应融合", 0.65, 1.0, 0.75,
            False, 0.18, 1.5, 0.85, 0.0, 1.0,
        )[0]
        self.assertTrue(torch.equal(automatic, red))
        self.assertGreater(float(torch.mean(torch.abs(forced - red))), 0.01)

    def test_advanced_auto_color_ev_limits_highlights_and_independent_masks(self):
        frame = np.full((32, 32, 3), 0.10, dtype=np.float32)
        frame[:4, :4] = 0.95
        region = np.ones((32, 32), dtype=np.float32)
        _, _, protected = advanced_auto_color_correct(
            frame, region, False, "自适应融合", 0.65, 1.0, 0.75,
            True, 0.30, 0.50, 1.0, 0.45,
        )
        _, _, unprotected = advanced_auto_color_correct(
            frame, region, False, "自适应融合", 0.65, 1.0, 0.75,
            True, 0.30, 0.50, 0.0, 0.45,
        )
        self.assertLessEqual(abs(protected["target_exposure_ev"]), 0.5 + 1e-7)
        self.assertLess(protected["effective_exposure_ev"], unprotected["effective_exposure_ev"])

        image = torch.full((1, 24, 32, 3), 0.2)
        analysis_mask = torch.zeros((1, 12, 16)); analysis_mask[:, :, :8] = 1.0
        effect_mask = torch.zeros((2, 24, 32)); effect_mask[:, :, 16:] = 1.0
        result = TUT_AutoColorCorrectAdvanced().correct_advanced(
            image, False, "自适应融合", 0.65, 1.0, 0.75,
            True, 0.30, 1.0, 0.0, 0.45, 1.0,
            analysis_mask, effect_mask,
        )
        self.assertEqual(tuple(result[0].shape), (2, 24, 32, 3))
        self.assertTrue(torch.equal(result[0][:, :, :16], image.repeat(2, 1, 1, 1)[:, :, :16]))
        self.assertEqual(float(result[3][:, :, 17:].max()), 0.0)
        self.assertGreater(float(result[3][:, :, :15].max()), 0.0)
        self.assertGreater(float(result[2][:, :, 16:].max()), 0.0)

    def test_advanced_auto_color_methods_and_white_balance_limit(self):
        cast = np.empty((32, 32, 3), dtype=np.float32)
        cast[..., 0], cast[..., 1], cast[..., 2] = 0.58, 0.34, 0.20
        region = np.ones((32, 32), dtype=np.float32)
        for method in ("自适应融合", "中性像素", "灰世界", "灰度幂均值"):
            output, sample, diagnostic = advanced_auto_color_correct(
                cast, region, True, method, 0.65, 1.0, 0.25,
                False, 0.18, 1.5, 0.85, 0.0,
            )
            self.assertTrue(np.isfinite(output).all(), method)
            self.assertGreater(float(sample.max()), 0.0, method)
            effective_ev = np.abs(np.log2(np.maximum(diagnostic["effective_gains"], 1e-8)))
            self.assertLessEqual(float(effective_ev.max()), 0.25 + 1e-7, method)

    def test_advanced_auto_color_exact_bypasses_and_errors(self):
        node = TUT_AutoColorCorrectAdvanced()
        disabled = node.correct_advanced(
            self.image, False, "自适应融合", 0.65, 1.0, 0.75,
            False, 0.18, 1.5, 0.85, 0.45, 1.0,
        )
        self.assertTrue(torch.equal(disabled[0], self.image))
        self.assertEqual(float(disabled[2].max()), 0.0)
        zero = node.correct_advanced(
            self.image, True, "自适应融合", 0.65, 1.0, 0.75,
            True, 0.18, 1.5, 0.85, 0.45, 0.0,
        )
        self.assertTrue(torch.equal(zero[0], self.image))
        black = torch.zeros((1, 32, 40))
        empty = node.correct_advanced(
            self.image, True, "自适应融合", 0.65, 1.0, 0.75,
            True, 0.18, 1.5, 0.85, 0.45, 1.0, black,
        )
        self.assertTrue(torch.equal(empty[0], self.image))
        self.assertIn("没有有效分析像素", empty[4])
        with self.assertRaisesRegex(ValueError, "高级白平衡"):
            node.correct_advanced(
                self.image, True, "未知", 0.65, 1.0, 0.75,
                True, 0.18, 1.5, 0.85, 0.45, 1.0,
            )
        with self.assertRaisesRegex(ValueError, "批次"):
            node.correct_advanced(
                self.image, True, "自适应融合", 0.65, 1.0, 0.75,
                True, 0.18, 1.5, 0.85, 0.45, 1.0,
                torch.ones((3, 32, 40)),
            )

    def test_basic_color_directions_and_vibrance(self):
        neutral = torch.full((1, 16, 16, 3), 0.4)
        warm = TUT_BasicColor().adjust_color(neutral, 0.5, 0, 0, 0, 1.0)[0]
        self.assertGreater(float(warm[..., 0].mean()), float(neutral[..., 0].mean()))
        self.assertLess(float(warm[..., 2].mean()), float(neutral[..., 2].mean()))
        tinted = TUT_BasicColor().adjust_color(neutral, 0, 0.5, 0, 0, 1.0)[0]
        self.assertGreater(float((tinted[..., [0, 2]].mean() - tinted[..., 1].mean())), 0.0)

        colors = torch.tensor([[[[0.50, 0.45, 0.45], [0.90, 0.10, 0.10]]]], dtype=torch.float32)
        vivid = TUT_BasicColor().adjust_color(colors, 0, 0, 0, 1.0, 1.0)[0]
        before = colors.max(dim=-1).values - colors.min(dim=-1).values
        after = vivid.max(dim=-1).values - vivid.min(dim=-1).values
        self.assertGreater(float((after - before)[0, 0, 0]), float((after - before)[0, 0, 1]))

    def test_detail_determinism_denoise_and_hsl_selection(self):
        generator = torch.Generator().manual_seed(9)
        noisy = torch.clamp(torch.full((1, 48, 48, 3), 0.5) + torch.randn((1, 48, 48, 1), generator=generator) * 0.08, 0, 1)
        denoised = TUT_DetailEnhance().enhance_detail(
            noisy, 0, 0, 0, 0, 1, 0, 0, 0, 0, 7, 1.0,
        )[0]
        self.assertLess(float(denoised.var()), float(noisy.var()))
        grain_a = TUT_DetailEnhance().enhance_detail(
            noisy, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 77, 1.0,
        )[0]
        grain_b = TUT_DetailEnhance().enhance_detail(
            noisy, 0, 0, 0, 0, 0, 0, 0.5, 0, 0, 77, 1.0,
        )[0]
        self.assertTrue(torch.equal(grain_a, grain_b))

        colors = torch.zeros((1, 16, 32, 3), dtype=torch.float32)
        colors[:, :, :16, 0] = 1.0
        colors[:, :, 16:, 2] = 1.0
        adjusted = TUT_HSLBasic().adjust_hsl(colors, "红色", 0, -1, 0, 1.0)[0]
        self.assertFalse(torch.equal(adjusted[:, :, :16], colors[:, :, :16]))
        self.assertTrue(torch.allclose(adjusted[:, :, 16:], colors[:, :, 16:], atol=1e-6))

        for height, width in ((31, 47), (96, 128)):
            sample = torch.rand((1, height, width, 3), generator=generator)
            result = TUT_DetailEnhance().enhance_detail(
                sample, 0.3, 0.2, 0.2, 0.4, 0.1, 0.1, 0, 0, -0.2, 5, 1.0,
            )[0]
            self.assertEqual(tuple(result.shape), tuple(sample.shape))
            self.assertTrue(torch.isfinite(result).all())

    def test_color_match_methods_and_reference_broadcast(self):
        node = TUT_ColorMatch()
        for method in ("Lab均值方差", "直方图匹配", "分区分布"):
            result = node.match_color(self.image, self.reference, method, "颜色和亮度", False, 0.75, self.mask, None)
            self.assertEqual(tuple(result[0].shape), (2, 32, 40, 3))
            self.assertEqual(tuple(result[2].shape), (2, 32, 40))
            self.assertEqual(tuple(result[3].shape), (2, 32, 40, 3))
            self.assertTrue(torch.isfinite(result[0]).all(), method)
        source_mean = self.image.mean(dim=(0, 1, 2))
        reference_mean = self.reference.mean(dim=(0, 1, 2))
        matched = node.match_color(self.image, self.reference, "Lab均值方差", "颜色和亮度", False, 1.0)[0]
        self.assertLess(float(torch.linalg.vector_norm(matched.mean(dim=(0, 1, 2)) - reference_mean)),
                        float(torch.linalg.vector_norm(source_mean - reference_mean)))

        black_reference = torch.zeros((1, 24, 36), dtype=torch.float32)
        disabled = node.match_color(
            self.image, self.reference, "Lab均值方差", "颜色和亮度", False,
            1.0, None, black_reference,
        )
        self.assertTrue(torch.equal(disabled[0], self.image))
        self.assertEqual(float(disabled[2].max()), 0.0)

    def test_modes_and_film_curve_monotonic(self):
        for mode in ("柔光镜", "黑柔", "薄雾", "梦幻扩散"):
            result = TUT_LensDiffusion().apply_diffusion(
                self.image[:1], "自定义", mode, 3.0, 0.6, 0.5, 1.0
            )[0]
            self.assertTrue(torch.isfinite(result).all(), mode)
        ramp = np.linspace(0, 1, 1024, dtype=np.float32).reshape(1, 1024, 1).repeat(3, axis=2)
        shaped = film_tone(ramp, 0.3, 0.25, 1.0, 0.0, 0.0, 0.0)
        self.assertTrue(np.all(np.diff(shaped[:, :, 0], axis=1) >= -1e-6))

    def test_halation_responds_to_highlights(self):
        dark = torch.zeros((1, 32, 40, 3), dtype=torch.float32)
        bright = dark.clone(); bright[:, 14:18, 18:22] = 1.0
        node = TUT_Halation()
        dark_mask = node.apply_halation(dark, "自定义", 0.7, 0.2, 5.0, 0.6, 0.5, 1.0)[2]
        bright_mask = node.apply_halation(bright, "自定义", 0.7, 0.2, 5.0, 0.6, 0.5, 1.0)[2]
        self.assertEqual(float(dark_mask.max()), 0.0)
        self.assertGreater(float(bright_mask.max()), 0.0)

    def test_color_compressor_converges_and_invalid_hex_is_clear(self):
        red = torch.zeros((1, 16, 16, 3)); red[..., 0] = 1.0
        node = TUT_ColorCompressor()
        output = node.compress_color(red, "自定义", "#00FF00", 180.0, 1.0, False, False, 1.0)[0]
        self.assertGreater(float(output[..., 1].mean()), float(red[..., 1].mean()))
        with self.assertRaisesRegex(ValueError, "颜色"):
            node.compress_color(red, "自定义", "zzzzzz", 90.0, 1.0, True, False, 1.0)

    def test_illegal_batch_raises(self):
        bad_mask = torch.ones((3, 32, 40))
        with self.assertRaises(ValueError):
            self.calls(mask=bad_mask)[0]()
        bad_reference = torch.rand((3, 24, 36, 3))
        with self.assertRaises(ValueError):
            TUT_ColorMatch().match_color(self.image, bad_reference, "Lab均值方差", "颜色和亮度", False, 1.0)
        with self.assertRaisesRegex(ValueError, "批次"):
            TUT_BasicColor().adjust_color(self.image, 0, 0, 0.1, 0, 1.0, bad_mask)


if __name__ == "__main__":
    unittest.main()
