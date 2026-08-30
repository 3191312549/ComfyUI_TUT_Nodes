import unittest

import torch

from ComfyUI_TUT_Nodes.core.filters import _smooth_noise

from ComfyUI_TUT_Nodes.nodes.image.filter_animation import TUT_AnimatedFilterSequence
from ComfyUI_TUT_Nodes.nodes.image.filters import (
    TUT_ComicFilter,
    TUT_GlassRefractionFilter,
    TUT_GlitchArtFilter,
    TUT_KaleidoscopeFilter,
    TUT_PixelArtFilter,
    TUT_RetroPrintFilter,
)


class TUTFilterTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(20260820)
        self.image = torch.rand((2, 32, 40, 3), generator=generator)
        self.one = self.image[:1]

    def calls(self, image=None, strength=1.0, mask=None):
        source = self.one if image is None else image
        return [
            lambda: TUT_RetroPrintFilter().apply_filter(source, "自定义", "三色", "#111111,#E2314D,#1D91C0", 4, 15.0, 1.0, 1.0, 0.2, 7, strength, mask),
            lambda: TUT_ComicFilter().apply_filter(source, "自定义", 5, 2, 1.2, 0.18, 0.42, True, 4, strength, mask),
            lambda: TUT_KaleidoscopeFilter().apply_filter(source, 8, 0.5, 0.5, 0.0, 1.0, True, 0.0, strength, mask),
            lambda: TUT_PixelArtFilter().apply_filter(source, "自定义", 4, 8, "自动", "#000000,#FFFFFF", "Bayer 2x2", 0.3, 0.3, strength, mask),
            lambda: TUT_GlassRefractionFilter().apply_filter(source, "自定义", "波纹玻璃", 5.0, 16.0, 0.0, 0.0, 1.0, 0.3, 7, strength, mask),
            lambda: TUT_GlitchArtFilter().apply_filter(source, "自定义", "数据损坏", 3, 3, 4, 0.2, 0.5, 0.05, 7, strength, mask),
        ]

    def test_static_filters_shapes_ranges_and_special_outputs(self):
        expected_lengths = (4, 5, 3, 4, 4, 4)
        for call, length in zip(self.calls(), expected_lengths):
            result = call()
            self.assertEqual(len(result), length)
            self.assertEqual(tuple(result[0].shape), (1, 32, 40, 3))
            self.assertTrue(torch.isfinite(result[0]).all())
            self.assertGreaterEqual(float(result[0].min()), 0.0)
            self.assertLessEqual(float(result[0].max()), 1.0)
            self.assertEqual(tuple(result[2].shape), (1, 32, 40))

    def test_zero_strength_and_black_mask_are_exact_identity(self):
        for call in self.calls(image=self.image, strength=0.0):
            self.assertTrue(torch.equal(call()[0], self.image))
        black = torch.zeros((1, 32, 40), dtype=torch.float32)
        for call in self.calls(image=self.image, mask=black):
            self.assertTrue(torch.equal(call()[0], self.image))

    def test_seeded_filters_are_deterministic(self):
        for index in (0, 4, 5):
            first = self.calls()[index]()[0]
            second = self.calls()[index]()[0]
            self.assertTrue(torch.equal(first, second), index)

    def test_illegal_mask_batch_raises(self):
        mask = torch.ones((3, 32, 40), dtype=torch.float32)
        with self.assertRaises(ValueError):
            self.calls(image=self.image, mask=mask)[0]()

    def test_animated_filters_output_input_major_and_are_deterministic(self):
        effects = ("色相循环", "扫光", "波纹", "像素化渐变", "万花筒旋转", "故障动画", "胶片闪烁")
        node = TUT_AnimatedFilterSequence()
        for effect in effects:
            args = (self.image, effect, 3, 1.0, 0.8, True, 19, 1.0)
            first = node.generate_sequence(*args)
            second = node.generate_sequence(*args)
            self.assertEqual(tuple(first[0].shape), (6, 32, 40, 3), effect)
            self.assertEqual(tuple(first[2].shape), (6, 32, 40), effect)
            self.assertTrue(torch.equal(first[0], second[0]), effect)

        zero = node.generate_sequence(self.image, "色相循环", 3, 1.0, 1.0, True, 1, 0.0)[0]
        self.assertTrue(torch.equal(zero, self.image.repeat_interleave(3, dim=0)))
        black = torch.zeros((1, 32, 40), dtype=torch.float32)
        masked = node.generate_sequence(self.image, "扫光", 3, 1.0, 1.0, True, 1, 1.0, black)[0]
        self.assertTrue(torch.equal(masked, self.image.repeat_interleave(3, dim=0)))

    def test_smoothed_noise_is_deterministic_and_bounded(self):
        import numpy as np

        first = _smooth_noise(np.random.default_rng(7), (40, 32), 0.3)
        second = _smooth_noise(np.random.default_rng(7), (40, 32), 0.3)
        self.assertEqual(first.shape, (32, 40))
        self.assertTrue(np.array_equal(first, second))
        self.assertGreaterEqual(float(first.min()), 0.0)
        self.assertLessEqual(float(first.max()), 1.0)


if __name__ == "__main__":
    unittest.main()
