import unittest

import torch

from TUT_Nodes.core.fonts import font_options
from TUT_Nodes.nodes.image.text import TUT_AutoContrastWatermark


class TUTAdaptiveWatermarkTests(unittest.TestCase):
    def setUp(self):
        self.font_name = font_options()[0]
        self.node = TUT_AutoContrastWatermark()

    def render(self, image, position="左上角", **overrides):
        values = {
            "text": "TUT",
            "font_name": self.font_name,
            "size_percent": 12.0,
            "max_width_percent": 50.0,
            "x_margin_percent": 5.0,
            "y_margin_percent": 7.0,
            "opacity": 1.0,
        }
        values.update(overrides)
        return self.node.watermark(image, position=position, **values)

    def test_four_corners_and_separate_margins(self):
        image = torch.zeros((1, 100, 200, 3), dtype=torch.float32)
        expected_quadrants = {
            "左上角": (False, False),
            "右上角": (True, False),
            "左下角": (False, True),
            "右下角": (True, True),
        }
        for position, (right, bottom) in expected_quadrants.items():
            _, _, mask = self.render(image, position)
            points = torch.nonzero(mask[0] > 0.05)
            self.assertGreater(points.shape[0], 0, position)
            center_y, center_x = points.float().mean(dim=0)
            self.assertEqual(bool(center_x > 100), right, position)
            self.assertEqual(bool(center_y > 50), bottom, position)
            self.assertGreaterEqual(int(points[:, 1].min()), 5 if not right else 0)
            self.assertGreaterEqual(int(points[:, 0].min()), 7 if not bottom else 0)

    def test_uniform_background_uses_exact_pixel_inverse_per_batch(self):
        image = torch.empty((2, 100, 200, 3), dtype=torch.float32)
        image[0] = torch.tensor([0.2, 0.4, 0.8])
        image[1] = torch.tensor([1.0, 1.0, 1.0])
        output, _, mask = self.render(image)
        solid = mask > 0.99
        first = output[0][solid[0]].mean(dim=0)
        second = output[1][solid[1]].mean(dim=0)
        self.assertTrue(torch.allclose(first, torch.tensor([0.8, 0.6, 0.2]), atol=1.0 / 255.0))
        self.assertTrue(torch.allclose(second, torch.zeros(3), atol=1.0 / 255.0))

    def test_nonuniform_background_inverts_each_masked_pixel(self):
        values = torch.arange(200, dtype=torch.float32).view(1, 1, 200, 1) / 255.0
        image = values.expand(1, 100, 200, 3).contiguous()
        output, _, mask = self.render(image)
        solid = mask[0] > 0.99
        self.assertGreater(int(solid.sum()), 0)
        self.assertTrue(torch.allclose(output[0][solid], 1.0 - image[0][solid], atol=1.0 / 255.0))
        self.assertTrue(torch.equal(output[0][mask[0] == 0], image[0][mask[0] == 0]))

    def test_opacity_blends_inverted_pixels_without_changing_raw_mask(self):
        image = torch.full((1, 100, 200, 3), 51.0 / 255.0, dtype=torch.float32)
        full_output, _, full_mask = self.render(image, opacity=1.0)
        half_output, _, half_mask = self.render(image, opacity=0.5)
        expected = image * (1.0 - full_mask.unsqueeze(-1) * 0.5)
        expected += (1.0 - image) * (full_mask.unsqueeze(-1) * 0.5)
        self.assertTrue(torch.equal(half_mask, full_mask))
        self.assertTrue(torch.allclose(half_output, expected, atol=2.0 / 255.0))
        self.assertTrue(torch.allclose(full_output[0][full_mask[0] > 0.99], torch.full((3,), 0.8), atol=1.0 / 255.0))

    def test_auto_scale_and_long_text_width_limit(self):
        small = torch.zeros((1, 100, 200, 3), dtype=torch.float32)
        large = torch.zeros((1, 200, 400, 3), dtype=torch.float32)
        _, _, small_mask = self.render(small, text="Watermark")
        _, _, large_mask = self.render(large, text="Watermark")
        small_box = torch.nonzero(small_mask[0] > 0.05)
        large_box = torch.nonzero(large_mask[0] > 0.05)
        small_height = int(small_box[:, 0].max() - small_box[:, 0].min() + 1)
        large_height = int(large_box[:, 0].max() - large_box[:, 0].min() + 1)
        self.assertGreater(large_height, small_height)

        _, _, long_mask = self.render(small, text="This watermark is deliberately very long", max_width_percent=35.0)
        long_box = torch.nonzero(long_mask[0] > 0.05)
        rendered_width = int(long_box[:, 1].max() - long_box[:, 1].min() + 1)
        self.assertLessEqual(rendered_width, 72)

    def test_empty_text_and_zero_opacity_are_exact_identity(self):
        image = torch.rand((2, 48, 64, 3), dtype=torch.float32)
        for values in ({"text": ""}, {"opacity": 0.0}):
            output, _, mask = self.render(image, **values)
            self.assertTrue(torch.equal(output, image))
            self.assertEqual(float(mask.max()), 0.0)
            self.assertEqual(output.dtype, torch.float32)


if __name__ == "__main__":
    unittest.main()
