import unittest

import torch
from PIL import Image, ImageDraw

from ComfyUI_TUT_Nodes.core.fonts import font_options
from ComfyUI_TUT_Nodes.nodes.image.text_animation import ANIMATIONS, TUT_动态文字序列
from ComfyUI_TUT_Nodes.nodes.image.text_geometry import TRANSFORMS, TUT_文字变形, TUT_文字沿路径
from ComfyUI_TUT_Nodes.core.text_layout import sample_mask_path


class TUTTextSequenceTests(unittest.TestCase):
    def setUp(self):
        self.font = font_options()[0]
        self.image = torch.zeros((2, 72, 128, 3), dtype=torch.float32)

    def test_every_text_animation_is_input_major(self):
        node = TUT_动态文字序列()
        for animation in ANIMATIONS:
            image, _, mask = node.animate_text(
                self.image, "AB", self.font, 28, animation, "字符", 4, 1, 1,
                "从左", "center", "center", 4, 0, 0, "white",
            )
            self.assertEqual(tuple(image.shape), (8, 72, 128, 3), animation)
            self.assertEqual(tuple(mask.shape), (8, 72, 128), animation)
            self.assertGreater(float(mask.max()), 0.0, animation)

    def test_every_text_transform_preserves_background_size(self):
        node = TUT_文字变形()
        for transform in TRANSFORMS:
            image, _, mask = node.warp_text(
                self.image, "TUT", self.font, 30, transform, 0.4, 1.0,
                "水平", "center", "center", 4, 0, 0, 0, 0, "white",
            )
            self.assertEqual(tuple(image.shape), (2, 72, 128, 3), transform)
            self.assertEqual(tuple(mask.shape), (2, 72, 128), transform)

    def test_text_on_path_broadcast_and_short_path_error(self):
        path = torch.zeros((1, 72, 128), dtype=torch.float32)
        path[:, 34:38, 8:120] = 1.0
        image, _, mask = TUT_文字沿路径().text_on_path(
            self.image, path, "AB", self.font, 18, 0.0, 0.0, 0.0,
            False, "保持直立", "截断", "white",
        )
        self.assertEqual(tuple(image.shape), (2, 72, 128, 3))
        self.assertEqual(tuple(mask.shape), (2, 72, 128))

        empty = torch.zeros((1, 72, 128), dtype=torch.float32)
        with self.assertRaises(ValueError):
            TUT_文字沿路径().text_on_path(
                self.image[:1], empty, "AB", self.font, 18, 0.0, 0.0, 0.0,
                False, "保持直立", "报错", "white",
            )

    def test_path_points_are_ordered_and_deterministic(self):
        mask = Image.new("L", (64, 32), 0)
        ImageDraw.Draw(mask).line((4, 16, 58, 16), fill=255, width=3)
        first = sample_mask_path(mask)
        second = sample_mask_path(mask)
        self.assertEqual(first, second)
        self.assertGreater(len(first), 2)
        self.assertLessEqual(max(abs(first[i + 1][0] - first[i][0]) for i in range(len(first) - 1)), 1.0)


if __name__ == "__main__":
    unittest.main()
