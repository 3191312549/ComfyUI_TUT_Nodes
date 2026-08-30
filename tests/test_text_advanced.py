import unittest

import torch

from ComfyUI_TUT_Nodes.core.fonts import font_display_name, font_metadata, font_options, resolve_font
from ComfyUI_TUT_Nodes.core.shaping import harfbuzz_available, shape_text
from ComfyUI_TUT_Nodes.core.text_layout import measure_text_layout, render_layout_mask, segment_text
from ComfyUI_TUT_Nodes.nodes.image.text_advanced import (
    TUT_区域自适应文字,
    TUT_字体预览墙,
    TUT_逐字逐词遮罩,
)


class TUTAdvancedTextTests(unittest.TestCase):
    def setUp(self):
        self.font_name = font_options()[0]

    def test_fit_text_broadcasts_region_mask(self):
        images = torch.zeros((2, 120, 240, 3), dtype=torch.float32)
        region = torch.zeros((1, 120, 240), dtype=torch.float32)
        region[:, 15:105, 20:220] = 1.0
        output, help_text, mask = TUT_区域自适应文字().fit_text(
            images, region, "自动换行文字测试", self.font_name, 10, 72, 8, 3,
            "center", "center", "缩小字号", "white",
        )
        self.assertEqual(tuple(output.shape), (2, 120, 240, 3))
        self.assertEqual(tuple(mask.shape), (2, 120, 240))
        self.assertGreater(float(mask.max()), 0.0)
        self.assertIn("TUT_区域自适应文字", help_text)

    def test_split_text_preserves_graphemes_and_input_major_order(self):
        self.assertEqual(segment_text("A\u0301B", "字符"), ["A\u0301", "B"])
        self.assertEqual(segment_text("👨‍👩‍👧‍👦🇨🇳", "字符"), ["👨‍👩‍👧‍👦", "🇨🇳"])
        images = torch.zeros((2, 96, 160, 3), dtype=torch.float32)
        output, _, masks = TUT_逐字逐词遮罩().split_masks(
            images, "A\u0301B", self.font_name, 42, "字符", False,
            "center", "center", 4, 0, 0, "white",
        )
        self.assertEqual(tuple(output.shape), (4, 96, 160, 3))
        self.assertEqual(tuple(masks.shape), (4, 96, 160))

    def test_font_preview_wall_outputs_page(self):
        image, help_text, mask = TUT_字体预览墙().preview_fonts(
            "TUT 字体", "全部", 1, 2, 2, 240, 100, 28,
            "black", "white",
        )
        self.assertEqual(tuple(image.shape), (1, 100, 480, 3))
        self.assertEqual(tuple(mask.shape), (1, 100, 480))
        self.assertGreater(float(mask.max()), 0.0)
        self.assertIn("TUT_字体预览墙", help_text)

    def test_font_metadata_has_a_display_name(self):
        metadata = font_metadata(self.font_name)
        self.assertTrue(metadata.family)
        self.assertTrue(font_display_name(self.font_name))

    def test_harfbuzz_shapes_and_rasterizes_real_font(self):
        if not harfbuzz_available():
            self.skipTest("uharfbuzz 未安装")
        real_font = next((token for token in font_options() if resolve_font(token) is not None), None)
        if real_font is None:
            self.skipTest("没有可用的轮廓字体")
        shaped = shape_text("A\u0301 fi", resolve_font(real_font), 36)
        self.assertIsNotNone(shaped)
        self.assertGreater(shaped.advance, 0.0)
        self.assertGreater(len(shaped.glyphs), 0)
        rtl = shape_text("\u0633\u0644\u0627\u0645", resolve_font(real_font), 36)
        self.assertEqual(rtl.direction, "rtl")
        layout = measure_text_layout("A\u0301 fi", real_font, 36, (240, 96))
        mask = render_layout_mask(layout)
        self.assertIsNotNone(mask.getbbox())


if __name__ == "__main__":
    unittest.main()
