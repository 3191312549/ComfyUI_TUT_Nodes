import unittest

import torch

from ComfyUI_TUT_Nodes.nodes.image.composite import PRESETS, TUT_柔边图层合成


SHAPES = ("圆角", "切角", "波浪", "撕裂")
TRANSITIONS = ("羽化", "噪声溶解", "像素崩解", "墨水扩散")
MATERIALS = ("白边贴纸", "纸张纤维", "烧焦", "玻璃切边")
DEPTHS = ("柔投影", "斜面浮雕", "霓虹边光", "伪厚度")


class TUTSoftLayerCompositeTests(unittest.TestCase):
    def setUp(self):
        self.node = TUT_柔边图层合成()
        self.layer = torch.ones((1, 36, 48, 3), dtype=torch.float32)
        self.layer[..., 1:] = 0.15
        self.background = torch.zeros((1, 64, 80, 3), dtype=torch.float32)
        self.background[..., 2] = 0.35

    def call(self, layer=None, background=None, mask=None, **changes):
        values = dict(
            image_layer=self.layer if layer is None else layer,
            image_background=self.background if background is None else background,
            preset="自定义", size_mode="适应背景", scale=0.65,
            position_x=0.5, position_y=0.5, rotation=0.0, tilt_x=0.0,
            tilt_y=0.0, opacity=1.0, blend_mode="normal",
            shape_mode="圆角", shape_amount=8, shape_strength=1.0,
            transition_mode="羽化", transition_strength=1.0,
            material_mode="关闭", material_strength=1.0,
            depth_mode="柔投影", depth_strength=0.5, edge_width=5,
            edge_color="white", edge_color_hex="#FFFFFF", detail_scale=5.0,
            irregularity=0.65, background_wrap=0.15, background_blur=0.0,
            depth_offset_x=4, depth_offset_y=4, shadow_blur=4.0, seed=23,
            layer_mask=mask,
        )
        values.update(changes)
        return self.node.composite_layer(**values)[0]

    def test_single_image_output_contract(self):
        self.assertEqual(self.node.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(len(PRESETS), 13)
        self.assertEqual(PRESETS[0], "自定义")
        self.assertEqual(self.node.INPUT_TYPES()["required"]["preset"][1]["default"], "自然悬浮")
        output = self.call()
        self.assertEqual(tuple(output.shape), (1, 64, 80, 3))
        self.assertEqual(output.dtype, torch.float32)
        self.assertGreater(float(output.max()), 0.0)
        self.assertGreaterEqual(float(output.min()), 0.0)
        self.assertLessEqual(float(output.max()), 1.0)

    def test_all_blend_modes_render(self):
        for mode in ("normal", "multiply", "screen", "overlay"):
            output = self.call(blend_mode=mode)
            self.assertEqual(tuple(output.shape), (1, 64, 80, 3), mode)

    def test_all_sixteen_edge_effects_render(self):
        groups = (
            ("shape_mode", SHAPES), ("transition_mode", TRANSITIONS),
            ("material_mode", MATERIALS), ("depth_mode", DEPTHS),
        )
        for name, modes in groups:
            for mode in modes:
                output = self.call(**{name: mode})
                self.assertEqual(tuple(output.shape), (1, 64, 80, 3), mode)
                self.assertTrue(torch.isfinite(output).all(), mode)

    def test_mask_transform_and_batch_broadcast(self):
        layers = self.layer.repeat(2, 1, 1, 1)
        mask = torch.zeros((1, 18, 24), dtype=torch.float32)
        mask[:, 3:15, 4:20] = 1.0
        output = self.call(
            layer=layers, mask=mask, rotation=18.0, tilt_x=25.0,
            tilt_y=-20.0, position_x=0.25, position_y=0.7,
        )
        self.assertEqual(tuple(output.shape), (2, 64, 80, 3))

    def test_illegal_batch_raises(self):
        layer = self.layer.repeat(2, 1, 1, 1)
        background = self.background.repeat(3, 1, 1, 1)
        with self.assertRaises(ValueError):
            self.call(layer=layer, background=background)

    def test_zero_opacity_black_mask_and_off_canvas_are_exact_background(self):
        self.assertTrue(torch.equal(self.call(opacity=0.0), self.background))
        black = torch.zeros((1, 36, 48), dtype=torch.float32)
        self.assertTrue(torch.equal(self.call(mask=black), self.background))
        self.assertTrue(torch.equal(self.call(position_x=-1.0, position_y=-1.0), self.background))

    def test_white_mask_matches_unmasked_layer(self):
        white = torch.ones((1, 18, 24), dtype=torch.float32)
        self.assertTrue(torch.equal(self.call(mask=white), self.call()))

    def test_seed_is_deterministic_and_batch_index_changes_noise(self):
        first = self.call(shape_mode="撕裂", transition_mode="噪声溶解")
        second = self.call(shape_mode="撕裂", transition_mode="噪声溶解")
        self.assertTrue(torch.equal(first, second))
        batch = self.call(
            layer=self.layer.repeat(2, 1, 1, 1),
            shape_mode="撕裂", transition_mode="噪声溶解",
        )
        self.assertFalse(torch.equal(batch[0], batch[1]))


if __name__ == "__main__":
    unittest.main()
