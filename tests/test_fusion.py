import unittest

import torch

from ComfyUI_TUT_Nodes.categories import IMAGE_COMPOSITE, IMAGE_KEYING
from ComfyUI_TUT_Nodes.nodes.image.fusion import (
    NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS, TUT_ChannelBoolean,
    TUT_CornerPinComposite, TUT_DepthMerge, TUT_DifferenceKeying,
    TUT_DisplaceComposite, TUT_LightWrapComposite, TUT_MatteFinesse,
)


class FusionNodeTests(unittest.TestCase):
    def setUp(self):
        self.a = torch.zeros((1, 24, 32, 3), dtype=torch.float32); self.a[..., 0] = .8
        self.b = torch.zeros((1, 30, 40, 3), dtype=torch.float32); self.b[..., 2] = .6
        self.mask = torch.zeros((1, 24, 32), dtype=torch.float32); self.mask[:, 4:20, 6:26] = 1

    def assert_valid(self, value, shape=None):
        self.assertEqual(value.dtype, torch.float32)
        self.assertTrue(torch.isfinite(value).all())
        self.assertGreaterEqual(float(value.min()), 0); self.assertLessEqual(float(value.max()), 1)
        if shape: self.assertEqual(tuple(value.shape), shape)

    def test_mappings_contracts_and_categories(self):
        expected = {"TUT_DifferenceKeying", "TUT_MatteFinesse", "TUT_LightWrapComposite", "TUT_DepthMerge", "TUT_CornerPinComposite", "TUT_ChannelBoolean", "TUT_DisplaceComposite"}
        self.assertEqual(set(NODE_CLASS_MAPPINGS), expected); self.assertEqual(set(NODE_DISPLAY_NAME_MAPPINGS), expected)
        self.assertTrue(all(name.startswith("TUT_") for name in NODE_DISPLAY_NAME_MAPPINGS.values()))
        self.assertEqual(TUT_DifferenceKeying.CATEGORY, IMAGE_KEYING); self.assertEqual(TUT_MatteFinesse.CATEGORY, IMAGE_KEYING)
        for cls in (TUT_LightWrapComposite, TUT_DepthMerge, TUT_CornerPinComposite, TUT_ChannelBoolean, TUT_DisplaceComposite): self.assertEqual(cls.CATEGORY, IMAGE_COMPOSITE)
        self.assertEqual(TUT_DifferenceKeying.RETURN_NAMES, ("foreground", "show_help", "foreground_mask", "background_mask", "difference_image"))

    def test_difference_key_spaces_and_complement(self):
        current = self.b.clone(); current[:, 8:20, 10:25, 0] = 1
        for space in ("RGB", "Lab", "亮度"):
            result = TUT_DifferenceKeying().key(current, self.b, space, .03, .08, 0, 0, 0, False)
            self.assertEqual(len(result), 5); self.assert_valid(result[0], (1, 30, 40, 3)); self.assert_valid(result[2], (1, 30, 40))
            self.assertTrue(torch.allclose(result[2] + result[3], torch.ones_like(result[2])))
            self.assertGreater(float(result[2][:, 8:20, 10:25].mean()), .3)
        same = TUT_DifferenceKeying().key(self.b, self.b, "RGB", .01, .1, 0, 0, 0, False)
        self.assertEqual(float(same[2].max()), 0.)

    def test_matte_finesse_outputs_holes_and_levels(self):
        mask = self.mask.clone(); mask[:, 10:12, 14:16] = 0
        result = TUT_MatteFinesse().refine(self.a, mask, 0, 1, 1, True, 1, 1, .5, "green", "#00ff00", .5)
        self.assertEqual(len(result), 5); self.assert_valid(result[0], (1, 24, 32, 3)); self.assert_valid(result[2], (1, 24, 32))
        self.assertTrue(torch.allclose(result[2] + result[3], torch.ones_like(result[2])))

        corner_touching = torch.zeros_like(self.mask)
        corner_touching[:, :12, :14] = 1
        refined = TUT_MatteFinesse().refine(
            self.a, corner_touching, 0, 1, 1, True, 0, 0, 0,
            "green", "#00ff00", 0,
        )[2]
        self.assertEqual(float(refined[:, -1, -1].max()), 0.0)

    def test_light_wrap_is_edge_limited_and_strength_zero(self):
        node = TUT_LightWrapComposite()
        zero = node.composite(self.a, self.b, self.mask, 4, .1, .8, 0, 2, .8)
        self.assertTrue(torch.equal(zero[0], self.b)); self.assertEqual(float(zero[3].max()), 0)
        result = node.composite(self.a, self.b, self.mask, 4, .1, .8, .8, 2, .8)
        self.assert_valid(result[0], (1, 30, 40, 3)); self.assert_valid(result[3], (1, 30, 40))

    def test_depth_merge_near_far_and_equal_depth(self):
        da = torch.ones((1, 24, 32)); db = torch.zeros((1, 30, 40))
        result = TUT_DepthMerge().merge(self.a, self.b, da, db, "白近", 0, 0, 0)
        expected_a = torch.nn.functional.interpolate(self.a.permute(0, 3, 1, 2), (30, 40), mode="bicubic", align_corners=False).permute(0, 2, 3, 1).clamp(0, 1)
        self.assertGreater(float(result[2].mean()), .99); self.assert_valid(result[0], (1, 30, 40, 3))
        equal = TUT_DepthMerge().merge(self.b, self.b, torch.zeros((1, 30, 40)), torch.zeros((1, 30, 40)), "白近", 0, .1, 0)
        self.assertTrue(torch.equal(equal[0], self.b))

    def test_corner_pin_identity_opacity_and_mask(self):
        node = TUT_CornerPinComposite(); args = (0, 0, 1, 0, 1, 1, 0, 1)
        zero = node.pin(self.a, self.b, *args, 0, "normal", 0, self.mask)
        self.assertTrue(torch.equal(zero[0], self.b))
        result = node.pin(self.a, self.b, *args, 1, "normal", 0, self.mask)
        self.assert_valid(result[0], (1, 30, 40, 3)); self.assert_valid(result[3], (1, 30, 40))

    def test_channel_boolean_expressions_and_broadcast(self):
        node = TUT_ChannelBoolean()
        expressions = ("A.对应通道", "B.对应通道", "A+B", "差值")
        result = node.combine(self.a.repeat(2, 1, 1, 1), self.b, *expressions)
        self.assert_valid(result[0], (2, 24, 32, 3)); self.assert_valid(result[2], (2, 24, 32))
        for expression in ("A.R", "A.G", "A.B", "A.A", "A.亮度", "B.R", "B.G", "B.B", "B.A", "B.亮度", "A+B", "A-B", "B-A", "A*B", "最小", "最大", "差值", "0", "1"):
            self.assert_valid(node.combine(self.a, self.b, expression, expression, expression, expression)[0])

    def test_displace_identity_directions_boundaries_and_mask(self):
        gradient = torch.linspace(0, 1, 32).view(1, 1, 32, 1).repeat(1, 24, 1, 3)
        disp = torch.ones((1, 12, 16, 3)); black = torch.zeros((1, 24, 32))
        node = TUT_DisplaceComposite()
        zero = node.apply(gradient, disp, "红", "绿", 0, 0, .5, "双线性", "钳制")
        self.assertTrue(torch.equal(zero[0], gradient))
        masked = node.apply(gradient, disp, "红", "绿", 10, 0, .5, "双线性", "钳制", black)
        self.assertTrue(torch.equal(masked[0], gradient))
        for boundary in ("裁切", "钳制", "镜像", "循环"):
            result = node.apply(gradient, disp, "红", "亮度", 8, -4, .5, "双三次", boundary)
            self.assert_valid(result[0], (1, 24, 32, 3)); self.assert_valid(result[2], (1, 24, 32, 3)); self.assert_valid(result[3], (1, 24, 32))

    def test_invalid_batches_raise_chinese_error(self):
        with self.assertRaisesRegex(ValueError, "批次数量无法匹配"):
            TUT_DifferenceKeying().key(self.a.repeat(2, 1, 1, 1), self.b.repeat(3, 1, 1, 1), "RGB", .1, .1, 0, 0, 0, False)


if __name__ == "__main__": unittest.main()
