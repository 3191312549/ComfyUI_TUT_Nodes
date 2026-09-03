import json
import unittest
from collections import deque
from pathlib import Path
from unittest.mock import patch

import torch

from ComfyUI_TUT_Nodes.categories import IMAGE_COMIC, PENDING_IMAGE_COMIC
from ComfyUI_TUT_Nodes.core.comic import BUBBLE_SHAPES, CUSTOM_LAYOUT, parse_bubble_data, parse_panel_data
from ComfyUI_TUT_Nodes.core.fonts import font_for_text, font_options
from ComfyUI_TUT_Nodes.nodes.pending.comic import (
    NODE_DISPLAY_NAME_MAPPINGS,
    TUT_ComicPanelCanvas,
    TUT_ComicSpeechBubble,
)


class ComicPanelTests(unittest.TestCase):
    def setUp(self):
        self.node = TUT_ComicPanelCanvas()

    def compose(self, images, layout="自动匹配数量", empty_fill="留空", panel_data='{"version":1,"panels":[]}'):
        return self.node.compose(
            images, layout, 240, 320, 12, 8, 4, "#111111", "#ffffff",
            "裁切填充", empty_fill, panel_data,
        )

    def test_interface_auto_layout_and_pagination(self):
        self.assertEqual(self.node.CATEGORY, IMAGE_COMIC)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["TUT_ComicPanelCanvas"], "TUT_漫画分镜画布")
        images = torch.rand((7, 72, 96, 3), dtype=torch.float32)
        output, help_text, panels, borders, canonical = self.compose(images)
        self.assertEqual(tuple(output.shape), (2, 320, 240, 3))
        self.assertEqual(tuple(panels.shape), (2, 320, 240))
        self.assertEqual(tuple(borders.shape), (2, 320, 240))
        self.assertGreater(float(panels.min()), -1e-6)
        self.assertLessEqual(float(panels.max()), 1.0)
        self.assertGreater(float(borders.max()), 0.0)
        self.assertIn("TUT_ComicPanelCanvas", help_text)
        parsed = json.loads(canonical)
        self.assertEqual(parsed["version"], 1)
        self.assertEqual(len(parsed["panels"]), 20)
        self.assertEqual(set(self.node.INPUT_TYPES()["hidden"]), {"prompt", "extra_pnginfo"})

    def test_execution_returns_input_preview_metadata(self):
        images = torch.rand((2, 32, 32, 3), dtype=torch.float32)
        preview = [{"filename": "comic.png", "type": "temp", "subfolder": ""}]
        with patch.object(self.node, "_save_input_previews", return_value=preview):
            result = self.node.compose(
                images, "左右双格", 240, 320, 12, 8, 4, "#111111", "#ffffff",
                "裁切填充", "留空", '{"version":1,"panels":[]}', prompt={}, extra_pnginfo={},
            )
        self.assertEqual(result["ui"]["input_previews"], preview)
        self.assertEqual(len(result["result"]), 5)

    def test_manual_layout_fill_modes_and_focus(self):
        solid = torch.zeros((1, 60, 120, 3), dtype=torch.float32)
        solid[..., 0] = 1.0
        for mode in ("留空", "循环填充", "复制最后一张"):
            output, _, _, _, _ = self.compose(solid, "四宫格", mode)
            self.assertEqual(tuple(output.shape), (1, 320, 240, 3))
            if mode == "留空":
                self.assertGreater(float(output[..., :].mean()), 0.3)
            else:
                self.assertGreater(int((output[..., 0] > 0.9).sum()), 1000)

        gradient = torch.zeros((1, 60, 180, 3), dtype=torch.float32)
        gradient[0, ..., 0] = torch.linspace(0, 1, 180)[None, :]
        left_data = json.dumps({"version": 1, "panels": [{"focus_x": 0, "focus_y": .5, "zoom": 1, "flip": False}]})
        right_data = json.dumps({"version": 1, "panels": [{"focus_x": 1, "focus_y": .5, "zoom": 1, "flip": False}]})
        left = self.compose(gradient, "整页单格", panel_data=left_data)[0]
        right = self.compose(gradient, "整页单格", panel_data=right_data)[0]
        self.assertLess(float(left[..., 0].mean()), float(right[..., 0].mean()))

    def test_custom_panel_rectangles_change_frame_sizes(self):
        images = torch.zeros((2, 80, 80, 3), dtype=torch.float32)
        images[0, ..., 0] = 1.0
        images[1, ..., 1] = 1.0
        panel_data = json.dumps({
            "version": 1,
            "panels": [],
            "layout_overrides": {"左右双格": [[0, 0, .25, 1], [.25, 0, 1, 1]]},
        })
        output, _, _, _, canonical = self.compose(images, "左右双格", panel_data=panel_data)
        self.assertGreater(float(output[0, 160, 40, 0]), .9)
        self.assertGreater(float(output[0, 160, 100, 1]), .9)
        self.assertEqual(json.loads(canonical)["layout_overrides"]["左右双格"][0][2], .25)

    def test_free_layout_accepts_one_to_six_frames(self):
        images = torch.rand((3, 50, 70, 3), dtype=torch.float32)
        data = json.dumps({
            "version": 1, "panels": [],
            "layout_overrides": {CUSTOM_LAYOUT: [[.05, .05, .48, .48], [.52, .05, .95, .48], [.05, .52, .95, .95]]},
        })
        output, _, panel_mask, _, canonical = self.compose(images, CUSTOM_LAYOUT, panel_data=data)
        self.assertEqual(tuple(output.shape), (1, 320, 240, 3))
        self.assertGreater(float(panel_mask.max()), 0.0)
        self.assertEqual(len(json.loads(canonical)["layout_overrides"][CUSTOM_LAYOUT]), 3)

    def test_custom_frames_obey_forced_page_margin(self):
        image = torch.zeros((1, 40, 40, 3), dtype=torch.float32)
        image[..., 0] = 1.0
        data = json.dumps({
            "version": 1, "panels": [],
            "layout_overrides": {CUSTOM_LAYOUT: [[0, 0, 1, 1]]},
        })
        output, _, panel_mask, _, _ = self.compose(image, CUSTOM_LAYOUT, panel_data=data)
        self.assertGreater(float(output[0, 2, 2].mean()), .95)
        self.assertEqual(float(panel_mask[0, 2, 2]), 0.0)
        self.assertGreater(float(output[0, 160, 120, 0]), .9)
        self.assertGreater(float(panel_mask[0, 160, 120]), .9)

    def test_quad_overrides_render_trapezoid_and_concave_shapes(self):
        image = torch.zeros((1, 80, 80, 3), dtype=torch.float32)
        image[..., 0] = 1.0
        for quad in (
            [[.25, .2], [.75, .28], [.68, .8], [.18, .72]],
            [[.2, .2], [.8, .2], [.48, .5], [.2, .8]],
        ):
            data = json.dumps({
                "version": 1, "panels": [],
                "quad_overrides": {CUSTOM_LAYOUT: [quad]},
            })
            output, _, panel_mask, border_mask, canonical = self.compose(image, CUSTOM_LAYOUT, panel_data=data)
            with self.subTest(quad=quad):
                self.assertGreater(float(panel_mask.mean()), .05)
                self.assertEqual(float(panel_mask[0, 20, 20]), 0.0)
                self.assertGreater(float(border_mask.max()), .9)
                self.assertEqual(json.loads(canonical)["quad_overrides"][CUSTOM_LAYOUT][0], quad)
                self.assertGreater(int((output[..., 0] > .9).sum()), 1000)

    def test_quad_diagonal_masks_are_antialiased(self):
        image = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
        image[..., 0] = 1.0
        data = json.dumps({
            "version": 1, "panels": [],
            "quad_overrides": {CUSTOM_LAYOUT: [[[.2, .2], [.78, .31], [.72, .8], [.16, .68]]]},
        })
        output, _, panel_mask, border_mask, _ = self.compose(image, CUSTOM_LAYOUT, panel_data=data)
        soft_panel = (panel_mask > 0.01) & (panel_mask < 0.99)
        soft_border = (border_mask > 0.01) & (border_mask < 0.99)
        self.assertGreater(int(soft_panel.sum()), 20)
        self.assertGreater(int(soft_border.sum()), 20)
        self.assertGreater(int(((output[..., 0] > .01) & (output[..., 0] < .99)).sum()), 20)

    def test_quad_precedence_and_actual_edge_opening(self):
        image = torch.zeros((1, 40, 40, 3), dtype=torch.float32)
        image[..., 0] = 1.0
        quad = [[.3, .25], [.7, .35], [.65, .75], [.25, .65]]
        data = {
            "version": 1,
            "panels": [{"zoom": 2, "open_edges": [True, False, False, False]}],
            "layout_overrides": {CUSTOM_LAYOUT: [[.05, .05, .95, .95]]},
            "quad_overrides": {CUSTOM_LAYOUT: [quad]},
        }
        output, _, panel_mask, border_mask, canonical = self.compose(image, CUSTOM_LAYOUT, panel_data=json.dumps(data))
        self.assertGreater(float(panel_mask[0, 20, 120]), .9)
        self.assertEqual(float(panel_mask[0, 300, 120]), 0.0)
        self.assertGreater(float(output[0, 20, 120, 0]), .9)
        self.assertLess(float(border_mask[0, 96, 120]), .1)
        parsed = json.loads(canonical)
        self.assertEqual(parsed["panels"][0]["open_edges"], [True, False, False, False])
        self.assertTrue(parsed["panels"][0]["overflow_top"])

    def test_quad_vertices_must_obey_forced_page_margin(self):
        image = torch.rand((1, 32, 32, 3), dtype=torch.float32)
        data = json.dumps({
            "version": 1, "panels": [],
            "quad_overrides": {CUSTOM_LAYOUT: [[[.01, .2], [.8, .2], [.8, .8], [.2, .8]]]},
        })
        with self.assertRaisesRegex(ValueError, "超出强制页边距"):
            self.compose(image, CUSTOM_LAYOUT, panel_data=data)

    def test_each_open_edge_allows_only_that_side_to_overflow(self):
        image = torch.zeros((1, 40, 40, 3), dtype=torch.float32)
        image[..., 0] = 1.0
        samples = {
            "overflow_top": ((40, 120), (280, 120)),
            "overflow_bottom": ((280, 120), (40, 120)),
            "overflow_left": ((160, 30), (160, 210)),
            "overflow_right": ((160, 210), (160, 30)),
        }
        for key, (opened_point, closed_point) in samples.items():
            panel = {"focus_x": .5, "focus_y": .5, "zoom": 2, "flip": False, key: True}
            data = json.dumps({
                "version": 1, "panels": [panel],
                "layout_overrides": {CUSTOM_LAYOUT: [[.25, .25, .75, .75]]},
            })
            output, _, panel_mask, _, canonical = self.compose(image, CUSTOM_LAYOUT, panel_data=data)
            oy, ox = opened_point; cy, cx = closed_point
            with self.subTest(edge=key):
                self.assertLess(float(output[0, oy, ox, 1]), .1)
                self.assertGreater(float(panel_mask[0, oy, ox]), .9)
                self.assertGreater(float(output[0, cy, cx, 1]), .9)
                self.assertEqual(json.loads(canonical)["panels"][0][key], True)

    def test_open_image_covers_frame_line_and_layer_order_controls_overlap(self):
        red = torch.zeros((1, 40, 40, 3), dtype=torch.float32); red[..., 0] = 1.0
        open_right = {
            "focus_x": .5, "focus_y": .5, "zoom": 2, "flip": False, "overflow_right": True,
        }
        frame = {"version": 1, "panels": [open_right], "layout_overrides": {CUSTOM_LAYOUT: [[.25, .25, .75, .75]]}}
        opened = self.compose(red, CUSTOM_LAYOUT, panel_data=json.dumps(frame))[0]
        frame["panels"][0]["overflow_right"] = False
        closed = self.compose(red, CUSTOM_LAYOUT, panel_data=json.dumps(frame))[0]
        self.assertGreater(float(opened[0, 160, 178, 0]), .9)
        self.assertLess(float(closed[0, 160, 178, 0]), .2)

        images = torch.zeros((2, 40, 40, 3), dtype=torch.float32)
        images[0, ..., 0] = 1.0; images[1, ..., 1] = 1.0
        panels = [
            {"zoom": 2, "overflow_top": True, "overflow_bottom": True, "overflow_left": True, "overflow_right": True},
            {"zoom": 2, "overflow_top": True, "overflow_bottom": True, "overflow_left": True, "overflow_right": True},
        ]
        base = {"version": 1, "panels": panels, "layout_overrides": {CUSTOM_LAYOUT: [[.2, .2, .6, .7], [.4, .3, .8, .8]]}}
        green_front = self.compose(images, CUSTOM_LAYOUT, panel_data=json.dumps(base))[0]
        base["layer_orders"] = {CUSTOM_LAYOUT: [1, 0]}
        red_front, _, _, _, canonical = self.compose(images, CUSTOM_LAYOUT, panel_data=json.dumps(base))
        self.assertGreater(float(green_front[0, 160, 120, 1]), .9)
        self.assertGreater(float(red_front[0, 160, 120, 0]), .9)
        self.assertEqual(json.loads(canonical)["layer_orders"][CUSTOM_LAYOUT], [1, 0])

    def test_rgba_input_is_composited_over_page_background_instead_of_black(self):
        rgba = torch.zeros((1, 40, 40, 4), dtype=torch.float32)
        rgba[:, 10:30, 10:30, 0] = 1.0
        rgba[:, 10:30, 10:30, 3] = 1.0

        rectangular = self.compose(rgba, "整页单格")[0]
        self.assertTrue(torch.all(rectangular[0, 30, 30] > 0.95))
        self.assertGreater(float(rectangular[0, 160, 120, 0]), 0.95)
        self.assertLess(float(rectangular[0, 160, 120, 1]), 0.05)

        quad_data = json.dumps({
            "version": 1,
            "panels": [],
            "quad_overrides": {
                "整页单格": [[[0.1, 0.1], [0.9, 0.14], [0.86, 0.9], [0.14, 0.86]]]
            },
        })
        quadrilateral = self.compose(rgba, "整页单格", panel_data=quad_data)[0]
        self.assertTrue(torch.all(quadrilateral[0, 60, 60] > 0.95))
        self.assertGreater(float(quadrilateral[0, 160, 120, 0]), 0.95)
        self.assertLess(float(quadrilateral[0, 160, 120, 1]), 0.05)

    def test_panel_json_validation(self):
        panels, canonical = parse_panel_data("")
        self.assertEqual(len(panels), 20)
        self.assertEqual(json.loads(canonical)["version"], 1)
        invalid = (
            ('{"panels":[]}', "version"),
            ('{"version":1,"panels":{}}', "panels"),
            ('{"version":1,"panels":[{"focus_x":-1}]}', "0.0 到 1.0"),
            ('{"version":1,"panels":[{"zoom":"nan"}]}', "有限数值"),
            ('{"version":1,"panels":[{"flip":1}]}', "布尔值"),
            ('{"version":1,"panels":[{"overflow_top":1}]}', "上边缘开放必须是布尔值"),
            ('{"version":1,"panels":[],"layout_overrides":[]}', "layout_overrides"),
            ('{"version":1,"panels":[],"layout_overrides":{"未知":[]}}', "未知模板"),
            ('{"version":1,"panels":[],"layout_overrides":{"左右双格":[[0,0,1,1]]}}', "数量必须为 2"),
            ('{"version":1,"panels":[],"layout_overrides":{"整页单格":[[0,0,0,1]]}}', "必须大于"),
            ('{"version":1,"panels":[],"layout_overrides":{"自由画框":[]}}', "1 到 20"),
            ('{"version":1,"panels":[{"open_edges":[true,false]}]}', "4 个布尔值"),
            ('{"version":1,"panels":[{"open_edges":[true,false,0,false]}]}', "布尔值"),
            ('{"version":1,"panels":[],"quad_overrides":[]}', "quad_overrides"),
            ('{"version":1,"panels":[],"quad_overrides":{"未知":[]}}', "未知模板"),
            ('{"version":1,"panels":[],"quad_overrides":{"左右双格":[[[0,0],[1,0],[1,1],[0,1]]]}}', "数量必须为 2"),
            ('{"version":1,"panels":[],"quad_overrides":{"整页单格":[[[0,0],[1,1],[1,0],[0,1]]]}}', "不能自交"),
            ('{"version":1,"panels":[],"quad_overrides":{"整页单格":[[[0,0],[0.001,0],[0.001,0.001],[0,0.001]]]}}', "面积过小|边过短"),
            ('{"version":1,"panels":[],"quad_overrides":{"整页单格":[[[0,0],[1,0],[1,1],[0,"nan"]]]}}', "有限数值"),
            ('{"version":1,"panels":[],"layer_orders":[]}', "layer_orders"),
            ('{"version":1,"panels":[],"layer_orders":{"未知":[0]}}', "未知模板"),
            ('{"version":1,"panels":[],"layer_orders":{"左右双格":[0,0]}}', "完整排列"),
        )
        for value, message in invalid:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                parse_panel_data(value)


class ComicBubbleTests(unittest.TestCase):
    def setUp(self):
        self.node = TUT_ComicSpeechBubble()
        self.font = font_options()[0]
        self.image = torch.rand((2, 240, 320, 3), dtype=torch.float32)

    def render(self, bubbles, enabled=True, merge_overlaps=False):
        data = json.dumps({"version": 1, "merge_overlaps": merge_overlaps, "bubbles": bubbles}, ensure_ascii=False)
        return self.node.render(self.image, enabled, self.font, data)

    def bubble(self, shape, index=0, text="中文台词自动换行测试"):
        return {
            "id": f"bubble-{index}", "shape": shape,
            "x": .25 + (index % 3) * .25, "y": .25 + (index // 3) * .42,
            "w": .24, "h": .25, "tail_x": .5, "tail_y": .8,
            "text": text, "font_name": self.font, "font_size": 28,
            "text_color": "#111111", "fill_color": "#ffffff",
            "border_color": "#111111", "border_width": 3, "opacity": 1,
        }

    def test_interface_eight_shapes_batch_and_masks(self):
        self.assertEqual(self.node.CATEGORY, PENDING_IMAGE_COMIC)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["TUT_ComicSpeechBubble"], "TUT_[待测试]漫画对话框")
        self.assertEqual(set(self.node.INPUT_TYPES()["hidden"]), {"prompt", "extra_pnginfo"})
        bubbles = [
            self.bubble(shape, index) | {"x": .14 + (index % 4) * .24, "y": .28 + (index // 4) * .44, "w": .19, "h": .24}
            for index, shape in enumerate(BUBBLE_SHAPES)
        ]
        output, help_text, bubble_mask, text_mask, canonical = self.render(bubbles)
        self.assertEqual(tuple(output.shape), tuple(self.image.shape))
        self.assertEqual(tuple(bubble_mask.shape), (2, 240, 320))
        self.assertEqual(tuple(text_mask.shape), (2, 240, 320))
        self.assertGreater(float(bubble_mask.max()), 0.0)
        self.assertGreater(float(text_mask.max()), 0.0)
        self.assertFalse(torch.equal(output, self.image))
        self.assertIn("TUT_ComicSpeechBubble", help_text)
        canonical_bubbles = json.loads(canonical)["bubbles"]
        self.assertEqual(len(canonical_bubbles), 8)
        self.assertNotIn("tail_x", canonical_bubbles[0])
        self.assertNotIn("tail_y", canonical_bubbles[0])
        self.assertEqual(canonical_bubbles[0]["spike_count"], 16)
        self.assertAlmostEqual(canonical_bubbles[0]["spike_depth"], .22)
        self.assertEqual(canonical_bubbles[0]["cloud_lobes"], 10)
        self.assertAlmostEqual(canonical_bubbles[0]["cloud_depth"], .14)
        self.assertEqual(canonical_bubbles[0]["text_direction"], "horizontal")

    def test_cloud_is_filled_single_contour_and_burst_controls_change_shape(self):
        cloud = self.bubble("云朵思考框", text="")
        _, _, cloud_mask, _, _ = self.render([cloud])
        mask = cloud_mask[0].numpy() > .5
        self.assertTrue(mask[60, 80])
        background = ~mask
        reachable = background.copy()
        reachable[:] = False
        queue = deque()
        for y in range(background.shape[0]):
            for x in (0, background.shape[1] - 1):
                if background[y, x] and not reachable[y, x]: reachable[y, x] = True; queue.append((y, x))
        for x in range(background.shape[1]):
            for y in (0, background.shape[0] - 1):
                if background[y, x] and not reachable[y, x]: reachable[y, x] = True; queue.append((y, x))
        while queue:
            y, x = queue.popleft()
            for next_y, next_x in ((y - 1, x), (y + 1, x), (y, x - 1), (y, x + 1)):
                if 0 <= next_y < background.shape[0] and 0 <= next_x < background.shape[1] and background[next_y, next_x] and not reachable[next_y, next_x]:
                    reachable[next_y, next_x] = True; queue.append((next_y, next_x))
        self.assertFalse((background & ~reachable).any())

        few_lobes = cloud | {"cloud_lobes": 6, "cloud_depth": .05}
        many_lobes = cloud | {"cloud_lobes": 16, "cloud_depth": .30}
        few_lobes_mask = self.render([few_lobes])[2]
        many_lobes_mask = self.render([many_lobes])[2]
        self.assertFalse(torch.equal(few_lobes_mask, many_lobes_mask))

        shallow = self.bubble("爆炸喊话框", text="") | {"spike_count": 6, "spike_depth": .05}
        deep = self.bubble("爆炸喊话框", text="") | {"spike_count": 32, "spike_depth": .70}
        shallow_mask = self.render([shallow])[2]
        deep_mask = self.render([deep])[2]
        self.assertFalse(torch.equal(shallow_mask, deep_mask))

    def test_reference_explosion_and_flash_are_distinct_renderable_shapes(self):
        explosion = self.bubble("爆炸对话框", text="") | {"x": .5, "y": .5, "w": .45, "h": .55}
        flash = self.bubble("闪光对话框", text="") | {"x": .5, "y": .5, "w": .45, "h": .55, "border_width": 3}
        explosion_mask = self.render([explosion])[2]
        flash_output, _, flash_mask, _, _ = self.render([flash])
        self.assertGreater(float(explosion_mask.max()), 0.0)
        self.assertGreater(float(flash_mask.max()), 0.0)
        self.assertFalse(torch.equal(explosion_mask, flash_mask))
        self.assertFalse(torch.equal(flash_output, self.image))

        no_rays = self.render([flash | {"border_width": 0}])[2]
        self.assertLess(float(no_rays.sum()), float(flash_mask.sum()))

        self.image = torch.zeros((1, 240, 320, 3), dtype=torch.float32)
        gradient = self.render([flash | {"fill_color": "#ffffff", "border_width": 0}])[0][0]
        center = float(gradient[120, 160].mean())
        fading_edge = float(gradient[120, 220].mean())
        canvas = float(gradient[120, 236].mean())
        self.assertGreater(center, fading_edge)
        self.assertGreater(fading_edge, canvas)

        self.image = torch.ones((1, 240, 320, 3), dtype=torch.float32)
        inward_rays = self.render([flash | {"fill_color": "#ffffff", "border_width": 3}])[0][0]
        without_rays = self.render([flash | {"fill_color": "#ffffff", "border_width": 0}])[0][0]
        inward_region = (slice(116, 125), slice(202, 211), slice(None))
        self.assertLess(float(inward_rays[inward_region].min()), float(without_rays[inward_region].min()) - .5)

    def test_burst_parameter_defaults_and_limits(self):
        default_bubbles, _ = parse_bubble_data(
            '{"version":1,"bubbles":[{"shape":"爆炸喊话框"}]}', self.font,
        )
        self.assertEqual(default_bubbles[0]["spike_count"], 16)
        self.assertAlmostEqual(default_bubbles[0]["spike_depth"], .22)
        self.assertEqual(default_bubbles[0]["cloud_lobes"], 10)
        self.assertAlmostEqual(default_bubbles[0]["cloud_depth"], .14)
        limits, _ = parse_bubble_data(
            '{"version":1,"bubbles":['
            '{"shape":"爆炸喊话框","spike_count":6,"spike_depth":0.05},'
            '{"shape":"爆炸喊话框","spike_count":32,"spike_depth":0.70}]}', self.font,
        )
        self.assertEqual((limits[0]["spike_count"], limits[1]["spike_count"]), (6, 32))
        self.assertEqual((limits[0]["spike_depth"], limits[1]["spike_depth"]), (.05, .70))

    def test_vertical_text_supports_both_column_directions(self):
        bubble = self.bubble("椭圆对白框", text="甲乙丙丁戊己庚辛") | {"x": .5, "y": .5, "w": .35, "h": .5, "font_size": 28}
        left_to_right = self.render([bubble | {"text_direction": "vertical_ltr"}])[3]
        right_to_left = self.render([bubble | {"text_direction": "vertical_rtl"}])[3]
        horizontal = self.render([bubble | {"text_direction": "horizontal"}])[3]
        self.assertGreater(float(left_to_right.max()), 0.0)
        self.assertFalse(torch.equal(left_to_right, right_to_left))
        self.assertFalse(torch.equal(left_to_right, horizontal))

    def test_missing_chinese_glyphs_use_a_compatible_fallback_font(self):
        latin_only = next(
            (token for token in font_options() if font_for_text(token, "中文") != token),
            None,
        )
        if latin_only is None:
            self.skipTest("没有可用于验证中文回退的纯西文字体")
        fallback = font_for_text(latin_only, "中文台词")
        self.assertNotEqual(fallback, latin_only)
        bubble = self.bubble("椭圆对白框", text="中文台词") | {"font_name": latin_only}
        self.assertGreater(float(self.render([bubble])[3].max()), 0.0)

    def test_overlapping_bubbles_can_share_one_outer_outline(self):
        self.image = torch.ones((1, 240, 320, 3), dtype=torch.float32)
        left = self.bubble("椭圆对白框", text="") | {"x": .42, "y": .5, "w": .5, "h": .4, "border_width": 8}
        right = self.bubble("椭圆对白框", 1, text="") | {"x": .58, "y": .5, "w": .5, "h": .4, "border_width": 8}
        separate = self.render([left, right], merge_overlaps=False)[0]
        merged, _, merged_mask, _, canonical = self.render([left, right], merge_overlaps=True)
        overlap_boundary = (slice(None), slice(116, 125), slice(101, 112), slice(None))
        self.assertGreater(float(merged[overlap_boundary].mean()), float(separate[overlap_boundary].mean()) + .2)
        self.assertGreater(float(merged_mask.max()), 0.0)
        self.assertTrue(json.loads(canonical)["merge_overlaps"])

        styled_left = left | {"fill_color": "#ffdddd", "border_width": 0}
        styled_right = right | {"fill_color": "#336699", "border_width": 0}
        unified = self.render([styled_left, styled_right], merge_overlaps=True)[0][0]
        self.assertTrue(torch.allclose(unified[120, 80], unified[120, 240], atol=.01))
        self.assertTrue(torch.allclose(unified[120, 160], unified[120, 240], atol=.01))

    def test_execution_returns_original_input_preview_metadata(self):
        bubble = self.bubble("椭圆对白框")
        preview = [{"filename": "bubble-input.png", "type": "temp", "subfolder": ""}]
        with patch.object(self.node, "_save_input_preview", return_value=preview):
            result = self.node.render(
                self.image, True, self.font,
                json.dumps({"version": 1, "bubbles": [bubble]}, ensure_ascii=False),
                prompt={}, extra_pnginfo={},
            )
        self.assertEqual(result["ui"]["input_previews"], preview)
        self.assertEqual(len(result["result"]), 5)

    def test_empty_text_borderless_and_exact_disabled_bypass(self):
        empty = self.bubble("椭圆对白框", text="")
        output, _, bubble_mask, text_mask, _ = self.render([empty])
        self.assertGreater(float(bubble_mask.max()), 0.0)
        self.assertEqual(float(text_mask.max()), 0.0)
        self.assertFalse(torch.equal(output, self.image))

        borderless = self.bubble("无边框文字")
        _, _, borderless_mask, borderless_text, _ = self.render([borderless])
        self.assertEqual(float(borderless_mask.max()), 0.0)
        self.assertGreater(float(borderless_text.max()), 0.0)

        bypass, _, bypass_bubble, bypass_text, canonical = self.render([empty], enabled=False)
        self.assertIs(bypass, self.image)
        self.assertTrue(torch.equal(bypass, self.image))
        self.assertEqual(float(bypass_bubble.max()), 0.0)
        self.assertEqual(float(bypass_text.max()), 0.0)
        self.assertEqual(json.loads(canonical)["version"], 1)

    def test_bubble_json_validation(self):
        bubbles, canonical = parse_bubble_data("", self.font)
        self.assertEqual(bubbles, [])
        self.assertEqual(json.loads(canonical)["version"], 1)
        self.assertFalse(json.loads(canonical)["merge_overlaps"])
        invalid = (
            ('{"bubbles":[]}', "version"),
            ('{"version":1,"bubbles":{}}', "bubbles"),
            ('{"version":1,"bubbles":[{"shape":"未知"}]}', "形状无效"),
            ('{"version":1,"bubbles":[{"opacity":2}]}', "0.0 到 1.0"),
            ('{"version":1,"bubbles":[{"x":"Infinity"}]}', "有限数值"),
            ('{"version":1,"bubbles":[{"fill_color":"oops"}]}', "颜色"),
            ('{"version":1,"bubbles":[{"spike_count":5}]}', "6 到 32"),
            ('{"version":1,"bubbles":[{"spike_count":33}]}', "6 到 32"),
            ('{"version":1,"bubbles":[{"spike_count":16.5}]}', "必须是整数"),
            ('{"version":1,"bubbles":[{"spike_depth":0.04}]}', "0.05 到 0.7"),
            ('{"version":1,"bubbles":[{"spike_depth":0.71}]}', "0.05 到 0.7"),
            ('{"version":1,"bubbles":[{"cloud_lobes":5}]}', "6 到 16"),
            ('{"version":1,"bubbles":[{"cloud_lobes":17}]}', "6 到 16"),
            ('{"version":1,"bubbles":[{"cloud_lobes":10.5}]}', "必须是整数"),
            ('{"version":1,"bubbles":[{"cloud_depth":0.04}]}', "0.05 到 0.3"),
            ('{"version":1,"bubbles":[{"cloud_depth":0.31}]}', "0.05 到 0.3"),
            ('{"version":1,"merge_overlaps":"true","bubbles":[]}', "merge_overlaps 必须是布尔值"),
            ('{"version":1,"bubbles":[{"text_direction":"diagonal"}]}', "文字方向无效"),
        )
        for value, message in invalid:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                parse_bubble_data(value, self.font)


class ComicFrontendTests(unittest.TestCase):
    def test_editors_and_chinese_labels_are_registered(self):
        plugin_root = Path(__file__).resolve().parents[1]
        editor = (plugin_root / "js" / "tut_comic_editors.js").read_text(encoding="utf-8")
        canvas = (plugin_root / "js" / "tut_comic_canvas.js").read_text(encoding="utf-8")
        labels = (plugin_root / "js" / "tut_chinese_ui.js").read_text(encoding="utf-8")
        for node_id in ("TUT_ComicPanelCanvas", "TUT_ComicSpeechBubble"):
            self.assertIn(node_id, editor)
        for action in ("添加", "删除", "上移一层", "下移一层", "水平翻转", "重置当前格", "调整画框", "调整镜头", "恢复模板"):
            self.assertIn(action, editor)
        for feature in ("layout_overrides", "canvas_width", "canvas_height", "1024 × 1536"):
            self.assertIn(feature, editor)
        for feature in (
            "添加画框", "自由画框", "switchToCustomLayout", "画布与间距", "画框与镜头", "页面样式",
            "tut-comic-monitor", "tut-comic-sidebar", "tut-comic-sidebar-tabs", "tut-comic-workspace",
            "border-radius:0", "Math.max(node.size?.[0] || 0, 1100)", "tut-comic-fields", "16:9", "9:16",
            "强制页边距", "remapOverrides", "pageBounds", "input_previews", "drawPreview", "__tutComicSetPreviews",
            "水平翻转", "重置镜头", 'addEventListener("wheel"', "passive: false", "frameStroke", "borderColor.value",
            "吸附对齐", "snap_enabled", "alignmentTargets", "snapMove", "snapCoordinate", "#ff4fc8",
            "逐边开放", "边 1", "边 4", "open_edges", "quad_overrides", "rectToQuad", "validQuad", "pointInQuad", "edgeExtrusion", "#f59e0b",
            "ResizeObserver", "resizeObserver.disconnect",
            "edgeMotion", "snapEdgeMove", "quadInPage", "hoveredCanvasEdge", "ew-resize", "ns-resize", "nwse-resize", "nesw-resize",
            "overflow_top", "overflow_bottom", "overflow_left", "overflow_right",
            "镜头图层", "置于顶层", "上移一层", "下移一层", "置于底层", "恢复默认顺序", "layer_orders", "moveSelectedLayer", "renderLayerList",
        ):
            self.assertIn(feature, canvas)
        self.assertNotIn("drawBtn.disabled = !free", canvas)
        self.assertNotIn("deleteBtn.disabled = !free", canvas)
        for feature in (
            "tut-bubble-workspace", "tut-bubble-monitor", "tut-bubble-sidebar", "tut-bubble-tabs",
            "内容", "样式", "图层", "添加对话框", "置于顶层", "置于底层", "对话框图层",
            "input_previews", "__tutBubbleSetPreviews", "drawBubbleText", "drawBubblePath",
            "搜索字体名称或路径", "未找到匹配字体", "当前字体未改变", "filterFonts", "hideStore(fontWidget)",
            "尖角数量", "尖角深度", "spike_count", "spike_depth", "tut-bubble-burst",
            "云瓣数量", "云瓣起伏", "cloud_lobes", "cloud_depth", "tut-bubble-cloud",
            "store.options.hidden = true", "store.draw = () => {}",
            "quadraticCurveTo", "function drawCanvas",
            "合并重叠边框", "merge_overlaps", "相交气泡合为一个轮廓", "最上层气泡的外观",
            "text_direction", "vertical_ltr", "vertical_rtl", "竖排列方向", "列从左到右", "列从右到左",
            "爆炸对话框", "闪光对话框", "drawFlashRays", "drawFlashFill", "createRadialGradient", "bezierCurveTo", "rayCount = 96",
            "/tut_nodes/fonts/catalog", "/tut_nodes/fonts/file", "FontFace", "document.fonts.add",
            "preview_family", "预览已使用所选字体", "旧路径字体无法在浏览器预览",
        ):
            self.assertIn(feature, editor)
        self.assertNotIn("context.arc", editor)
        for removed in ("tail_x", "tail_y", "调整尾巴", "黄色圆点"):
            self.assertNotIn(removed, editor)
        for removed in ("X%", "Y%", "宽%", "高%", "#ffd400"):
            self.assertNotIn(removed, canvas)
        for field in ("panel_data", "bubble_data", "page_margin", "empty_fill", "default_font"):
            self.assertIn(f"{field}:", labels)

    def test_bubble_drag_only_redraws_canvas(self):
        plugin_root = Path(__file__).resolve().parents[1]
        editor = (plugin_root / "js" / "tut_comic_editors.js").read_text(encoding="utf-8")
        bubble_editor = editor.index("function installBubbleEditor")
        move_start = editor.index("const move = (event) =>", bubble_editor)
        move_end = editor.index("const up = (event) =>", move_start)
        move_handler = editor[move_start:move_end]
        self.assertIn("drawCanvas();", move_handler)
        self.assertNotIn("render();", move_handler)
        self.assertNotIn("filterFonts", move_handler)
        self.assertNotIn("renderLayerList", move_handler)

    def test_normal_bubble_draw_does_not_treat_foreach_array_as_draw_options(self):
        plugin_root = Path(__file__).resolve().parents[1]
        editor = (plugin_root / "js" / "tut_comic_editors.js").read_text(encoding="utf-8")
        self.assertIn("data.bubbles.forEach((bubble, index) => drawBubble(bubble, index));", editor)
        self.assertNotIn("data.bubbles.forEach(drawBubble);", editor)

    def test_bubble_preview_never_draws_text_with_generic_sans_serif(self):
        plugin_root = Path(__file__).resolve().parents[1]
        editor = (plugin_root / "js" / "tut_comic_editors.js").read_text(encoding="utf-8")
        start = editor.index("function drawBubbleText")
        end = editor.index("function installBubbleEditor", start)
        text_renderer = editor[start:end]
        self.assertIn("loadedComicFontFamily", text_renderer)
        self.assertNotIn("sans-serif", text_renderer)

    def test_preview_scales_pixel_sizes_from_source_image_to_editor_canvas(self):
        plugin_root = Path(__file__).resolve().parents[1]
        editor = (plugin_root / "js" / "tut_comic_editors.js").read_text(encoding="utf-8")
        self.assertIn("sourceFontSize * previewScale", editor)
        self.assertIn("2 * previewScale", editor)
        self.assertIn("scaledBorderWidth = (style.border_width || 0) * previewScale", editor)
        self.assertIn("nextWidth / naturalWidth", editor)
        self.assertIn("nextHeight / naturalHeight", editor)
        self.assertNotIn("Math.min(72, Number(bubble.font_size)", editor)


if __name__ == "__main__":
    unittest.main()
