import json
import unittest
from unittest.mock import patch

import torch

from ComfyUI_TUT_Nodes.categories import IMAGE_COMIC
from ComfyUI_TUT_Nodes.core.comic import CUSTOM_LAYOUT, parse_panel_data
from ComfyUI_TUT_Nodes.nodes.pending.comic import (
    NODE_DISPLAY_NAME_MAPPINGS,
    TUT_ComicPanelCanvas,
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
        self.assertEqual(len(parsed["panels"]), 6)
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

    def test_panel_json_validation(self):
        panels, canonical = parse_panel_data("")
        self.assertEqual(len(panels), 6)
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
            ('{"version":1,"panels":[],"layout_overrides":{"自由画框":[]}}', "1 到 6"),
            ('{"version":1,"panels":[],"layer_orders":[]}', "layer_orders"),
            ('{"version":1,"panels":[],"layer_orders":{"未知":[0]}}', "未知模板"),
            ('{"version":1,"panels":[],"layer_orders":{"左右双格":[0,0]}}', "完整排列"),
        )
        for value, message in invalid:
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, message):
                parse_panel_data(value)
