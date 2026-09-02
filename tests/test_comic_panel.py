import json
import unittest
from pathlib import Path
from unittest.mock import patch

import torch

from ComfyUI_TUT_Nodes.categories import IMAGE_COMIC, PENDING_IMAGE_COMIC
from ComfyUI_TUT_Nodes.core.comic import BUBBLE_SHAPES, CUSTOM_LAYOUT, parse_bubble_data, parse_panel_data
from ComfyUI_TUT_Nodes.core.fonts import font_options
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

    def render(self, bubbles, enabled=True):
        data = json.dumps({"version": 1, "bubbles": bubbles}, ensure_ascii=False)
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

    def test_interface_six_shapes_batch_and_masks(self):
        self.assertEqual(self.node.CATEGORY, PENDING_IMAGE_COMIC)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["TUT_ComicSpeechBubble"], "TUT_[待测试]漫画对话框")
        self.assertEqual(set(self.node.INPUT_TYPES()["hidden"]), {"prompt", "extra_pnginfo"})
        bubbles = [self.bubble(shape, index) for index, shape in enumerate(BUBBLE_SHAPES)]
        output, help_text, bubble_mask, text_mask, canonical = self.render(bubbles)
        self.assertEqual(tuple(output.shape), tuple(self.image.shape))
        self.assertEqual(tuple(bubble_mask.shape), (2, 240, 320))
        self.assertEqual(tuple(text_mask.shape), (2, 240, 320))
        self.assertGreater(float(bubble_mask.max()), 0.0)
        self.assertGreater(float(text_mask.max()), 0.0)
        self.assertFalse(torch.equal(output, self.image))
        self.assertIn("TUT_ComicSpeechBubble", help_text)
        canonical_bubbles = json.loads(canonical)["bubbles"]
        self.assertEqual(len(canonical_bubbles), 6)
        self.assertNotIn("tail_x", canonical_bubbles[0])
        self.assertNotIn("tail_y", canonical_bubbles[0])

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
        invalid = (
            ('{"bubbles":[]}', "version"),
            ('{"version":1,"bubbles":{}}', "bubbles"),
            ('{"version":1,"bubbles":[{"shape":"未知"}]}', "形状无效"),
            ('{"version":1,"bubbles":[{"opacity":2}]}', "0.0 到 1.0"),
            ('{"version":1,"bubbles":[{"x":"Infinity"}]}', "有限数值"),
            ('{"version":1,"bubbles":[{"fill_color":"oops"}]}', "颜色"),
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
            "绘制画框", "自由画框", "画布与间距", "画框与镜头", "页面样式",
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
        for feature in (
            "tut-bubble-workspace", "tut-bubble-monitor", "tut-bubble-sidebar", "tut-bubble-tabs",
            "内容", "样式", "图层", "添加对话框", "置于顶层", "置于底层", "对话框图层",
            "input_previews", "__tutBubbleSetPreviews", "drawBubbleText", "drawBubblePath",
        ):
            self.assertIn(feature, editor)
        for removed in ("tail_x", "tail_y", "调整尾巴", "黄色圆点"):
            self.assertNotIn(removed, editor)
        for removed in ("X%", "Y%", "宽%", "高%", "#ffd400"):
            self.assertNotIn(removed, canvas)
        for field in ("panel_data", "bubble_data", "page_margin", "empty_fill", "default_font"):
            self.assertIn(f"{field}:", labels)


if __name__ == "__main__":
    unittest.main()
