import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from ComfyUI_TUT_Nodes.nodes.image.text import TUT_DrawText
from ComfyUI_TUT_Nodes.nodes.tools.help import TUT_NodeHelp, format_node_help


class TUTNodeHelpTests(unittest.TestCase):
    def test_help_node_accepts_raw_wildcard_link(self):
        input_types = TUT_NodeHelp.INPUT_TYPES()
        self.assertEqual(input_types["required"]["node"][0], "*")
        self.assertTrue(input_types["required"]["node"][1]["rawLink"])

    def test_formats_node_description_inputs_and_outputs(self):
        document = format_node_help("TUT_DrawText", TUT_DrawText, "TUT_绘制文字")
        self.assertIn("TUT_绘制文字", document)
        self.assertIn("在纯色画布上绘制", document)
        self.assertIn("image_width", document)
        self.assertIn("text_mask", document)

    def test_resolves_upstream_class_from_raw_link_and_prompt(self):
        fake_nodes = SimpleNamespace(
            NODE_CLASS_MAPPINGS={"TUT_DrawText": TUT_DrawText},
            NODE_DISPLAY_NAME_MAPPINGS={"TUT_DrawText": "TUT_绘制文字"},
        )
        prompt = {"upstream": {"class_type": "TUT_DrawText", "inputs": {}}}
        with patch.dict(sys.modules, {"nodes": fake_nodes}):
            result = TUT_NodeHelp().show_help(["upstream", 0], prompt=prompt)

        self.assertEqual(result["result"][0], result["ui"]["text"][0])
        self.assertIn("TUT_绘制文字", result["result"][0])
        self.assertIn("TUT_DrawText", result["result"][0])


if __name__ == "__main__":
    unittest.main()
