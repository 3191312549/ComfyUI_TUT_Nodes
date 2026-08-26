import unittest
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch

from TUT_Nodes.nodes.video.animation import TUT_GIMMVFIInterpolate


class TUTVideoAnimationTests(unittest.TestCase):
    def test_gimmvfi_interface(self):
        inputs = TUT_GIMMVFIInterpolate.INPUT_TYPES()["required"]
        self.assertEqual(inputs["interpolation_factor"][1]["default"], 2)
        self.assertEqual(inputs["interpolation_factor"][1]["min"], 2)
        self.assertEqual(TUT_GIMMVFIInterpolate.RETURN_TYPES, ("IMAGE", "IMAGE"))
        self.assertEqual(TUT_GIMMVFIInterpolate.RETURN_NAMES, ("images", "flow_tensors"))

    def test_gimmvfi_requires_two_frames(self):
        with self.assertRaisesRegex(ValueError, "至少需要两帧"):
            TUT_GIMMVFIInterpolate().interpolate(
                torch.zeros((1, 32, 32, 3)),
                "gimmvfi_f_arb_lpips_fp32.safetensors",
                "fp32",
                False,
                1.0,
                2,
                0,
                False,
            )

    def test_gimmvfi_validates_channels_and_controls(self):
        node = TUT_GIMMVFIInterpolate()
        arguments = [
            torch.zeros((2, 32, 32, 3)),
            "gimmvfi_f_arb_lpips_fp32.safetensors",
            "fp32",
            False,
            1.0,
            2,
            0,
            False,
        ]
        for index, value, message in (
            (0, torch.zeros((2, 32, 32, 4)), "三通道 RGB"),
            (4, 0.0, "DS 因子"),
            (5, 1, "补帧因子"),
            (6, -1, "随机种子"),
        ):
            invalid = list(arguments)
            invalid[index] = value
            with self.assertRaisesRegex(ValueError, message):
                node.interpolate(*invalid)

    def test_gimmvfi_reports_missing_original_plugin(self):
        with patch.dict("sys.modules", {
            "nodes": SimpleNamespace(NODE_CLASS_MAPPINGS={}),
        }):
            with self.assertRaisesRegex(RuntimeError, "ComfyUI-GIMM-VFI"):
                TUT_GIMMVFIInterpolate().interpolate(
                    torch.zeros((2, 32, 32, 3)),
                    "gimmvfi_f_arb_lpips_fp32.safetensors",
                    "fp32",
                    False,
                    1.0,
                    2,
                    0,
                    False,
                )

    def test_gimmvfi_expands_to_original_nodes(self):
        class FakeNode:
            def __init__(self, node_id, class_type, inputs):
                self.node_id = node_id
                self.class_type = class_type
                self.inputs = inputs

            def out(self, index):
                return [self.node_id, index]

        class FakeGraphBuilder:
            def __init__(self):
                self.nodes = []

            def node(self, class_type, **inputs):
                node = FakeNode(str(len(self.nodes) + 1), class_type, inputs)
                self.nodes.append(node)
                return node

            def finalize(self):
                return {
                    node.node_id: {"class_type": node.class_type, "inputs": node.inputs}
                    for node in self.nodes
                }

        fake_nodes = SimpleNamespace(NODE_CLASS_MAPPINGS={
            "DownloadAndLoadGIMMVFIModel": object,
            "GIMMVFI_interpolate": object,
        })
        fake_package = ModuleType("comfy_execution")
        fake_graph_utils = ModuleType("comfy_execution.graph_utils")
        fake_graph_utils.GraphBuilder = FakeGraphBuilder
        with patch.dict("sys.modules", {
            "nodes": fake_nodes,
            "comfy_execution": fake_package,
            "comfy_execution.graph_utils": fake_graph_utils,
        }):
            result = TUT_GIMMVFIInterpolate().interpolate(
                torch.zeros((2, 32, 32, 3)),
                "gimmvfi_f_arb_lpips_fp32.safetensors",
                "fp32",
                False,
                0.75,
                4,
                123,
                True,
            )

        expanded = list(result["expand"].values())
        self.assertEqual([node["class_type"] for node in expanded], [
            "DownloadAndLoadGIMMVFIModel",
            "GIMMVFI_interpolate",
        ])
        self.assertEqual(expanded[0]["inputs"]["model"], "gimmvfi_f_arb_lpips_fp32.safetensors")
        self.assertEqual(expanded[1]["inputs"]["interpolation_factor"], 4)
        self.assertEqual(expanded[1]["inputs"]["seed"], 123)
        self.assertTrue(expanded[1]["inputs"]["output_flows"])
        self.assertEqual(tuple(result["result"]), (["2", 0], ["2", 1]))


if __name__ == "__main__":
    unittest.main()
