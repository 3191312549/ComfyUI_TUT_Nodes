import unittest
from pathlib import Path

import torch

from ComfyUI_TUT_Nodes.categories import TOOLS_BATCH
from ComfyUI_TUT_Nodes.nodes.pending.batch import (
    NODE_DISPLAY_NAME_MAPPINGS,
    TUT_ImageToBatch,
)


class ImageToBatchTests(unittest.TestCase):
    def setUp(self):
        self.node = TUT_ImageToBatch()

    def test_interface_category_and_single_output(self):
        inputs = self.node.INPUT_TYPES()
        self.assertEqual(inputs["required"], {})
        self.assertEqual(tuple(inputs["optional"]), ("image_1",))
        self.assertEqual(self.node.RETURN_TYPES, ("IMAGE",))
        self.assertEqual(self.node.RETURN_NAMES, ("images",))
        self.assertEqual(self.node.CATEGORY, TOOLS_BATCH)
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS["TUT_ImageToBatch"], "TUT_图像到批次")

    def test_single_input_is_returned_as_one_batch(self):
        image = torch.rand((2, 8, 12, 3), dtype=torch.float32)
        output = self.node.make_batch(image_1=image)[0]
        self.assertTrue(torch.equal(output, image))

    def test_inputs_are_sorted_resized_and_batches_are_expanded(self):
        first = torch.zeros((1, 8, 12, 3), dtype=torch.float32)
        second = torch.ones((2, 4, 6, 3), dtype=torch.float32)
        third = torch.full((1, 8, 12, 3), 0.5, dtype=torch.float32)
        output = self.node.make_batch(image_3=third, image_1=first, image_2=second)[0]
        self.assertEqual(tuple(output.shape), (4, 8, 12, 3))
        self.assertEqual(float(output[0].max()), 0.0)
        self.assertEqual(float(output[1:3].min()), 1.0)
        self.assertTrue(torch.allclose(output[3], torch.full_like(output[3], 0.5)))

    def test_clear_errors_and_ten_input_limit(self):
        with self.assertRaisesRegex(ValueError, "至少需要连接一个"):
            self.node.make_batch()
        image = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "最多只能有 10 个"):
            self.node.make_batch(image_1=image, image_11=image)
        with self.assertRaisesRegex(ValueError, "非空 IMAGE 批次"):
            self.node.make_batch(image_1=torch.zeros((4, 4, 3)))

    def test_rgb_rgba_and_grayscale_are_automatically_normalized(self):
        rgba = torch.zeros((1, 4, 6, 4), dtype=torch.float32)
        rgba[..., 0] = 0.75; rgba[..., 3] = 0.25
        rgb = torch.full((1, 2, 3, 3), 0.5, dtype=torch.float32)
        mixed = self.node.make_batch(image_1=rgba, image_2=rgb)[0]
        self.assertEqual(tuple(mixed.shape), (2, 4, 6, 4))
        self.assertTrue(torch.allclose(mixed[0, ..., 3], torch.full((4, 6), 0.25)))
        self.assertTrue(torch.equal(mixed[1, ..., 3], torch.ones((4, 6))))

        grayscale = torch.full((1, 4, 6, 1), 0.4, dtype=torch.float32)
        gray_batch = self.node.make_batch(image_1=grayscale, image_2=rgb)[0]
        self.assertEqual(tuple(gray_batch.shape), (2, 4, 6, 3))
        self.assertTrue(torch.allclose(gray_batch[0], torch.full((4, 6, 3), 0.4)))

    def test_frontend_automatically_adds_and_removes_inputs(self):
        source = (
            Path(__file__).resolve().parents[1] / "js" / "tut_image_to_batch.js"
        ).read_text(encoding="utf-8")
        self.assertIn('const MAX_INPUTS = 10', source)
        self.assertIn("onConnectionsChange", source)
        self.assertIn("addInput", source)
        self.assertIn("removeInput", source)
        self.assertIn("connected.length < MAX_INPUTS", source)


if __name__ == "__main__":
    unittest.main()
