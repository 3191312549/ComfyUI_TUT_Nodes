import unittest
from unittest.mock import patch

import torch

from TUT_Nodes.categories import IMAGE
from TUT_Nodes.nodes.image.compare import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    TUT_ImageCompare,
)


class ImageCompareTests(unittest.TestCase):
    def test_public_contract(self):
        self.assertEqual(NODE_CLASS_MAPPINGS, {"TUT_ImageCompare": TUT_ImageCompare})
        self.assertEqual(NODE_DISPLAY_NAME_MAPPINGS, {"TUT_ImageCompare": "TUT_图像对比"})
        self.assertEqual(TUT_ImageCompare.CATEGORY, IMAGE)
        self.assertEqual(TUT_ImageCompare.FUNCTION, "compare_images")
        self.assertTrue(TUT_ImageCompare.OUTPUT_NODE)
        self.assertEqual(TUT_ImageCompare.RETURN_TYPES, ())
        self.assertEqual(list(TUT_ImageCompare.INPUT_TYPES()["required"]), ["image_a", "image_b"])

    def test_first_batch_items_are_saved_separately(self):
        image_a = torch.zeros((3, 8, 10, 3), dtype=torch.float32)
        image_b = torch.ones((2, 6, 7, 3), dtype=torch.float32)
        saved = []

        def fake_save(image, prefix, prompt, extra_pnginfo):
            saved.append((image.clone(), prefix, prompt, extra_pnginfo))
            return [{"filename": prefix + "png", "subfolder": "", "type": "temp"}]

        node = TUT_ImageCompare()
        with patch.object(node, "_save_preview", side_effect=fake_save):
            result = node.compare_images(image_a, image_b, prompt={"1": {}}, extra_pnginfo={"x": 1})

        self.assertEqual(len(saved), 2)
        self.assertEqual(tuple(saved[0][0].shape), (1, 8, 10, 3))
        self.assertEqual(tuple(saved[1][0].shape), (1, 6, 7, 3))
        self.assertTrue(torch.equal(saved[0][0], image_a[:1]))
        self.assertTrue(torch.equal(saved[1][0], image_b[:1]))
        self.assertIn("A", saved[0][1])
        self.assertIn("B", saved[1][1])
        self.assertTrue(result["ui"]["a_images"][0]["filename"].startswith("TUT.compare.A"))
        self.assertTrue(result["ui"]["b_images"][0]["filename"].startswith("TUT.compare.B"))

    def test_empty_or_invalid_inputs_raise_chinese_errors(self):
        node = TUT_ImageCompare()
        valid = torch.zeros((1, 4, 4, 3), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "图像 A"):
            node.compare_images(torch.zeros((0, 4, 4, 3)), valid)
        with self.assertRaisesRegex(ValueError, "图像 B"):
            node.compare_images(valid, torch.zeros((4, 4, 3)))


if __name__ == "__main__":
    unittest.main()
