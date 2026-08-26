import io
import sys
import types
import unittest
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from TUT_Nodes.nodes.image.keying import (
    TUT_AIKeying,
    TUT_ColorKeying,
    TUT_SAMMaskKeying,
)
from TUT_Nodes.core.keying import clear_rembg_session_cache


class KeyingTestCase(unittest.TestCase):
    def assert_keying_outputs(self, result, batch, height, width):
        self.assertEqual(len(result), 4)
        foreground, show_help, foreground_mask, background_mask = result
        self.assertEqual(tuple(foreground.shape), (batch, height, width, 3))
        self.assertEqual(tuple(foreground_mask.shape), (batch, height, width))
        self.assertEqual(tuple(background_mask.shape), (batch, height, width))
        self.assertEqual(foreground.dtype, torch.float32)
        self.assertEqual(foreground_mask.dtype, torch.float32)
        self.assertEqual(background_mask.dtype, torch.float32)
        self.assertIsInstance(show_help, str)
        self.assertTrue(show_help.startswith("TUT_"))
        for tensor in (foreground, foreground_mask, background_mask):
            self.assertGreaterEqual(float(tensor.min()), 0.0)
            self.assertLessEqual(float(tensor.max()), 1.0)
        self.assertTrue(torch.allclose(
            foreground_mask + background_mask,
            torch.ones_like(foreground_mask),
            atol=1e-6,
        ))
        return foreground, foreground_mask, background_mask


class ColorKeyingTests(KeyingTestCase):
    @staticmethod
    def green_screen(batch=1):
        image = torch.zeros((batch, 24, 32, 3), dtype=torch.float32)
        image[..., 1] = 1.0
        image[:, 6:18, 10:22] = torch.tensor([1.0, 0.0, 0.0])
        return image

    def test_green_screen_separates_subject_and_background(self):
        image = self.green_screen()
        result = TUT_ColorKeying().key(
            image, "green", "#00FF00", 20.0, 10.0, 0.6, 0, 0.0, False,
        )
        foreground, mask, _ = self.assert_keying_outputs(result, 1, 24, 32)

        self.assertLess(float(mask[0, 2, 2]), 0.05)
        self.assertGreater(float(mask[0, 12, 16]), 0.95)
        self.assertLess(float(foreground[0, 2, 2].abs().max()), 0.05)
        self.assertGreater(float(foreground[0, 12, 16, 0]), 0.9)

    def test_custom_color_and_batch_are_supported(self):
        image = self.green_screen(batch=2)
        result = TUT_ColorKeying().key(
            image, "custom", "#00ff00", 20.0, 8.0, 0.0, 0, 0.0, False,
        )
        _, mask, _ = self.assert_keying_outputs(result, 2, 24, 32)
        self.assertTrue(torch.equal(mask[0], mask[1]))


class SAMMaskKeyingTests(KeyingTestCase):
    @staticmethod
    def image_batch(batch=2):
        image = torch.ones((batch, 16, 20, 3), dtype=torch.float32)
        if batch > 1:
            image[1, ..., 0] = 0.25
        return image

    def test_singleton_mask_broadcasts_and_is_applied(self):
        image = self.image_batch(2)
        sam_mask = torch.zeros((1, 16, 20), dtype=torch.float32)
        sam_mask[:, 4:12, 5:15] = 1.0

        result = TUT_SAMMaskKeying().key(image, sam_mask, 0, 0.0, False)
        foreground, mask, _ = self.assert_keying_outputs(result, 2, 16, 20)

        self.assertTrue(torch.equal(mask[0], mask[1]))
        self.assertTrue(torch.allclose(foreground, image * mask.unsqueeze(-1)))

    def test_black_white_and_inverted_masks(self):
        image = self.image_batch(1)
        node = TUT_SAMMaskKeying()

        black = torch.zeros((1, 16, 20), dtype=torch.float32)
        black_result = node.key(image, black, 0, 0.0, False)
        black_foreground, black_mask, _ = self.assert_keying_outputs(black_result, 1, 16, 20)
        self.assertTrue(torch.equal(black_foreground, torch.zeros_like(image)))
        self.assertTrue(torch.equal(black_mask, black))

        white = torch.ones((1, 16, 20), dtype=torch.float32)
        white_result = node.key(image, white, 0, 0.0, False)
        white_foreground, white_mask, _ = self.assert_keying_outputs(white_result, 1, 16, 20)
        self.assertTrue(torch.equal(white_foreground, image))
        self.assertTrue(torch.equal(white_mask, white))

        inverted_result = node.key(image, white, 0, 0.0, True)
        inverted_foreground, inverted_mask, _ = self.assert_keying_outputs(inverted_result, 1, 16, 20)
        self.assertTrue(torch.equal(inverted_foreground, torch.zeros_like(image)))
        self.assertTrue(torch.equal(inverted_mask, black))

    def test_grow_shrink_and_feather_refine_mask(self):
        image = self.image_batch(1)
        sam_mask = torch.zeros((1, 16, 20), dtype=torch.float32)
        sam_mask[:, 6:10, 8:12] = 1.0
        node = TUT_SAMMaskKeying()

        _, _, original, _ = node.key(image, sam_mask, 0, 0.0, False)
        _, _, grown, _ = node.key(image, sam_mask, 2, 0.0, False)
        _, _, shrunk, _ = node.key(image, sam_mask, -1, 0.0, False)
        _, _, feathered, _ = node.key(image, sam_mask, 0, 2.0, False)

        self.assertGreater(float(grown.sum()), float(original.sum()))
        self.assertLess(float(shrunk.sum()), float(original.sum()))
        self.assertTrue(torch.any((feathered > 0.0) & (feathered < 1.0)))

    def test_incompatible_batches_raise_clear_error(self):
        image = self.image_batch(2)
        sam_mask = torch.zeros((3, 16, 20), dtype=torch.float32)
        with self.assertRaisesRegex((ValueError, RuntimeError), "批次|batch"):
            TUT_SAMMaskKeying().key(image, sam_mask, 0, 0.0, False)


class AIKeyingTests(KeyingTestCase):
    def setUp(self):
        clear_rembg_session_cache()

    def tearDown(self):
        clear_rembg_session_cache()

    @staticmethod
    def fake_rembg(calls):
        module = types.ModuleType("rembg")

        def new_session(model_name, providers=None, **kwargs):
            session = {"model_name": model_name, "providers": providers}
            calls["sessions"].append(session)
            return session

        def remove(data, session=None, only_mask=False, **kwargs):
            calls["removes"].append({
                "session": session,
                "only_mask": only_mask,
                "kwargs": kwargs,
            })
            if isinstance(data, bytes):
                with Image.open(io.BytesIO(data)) as opened:
                    width, height = opened.size
            elif isinstance(data, Image.Image):
                width, height = data.size
            else:
                array = np.asarray(data)
                height, width = array.shape[:2]
            mask = np.zeros((height, width), dtype=np.uint8)
            mask[height // 4:height * 3 // 4, width // 4:width * 3 // 4] = 255
            return mask

        module.new_session = new_session
        module.remove = remove
        return module

    def test_ai_keying_uses_mocked_rembg_without_network(self):
        calls = {"sessions": [], "removes": []}
        fake_rembg = self.fake_rembg(calls)
        image = torch.rand((2, 18, 26, 3), generator=torch.Generator().manual_seed(7))

        with patch.dict(sys.modules, {"rembg": fake_rembg}):
            result = TUT_AIKeying().key(
                image, "birefnet-general-lite", "cpu", 0, 0.0, False,
            )

        foreground, mask, _ = self.assert_keying_outputs(result, 2, 18, 26)
        self.assertEqual(len(calls["sessions"]), 1)
        self.assertEqual(calls["sessions"][0]["model_name"], "birefnet-general-lite")
        providers = calls["sessions"][0]["providers"]
        if providers is not None:
            self.assertIn("CPUExecutionProvider", providers)
        self.assertEqual(len(calls["removes"]), 2)
        self.assertTrue(all(call["only_mask"] for call in calls["removes"]))
        self.assertTrue(all(call["session"] is calls["sessions"][0] for call in calls["removes"]))
        self.assertTrue(torch.allclose(foreground, image * mask.unsqueeze(-1)))

    def test_ai_session_is_reused_for_same_model_and_provider(self):
        calls = {"sessions": [], "removes": []}
        fake_rembg = self.fake_rembg(calls)
        image = torch.ones((1, 12, 14, 3), dtype=torch.float32)
        node = TUT_AIKeying()

        with patch.dict(sys.modules, {"rembg": fake_rembg}):
            node.key(image, "u2net", "cpu", 0, 0.0, False)
            node.key(image, "u2net", "cpu", 0, 0.0, False)

        self.assertEqual(len(calls["sessions"]), 1)
        self.assertEqual(len(calls["removes"]), 2)

    def test_missing_rembg_has_clear_chinese_error(self):
        image = torch.ones((1, 8, 8, 3), dtype=torch.float32)
        real_import = __import__("importlib").import_module

        def import_without_rembg(name, *args, **kwargs):
            if name == "rembg":
                raise ModuleNotFoundError("rembg")
            return real_import(name, *args, **kwargs)

        with patch("TUT_Nodes.core.keying.importlib.import_module", side_effect=import_without_rembg):
            with self.assertRaisesRegex(RuntimeError, "rembg"):
                TUT_AIKeying().key(
                    image, "birefnet-general-lite", "cpu", 0, 0.0, False,
                )


if __name__ == "__main__":
    unittest.main()
