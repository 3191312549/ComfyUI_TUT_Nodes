import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from PIL import Image

from ComfyUI_TUT_Nodes.categories import IMAGE_COLOR
from ComfyUI_TUT_Nodes.core.color_lut import (
    _load_lut_cached,
    apply_3d_lut,
    apply_lut_data,
    apply_curves,
    clear_lut_cache,
    curve_lut,
    load_lut,
    load_lut_data,
    parse_curve_data,
)
from ComfyUI_TUT_Nodes.nodes.image.color_lut import (
    DEFAULT_CURVES,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    TUT_ColorCurves,
    TUT_LUT,
    TUT_LUTLoaderPreview,
)


def write_cube(path, transform=lambda r, g, b: (r, g, b), size=2, domain=None):
    lines = [f"LUT_3D_SIZE {size}"]
    if domain is not None:
        lines.extend([
            "DOMAIN_MIN " + " ".join(map(str, domain[0])),
            "DOMAIN_MAX " + " ".join(map(str, domain[1])),
        ])
    for blue in range(size):
        for green in range(size):
            for red in range(size):
                values = transform(red / (size - 1), green / (size - 1), blue / (size - 1))
                lines.append(" ".join(f"{value:.7f}" for value in values))
    path.write_text("\n".join(lines), encoding="utf-8")


def write_identity_hald(path, level=2):
    size = level ** 2
    entries = []
    # Standard Hald order: red changes fastest, then green, then blue.
    for blue in range(size):
        for green in range(size):
            for red in range(size):
                entries.append([red, green, blue])
    pixels = np.asarray(entries, dtype=np.float32) * (255.0 / (size - 1))
    side = level ** 3
    Image.fromarray(np.round(pixels).astype(np.uint8).reshape(side, side, 3), "RGB").save(path)


def write_identity_reshade(path, size=2):
    image = np.zeros((size, size * size, 3), dtype=np.uint8)
    for blue in range(size):
        for green in range(size):
            for red in range(size):
                image[green, blue * size + red] = np.round(
                    np.asarray([red, green, blue]) * (255.0 / (size - 1))
                )
    Image.fromarray(image, "RGB").save(path)


def write_identity_3dl(path, size=2, swap=False):
    maximum = 1023
    lines = [" ".join(str(round(index * maximum / (size - 1))) for index in range(size))]
    # Autodesk 3DL advances blue fastest, then green, then red.
    for red in range(size):
        for green in range(size):
            for blue in range(size):
                values = (blue, green, red) if swap else (red, green, blue)
                lines.append(" ".join(str(round(value * maximum / (size - 1))) for value in values))
    path.write_text("\n".join(lines), encoding="utf-8")


class TUTColorLUTTests(unittest.TestCase):
    def setUp(self):
        generator = torch.Generator().manual_seed(20260820)
        self.image = torch.rand((2, 48, 64, 3), generator=generator)
        self.mask = torch.ones((1, 24, 32), dtype=torch.float32)
        clear_lut_cache()

    def test_public_contracts(self):
        expected = {"TUT_ColorCurves", "TUT_LUT", "TUT_LUTLoaderPreview"}
        self.assertEqual(set(NODE_CLASS_MAPPINGS), expected)
        self.assertEqual(set(NODE_DISPLAY_NAME_MAPPINGS), expected)
        for node_id, node_class in NODE_CLASS_MAPPINGS.items():
            self.assertEqual(node_class.CATEGORY, IMAGE_COLOR)
        self.assertEqual(TUT_ColorCurves.RETURN_NAMES, ("image", "show_help", "effect_mask"))
        self.assertEqual(TUT_LUT.RETURN_NAMES, ("image", "show_help", "effect_mask"))
        self.assertEqual(TUT_LUTLoaderPreview.RETURN_TYPES, ("TUT_LUT_DATA",))
        self.assertTrue(TUT_LUTLoaderPreview.OUTPUT_NODE)
        self.assertEqual(
            TUT_ColorCurves.INPUT_TYPES()["required"]["interpolation"][0],
            ["单调三次"],
        )

    def test_public_color_float_parameters_use_sliders(self):
        for node_class in (TUT_ColorCurves, TUT_LUT, TUT_LUTLoaderPreview):
            inputs = node_class.INPUT_TYPES()
            for section in ("required", "optional"):
                for name, definition in inputs.get(section, {}).items():
                    if definition[0] == "FLOAT":
                        self.assertEqual(
                            definition[1].get("display"), "slider",
                            f"{node_class.__name__}.{name} 应显示为滑块",
                        )

    def test_curves_identity_channels_and_strict_json(self):
        node = TUT_ColorCurves()
        identity = node.adjust_curves(self.image, DEFAULT_CURVES, "单调三次", 1.0, self.mask)
        self.assertTrue(torch.equal(identity[0], self.image))
        self.assertEqual(tuple(identity[2].shape), (2, 48, 64))

        payload = json.loads(DEFAULT_CURVES)
        payload["channels"]["RGB"] = [{"x": 0, "y": 1}, {"x": 1, "y": 0}]
        inverted = node.adjust_curves(self.image, json.dumps(payload), "单调三次", 1.0)[0]
        self.assertTrue(torch.allclose(inverted, 1.0 - self.image, atol=2e-6))

        payload = json.loads(DEFAULT_CURVES)
        payload["channels"]["R"] = [{"x": 0, "y": 0}, {"x": 1, "y": 0}]
        channel = node.adjust_curves(self.image, json.dumps(payload), "单调三次", 1.0)[0]
        self.assertEqual(float(channel[..., 0].max()), 0.0)
        self.assertTrue(torch.allclose(channel[..., 1:], self.image[..., 1:], atol=2e-6))

        for bad, message in [
            ("{", "JSON"),
            (json.dumps({"version": 2, "channels": {}}), "version"),
        ]:
            with self.assertRaisesRegex(ValueError, message):
                parse_curve_data(bad)
        duplicate = json.loads(DEFAULT_CURVES)
        duplicate["channels"]["RGB"] = [{"x": 0, "y": 0}, {"x": 0, "y": 0.5}, {"x": 1, "y": 1}]
        with self.assertRaisesRegex(ValueError, "不能重复"):
            parse_curve_data(json.dumps(duplicate))
        with self.assertRaisesRegex(ValueError, "只支持单调三次"):
            node.adjust_curves(self.image, DEFAULT_CURVES, "线性", 1.0)

    def test_monotone_curve_stays_within_neighbor_values(self):
        points = np.asarray([[0.0, 0.0], [0.2, 0.8], [0.7, 0.35], [1.0, 1.0]], dtype=np.float32)
        lut = curve_lut(points, "单调三次", 1024)
        self.assertTrue(np.isfinite(lut).all())
        self.assertGreaterEqual(float(lut.min()), 0.0)
        self.assertLessEqual(float(lut.max()), 1.0)
        for left, right in zip(points[:-1], points[1:]):
            segment = lut[round(left[0] * 1023):round(right[0] * 1023) + 1]
            self.assertGreaterEqual(float(segment.min()), min(left[1], right[1]) - 2e-3)
            self.assertLessEqual(float(segment.max()), max(left[1], right[1]) + 2e-3)
        for x, y in points:
            self.assertAlmostEqual(float(lut[round(float(x) * 1023)]), float(y), delta=2e-3)
        for interpolation in ("线性", "未知"):
            with self.assertRaisesRegex(ValueError, "不支持"):
                curve_lut(points, interpolation, 1024)

    def test_strength_zero_and_black_mask_are_exact_without_lut_io(self):
        node = TUT_LUT()
        strength_zero = node.apply_lut(self.image, "definitely-missing.cube", 0.0)
        self.assertTrue(torch.equal(strength_zero[0], self.image))
        black = torch.zeros((1, 48, 64), dtype=torch.float32)
        masked = node.apply_lut(self.image, "definitely-missing.cube", 1.0, black)
        self.assertTrue(torch.equal(masked[0], self.image))
        self.assertEqual(float(masked[2].max()), 0.0)

    def test_cube_identity_channel_swap_domain_and_cache_invalidation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "grade.cube"
            write_cube(path)
            lut, low, high = load_lut(path)
            frame = self.image[0].numpy()
            self.assertTrue(np.allclose(apply_3d_lut(frame, lut, low, high), frame, atol=2e-6))
            first_misses = _load_lut_cached.cache_info().misses
            load_lut(path)
            self.assertEqual(_load_lut_cached.cache_info().misses, first_misses)

            write_cube(path, lambda r, g, b: (b, g, r))
            stat = path.stat()
            os.utime(path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
            swapped_lut, low, high = load_lut(path)
            self.assertGreater(_load_lut_cached.cache_info().misses, first_misses)
            swapped = apply_3d_lut(frame, swapped_lut, low, high)
            self.assertTrue(np.allclose(swapped[..., 0], frame[..., 2], atol=2e-6))
            self.assertTrue(np.allclose(swapped[..., 2], frame[..., 0], atol=2e-6))

            domain_path = Path(directory) / "domain.cube"
            write_cube(domain_path, domain=([0.25, 0.25, 0.25], [0.75, 0.75, 0.75]))
            domain_lut, low, high = load_lut(domain_path)
            samples = np.asarray([[[0.25, 0.5, 0.75]]], dtype=np.float32)
            corrected = apply_3d_lut(samples, domain_lut, low, high)
            self.assertTrue(np.allclose(corrected, [[[0.0, 0.5, 1.0]]], atol=2e-6))

    def test_standard_hald_r_fastest_indexing(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.png"
            write_identity_hald(path, level=2)
            lut, low, high = load_lut(path)
            self.assertEqual(tuple(lut.shape), (4, 4, 4, 3))
            self.assertTrue(np.allclose(lut[0, 0, 1], [1 / 3, 0, 0], atol=1 / 255))
            self.assertTrue(np.allclose(lut[0, 1, 0], [0, 1 / 3, 0], atol=1 / 255))
            self.assertTrue(np.allclose(lut[1, 0, 0], [0, 0, 1 / 3], atol=1 / 255))
            frame = self.image[0].numpy()
            self.assertTrue(np.allclose(apply_3d_lut(frame, lut, low, high), frame, atol=1 / 255 + 1e-6))

    def test_cube_1d_1dlut_3dl_and_reshade(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cube_1d = root / "identity_1d.cube"
            cube_1d.write_text(
                "LUT_1D_SIZE 3\nDOMAIN_MIN 0 0 0\nDOMAIN_MAX 1 1 1\n"
                "0 0 0\n0.5 0.5 0.5\n1 1 1",
                encoding="utf-8",
            )
            one_d = root / "swap.1dlut"
            one_d.write_text("LUT_1D_SIZE 2\n0 0 0\n0 1 1", encoding="utf-8")
            classic = root / "identity.3dl"
            reshade = root / "identity.png"
            write_identity_3dl(classic)
            write_identity_reshade(reshade)

            frame = self.image[0].numpy()
            self.assertTrue(np.allclose(apply_lut_data(frame, load_lut_data(cube_1d)), frame, atol=2e-6))
            mapped_1d = apply_lut_data(np.ones((1, 1, 3), np.float32), load_lut_data(one_d))
            self.assertTrue(np.allclose(mapped_1d, [[[0, 1, 1]]], atol=1e-6))
            self.assertTrue(np.allclose(apply_lut_data(frame, load_lut_data(classic)), frame, atol=2e-6))
            self.assertTrue(np.allclose(
                apply_lut_data(frame, load_lut_data(reshade)), frame, atol=1 / 255 + 1e-6
            ))

    def test_loader_output_application_and_preview_precedence(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "identity.cube"
            write_cube(path)
            loader = TUT_LUTLoaderPreview()
            self.assertIs(TUT_LUTLoaderPreview.VALIDATE_INPUTS(str(path), "未选择"), True)
            self.assertIn("找不到", TUT_LUTLoaderPreview.VALIDATE_INPUTS(str(path.with_name("missing.cube")), "未选择"))
            no_preview = loader.load_and_preview(str(path), 1.0, "未选择")
            data = no_preview["result"][0]
            self.assertEqual(data["kind"], "3d")
            self.assertEqual(no_preview["ui"]["message"], ["未选择预览图片"])
            applied = TUT_LUT().apply_lut(self.image, "missing.cube", 1.0, lut_data=data)[0]
            self.assertTrue(torch.allclose(applied, self.image, atol=2e-6))

            with patch.object(loader, "_load_preview_image", side_effect=AssertionError("不应加载手动图片")), \
                 patch.object(loader, "_save_preview", side_effect=[[{"filename": "a.png"}], [{"filename": "b.png"}]]):
                preview = loader.load_and_preview(str(path), 1.0, "manual.png", image=self.image[:1])
            self.assertEqual(preview["ui"]["original_images"][0]["filename"], "a.png")
            self.assertEqual(preview["ui"]["graded_images"][0]["filename"], "b.png")

    def test_lut_errors_are_clear(self):
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            bad = directory / "bad.cube"
            bad.write_text("LUT_3D_SIZE 2\n0 0 0", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "数据量"):
                load_lut(bad)
            text = directory / "grade.txt"
            text.write_text("", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "只支持"):
                load_lut(text)

    def test_batch_mask_contract_and_illegal_batch(self):
        curve = TUT_ColorCurves().adjust_curves(self.image, DEFAULT_CURVES, "单调三次", 0.5, self.mask)
        self.assertEqual(tuple(curve[0].shape), (2, 48, 64, 3))
        self.assertEqual(tuple(curve[2].shape), (2, 48, 64))
        bad_mask = torch.ones((3, 48, 64), dtype=torch.float32)
        with self.assertRaisesRegex(ValueError, "批次"):
            TUT_ColorCurves().adjust_curves(self.image, DEFAULT_CURVES, "单调三次", 1.0, bad_mask)


if __name__ == "__main__":
    unittest.main()
