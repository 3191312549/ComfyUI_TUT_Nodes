import hashlib
import importlib
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

import torch
import torch.nn.functional as F

from ComfyUI_TUT_Nodes.categories import LATENT_UPSCALING
from ComfyUI_TUT_Nodes.core import latent_upscale
from ComfyUI_TUT_Nodes.core.sesqui_lsr import (
    make_flux,
    make_flux2,
    make_ideogram4,
    make_identity,
    make_sdxl,
)
from ComfyUI_TUT_Nodes.nodes.pending.latent import (
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    TUT_SesquiLatentUpscale,
)


class _ResizeModel:
    """Cheap stand-in that exercises all adaptor and layout code."""

    def __call__(self, value, target_hw):
        return F.interpolate(value, size=target_hw, mode="nearest")


class SesquiNodeContractTests(unittest.TestCase):
    def test_node_contract_mapping_and_formal_category(self):
        self.assertEqual(NODE_CLASS_MAPPINGS, {
            "TUT_SesquiLatentUpscale": TUT_SesquiLatentUpscale,
        })
        self.assertEqual(
            NODE_DISPLAY_NAME_MAPPINGS["TUT_SesquiLatentUpscale"],
            "TUT_SesquiLSR潜空间放大",
        )
        self.assertTrue(TUT_SesquiLatentUpscale.__name__.startswith("TUT_"))
        self.assertEqual(TUT_SesquiLatentUpscale.CATEGORY, LATENT_UPSCALING)
        self.assertEqual(TUT_SesquiLatentUpscale.RETURN_TYPES, ("LATENT",))
        self.assertEqual(TUT_SesquiLatentUpscale.RETURN_NAMES, ("latent",))
        self.assertEqual(TUT_SesquiLatentUpscale.FUNCTION, "upscale")

        required = TUT_SesquiLatentUpscale.INPUT_TYPES()["required"]
        self.assertEqual(tuple(required), ("latent", "model_format", "scale", "half_precision"))
        self.assertEqual(required["latent"], ("LATENT",))
        self.assertEqual(
            required["model_format"][0],
            ["SDXL", "Flux", "Flux2", "Ideogram 4", "Wan 2.1"],
        )
        self.assertEqual(required["model_format"][1]["default"], "SDXL")
        self.assertEqual(
            required["scale"],
            ("FLOAT", {"default": 1.5, "min": 1.0, "max": 2.0, "step": 0.05}),
        )
        self.assertEqual(required["half_precision"], ("BOOLEAN", {"default": True}))

    def test_import_is_lazy_and_never_downloads(self):
        with mock.patch.object(
            latent_upscale.urllib.request,
            "urlretrieve",
            side_effect=AssertionError("模块导入不得访问网络"),
        ) as download:
            importlib.reload(latent_upscale)
        download.assert_not_called()

    def test_pinned_weight_manifest_has_four_verified_files(self):
        self.assertEqual(
            latent_upscale.UPSTREAM_COMMIT,
            "befae004248c403f38b76b9f65fd43b901ea3eaa",
        )
        self.assertEqual(len(latent_upscale.MODEL_INFO), 4)
        self.assertEqual(set(latent_upscale.FORMAT_CONFIG), {
            "SDXL", "Flux", "Flux2", "Ideogram 4", "Wan 2.1",
        })
        for filename, info in latent_upscale.MODEL_INFO.items():
            with self.subTest(filename=filename):
                self.assertGreater(info["size"], 0)
                self.assertRegex(info["sha256"], r"^[0-9a-f]{64}$")
                self.assertIn(latent_upscale.UPSTREAM_COMMIT, info["url"])
                self.assertTrue(info["url"].endswith("/" + filename))
        self.assertEqual(
            latent_upscale.FORMAT_CONFIG["Flux2"]["model_file"],
            latent_upscale.FORMAT_CONFIG["Ideogram 4"]["model_file"],
        )


class SesquiAdaptorTests(unittest.TestCase):
    def test_sdxl_and_flux_affine_values_and_round_trip(self):
        value4 = torch.linspace(-2.0, 2.0, 4 * 3 * 5).reshape(1, 4, 3, 5)
        sdxl = make_sdxl()
        self.assertEqual(sdxl.external_channels, 4)
        self.assertTrue(torch.allclose(sdxl.to_vae_latent(value4), value4 * 0.13025))
        self.assertTrue(torch.allclose(
            sdxl.from_vae_latent(sdxl.to_vae_latent(value4)), value4, atol=1e-6,
        ))

        value16 = torch.linspace(-1.0, 1.0, 16 * 3 * 5).reshape(1, 16, 3, 5)
        flux = make_flux()
        expected = (value16 - 0.1159) * 0.3611
        self.assertEqual(flux.external_channels, 16)
        self.assertTrue(torch.allclose(flux.to_vae_latent(value16), expected))
        self.assertTrue(torch.allclose(
            flux.from_vae_latent(flux.to_vae_latent(value16)), value16, atol=1e-6,
        ))

    def test_flux2_and_ideogram_pack_round_trip_and_target_size(self):
        packed = torch.arange(2 * 128 * 3 * 5, dtype=torch.float32).reshape(2, 128, 3, 5)
        for factory in (make_flux2, make_ideogram4):
            with self.subTest(factory=factory.__name__):
                adaptor = factory()
                unpacked = adaptor.to_vae_latent(packed)
                self.assertEqual(adaptor.external_channels, 128)
                self.assertEqual(tuple(unpacked.shape), (2, 32, 6, 10))
                self.assertEqual(adaptor.vae_target_size((7, 11)), (14, 22))
                restored = adaptor.from_vae_latent(unpacked)
                self.assertEqual(tuple(restored.shape), tuple(packed.shape))
                self.assertTrue(torch.allclose(restored, packed, atol=2e-5, rtol=2e-5))

    def test_wan_comfyui_adaptor_is_identity(self):
        adaptor = make_identity(16)
        value = torch.randn((2, 16, 3, 5))
        self.assertEqual(adaptor.external_channels, 16)
        self.assertEqual(adaptor.vae_target_size((8, 10)), (8, 10))
        self.assertTrue(torch.equal(adaptor.to_vae_latent(value), value))
        self.assertTrue(torch.equal(adaptor.from_vae_latent(value), value))


class SesquiLayoutAndExecutionTests(unittest.TestCase):
    def _run(self, samples, model_format, scale, **metadata):
        latent = {"samples": samples, **metadata}
        with mock.patch.object(
            latent_upscale,
            "load_model",
            return_value=(_ResizeModel(), latent_upscale.FORMAT_CONFIG[model_format]["adaptor_fn"]()),
        ), mock.patch.object(
            latent_upscale, "get_torch_device", return_value=torch.device("cpu")
        ):
            return latent_upscale.upscale_latent(latent, model_format, scale, True)

    def test_all_five_formats_4d_and_python_round_dimensions(self):
        channels = {"SDXL": 4, "Flux": 16, "Flux2": 128, "Ideogram 4": 128, "Wan 2.1": 16}
        for model_format, count in channels.items():
            with self.subTest(model_format=model_format):
                source = torch.randn((2, count, 5, 7), dtype=torch.float32)
                output = self._run(source, model_format, 1.5)
                # Python round(): round(7.5) == 8 and round(10.5) == 10.
                self.assertEqual(tuple(output["samples"].shape), (2, count, 8, 10))
                self.assertEqual(output["samples"].dtype, source.dtype)
                self.assertEqual(output["samples"].device, source.device)

    def test_scale_boundaries_batch_and_odd_dimensions(self):
        source = torch.randn((3, 4, 5, 7), dtype=torch.float64)
        same = self._run(source, "SDXL", 1.0)["samples"]
        doubled = self._run(source, "SDXL", 2.0)["samples"]
        self.assertEqual(tuple(same.shape), (3, 4, 5, 7))
        self.assertEqual(tuple(doubled.shape), (3, 4, 10, 14))
        self.assertEqual(same.dtype, torch.float64)

    def test_bcthw_and_btchw_are_restored(self):
        bcthw = torch.randn((2, 16, 3, 5, 7))
        btchw = bcthw.permute(0, 2, 1, 3, 4).contiguous()
        out_bcthw = self._run(bcthw, "Wan 2.1", 2.0)["samples"]
        out_btchw = self._run(btchw, "Wan 2.1", 2.0)["samples"]
        self.assertEqual(tuple(out_bcthw.shape), (2, 16, 3, 10, 14))
        self.assertEqual(tuple(out_btchw.shape), (2, 3, 16, 10, 14))
        self.assertTrue(torch.equal(out_bcthw.permute(0, 2, 1, 3, 4), out_btchw))

    def test_noise_mask_nearest_resize_and_all_metadata_preserved(self):
        samples = torch.randn((2, 4, 5, 7))
        noise_mask = torch.tensor([[[[0, 1], [1, 0]]]], dtype=torch.bool)
        batch_index = torch.tensor([8, 9])
        marker = object()
        latent = {
            "samples": samples, "noise_mask": noise_mask,
            "batch_index": batch_index, "custom": marker,
        }
        with mock.patch.object(
            latent_upscale,
            "load_model",
            return_value=(_ResizeModel(), make_sdxl()),
        ), mock.patch.object(
            latent_upscale, "get_torch_device", return_value=torch.device("cpu")
        ):
            output = latent_upscale.upscale_latent(latent, "SDXL", 2.0, True)
        self.assertIsNot(output, latent)
        self.assertIs(latent["samples"], samples)
        self.assertIs(latent["noise_mask"], noise_mask)
        self.assertEqual(tuple(output["noise_mask"].shape), (1, 1, 10, 14))
        self.assertEqual(output["noise_mask"].dtype, torch.bool)
        self.assertFalse(bool(output["noise_mask"][0, 0, 0, 0]))
        self.assertTrue(output["noise_mask"][0, 0, 0, -1])
        self.assertIs(output["batch_index"], batch_index)
        self.assertIs(output["custom"], marker)

    def test_flatten_restore_helpers_and_errors(self):
        source = torch.randn((2, 16, 3, 5, 7))
        flat, layout = latent_upscale.flatten_to_4d(source, 16)
        self.assertEqual(tuple(flat.shape), (6, 16, 5, 7))
        self.assertTrue(torch.equal(latent_upscale.restore_from_4d(flat, layout), source))
        with self.assertRaisesRegex(ValueError, "4D 或 5D"):
            latent_upscale.flatten_to_4d(torch.zeros((4, 5, 6)), 4)
        with self.assertRaisesRegex(ValueError, "通道"):
            latent_upscale.flatten_to_4d(torch.zeros((1, 8, 5, 7)), 4)
        with self.assertRaisesRegex(ValueError, "通道轴"):
            latent_upscale.flatten_to_4d(torch.zeros((1, 8, 9, 5, 7)), 16)

    def test_validation_and_node_chinese_error_context(self):
        invalid = (
            ({}, "SDXL", 1.5),
            ({"samples": torch.zeros((1, 4, 5))}, "SDXL", 1.5),
            ({"samples": torch.zeros((1, 4, 2, 5))}, "SDXL", 1.5),
            ({"samples": torch.zeros((1, 4, 5, 7))}, "SDXL", 0.9),
            ({"samples": torch.zeros((1, 4, 5, 7))}, "unknown", 1.5),
        )
        for latent, model_format, scale in invalid:
            with self.subTest(model_format=model_format, scale=scale):
                with self.assertRaises((ValueError, RuntimeError)):
                    latent_upscale.upscale_latent(latent, model_format, scale, False)
        with mock.patch(
            "ComfyUI_TUT_Nodes.nodes.pending.latent.upscale_latent",
            side_effect=ValueError("测试错误"),
        ):
            with self.assertRaisesRegex(ValueError, "潜空间放大失败.*测试错误.*通道"):
                TUT_SesquiLatentUpscale().upscale(
                    {"samples": torch.zeros((1, 4, 5, 7))}, "SDXL", 1.5, True,
                )


class SesquiDtypeDownloadAndCacheTests(unittest.TestCase):
    def tearDown(self):
        latent_upscale.clear_model_cache()

    def test_dtype_selection_cpu_cuda_bf16_and_fp16(self):
        self.assertEqual(
            latent_upscale.resolve_dtype(True, torch.device("cpu")), torch.float32,
        )
        self.assertEqual(
            latent_upscale.resolve_dtype(False, torch.device("cuda")), torch.float32,
        )
        with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=True):
            self.assertEqual(
                latent_upscale.resolve_dtype(True, torch.device("cuda")), torch.bfloat16,
            )
        with mock.patch.object(torch.cuda, "is_bf16_supported", return_value=False):
            self.assertEqual(
                latent_upscale.resolve_dtype(True, torch.device("cuda")), torch.float16,
            )

    def test_download_hash_atomic_reuse_and_failure_cleanup(self):
        payload = b"fake-sesqui-checkpoint"
        digest = hashlib.sha256(payload).hexdigest()
        filename = "fake.safetensors"
        good_info = {
            filename: {
                "size": len(payload), "sha256": digest,
                "url": "https://invalid.test/fake.safetensors",
            },
        }
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(latent_upscale, "MODEL_INFO", good_info), \
             mock.patch.object(latent_upscale, "_model_directory", return_value=Path(directory)), \
             mock.patch.object(
                 latent_upscale.urllib.request, "urlretrieve",
                 side_effect=lambda _url, path: Path(path).write_bytes(payload),
             ) as download:
            path = latent_upscale.ensure_sesqui_model(filename)
            self.assertEqual(path.read_bytes(), payload)
            self.assertEqual(latent_upscale.ensure_sesqui_model(filename), path)
            self.assertEqual(download.call_count, 1)
            self.assertFalse(any(Path(directory).glob("*.tmp")))

        bad_info = {filename: {**good_info[filename], "sha256": "0" * 64}}
        with tempfile.TemporaryDirectory() as directory, \
             mock.patch.object(latent_upscale, "MODEL_INFO", bad_info), \
             mock.patch.object(latent_upscale, "_model_directory", return_value=Path(directory)), \
             mock.patch.object(
                 latent_upscale.urllib.request, "urlretrieve",
                 side_effect=lambda _url, path: Path(path).write_bytes(payload),
             ):
            with self.assertRaisesRegex(RuntimeError, "SHA256 校验失败"):
                latent_upscale.ensure_sesqui_model(filename)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_existing_corrupt_file_is_rejected_without_network(self):
        filename = "fake.safetensors"
        info = {
            filename: {
                "size": 4, "sha256": hashlib.sha256(b"good").hexdigest(),
                "url": "https://invalid.test/fake.safetensors",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            Path(directory, filename).write_bytes(b"bad")
            with mock.patch.object(latent_upscale, "MODEL_INFO", info), \
                 mock.patch.object(latent_upscale, "_model_directory", return_value=Path(directory)), \
                 mock.patch.object(latent_upscale.urllib.request, "urlretrieve") as download:
                with self.assertRaisesRegex(RuntimeError, "大小校验失败"):
                    latent_upscale.ensure_sesqui_model(filename)
                download.assert_not_called()

    def test_strict_weight_loading_and_single_active_cache(self):
        class FakeUpscaler:
            instances = []

            def __init__(self, in_channels):
                self.in_channels = in_channels
                self.strict = None
                FakeUpscaler.instances.append(self)

            def load_state_dict(self, state, strict):
                self.state = state
                self.strict = strict

            def to(self, **kwargs):
                self.to_kwargs = kwargs
                return self

            def eval(self):
                return self

            def requires_grad_(self, enabled):
                self.requires_grad = enabled
                return self

        safe_pkg = types.ModuleType("safetensors")
        safe_torch = types.ModuleType("safetensors.torch")
        safe_torch.load_file = mock.Mock(return_value={"weight": torch.tensor(1.0)})
        safe_pkg.torch = safe_torch
        fake_path = Path("C:/mock/upscaler.safetensors")
        latent_upscale.clear_model_cache()
        with mock.patch.dict(sys.modules, {
            "safetensors": safe_pkg, "safetensors.torch": safe_torch,
        }), mock.patch.object(
            latent_upscale, "ensure_sesqui_model", return_value=fake_path,
        ), mock.patch.object(latent_upscale, "LatentUpscaler", FakeUpscaler):
            first, _ = latent_upscale.load_model("SDXL", torch.float32, torch.device("cpu"))
            again, _ = latent_upscale.load_model("SDXL", torch.float32, torch.device("cpu"))
            flux, _ = latent_upscale.load_model("Flux", torch.float32, torch.device("cpu"))
            flux2, _ = latent_upscale.load_model("Flux2", torch.float32, torch.device("cpu"))
            wan, _ = latent_upscale.load_model("Wan 2.1", torch.float32, torch.device("cpu"))
        self.assertIs(first, again)
        self.assertEqual(len({id(first), id(flux), id(flux2), id(wan)}), 4)
        self.assertEqual([item.in_channels for item in FakeUpscaler.instances], [4, 16, 32, 16])
        self.assertTrue(all(item.strict is True for item in FakeUpscaler.instances))
        self.assertTrue(all(item.requires_grad is False for item in FakeUpscaler.instances))
        self.assertEqual(safe_torch.load_file.call_count, 4)

    def test_missing_safetensors_and_checkpoint_errors_are_chinese(self):
        fake_path = Path("C:/mock/upscaler.safetensors")
        latent_upscale.clear_model_cache()
        with mock.patch.object(
            latent_upscale, "ensure_sesqui_model", return_value=fake_path,
        ), mock.patch.dict(sys.modules, {"safetensors": None, "safetensors.torch": None}):
            with self.assertRaisesRegex(RuntimeError, "缺少 safetensors"):
                latent_upscale.load_model("SDXL", torch.float32, torch.device("cpu"))


if __name__ == "__main__":
    unittest.main()
