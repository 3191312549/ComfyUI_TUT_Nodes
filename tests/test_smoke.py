import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch
from PIL import Image

from ComfyUI_TUT_Nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from ComfyUI_TUT_Nodes.categories import (
    IMAGE,
    IMAGE_ANIMATION,
    IMAGE_COLOR,
    IMAGE_COMPOSITE,
    IMAGE_FILTER,
    IMAGE_KEYING,
    IMAGE_COMIC,
    IMAGE_TEXT,
    LATENT_UPSCALING,
    MODEL_LORA,
    TOOLS_BATCH,
    TOOLS_EXCEL,
    TOOLS_WORKFLOW,
    TOOLS_TEXT,
    VIDEO_ANIMATION,
)
from ComfyUI_TUT_Nodes.core.fonts import font_options
from ComfyUI_TUT_Nodes.core.imaging import render_text_mask
from ComfyUI_TUT_Nodes.nodes.image.text import (
    TUT_CompositeText,
    TUT_DrawText,
    TUT_MaskText,
    TUT_OverlayText,
    TUT_AutoContrastWatermark,
    TUT_SelectFont,
    TUT_SimpleTextWatermark,
    TUT_TextEffect,
)
from ComfyUI_TUT_Nodes.nodes.image.animation import (
    GIF_COLOR_LEVELS,
    GIF_COMPRESSION_PRESETS,
    TUT_SaveAnimatedGIF,
)


EXPECTED_NODE_IDS = {
    "TUT_AnimatedFilterSequence", "TUT_AnimatedTextSequence", "TUT_ComicFilter",
    "TUT_CompositeText", "TUT_DrawText", "TUT_FitTextToRegion",
    "TUT_FontPreviewWall", "TUT_GlassRefractionFilter", "TUT_GlitchArtFilter",
    "TUT_KaleidoscopeFilter", "TUT_MaskText", "TUT_OverlayText",
    "TUT_PixelArtFilter", "TUT_RetroPrintFilter", "TUT_SaveAnimatedGIF",
    "TUT_SelectFont", "TUT_SimpleTextWatermark", "TUT_AutoContrastWatermark", "TUT_SplitTextMasks",
    "TUT_TextEffect", "TUT_TextOnPath", "TUT_WarpText",
    "TUT_SoftLayerComposite",
    "TUT_ColorKeying", "TUT_AIKeying", "TUT_SAMMaskKeying",
    "TUT_DifferenceKeying", "TUT_MatteFinesse", "TUT_LightWrapComposite",
    "TUT_DepthMerge", "TUT_CornerPinComposite", "TUT_ChannelBoolean",
    "TUT_DisplaceComposite", "TUT_AutoColorCorrect", "TUT_AutoColorCorrectAdvanced", "TUT_BasicTone",
    "TUT_BasicColor", "TUT_DetailEnhance", "TUT_HSLBasic", "TUT_ColorMatch", "TUT_FilmTone",
    "TUT_Halation", "TUT_LensDiffusion", "TUT_ColorCompressor",
    "TUT_NodeHelp",
    "TUT_SplitTextBatch",
    "TUT_SelectBatchItem",
    "TUT_DelayPassThrough",
    "TUT_LoadExcel", "TUT_ReadExcelBatch", "TUT_ReadExcelMerged", "TUT_ReadExcelSingleLine",
    "TUT_ColorCurves", "TUT_LUT", "TUT_LUTLoaderPreview",
    "TUT_ImageCompare",
    "TUT_GIMMVFIInterpolate",
    "TUT_ImageToBatch",
    "TUT_ComicPanelCanvas",
    "TUT_SesquiLatentUpscale",
    "TUT_LoraStrengthTester",
}


class TUTNodesSmokeTests(unittest.TestCase):
    def setUp(self):
        self.font_name = font_options()[0]

    def test_registry_and_category(self):
        self.assertEqual(set(NODE_CLASS_MAPPINGS), EXPECTED_NODE_IDS)
        self.assertEqual(set(NODE_CLASS_MAPPINGS), set(NODE_DISPLAY_NAME_MAPPINGS))
        self.assertTrue(all(name.startswith("TUT_") for name in NODE_CLASS_MAPPINGS))
        self.assertTrue(all(name.startswith("TUT_") for name in NODE_DISPLAY_NAME_MAPPINGS.values()))
        self.assertTrue(all(node.CATEGORY == IMAGE_TEXT for node in (
            TUT_OverlayText, TUT_DrawText, TUT_MaskText, TUT_CompositeText,
            TUT_SimpleTextWatermark, TUT_AutoContrastWatermark, TUT_SelectFont, TUT_TextEffect,
        )))
        self.assertEqual(TUT_SaveAnimatedGIF.CATEGORY, IMAGE_ANIMATION)
        self.assertEqual(TUT_SaveAnimatedGIF.RETURN_TYPES, ("STRING",))
        self.assertEqual(TUT_SaveAnimatedGIF.RETURN_NAMES, ("filename",))
        for node_id, node_class in NODE_CLASS_MAPPINGS.items():
            if node_id == "TUT_SaveAnimatedGIF":
                continue
            if node_id == "TUT_ImageCompare":
                self.assertEqual(node_class.CATEGORY, IMAGE)
                continue
            if node_id == "TUT_GIMMVFIInterpolate":
                self.assertEqual(node_class.CATEGORY, VIDEO_ANIMATION)
                continue
            if node_id == "TUT_ComicPanelCanvas":
                self.assertEqual(node_class.CATEGORY, IMAGE_COMIC)
                continue
            if node_id == "TUT_SesquiLatentUpscale":
                self.assertEqual(node_class.CATEGORY, LATENT_UPSCALING)
                continue
            if node_id == "TUT_LoraStrengthTester":
                self.assertEqual(node_class.CATEGORY, MODEL_LORA)
                continue
            if node_id in {
                "TUT_SoftLayerComposite", "TUT_LightWrapComposite", "TUT_DepthMerge",
                "TUT_CornerPinComposite", "TUT_ChannelBoolean", "TUT_DisplaceComposite",
            }:
                expected = IMAGE_COMPOSITE
            elif node_id in {
                "TUT_ColorKeying", "TUT_AIKeying", "TUT_SAMMaskKeying",
                "TUT_DifferenceKeying", "TUT_MatteFinesse",
            }:
                expected = IMAGE_KEYING
            elif node_id in {
                "TUT_AutoColorCorrect", "TUT_AutoColorCorrectAdvanced", "TUT_BasicTone", "TUT_BasicColor",
                "TUT_DetailEnhance", "TUT_HSLBasic", "TUT_ColorCurves",
                "TUT_ColorMatch", "TUT_FilmTone", "TUT_Halation",
                "TUT_LensDiffusion", "TUT_ColorCompressor", "TUT_LUT",
                "TUT_LUTLoaderPreview",
            }:
                expected = IMAGE_COLOR
            elif node_id in {"TUT_NodeHelp", "TUT_SplitTextBatch"}:
                expected = TOOLS_TEXT
            elif node_id in {
                "TUT_LoadExcel", "TUT_ReadExcelBatch", "TUT_ReadExcelMerged",
                "TUT_ReadExcelSingleLine",
            }:
                expected = TOOLS_EXCEL
            elif node_id in {"TUT_SelectBatchItem", "TUT_ImageToBatch"}:
                expected = TOOLS_BATCH
            elif node_id == "TUT_DelayPassThrough":
                expected = TOOLS_WORKFLOW
            else:
                expected = IMAGE_FILTER if "Filter" in node_id or node_id in {
                "TUT_ComicFilter", "TUT_KaleidoscopeFilter", "TUT_PixelArtFilter",
                "TUT_RetroPrintFilter", "TUT_GlassRefractionFilter", "TUT_GlitchArtFilter",
                } else IMAGE_TEXT
            self.assertEqual(node_class.CATEGORY, expected, node_id)

    def test_save_animated_gif_from_image_batch(self):
        source = torch.zeros((2, 24, 32, 3), dtype=torch.float32)
        source[1, :, :, 0] = 1.0

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "batch.gif"
            original_next_path = TUT_SaveAnimatedGIF._next_path
            TUT_SaveAnimatedGIF._next_path = staticmethod(
                lambda filename_prefix, width, height: (output_path, output_path.name, "")
            )
            try:
                result = TUT_SaveAnimatedGIF().save_gif(source, fps=10.0, pingpong=True)
            finally:
                TUT_SaveAnimatedGIF._next_path = original_next_path

            self.assertTrue(output_path.exists())
            self.assertEqual(
                result["ui"]["gif_preview"],
                [{"filename": "batch.gif", "subfolder": "", "type": "output"}],
            )
            self.assertNotIn("images", result["ui"])
            self.assertNotIn("animated", result["ui"])
            self.assertEqual(result["result"], ("batch.gif",))
            with Image.open(output_path) as gif:
                self.assertEqual(gif.n_frames, 2)

    def test_gif_compression_presets(self):
        self.assertEqual(list(GIF_COMPRESSION_PRESETS), ["自定义", "高画质", "均衡", "小体积"])
        self.assertEqual(
            GIF_COMPRESSION_PRESETS["高画质"],
            {"resize_scale": 0.85, "max_colors": 256, "frame_step": 1, "dither": True, "optimize": True},
        )
        self.assertEqual(
            GIF_COMPRESSION_PRESETS["均衡"],
            {"resize_scale": 0.75, "max_colors": 128, "frame_step": 1, "dither": False, "optimize": True},
        )
        self.assertEqual(
            GIF_COMPRESSION_PRESETS["小体积"],
            {"resize_scale": 0.5, "max_colors": 64, "frame_step": 2, "dither": False, "optimize": True},
        )

        inputs = TUT_SaveAnimatedGIF.INPUT_TYPES()["optional"]
        self.assertEqual(inputs["compression_preset"][1]["default"], "自定义")
        self.assertEqual(inputs["resize_scale"][1]["default"], 1.0)
        self.assertEqual(
            GIF_COLOR_LEVELS,
            ["2 色", "4 色", "8 色", "16 色", "32 色", "64 色", "128 色", "256 色"],
        )
        self.assertEqual(inputs["max_colors"][0], GIF_COLOR_LEVELS)
        self.assertEqual(inputs["max_colors"][1]["default"], "256 色")
        self.assertEqual(inputs["frame_step"][1]["default"], 1)
        self.assertFalse(inputs["dither"][1]["default"])
        self.assertTrue(inputs["save_metadata"][1]["default"])

    def test_gif_metadata_switch(self):
        source = torch.zeros((2, 16, 16, 3), dtype=torch.float32)

        with TemporaryDirectory() as temp_dir:
            metadata_path = Path(temp_dir) / "with_metadata.gif"
            clean_path = Path(temp_dir) / "without_metadata.gif"
            paths = iter((metadata_path, clean_path))
            original_next_path = TUT_SaveAnimatedGIF._next_path
            TUT_SaveAnimatedGIF._next_path = staticmethod(
                lambda filename_prefix, width, height: (
                    (path := next(paths)), path.name, ""
                )
            )
            try:
                node = TUT_SaveAnimatedGIF()
                node.save_gif(
                    source,
                    save_metadata=True,
                    prompt={"1": {"class_type": "TestNode"}},
                    extra_pnginfo={"workflow": {"nodes": [{"id": 1}]}},
                )
                node.save_gif(
                    source,
                    save_metadata=False,
                    prompt={"secret": "must-not-be-saved"},
                    extra_pnginfo={"workflow": {"secret": True}},
                )
            finally:
                TUT_SaveAnimatedGIF._next_path = original_next_path

            with Image.open(metadata_path) as gif:
                metadata = json.loads(gif.info["comment"].decode("utf-8"))
                self.assertEqual(metadata["prompt"]["1"]["class_type"], "TestNode")
                self.assertEqual(metadata["workflow"]["nodes"][0]["id"], 1)
            with Image.open(clean_path) as gif:
                self.assertNotIn("comment", gif.info)

    def test_gif_manual_compression_controls_output(self):
        source = torch.zeros((5, 40, 60, 3), dtype=torch.float32)
        for index in range(5):
            source[index, :, :, 0] = index / 4.0
            source[index, :, :, 1] = (4 - index) / 4.0

        with TemporaryDirectory() as temp_dir:
            output_path = Path(temp_dir) / "compressed.gif"
            original_next_path = TUT_SaveAnimatedGIF._next_path
            TUT_SaveAnimatedGIF._next_path = staticmethod(
                lambda filename_prefix, width, height: (output_path, output_path.name, "")
            )
            try:
                # The preset name is intentionally different from the manual
                # values: manual widgets remain authoritative after preset fill.
                TUT_SaveAnimatedGIF().save_gif(
                    source,
                    fps=10.0,
                    optimize=True,
                    compression_preset="小体积",
                    resize_scale=0.8,
                    max_colors="16 色",
                    frame_step=2,
                    dither=False,
                )
            finally:
                TUT_SaveAnimatedGIF._next_path = original_next_path

            with Image.open(output_path) as gif:
                self.assertEqual(gif.size, (48, 32))
                self.assertEqual(gif.n_frames, 3)
                durations = []
                for frame_index in range(gif.n_frames):
                    gif.seek(frame_index)
                    durations.append(gif.info["duration"])
                    self.assertIsNotNone(gif.convert("RGB").getcolors(maxcolors=16))
                self.assertEqual(sum(durations), 500)

    def test_gif_small_settings_reduce_file_size(self):
        generator = torch.Generator().manual_seed(20260820)
        source = torch.rand((8, 64, 64, 3), generator=generator, dtype=torch.float32)

        with TemporaryDirectory() as temp_dir:
            original_path = Path(temp_dir) / "original.gif"
            small_path = Path(temp_dir) / "small.gif"
            paths = iter((original_path, small_path))
            original_next_path = TUT_SaveAnimatedGIF._next_path
            TUT_SaveAnimatedGIF._next_path = staticmethod(
                lambda filename_prefix, width, height: (
                    (path := next(paths)), path.name, ""
                )
            )
            try:
                node = TUT_SaveAnimatedGIF()
                node.save_gif(source, fps=12.0)
                small = GIF_COMPRESSION_PRESETS["小体积"]
                node.save_gif(
                    source,
                    fps=12.0,
                    compression_preset="小体积",
                    resize_scale=small["resize_scale"],
                    max_colors=small["max_colors"],
                    frame_step=small["frame_step"],
                    dither=small["dither"],
                    optimize=small["optimize"],
                )
            finally:
                TUT_SaveAnimatedGIF._next_path = original_next_path

            self.assertLess(small_path.stat().st_size, original_path.stat().st_size)

    def test_draw_text_with_effects(self):
        image, help_text, mask = TUT_DrawText().draw_text(
            256, 128, "TUT", self.font_name, 48, "white", "black",
            "center", "center", 0, 4, 1, 0, 0, 0.0, "text center",
            effect_preset="neon",
            glow_color="cyan",
        )
        self.assertEqual(tuple(image.shape), (1, 128, 256, 3))
        self.assertEqual(tuple(mask.shape), (1, 128, 256))
        self.assertGreater(float(mask.max()), 0.0)
        self.assertIn("TUT_DrawText", help_text)

    def test_overlay_preserves_batch(self):
        source = torch.zeros((2, 96, 192, 3), dtype=torch.float32)
        image, _, mask = TUT_OverlayText().overlay_text(
            source, "batch", self.font_name, 32, "white", "center", "center",
            0, 0, 0, 0, 0, 0.0, "text center", effect_preset="outline",
        )
        self.assertEqual(tuple(image.shape), (2, 96, 192, 3))
        self.assertEqual(tuple(mask.shape), (2, 96, 192))

    def test_independent_effect_broadcasts_mask(self):
        background = torch.zeros((2, 64, 64, 3), dtype=torch.float32)
        mask = torch.zeros((1, 64, 64), dtype=torch.float32)
        mask[:, 20:44, 20:44] = 1.0
        image, _, returned_mask = TUT_TextEffect().apply_effect(
            background, mask, "red", effect_preset="shadow",
        )
        self.assertEqual(tuple(image.shape), (2, 64, 64, 3))
        self.assertEqual(tuple(returned_mask.shape), (2, 64, 64))
        self.assertGreater(float(image.max()), 0.0)

    def test_mask_composite_watermark_and_font_nodes(self):
        source = torch.ones((1, 96, 192, 3), dtype=torch.float32)
        background = torch.zeros((1, 96, 192, 3), dtype=torch.float32)

        masked, _, masked_text = TUT_MaskText().mask_text(
            source, "MASK", self.font_name, 30, "black", "center", "center",
            0, 0, 0, 0, 0, 0.0, "text center", effect_preset="glow",
        )
        composited, _, composite_mask = TUT_CompositeText().composite_text(
            source, background, "TEXT", self.font_name, 30, "center", "center",
            0, 0, 0, 0, 0, 0.0, "text center", effect_preset="extrude",
        )
        watermarked, _, watermark_mask = TUT_SimpleTextWatermark().watermark(
            background, "TUT", "bottom right", 0.5, self.font_name, 24, "white", 10, 10,
            effect_preset="shadow",
        )

        for image in (masked, composited, watermarked):
            self.assertEqual(tuple(image.shape), (1, 96, 192, 3))
        for mask in (masked_text, composite_mask, watermark_mask):
            self.assertEqual(tuple(mask.shape), (1, 96, 192))
            self.assertGreater(float(mask.max()), 0.0)
        self.assertEqual(TUT_SelectFont().select_font(self.font_name)[0], self.font_name)

    def test_every_effect_preset_renders(self):
        background = torch.zeros((1, 64, 96, 3), dtype=torch.float32)
        mask = torch.zeros((1, 64, 96), dtype=torch.float32)
        mask[:, 20:44, 24:72] = 1.0
        for preset in ("none", "custom", "outline", "shadow", "glow", "neon", "gradient", "emboss", "extrude"):
            image, _, _ = TUT_TextEffect().apply_effect(
                background, mask, "white", effect_preset=preset,
                fill_mode="radial_gradient", outline_width=2, glow_radius=3,
                shadow_opacity=0.4, emboss_depth=1, extrude_depth=2,
            )
            self.assertEqual(tuple(image.shape), (1, 64, 96, 3), preset)

    def test_multilayer_text_effect_defaults_and_rendering(self):
        background = torch.zeros((1, 80, 120, 3), dtype=torch.float32)
        mask = torch.zeros((1, 80, 120), dtype=torch.float32)
        mask[:, 24:56, 36:84] = 1.0

        legacy, _, _ = TUT_TextEffect().apply_effect(
            background, mask, "white", effect_preset="custom", outline_width=3,
        )
        explicit_defaults, _, _ = TUT_TextEffect().apply_effect(
            background, mask, "white", effect_preset="custom", outline_width=3,
            outer_outline2_width=0, inner_outline_width=0, highlight_strength=0.0,
        )
        self.assertTrue(torch.equal(legacy, explicit_defaults))

        multilayer, _, returned_mask = TUT_TextEffect().apply_effect(
            background, mask, "white", effect_preset="custom",
            outline_width=2, outer_outline2_width=4, outline_gap=1,
            outer_outline2_color="red", inner_outline_width=3,
            inner_outline_color="blue", highlight_strength=0.7,
        )
        self.assertEqual(tuple(multilayer.shape), (1, 80, 120, 3))
        self.assertTrue(torch.equal(returned_mask, mask))
        self.assertFalse(torch.equal(multilayer, legacy))

    def test_bottom_alignment_respects_margin(self):
        mask = render_text_mask(
            (240, 100), "Ag", self.font_name, 50,
            align="bottom", justify="center", margins=10,
        )
        bbox = mask.getbbox()
        self.assertIsNotNone(bbox)
        self.assertLessEqual(bbox[3], 90)

    def test_zero_watermark_opacity_hides_all_effects(self):
        background = torch.zeros((1, 96, 192, 3), dtype=torch.float32)
        image, _, _ = TUT_SimpleTextWatermark().watermark(
            background, "hidden", "bottom right", 0.0, self.font_name, 30,
            "white", 10, 10, effect_preset="neon",
        )
        self.assertTrue(torch.equal(image, background))


if __name__ == "__main__":
    unittest.main()
