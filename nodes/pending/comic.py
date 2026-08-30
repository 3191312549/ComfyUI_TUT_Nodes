"""Comic panel canvas node."""

from __future__ import annotations

from ...categories import IMAGE_COMIC
from ...core.comic import AUTO_LAYOUT, PANEL_LAYOUTS, render_comic_panels


class TUT_ComicPanelCanvas:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "images": ("IMAGE",),
            "layout": ([AUTO_LAYOUT, *PANEL_LAYOUTS.keys()], {"default": AUTO_LAYOUT}),
            "canvas_width": ("INT", {"default": 1024, "min": 64, "max": 8192}),
            "canvas_height": ("INT", {"default": 1536, "min": 64, "max": 8192}),
            "page_margin": ("INT", {"default": 36, "min": 0, "max": 2048}),
            "gutter": ("INT", {"default": 18, "min": 0, "max": 1024}),
            "border_width": ("INT", {"default": 6, "min": 0, "max": 128}),
            "border_color": ("STRING", {"default": "#111111"}),
            "background_color": ("STRING", {"default": "#ffffff"}),
            "fit_mode": (["裁切填充", "完整显示"], {"default": "裁切填充"}),
            "empty_fill": (["留空", "循环填充", "复制最后一张"], {"default": "留空"}),
            "panel_data": ("STRING", {"multiline": True, "default": '{"version":1,"panels":[]}'}),
        }, "hidden": {
            "prompt": "PROMPT",
            "extra_pnginfo": "EXTRA_PNGINFO",
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "show_help", "panel_mask", "border_mask", "panel_data")
    FUNCTION = "compose"
    CATEGORY = IMAGE_COMIC
    DESCRIPTION = "将 IMAGE 批次自动分页填入一至六格漫画分镜，并支持逐格焦点、缩放和翻转。"

    @staticmethod
    def _save_input_previews(images, prompt, extra_pnginfo):
        from nodes import PreviewImage

        saved = PreviewImage().save_images(
            images[:6], filename_prefix="TUT.comic.input.",
            prompt=prompt, extra_pnginfo=extra_pnginfo,
        )
        return saved.get("ui", {}).get("images", [])

    def compose(self, images, layout, canvas_width, canvas_height, page_margin, gutter,
                border_width, border_color, background_color, fit_mode, empty_fill, panel_data,
                prompt=None, extra_pnginfo=None):
        output, panel_mask, border_mask, canonical = render_comic_panels(
            images, layout, canvas_width, canvas_height, page_margin, gutter,
            border_width, border_color, background_color, fit_mode, empty_fill, panel_data,
        )
        result = (output, "TUT_ComicPanelCanvas：批次漫画分镜画布", panel_mask, border_mask, canonical)
        if prompt is None and extra_pnginfo is None:
            return result
        previews = self._save_input_previews(images, prompt, extra_pnginfo)
        return {"ui": {"input_previews": previews}, "result": result}


NODE_CLASS_MAPPINGS = {
    "TUT_ComicPanelCanvas": TUT_ComicPanelCanvas,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_ComicPanelCanvas": "TUT_漫画分镜画布",
}
