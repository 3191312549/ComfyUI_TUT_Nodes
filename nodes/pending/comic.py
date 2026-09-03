"""Pending comic panel and speech-bubble nodes."""

from __future__ import annotations

from ...categories import IMAGE_COMIC, PENDING_IMAGE_COMIC
from ...core.comic import AUTO_LAYOUT, PANEL_LAYOUTS, render_comic_panels, render_speech_bubbles
from ...core.fonts import font_options
from ...web_routes import register_font_routes


register_font_routes()


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
    DESCRIPTION = "将 IMAGE 批次自动分页填入一至六格漫画分镜，支持抗锯齿自由四边形、拖边拉伸、逐边开放、图层与逐格镜头。"

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


class TUT_ComicSpeechBubble:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "image": ("IMAGE",),
            "enabled": ("BOOLEAN", {"default": True}),
            "default_font": (list(font_options()),),
            "bubble_data": ("STRING", {"multiline": True, "default": '{"version":1,"bubbles":[]}'}),
        }, "hidden": {
            "prompt": "PROMPT",
            "extra_pnginfo": "EXTRA_PNGINFO",
        }}

    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "MASK", "STRING")
    RETURN_NAMES = ("image", "show_help", "bubble_mask", "text_mask", "bubble_data")
    FUNCTION = "render"
    CATEGORY = PENDING_IMAGE_COMIC
    DESCRIPTION = "在漫画或任意图片上添加可保存、可拖动的对白框、思考框、喊话框、旁白框和文字。"

    @staticmethod
    def _save_input_preview(image, prompt, extra_pnginfo):
        from nodes import PreviewImage

        saved = PreviewImage().save_images(
            image[:1], filename_prefix="TUT.comic.bubble.input.",
            prompt=prompt, extra_pnginfo=extra_pnginfo,
        )
        return saved.get("ui", {}).get("images", [])

    def render(self, image, enabled, default_font, bubble_data, prompt=None, extra_pnginfo=None):
        output, bubble_mask, text_mask, canonical = render_speech_bubbles(
            image, enabled, default_font, bubble_data,
        )
        result = (output, "TUT_ComicSpeechBubble：可视化漫画对话框", bubble_mask, text_mask, canonical)
        if prompt is None and extra_pnginfo is None:
            return result
        previews = self._save_input_preview(image, prompt, extra_pnginfo)
        return {"ui": {"input_previews": previews}, "result": result}


NODE_CLASS_MAPPINGS = {
    "TUT_ComicPanelCanvas": TUT_ComicPanelCanvas,
    "TUT_ComicSpeechBubble": TUT_ComicSpeechBubble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_ComicPanelCanvas": "TUT_漫画分镜画布",
    "TUT_ComicSpeechBubble": "TUT_[待测试]漫画对话框",
}
