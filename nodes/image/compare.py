"""Interactive image comparison preview node."""

from ...categories import IMAGE


class TUT_ImageCompare:
    """Show the first image from two inputs in an interactive A/B preview."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_a": ("IMAGE",),
                "image_b": ("IMAGE",),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    RETURN_TYPES = ()
    FUNCTION = "compare_images"
    OUTPUT_NODE = True
    CATEGORY = IMAGE
    DESCRIPTION = "叠加预览图像 A 与图像 B，并通过带外部 A/B 标识的滑动分割线进行对比。"

    @staticmethod
    def _save_preview(image, filename_prefix, prompt, extra_pnginfo):
        # ComfyUI is deliberately imported only when the node executes.  This
        # keeps TUT_Nodes importable in documentation and unit-test contexts.
        from nodes import PreviewImage

        result = PreviewImage().save_images(
            image,
            filename_prefix=filename_prefix,
            prompt=prompt,
            extra_pnginfo=extra_pnginfo,
        )
        return result.get("ui", {}).get("images", [])

    def compare_images(self, image_a, image_b, prompt=None, extra_pnginfo=None):
        if image_a.ndim != 4 or image_a.shape[0] < 1:
            raise ValueError("图像 A 必须是至少包含一张图片的 IMAGE 批次")
        if image_b.ndim != 4 or image_b.shape[0] < 1:
            raise ValueError("图像 B 必须是至少包含一张图片的 IMAGE 批次")

        a_images = self._save_preview(
            image_a[:1], "TUT.compare.A.", prompt, extra_pnginfo
        )
        b_images = self._save_preview(
            image_b[:1], "TUT.compare.B.", prompt, extra_pnginfo
        )
        return {"ui": {"a_images": a_images, "b_images": b_images}}


NODE_CLASS_MAPPINGS = {"TUT_ImageCompare": TUT_ImageCompare}
NODE_DISPLAY_NAME_MAPPINGS = {"TUT_ImageCompare": "TUT_图像对比"}
