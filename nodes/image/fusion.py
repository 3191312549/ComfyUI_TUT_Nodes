"""Fusion-style keying and two-image compositing nodes."""

from __future__ import annotations

from ...categories import IMAGE_COMPOSITE, IMAGE_KEYING
from ...core.fusion import (
    channel_boolean, corner_pin, depth_merge, difference_key, displace,
    light_wrap, matte_finesse,
)
from ...core.imaging import parse_color


class TUT_DifferenceKeying:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "clean_background": ("IMAGE",),
            "color_space": (["RGB", "Lab", "亮度"], {"default": "Lab"}),
            "threshold": ("FLOAT", {"default": .08, "min": 0., "max": 1., "step": .005}),
            "softness": ("FLOAT", {"default": .12, "min": 0., "max": 1., "step": .005}),
            "denoise": ("INT", {"default": 1, "min": 0, "max": 16}),
            "grow_shrink": ("INT", {"default": 0, "min": -64, "max": 64}),
            "edge_feather": ("FLOAT", {"default": 1., "min": 0., "max": 64., "step": .1}),
            "invert_mask": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "MASK", "IMAGE")
    RETURN_NAMES = ("foreground", "show_help", "foreground_mask", "background_mask", "difference_image")
    FUNCTION = "key"; CATEGORY = IMAGE_KEYING
    DESCRIPTION = "使用当前画面与干净背景的差异提取前景。"
    def key(self, image, clean_background, color_space, threshold, softness, denoise, grow_shrink, edge_feather, invert_mask):
        fg, mask, bg_mask, diff = difference_key(image, clean_background, color_space, threshold, softness, denoise, grow_shrink, edge_feather, invert_mask)
        return fg, f"TUT_DifferenceKeying：{color_space} 差异抠像", mask, bg_mask, diff


class TUT_MatteFinesse:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "mask": ("MASK",),
            "black_point": ("FLOAT", {"default": 0., "min": 0., "max": 1., "step": .01}),
            "white_point": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": .01}),
            "edge_contrast": ("FLOAT", {"default": 1., "min": .1, "max": 4., "step": .05}),
            "fill_holes": ("BOOLEAN", {"default": False}),
            "grow_shrink": ("INT", {"default": 0, "min": -64, "max": 64}),
            "edge_feather": ("FLOAT", {"default": 0., "min": 0., "max": 64., "step": .1}),
            "detail_recovery": ("FLOAT", {"default": 0., "min": 0., "max": 1., "step": .01}),
            "despill_color": (["green", "blue", "custom"], {"default": "green"}),
            "despill_color_hex": ("STRING", {"default": "#00FF00"}),
            "despill_strength": ("FLOAT", {"default": 0., "min": 0., "max": 1., "step": .01})}}
    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "MASK", "MASK")
    RETURN_NAMES = ("foreground", "show_help", "foreground_mask", "background_mask", "edge_mask")
    FUNCTION = "refine"; CATEGORY = IMAGE_KEYING
    DESCRIPTION = "清理、扩缩、羽化并去除遮罩边缘污染色。"
    def refine(self, image, mask, black_point, white_point, edge_contrast, fill_holes, grow_shrink, edge_feather, detail_recovery, despill_color, despill_color_hex, despill_strength):
        color = parse_color(despill_color, despill_color_hex)
        fg, alpha, background, edge = matte_finesse(image, mask, black_point, white_point, edge_contrast, fill_holes, grow_shrink, edge_feather, detail_recovery, color, despill_strength)
        return fg, "TUT_MatteFinesse：遮罩边缘精修", alpha, background, edge


class TUT_LightWrapComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"foreground": ("IMAGE",), "background": ("IMAGE",), "foreground_mask": ("MASK",),
            "wrap_width": ("INT", {"default": 12, "min": 1, "max": 256}),
            "highlight_threshold": ("FLOAT", {"default": .45, "min": 0., "max": .99, "step": .01}),
            "color_bleed": ("FLOAT", {"default": .7, "min": 0., "max": 1., "step": .01}),
            "strength": ("FLOAT", {"default": .5, "min": 0., "max": 1., "step": .01}),
            "blur": ("FLOAT", {"default": 5., "min": 0., "max": 128., "step": .5}),
            "inner_ratio": ("FLOAT", {"default": .8, "min": 0., "max": 1., "step": .01})}}
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "show_help", "wrapped_foreground", "wrap_mask")
    FUNCTION = "composite"; CATEGORY = IMAGE_COMPOSITE
    DESCRIPTION = "采样背景亮部和颜色包裹前景轮廓。"
    def composite(self, foreground, background, foreground_mask, wrap_width, highlight_threshold, color_bleed, strength, blur, inner_ratio):
        image, wrapped, mask = light_wrap(foreground, background, foreground_mask, wrap_width, highlight_threshold, color_bleed, strength, blur, inner_ratio)
        return image, "TUT_LightWrapComposite：背景光线包裹", wrapped, mask


class TUT_DepthMerge:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_a": ("IMAGE",), "image_b": ("IMAGE",), "depth_a": ("MASK",), "depth_b": ("MASK",),
            "depth_mode": (["白近", "黑近"], {"default": "白近"}),
            "depth_offset": ("FLOAT", {"default": 0., "min": -1., "max": 1., "step": .01}),
            "edge_softness": ("FLOAT", {"default": .02, "min": 0., "max": 1., "step": .005}),
            "antialias": ("FLOAT", {"default": .5, "min": 0., "max": 16., "step": .1})}}
    RETURN_TYPES = ("IMAGE", "STRING", "MASK", "IMAGE")
    RETURN_NAMES = ("image", "show_help", "selection_mask", "depth_difference")
    FUNCTION = "merge"; CATEGORY = IMAGE_COMPOSITE
    DESCRIPTION = "依据两张深度图决定图一和图二的逐像素前后关系。"
    def merge(self, image_a, image_b, depth_a, depth_b, depth_mode, depth_offset, edge_softness, antialias):
        image, selection, difference = depth_merge(image_a, image_b, depth_a, depth_b, depth_mode, depth_offset, edge_softness, antialias)
        return image, "TUT_DepthMerge：深度图合成", selection, difference


class TUT_CornerPinComposite:
    @classmethod
    def INPUT_TYPES(cls):
        points = {f"{corner}_{axis}": ("FLOAT", {"default": default, "min": -2., "max": 3., "step": .01}) for corner, defaults in (("top_left", (0., 0.)), ("top_right", (1., 0.)), ("bottom_right", (1., 1.)), ("bottom_left", (0., 1.))) for axis, default in zip(("x", "y"), defaults)}
        required = {"foreground": ("IMAGE",), "background": ("IMAGE",), **points,
            "opacity": ("FLOAT", {"default": 1., "min": 0., "max": 1., "step": .01}),
            "blend_mode": (["normal", "multiply", "screen", "overlay"], {"default": "normal"}),
            "edge_feather": ("FLOAT", {"default": 0., "min": 0., "max": 64., "step": .1})}
        return {"required": required, "optional": {"foreground_mask": ("MASK",)}}
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "show_help", "transformed_layer", "transformed_mask")
    FUNCTION = "pin"; CATEGORY = IMAGE_COMPOSITE
    DESCRIPTION = "将前景四角精确映射到背景归一化坐标。"
    def pin(self, foreground, background, top_left_x, top_left_y, top_right_x, top_right_y, bottom_right_x, bottom_right_y, bottom_left_x, bottom_left_y, opacity, blend_mode, edge_feather, foreground_mask=None):
        corners = ((top_left_x, top_left_y), (top_right_x, top_right_y), (bottom_right_x, bottom_right_y), (bottom_left_x, bottom_left_y))
        image, layer, mask = corner_pin(foreground, background, foreground_mask, corners, opacity, blend_mode, edge_feather)
        return image, "TUT_CornerPinComposite：四角定位合成", layer, mask


CHANNEL_EXPRESSIONS = ["A.对应通道", "B.对应通道", "A.R", "A.G", "A.B", "A.A", "A.亮度", "B.R", "B.G", "B.B", "B.A", "B.亮度", "A+B", "A-B", "B-A", "A*B", "最小", "最大", "差值", "0", "1"]


class TUT_ChannelBoolean:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image_a": ("IMAGE",), "image_b": ("IMAGE",),
            "red_expression": (CHANNEL_EXPRESSIONS, {"default": "A.对应通道"}),
            "green_expression": (CHANNEL_EXPRESSIONS, {"default": "A.对应通道"}),
            "blue_expression": (CHANNEL_EXPRESSIONS, {"default": "A.对应通道"}),
            "alpha_expression": (CHANNEL_EXPRESSIONS, {"default": "1"})}}
    RETURN_TYPES = ("IMAGE", "STRING", "MASK")
    RETURN_NAMES = ("image", "show_help", "alpha_mask")
    FUNCTION = "combine"; CATEGORY = IMAGE_COMPOSITE
    DESCRIPTION = "逐通道选择、运算和重组两张图片。"
    def combine(self, image_a, image_b, red_expression, green_expression, blue_expression, alpha_expression):
        image, alpha = channel_boolean(image_a, image_b, (red_expression, green_expression, blue_expression, alpha_expression))
        return image, "TUT_ChannelBoolean：通道布尔合成", alpha


class TUT_DisplaceComposite:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"image": ("IMAGE",), "displacement_image": ("IMAGE",),
            "channel_x": (["红", "绿", "蓝", "亮度"], {"default": "红"}),
            "channel_y": (["红", "绿", "蓝", "亮度"], {"default": "绿"}),
            "strength_x": ("FLOAT", {"default": 20., "min": -1024., "max": 1024., "step": .5}),
            "strength_y": ("FLOAT", {"default": 20., "min": -1024., "max": 1024., "step": .5}),
            "neutral": ("FLOAT", {"default": .5, "min": 0., "max": 1., "step": .01}),
            "interpolation": (["双线性", "双三次"], {"default": "双线性"}),
            "boundary": (["裁切", "钳制", "镜像", "循环"], {"default": "钳制"})},
            "optional": {"mask": ("MASK",)}}
    RETURN_TYPES = ("IMAGE", "STRING", "IMAGE", "MASK")
    RETURN_NAMES = ("image", "show_help", "displacement_map", "affected_mask")
    FUNCTION = "apply"; CATEGORY = IMAGE_COMPOSITE
    DESCRIPTION = "使用位移图通道对源图执行确定性的二维重映射。"
    def apply(self, image, displacement_image, channel_x, channel_y, strength_x, strength_y, neutral, interpolation, boundary, mask=None):
        output, displacement_map, affected = displace(image, displacement_image, mask, channel_x, channel_y, strength_x, strength_y, neutral, interpolation, boundary)
        return output, "TUT_DisplaceComposite：图像位移合成", displacement_map, affected


NODE_CLASS_MAPPINGS = {
    "TUT_DifferenceKeying": TUT_DifferenceKeying, "TUT_MatteFinesse": TUT_MatteFinesse,
    "TUT_LightWrapComposite": TUT_LightWrapComposite, "TUT_DepthMerge": TUT_DepthMerge,
    "TUT_CornerPinComposite": TUT_CornerPinComposite, "TUT_ChannelBoolean": TUT_ChannelBoolean,
    "TUT_DisplaceComposite": TUT_DisplaceComposite,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_DifferenceKeying": "TUT_差异抠像", "TUT_MatteFinesse": "TUT_遮罩边缘精修",
    "TUT_LightWrapComposite": "TUT_光线包裹合成", "TUT_DepthMerge": "TUT_深度图合成",
    "TUT_CornerPinComposite": "TUT_四角定位合成", "TUT_ChannelBoolean": "TUT_通道布尔合成",
    "TUT_DisplaceComposite": "TUT_图像位移合成",
}
