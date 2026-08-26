"""General text utility nodes."""

from __future__ import annotations

from ...categories import TOOLS_TEXT
from ...core.text_tools import decode_separator


class TUT_SplitTextBatch:
    """Split one string into a ComfyUI list for downstream batch execution."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": (
                    "STRING",
                    {
                        "default": "",
                        "multiline": True,
                        "tooltip": "需要拆分为批次的文本。",
                    },
                ),
                "separator": (
                    "STRING",
                    {
                        "default": r"\n",
                        "multiline": False,
                        "tooltip": r"拆分分隔符；支持 \n、\t、\r 和 \\。",
                    },
                ),
                "strip_whitespace": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "去除每个文本项首尾的空格与换行。",
                    },
                ),
                "remove_empty": (
                    "BOOLEAN",
                    {
                        "default": True,
                        "tooltip": "忽略拆分后产生的空文本项。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "INT")
    RETURN_NAMES = ("text_items", "item_count")
    OUTPUT_IS_LIST = (True, False)
    FUNCTION = "split_text"
    CATEGORY = TOOLS_TEXT
    DESCRIPTION = "按自定义分隔符把文本拆成 STRING 列表，逐项驱动下游节点。"

    def split_text(self, text, separator, strip_whitespace=True, remove_empty=True):
        decoded_separator = decode_separator(separator)
        if decoded_separator == "":
            raise ValueError("文本拆分分隔符不能为空。")

        items = str(text).split(decoded_separator)
        if strip_whitespace:
            items = [item.strip() for item in items]
        if remove_empty:
            items = [item for item in items if item != ""]
        return (items, len(items))


NODE_CLASS_MAPPINGS = {"TUT_SplitTextBatch": TUT_SplitTextBatch}
NODE_DISPLAY_NAME_MAPPINGS = {"TUT_SplitTextBatch": "TUT_文本分隔批次"}
