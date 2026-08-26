"""Generic ComfyUI list/batch utility nodes."""

from __future__ import annotations

from ...categories import TOOLS_BATCH


def _first_control_value(value, name: str):
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError(f"{name} 没有可用值。")
        return value[0]
    return value


class TUT_SelectBatchItem:
    """Select one item from a ComfyUI execution list by its zero-based index."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "batch": (
                    "*",
                    {"tooltip": "连接带列表标记的批次输出。"},
                ),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 2_147_483_647,
                        "step": 1,
                        "tooltip": "从 0 开始编号：0 取第 0 项，1 取第 1 项。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("*", "INT", "INT")
    RETURN_NAMES = ("selected_item", "selected_index", "item_count")
    INPUT_IS_LIST = True
    FUNCTION = "select_item"
    CATEGORY = TOOLS_BATCH
    DESCRIPTION = "按从 0 开始的编号，从 ComfyUI 列表批次中取出一个元素。"

    def select_item(self, batch, index):
        if not isinstance(batch, (list, tuple)) or not batch:
            raise ValueError("输入批次为空，无法按编号加载元素。")

        selected_index = int(_first_control_value(index, "编号"))
        if selected_index < 0 or selected_index >= len(batch):
            raise ValueError(
                f"批次编号 {selected_index} 超出范围；当前共有 {len(batch)} 项，"
                f"可用编号为 0 到 {len(batch) - 1}。"
            )
        return (batch[selected_index], selected_index, len(batch))


NODE_CLASS_MAPPINGS = {"TUT_SelectBatchItem": TUT_SelectBatchItem}
NODE_DISPLAY_NAME_MAPPINGS = {"TUT_SelectBatchItem": "TUT_批次按编号加载"}
