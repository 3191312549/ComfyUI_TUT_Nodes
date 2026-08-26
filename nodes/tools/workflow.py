"""Workflow-control utility nodes."""

from __future__ import annotations

import time

from ...categories import TOOLS_WORKFLOW


class TUT_DelayPassThrough:
    """Wait for a configured duration, then return the input unchanged."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "value": (
                    "*",
                    {
                        "display_name": "输入",
                        "tooltip": "可连接任意类型；节点不会修改输入内容。",
                    },
                ),
                "delay_ms": (
                    "INT",
                    {
                        "default": 1000,
                        "min": 0,
                        "max": 3_600_000,
                        "step": 1,
                        "display_name": "等待时间（毫秒）",
                        "tooltip": "收到输入后等待指定毫秒数，再原样输出。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("*",)
    RETURN_NAMES = ("输出",)
    FUNCTION = "wait_and_pass"
    CATEGORY = TOOLS_WORKFLOW
    DESCRIPTION = "等待指定毫秒数后，将任意类型的输入原样输出。"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        # A delay is an execution-side effect, so it must not be skipped by cache.
        return float("nan")

    def wait_and_pass(self, value, delay_ms):
        delay_ms = int(delay_ms)
        if delay_ms < 0:
            raise ValueError("等待时间不能小于 0 毫秒。")
        time.sleep(delay_ms / 1000.0)
        return (value,)


NODE_CLASS_MAPPINGS = {"TUT_DelayPassThrough": TUT_DelayPassThrough}
NODE_DISPLAY_NAME_MAPPINGS = {"TUT_DelayPassThrough": "TUT_等待"}
