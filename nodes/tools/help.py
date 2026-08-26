"""Generic help node that can inspect any upstream ComfyUI node."""

from __future__ import annotations

from collections.abc import Mapping

from ...categories import TOOLS_TEXT


def _as_link(value):
    """Return a raw ComfyUI link as ``(node_id, output_index)``."""

    if isinstance(value, (list, tuple)) and len(value) == 1:
        value = value[0]
    if not isinstance(value, (list, tuple)) or len(value) < 2:
        return None
    return value[0], value[1]


def _lookup_prompt_node(node_id, prompt, dynprompt):
    if dynprompt is not None:
        try:
            return dynprompt.get_node(node_id)
        except Exception:
            try:
                return dynprompt.get_node(str(node_id))
            except Exception:
                pass

    if not isinstance(prompt, Mapping):
        return None
    return prompt.get(node_id) or prompt.get(str(node_id))


def _node_class_and_display_name(class_type: str):
    """Resolve a loaded node class without importing ComfyUI at plugin load time."""

    try:
        import nodes as comfy_nodes

        node_class = getattr(comfy_nodes, "NODE_CLASS_MAPPINGS", {}).get(class_type)
        display_name = getattr(comfy_nodes, "NODE_DISPLAY_NAME_MAPPINGS", {}).get(
            class_type, class_type
        )
        if node_class is not None:
            return node_class, display_name
    except Exception:
        pass

    # This fallback keeps the node useful in unit tests and in lightweight imports.
    try:
        from ...registry import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

        return (
            NODE_CLASS_MAPPINGS.get(class_type),
            NODE_DISPLAY_NAME_MAPPINGS.get(class_type, class_type),
        )
    except Exception:
        return None, class_type


def _type_name(spec) -> str:
    if isinstance(spec, (list, tuple)) and spec:
        value = spec[0]
    else:
        value = spec
    if isinstance(value, (list, tuple)):
        choices = [str(item) for item in value]
        preview = ", ".join(choices[:5])
        if len(choices) > 5:
            preview += f", …（共 {len(choices)} 项）"
        return f"选择（{preview}）"
    return getattr(value, "name", None) or str(value)


def _format_inputs(node_class) -> list[str]:
    try:
        input_types = node_class.INPUT_TYPES()
    except Exception as exc:
        return [f"  （输入定义读取失败：{type(exc).__name__}）"]

    lines = []
    for section, title in (("required", "必填输入"), ("optional", "可选输入")):
        values = input_types.get(section, {}) if isinstance(input_types, Mapping) else {}
        if not values:
            continue
        lines.append(f"{title}：")
        for name, spec in values.items():
            line = f"  - {name}: {_type_name(spec)}"
            if isinstance(spec, (list, tuple)) and len(spec) > 1 and isinstance(spec[1], Mapping):
                options = spec[1]
                if "default" in options:
                    line += f"，默认={options['default']!r}"
                tooltip = options.get("tooltip") or options.get("description")
                if tooltip:
                    line += f"；{tooltip}"
            lines.append(line)
    return lines or ["输入：无可见输入"]


def _format_outputs(node_class) -> list[str]:
    return_types = tuple(getattr(node_class, "RETURN_TYPES", ()) or ())
    return_names = tuple(getattr(node_class, "RETURN_NAMES", ()) or ())
    if not return_types:
        return ["输出：无（通常为保存或其他副作用节点）"]
    lines = ["输出："]
    for index, return_type in enumerate(return_types):
        name = return_names[index] if index < len(return_names) else f"output_{index}"
        lines.append(f"  - {index}: {name} ({_type_name(return_type)})")
    return lines


def format_node_help(class_type: str, node_class=None, display_name: str | None = None) -> str:
    """Build a readable, dependency-free document from a loaded node class."""

    display_name = display_name or class_type
    if node_class is None:
        return (
            f"节点帮助\n\n名称：{display_name}\n内部 ID：{class_type}\n\n"
            "当前运行环境没有找到该节点的 Python 定义，无法读取详细输入和输出。"
        )

    description = str(getattr(node_class, "DESCRIPTION", "") or "").strip()
    category = str(getattr(node_class, "CATEGORY", "") or "").strip()
    function = str(getattr(node_class, "FUNCTION", "") or "").strip()
    lines = [
        "节点帮助",
        "",
        f"名称：{display_name}",
        f"内部 ID：{class_type}",
    ]
    if category:
        lines.append(f"分类：{category}")
    if function:
        lines.append(f"执行函数：{function}")
    if description:
        lines.extend(("", f"说明：{description}"))
    lines.extend(("", *_format_inputs(node_class), "", *_format_outputs(node_class)))
    return "\n".join(lines)


class TUT_NodeHelp:
    """Inspect the node connected to the wildcard input and show its help document."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "node": (
                    "*",
                    {
                        "rawLink": True,
                        "tooltip": "连接任意节点的任意输出端口；帮助节点会读取上游节点定义。",
                    },
                ),
            },
            "hidden": {
                "prompt": "PROMPT",
                "dynprompt": "DYNPROMPT",
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("help_document",)
    FUNCTION = "show_help"
    OUTPUT_NODE = True
    CATEGORY = TOOLS_TEXT
    DESCRIPTION = "连接任意节点后，读取并显示该节点的名称、说明、输入和输出。"

    def show_help(self, node, prompt=None, dynprompt=None):
        link = _as_link(node)
        if link is None:
            document = "节点帮助\n\n请将任意节点的输出端口连接到 TUT_节点帮助。"
            return {"ui": {"text": (document,)}, "result": (document,)}

        upstream_id, output_index = link
        prompt_node = _lookup_prompt_node(upstream_id, prompt, dynprompt)
        if not isinstance(prompt_node, Mapping):
            document = (
                "节点帮助\n\n"
                f"找不到上游节点：{upstream_id!r}（输出端口 {output_index!r}）。\n"
                "请重新执行工作流，或确认连接仍然有效。"
            )
            return {"ui": {"text": (document,)}, "result": (document,)}

        class_type = str(prompt_node.get("class_type", ""))
        if not class_type:
            document = f"节点帮助\n\n上游节点 {upstream_id!r} 没有 class_type。"
            return {"ui": {"text": (document,)}, "result": (document,)}

        node_class, display_name = _node_class_and_display_name(class_type)
        document = format_node_help(class_type, node_class, display_name)
        return {"ui": {"text": (document,)}, "result": (document,)}


NODE_CLASS_MAPPINGS = {"TUT_NodeHelp": TUT_NodeHelp}
NODE_DISPLAY_NAME_MAPPINGS = {"TUT_NodeHelp": "TUT_节点帮助"}
