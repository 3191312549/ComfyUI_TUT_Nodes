"""Excel loading and text-processing utility nodes."""

from __future__ import annotations

from ...categories import TOOLS_EXCEL
from ...core.excel import (
    FORMULA_MODES,
    ORIENTATIONS,
    READ_DIRECTIONS,
    ExcelTable,
    decode_separator,
    excel_file_fingerprint,
    read_excel_table,
    select_excel_line,
    slice_excel_table,
    table_text_items,
)


def _loader_inputs():
    return {
        "excel_path": (
            "STRING",
            {"default": "", "multiline": False, "tooltip": "通过节点按钮选择 .xlsx 或 .xlsm 文件。"},
        ),
        "sheet_name": (
            "STRING",
            {"default": "", "multiline": False, "tooltip": "工作表名称；留空读取第一个工作表。"},
        ),
        "cell_range": (
            "STRING",
            {"default": "", "multiline": False, "tooltip": "例如 A1:F100；留空读取工作表有效区域。"},
        ),
        "formula_mode": (FORMULA_MODES, {"default": "缓存值"}),
    }


def _processor_inputs():
    return {
        "excel_data": (
            "TUT_EXCEL_DATA",
            {"tooltip": "连接“TUT_读取Excel”节点的 Excel 数据输出。"},
        ),
        "orientation": (ORIENTATIONS, {"default": "按行"}),
        "skip_rows": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "step": 1}),
        "skip_columns": ("INT", {"default": 0, "min": 0, "max": 16_384, "step": 1}),
        "cell_separator": (
            "STRING",
            {
                "default": r"\n",
                "multiline": False,
                "tooltip": r"同一行或列内的单元格分隔符；支持 \n、\t、\r。",
            },
        ),
        "max_items": (
            "INT",
            {
                "default": 0,
                "min": 0,
                "max": 1_000_000,
                "step": 1,
                "tooltip": "限制最终读取的行数或列数；0 表示无上限。",
            },
        ),
    }


def _require_excel_table(excel_data) -> ExcelTable:
    if not isinstance(excel_data, ExcelTable):
        raise ValueError("Excel 数据无效，请连接“TUT_读取Excel”节点。")
    return excel_data


def _read_items(
    excel_data,
    orientation,
    skip_rows,
    skip_columns,
    cell_separator,
    max_items,
):
    table = slice_excel_table(
        _require_excel_table(excel_data),
        skip_rows=skip_rows,
        skip_columns=skip_columns,
        orientation=orientation,
        max_items=max_items,
    )
    items = table_text_items(table, orientation, cell_separator)
    metadata = (
        "\n".join(table.sheet_names),
        len(items),
        table.row_count,
        table.column_count,
    )
    return items, metadata


class TUT_LoadExcel:
    """Load one worksheet into reusable immutable Excel data."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _loader_inputs()}

    RETURN_TYPES = ("TUT_EXCEL_DATA", "STRING", "STRING", "INT", "INT")
    RETURN_NAMES = ("excel_data", "sheet_names", "selected_sheet", "row_count", "column_count")
    FUNCTION = "load_excel"
    CATEGORY = TOOLS_EXCEL
    DESCRIPTION = "选择并读取 Excel 工作表，输出给批次、合并和单行/单列处理节点。"

    @classmethod
    def IS_CHANGED(cls, excel_path, **kwargs):
        del kwargs
        return excel_file_fingerprint(excel_path)

    def load_excel(self, excel_path, sheet_name="", cell_range="", formula_mode="缓存值"):
        table = read_excel_table(
            excel_path=excel_path,
            sheet_name=sheet_name,
            cell_range=cell_range,
            formula_mode=formula_mode,
        )
        return (
            table,
            "\n".join(table.sheet_names),
            table.selected_sheet,
            table.row_count,
            table.column_count,
        )


class _ExcelProcessorBase:
    CATEGORY = TOOLS_EXCEL
    RETURN_TYPES = ("STRING", "STRING", "INT", "INT", "INT")
    RETURN_NAMES = ("text", "sheet_names", "item_count", "row_count", "column_count")


class TUT_ReadExcelBatch(_ExcelProcessorBase):
    """Emit one list item for every selected worksheet row or column."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": _processor_inputs()}

    RETURN_NAMES = ("text_items", "sheet_names", "item_count", "row_count", "column_count")
    OUTPUT_IS_LIST = (True, False, False, False, False)
    FUNCTION = "read_batch"
    DESCRIPTION = "将 Excel 数据按行或按列输出 STRING 列表，以批量驱动下游节点。"

    def read_batch(self, **kwargs):
        items, metadata = _read_items(**kwargs)
        return (items, *metadata)


class TUT_ReadExcelMerged(_ExcelProcessorBase):
    """Merge selected worksheet rows or columns into one string."""

    @classmethod
    def INPUT_TYPES(cls):
        inputs = _processor_inputs()
        inputs["group_separator"] = (
            "STRING",
            {
                "default": r"\n\n",
                "multiline": False,
                "tooltip": r"各行或各列之间的分隔符；支持 \n、\t、\r。",
            },
        )
        return {"required": inputs}

    FUNCTION = "read_merged"
    DESCRIPTION = "将 Excel 数据中的行或列使用自定义分隔符合并为一个 STRING。"

    def read_merged(self, group_separator, **kwargs):
        items, metadata = _read_items(**kwargs)
        return (decode_separator(group_separator).join(items), *metadata)


class TUT_ReadExcelSingleLine:
    """Read cells from one zero-based row or column as batch and merged text."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "excel_data": (
                    "TUT_EXCEL_DATA",
                    {"tooltip": "连接“TUT_读取Excel”节点的 Excel 数据输出。"},
                ),
                "read_direction": (READ_DIRECTIONS, {"default": "单行"}),
                "index": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1_000_000,
                        "step": 1,
                        "tooltip": "从 0 开始；0 表示第一行或第一列。",
                    },
                ),
                "skip_items": ("INT", {"default": 0, "min": 0, "max": 1_000_000, "step": 1}),
                "max_items": (
                    "INT",
                    {
                        "default": 0,
                        "min": 0,
                        "max": 1_000_000,
                        "step": 1,
                        "tooltip": "0 表示无上限。",
                    },
                ),
                "merge_separator": (
                    "STRING",
                    {
                        "default": r"\n",
                        "multiline": False,
                        "tooltip": r"合并输出中的单元格分隔符；支持 \n、\t、\r。",
                    },
                ),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT")
    RETURN_NAMES = ("text_items", "text", "item_count")
    OUTPUT_IS_LIST = (True, False, False)
    FUNCTION = "read_single_line"
    CATEGORY = TOOLS_EXCEL
    DESCRIPTION = "按从 0 开始的编号读取一行或一列，同时输出单元格批次与合并文本。"

    def read_single_line(
        self,
        excel_data,
        read_direction="单行",
        index=0,
        skip_items=0,
        max_items=0,
        merge_separator=r"\n",
    ):
        items = select_excel_line(
            _require_excel_table(excel_data),
            read_direction=read_direction,
            index=index,
            skip_items=skip_items,
            max_items=max_items,
        )
        return (list(items), decode_separator(merge_separator).join(items), len(items))


NODE_CLASS_MAPPINGS = {
    "TUT_LoadExcel": TUT_LoadExcel,
    "TUT_ReadExcelBatch": TUT_ReadExcelBatch,
    "TUT_ReadExcelMerged": TUT_ReadExcelMerged,
    "TUT_ReadExcelSingleLine": TUT_ReadExcelSingleLine,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "TUT_LoadExcel": "TUT_读取Excel",
    "TUT_ReadExcelBatch": "TUT_Excel批次读取",
    "TUT_ReadExcelMerged": "TUT_Excel合并读取",
    "TUT_ReadExcelSingleLine": "TUT_Excel只读单行/单列",
}


# The route is optional in lightweight imports/tests and available in ComfyUI.
from ...web_routes import register_excel_routes

register_excel_routes()
