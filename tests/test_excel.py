import os
import unittest
from datetime import date, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

from openpyxl import Workbook

from ComfyUI_TUT_Nodes.categories import TOOLS_EXCEL
from ComfyUI_TUT_Nodes.core.excel import (
    decode_separator,
    excel_file_fingerprint,
    inspect_excel_workbook,
    list_excel_input_files,
    read_excel_table,
    select_excel_line,
    slice_excel_table,
)
from ComfyUI_TUT_Nodes.nodes.tools.excel import (
    TUT_LoadExcel,
    TUT_ReadExcelBatch,
    TUT_ReadExcelMerged,
    TUT_ReadExcelSingleLine,
)


def _save_workbook(path: Path):
    workbook = Workbook()
    sheet = workbook.active
    sheet.title = "数据"
    sheet.append([None, None, None, None, None])
    sheet.append([None, "姓名", "数量", "启用", None])
    sheet.append([None, "甲", 2, True, None])
    sheet.append([None, "乙", None, False, None])
    sheet.append([None, date(2026, 8, 25), datetime(2026, 8, 25, 12, 30), None, None])
    sheet.append([None, None, None, None, None])
    formulas = workbook.create_sheet("公式")
    formulas["A1"] = 2
    formulas["A2"] = 3
    formulas["A3"] = "=SUM(A1:A2)"
    workbook.save(path)
    workbook.close()


class TUTExcelNodeTests(unittest.TestCase):
    def setUp(self):
        self.temp_dir = TemporaryDirectory()
        self.path = Path(self.temp_dir.name) / "sample.xlsx"
        _save_workbook(self.path)
        self.table = TUT_LoadExcel().load_excel(str(self.path), "数据", "", "缓存值")[0]

    def tearDown(self):
        self.temp_dir.cleanup()

    def _processor_inputs(self, **overrides):
        values = {
            "excel_data": self.table,
            "orientation": "按行",
            "skip_rows": 0,
            "skip_columns": 0,
            "cell_separator": r"\t",
            "max_items": 0,
        }
        values.update(overrides)
        return values

    def test_public_contracts_and_new_data_flow(self):
        for node in (TUT_LoadExcel, TUT_ReadExcelBatch, TUT_ReadExcelMerged, TUT_ReadExcelSingleLine):
            self.assertEqual(node.CATEGORY, TOOLS_EXCEL)
        self.assertEqual(
            TUT_LoadExcel.RETURN_TYPES,
            ("TUT_EXCEL_DATA", "STRING", "STRING", "INT", "INT"),
        )
        self.assertEqual(TUT_ReadExcelBatch.OUTPUT_IS_LIST, (True, False, False, False, False))
        self.assertFalse(hasattr(TUT_ReadExcelMerged, "OUTPUT_IS_LIST"))
        self.assertEqual(TUT_ReadExcelSingleLine.OUTPUT_IS_LIST, (True, False, False))
        batch_inputs = TUT_ReadExcelBatch.INPUT_TYPES()["required"]
        self.assertEqual(batch_inputs["excel_data"][0], "TUT_EXCEL_DATA")
        self.assertNotIn("excel_path", batch_inputs)
        self.assertEqual(batch_inputs["orientation"][0], ("按行", "按列"))
        self.assertEqual(batch_inputs["max_items"][1]["default"], 0)

    def test_loader_outputs_cropped_reusable_data_and_metadata(self):
        result = TUT_LoadExcel().load_excel(str(self.path), "数据", "", "缓存值")
        table = result[0]
        self.assertEqual(table.rows[0], ("姓名", "数量", "启用"))
        self.assertEqual(table.rows[2], ("乙", "", "false"))
        self.assertEqual(result[1:], ("数据\n公式", "数据", 4, 3))

    def test_batch_rows_preserve_internal_empty_cells_and_limit(self):
        result = TUT_ReadExcelBatch().read_batch(**self._processor_inputs(max_items=2))
        self.assertEqual(result[0], ["姓名\t数量\t启用", "甲\t2\ttrue"])
        self.assertEqual(result[1:], ("数据\n公式", 2, 2, 3))

    def test_batch_columns_apply_skips_before_limit(self):
        result = TUT_ReadExcelBatch().read_batch(
            **self._processor_inputs(
                orientation="按列",
                skip_rows=1,
                skip_columns=1,
                cell_separator=" | ",
                max_items=1,
            )
        )
        self.assertEqual(result[0], ["2 |  | 2026-08-25 12:30:00"])
        self.assertEqual(result[2:], (1, 3, 1))

    def test_zero_limit_is_unlimited_for_batch_and_merged(self):
        batch = TUT_ReadExcelBatch().read_batch(**self._processor_inputs(max_items=0))
        self.assertEqual(batch[2], 4)
        merged = TUT_ReadExcelMerged().read_merged(
            group_separator=r"\n---\n",
            **self._processor_inputs(skip_rows=1, max_items=2),
        )
        self.assertEqual(merged[0], "甲\t2\ttrue\n---\n乙\t\tfalse")
        self.assertEqual(merged[2:], (2, 2, 3))

    def test_single_row_outputs_batch_and_merged_text(self):
        result = TUT_ReadExcelSingleLine().read_single_line(
            self.table,
            read_direction="单行",
            index=2,
            skip_items=1,
            max_items=0,
            merge_separator=" | ",
        )
        self.assertEqual(result, (["", "false"], " | false", 2))

    def test_single_column_skip_and_limit_count_empty_cells(self):
        result = TUT_ReadExcelSingleLine().read_single_line(
            self.table,
            read_direction="单列",
            index=1,
            skip_items=1,
            max_items=2,
            merge_separator=r"\t",
        )
        self.assertEqual(result, (["2", ""], "2\t", 2))

    def test_single_line_clear_errors(self):
        with self.assertRaisesRegex(ValueError, "可用编号为 0 到 3"):
            select_excel_line(self.table, "单行", 4)
        with self.assertRaisesRegex(ValueError, "没有剩余数据"):
            select_excel_line(self.table, "单列", 0, skip_items=99)
        with self.assertRaisesRegex(ValueError, "读取数量上限不能小于 0"):
            select_excel_line(self.table, "单行", 0, max_items=-1)
        with self.assertRaisesRegex(ValueError, "TUT_读取Excel"):
            TUT_ReadExcelBatch().read_batch(
                **self._processor_inputs(excel_data=None)
            )

    def test_table_limit_tracks_result_dimensions(self):
        rows = slice_excel_table(self.table, orientation="按行", max_items=1)
        columns = slice_excel_table(self.table, orientation="按列", max_items=2)
        self.assertEqual((rows.row_count, rows.column_count), (1, 3))
        self.assertEqual((columns.row_count, columns.column_count), (4, 2))

    def test_formula_text_and_cached_value_modes(self):
        formula = read_excel_table(str(self.path), "公式", "A3", formula_mode="公式文本")
        cached = read_excel_table(str(self.path), "公式", "A3", formula_mode="缓存值")
        self.assertEqual(formula.rows, (("=SUM(A1:A2)",),))
        self.assertEqual(cached.rows, (("",),))

    def test_inspection_returns_sheet_dropdown_metadata_and_effective_ranges(self):
        metadata = inspect_excel_workbook(str(self.path))
        self.assertEqual(metadata["file_name"], "sample.xlsx")
        self.assertEqual(
            metadata["sheets"],
            [
                {"name": "数据", "range": "B2:D5", "row_count": 4, "column_count": 3},
                {"name": "公式", "range": "A1:A3", "row_count": 3, "column_count": 1},
            ],
        )

    def test_comfy_input_token_resolves_without_breaking_absolute_paths(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        fake_folder_paths = SimpleNamespace(get_annotated_filepath=lambda _token: str(self.path))
        with patch.dict(sys.modules, {"folder_paths": fake_folder_paths}):
            table = read_excel_table("TUT_Nodes/excel/sample.xlsx", "数据", "B2:C3")
        self.assertEqual(table.rows, (("姓名", "数量"), ("甲", "2")))

    def test_uploaded_workbook_dropdown_lists_only_supported_excel_files(self):
        import sys
        from types import SimpleNamespace
        from unittest.mock import patch

        input_root = Path(self.temp_dir.name) / "input"
        excel_root = input_root / "TUT_Nodes" / "excel"
        nested = excel_root / "子目录"
        nested.mkdir(parents=True)
        (excel_root / "甲.xlsx").write_bytes(b"xlsx")
        (nested / "乙.xlsm").write_bytes(b"xlsm")
        (excel_root / "忽略.xls").write_bytes(b"xls")
        fake_folder_paths = SimpleNamespace(get_input_directory=lambda: str(input_root))
        with patch.dict(sys.modules, {"folder_paths": fake_folder_paths}):
            files = list_excel_input_files()
        self.assertEqual(
            files,
            [
                {"name": "乙.xlsm", "token": "TUT_Nodes/excel/子目录/乙.xlsm"},
                {"name": "甲.xlsx", "token": "TUT_Nodes/excel/甲.xlsx"},
            ],
        )

    def test_file_fingerprint_changes_with_file_metadata(self):
        before = excel_file_fingerprint(str(self.path))
        stat = self.path.stat()
        os.utime(self.path, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000_000))
        after = excel_file_fingerprint(str(self.path))
        self.assertNotEqual(before, after)
        self.assertEqual(TUT_LoadExcel.IS_CHANGED(str(self.path)), after)

    def test_escape_sequences_and_clear_file_errors(self):
        self.assertEqual(decode_separator(r"前缀\n中文\t尾部\\"), "前缀\n中文\t尾部\\")
        with self.assertRaisesRegex(ValueError, "绝对路径"):
            read_excel_table("relative.xlsx")
        with self.assertRaisesRegex(ValueError, "仅支持"):
            read_excel_table(str(self.path.with_suffix(".xls")))
        with self.assertRaisesRegex(ValueError, "找不到工作表"):
            read_excel_table(str(self.path), "不存在")
        with self.assertRaisesRegex(ValueError, "范围格式无效"):
            read_excel_table(str(self.path), "数据", "A:F")
        with self.assertRaisesRegex(ValueError, "没有剩余数据"):
            read_excel_table(str(self.path), "数据", skip_rows=99)

        broken = Path(self.temp_dir.name) / "broken.xlsx"
        broken.write_text("not an excel file", encoding="utf-8")
        with self.assertRaisesRegex(ValueError, "损坏或受密码保护"):
            read_excel_table(str(broken))


if __name__ == "__main__":
    unittest.main()
