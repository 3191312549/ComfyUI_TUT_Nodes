import unittest

from TUT_Nodes.categories import TOOLS_TEXT
from TUT_Nodes.nodes.tools.text import TUT_SplitTextBatch


class TUTSplitTextBatchTests(unittest.TestCase):
    def test_public_interface_and_defaults(self):
        inputs = TUT_SplitTextBatch.INPUT_TYPES()["required"]
        self.assertEqual(inputs["separator"][1]["default"], r"\n")
        self.assertTrue(inputs["strip_whitespace"][1]["default"])
        self.assertTrue(inputs["remove_empty"][1]["default"])
        self.assertEqual(TUT_SplitTextBatch.RETURN_TYPES, ("STRING", "INT"))
        self.assertEqual(TUT_SplitTextBatch.RETURN_NAMES, ("text_items", "item_count"))
        self.assertEqual(TUT_SplitTextBatch.OUTPUT_IS_LIST, (True, False))
        self.assertEqual(TUT_SplitTextBatch.CATEGORY, TOOLS_TEXT)

    def test_splits_newlines_into_clean_batch(self):
        result = TUT_SplitTextBatch().split_text(
            "  第一项  \n\n第二项\n  第三项",
            r"\n",
            strip_whitespace=True,
            remove_empty=True,
        )
        self.assertEqual(result, (["第一项", "第二项", "第三项"], 3))

    def test_supports_multicharacter_and_tab_separators(self):
        node = TUT_SplitTextBatch()
        self.assertEqual(node.split_text("甲||乙||丙", "||"), (["甲", "乙", "丙"], 3))
        self.assertEqual(node.split_text("甲\t乙\t丙", r"\t"), (["甲", "乙", "丙"], 3))

    def test_can_preserve_whitespace_and_empty_items(self):
        result = TUT_SplitTextBatch().split_text(
            " 甲 ,, 乙 ,",
            ",",
            strip_whitespace=False,
            remove_empty=False,
        )
        self.assertEqual(result, ([" 甲 ", "", " 乙 ", ""], 4))

    def test_empty_separator_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "分隔符不能为空"):
            TUT_SplitTextBatch().split_text("文本", "")


if __name__ == "__main__":
    unittest.main()
