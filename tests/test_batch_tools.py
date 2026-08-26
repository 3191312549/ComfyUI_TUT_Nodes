import unittest

from TUT_Nodes.categories import TOOLS_BATCH
from TUT_Nodes.nodes.tools.batch import TUT_SelectBatchItem


class TUTSelectBatchItemTests(unittest.TestCase):
    def test_public_list_interface(self):
        inputs = TUT_SelectBatchItem.INPUT_TYPES()["required"]
        self.assertEqual(inputs["batch"][0], "*")
        self.assertEqual(inputs["index"][1]["default"], 0)
        self.assertTrue(TUT_SelectBatchItem.INPUT_IS_LIST)
        self.assertEqual(TUT_SelectBatchItem.RETURN_TYPES, ("*", "INT", "INT"))
        self.assertEqual(
            TUT_SelectBatchItem.RETURN_NAMES,
            ("selected_item", "selected_index", "item_count"),
        )
        self.assertEqual(TUT_SelectBatchItem.CATEGORY, TOOLS_BATCH)

    def test_zero_and_one_select_expected_items(self):
        node = TUT_SelectBatchItem()
        self.assertEqual(node.select_item(["第0项", "第1项", "第2项"], [0]), ("第0项", 0, 3))
        self.assertEqual(node.select_item(["第0项", "第1项", "第2项"], [1]), ("第1项", 1, 3))

    def test_accepts_scalar_index_for_direct_calls(self):
        self.assertEqual(TUT_SelectBatchItem().select_item([10, 20], 1), (20, 1, 2))

    def test_empty_and_out_of_range_errors_are_clear(self):
        node = TUT_SelectBatchItem()
        with self.assertRaisesRegex(ValueError, "输入批次为空"):
            node.select_item([], [0])
        with self.assertRaisesRegex(ValueError, "可用编号为 0 到 1"):
            node.select_item(["甲", "乙"], [2])
        with self.assertRaisesRegex(ValueError, "超出范围"):
            node.select_item(["甲"], [-1])


if __name__ == "__main__":
    unittest.main()
