import math
import unittest
from unittest.mock import patch

from TUT_Nodes.categories import TOOLS_WORKFLOW
from TUT_Nodes.nodes.tools.workflow import TUT_DelayPassThrough


class TUTDelayPassThroughTests(unittest.TestCase):
    def test_public_interface(self):
        inputs = TUT_DelayPassThrough.INPUT_TYPES()["required"]
        self.assertEqual(inputs["value"][0], "*")
        self.assertEqual(inputs["value"][1]["display_name"], "输入")
        self.assertEqual(inputs["delay_ms"][0], "INT")
        self.assertEqual(inputs["delay_ms"][1]["default"], 1000)
        self.assertEqual(inputs["delay_ms"][1]["display_name"], "等待时间（毫秒）")
        self.assertEqual(TUT_DelayPassThrough.RETURN_TYPES, ("*",))
        self.assertEqual(TUT_DelayPassThrough.RETURN_NAMES, ("输出",))
        self.assertEqual(TUT_DelayPassThrough.CATEGORY, TOOLS_WORKFLOW)

    def test_waits_in_seconds_and_returns_same_object(self):
        value = {"任意": [1, 2, 3]}
        with patch("TUT_Nodes.nodes.tools.workflow.time.sleep") as sleep:
            result = TUT_DelayPassThrough().wait_and_pass(value, 1250)
        sleep.assert_called_once_with(1.25)
        self.assertIs(result[0], value)

    def test_zero_delay_is_supported(self):
        with patch("TUT_Nodes.nodes.tools.workflow.time.sleep") as sleep:
            self.assertEqual(TUT_DelayPassThrough().wait_and_pass("内容", 0), ("内容",))
        sleep.assert_called_once_with(0.0)

    def test_negative_direct_call_has_clear_error(self):
        with self.assertRaisesRegex(ValueError, "不能小于 0"):
            TUT_DelayPassThrough().wait_and_pass("内容", -1)

    def test_cache_is_disabled_for_each_execution(self):
        self.assertTrue(math.isnan(TUT_DelayPassThrough.IS_CHANGED()))


if __name__ == "__main__":
    unittest.main()
