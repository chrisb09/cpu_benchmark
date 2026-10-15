import unittest

from check_maia_warmup import compare


class WarmupComparisonTests(unittest.TestCase):
    def test_second_call_changes_six_call_mean(self):
        result = compare([9000, 2000, 1000, 1000, 1000, 1000, 1000])
        self.assertEqual(result["call2_delta_pct"], 100)
        self.assertAlmostEqual(result["mean_change_pct"], 100 / 6)
        self.assertTrue(result["call2_above_later_max"])
        self.assertEqual(result["later_mean_s"], 1)

    def test_first_call_does_not_affect_comparison(self):
        self.assertEqual(compare([1, 2, 3, 4, 5, 6, 7]),
                         compare([10000, 2, 3, 4, 5, 6, 7]))

    def test_identical_calls(self):
        result = compare([1000] * 7)
        self.assertEqual(result["mean_change_pct"], 0)
        self.assertFalse(result["call2_above_later_max"])

    def test_invalid_calls(self):
        for values in ([1] * 6, [0] * 7, [float("nan")] * 7, [float("inf")] * 7):
            with self.subTest(values=values), self.assertRaises(ValueError):
                compare(values)


if __name__ == "__main__":
    unittest.main()
