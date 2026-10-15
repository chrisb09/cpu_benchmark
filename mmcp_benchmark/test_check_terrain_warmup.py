import unittest

from check_terrain_warmup import compare


class TerrainWarmupTests(unittest.TestCase):
    def test_second_call_excess(self):
        result = compare([2000] + [1000] * 9)
        self.assertEqual(result["call2_delta_pct"], 100)
        self.assertEqual(result["current_mean_ms"], 1100)
        self.assertEqual(result["later_mean_ms"], 1000)
        self.assertTrue(result["call2_above_later_max"])

    def test_constant_calls(self):
        result = compare([1000] * 10)
        self.assertEqual(result["call2_delta_pct"], 0)
        self.assertFalse(result["call2_above_later_max"])

    def test_invalid_calls(self):
        for values in ([1] * 9, [0] * 10, [float("nan")] * 10):
            with self.subTest(values=values), self.assertRaises(ValueError):
                compare(values)


if __name__ == "__main__":
    unittest.main()
