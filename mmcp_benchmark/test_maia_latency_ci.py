"""Run-mean m-aia CI contracts, without Slurm or real runs."""
import importlib.util
import math
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("maia_ci", Path(__file__).with_name("maia_latency_ci.py"))
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


def write_run(root, means):
    run = Path(root)
    run.mkdir(parents=True, exist_ok=True)
    lines = []
    for i, value in enumerate(means, start=1):
        step = 15 + 29 * (i - 1)
        lines.append(f"ML_STEP_TIMING step={step} ml_ms={value} rank_min_ms={value} rank_max_ms={value}")
    (run / "solver.log").write_text("\n".join(lines) + "\n")
    return run


class LatencyCiTests(unittest.TestCase):
    def test_ci_constant_runs_is_zero(self):
        mean, half = ci.cell_ci([3, 3, 3, 3, 3])
        self.assertEqual(mean, 3)
        self.assertEqual(half, 0)

    def test_ci_formula(self):
        mean, half = ci.cell_ci([1, 2, 3, 4, 5])
        expected = ci.T95 * math.sqrt(2.5 / 5)
        self.assertAlmostEqual(half, expected, places=12)

    def test_ci_needs_five_runs(self):
        for values in ([1], [1, 2, 3], [1, 2, 3, 4]):
            with self.subTest(values=values), self.assertRaises(ValueError):
                ci.cell_ci(values)

    def test_run_mean_uses_five_steady_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            # Calls 1-2 are warmups; the run mean uses calls 3-7 only.
            run = write_run(tmp, [9000, 8000, 10, 20, 30, 40, 50])
            self.assertEqual(ci.run_mean(run), 30)

    def test_bad_schedule_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, [1] * 6)
            with self.assertRaisesRegex(ValueError, "schedule"):
                ci.calls(run)

    def test_nonfinite_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            run = write_run(tmp, [1, 1, 1, 1, 1, 1, 1])
            (run / "solver.log").write_text("ML_STEP_TIMING step=15 ml_ms=nan rank_min_ms=1 rank_max_ms=nan\n")
            with self.assertRaises(ValueError):
                ci.calls(run)


if __name__ == "__main__":
    unittest.main()
