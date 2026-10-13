"""Selector and raw normalization contracts; no benchmark launches."""
import copy
import csv
import math
import sys
import unittest
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import regenerate_cpu_final as analysis


class FinalCPU(unittest.TestCase):
    def fixture(self):
        meta = dict(MEASUREMENT="timing", WARMUP_CALLS="2", SOLVER_RANKS="96", RUN_STEPS="200",
                    EXPECTED_INFERENCE_CALLS="7", ML_CUBE_OVERLAP="0", PROVIDER="PHYDLL",
                    SOLVER_EXIT_STATUS="0", EXIT_STATUS="0", PHYDLL_MODE="readiness", DL_RANKS="96",
                    MODEL="/input/mmcp_test_mlp_m5_cpu.pt", MAIA="/maia/build_cpu_retest_plain/bin/maia")
        rows = [dict(step=15+29*i, ml_ms=10.0, rank_min_ms=5.0, rank_max_ms=10.0) for i in range(7)]
        return meta, rows

    def test_valid(self):
        meta, rows = self.fixture()
        analysis.validate_cell(meta, "mlp_ov0", "timing", "readiness", rows)

    def test_metadata_failures(self):
        for key, value in (("MEASUREMENT", "profile"), ("ML_CUBE_OVERLAP", "6"),
                           ("MODEL", "transformer_inference_scripted_fw2.pt"),
                           ("PHYDLL_MODE", "blocking"), ("EXIT_STATUS", "1"),
                           ("SOLVER_EXIT_STATUS", "124"), ("WARMUP_CALLS", "1"),
                           ("MAIA", "/build_cpu_retest_scorep/bin/maia"), ("SOLVER_RANKS", "95")):
            with self.subTest(key=key):
                meta, rows = self.fixture()
                meta[key] = value
                with self.assertRaises(ValueError):
                    analysis.validate_cell(meta, "mlp_ov0", "timing", "readiness", rows)

    def test_call_failures(self):
        for kind in ("missing", "duplicate", "wrong", "nan", "max"):
            with self.subTest(kind=kind):
                meta, rows = self.fixture()
                if kind == "missing":
                    rows.pop()
                elif kind == "duplicate":
                    rows.append(rows[-1])
                elif kind == "wrong":
                    rows[-1]["step"] = 190
                elif kind == "nan":
                    rows[0]["ml_ms"] = float("nan")
                else:
                    rows[0]["ml_ms"] = 6
                with self.assertRaises(ValueError):
                    analysis.validate_cell(meta, "mlp_ov0", "timing", "readiness", rows)

    def test_selector_duplicate_and_matrix(self):
        jobs = copy.deepcopy(analysis.JOBS)
        jobs[("transformer", "timing")] = (4952696,) * 7
        with self.assertRaisesRegex(ValueError, "duplicate job"):
            analysis.select(jobs)
        jobs.pop(("mlp_ov4", "memory"))
        with self.assertRaisesRegex(ValueError, "incomplete matrix"):
            analysis.select(jobs)

    def test_normalization(self):
        result = analysis.normalize(192, 96, 672, 96)
        self.assertEqual(result["inclusive_rankavg_s"], 2)
        self.assertEqual(result["exclusive_rankavg_s"], 1)
        self.assertEqual(result["visits_rankavg"], 7)
        self.assertAlmostEqual(result["exclusive_ms_per_visit"], 1000/7)
        # Extraction's five visits per call must not be replaced with seven ML calls.
        self.assertAlmostEqual(analysis.normalize(96, 96, 3360, 96)["exclusive_ms_per_visit"], 1000/35)
        self.assertEqual(analysis.normalize(0, 0, 0, 96)["exclusive_ms_per_visit"], "")

    def test_bad_normalization(self):
        for values in ((1, 2, 7, 96), (1, -1, 7, 96), (1, 1, 7, 0),
                       (float("nan"), 1, 7, 96), (1, 1, -1, 96)):
            with self.assertRaises(ValueError):
                analysis.normalize(*values)

    def test_memory_failures(self):
        meta, rows = self.fixture()
        meta.update(MEASUREMENT="memory", RUN_STEPS="80", EXPECTED_INFERENCE_CALLS="3",
                    MEMORY_VALID="true", MEMORY_PEAK_BYTES="1073741824",
                    MEMORY_SCOPE="single-cell-job-cgroup-including-startup-and-page-cache",
                    MEM_CGROUP_PEAK_PATH="/sys/job_1/memory.peak")
        analysis.validate_cell(meta, "mlp_ov0", "memory", "readiness", rows[:3])
        for key, value in (("MEMORY_VALID", "false"), ("MEMORY_PEAK_BYTES", "0"),
                           ("MEMORY_SCOPE", "task"), ("MEM_CGROUP_PEAK_PATH", "/sys/memory.current")):
            changed = dict(meta, **{key: value})
            with self.subTest(key=key), self.assertRaises(ValueError):
                analysis.validate_cell(changed, "mlp_ov0", "memory", "readiness", rows[:3])

    def test_smartsim_variant(self):
        meta, rows = self.fixture()
        meta.update(PROVIDER="SMARTSIM", SS_CONFIG="SS_DEFAULT")
        analysis.validate_cell(meta, "mlp_ov0", "timing", "SS_DEFAULT", rows)
        with self.assertRaises(ValueError):
            analysis.validate_cell(meta, "mlp_ov0", "timing", "SS_TPQ1_I96_B96", rows)

    @unittest.skipUnless((analysis.OUTPUT / "profile_exclusive_bins.csv").exists(), "run renderer first")
    def test_rendered_profile_conservation_and_visits(self):
        with (analysis.OUTPUT / "profile_exclusive_bins.csv").open() as handle:
            bins = list(csv.DictReader(handle))
        groups = {}
        for row in bins:
            groups.setdefault((row["job"], row["scope"]), []).append(row)
        self.assertEqual(len(groups), 27)  # 21 solver + six PhyDLL DL communicators
        for rows in groups.values():
            self.assertEqual(len(rows), 4)
            self.assertTrue(math.isclose(sum(float(r["exclusive_rankavg_s"]) for r in rows),
                                         float(rows[0]["root_inclusive_rankavg_s"]), rel_tol=2e-4))
        with (analysis.OUTPUT / "profile_regions.csv").open() as handle:
            regions = list(csv.DictReader(handle))
        for row in regions:
            visits = float(row["visits_rankavg"])
            if visits:
                self.assertAlmostEqual(float(row["exclusive_ms_per_visit"]),
                                       1000 * float(row["exclusive_rankavg_s"]) / visits, places=6)
        extraction = [r for r in regions if r["region"] == "flowex_extract_cubes"]
        self.assertEqual(len(extraction), 21)
        self.assertTrue(all(float(r["visits_rankavg"]) == 35 for r in extraction))

    def test_real_final_selection(self):
        manifest, timing, memory = analysis.select()
        self.assertEqual(len(manifest), 63)
        self.assertEqual(sum(r["steady"] for r in timing), 105)
        self.assertEqual(len(timing), 147)
        self.assertEqual(len(memory), 21)
        self.assertEqual(len({r["job"] for r in manifest}), 63)


if __name__ == "__main__":
    unittest.main()
