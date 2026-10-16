"""Synthetic diagnostic contracts; no Slurm, Cube installation, or plotting needed."""
import csv
import importlib.util
import math
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location("variance", Path(__file__).with_name("phydll_variance.py"))
v = importlib.util.module_from_spec(spec)
spec.loader.exec_module(v)


def rows():
    result = []
    for scope, names in v.REGIONS.items():
        for cnode, name in enumerate(names):
            for rank in range(96) if scope == "solver_steady" else range(96, 192):
                result.append(dict(scope=scope, region=name, cnode=cnode, rank=rank,
                                   visits=5, inclusive_s=rank + 1, exclusive_s=0))
    return result


FRAMES = {"solver_steady": 5, "dl_steady": 5}


class VarianceTests(unittest.TestCase):
    def test_distribution_and_units(self):
        ranks, phases = v.phase_metrics(rows(), FRAMES)
        self.assertEqual(len(ranks), 12 * 96)
        first = phases[0]
        self.assertEqual(first["mean_ms"], 48.5 * 200)
        self.assertEqual(first["maxrank"], 95)
        self.assertAlmostEqual(first["p10_ms"], 10.5 * 200)
        self.assertEqual(ranks[0]["total_s"], 1)
        self.assertEqual(ranks[0]["ms_per_call"], 200)

    def test_optional_absence_is_not_zero(self):
        data = [r for r in rows() if r["region"] != "dl_frame_copy"]
        _, phases = v.phase_metrics(data, FRAMES)
        absent = next(r for r in phases if r["region"] == "dl_frame_copy")
        self.assertEqual(absent["status"], "missing_optional")
        self.assertNotIn("mean_ms", absent)

    def test_child_visits_are_not_the_denominator(self):
        data = rows()
        for row in data:
            if row["region"] == "flowex_extract_cubes":
                row["visits"] = 25
        ranks, _ = v.phase_metrics(data, FRAMES)
        child = next(r for r in ranks if r["region"] == "flowex_extract_cubes")
        self.assertEqual(child["ms_per_call"], 200)

    def test_missing_core_wrong_scope_and_frames(self):
        for name in v.CORE:
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "core"):
                v.phase_metrics([r for r in rows() if r["region"] != name], FRAMES)
        data = rows()
        for row in data:
            if row["region"] == "dl_torch_forward":
                row["scope"] = "solver_steady"
        with self.assertRaisesRegex(ValueError, "core"):
            v.phase_metrics(data, FRAMES)
        for frames in ({}, {"solver_steady": 5}, {"solver_steady": 5, "dl_steady": 7}):
            with self.assertRaisesRegex(ValueError, "frames"):
                v.phase_metrics(rows(), frames)

    def test_partial_duplicate_rank_cnode_visits_and_nonfinite(self):
        for field, value in (("rank", 1), ("cnode", 999), ("visits", 4),
                             ("inclusive_s", math.nan), ("inclusive_s", math.inf),
                             ("inclusive_s", 1e308),
                             ("exclusive_s", -1)):
            data = rows()
            data[0][field] = value
            with self.subTest(field=field, value=value), self.assertRaises(ValueError):
                v.phase_metrics(data, FRAMES)
        with self.assertRaises(ValueError):
            v.phase_metrics(rows()[1:], FRAMES)
        with self.assertRaises(ValueError):
            v.phase_metrics(rows() + [rows()[0]], FRAMES)

    def test_ci_only_five_and_sample_std(self):
        ci = v.load("maia_latency_ci")
        runs = [dict(workload="transformer", variant="blocking", mean_ms=i) for i in range(1, 6)]
        for n in range(1, 6):
            summary = v.aggregate([], runs[:n], ci)[0]
            self.assertEqual(summary["n"], n)
            self.assertEqual(summary["ci_half_ms"] != "", n == 5)
        summary = v.aggregate([], runs, ci)[0]
        self.assertAlmostEqual(summary["std_ms"], math.sqrt(2.5))
        self.assertAlmostEqual(summary["cv"], math.sqrt(2.5) / 3)

    def test_verified_pending_and_invalid_completed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            log = root / "run.out"
            module = SimpleNamespace(run_dir=lambda path: root / "absent")
            entry = dict(log=log, rep=2)
            self.assertIsNone(v.verified_run(entry, module, True))
            entry["rep"] = 1
            with self.assertRaises(ValueError):
                v.verified_run(entry, module, True)
            entry["rep"] = 2
            log.write_text("NOT VERIFIED\n")
            self.assertIsNone(v.verified_run(entry, module, True))
            log.write_text(f"RUN_DIR={root / 'absent'}\nVERIFIED\n")
            with self.assertRaisesRegex(ValueError, "preserved run"):
                v.verified_run(entry, module, True)
            log.write_text("VERIFIED\n")
            with self.assertRaisesRegex(ValueError, "RUN_DIR"):
                v.verified_run(entry, module, True)
            log.write_text(f"RUN_DIR={root}\nVERIFIED\n")
            module.run_dir = lambda path: root
            self.assertEqual(v.verified_run(entry, module, False), root.resolve())

    def test_manifest_duplicate_job_cell_and_invalid_scope(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            jobs = {job: (w, variant, 96) for job, (w, variant) in zip(v.BASELINES, sorted(v.CELLS))}
            module = SimpleNamespace(LOGS=root, JOBS=jobs)
            for job in v.BASELINES:
                (root / f"retest_profile_test_{job}.out").write_text("VERIFIED\n")
            def write(data):
                with (root / "jobs.csv").open("w", newline="") as stream:
                    writer = csv.DictWriter(stream, fieldnames=["job", "workload", "variant", "rep", "log"])
                    writer.writeheader()
                    writer.writerows(data)
            row = dict(job="900", workload="transformer", variant="blocking", rep=2, log=str(root / "extra.out"))
            write([row])
            self.assertEqual(len(v.manifest(root, module)), 5)
            for other in (row, dict(row, job="901"), dict(row, job=str(v.BASELINES[0]), rep=3),
                          dict(row, rep=1), dict(row, variant="AIX"), dict(row, log="relative")):
                write([row, other])
                with self.subTest(other=other), self.assertRaises(ValueError):
                    v.manifest(root, module)
            (root / f"retest_profile_test_{v.BASELINES[0]}.out").unlink()
            write([])
            with self.assertRaisesRegex(ValueError, "baseline"):
                v.manifest(root, module)

    @unittest.skipUnless(importlib.util.find_spec("matplotlib"), "optional matplotlib not installed")
    def test_optional_pngs(self):
        rank_rows, phase_rows = v.phase_metrics(rows(), FRAMES)
        identity = dict(job="900", workload="transformer", variant="blocking", rep=1)
        rank_rows = [dict(identity, **row) for row in rank_rows]
        phase_rows = [dict(identity, **row) for row in phase_rows]
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            v.plots(root, rank_rows, phase_rows, [dict(identity, mean_ms=100)])
            self.assertEqual(len(list(root.glob("*.png"))), 5)
            self.assertTrue(all(p.stat().st_size > 1000 for p in root.glob("*.png")))

    def test_analysis_outputs_duplicate_runs_and_completed_errors(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            campaign = root / "campaign"
            campaign.mkdir()
            (campaign / "jobs.csv").write_text("job,workload,variant,rep,log\n")
            cm = v.load("extract_steady_campaign")
            cm.LOGS = root
            extract = v.load("extract_steady_profiles")
            ci = v.load("maia_latency_ci")
            modules = {"extract_steady_campaign": cm, "extract_steady_profiles": extract,
                       "maia_latency_ci": ci}
            directories = []
            for job in v.BASELINES:
                directory = root / str(job)
                directory.mkdir()
                directories.append(directory)
                lines = [f"ML_STEP_TIMING step={15 + 29*i} ml_ms={i+1} rank_min_ms=1 rank_max_ms={i+1}"
                         for i in range(7)]
                (directory / "solver.log").write_text("\n".join(lines))
                (root / f"retest_profile_test_{job}.out").write_text(f"RUN_DIR={directory}\nVERIFIED profile\n")
            with patch.object(v, "load", side_effect=modules.__getitem__), \
                 patch.object(extract, "extract_cube", return_value=([], rows(), FRAMES)):
                with self.assertRaisesRegex(ValueError, "missing repetitions"):
                    v.analyze(campaign)
                self.assertFalse((campaign / "analysis").exists())
                output = v.analyze(campaign, allow_missing=True)
                self.assertEqual(output, campaign.resolve() / "analysis")
                with (output / "steady_calls.csv").open() as stream:
                    self.assertEqual(len(list(csv.DictReader(stream))), 20)
                with (output / "phase_summary.csv").open() as stream:
                    summaries = list(csv.DictReader(stream))
                self.assertTrue(all(r["n"] == "1" and r["ci_half_ms"] == "" for r in summaries))
                self.assertIn("NOT", (output / "README.md").read_text())
                self.assertEqual({p.name for p in campaign.iterdir()}, {"jobs.csv", "analysis"})
                second_log = root / f"retest_profile_test_{v.BASELINES[1]}.out"
                second_log.write_text(f"RUN_DIR={directories[0]}\nVERIFIED profile\n")
                with self.assertRaisesRegex(ValueError, "duplicate run"):
                    v.analyze(campaign, allow_missing=True)
                second_log.write_text(f"RUN_DIR={directories[1]}\nVERIFIED profile\n")
                with patch.object(extract, "extract_cube", side_effect=ValueError("invalid completed cube")):
                    with self.assertRaisesRegex(ValueError, "invalid completed cube"):
                        v.analyze(campaign, allow_missing=True)
                extra = root / "extra.out"
                extra.write_text(f"RUN_DIR={root / 'nonexistent'}\nVERIFIED profile\n")
                (campaign / "jobs.csv").write_text(
                    f"job,workload,variant,rep,log\n900,transformer,blocking,2,{extra}\n")
                with self.assertRaisesRegex(ValueError, "invalid preserved run"):
                    v.analyze(campaign, allow_missing=True)


if __name__ == "__main__":
    unittest.main()
