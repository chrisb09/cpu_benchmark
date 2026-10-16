"""Statistical, scope and rendering checks for the read-only comparison pipeline."""
import json
import math
import tempfile
import unittest
from pathlib import Path

import matplotlib.pyplot as plt
import plot_cpu_comparison as plot


class StatisticsTests(unittest.TestCase):
    def test_five_run_t_ci(self):
        mean, ci, status = plot.mean_ci([1, 2, 3, 4, 5])
        self.assertEqual(mean, 3)
        self.assertAlmostEqual(ci, 2.776445 * math.sqrt(2.5) / math.sqrt(5))
        self.assertEqual(status, "available_95pct_t_df4")

    def test_single_run_no_ci(self):
        self.assertIsNone(plot.mean_ci([3])[1])
        self.assertIsNone(plot.mean_ci([1, 2, 3, 4, 5], independent=False)[1])

    def test_wrong_independent_count_rejected(self):
        with self.assertRaises(ValueError):
            plot.mean_ci([1, 2, 3])

    def test_nonfinite_rejected(self):
        for value in ("nan", "inf", "-inf", -1):
            with self.subTest(value=value), self.assertRaises(ValueError):
                plot.mean_ci([value])

    def test_empty_rejected(self):
        with self.assertRaises(ValueError):
            plot.mean_ci([])


class PhaseTests(unittest.TestCase):
    def rows(self):
        return [{"region": "dl_recv", "cnode": "1", "inclusive_rankavg_s": "6", "visits_rankavg": "6"},
                {"region": "dl_frame_copy", "cnode": "2", "inclusive_rankavg_s": "12", "visits_rankavg": "6"}]

    def test_sum_uses_six_frames_not_summed_visits(self):
        self.assertEqual(plot.inclusive_phase(self.rows(), ("dl_recv", "dl_frame_copy"), 6,
                                             {"1": "0", "2": "0"}), 3000)

    def test_seven_not_six_rejected(self):
        with self.assertRaises(ValueError):
            plot.inclusive_phase(self.rows(), ("dl_recv",), 7, {"1": "0"})

    def test_inclusive_ancestor_rejected(self):
        with self.assertRaises(ValueError):
            plot.inclusive_phase(self.rows(), ("dl_recv", "dl_frame_copy"), 6,
                                 {"1": "0", "2": "1"})

    def test_duplicate_rejected(self):
        with self.assertRaises(ValueError):
            plot.assert_disjoint(["1", "1"], {"1": "0"})

    def test_missing_rejected(self):
        with self.assertRaises(ValueError):
            plot.inclusive_phase(self.rows(), ("not_recorded",), 6, {})

    def test_ambiguous_rejected(self):
        with self.assertRaises(ValueError):
            plot.inclusive_phase(self.rows() * 2, ("dl_recv",), 6, {})


class SourceAndRenderingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.latency, cls.memory, cls.phases, cls.audit, cls.pre, cls.hashes = plot.load_data()
        plot.style()

    def tearDown(self):
        plt.close("all")

    def test_source_counts_and_hashes(self):
        counts = {Path(r["path"]).name: r["rows"] for r in self.hashes}
        self.assertEqual(counts["source_manifest.csv"], 63)
        for name in ("profile_manifest.csv", "timing_replicates.csv", "memory_replicates.csv"):
            self.assertEqual(counts[name], 70)
        self.assertTrue(all(len(r["sha256"]) == 64 for r in self.hashes))

    def test_main_excludes_readiness(self):
        self.assertEqual(len(plot.MAIN), 6)
        self.assertNotIn("readiness", plot.MAIN)
        fig = plot.comparison(["watercnn"], self.latency, self.memory, self.phases)
        self.assertFalse(any("readiness" in t.get_text() for ax in fig.axes for t in ax.get_xticklabels()))

    def test_optional_ov0_selection(self):
        self.assertNotIn("mlp_ov0", plot.selected_models())
        self.assertIn("mlp_ov0", plot.selected_models(True))

    def test_maia_ci_and_recorded_phases(self):
        # MAIA latency is now a five-independent-run mean with a 95% t CI.
        rows = [r for r in self.latency if r["model"] in plot.MODELS[2:]]
        self.assertTrue(rows)
        self.assertTrue(all(r["independent_runs"] == 5 and r["ci_half_ms"] is not None
                            and r["ci_status"] == "available_95pct_t_df4" for r in rows))
        # Two-warmup scope means every provider, SmartSim included, is recorded.
        self.assertFalse([r for r in self.phases if r["status"] == "unavailable_not_recorded"])
        maia = [r for r in self.phases if r["model"] in plot.MODELS[2:]]
        self.assertTrue(maia and all(r["frames"] == plot.STEADY_CALLS for r in maia))

    def test_maia_ci_uses_five_run_means(self):
        for r in [x for x in self.latency if x["model"] in plot.MODELS[2:]]:
            self.assertEqual(len(json.loads(r["samples_ms"])), 5)

    def test_visit_counts_and_no_send_descendant(self):
        self.assertTrue(all(r["visits_min"] == r["visits_max"] == plot.STEADY_CALLS for r in self.audit))
        self.assertTrue(all(r["denominator_frames"] == plot.STEADY_CALLS for r in self.audit))
        self.assertNotIn("dl_send", {r["region"] for r in self.audit})
        self.assertTrue(all(r["visits_rankavg"] == plot.STEADY_CALLS
                            and r["denominator_calls"] == plot.STEADY_CALLS for r in self.pre))

    def test_memory_is_maximum_without_ci(self):
        self.assertTrue(all(r["ci_status"] == "not_applicable_maximum" for r in self.memory))
        self.assertEqual({r["metric"] for r in self.memory if r["model"] == "watercnn"}, {"PSS", "RSS"})
        self.assertEqual({r["metric"] for r in self.memory if r["model"] == "transformer"}, {"Kernel"})

    def test_linear_axes_and_independent_scales(self):
        fig = plot.comparison(plot.MODELS[:2], self.latency, self.memory, self.phases)
        self.assertEqual(len(fig.axes), 6)
        self.assertTrue(all(ax.get_yscale() == "linear" and ax.get_ylim()[0] == 0 for ax in fig.axes))
        self.assertNotEqual(fig.axes[0].get_ylim(), fig.axes[3].get_ylim())

    def test_readiness_four_paired_phase_groups(self):
        fig = plot.comparison(["transformer"], self.latency, self.memory, self.phases, True)
        self.assertEqual(len(fig.axes), 3)
        self.assertEqual(len(fig.axes[1].patches), 8)
        self.assertEqual(len(fig.axes[1].get_xticklabels()), 4)
        self.assertEqual(len(fig.axes[0].patches), 3)
        self.assertIn("PhyDLL\ndefault", [t.get_text() for t in fig.axes[0].get_xticklabels()])

    def test_single_memory_bar_per_configuration(self):
        for model in plot.MODELS:
            fig, ax = plt.subplots()
            plot.memory_panel(ax, model, self.memory, plot.MAIN)
            self.assertEqual(len(ax.patches), len(plot.MAIN))
            metric = "PSS" if model in plot.MODELS[:2] else "Kernel"
            expected = [next(r["value_GiB"] for r in self.memory
                             if (r["model"], r["variant"], r["metric"]) == (model, v, metric))
                        for v in plot.MAIN]
            self.assertEqual([p.get_height() for p in ax.patches], expected)

    def test_solver_layouts_and_uniform_phase_width(self):
        for models in (plot.MODELS[:2], plot.MODELS[2:]):
            fig = plot.solver_summary(models, self.latency, self.memory, self.phases)
            self.assertEqual(len(fig.axes), 4)
            self.assertAlmostEqual(fig.get_figwidth() * 25.4, 150, places=3)
            self.assertTrue(all(not ax.get_title(loc="left") for ax in fig.axes))
            fig = plot.solver_summary(models, self.latency, self.memory, self.phases, phase_only=True)
            self.assertEqual(len(fig.axes), 2)
            for ax in fig.axes:
                self.assertEqual(len({round(p.get_width(), 8) for p in ax.patches}), 1)
                self.assertIn("Forward", [t.get_text() for t in ax.get_xticklabels()])
                self.assertTrue(any(t.get_text() == "AIx" for t in ax.texts))
                self.assertEqual(ax.get_ylabel(), "Phase time (s / rank)")

    def test_native_panel_export_dimensions(self):
        from matplotlib.transforms import Bbox
        from PIL import Image
        fig = plot.solver_summary(plot.MODELS[:2], self.latency, self.memory, self.phases, captions=False)
        slot = (plot.THESIS_WIDTH_IN - 3 * plot.PANEL_GAP_IN) / 4
        with tempfile.TemporaryDirectory(dir=plot.ROOT / "tmp") as tmp:
            records = []
            plot.save(fig, Path(tmp), "panel", records, native=True,
                      bbox=Bbox.from_bounds(0, 0, slot, fig.get_figheight()))
            self.assertAlmostEqual(records[0]["native_width_mm"], 36, places=3)
            self.assertEqual(records[0]["panels"], 1)
            with Image.open(Path(tmp) / "panel.png") as image:
                self.assertAlmostEqual(image.width, slot * 170, delta=1)

    def test_solver_captions_and_full_width_phases(self):
        fig = plot.solver_summary(plot.MODELS[:2], self.latency, self.memory, self.phases)
        self.assertEqual(len(fig.texts), 4)
        self.assertTrue(fig.texts[0].get_text().startswith("(a) [TS]"))
        self.assertTrue(fig.texts[-1].get_text().startswith("(d) [TS]"))
        self.assertIn("SmartSim default", [t.get_text() for t in fig.axes[0].get_xticklabels()])
        standalone = plot.solver_summary(plot.MODELS[:2], self.latency, self.memory, self.phases, captions=False)
        self.assertFalse(standalone.texts)
        for model in plot.MODELS:
            fig = plot.solver_summary([model], self.latency, self.memory, self.phases,
                                      phase_only=True, captions=False)
            self.assertEqual(len(fig.axes), 1)
            self.assertFalse(fig.texts)
            self.assertAlmostEqual(fig.get_figwidth() * 25.4, 150, places=3)
            self.assertTrue(any(t.get_text().startswith("SmartSim") for t in fig.axes[0].texts))

    def test_rendered_dimensions_and_panels(self):
        with tempfile.TemporaryDirectory(dir=plot.ROOT / "tmp") as tmp:
            records = []
            fig = plot.comparison(["watercnn"], self.latency, self.memory, self.phases)
            plot.save(fig, Path(tmp), "comparison", records)
            self.assertEqual(records[0]["panels"], 3)
            self.assertGreater(records[0]["width"], 2000)
            self.assertGreater(records[0]["height"], 350)
            self.assertLess(records[0]["height"], 700)
            self.assertGreater(records[0]["pixel_std"], 8)

    def test_generated_manifest_and_full_legend(self):
        manifest = json.loads((plot.OUTPUT / "manifest.json").read_text())
        self.assertGreaterEqual(len(manifest["figures"]), 14)
        self.assertEqual(manifest["main_models"], list(plot.MODELS))
        self.assertEqual(manifest["phase_unavailable_rows"], 0)
        for record in manifest["figures"]:
            self.assertTrue((plot.OUTPUT / (record["name"] + ".png")).is_file())
            self.assertTrue((plot.OUTPUT / (record["name"] + ".pdf")).is_file())
        fig = plt.figure()
        plot.phase_legend(fig, full=True)
        self.assertEqual({t.get_text() for t in fig.legends[0].get_texts()}, set(plot.FULL_LEGEND.values()))


if __name__ == "__main__":
    unittest.main()
