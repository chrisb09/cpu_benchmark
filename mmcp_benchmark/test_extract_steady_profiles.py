"""Steady-subtree extraction contracts, without cube_dump or real profiles."""
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

spec = importlib.util.spec_from_file_location(
    "steady_extract", Path(__file__).with_name("extract_steady_profiles.py"))
extract = importlib.util.module_from_spec(spec)
spec.loader.exec_module(extract)


def node(cid, name, parent, visits, incl=1.0, excl=0.0):
    return SimpleNamespace(id=cid, name=name, parent_id=parent, children_ids=[],
                           visits=visits, incl_time=incl, excl_time=excl)


def link(tree, parent_id, *children):
    for child in children:
        tree[parent_id].children_ids.append(child.id)
        tree[child.id] = child


def solver_tree(visits=5):
    tree = {1: node(1, "m-AIA-Solver::run", None, 1),
            2: node(2, "cmi_inference_steady", 1, visits)}
    link(tree, 2, node(3, "app_provider_inference", 2, visits),
         node(4, "app_finalize_output", 2, visits))
    link(tree, 3, node(5, "smartsim_run_model", 3, visits))
    return tree


def dl_tree(visits=5):
    tree = {10: node(10, "phydll_dl_client", None, 1),
            11: node(11, "dl_inference_steady", 10, visits)}
    link(tree, 11, node(12, "dl_recv", 11, visits), node(13, "dl_torch_forward", 11, visits))
    return tree


class ExtractTests(unittest.TestCase):
    def run_extract(self, trees, solver_ranks, dl_ranks=0, workload="mlp_ov4", variant="X"):
        with tempfile.TemporaryDirectory() as tmp:
            cube = Path(tmp) / "profile.cubex"
            xml = "<cube>" + "".join(
                f'<locationgroup><type>process</type><rank>{i}</rank>'
                f'<location Id="{i}"><type>thread</type></location></locationgroup>'
                for i in range(len(trees))) + "</cube>"
            with tarfile.open(cube, "w") as archive:
                member = tarfile.TarInfo("anchor.xml")
                member.size = len(xml)
                archive.addfile(member, io.BytesIO(xml.encode()))
            module = SimpleNamespace(parse_profile=lambda _: trees)
            return extract.extract_cube(module, cube, workload, variant, "1", solver_ranks, dl_ranks)

    def test_solver_scope_aggregates_per_visit(self):
        trees = [solver_tree(5), solver_tree(5)]
        regions, ranks, frames = self.run_extract(trees, solver_ranks=2)
        self.assertEqual(frames, {"solver_steady": 5})
        parent = next(r for r in regions if r["region"] == "cmi_inference_steady")
        self.assertEqual(parent["rank_count"], 2)
        self.assertEqual(parent["visits_rankavg"], 5)
        # Parent plus three descendants (app_provider_inference, app_finalize_output,
        # smartsim_run_model); the solver root itself is an ancestor, not a member.
        self.assertEqual(len([r for r in ranks if r["scope"] == "solver_steady"]), 2 * 4)

    def test_dl_scope_uses_only_dl_ranks(self):
        trees = [solver_tree(5), dl_tree(5)]
        regions, _, frames = self.run_extract(trees, solver_ranks=1, dl_ranks=1)
        self.assertEqual(frames, {"solver_steady": 5, "dl_steady": 5})
        dl = [r for r in regions if r["scope"] == "dl_steady"]
        self.assertTrue(dl and all(r["rank_count"] == 1 for r in dl))
        names = {r["region"] for r in dl}
        self.assertIn("dl_recv", names)
        self.assertNotIn("app_provider_inference", names)

    def test_denominator_follows_parent_not_assumed(self):
        regions, _, frames = self.run_extract([solver_tree(3)], solver_ranks=1)
        self.assertEqual(frames, {"solver_steady": 3})

    def test_inconsistent_visits_rejected(self):
        with self.assertRaisesRegex(ValueError, "inconsistent"):
            self.run_extract([solver_tree(5), solver_tree(4)], solver_ranks=2)

    def test_missing_parent_rejected(self):
        tree = solver_tree(5)
        tree[2].name = "not_the_parent"
        with self.assertRaisesRegex(ValueError, "cmi_inference_steady"):
            self.run_extract([tree], solver_ranks=1)

    def test_rank_set_mismatch_rejected(self):
        with self.assertRaisesRegex(ValueError, "rank set"):
            self.run_extract([solver_tree(5)], solver_ranks=2)


if __name__ == "__main__":
    unittest.main()
