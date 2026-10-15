"""Numeric steady-v1 contracts, without cube_dump, Slurm, or inference runs."""
import importlib.util
import io
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch


spec = importlib.util.spec_from_file_location("steady_verify", Path(__file__).with_name("verify_cpu_run.py"))
verify = importlib.util.module_from_spec(spec)
spec.loader.exec_module(verify)


def node(cid, name, parent, visits=5, time=1):
    return SimpleNamespace(id=cid, name=name, parent_id=parent, visits=visits,
                           incl_time=time, excl_time=0)


def solver_tree(provider, calls=7):
    count = max(calls - 2, 0)
    names = {
        "AIX": ["aix_library_static_step", "aix_inference", "inference", "gatherInputData",
                "scatterOutputData", "torchInference::forward"],
        "PHYDLL": ["phydll_library_static_step", "phydll_send", "phydll_recv", "phydll_unpack"],
        "SMARTSIM": ["smartsim_library_static_step", "smartsim_chunk_plan", "smartsim_put_tensor",
                     "smartsim_run_model", "smartsim_unpack_tensor"],
    }[provider]
    tree = {0: node(0, "m-AIA-Solver::run", None, 1),
            1: node(1, "cmi_inference_steady", 0, count)}
    for cid, name in enumerate(names, 2):
        tree[cid] = node(cid, name, 1 if cid == 2 else 2,
                         count if cid == 2 else count * 3)
    return tree


class SteadyTests(unittest.TestCase):
    def run_profile(self, trees, provider="AIX", calls=7, version="steady-v1", ranks=None):
        if ranks is None:
            ranks = list(range(len(trees)))
        dl = 96 if provider == "PHYDLL" else 0
        solver = len(trees) - dl
        with tempfile.TemporaryDirectory() as tmp:
            run = Path(tmp)
            (run / "run.info").write_text(
                f"EXPECTED_INFERENCE_CALLS={calls}\nMEASUREMENT=profile\nSOLVER_RANKS={solver}\n"
                f"DL_RANKS={dl}\nPROVIDER={provider}\nPROFILE_SCOPE_VERSION={version}\n")
            (run / "solver.log").write_text("\n".join(
                f"ML_STEP_TIMING step={15 + 29 * i}" for i in range(calls)))
            (run / "scorep").mkdir()
            # Location IDs intentionally do not equal MPI ranks.
            xml = "<cube>" + "".join(
                f'<locationgroup><type>process</type><rank>{rank}</rank>'
                f'<location Id="{100 + i * 2}"><type>thread</type></location></locationgroup>'
                for i, rank in enumerate(ranks))
            xml += "".join(f"<region><name>{name}</name></region>" for name in
                           ["m-AIA-Solver::run", "dl_recv", "dl_send", "dl_torch_forward"]) + "</cube>"
            with tarfile.open(run / "scorep/profile.cubex", "w") as archive:
                member = tarfile.TarInfo("anchor.xml")
                member.size = len(xml)
                archive.addfile(member, io.BytesIO(xml.encode()))
            with patch.object(verify, "parse_profile", return_value=trees) as parser:
                verify.verify(run)
                self.assertEqual(parser.call_count, int(version == "steady-v1"))

    def test_production_and_smoke(self):
        for provider in ("AIX", "SMARTSIM"):
            for calls in (3, 7):
                with self.subTest(provider=provider, calls=calls):
                    self.run_profile([solver_tree(provider, calls)], provider, calls)

    def test_count_checked_per_rank_not_average(self):
        trees = [solver_tree("AIX"), solver_tree("AIX")]
        trees[0][1].visits = 4
        trees[1][1].visits = 6
        with self.assertRaisesRegex(ValueError, "visits on rank 0"):
            self.run_profile(trees)

    def test_static_count(self):
        tree = solver_tree("AIX")
        tree[2].visits = 7
        with self.assertRaisesRegex(ValueError, "aix_library_static_step visits"):
            self.run_profile([tree])

    def test_misplaced_parent(self):
        tree = solver_tree("AIX")
        tree[1].parent_id = None
        with self.assertRaisesRegex(ValueError, "must be under"):
            self.run_profile([tree])

    def test_aix_whole_profile_phases_do_not_satisfy_scope(self):
        tree = solver_tree("AIX")
        tree[5].parent_id = 0
        with self.assertRaisesRegex(ValueError, "gatherInputData"):
            self.run_profile([tree])

    def test_unconditional_aix_warmup_and_repeated_forward_allowed(self):
        tree = solver_tree("AIX")
        tree[20] = node(20, "inference", 0, 2)
        tree[21] = node(21, "torchInference::forward", 20, 8)
        self.run_profile([tree])

    def test_missing_or_unscoped_smart_regions(self):
        for name in ("smartsim_put_tensor", "smartsim_run_model", "smartsim_unpack_tensor"):
            for misplaced in (False, True):
                tree = solver_tree("SMARTSIM")
                cid = next(cid for cid, n in tree.items() if n.name == name)
                if misplaced:
                    tree[cid].parent_id = 0
                else:
                    del tree[cid]
                with self.subTest(name=name, misplaced=misplaced):
                    with self.assertRaisesRegex(ValueError, name):
                        self.run_profile([tree], "SMARTSIM")

    def test_smart_regions_need_positive_aggregate_not_each_rank(self):
        trees = [solver_tree("SMARTSIM"), solver_tree("SMARTSIM")]
        run_model = next(cid for cid, n in trees[1].items() if n.name == "smartsim_run_model")
        del trees[1][run_model]
        self.run_profile(trees, "SMARTSIM")

    def test_invalid_parent_time(self):
        for time in (0, -1, float("nan"), float("inf")):
            tree = solver_tree("AIX")
            tree[1].incl_time = time
            with self.subTest(time=time), self.assertRaisesRegex(ValueError, "cmi_inference_steady"):
                self.run_profile([tree])

    def test_no_steady_calls(self):
        self.run_profile([solver_tree("AIX", 2)], calls=2)

    def test_legacy_does_not_load_numeric_parser(self):
        for version in ("", "future-v2"):
            self.run_profile([{}], version=version)

    def phydll_trees(self):
        dl = {0: node(0, "phydll_dl_client", None, 1),
              1: node(1, "dl_inference_steady", 0),
              2: node(2, "dl_recv", 1), 3: node(3, "dl_torch_forward", 1, 15),
              4: node(4, "dl_send", 1)}
        return [solver_tree("PHYDLL")] + [dict((cid, SimpleNamespace(**vars(n)))
                                               for cid, n in dl.items()) for _ in range(96)]

    def test_phydll_all_dl_ranks_and_location_mapping(self):
        trees = self.phydll_trees()
        self.run_profile(trees[1:] + trees[:1], "PHYDLL", ranks=list(range(1, 97)) + [0])

    def test_phydll_bad_dl_count(self):
        trees = self.phydll_trees()
        trees[-1][1].visits = 4
        with self.assertRaisesRegex(ValueError, "dl_inference_steady visits on rank 96"):
            self.run_profile(trees, "PHYDLL")

    def test_unexpected_parent_visit_rank(self):
        trees = self.phydll_trees()
        trees[0][20] = node(20, "dl_inference_steady", 0)
        with self.assertRaisesRegex(ValueError, "unexpected dl_inference_steady visit rank 0"):
            self.run_profile(trees, "PHYDLL")

    def test_missing_numeric_trees(self):
        with self.assertRaisesRegex(ValueError, "numeric rank trees"):
            self.run_profile([{}])


if __name__ == "__main__":
    unittest.main()
