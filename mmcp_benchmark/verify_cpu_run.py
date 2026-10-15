#!/usr/bin/env python3
"""Fail incomplete retest cells; verify Score-P's process topology and user regions."""
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import xml.etree.ElementTree as ET
from collections import defaultdict
from pathlib import Path
from types import SimpleNamespace


def _cube_dump():
    # The engine sources the Score-P module; accept PATH or the module root so
    # verification needs no pandas/numpy plotting dependencies.
    found = shutil.which("cube_dump")
    if found:
        return found
    root = os.environ.get("SCOREP_ROOT_DIR")
    if root:
        candidate = Path(root) / "bin" / "cube_dump"
        if candidate.is_file():
            return str(candidate)
    raise RuntimeError("cube_dump not found; source the Score-P environment")


def parse_profile(cube):
    """Dependency-free CUBE call-tree parse; one node map per Score-P location ID."""
    dump = _cube_dump()

    def dump_arg(*args):
        return subprocess.check_output([dump, *args, str(cube)], text=True, stderr=subprocess.DEVNULL)

    tree_out = dump_arg("-w", "calltree")
    incl_out = dump_arg("-m", "time", "-z", "incl", "-s", "csv2", "-c", "all")
    excl_out = dump_arg("-m", "time", "-z", "excl", "-s", "csv2", "-c", "all")
    vis_out = dump_arg("-m", "visits", "-z", "excl", "-s", "csv2", "-c", "all")

    def metric_map(raw):
        result = defaultdict(dict)
        for line in raw.splitlines():
            parts = line.strip().split(",")
            if len(parts) < 3 or line.startswith("Cnode ID"):
                continue
            try:
                result[int(parts[1])][int(parts[0])] = float(parts[2])
            except ValueError:
                continue
        return result

    incl_map, excl_map, vis_map = metric_map(incl_out), metric_map(excl_out), metric_map(vis_out)
    thread_ids = sorted(incl_map) or [0]

    skeleton, parent_stack, in_tree = [], [], False
    for line in tree_out.splitlines():
        if "--- CALL TREE ---" in line:
            in_tree = True
            continue
        if not in_tree or not line.strip():
            continue
        idx = line.find("[ ( id=")
        if idx == -1:
            continue
        prefix = line[:idx].rstrip()
        try:
            cid = int(line[idx + len("[ ( id="):].split(",")[0].split(")")[0].strip())
        except ValueError:
            continue
        name = prefix.lstrip(" |-\t")
        depth = len(prefix) - len(name)
        while parent_stack and parent_stack[-1][0] >= depth:
            parent_stack.pop()
        parent_id = parent_stack[-1][1] if parent_stack else None
        parent_stack.append((depth, cid))
        skeleton.append((cid, name, parent_id))

    trees = []
    for tid in thread_ids:
        nodes = {}
        for cid, name, parent_id in skeleton:
            nodes[cid] = SimpleNamespace(id=cid, name=name.removeprefix("user:"), parent_id=parent_id,
                                         children_ids=[], incl_time=incl_map.get(tid, {}).get(cid, 0.0),
                                         excl_time=excl_map.get(tid, {}).get(cid, 0.0),
                                         visits=int(vis_map.get(tid, {}).get(cid, 0)))
        for node in nodes.values():
            if node.parent_id is not None and node.parent_id in nodes:
                nodes[node.parent_id].children_ids.append(node.id)
        trees.append(nodes)
    return trees


def verify_steady_profile(cube, root, meta, expected):
    solver_count = int(meta["SOLVER_RANKS"])
    dl_count = int(meta.get("DL_RANKS", 0))
    provider = meta["PROVIDER"]
    if provider == "PHYDLL" and dl_count != 96:
        raise ValueError("steady-v1 PHYDLL requires 96 DL ranks")
    steady = max(expected - 2, 0)
    location_rank = {}
    for group in root.findall(".//locationgroup"):
        if group.findtext("type") != "process":
            continue
        locations = group.findall("location")
        if len(locations) != 1 or locations[0].findtext("type") != "thread":
            raise ValueError("steady-v1 requires one thread location per MPI rank")
        lid = int(locations[0].attrib["Id"])
        if lid in location_rank:
            raise ValueError("duplicate Score-P location ID")
        location_rank[lid] = int(group.findtext("rank"))
    trees = parse_profile(cube)
    if len(trees) != solver_count + dl_count or len(location_rank) != len(trees) or not all(trees):
        raise ValueError("empty/missing Score-P numeric rank trees")
    # The parser orders by location ID, not by MPI rank, and keeps zero-visit nodes.
    ranktrees = dict(zip((location_rank[i] for i in sorted(location_rank)), trees))
    aggregate = {}
    for rank, tree in ranktrees.items():
        def below(node, ancestor):
            seen = set()
            current = node.parent_id
            while current is not None and current in tree and current not in seen:
                if current == ancestor.id:
                    return True
                seen.add(current)
                current = tree[current].parent_id
            return False

        def metrics(nodes, label, visits=None):
            if any(not all(math.isfinite(v) and v >= 0 for v in
                           (n.visits, n.incl_time, n.excl_time)) for n in nodes):
                raise ValueError(f"invalid Score-P metrics for {label} on rank {rank}")
            total_visits = sum(n.visits for n in nodes)
            if visits is not None and total_visits != visits:
                raise ValueError(f"{label} visits on rank {rank}: {total_visits}, expected {visits}")
            values = aggregate.setdefault(label, [0, 0.0])
            values[0] += total_visits
            values[1] += sum(n.incl_time for n in nodes)

        solver = rank < solver_count
        parent_name = "cmi_inference_steady" if solver else "dl_inference_steady"
        for name in ("cmi_inference_steady", "dl_inference_steady"):
            if name != parent_name and any(n.name == name and n.visits != 0 for n in tree.values()):
                raise ValueError(f"unexpected {name} visit rank {rank}")
        roots = [n for n in tree.values() if n.name ==
                 ("m-AIA-Solver::run" if solver else "phydll_dl_client") and n.visits > 0]
        if len(roots) != 1:
            raise ValueError(f"missing/ambiguous Score-P root on rank {rank}")
        parents = [n for n in tree.values() if n.name == parent_name]
        if not parents or any(not below(n, roots[0]) for n in parents):
            raise ValueError(f"{parent_name} must be under {roots[0].name} on rank {rank}")
        metrics(parents, parent_name, steady)
        scoped = [n for n in tree.values() if any(below(n, p) for p in parents)]
        if solver:
            static = {"AIX": "aix_library_static_step", "PHYDLL": "phydll_library_static_step",
                      "SMARTSIM": "smartsim_library_static_step"}[provider]
            metrics([n for n in scoped if n.name == static], static, steady)
            phases = {"AIX": ("aix_inference", "inference", "gatherInputData", "scatterOutputData"),
                      "PHYDLL": ("phydll_send", "phydll_recv", "phydll_unpack"),
                      "SMARTSIM": ("smartsim_chunk_plan", "smartsim_put_tensor",
                                   "smartsim_run_model", "smartsim_unpack_tensor")}[provider]
            if provider == "AIX":
                metrics([n for n in scoped if n.name == "torchInference::forward"],
                        "torchInference::forward")
        else:
            phases = ("dl_recv", "dl_torch_forward", "dl_send")
        for name in phases:
            metrics([n for n in scoped if n.name == name], name)
    if steady:
        for name, (visits, inclusive) in aggregate.items():
            if visits <= 0 or not math.isfinite(inclusive) or inclusive <= 0:
                raise ValueError(f"missing/zero aggregate time for {name} within steady parent")


def verify(run):
    meta = dict(line.split("=", 1) for line in (run / "run.info").read_text().splitlines() if "=" in line)
    steps = [int(s) for s in re.findall(r"ML_STEP_TIMING step=(\d+)", (run / "solver.log").read_text())]
    expected = int(meta["EXPECTED_INFERENCE_CALLS"])
    if steps != [15 + 29 * i for i in range(expected)]:
        raise ValueError(f"inference schedule incomplete: {steps}, expected {expected} calls")
    if meta["MEASUREMENT"] == "profile":
        cube = run / "scorep" / "profile.cubex"
        if not cube.is_file() or not cube.stat().st_size:
            raise ValueError("missing Score-P profile.cubex")
        # Verified against the artifact's real CUBE4 tar/anchor.xml format.
        with tarfile.open(cube) as archive:
            root = ET.parse(archive.extractfile("anchor.xml")).getroot()
        ranks = sorted(int(group.findtext("rank")) for group in root.findall(".//locationgroup")
                       if group.findtext("type") == "process")
        count = int(meta["SOLVER_RANKS"]) + int(meta.get("DL_RANKS", 0))
        if ranks != list(range(count)):
            raise ValueError(f"Score-P MPI ranks {ranks}, expected 0..{count - 1}")
        regions = {region.findtext("name") for region in root.findall(".//region")}
        required = {"m-AIA-Solver::run"}
        if meta["PROVIDER"] == "PHYDLL":
            required |= {"dl_torch_forward", "dl_recv", "dl_send"}
        if not required <= regions:
            raise ValueError(f"missing Score-P user regions: {required - regions}")
        if meta.get("PROFILE_SCOPE_VERSION") == "steady-v1":
            verify_steady_profile(cube, root, meta, expected)
    print(f"VERIFIED {run.name}: {expected} inference calls")


if __name__ == "__main__":
    try:
        verify(Path(sys.argv[1]))
    except Exception as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
