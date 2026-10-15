#!/usr/bin/env python3
"""Extract two-warmup steady inference phases from m-aia Score-P profiles.

Unlike the earlier whole-run scope, this selects only the subtree beneath the
instrumented steady parents ``cmi_inference_steady`` (solver ranks) and
``dl_inference_steady`` (PhyDLL DL ranks). Those parents are entered only after
two warmup inference calls, so five measured calls remain in production and one
in the short smoke. The denominator is derived from the parent's visit count,
never assumed.

The parser is the dependency-free ``cube_dump`` reader in ``verify_cpu_run`` so
the extraction needs Score-P but not pandas/numpy.
"""
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path

HERE = Path(__file__).resolve().parent
SOLVER_PARENT = "cmi_inference_steady"
DL_PARENT = "dl_inference_steady"
SOLVER_SCOPE = "solver_steady"
DL_SCOPE = "dl_steady"
REGION_FIELDS = ("workload", "variant", "job", "cubex", "scope", "rank_count", "cnode",
                 "region", "inclusive_rankavg_s", "exclusive_rankavg_s", "visits_rankavg",
                 "inclusive_ms_per_visit", "exclusive_ms_per_visit")
RANK_FIELDS = ("workload", "variant", "job", "cubex", "scope", "rank", "cnode", "parent",
               "region", "inclusive_s", "exclusive_s", "visits")


def _verify_module():
    spec = importlib.util.spec_from_file_location("steady_verify", HERE / "verify_cpu_run.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def location_ranks(root):
    """Map Score-P location IDs to MPI ranks; require one thread per process."""
    mapping = {}
    for group in root.findall(".//locationgroup"):
        if group.findtext("type") != "process":
            continue
        locations = group.findall("location")
        if len(locations) != 1 or locations[0].findtext("type") != "thread":
            raise ValueError("steady extraction requires one thread location per MPI rank")
        loc = int(locations[0].attrib["Id"])
        if loc in mapping:
            raise ValueError("duplicate Score-P location ID")
        mapping[loc] = int(group.findtext("rank"))
    return mapping


def _descendants(tree, parent_id):
    members, stack = [], [parent_id]
    while stack:
        current = stack.pop()
        node = tree.get(current)
        if node is None:
            continue
        members.append(node)
        stack.extend(node.children_ids)
    return members


def steady_subtree(tree, parent_name):
    """Return the steady parent node and every node at or below it."""
    parents = [n for n in tree.values() if n.name == parent_name and n.visits > 0]
    if len(parents) != 1:
        raise ValueError(f"expected exactly one {parent_name} node, found {len(parents)}")
    parent = parents[0]
    return parent, _descendants(tree, parent.id)


def extract_cube(module, cube, workload, variant, job, solver_ranks, dl_ranks):
    """Return (region rows, per-rank rows) for both solver and DL steady scopes."""
    with tarfile.open(cube) as archive:
        root = ET.parse(archive.extractfile("anchor.xml")).getroot()
    loc_rank = location_ranks(root)
    count = solver_ranks + dl_ranks
    if sorted(loc_rank.values()) != list(range(count)):
        raise ValueError(f"unexpected MPI rank set in {cube}")
    trees = module.parse_profile(cube)
    if len(trees) != count or not all(trees):
        raise ValueError(f"empty or incomplete rank trees in {cube}")
    ranktrees = dict(zip((loc_rank[i] for i in sorted(loc_rank)), trees))

    regions, ranks = [], []
    specs = (("solver", SOLVER_PARENT, SOLVER_SCOPE, range(solver_ranks)),
             ("dl", DL_PARENT, DL_SCOPE, range(solver_ranks, count)))
    frames_seen = {}
    for kind, parent_name, scope, rank_set in specs:
        rank_set = list(rank_set)
        if not rank_set:
            continue
        aggregate, per_rank = {}, {}
        for rank in rank_set:
            parent, members = steady_subtree(ranktrees[rank], parent_name)
            frames_seen.setdefault(scope, parent.visits)
            if frames_seen[scope] != parent.visits:
                raise ValueError(f"inconsistent {scope} steady visit count across ranks")
            for node in members:
                if not (math.isfinite(node.incl_time) and math.isfinite(node.excl_time)
                        and node.incl_time >= 0 and node.excl_time >= 0 and node.visits >= 0):
                    raise ValueError(f"invalid metrics for {node.name} on rank {rank}")
                ranks.append(dict(workload=workload, variant=variant, job=job, cubex=str(cube),
                                  scope=scope, rank=rank, cnode=node.id, parent=node.parent_id or 0,
                                  region=node.name, inclusive_s=node.incl_time,
                                  exclusive_s=node.excl_time, visits=node.visits))
                entry = aggregate.setdefault((node.id, node.name), [0.0, 0.0, 0])
                entry[0] += node.incl_time
                entry[1] += node.excl_time
                entry[2] += node.visits
        for (cnode, name), (incl, excl, visits) in sorted(aggregate.items(), key=lambda kv: kv[0][0]):
            per_rank[(cnode, name)] = None
            regions.append(dict(workload=workload, variant=variant, job=job, cubex=str(cube),
                                scope=scope, rank_count=len(rank_set), cnode=cnode, region=name,
                                inclusive_rankavg_s=incl / len(rank_set),
                                exclusive_rankavg_s=excl / len(rank_set),
                                visits_rankavg=visits / len(rank_set),
                                inclusive_ms_per_visit=1000 * incl / visits if visits else "",
                                exclusive_ms_per_visit=1000 * excl / visits if visits else ""))
    return regions, ranks, frames_seen


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields))
        writer.writeheader()
        writer.writerows(rows)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", required=True, type=Path, help="preserved run directory")
    parser.add_argument("--workload", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--job", required=True)
    parser.add_argument("--solver-ranks", type=int, default=96)
    parser.add_argument("--dl-ranks", type=int, default=0)
    parser.add_argument("--regions", type=Path, required=True)
    parser.add_argument("--ranks-out", type=Path, required=True)
    args = parser.parse_args()
    module = _verify_module()
    regions, ranks, frames = extract_cube(module, args.run / "scorep/profile.cubex",
                                          args.workload, args.variant, args.job,
                                          args.solver_ranks, args.dl_ranks)
    write_csv(args.regions, REGION_FIELDS, regions)
    write_csv(args.ranks_out, RANK_FIELDS, ranks)
    print(f"steady frames by scope: {frames}; regions={len(regions)} rank-rows={len(ranks)}")


if __name__ == "__main__":
    main()
