#!/usr/bin/env python3
"""Render the m-aia Phase-1 characterization: warm ML-step latency vs rank count.

Consumes the per-step CSV written by ``parse_ml_step_timing.py`` and produces a
boxplot of warm ``ml_ms`` per run, plus a mean-vs-ranks line if rank metadata is
present. Figures go through the shared ``hh3_render`` hook (``--pgf`` for the
thesis style).
"""
from __future__ import annotations

import argparse
import csv
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "mini_app"))
from hh3_render import close_figure, save_figure  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402


def read_rows(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timings", type=Path, required=True, help="per-step CSV")
    parser.add_argument("--output", type=Path, required=True, help="figure path (.png/.pdf)")
    parser.add_argument("--title", default="m-aia warm ML-step latency (AIX)")
    args = parser.parse_args()

    rows = [r for r in read_rows(args.timings) if r.get("warm", "").lower() == "true"]
    if not rows:
        print("no warm rows", file=sys.stderr)
        return 1

    groups: dict[str, list[float]] = defaultdict(list)
    ranks_of: dict[str, str] = {}
    for r in rows:
        groups[r["run"]].append(float(r["ml_ms"]))
        ranks_of[r["run"]] = r.get("ranks", "")

    labels = sorted(groups, key=lambda k: int(ranks_of.get(k) or 0))
    data = [groups[k] for k in labels]
    xlabels = [f"{ranks_of.get(k, '?')}" for k in labels]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.boxplot(data, showmeans=True)
    ax.set_xticks(range(1, len(xlabels) + 1))
    ax.set_xticklabels(xlabels)
    for i, vals in enumerate(data, start=1):
        ax.scatter([i] * len(vals), vals, s=10, alpha=0.5, zorder=3)
    ax.set_xlabel("solver ranks")
    ax.set_ylabel("warm ML-step time (ms)")
    ax.set_title(args.title)
    fig.tight_layout()
    save_figure(fig, args.output)
    close_figure(fig)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
