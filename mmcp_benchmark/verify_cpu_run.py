#!/usr/bin/env python3
"""Fail incomplete retest cells; verify Score-P's process topology and user regions."""
import re
import sys
import tarfile
import xml.etree.ElementTree as ET
from pathlib import Path


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
    print(f"VERIFIED {run.name}: {expected} inference calls")


if __name__ == "__main__":
    try:
        verify(Path(sys.argv[1]))
    except Exception as exc:
        print(f"VERIFICATION FAILED: {exc}", file=sys.stderr)
        sys.exit(1)
