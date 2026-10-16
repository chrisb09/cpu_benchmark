#!/usr/bin/env bash
# Small postprocessing job, submitted after all diagnostic profiling jobs finish.
set -euo pipefail
cd "${PROJECT_DIR:?}"
source /hpcwork/ro092286/MMCP_2026_Artifact_Hybrid_Inference/setup_env_claix23.sh
export PYTHONDONTWRITEBYTECODE=1 MPLBACKEND=Agg
/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cuda-12/bin/python3 \
    mmcp_benchmark/phydll_variance.py --campaign "${CAMPAIGN_DIR:?}" --plots
