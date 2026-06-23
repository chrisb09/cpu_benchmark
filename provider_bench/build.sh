#!/bin/bash


#get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Running provider benchmark with script from ${SCRIPT_DIR}"

BUILD_DIR="${SCRIPT_DIR}/build"

mkdir -p ${BUILD_DIR}

cd ${BUILD_DIR}

cmake .. -DCMAKE_BUILD_TYPE=Release

build_jobs="${SLURM_CPUS_ON_NODE:-$(nproc)}"
echo "Building with -j${build_jobs} parallel jobs..."
cmake --build . -j ${build_jobs}