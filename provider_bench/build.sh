#!/bin/bash


#get script directory
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Running provider benchmark with script from ${SCRIPT_DIR}"

BUILD_DIR="${SCRIPT_DIR}/build"

mkdir -p ${BUILD_DIR}

cd ${BUILD_DIR}

cmake .. -DCMAKE_BUILD_TYPE=Release

NUM_CORES=$(nproc)

cmake --build . -j ${NUM_CORES}