import os
import subprocess
import time

BENCH_DIR = "/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench"
OUT_DIR = os.path.join(BENCH_DIR, "results_c23mm")
SCRIPTS_DIR = os.path.join(OUT_DIR, "scripts")
LOGS_DIR = os.path.join(OUT_DIR, "logs")

os.makedirs(SCRIPTS_DIR, exist_ok=True)
os.makedirs(LOGS_DIR, exist_ok=True)

# First compile the benchmark solver
print("Compiling benchmark solver...")
build_cmd = f"cd {BENCH_DIR}/build && cmake .. -DSMARTSIM_PYTHON=/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/python/smartsim_cpu/bin/python -DTorch_DIR=/hpcwork/ro092286/smartsim/CPP-ML-Interface/extern/libtorch/share/cmake/Torch && make -j"
subprocess.run(build_cmd, shell=True, check=True)

configs = [
    # Multi-rank configs (ntasks=96, cpus-per-task=1)
    {"name": "AIX", "type": "multi", "tpq": "N/A", "intra": "N/A", "bind": "N/A", "ss_args": ""},
    {"name": "SS_DEFAULT", "type": "multi", "tpq": "N/A", "intra": "N/A", "bind": "-1", "ss_args": "--use-default-cpu-settings"},
    {"name": "SS_TPQ1_I1_B96", "type": "multi", "tpq": "1", "intra": "1", "bind": "96", "ss_args": "--cpu-cores-per-node 96"},
    {"name": "SS_TPQ1_I1_NOBIND", "type": "multi", "tpq": "1", "intra": "1", "bind": "0", "ss_args": "--no-cpu-bind"},
    {"name": "SS_TPQ2_I48_B96", "type": "multi", "tpq": "2", "intra": "48", "bind": "96", "ss_args": "--threads-per-queue 2 --intra-op-threads 48 --cpu-cores-per-node 96"},
    {"name": "SS_TPQ4_I24_B96", "type": "multi", "tpq": "4", "intra": "24", "bind": "96", "ss_args": "--threads-per-queue 4 --intra-op-threads 24 --cpu-cores-per-node 96"},
    {"name": "SS_TPQ12_I8_B96", "type": "multi", "tpq": "12", "intra": "8", "bind": "96", "ss_args": "--threads-per-queue 12 --intra-op-threads 8 --cpu-cores-per-node 96"},
    {"name": "SS_TPQ24_I4_B96", "type": "multi", "tpq": "24", "intra": "4", "bind": "96", "ss_args": "--threads-per-queue 24 --intra-op-threads 4 --cpu-cores-per-node 96"},
    {"name": "SS_TPQ48_I2_B96", "type": "multi", "tpq": "48", "intra": "2", "bind": "96", "ss_args": "--threads-per-queue 48 --intra-op-threads 2 --cpu-cores-per-node 96"},
    {"name": "SS_TPQ96_I1_NOBIND", "type": "multi", "tpq": "96", "intra": "1", "bind": "0", "ss_args": "--threads-per-queue 96 --intra-op-threads 1 --no-cpu-bind"},
    {"name": "SS_TPQ96_I1_B96", "type": "multi", "tpq": "96", "intra": "1", "bind": "96", "ss_args": "--threads-per-queue 96 --intra-op-threads 1 --cpu-cores-per-node 96"},
    
    # Single-rank configs (ntasks=1, cpus-per-task=96)
    {"name": "SINGLE_SS_DEFAULT", "type": "single", "tpq": "N/A", "intra": "N/A", "bind": "-1", "ss_args": "--use-default-cpu-settings"},
    {"name": "SINGLE_SS_TPQ1_I96_NOBIND", "type": "single", "tpq": "1", "intra": "96", "bind": "0", "ss_args": "--no-cpu-bind"},
    {"name": "SINGLE_SS_TPQ1_I96_B96", "type": "single", "tpq": "1", "intra": "96", "bind": "96", "ss_args": "--cpu-cores-per-node 96"},
    {"name": "SINGLE_SS_TPQ48_I2_B96", "type": "single", "tpq": "48", "intra": "2", "bind": "96", "ss_args": "--threads-per-queue 48 --intra-op-threads 2 --cpu-cores-per-node 96"},
    {"name": "SINGLE_SS_TPQ96_I1_B96", "type": "single", "tpq": "96", "intra": "1", "bind": "96", "ss_args": "--threads-per-queue 96 --intra-op-threads 1 --cpu-cores-per-node 96"},
]

template = """#!/bin/bash
#SBATCH --job-name={name}
#SBATCH --nodes=1
#SBATCH --ntasks={ntasks}
#SBATCH --cpus-per-task={cpus}
#SBATCH --output={out_dir}/logs/{name}_%j.log
#SBATCH --time=05:00:00
#SBATCH --partition=c23mm
#SBATCH --exclusive
#SBATCH --account=thes2181

set -e
BENCH_DIR="/hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench"
BASE_DIR="/hpcwork/ro092286/smartsim"
PYTHON_RUNTIME_ROOT="${{BASE_DIR}}/CPP-ML-Interface/extern/python"
RUNTIME_DEVICE="smartsim_cpu"
SMARTSIM_PYTHON="${{PYTHON_RUNTIME_ROOT}}/${{RUNTIME_DEVICE}}/bin/python"

cd "${{BASE_DIR}}/CPP-ML-Interface" && source ./install.sh cpu

RUNTIME_EXTRA_LIB_DIR="${{PYTHON_RUNTIME_ROOT}}/${{RUNTIME_DEVICE}}/runtime_libs"
if [ -d "${{RUNTIME_EXTRA_LIB_DIR}}" ]; then
    export LD_LIBRARY_PATH="${{RUNTIME_EXTRA_LIB_DIR}}:${{LD_LIBRARY_PATH:-}}"
fi
PHYDLL_LIB_DIR="${{BASE_DIR}}/CPP-ML-Interface/extern/phydll/build/lib"
export LD_LIBRARY_PATH="${{PHYDLL_LIB_DIR}}:${{LD_LIBRARY_PATH:-}}"

export MLCOUPLING_LOG_LEVEL=DEBUG
export SR_MODEL_TIMEOUT=2000000
export SR_CMD_TIMEOUT=2000000
export SR_SOCKET_TIMEOUT=2000000

cd $BENCH_DIR/build

RESULTS_CSV="{out_dir}/{name}.csv"
echo "label,model,provider,tpq,intra_threads,bind_cores,time_s,max_rss_mb,status" > "$RESULTS_CSV"

PNAME="{name}"
TPQ="{tpq}"
INTRA="{intra}"
BIND="{bind}"

ENDPOINT_FILE="{out_dir}/.ssdb_endpoint_${{PNAME}}"
DONE_FILE="{out_dir}/.solver_done_${{PNAME}}"

if [ "$PNAME" != "AIX" ]; then
    rm -f "${{ENDPOINT_FILE}}" "${{DONE_FILE}}"
    ${{SMARTSIM_PYTHON}} "${{BASE_DIR}}/CPP-ML-Interface/dl_clients/smartsim_controller.py" \\
        --auto-port \\
        --endpoint-file "${{ENDPOINT_FILE}}" \\
        --done-file "${{DONE_FILE}}" \\
        --exp-dir "{out_dir}/smartsim_experiments_${{PNAME}}" \\
        --silent \\
        {ss_args} &
    DRIVER_PID=$!
    
    for _ in $(seq 1 120); do
        if [ -s "${{ENDPOINT_FILE}}" ]; then break; fi
        sleep 0.5
    done
    if [ ! -s "${{ENDPOINT_FILE}}" ]; then
        echo "Timed out waiting for SmartSim DB"
        kill $DRIVER_PID 2>/dev/null || true
        exit 1
    fi
    export SSDB="$(tr -d '\\n' < "${{ENDPOINT_FILE}}")"
fi

MODELS=(
    "giant|100000|mini_app|${{BASE_DIR}}/mini_app/train_models/model_a/giant_cpu.pt"
    "mmcp_transformer|10000|mmcp|${{BASE_DIR}}/MMCP_TOM/input/transformer_inference_scripted_fw2.pt"
    "transformer|1000000|mini_app|${{BASE_DIR}}/mini_app/train_models/model_a/transformer_cpu.pt"
    "perfect|100000000|mini_app|${{BASE_DIR}}/mini_app/train_models/model_a/perfect_cpu.pt"
    "watercnn|10000000|mini_app|${{BASE_DIR}}/mini_app/train_models/model_a/watercnn_cpu.pt"
)

for m_data in "${{MODELS[@]}}"; do
    MODEL_NAME=$(echo "$m_data" | cut -d'|' -f1)
    INPUTS=$(echo "$m_data" | cut -d'|' -f2)
    SCHEMA=$(echo "$m_data" | cut -d'|' -f3)
    MODEL_PATH=$(echo "$m_data" | cut -d'|' -f4)
    LABEL="${{MODEL_NAME}}_${{PNAME}}"
    
    for run in {{1..10}}; do
        echo "=========================================================="
        echo "Model: $MODEL_NAME | Provider: $PNAME | Run: $run/10"
        
        OUTPUT_FILE="{out_dir}/logs/${{MODEL_NAME}}_${{PNAME}}_run${{run}}.log"
        
        set +e
        if [ "$PNAME" = "AIX" ]; then
            mpirun -n {ntasks} ./benchmark_solver \\
                --provider AIX --model "$MODEL_PATH" --schema "$SCHEMA" --inputs "$INPUTS" \\
                > "$OUTPUT_FILE" 2>&1
            RC=$?
        else
            mpirun -n {ntasks} ./benchmark_solver \\
                --provider SMARTSIM --model "$MODEL_PATH" --schema "$SCHEMA" --inputs "$INPUTS" \\
                > "$OUTPUT_FILE" 2>&1
            RC=$?
        fi
        set -e
        
        if [ $RC -eq 0 ]; then
            RES_LINE=$(grep "^RESULT:" "$OUTPUT_FILE" | tail -n 1 | cut -d':' -f2)
            if [ -n "$RES_LINE" ]; then
                T_S=$(echo "$RES_LINE" | cut -d',' -f1)
                M_MB=$(echo "$RES_LINE" | cut -d',' -f2)
                echo "$LABEL,$MODEL_NAME,$PNAME,$TPQ,$INTRA,$BIND,$T_S,$M_MB,SUCCESS" >> "$RESULTS_CSV"
            else
                echo "$LABEL,$MODEL_NAME,$PNAME,$TPQ,$INTRA,$BIND,-1,-1,FAILED_PARSE" >> "$RESULTS_CSV"
            fi
        else
            echo "$LABEL,$MODEL_NAME,$PNAME,$TPQ,$INTRA,$BIND,-1,-1,FAILED_RC_${{RC}}" >> "$RESULTS_CSV"
        fi
    done
done

if [ "$PNAME" != "AIX" ]; then
    touch "${{DONE_FILE}}"
    wait "${{DRIVER_PID}}" || true
fi
echo "Benchmark completed successfully."
"""

print(f"Submitting {len(configs)} benchmarking jobs...")

for config in configs:
    ntasks = 96 if config["type"] == "multi" else 1
    cpus = 1 if config["type"] == "multi" else 96
    
    script_content = template.format(
        name=config["name"],
        ntasks=ntasks,
        cpus=cpus,
        out_dir=OUT_DIR,
        tpq=config["tpq"],
        intra=config["intra"],
        bind=config["bind"],
        ss_args=config["ss_args"]
    )
    
    script_path = os.path.join(SCRIPTS_DIR, f"{config['name']}.sh")
    with open(script_path, "w") as f:
        f.write(script_content)
    
    # Submit job
    res = subprocess.run(["sbatch", script_path], capture_output=True, text=True, check=True)
    out = res.stdout.strip()
    print(f"Submitted {config['name']}: {out}")
    
    # Parse job id to run swatch
    # Output is typically "Submitted batch job 123456"
    if "Submitted batch job" in out:
        job_id = out.split()[-1]
        subprocess.Popen(f"/home/ro092286/scripts/swatch {job_id} > /dev/null 2>&1 &", shell=True)
    
    time.sleep(0.5)

print("All jobs submitted and swatch monitors attached!")
