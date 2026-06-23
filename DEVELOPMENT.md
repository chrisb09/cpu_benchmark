# CPU Benchmarking & Model Multiplexing Development Guide

This repository contains benchmarks and scaling analysis of different Machine Learning coupling providers (AIxelerator, PhyDLL, and SmartSim) on CPU architectures, run on the RWTH Aachen CLAIX cluster (partition `c23mm`, featuring a dual-socket Intel Sapphire Rapids node with 96 cores and 512 GB of RAM).

---

## 1. Benchmarking Suite Structure

* **`provider_bench/benchmark_solver.cpp`**: The C++ executable that initializes the specified ML coupling provider, loads a PyTorch model, and executes the inference loop using dummy input datasets.
* **`provider_bench/submit_benchmarks.py`**: A python script that automates the generation and submission of 10-run SLURM job scripts for each configuration swept on the `c23mm` partition.
* **`provider_bench/results_c23mm/`**: Holds the output results including:
  * Individual configuration CSVs (`*.csv`) tracking execution time, Max RSS memory, and status.
  * Run logs (`logs/*.log`) of the 10 consecutive executions per model/provider configuration.
  * Generated SLURM batch scripts (`scripts/*.sh`).
* **`provider_bench/generate_final_analysis.py`**: Reads all configuration CSVs from `results_c23mm/` and outputs a comprehensive Jupyter Notebook analyzing median inference time and memory usage.
* **`provider_bench/final_c23mm_analysis.ipynb`**: Executed Jupyter Notebook featuring annotated bar plots and box plots with 95% Confidence Interval (CI) whiskers.

---

## 2. Configuration Matrix

The benchmark sweeps 17 distinct configurations (each run 10 times across 5 model sizes):
1. **`AIX`**: Baseline CPU inference using the AIxelerator framework.
2. **`SS_DEFAULT`**: SmartSim default configuration (no thread/binding constraints).
3. **`SS_B96` / `SS_NOBIND`**: SmartSim with default threads, either bound to all 96 cores or completely unbound.
4. **Thread Sweep (`SS_TPQ=N_I=M_B=96`)**: Sweeps the Threads Per Queue (TPQ) and Intra-Op threads (I) under a fixed 96-core bind constraint (e.g. TPQ=12, Intra=8).
5. **Single-Rank (`SINGLE_SS_*`)**: Run configurations using a single solver rank with 96 CPUs-per-task to measure single-client database capabilities.
6. **Multiplexed (`SS_TPQ96_I1_B96_MULTI`)**: *Added in this session*. Runs 96 MPI ranks, where each rank loads and executes its own model key (`benchmark_model_0` to `benchmark_model_95`) inside the database rather than sharing a single global key.

---

## 3. Session Implementation Details & Code Changes

### A. Code Modifications
To enable rank-level model multiplexing without breaking other configurations, the following changes were made:
1. **SmartSim Provider ([ml_coupling_provider_smartsim.hpp](file:///hpcwork/ro092286/smartsim/CPP-ML-Interface/include/provider/ml_coupling_provider_smartsim.hpp))**:
   * Introduced support for the `MLCOUPLING_MULTI_MODEL` environment variable.
   * If `MLCOUPLING_MULTI_MODEL` is set, the condition `if (this->rank == 0)` is expanded to allow all ranks to execute `client->set_model_from_file` and load their own model keys into the database concurrently.
2. **Benchmark Solver ([benchmark_solver.cpp](file:///hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/benchmark_solver.cpp))**:
   * Checks for `MLCOUPLING_MULTI_MODEL`. If active, it appends the suffix `_world_rank` to the model name (resulting in keys `benchmark_model_0` through `benchmark_model_95`).

### B. Database Isolation in Batch Script
Due to physical memory limitations on the node, loading 96 instances of the 1.6 GB `giant` model triggers OOM kills, crashing the database and failing the entire job.
* To prevent failure propagation, [SS_TPQ96_I1_B96_MULTI.sh](file:///hpcwork/ro092286/smartsim/cpu_benchmark/provider_bench/results_c23mm/scripts/SS_TPQ96_I1_B96_MULTI.sh) was structured to start a fresh database controller instance at the beginning of each model, run the 10 inferences, and then tear the database down. This isolates failures and allows the remaining models to complete successfully.

### C. Notebook Execution
The notebook `final_c23mm_analysis.ipynb` is regenerated and executed using the dedicated `analysis` virtual environment:
```bash
/hpcwork/ro092286/analysis/bin/python -m jupyter nbconvert --to notebook --execute --inplace final_c23mm_analysis.ipynb
```
*(Note: `nbconvert` and `nbformat` were installed in the `analysis` environment to support programmatic in-place compilation).*

---

## 4. Key Findings

* **Multiplexing Slowdown (Cache Thrashing)**: For the 699 MB `mmcp_transformer` model, model multiplexing took **~128.5 seconds** per inference compared to **~19.0 seconds** in the single shared model key configuration. When ranks share a single read-only key, the model weights reside compactly in the CPU's shared L3/L2 caches. With 96 multiplexed keys, the active weight set grows to **~67 GB** in memory, saturating the DRAM memory bus and causing continuous cache thrashing.
* **Small Model Overhead**: For models under 1 MB, the cache thrashing is avoided, but multiplexing remains slightly slower than single shared key due to Redis socket/lookup overhead.

---

## 5. `MLCOUPLING_MULTI_MODEL` Environment Variable

The benchmark solver and the C++ `MLCouplingProviderSmartsim` both check the
`MLCOUPLING_MULTI_MODEL` environment variable. When set:

* The benchmark solver appends `_world_rank` to the model name
  (`benchmark_model_0`..`benchmark_model_95`), so each rank uploads its
  own model under a unique Redis key.
* The SmartSim provider expands the `if (rank == 0)` upload guard to
  `if (rank == 0 || is_multi)`, allowing every rank to execute
  `set_model_from_file`.

**This is a benchmarking / research-only knob, not a production feature.**
For the 1.6 GB `giant` model, 96 multiplexed keys require ~153 GB of disk
space and even more RAM after TorchScript loading — large enough to OOM
the 512 GB `c23mm` node (the orchestrator process is killed with
`exit code 137` by the Linux OOM killer). For the 699 MB `mmcp_transformer`
model it survives but is ~6.8× slower than the single-shared-key
configuration due to L3 cache thrashing. For models under a few MB the
overhead is small but the configuration still has no production
benefit.

Recommended use: only as a stress test for the SmartSim database's
per-key upload path. In any real coupling scenario prefer the default
single-shared-key upload (rank 0 only) and let the database multiplex
inference.
