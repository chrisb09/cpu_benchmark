# SmartSim CPU-ML-Interface: Consolidated Benchmark Analysis

This document summarizes the performance characteristics and scaling strategies for running PyTorch models on 16-core CPU nodes, based on benchmarks of five distinct models (`giant`, `perfect`, `transformer`, `transformer_mmcp`, and `watercnn`).

## Executive Summary: The "Golden Rule"
For a node with **N physical cores** (e.g., 16), the highest throughput is achieved when:
**`Total Threads (Ranks * Intra-op Threads) == N`**

While 16 independent processes (Ranks) usually provide the absolute peak throughput, a hybrid approach of Ranks and Intra-op threads is often necessary to balance memory constraints.

---

## 1. Key Findings

### A. Rank Scaling is the "Performance King"
*   **The Trend:** Process-level parallelism (MPI Ranks) consistently outperforms thread-level parallelism (Intra-op threads). 
*   **The Result:** Running 16 ranks with 1 thread each was up to **3x faster** than running 1 rank with 16 threads for complex models like the Transformer.
*   **Why?** It avoids the overhead of PyTorch's internal thread synchronization and global interpreter/resource locks, allowing each core to operate on independent data streams.

### B. The Memory "Duplication Tax"
*   **The Cost:** Memory usage scales linearly with the number of MPI ranks.
*   **The Tradeoff:** Each additional rank adds a baseline overhead (loading the model + framework initialization).
    *   *Example:* In the `perfect` model, 16 ranks consumed **~19.4 GB**, whereas 2 ranks used only **~9.8 GB** for nearly identical performance.
*   **Strategy:** If you are memory-constrained, reduce ranks and increase intra-op threads to maintain core saturation.

### C. The "Oversubscription Cliff"
*   **The Danger:** For compute-heavy models (`giant`, `transformer_mmcp`), exceeding the physical core count (`Ranks * Intra > 16`) triggers a catastrophic performance drop.
*   **The Impact:** Execution times can skyrocket from **30 seconds to 1100+ seconds** (a 30x-50x penalty) due to extreme thread contention and context switching.
*   **Recommendation:** Never exceed the physical core count unless the model is extremely "light" or latency-bound.

### D. Inter-op Threads: Negligible Impact
*   **The Finding:** Adjusting `inter_op_threads` (for parallelizing independent ops in the graph) showed **no measurable benefit** for most models. 
*   **Default:** Keep this set to 1 or a low constant.

---

## 2. Recommended Deployment Strategy

| Priority | Goal | Configuration | When to use |
| :--- | :--- | :--- | :--- |
| **1. Peak Throughput** | Maximize IO/Inference | **16 Ranks, 1 Intra** | Memory is abundant; high throughput is required. |
| **2. Balanced / Safe** | Performance + RAM | **4 Ranks, 4 Intra** or **8 Ranks, 2 Intra** | Standard production workloads; prevents OOMs. |
| **3. Memory Saving** | Minimum RAM usage | **1-2 Ranks, 8-16 Intra** | Large models on nodes with low memory-per-core. |

## 3. Configuration Cheat Sheet
When initializing the `AIxeleratorService` or setting PyTorch environment variables:

```bash
# Optimal for 16-core node
export OMP_NUM_THREADS=1
export TORCH_NUM_THREADS=1 # (Maps to Intra-op)
export TORCH_NUM_INTEROP_THREADS=1

# Use MPI to launch 16 ranks
mpirun -n 16 ./your_executable
```

---
*Analysis generated on 2026-06-03 based on empirical benchmark data.*
