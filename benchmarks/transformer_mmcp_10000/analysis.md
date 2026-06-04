# CPU Benchmark Analysis: transformer [mmcp]

### Performance Analysis: Transformer Model (mmcp) on 16-Core CPU

#### 1. Optimal Configuration
The benchmark results identify a clear winner for throughput. The most efficient configuration is **16 Ranks, 1 Intra-op Thread, and 16 Inter-op Threads**, achieving the total workload in **12.07 seconds**.

*   **Ranks over Threads**: There is a significant performance advantage to increasing the number of MPI ranks (data parallelism) rather than increasing the number of threads per rank (operator parallelism). Comparing configurations using all 16 cores:
    *   **1 Rank, 16 Threads**: 36.80 seconds
    *   **16 Ranks, 1 Thread**: 12.14 seconds
    *   **Conclusion**: Running 16 independent processes is **~3x faster** than running one process with 16 threads for this model.

#### 2. Scaling & Efficiency
*   **Speedup**: The transition from 1 Rank (284.6s) to 16 Ranks (12.1s) represents a **23.5x speedup** on 16 physical cores. This "super-linear" appearance suggests that the single-threaded baseline or the multi-threaded operator parallelism is highly inefficient for this specific Transformer implementation (possibly due to cache thrashing or synchronization overhead in the attention layers).
*   **Inter-op Parallelism**: The `inter` parameter (inter-op parallelism) had almost no impact on performance across all tested configurations. The difference between `inter=1` and `inter=16` at 16 ranks was less than 0.1 seconds.

#### 3. Oversubscription Performance Cliff
A massive performance degradation occurs when the total thread count (`ranks * intra`) exceeds the physical core count (16), specifically with medium `intra` values:
*   **The "Dead Zone"**: Configurations with `intra=4` or `intra=8` on 8 or 16 ranks saw execution times skyrocket to **186s – 513s** (up to **42x slower** than the optimal 12s).
*   **Observation**: Interestingly, `intra=16` at 16 ranks (256 total threads) was actually *faster* (14.2s) than `intra=4` at 16 ranks (316s), suggesting a specific pathological interaction between PyTorch's thread pool management and the OS scheduler at medium thread counts for this model.

#### 4. Memory Scaling (Max RSS)
*   **Baseline**: A single rank consumes ~9.7 GB of RSS.
*   **Scaling**: Memory consumption scales with the number of ranks but not perfectly linearly. 
    *   16 Ranks (1 thread) consumes **26.98 GB** total. 
    *   While each rank loads a 698 MB model, the total footprint is significantly higher than the sum of model sizes, likely due to shared libraries, internal PyTorch buffers, and the 'mmcp' schema overhead.
