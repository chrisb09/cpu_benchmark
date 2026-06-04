# CPU Benchmark Analysis: perfect

### Detailed Analysis of PyTorch CPU Benchmarks ('perfect' model)

**1. Overall Performance Trends**
Execution times range from a maximum of ~12.99s (1 rank, 1 intra, 16 inter) down to ~1.44s (16 ranks, 4 intra, 16 inter). Generally, increasing the number of MPI ranks and utilizing more intra-op threads decreases the execution time significantly. Scaling from 1 rank to 16 ranks provides a substantial speedup, but the improvements experience diminishing returns past 2-4 ranks, especially when considering the memory tradeoffs.

**2. Optimal Configurations**
- **Absolute Fastest**: Run 114 (16 ranks, 4 intra, 16 inter) achieves the lowest time at **1.4386s**, utilizing 19334.02 MB of memory. Run 109 (16 ranks, 2 intra, 16 inter) is practically tied at 1.4432s.
- **Best Balance (Time vs. Memory)**: Run 44 (2 ranks, 8 intra, 16 inter) completes in **1.5398s** while consuming only **9781.00 MB** of memory. This configuration is only ~7% slower than the absolute fastest run but saves nearly 50% in memory overhead, making it highly attractive for environments where memory is constrained.

**3. Memory Scaling Analysis**
The theoretical minimum data size is ~6866 MB. Since the total input data is distributed across all ranks and each rank loads its own copy of the model, we see a clear linear increase in memory usage as the number of ranks grows:
- **1 Rank**: ~9300 MB
- **2 Ranks**: ~9780 MB (+480 MB)
- **4 Ranks**: ~10730 MB (+950 MB over 2 ranks)
- **8 Ranks**: ~12660 MB (+1930 MB over 4 ranks)
- **16 Ranks**: ~19450 MB (+6790 MB over 8 ranks)

This confirms that while the data payload is divided, the baseline overhead of initializing the PyTorch runtime and duplicating the model per MPI rank adds roughly 500-800 MB per rank. At 16 ranks, the framework and model overhead vastly exceeds the actual data size.

**4. Thread Configuration & Oversubscription Impact**
On a 16-core node, configurations with high thread counts (e.g., 16 ranks * 4 intra = 64 compute threads, or 2 ranks * 16 intra = 32 compute threads) technically oversubscribe the physical cores. Interestingly, PyTorch handles this oversubscription surprisingly well for this specific workload. The fastest times consistently appear when `inter` threads are set to 16, regardless of the number of ranks. This suggests that the model benefits heavily from concurrent operations, and the OS/thread scheduler efficiently interleaves the workload without suffering severe context-switching penalties.
