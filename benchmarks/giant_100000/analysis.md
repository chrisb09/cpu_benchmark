# CPU Benchmark Analysis: giant

### Performance and Optimal Configuration
The benchmark results clearly demonstrate that performance is optimal when the total number of active compute threads (`ranks` * `intra`) perfectly matches the 16 hardware cores of the node. The absolute fastest execution was Run 101, which utilized 16 ranks and 1 intra-op thread, finishing in 30.37 seconds. Other configurations matching the 16-core threshold performed similarly well:
- 1 rank, 16 intra: ~34.0s
- 2 ranks, 8 intra: ~34.2s
- 4 ranks, 4 intra: ~33.3s
- 8 ranks, 2 intra: ~33.3s
- 16 ranks, 1 intra: ~30.4s
Running 16 independent MPI ranks with 1 thread each slightly edges out the others, likely by avoiding the overhead of PyTorch's internal thread synchronization.

### Thread Oversubscription Impact
Oversubscribing the CPU cores (`ranks` * `intra` > 16) causes severe performance degradation. For example, using 4 ranks with 8 intra-op threads (32 active threads) resulted in execution times skyrocketing from ~33s to over 1100s, and triggered multiple TIMEOUTs. This highlights a massive penalty from thread contention and context-switching when PyTorch threads compete for physical cores.

### Memory Scaling
Peak memory consumption (`max_rss_mb`) scales linearly with the number of MPI ranks. This aligns with the context that each rank loads its own copy of the model:
- 1 Rank: ~6.8 GB
- 2 Ranks: ~9.0 GB
- 4 Ranks: ~13.2 GB
- 8 Ranks: ~21.6 GB
- 16 Ranks: ~38.4 GB
Each additional rank adds approximately 2.1 to 2.2 GB of memory overhead, which logically accounts for the 1.6 GB model weights plus PyTorch runtime overhead and the partitioned data tensors.

### Impact of Inter-op Threads
Modifying the number of inter-op threads (`inter`) between 1 and 16 showed no significant impact on execution time or memory usage across the board. The model's execution bottleneck is strictly tied to intra-op parallelism and the overall distribution of MPI ranks.
