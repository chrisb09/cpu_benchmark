# CPU Benchmark Analysis: watercnn

### Detailed Analysis of WaterCNN CPU Benchmarks

#### 1. Optimal Configuration and Performance
The benchmark results identify two primary 'sweet spots' for performance on the 16-core node:
- **Low Rank/High Intra**: Run 46 achieved the absolute lowest time of **1.2392s** using **2 ranks and 16 intra-op threads**. This configuration utilizes oversubscription (32 threads on 16 cores), suggesting the workload benefits from Hyper-Threading or masks latency effectively.
- **High Rank/Low Intra**: Run 100 followed closely at **1.2631s** using **16 ranks and 1 intra-op thread**. This is the most efficient configuration for maximizing physical core utilization without thread contention within a single process.

#### 2. Throughput and Scaling Efficiency
- **Rank Scaling**: Moving from 1 rank to 16 ranks (with 1 intra thread each) reduced execution time from 8.01s to 1.26s, a **6.35x speedup**. While significant, this represents a parallel efficiency of ~40%, indicating overhead in process management or data distribution.
- **Intra-op Scaling**: Within a single rank, increasing threads from 1 to 16 improved performance from 8.01s to 1.40s. However, multi-rank setups consistently outperformed single-rank setups with the same total thread count (e.g., 16x1 was ~10% faster than 1x16).

#### 3. Memory Scaling and Overhead
- **Baseline Memory**: The results show a surprisingly high baseline memory footprint of **~13.5 GB** for a single rank, despite the model and theoretical data size being significantly smaller (~700 MB total). This suggests significant environment overhead or data replication in the benchmark harness.
- **Incremental Cost**: Memory usage scales linearly with ranks, adding approximately **450–500 MB per rank**. The peak RSS at 16 ranks reached **21.04 GB**, which is a ~56% increase over the single-rank baseline.

#### 4. Impact of Threading Parameters
- **Intra-op Threads**: Crucial for performance. The model shows strong scaling up to 4–8 threads, with diminishing returns thereafter.
- **Inter-op Threads**: This parameter had **negligible impact** across all tests. For example, at 16 ranks, varying `inter` from 1 to 16 resulted in a variance of less than 0.01s. This indicates the model has a shallow or highly sequential execution graph.
