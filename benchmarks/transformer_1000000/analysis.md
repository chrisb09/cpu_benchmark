# CPU Benchmark Analysis: transformer

### Performance Analysis: Transformer Model CPU Benchmark

#### 1. Optimal Configuration
The peak performance was achieved with **8 ranks, 4 intra-op threads, and 4 inter-op threads** (Run 87), resulting in an execution time of **8.84s**. This represents a **10.87x speedup** over the baseline single-rank, single-thread configuration (96.14s). 

However, a highly efficient and more stable configuration is **16 ranks with 1 intra-op thread** (Run 100), which achieved **9.03s**. Given the 16-core architecture, this configuration utilizes 1 rank per core without oversubscription, offering a cleaner scaling path.

#### 2. Scaling & Threading Efficiency
*   **Intra-op Threading (torch.set_num_threads):** On a single rank, scaling intra-op threads from 1 to 16 yielded significant gains up to 8 threads (6.46x speedup), but plateaued thereafter (Run 20: 11.04s). 
*   **Process-level Parallelism (MPI Ranks):** Distributing the workload across MPI ranks is more effective than thread-level parallelism alone. Comparing configurations using all 16 cores: 
    *   1 Rank x 16 Threads: 11.04s
    *   16 Ranks x 1 Thread: 9.03s (~18% faster)
*   **Inter-op Threading:** Varying `inter_op_threads` had negligible impact on performance across all rank counts, suggesting the transformer model's graph has few independent operators that PyTorch can parallelize at the inter-op level.

#### 3. Memory Scaling (RSS)
*   **Base Overhead:** A single rank starts with a high baseline memory footprint of **~20.4 GB**. This suggests a large fixed overhead from the dataset or the PyTorch environment initialization for this specific workload.
*   **Incremental Cost:** Each additional MPI rank adds approximately **500–530 MB** to the total RSS. 
*   **Total Usage:** Memory usage scaled from 20.4 GB (1 rank) to **28.2 GB (16 ranks)**. While the base is high, the incremental scaling is relatively efficient, allowing for high rank counts on standard HPC nodes.

#### 4. Oversubscription Impact
Running `8 ranks x 4 threads` (32 total threads) on a 16-core node provided a slight performance boost (8.84s) over the non-oversubscribed `16x1` (9.03s). This indicates that the transformer model's compute intensity or memory access patterns allow for some hyperthreading/context-switching gains without significant contention penalties.
