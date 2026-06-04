### CPU Torch Scaling "Benchmark"

- Varying ranks=parallel model instances, intra-op-threads and inter-op-threads from {1,2,4,8,16}
  - 125 combinations per model
- 16 core, 220GB memory allocations
  - devel
    - giant_mlp
      - 1.6GB model size
      - 100k inputs (a 18 fp32, 7.2MB)
      - runtime: 15h 22m 41s (includes job-restarts)
  - c23ms
    - transformer
      - 812KB model size
      - 1m inputs (a 18 fp32, 72MB)
      - runtime: 0h 50m 13s
    - mmcp_transformer
      - 699MB model size
      - 10k inputs (a 2560 fp32, 102.4MB)
      - runtime: 4h 37m 46s (includes job-restarts)
    - perfect
      - <8KB model size
      - 100m inputs (a 18 fp32, 7.2GB)
      - runtime: 0h 28m 50s
    - watercnn
      - 28KB model size
      - 10m inputs (a 18 fp32, 720MB)
      - runtime: 0h 20m 46s

  * (inputs vary to ensure inference time in the seconds-minutes range)
  * (job is designed around 1h limit on devel partition, if more than 30min into a job, stop it and schedule again)

- Parameters
  - Ranks:              Each rank runs one model instance (how AIx does it iirc.)
  - Intra-op-threads:   How many threads run one operation (torch::set_num_threads(M))
  - Inter-op-threads:   How many threads run independent ops concurrently (torch::set_num_interop_threads(N)), shouldn't help much with a strictly sequential model

- Scaling Highlights:
  - **giant:**
    - 1 rank, 1 intra, 1 inter
      - runtime: 423.08s
      - memory: 6.83 GB
    - 4 rank, 4 intra, 1 inter
      - runtime: 33.27s
      - memory: 13.25 GB
    - 16 rank, 1 intra, 1 inter
      - runtime: 31.01s
      - memory: 38.34 GB
    - 16 rank, 16 intra, 1 inter
      - runtime: 661.89s (catastrophic oversubscription)
      - memory: 38.38 GB
  - **transformer:**
    - 1 rank, 1 intra, 1 inter: 96.14s | 20.38 GB
    - 4 rank, 4 intra, 1 inter: 9.27s | 22.17 GB
    - 16 rank, 1 intra, 1 inter: 9.03s | 28.26 GB
    - 16 rank, 16 intra, 1 inter: 9.13s | 28.15 GB
  - **mmcp_transformer:**
    - 1 rank, 1 intra, 1 inter: 284.67s | 9.74 GB
    - 4 rank, 4 intra, 1 inter: 21.58s | 14.63 GB
    - 16 rank, 1 intra, 1 inter: 12.14s | 26.98 GB
    - 16 rank, 16 intra, 1 inter: 14.24s | 27.51 GB
  - **perfect:**
    - 1 rank, 1 intra, 1 inter: 12.94s | 9.31 GB
    - 4 rank, 4 intra, 1 inter: 2.41s | 10.73 GB
    - 16 rank, 1 intra, 1 inter: 1.65s | 19.44 GB
    - 16 rank, 16 intra, 1 inter: 1.73s | 19.64 GB
  - **watercnn:**
    - 1 rank, 1 intra, 1 inter: 8.01s | 13.47 GB
    - 4 rank, 4 intra, 1 inter: 1.36s | 14.96 GB
    - 16 rank, 1 intra, 1 inter: 1.26s | 21.04 GB
    - 16 rank, 16 intra, 1 inter: 1.39s | 21.04 GB

- Insights:
  - **Rank Parallelism is King (CPU):** Distributing work across MPI ranks (process-level) is significantly more efficient than using many threads per rank (operator-level). 16 ranks of 1 thread is the "gold standard" for throughput on 16-core nodes.
  - **GPU vs. CPU Scaling:** GPU (H100) provides massive speedups for compute-intensive models but is limited by PCIe/Memory bus for data-heavy models:
    - `giant`: ~15x speedup (1.97s total vs 31s CPU peak)
    - `transformer`: ~18x speedup (0.49s total vs 9s CPU peak)
    - `watercnn`: ~5.5x speedup (0.23s total vs 1.26s CPU peak)
    - `perfect`: ~1.3x speedup (1.20s total vs 1.65s CPU peak) -- Heavily bottlenecked by data transfer (~75% of time).
  - **Scripting Target Matters (_cpu vs _cuda):**
    - For `transformer` and `watercnn`, the **_cuda scripted** models were slightly faster on GPU than the _cpu versions (e.g., 0.49s vs 0.55s for transformer).
    - For `giant` and `perfect`, performance was nearly identical between variants, indicating compute or transfer bottlenecks dominate.
  - **The "Oversubscription Cliff" (CPU):** For heavy models like `giant`, exceeding physical cores causes a >20x performance drop.
  - **Memory/Throughput Trade-off (CPU):** 4-8 ranks with multiple threads often hits the performance "sweet spot" while significantly reducing the memory duplication tax.

- GPU Results (NVIDIA H100):
  (Values from `gpu_benchmark_output.txt`. Total Time = Transfer In + Inference + Transfer Out)
  - **giant** (100k): 1.97s (Xfer In: 0.0012s, Inf: 1.97s, Xfer Out: 0.0004s)
  - **transformer** (1m): 0.49s (Xfer In: 0.0074s, Inf: 0.48s, Xfer Out: 0.0016s)
  - **mmcp_transformer** (10k): 3.99s (Xfer In: 0.0129s, Inf: 3.97s, Xfer Out: 0.0001s)
  - **perfect** (100m): 1.20s (Xfer In: 0.70s, Inf: 0.31s, Xfer Out: 0.20s) -- Transfer is ~75% of total time.
  - **watercnn** (10m): 0.23s (Xfer In: 0.12s, Inf: 0.08s, Xfer Out: 0.03s)
