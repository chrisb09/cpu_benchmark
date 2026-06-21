import argparse
import time
import resource
import os
import torch
from mpi4py import MPI

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--intra", type=int, required=True)
    parser.add_argument("--inter", type=int, required=True)
    parser.add_argument("--model", type=str, default="giant",
                        help="Model name prefix or full path")
    parser.add_argument("--num-inputs", type=int, default=100000,
                        help="TOTAL inference inputs across all ranks")
    parser.add_argument("--schema", type=str, default="mini_app",
                        choices=["mini_app", "mmcp"],
                        help="Input schema: 'mini_app' (B, 18) or 'mmcp' (5, B, 512)")
    parser.add_argument("--seq-len", type=int, default=5, help="Sequence length for mmcp schema")
    parser.add_argument("--feature-dim", type=int, default=512, help="Feature dimension for mmcp schema")
    parser.add_argument("--max-batch-size", type=int, default=0, 
                        help="Maximum batch size per inference call. 0 = auto-detect based on schema.")
    args = parser.parse_args()

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()

    torch.set_num_threads(args.intra)
    torch.set_num_interop_threads(args.inter)

    # Resolve max batch size
    max_bs = args.max_batch_size
    if max_bs <= 0:
        if args.schema == "mmcp":
            max_bs = 5000
        else:
            max_bs = 1000000 # Effectively unlimited for MLP

    # Load model
    if os.path.exists(args.model):
        model_path = args.model
    else:
        model_path = f"../mini_app/train_models/model_a/{args.model}_cpu.pt"
    
    if not os.path.exists(model_path):
        if rank == 0:
            print(f"Model not found at {model_path}, aborting.", flush=True)
        comm.Abort(1)

    try:
        model = torch.jit.load(model_path, map_location="cpu")
        model.eval()
    except Exception as e:
        if rank == 0:
            print(f"FAILED to load model: {e}")
        comm.Abort(1)

    # Calculate inputs per rank for Strong Scaling
    total_inputs = args.num_inputs
    inputs_per_rank = total_inputs // comm.Get_size()
    if rank == comm.Get_size() - 1:
        inputs_per_rank += total_inputs % comm.Get_size()

    # Create dummy inputs based on schema
    if args.schema == "mini_app":
        warmup_input = torch.randn(1, 18, dtype=torch.float32)
        batch_input = torch.randn(inputs_per_rank, 18, dtype=torch.float32)
    elif args.schema == "mmcp":
        warmup_input = torch.randn(3 * 1, args.seq_len, args.feature_dim, dtype=torch.float32)
        batch_input = torch.randn(3 * inputs_per_rank, args.seq_len, args.feature_dim, dtype=torch.float32)

    # count warmup time
    warmup_start = time.perf_counter()
    with torch.no_grad():
        _ = model(warmup_input)
    warmup_end = time.perf_counter()
    if rank == 0:
        print(f"Warmup time: {warmup_end - warmup_start:.4f} seconds", flush=True)
        
    # Canary measurement
    comm.Barrier() 
    canary_start = time.perf_counter()
    canary_size = min(max_bs, max(1, inputs_per_rank // 100))
    
    if args.schema == "mmcp":
        canary_input = batch_input[:3 * canary_size, :, :]
    else:
        canary_input = batch_input[:canary_size]

    with torch.no_grad():
        _ = model(canary_input)
        
    canary_end = time.perf_counter()
    canary_time = canary_end - canary_start
    estimated_total_time = None
    if rank == 0:
        print(f"Canary time for {canary_size} samples: {canary_time:.4f} seconds", flush=True)
        estimated_total_time = canary_time * (inputs_per_rank / canary_size)
        print(f"Estimated total time per rank: {estimated_total_time:.4f} seconds", flush=True)

    skip_benchmark = False
    if rank == 0 and estimated_total_time is not None:
        if estimated_total_time > 3600: 
            print("Estimated time exceeds 1 hour. Exiting early.", flush=True)
            skip_benchmark = True

    skip_benchmark = comm.bcast(skip_benchmark, root=0)
    if skip_benchmark:
        if rank == 0:
            print("RESULT:EXPECTED_TIMEOUT,0.00", flush=True)
        comm.Barrier()
        MPI.Finalize()
        return

    # Wait for all ranks to be ready before timing
    comm.Barrier()

    # Benchmark with batching support
    start_time = time.perf_counter()
    with torch.no_grad():
        cursor = 0
        while cursor < inputs_per_rank:
            end_idx = min(cursor + max_bs, inputs_per_rank)
            if args.schema == "mmcp":
                batch = batch_input[3 * cursor : 3 * end_idx, :, :]
            else:
                batch = batch_input[cursor:end_idx]
            
            _ = model(batch)
            cursor = end_idx
            
    end_time = time.perf_counter()
    elapsed = end_time - start_time

    # Memory usage (KB -> MB)
    mem_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    mem_mb = mem_kb / 1024.0

    # Gather results to rank 0
    all_times = comm.gather(elapsed, root=0)
    all_mems = comm.gather(mem_mb, root=0)

    if rank == 0:
        max_time = max(all_times)
        total_mem = sum(all_mems)
        print(f"RESULT:{max_time:.4f},{total_mem:.2f}")
    
    # Clean up
    comm.Barrier()
    MPI.Finalize()

if __name__ == "__main__":
    main()
