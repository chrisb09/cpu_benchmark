import argparse
import time
import os
import torch

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gpu", type=int, default=3, help="GPU ID to use")
    parser.add_argument("--model", type=str, default="giant",
                        help="Model name prefix or full path")
    parser.add_argument("--num-inputs", type=int, default=100000,
                        help="TOTAL inference inputs")
    parser.add_argument("--schema", type=str, default="mini_app",
                        choices=["mini_app", "mmcp"],
                        help="Input schema: 'mini_app' (B, 18) or 'mmcp' (5, B, 512)")
    parser.add_argument("--seq-len", type=int, default=5, help="Sequence length for mmcp schema")
    parser.add_argument("--feature-dim", type=int, default=512, help="Feature dimension for mmcp schema")
    parser.add_argument("--max-batch-size", type=int, default=0, 
                        help="Maximum batch size per inference call. 0 = auto-detect based on schema.")
    parser.add_argument("--runs", type=int, default=10, help="Number of benchmark runs")
    args = parser.parse_args()

    # Check for GPU
    if not torch.cuda.is_available():
        print("CUDA not available. Aborting.")
        return
    
    device = torch.device(f"cuda:{args.gpu}")
    torch.cuda.set_device(device)
    print(f"Using device: {torch.cuda.get_device_name(device)}")

    # Resolve max batch size
    max_bs = args.max_batch_size
    if max_bs <= 0:
        if args.schema == "mmcp":
            max_bs = 5000
        elif "transformer" in args.model.lower():
            max_bs = 10000 # Transformer models need smaller batches to avoid OOM
        else:
            max_bs = 100000 

    # Load model
    model_path = args.model
    if not os.path.exists(model_path):
        paths_to_try = [
            f"../mini_app/train_models/model_a/{args.model}",
            f"../mini_app/train_models/model_a/{args.model}_cpu.pt",
            f"../mini_app/train_models/model_a/{args.model}_cuda.pt",
            f"../MMCP_TOM/input/{args.model}",
        ]
        found = False
        for p in paths_to_try:
            if os.path.exists(p):
                model_path = p
                found = True
                break
        if not found:
            print(f"Model not found. Tried: {paths_to_try}")
            return

    print(f"Loading model from: {model_path}")
    try:
        model = torch.jit.load(model_path, map_location="cpu")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    num_inputs = args.num_inputs
    if args.schema == "mini_app":
        warmup_input_cpu = torch.randn(1, 18, dtype=torch.float32)
        batch_input_cpu = torch.randn(num_inputs, 18, dtype=torch.float32)
    elif args.schema == "mmcp":
        warmup_input_cpu = torch.randn(3 * 1, args.seq_len, args.feature_dim, dtype=torch.float32)
        batch_input_cpu = torch.randn(3 * num_inputs, args.seq_len, args.feature_dim, dtype=torch.float32)

    # --- Warmup ---
    warmup_input_gpu = warmup_input_cpu.to(device)
    with torch.no_grad():
        _ = model(warmup_input_gpu)
    torch.cuda.synchronize()

    # --- Benchmark ---
    import numpy as np
    
    total_times = []

    for r in range(args.runs):
        transfer_in_start = time.perf_counter()
        batch_input_gpu = batch_input_cpu.to(device)
        torch.cuda.synchronize()
        transfer_in_time = time.perf_counter() - transfer_in_start

        inference_start = time.perf_counter()
        outputs_gpu = []
        with torch.no_grad():
            cursor = 0
            while cursor < num_inputs:
                end_idx = min(cursor + max_bs, num_inputs)
                if args.schema == "mmcp":
                    batch = batch_input_gpu[3 * cursor : 3 * end_idx, :, :]
                else:
                    batch = batch_input_gpu[cursor:end_idx]
                
                out = model(batch)
                outputs_gpu.append(out)
                cursor = end_idx
        torch.cuda.synchronize()
        inference_time = time.perf_counter() - inference_start

        transfer_out_start = time.perf_counter()
        outputs_cpu = [out.to("cpu") for out in outputs_gpu]
        torch.cuda.synchronize()
        transfer_out_time = time.perf_counter() - transfer_out_start

        total_time = transfer_in_time + inference_time + transfer_out_time
        total_times.append(total_time)
        
        print(f"Run {r+1}/{args.runs}: {total_time:.4f}s")
        
        # Cleanup to prevent OOM
        del batch_input_gpu
        del outputs_gpu
        del outputs_cpu
        torch.cuda.empty_cache()

    total_times = np.array(total_times)
    median = np.median(total_times)
    mean_arith = np.mean(total_times)
    mean_geom = np.exp(np.mean(np.log(total_times)))
    variance = np.var(total_times, ddof=1)
    
    # 95% CI (1.96 * std / sqrt(N))
    std_dev = np.std(total_times, ddof=1)
    ci_margin = 1.96 * (std_dev / np.sqrt(args.runs))
    ci_lower = mean_arith - ci_margin
    ci_upper = mean_arith + ci_margin

    print(f"\\nBenchmark Results for {args.model} ({num_inputs} inputs, {args.runs} runs):")
    print(f"  Median: {median:.4f}s")
    print(f"  Arith Mean: {mean_arith:.4f}s")
    print(f"  Geom Mean: {mean_geom:.4f}s")
    print(f"  Variance: {variance:.6f}")
    print(f"  95% CI: [{ci_lower:.4f}s, {ci_upper:.4f}s]")
    
    csv_file = "gpu_stats_results.csv"
    write_header = not os.path.exists(csv_file)
    with open(csv_file, "a") as f:
        if write_header:
            runs_cols = ",".join([f"run_{i+1}" for i in range(args.runs)])
            f.write(f"model,schema,num_inputs,{runs_cols},median,mean_arith,mean_geom,variance,ci_lower,ci_upper\n")
        
        runs_vals = ",".join([f"{t:.4f}" for t in total_times])
        model_basename = os.path.basename(args.model)
        f.write(f"{model_basename},{args.schema},{num_inputs},{runs_vals},{median:.4f},{mean_arith:.4f},{mean_geom:.4f},{variance:.6f},{ci_lower:.4f},{ci_upper:.4f}\n")

if __name__ == "__main__":
    main()
