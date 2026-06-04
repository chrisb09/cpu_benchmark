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
        # Try some default locations
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
        # Load to CPU first
        model = torch.jit.load(model_path, map_location="cpu")
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"FAILED to load model: {e}")
        return

    # Create dummy inputs on CPU
    num_inputs = args.num_inputs
    if args.schema == "mini_app":
        warmup_input_cpu = torch.randn(1, 18, dtype=torch.float32)
        batch_input_cpu = torch.randn(num_inputs, 18, dtype=torch.float32)
    elif args.schema == "mmcp":
        warmup_input_cpu = torch.randn(args.seq_len, 1, args.feature_dim, dtype=torch.float32)
        batch_input_cpu = torch.randn(args.seq_len, num_inputs, args.feature_dim, dtype=torch.float32)

    # --- Warmup ---
    # Transfer warmup input
    warmup_input_gpu = warmup_input_cpu.to(device)
    # Warmup inference
    with torch.no_grad():
        _ = model(warmup_input_gpu)
    torch.cuda.synchronize()

    # --- Benchmark ---
    
    # 1. Measure data transfer time: CPU -> GPU
    transfer_in_start = time.perf_counter()
    batch_input_gpu = batch_input_cpu.to(device)
    torch.cuda.synchronize()
    transfer_in_end = time.perf_counter()
    transfer_in_time = transfer_in_end - transfer_in_start

    # 2. Measure actual inference time
    inference_start = time.perf_counter()
    outputs_gpu = []
    with torch.no_grad():
        cursor = 0
        while cursor < num_inputs:
            end_idx = min(cursor + max_bs, num_inputs)
            if args.schema == "mmcp":
                batch = batch_input_gpu[:, cursor:end_idx, :]
            else:
                batch = batch_input_gpu[cursor:end_idx]
            
            out = model(batch)
            outputs_gpu.append(out)
            cursor = end_idx
    torch.cuda.synchronize()
    inference_end = time.perf_counter()
    inference_time = inference_end - inference_start

    # 3. Measure data transfer time: GPU -> CPU
    transfer_out_start = time.perf_counter()
    # To properly measure transfer out of all data, we move all outputs back
    outputs_cpu = [out.to("cpu") for out in outputs_gpu]
    # Also synchronize to ensure transfers are done
    torch.cuda.synchronize()
    transfer_out_end = time.perf_counter()
    transfer_out_time = transfer_out_end - transfer_out_start

    total_time = transfer_in_time + inference_time + transfer_out_time

    print(f"\nBenchmark Results for {args.model} ({num_inputs} inputs):")
    print(f"  Transfer CPU -> GPU: {transfer_in_time:.4f}s")
    print(f"  Actual Inference:    {inference_time:.4f}s")
    print(f"  Transfer GPU -> CPU: {transfer_out_time:.4f}s")
    print(f"  ------------------------------")
    print(f"  Total Time:          {total_time:.4f}s")
    
    # Simple CSV-style output for easy parsing
    print(f"RESULT_GPU:{args.model},{num_inputs},{transfer_in_time:.4f},{inference_time:.4f},{transfer_out_time:.4f},{total_time:.4f}")

if __name__ == "__main__":
    main()
