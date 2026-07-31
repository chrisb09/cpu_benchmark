#!/usr/bin/env python3
import argparse
import os
import shutil
import sys
import time
import socket
import subprocess
from pathlib import Path

from smartsim.experiment import Experiment

def find_free_port(start_port=6780, max_attempts=100):
    """Finds a free port starting from the given port."""
    for port in range(start_port, start_port + max_attempts):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            try:
                s.bind(('', port))
                return port
            except OSError:
                continue
    raise RuntimeError(f"Could not find a free port in range {start_port}-{start_port+max_attempts}")

def log_redis_telemetry(silent=False):
    """Logs Redis server PID, affinity mask, and environment details."""
    try:
        # Find redis-server process
        output = subprocess.check_output(["pgrep", "-f", "redis-server"], text=True)
        pids = output.strip().split()
        for pid in pids:
            status_file = Path(f"/proc/{pid}/status")
            if status_file.exists():
                with open(status_file, "r") as f:
                    for line in f:
                        if line.startswith("Cpus_allowed_list:"):
                            if not silent:
                                print(f"[TELEMETRY] Redis PID: {pid} | {line.strip()}", flush=True)
                            break
    except Exception as exc:
        if not silent:
            print(f"[TELEMETRY] Unable to query Redis process affinity: {exc}", flush=True)

def main() -> int:
    parser = argparse.ArgumentParser(description="Diagnostic SmartSim DB Orchestrator for Factorial Causality Sweeps.")
    
    # Network & Topology
    parser.add_argument("--port", type=int, default=6780, help="Base database port")
    parser.add_argument("--auto-port", action="store_true", help="Automatically find a free port")
    parser.add_argument("--interface", default="lo", help="Network interface (e.g., lo, ib0)")
    parser.add_argument("--db-nodes", type=int, default=1, help="Number of database nodes/shards")
    
    # Independent RedisAI & DB Affinity Knobs (un-coupled!)
    parser.add_argument("--rai-tpq", type=int, default=None, help="Explicit RedisAI THREADS_PER_QUEUE (omitted if not specified)")
    parser.add_argument("--rai-intra", type=int, default=None, help="Explicit RedisAI INTRA_OP_PARALLELISM (omitted if not specified)")
    parser.add_argument("--rai-inter", type=int, default=None, help="Explicit RedisAI INTER_OP_PARALLELISM (omitted if not specified)")
    parser.add_argument("--db-cpus", type=int, default=None, help="Explicit DB cores to bind via db.set_cpus() (omitted if not specified)")
    
    # Synchronization & Experiment Output
    parser.add_argument("--endpoint-file", default=".ssdb_endpoint", help="File to write host:port")
    parser.add_argument("--done-file", default=".solver_done", help="File that signals solver completion")
    parser.add_argument("--timeout-s", type=float, default=120.0, help="Readiness timeout for the DB")
    parser.add_argument("--launcher", default=None, choices=["local", "slurm"], help="Launcher to use.")
    parser.add_argument("--exp-dir", default=None, help="Base experiment directory")
    parser.add_argument("--silent", action="store_true", help="Suppress output")
    
    args = parser.parse_args()

    launcher = args.launcher or ("slurm" if "SLURM_JOB_ID" in os.environ else "local")

    port = args.port
    if args.auto_port:
        port = find_free_port(start_port=args.port)

    endpoint_file = Path(args.endpoint_file)
    done_file = Path(args.done_file)

    if endpoint_file.exists():
        endpoint_file.unlink()
    if done_file.exists():
        done_file.unlink()

    exp_name = f"smartsim_diag_{int(time.time())}"
    if args.exp_dir:
        exp_path = Path(args.exp_dir) / exp_name
    else:
        exp_path = Path(os.getcwd()) / "smartsim_experiments" / exp_name
    
    if exp_path.exists():
        shutil.rmtree(exp_path)
    exp_path.mkdir(parents=True, exist_ok=True)

    if not args.silent:
        print("=== Diagnostic Orchestrator Configuration ===")
        print(f"Launcher: {launcher} | Port: {port} | Interface: {args.interface}")
        print(f"RedisAI TPQ: {args.rai_tpq} (omitted if None)")
        print(f"RedisAI Intra-op: {args.rai_intra} (omitted if None)")
        print(f"RedisAI Inter-op: {args.rai_inter} (omitted if None)")
        print(f"DB set_cpus: {args.db_cpus} (omitted if None)")
        print(f"Exp Path: {exp_path}")
        print("==============================================", flush=True)

    exp = Experiment(name=exp_name, launcher=launcher, exp_path=str(exp_path))
    
    # Build create_database kwargs dynamically based strictly on specified flags
    db_kwargs = {
        "port": port,
        "interface": args.interface,
        "db_nodes": args.db_nodes,
        "single_cmd": False,
        "batch": False,
    }
    if args.rai_tpq is not None:
        db_kwargs["threads_per_queue"] = args.rai_tpq
    if args.rai_intra is not None:
        db_kwargs["intra_op_threads"] = args.rai_intra
    if args.rai_inter is not None:
        db_kwargs["inter_op_threads"] = args.rai_inter

    db = exp.create_database(**db_kwargs)

    if launcher == "slurm":
        db.set_run_arg("export", "ALL")
        db.set_run_arg("mem", "0")
        
    if args.db_cpus is not None and args.db_cpus > 0:
        db.set_cpus(args.db_cpus)
        if not args.silent:
            print(f"Calling db.set_cpus({args.db_cpus}).", flush=True)

    exp.start(db, block=False, summary=not args.silent)

    # Wait for DB readiness
    start = time.time()
    addresses = None
    while time.time() - start < args.timeout_s:
        try:
            addresses = db.get_address()
            if addresses:
                break
        except Exception:
            pass
        time.sleep(0.5)

    if not addresses:
        exp.stop(db)
        raise RuntimeError("SmartSim database did not become ready in time.")

    log_redis_telemetry(silent=args.silent)

    endpoint = ",".join(addresses)
    endpoint_file.write_text(endpoint + "\n", encoding="utf-8")
    if not args.silent:
        print(f"Database ready. SSDB={endpoint}", flush=True)

    # Set RedisAI MODEL_EXECUTION_TIMEOUT to prevent 5-second default timeout on long inferences
    try:
        import redis
        host, port_str = addresses[0].split(":")
        r = redis.Redis(host=host, port=int(port_str), socket_timeout=10)
        r.execute_command("AI.CONFIG", "MODEL_EXECUTION_TIMEOUT", "2000000")
        if not args.silent:
            print("[DIAGNOSTIC] Successfully set RedisAI MODEL_EXECUTION_TIMEOUT to 2,000,000 ms.", flush=True)
    except Exception as exc:
        if not args.silent:
            print(f"[DIAGNOSTIC] Warning: Failed to set RedisAI MODEL_EXECUTION_TIMEOUT: {exc}", flush=True)

    # Monitor done file
    while not done_file.exists():
        time.sleep(0.5)

    if not args.silent:
        print("Solver completion signaled. Shutting down database.", flush=True)
    
    import threading
    stop_thread = threading.Thread(target=exp.stop, args=(db,))
    stop_thread.start()
    stop_thread.join(timeout=15.0)
    
    try:
        done_file.unlink()
    except Exception:
        pass

    return 0

if __name__ == "__main__":
    sys.exit(main())
