#!/bin/bash
#SBATCH --partition=c23ms
#SBATCH --account=rwth2150
#SBATCH --ntasks=1
#SBATCH --mem=1G
#SBATCH --time=0:05:00
#SBATCH --output=logs/cgprobe_%j.out
echo "=== /proc/self/cgroup ==="; cat /proc/self/cgroup
echo "=== mountinfo cgroup ==="; grep cgroup /proc/self/mountinfo
echo "=== ls /sys/fs/cgroup ==="; ls /sys/fs/cgroup 2>&1 | head
echo "=== candidates ==="
for f in /sys/fs/cgroup/memory.current /sys/fs/cgroup/memory.peak /sys/fs/cgroup/memory/memory.usage_in_bytes; do
  printf '%s: ' "$f"; cat "$f" 2>&1 || true
done
echo "=== memory.current files under /sys/fs/cgroup ==="
find /sys/fs/cgroup -maxdepth 4 -name "memory.current" 2>/dev/null | head
echo "=== dirs matching job id ==="
find /sys/fs/cgroup -maxdepth 6 -type d -name "*${SLURM_JOB_ID}*" 2>/dev/null | head
