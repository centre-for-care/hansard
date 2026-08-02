#!/usr/bin/env bash
# Read-only cluster discovery. Run on the LOGIN node:
#   bash cluster/00_discovery.sh 2>&1 | tee discovery_$(hostname).txt
# Paste/share the output — every later choice (partitions, GPU count, FP8 vs
# tensor-parallel, where the HF cache lives, how models get downloaded) is
# decided from it. Nothing here submits jobs or changes state except one tiny
# 5-minute interactive GPU probe at the end (skippable with NO_GPU_PROBE=1).

set -uo pipefail
section() { printf '\n============ %s ============\n' "$*"; }

section "WHO/WHERE"
date; hostname; whoami
grep -e PRETTY_NAME /etc/os-release 2>/dev/null || true

section "SLURM PARTITIONS (name avail timelimit nodes gres mem cpus)"
sinfo -o "%P %a %l %D %G %m %c" 2>/dev/null || echo "sinfo not found — not a SLURM cluster?"

section "PARTITION DETAILS"
scontrol show partition 2>/dev/null | grep -E "PartitionName|MaxTime|TRES|DefMemPer|MaxMemPer|State" || true

section "GPU NODES (gres detail)"
sinfo -N -o "%N %P %G %m %c %t" 2>/dev/null | grep -iv "gres:$" | head -40 || true

section "MY ACCOUNT / QOS / LIMITS"
sacctmgr show assoc user="$USER" format=account,partition,qos,grptres%30,maxjobs,maxsubmit 2>/dev/null || true
sshare -U 2>/dev/null || true

section "STORAGE QUOTAS"
quota -s 2>/dev/null || true
for d in "$HOME" /scratch* /well* /gpfs* /data* /work*; do
  [ -d "$d" ] && df -h "$d" 2>/dev/null | tail -1 | sed "s|^|$d : |"
done 2>/dev/null | sort -u
echo "SCRATCH-like env vars:"; env | grep -iE "scratch|tmpdir" || true

section "MODULES (cuda / python / apptainer)"
module avail cuda 2>&1 | head -15 || true
module avail python 2>&1 | head -10 || true
module avail apptainer singularity 2>&1 | head -5 || true

section "TOOLING ON LOGIN NODE"
for t in python3 uv git curl rsync tmux; do
  printf '%-8s ' "$t"; command -v "$t" >/dev/null && "$t" --version 2>&1 | head -1 || echo "MISSING"
done

section "INTERNET FROM LOGIN NODE"
curl -sI --max-time 10 https://huggingface.co | head -1 || echo "NO internet from login node"

section "MAX ARRAY SIZE / SCHEDULER CONFIG"
scontrol show config 2>/dev/null | grep -iE "MaxArraySize|MaxJobCount|DefMemPer" || true

if [ -z "${NO_GPU_PROBE:-}" ]; then
  section "GPU PROBE (5-min interactive job; edit -p if it fails to schedule)"
  GPU_PART=$(sinfo -h -o "%P %G" 2>/dev/null | awk '$2 ~ /gpu/ {gsub(/\*/,"",$1); print $1; exit}')
  echo "trying partition: ${GPU_PART:-<none found>}"
  if [ -n "${GPU_PART:-}" ]; then
    srun -p "$GPU_PART" --gres=gpu:1 -t 5 --mem=8G bash -c '
      hostname; nvidia-smi
      echo "--- internet from COMPUTE node:"
      curl -sI --max-time 10 https://huggingface.co | head -1 || echo "NO internet from compute node"
    ' || echo "GPU probe did not schedule in time — run it manually later"
  fi
fi

section "DONE"
echo "Send this whole output back for the env setup step."
