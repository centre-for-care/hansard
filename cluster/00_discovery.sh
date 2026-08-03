#!/usr/bin/env bash
# Read-only discovery — slimmed to what the BMRC docs + walkthrough did NOT
# already establish (partitions/limits/accounts are documented; compute nodes
# have no internet). Remaining unknowns: /well quota headroom, GPU queue
# occupancy right now, python module + tooling on the login node. Run on a
# login node (cluster1-4.bmrc.ox.ac.uk):
#   bash cluster/00_discovery.sh 2>&1 | tee discovery.txt

set -uo pipefail
section() { printf '\n============ %s ============\n' "$*"; }

section "WHO/WHERE"
date; hostname; whoami; groups
echo "group dirs under /well that mention you:"
for g in $(groups); do [ -d "/well/$g/users/$USER" ] && echo "  /well/$g/users/$USER"; done

section "STORAGE HEADROOM (/well needs ~400GB for models + caches)"
quota -s 2>/dev/null || true
for g in $(groups); do
  d="/well/$g"
  [ -d "$d" ] && df -h "$d" 2>/dev/null | tail -1 | sed "s|^|$d : |"
done | sort -u

section "GPU QUEUE OCCUPANCY (idle nodes now)"
sinfo -p gpu_gh200_144gb,gpu_a100_80gb,gpu_a100_40gb,gpu_rtx8000_48gb,gpu_interactive \
      -o "%P %a %D %t" 2>/dev/null | sort -u

section "PYTHON MODULE"
module avail Python/3.11 2>&1 | head -5 || true

section "TOOLING ON LOGIN NODE"
for t in python3 uv git curl rsync tmux; do
  printf '%-8s ' "$t"; command -v "$t" >/dev/null && "$t" --version 2>&1 | head -1 || echo "MISSING"
done

section "INTERNET FROM LOGIN NODE (needed for model + wheel downloads)"
curl -sI --max-time 10 https://huggingface.co | head -1 || echo "NO internet from login node"
curl -sI --max-time 10 https://pypi.org | head -1 || echo "NO pypi from login node"

section "DONE"
