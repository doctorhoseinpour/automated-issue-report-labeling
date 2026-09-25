#!/usr/bin/env bash
# Idempotent launcher (run on an OSC login node). Reads run specs, one per line:
#   <run>|<tag>|<nshards>|<cluster>|<partition>|<time>|<run_llm.py args after --tag/--run>
# and submits every shard that has no done marker and no queued/running job of the same name.
# Safe to call repeatedly (e.g. every few minutes) to resubmit shards cancelled by preemption.
#   bash launch_nm.sh runs.txt
set -uo pipefail
ROOT=/users/PCS0289/alirezzzhp1378/nm
RAW=$ROOT/repo/results/issues11k/exploration/newllms/raw
SB=$ROOT/repo/scripts/experiments/newllms/osc/gpu_nm.sbatch
SPEC="${1:?runs file}"
active=$( (squeue -h -u "$USER" -o %j; squeue -M ascend -h -u "$USER" -o %j) 2>/dev/null | grep '^nm-' | sort -u)
n_sub=0; n_done=0; n_act=0; n_all=0
while IFS='|' read -r run tag nsh cluster part tlim rest; do
  [[ -z "$run" || "$run" == \#* ]] && continue
  for ((i = 0; i < nsh; i++)); do
    n_all=$((n_all + 1))
    name=$(printf "nm-%s-%s-%02d" "$tag" "$run" "$i")
    if [ -f "$RAW/$tag/$run/$(printf 'done_%02dof%02d.json' "$i" "$nsh")" ]; then n_done=$((n_done + 1)); continue; fi
    if grep -qx "$name" <<< "$active"; then n_act=$((n_act + 1)); continue; fi
    nfail=$(sacct -M "$cluster" -u "$USER" -S now-1days -X -n --name="$name" --format=State 2>/dev/null | grep -cE "FAILED|OUT_OF_ME|NODE_FAIL")
    if [ "$nfail" -ge 2 ]; then echo "GIVING UP $name ($nfail failures; see ~/nm/logs)"; continue; fi
    # shellcheck disable=SC2086
    sbatch -M "$cluster" --partition="$part" --time="$tlim" --job-name="$name" \
      --export=ALL,NM_SHARD=$i,NM_NSHARDS=$nsh "$SB" run_llm.py --tag "$tag" --run "$run" $rest >/dev/null \
      && n_sub=$((n_sub + 1)) && echo "submitted $name"
  done
done < "$SPEC"
echo "shards: $n_all total, $n_done done, $n_act active, $n_sub submitted now"
