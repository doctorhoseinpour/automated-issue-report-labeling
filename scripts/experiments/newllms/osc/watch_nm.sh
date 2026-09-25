#!/usr/bin/env bash
# Local PC: re-run launch_nm.sh for a runs file every 3 min until every shard is done
# (Ascend preemptible jobs are CANCELLED on preemption, not requeued). One line per round.
#   bash scripts/experiments/newllms/osc/watch_nm.sh runs_main.txt
RUNS="${1:?runs file name in osc/}"
L=/users/PCS0289/alirezzzhp1378/nm/repo/scripts/experiments/newllms/osc
while true; do
  out=$(timeout 120 ssh -o BatchMode=yes -o ConnectTimeout=20 -o ServerAliveInterval=15 alirezzzhp1378@cardinal.osc.edu "bash $L/launch_nm.sh $L/$RUNS" 2>/dev/null \
        | grep -E "^shards:|^submitted|^GIVING UP")
  echo "$(date +%T) $(echo "$out" | tr '\n' ' ')"
  if echo "$out" | grep -qE "^shards: ([0-9]+) total, \1 done"; then echo "ALL DONE $RUNS"; break; fi
  if echo "$out" | grep -q "GIVING UP"; then echo "FAILURES in $RUNS"; exit 1; fi
  sleep 180
done
