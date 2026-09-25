#!/bin/bash
# Sequential GPU job queue for the shared lab 4090.
# Before each job: wait until the GPU has had NO compute processes for IDLE_S seconds
# (longer than the gaps between another session's back-to-back runs), so we never
# start while someone else is using the card. Jobs are lines of QUEUE_FILE
# ("<name>|<command>"); finished names are recorded in DONE_FILE, so the queue is
# idempotent and can be appended to while running. Run under tmux/nohup.
set -u
cd ~/llm-labler
EXP=results/issues11k/exploration/rag_next
QUEUE_FILE=${1:-$EXP/queue.txt}
DONE_FILE=$EXP/logs/queue_done.txt
IDLE_S=${IDLE_S:-120}
mkdir -p $EXP/logs; touch "$DONE_FILE"

gpu_idle_for() {  # returns when the GPU has had no compute apps for $1 seconds
  local need=$1 since=$(date +%s)
  while true; do
    if [ -n "$(nvidia-smi --query-compute-apps=pid --format=csv,noheader)" ]; then
      since=$(date +%s)
    elif [ $(( $(date +%s) - since )) -ge "$need" ]; then
      return
    fi
    sleep 10
  done
}

while true; do
  job=""
  while IFS= read -r line; do
    [ -z "$line" ] && continue; [[ "$line" == \#* ]] && continue
    name=${line%%|*}
    grep -qxF "$name" "$DONE_FILE" || { job="$line"; break; }
  done < "$QUEUE_FILE"
  [ -z "$job" ] && { echo "$(date +%T) queue empty" >> $EXP/logs/progress.txt; break; }
  name=${job%%|*}; cmd=${job#*|}
  gpu_idle_for "$IDLE_S"
  echo "$(date +%T) START $name" >> $EXP/logs/progress.txt
  bash -c "$cmd" > "$EXP/logs/$name.log" 2>&1
  rc=$?
  echo "$(date +%T) END $name rc=$rc" >> $EXP/logs/progress.txt
  echo "$name" >> "$DONE_FILE"
  # after our own job, only a short idle check is needed for the next one
  IDLE_S=${IDLE_S_NEXT:-30}
done
