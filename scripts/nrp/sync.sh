#!/usr/bin/env bash
# sync.sh — local cron-driven daemon. Pulls new result tarballs from the NRP
# results-pvc into ./fetched/, untars, runs evaluate.py on any new prediction
# CSVs, then integrates new directories into ./results/.
#
# Designed to run on a local cron tick (every 15 min). Idempotent — uses
# ./fetched/.synced as a ledger of tarballs already pulled.
#
# Cron entry:
#   */15 * * * * cd /home/ahosein/llm-labler && bash scripts/nrp/sync.sh \
#       >> /tmp/llm-labler-sync.log 2>&1
#
# Mechanism:
#   1. Spawn a transient Pod whose only command is `ls /data/_outbox/`. It
#      exits naturally (NRP-policy compliant). kubectl logs gives the file
#      list.
#   2. For each tarball not in ./fetched/.synced, spawn another transient
#      Pod whose only command is `cat /data/_outbox/<file>`. kubectl logs
#      streams the bytes; redirect to ./fetched/<file>.
#   3. Untar, run evaluate.py on each new preds_*.csv, rsync the new
#      directories into ./results/.
#   4. Append to .synced.
set -euo pipefail

REPO_DIR="$(cd "$(dirname "$0")"/../.. && pwd)"
cd "$REPO_DIR"

FETCHED_DIR="$REPO_DIR/fetched"
LEDGER="$FETCHED_DIR/.synced"
TEMPLATE="$REPO_DIR/scripts/nrp/manifests/pod-puller-template.yaml"
LOCK_FILE="/tmp/llm-labler-sync.lock"

mkdir -p "$FETCHED_DIR"
touch "$LEDGER"

# Prevent overlapping runs (cron may fire while a prior pull is in flight)
exec 9>"$LOCK_FILE"
if ! flock -n 9; then
  echo "[sync] another instance is running; exiting"
  exit 0
fi

ts() { date -Is; }
echo "[sync] $(ts) start"

# ----------------------------------------------------------------------------
# spawn_pod NAME CMD  --  applies the templated Pod, waits for Succeeded,
#                          dumps stdout via kubectl logs, then deletes the Pod.
# Returns kubectl logs output on stdout. Failures go to stderr and the function
# exits non-zero (caller should set -e).
# ----------------------------------------------------------------------------
spawn_pod() {
  local name="$1"
  local cmd="$2"
  local manifest_tmp
  manifest_tmp=$(mktemp -t puller-pod-XXXXXX.yaml)
  trap 'rm -f "$manifest_tmp"' RETURN

  # Substitute placeholders. Note: __CMD__ is a shell command string passed to
  # `sh -c`. Quotes inside the command must already be escaped for YAML safety;
  # the callers below use only simple commands (`ls`, `cat`).
  sed -e "s|__NAME__|$name|g" -e "s|__CMD__|$cmd|g" "$TEMPLATE" >"$manifest_tmp"

  kubectl apply -f "$manifest_tmp" >&2
  # Wait for the Pod to terminate (success OR failure).
  kubectl wait --for=jsonpath='{.status.phase}'=Succeeded \
    "pod/$name" --timeout=300s >&2 || {
    # If it failed, surface logs to stderr and abort
    echo "[sync] pod $name did not Succeed; logs follow:" >&2
    kubectl logs "pod/$name" >&2 || true
    kubectl delete pod "$name" --wait=false --ignore-not-found >&2 || true
    rm -f "$manifest_tmp"
    return 1
  }
  kubectl logs "pod/$name"
  kubectl delete pod "$name" --wait=false --ignore-not-found >&2 || true
}

# Step 1: list the outbox
SUFFIX=$(date +%s)
list_pod="sync-lister-$SUFFIX"
remote_list=$(spawn_pod "$list_pod" "ls /data/_outbox/ 2>/dev/null || true" \
              | grep '\.tar\.gz$' || true)

if [[ -z "$remote_list" ]]; then
  echo "[sync] outbox empty, nothing to do"
  exit 0
fi

new_count=0
fail_count=0
while IFS= read -r tarball; do
  [[ -z "$tarball" ]] && continue
  if grep -Fxq "$tarball" "$LEDGER"; then
    continue
  fi
  echo "[sync] pulling $tarball"
  out_path="$FETCHED_DIR/$tarball"
  pull_pod="sync-puller-$SUFFIX-$(echo "$tarball" | tr -c 'a-z0-9-' '-' | cut -c1-30)"

  if ! spawn_pod "$pull_pod" "cat /data/_outbox/$tarball" >"$out_path"; then
    echo "[sync] FAILED to stream $tarball" >&2
    rm -f "$out_path"
    fail_count=$((fail_count + 1))
    continue
  fi

  # Verify gzip integrity
  if ! gzip -t "$out_path" 2>/dev/null; then
    echo "[sync] WARN: $tarball failed gzip integrity check; will retry" >&2
    rm -f "$out_path"
    fail_count=$((fail_count + 1))
    continue
  fi

  # Untar to a per-tarball staging dir
  staging="$FETCHED_DIR/${tarball%.tar.gz}"
  rm -rf "$staging"
  mkdir -p "$staging"
  tar xzf "$out_path" -C "$staging"

  # Run evaluate.py on each new prediction CSV the tarball brought in
  while IFS= read -r pred; do
    [[ -z "$pred" ]] && continue
    pred_dir="$(dirname "$pred")"
    eval_dir="$(dirname "$pred_dir")/evaluations"
    eval_file="$eval_dir/$(basename "$pred" | sed 's/^preds_/eval_/')"
    mkdir -p "$eval_dir"
    if [[ -f "$eval_file" ]]; then
      continue
    fi
    echo "[sync] eval: $(echo "$pred" | sed "s|$staging/||")"
    if ! python "$REPO_DIR/evaluate.py" \
        --preds_csv "$pred" \
        --output_csv "$eval_file" >/dev/null 2>&1; then
      echo "[sync] WARN: evaluate.py failed for $pred" >&2
    fi
  done < <(find "$staging" -type f -name "preds_*.csv")

  # Integrate into ./results/. Use rsync to merge directories without
  # overwriting existing files (paper-archival: never replace).
  if [[ -d "$staging/issues11k" ]]; then
    rsync -a --ignore-existing "$staging/issues11k/" "$REPO_DIR/results/issues11k/"
  fi
  # Other top-level subtrees (e.g. issues11k_random/ when out of scope re-enabled)
  for top in $(ls "$staging" 2>/dev/null); do
    [[ "$top" == "issues11k" ]] && continue
    [[ ! -d "$staging/$top" ]] && continue
    rsync -a --ignore-existing "$staging/$top/" "$REPO_DIR/results/$top/"
  done

  echo "$tarball" >> "$LEDGER"
  new_count=$((new_count + 1))
  echo "[sync] integrated: $tarball"
done <<< "$remote_list"

echo "[sync] $(ts) done. new=$new_count fail=$fail_count"
