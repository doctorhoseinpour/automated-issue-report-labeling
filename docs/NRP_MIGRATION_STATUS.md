# NRP Migration — Status

**Last updated:** 2026-05-04

This doc was originally a live snapshot of the 2026-04-27 1-epoch FT campaign. That campaign completed; the live-progress sections were removed because they had become misleading. The "Bugs caught" log is preserved as institutional knowledge — every item there is a class of failure future sessions should not re-discover.

For the current campaign state, read [paper/SESSION_HANDOFF.md](../paper/SESSION_HANDOFF.md). For the steady-state cluster setup (PVCs, image discipline, mega-runner pattern), read on.

---

## Steady-state setup

A **single mega-runner Job** holds one A6000 and processes all production cells sequentially via `subprocess.run` calls inside the same Pod. Idempotent: skips any cell whose `preds_*.csv` already exists on the PVC. Each completed cell is tarred to `_outbox/<name>.tar.gz` for `sync.sh` to pick up.

The runner reuses `submit.py`'s `WAVE_BUILDERS` (via Python import) to enumerate cells and build their commands — no business logic is duplicated. Code: [scripts/nrp/runners/run_remaining_cells.py](../scripts/nrp/runners/run_remaining_cells.py); manifest: [scripts/nrp/manifests/job-mega-runner.yaml](../scripts/nrp/manifests/job-mega-runner.yaml); plan: [scripts/nrp/plan.yaml](../scripts/nrp/plan.yaml) (waves field controls which cells run).

**Why this strategy:** cluster A6000 contention has been heavy (per-cell queues observed at 1–3 h). Submitting all cells in parallel risked a multi-day campaign of mostly-queued pods. Mega-runner: acquire one A6000 once, hold it for the whole campaign. NRP-policy compliant — continuous useful work, no `sleep infinity`.

| Resource | Where | Detail |
|---|---|---|
| Container image | `ghcr.io/doctorhoseinpour/llm-labler:<sha>` (public on GHCR) | ~12.3 GB; **always SHA-pinned**, never `:latest` |
| HF weight cache | PVC `hf-cache-pvc` in namespace `bgsu-cs-heydarnoori` | 100 Gi rook-cephfs RWX, ~60 GB used (4 LLMs + sentence-transformers) |
| Results storage | PVC `results-pvc` | 50 Gi rook-cephfs RWX |
| Outbox (per-cell tarballs) | `_outbox/` inside `results-pvc` | populated as cells complete; drained by `sync.sh` |
| Pod-level activeDeadlineSeconds | 360000 (100 h) in `job.yaml.j2` | clock starts when container runs, not at submission; queue time uncapped |

**Image SHA discipline:** every code change → new commit → rebuild → push → update `plan.yaml` SHA. After pushing, **always** verify the cluster pulls the new image with an ephemeral pod that greps for the expected change. Skipping that step has burned us before (see "Bugs caught" §1).

---

## Bugs caught + fixed during the migration

Each item is a class of failure that should not recur. Fix commit recorded.

- **Missing C compiler in the image** → triton's runtime kernel JIT failed; every prediction was the literal `ERROR: Failed to find C compiler` string. Fix: added `build-essential` to the Dockerfile. **Commit `ef6bd30`.**
- **Missing Python.h** → triton compiles `cuda_utils.c` with `#include <Python.h>` at runtime; needed `python3-dev` for the headers. Fix: added `python3-dev`. **Commit `7028c1a`.**
- **`:latest` mutable-tag cache stale on a node** → a previous-day image cached locally meant `imagePullPolicy: IfNotPresent` re-used the broken image. Fix: SHA-pinned tags from then on. **Commit `ed7bc20`.**
- **Useless 633 MB FT tarballs from default Trainer saves** → HF Trainer wrote intermediate checkpoints + a final adapter; both unused (we never reload). Fix: `TrainingArguments(save_strategy="no")` + new `--skip_save_adapter` flag in `fixed_fine-tune.py`, passed by `submit.py` for all NRP FT cells. Tarballs went 633 MB → 113 KB (5600× reduction). **Commit `a0e6b7e`.**
- **`sync.sh` wrong eval path for FT cells** → it assumed all preds live in a `predictions/` subfolder (true for RAGTAG, false for FT); FT evals landed at `<model>/evaluations/` instead of `<model>/finetune_fixed/evaluations/`. Fix: detect both layouts. **Commit `55ba8f1`.**
- **`pkill -f "scripts/nrp/submit.py"` self-immolation** → the bash script running `pkill` had `submit.py` in its own command line; pkill matched the bash itself. Workaround: kill by PID, not pattern.
- **Pod-level vs Job-level `activeDeadlineSeconds`** → Job-level counts queue time; pod-level only counts runtime. We use pod-level. **Commit `68a27e6`.**

---

## How to observe / control during a run

```bash
# Snapshot
kubectl -n bgsu-cs-heydarnoori get jobs,pods

# Live tail
kubectl -n bgsu-cs-heydarnoori logs -f job/mega-runner

# Just milestones (filters out chatty max_new_tokens warnings)
kubectl -n bgsu-cs-heydarnoori logs job/mega-runner | grep -E "^---|SKIP|OK in|FAILED|SUMMARY|MEGA-RUNNER"

# Pull tarballs whenever cells have completed (idempotent)
cd ~/llm-labler && bash scripts/nrp/sync.sh
```

### Stopping cleanly

```bash
kubectl -n bgsu-cs-heydarnoori delete job mega-runner

# Resume after a fix (idempotent skip handles already-done cells)
sed 's|__IMAGE__|ghcr.io/doctorhoseinpour/llm-labler:<NEW_SHA>|' \
  scripts/nrp/manifests/job-mega-runner.yaml | kubectl apply -f -
```

The runner exits the pod naturally when all cells are done — no manual intervention needed for the happy path.

---

## Resuming from a different machine

Three things needed:
1. `kubectl` + `kubelogin` for NRP OIDC auth
2. `~/.kube/config` (copy from the BGSU machine, then run any `kubectl get pods` to trigger browser auth)
3. Git clone of this repo at the active branch

The `fetched/.synced` ledger is per-machine, so a fresh machine will re-pull tarballs that the BGSU machine already has — harmless duplication, no data loss.
