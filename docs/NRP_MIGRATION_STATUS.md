# NRP Migration — Status

**Date:** 2026-04-27
**Status:** Mega-runner Job `mega-runner` running on `hcc-prp-c5036.unl.edu` (UNL A6000). Wave 1 (Qwen-32B RAGTAG) ~92% done; campaign ~12 h elapsed of an estimated ~66 h total wall.

This doc is a snapshot — written so any session (different machine, different conversation, days later) can pick up cleanly.

---

## 1. What's running

A **single mega-runner Job** holds one A6000 and processes all 58 production cells sequentially via `subprocess.run` calls inside the same Pod. Idempotent: skips any cell whose `preds_*.csv` already exists on the PVC, so a Pod restart resumes from the next un-finished cell. Each completed cell is tarred to `_outbox/<name>.tar.gz` for `sync.sh` to pick up.

The runner reuses `submit.py`'s `WAVE_BUILDERS` (via Python import) to enumerate cells and build their commands — no business logic is duplicated. Code is in [scripts/nrp/runners/run_remaining_cells.py](../scripts/nrp/runners/run_remaining_cells.py); manifest in [scripts/nrp/manifests/job-mega-runner.yaml](../scripts/nrp/manifests/job-mega-runner.yaml).

### Why this strategy

Cluster A6000 contention has been heavy (per-cell queues observed at 1-3 h). Submitting all 58 cells in parallel risked a 4-7 day campaign of mostly-queued pods. The mega-runner pattern: **acquire one A6000 once, hold it for the whole campaign**. NRP-policy compliant — the Job does continuous useful work, no `sleep infinity`.

---

## 2. Live progress (as of writing)

| | |
|---|---|
| Image | `ghcr.io/doctorhoseinpour/llm-labler:55ba8f1` |
| Pod | `mega-runner-4trxs` on `hcc-prp-c5036.unl.edu` (UNL A6000, 47.4 GB) |
| Elapsed | ~12 h |
| Cells done | **11/58** (1 canary SKIP + 10 real completions) |
| Currently on | `[12/58] w1-qwen32b-ragtag-agnostic`, k=1 at 56% (1858/3300) |
| Estimated remaining | **~54 h ≈ 2.3 days** |

### Per-cell timings observed (Wave 1 RAGTAG, project-specific)

| Cell | Wall | Notes |
|---|---|---|
| ansible | SKIP | canary already produced preds |
| bitcoin | 127 min | first real cell — paid cold-load penalty (model wasn't in node's page cache yet) |
| dart-lang | 45 min | |
| dotnet-roslyn | 50 min | |
| facebook-react | 48 min | |
| flutter-flutter | 80 min | longer issue bodies |
| kubernetes | 65 min | |
| microsoft-typescript | 72 min | longer issue bodies |
| microsoft-vscode | 58 min | |
| opencv | 61 min | |
| tensorflow | 73 min | |

Per-project average (excluding cold-start): ~62 min. ~38% slower than the canary's 45 min on a different node, attributed to node variance. Per-cell scaling ~1.38× over original projection.

### Wave queue (in execution order)

1. Wave 1 — Qwen-32B RAGTAG (12 cells: 1 ag + 11 proj)
2. Wave 2 — Qwen-14B FT (12 cells: 1 ag + 11 proj)
3. Wave 3 — Qwen-14B debias (11 cells, project-specific only)
4. Wave 4 — Qwen-32B debias (11 cells, project-specific only)
5. Wave 5 — Qwen-32B FT (12 cells: 1 ag + 11 proj)

Within each wave: project-specific cells first (faster, ~30-65 min each), agnostic cell last (4-8 h, the long pole).

---

## 3. What's in play

| Resource | Where | Size / detail |
|---|---|---|
| Container image | `ghcr.io/doctorhoseinpour/llm-labler:55ba8f1` (public on GHCR) | ~12.3 GB; SHA-pinned tag |
| HF weight cache | PVC `hf-cache-pvc` in namespace `bgsu-cs-heydarnoori` | 100 Gi rook-cephfs RWX, ~60 GB used (4 LLMs + sentence-transformers) |
| Results storage | PVC `results-pvc` | 50 Gi rook-cephfs RWX, ~3 GB used |
| Outbox (per-cell tarballs) | `_outbox/` inside `results-pvc` | empty as of latest sync; populates as cells complete |
| Safety fallback | git tag `pre-nrp-migration` + branch `backup/pre-nrp-migration` | both pushed to origin |
| `pod-level activeDeadlineSeconds` | 360000 (100 h) | clock starts when container runs, not at submission; queue time uncapped |

---

## 4. Bugs caught + fixed during the migration

Recorded so future sessions don't re-discover them. Each item has the fix commit:

- **Missing C compiler in the image** → triton's runtime kernel JIT failed; every prediction was the literal `ERROR: Failed to find C compiler` string. Fix: added `build-essential` to the Dockerfile. **Commit `ef6bd30`.**
- **Missing Python.h** → triton compiles `cuda_utils.c` with `#include <Python.h>` at runtime; needed `python3-dev` for the headers. Fix: added `python3-dev`. **Commit `7028c1a`.**
- **`:latest` mutable-tag cache stale on a node** → a previous-day image cached locally meant `imagePullPolicy: IfNotPresent` re-used the broken image. Fix: SHA-pinned tags (`:7028c1a`, then `:a0e6b7e`, now `:55ba8f1`). **Commit `ed7bc20`.**
- **Useless 633 MB FT tarballs from default Trainer saves** → HF Trainer wrote intermediate checkpoints + a final adapter; both unused (we never reload). Fix: `TrainingArguments(save_strategy="no")` + new `--skip_save_adapter` flag in `fixed_fine-tune.py`, passed by `submit.py` for all NRP FT cells. Tarballs went 633 MB → 113 KB (5600× reduction). **Commit `a0e6b7e`.**
- **`sync.sh` wrong eval path for FT cells** → it assumed all preds live in a `predictions/` subfolder (true for RAGTAG, false for FT); FT evals landed at `<model>/evaluations/` instead of `<model>/finetune_fixed/evaluations/`. Fix: detect both layouts. **Commit `55ba8f1` (bundled with mega-runner).**
- **`pkill -f "scripts/nrp/submit.py"` self-immolation** → the bash script running `pkill` had `submit.py` in its own command line; pkill matched the bash itself. Workaround: kill by PID, not pattern.
- **Pod-level vs Job-level `activeDeadlineSeconds`** → Job-level counts queue time; pod-level only counts runtime. We use pod-level. **Commit `68a27e6`.**

---

## 5. How to observe / control during the run

```bash
# Snapshot
kubectl get job,pod -l role=mega-runner -o wide

# Live tail
kubectl logs -f job/mega-runner

# Just milestones (filters out the chatty max_new_tokens warnings)
kubectl logs job/mega-runner | grep -E "^---|SKIP|OK in|FAILED|SUMMARY|MEGA-RUNNER"

# Pull tarballs whenever cells have completed (idempotent, safe to run anytime)
cd ~/llm-labler && bash scripts/nrp/sync.sh
```

### Stopping cleanly (if ever needed)

```bash
# Kill cleanly tonight, no auto-retry; resume later by re-applying the manifest
kubectl delete job mega-runner

# Resume after a fix (idempotent skip handles already-done cells)
sed 's|__IMAGE__|ghcr.io/doctorhoseinpour/llm-labler:<NEW_SHA>|' \
  scripts/nrp/manifests/job-mega-runner.yaml | kubectl apply -f -
```

The runner exits the pod naturally when all cells are done — no manual intervention needed for the happy path.

---

## 6. Resuming from a different machine

Three things needed:
1. `kubectl` + `kubelogin` for NRP OIDC auth
2. `~/.kube/config` (copy from the BGSU machine, then run any `kubectl get pods` to trigger browser auth)
3. Git clone of this repo at branch `11k-experiments`

Then the same `kubectl get` / `kubectl logs` / `bash scripts/nrp/sync.sh` workflow works. The `fetched/.synced` ledger is per-machine, so a fresh machine will re-pull tarballs that the BGSU machine already has — harmless duplication, no data loss.

---

## 7. After the campaign finishes

When the runner exits cleanly:
1. Final SUMMARY block in `kubectl logs job/mega-runner` lists pass/fail per cell
2. `bash scripts/nrp/sync.sh` one last time pulls any tarballs not yet integrated
3. Spot-check some eval CSVs in `results/issues11k/.../evaluations/`
4. Optional: clean up `_outbox/` on the PVC, delete the Job (or let the 7-day TTL handle it), revisit decluttering
5. Random-shot ablation (out of scope for this campaign) gets its own plan
