# NRP Kubernetes Jobs Migration — Status

**Date:** 2026-04-25
**Status:** Smoke test in flight (queued for tier-S GPU). All infrastructure in place. Production campaign not yet launched.

This doc captures the live state of the NRP migration so any future session — different machine, different Claude conversation, days later — can pick up cleanly.

---

## 1. What we built

A set-and-forget Kubernetes Jobs pipeline that replaces the JupyterHub workflow for the remaining 11k experiments. Five experimental waves run on NRP A6000 GPUs:

- **Wave 0** — indexing (FAISS over agnostic + 11 project splits), one Job
- **Wave 1** — Qwen-32B RAGTAG redo, 12 Jobs (1 agnostic + 11 per-project)
- **Wave 2** — Qwen-14B FT, 12 Jobs
- **Wave 3** — Qwen-14B debiased RAGTAG, 11 Jobs (project-specific only)
- **Wave 4** — Qwen-32B debiased RAGTAG, 11 Jobs
- **Wave 5** — Qwen-32B FT, 12 Jobs

A custom container image (`ghcr.io/doctorhoseinpour/llm-labler:latest`) bakes in the code + datasets + dependencies. Two CephFS RWX PVCs hold model weights (cached once) and outputs. Each Job tars its output to `_outbox/` on the results PVC; `scripts/nrp/sync.sh` pulls those tarballs back to `./results/` on the local machine when invoked manually. The full plan is in [/home/ahosein/.claude/plans/i-have-an-nrp-warm-allen.md](../../../home/ahosein/.claude/plans/i-have-an-nrp-warm-allen.md) on the BGSU machine.

---

## 2. Where things live

| Resource | Location | Notes |
|---|---|---|
| Code (orchestration) | `scripts/nrp/` in this repo | All committed to branch `11k-experiments` |
| Container image | `ghcr.io/doctorhoseinpour/llm-labler:b74eb2d` (pinned), `:ef6bd30` (latest), `:latest` | Public on GHCR |
| HF weight cache | PVC `hf-cache-pvc` (100Gi, rook-cephfs RWX) in namespace `bgsu-cs-heydarnoori` | Has all 4 LLMs + sentence-transformers |
| Results | PVC `results-pvc` (50Gi, rook-cephfs RWX) | Empty between runs except `_outbox/` between syncs |
| Safety fallback | git tag `pre-nrp-migration` + branch `backup/pre-nrp-migration` | Pushed to origin |
| Plan file | `/home/ahosein/.claude/plans/i-have-an-nrp-warm-allen.md` | BGSU machine only |

---

## 3. Setup checkpoints — all done

| Step | What | Done |
|---|---|---|
| P0 | Tag `pre-nrp-migration` on `63edef0` (pre-migration HEAD), pushed to origin | ✓ |
| P1 | GHCR push token configured locally | ✓ |
| P2 | PVCs `hf-cache-pvc` (100Gi) + `results-pvc` (50Gi) created on `rook-cephfs` | ✓ |
| P3 | Image built and pushed: tags `b74eb2d`, `ef6bd30`, `latest` | ✓ |
| P4 | Warmup Job downloaded all 5 model snapshots into `hf-cache-pvc` | ✓ |
| P5 | All wave deadlines bumped to 20h, moved to **pod-level** so queue time doesn't count | ✓ |

---

## 4. Smoke test — current state

The smoke test (`smoketest-indexing` + `smoketest-llama3b` Jobs) is the pre-flight check before launching the 60-job production campaign. It uses Llama-3B zero-shot on `ansible_ansible` (300 issues) under a sandboxed `_smoketest/` path on the PVC.

**As of this write:**
- `smoketest-indexing` is **Pending** (queued for a free RTX-3090/A5000), no node assigned.
- `smoketest-llama3b` not submitted yet (waits for indexing per `submit.py` serialization).
- A background watcher (`bnsw3ovgu` on the BGSU machine) is monitoring with `kubectl wait --timeout=72060s`.

**What we already learned from previous (broken) smoke attempts** — these are FIXED in the current image/scripts:
1. Image was missing a C compiler (`build-essential`) — triton kernel JIT failed at runtime, every prediction came back as the literal triton error string. Fixed in commit `ef6bd30`.
2. Stale predictions on the PVC caused `llm_labeler.py` to skip re-running and the Job exited rc=0 without doing real work — gave a false positive. Fixed by wiping `_smoketest/` between re-attempts.
3. `sync.sh` had four separate bugs: sed substitution choking on `|` in commands, pod names with trailing dashes (k8s rejects), `kubectl logs` corrupting binary tarballs (need base64 round-trip), and a `pkill -f` that matched its own bash script's command line and self-immolated. All fixed.

---

## 5. Verification needed before launching production

Once `smoketest-llama3b` completes:

1. Run `bash scripts/nrp/sync.sh` to pull the smoke test tarballs back.
2. Inspect `results/issues11k/_smoketest/ansible_ansible/unsloth_Llama_3_2_3B_Instruct/ragtag/predictions/preds_zero_shot.csv`.
3. Confirm:
   - `parsed_via` column shows `xml` or `regex` for most rows (NOT `failed`).
   - `predicted_label` is one of `bug`/`feature`/`question` for most rows (NOT `invalid`).
   - `raw_output` has actual `<label>X</label>` strings (NOT triton errors).
   - `model_load_time_s` from `cost_metrics.csv` is < 10 minutes (target).
4. Run `evaluate.py` and confirm macro-F1 matches the existing local Llama-3B `ansible_ansible` zero-shot eval at `results/issues11k/project_specific/ansible_ansible/unsloth_Llama_3_2_3B_Instruct/ragtag/evaluations/eval_zero_shot.csv` to within ±0.01.
5. If all four hold, clean up the sandbox: `rm -rf results/issues11k/_smoketest fetched/smoketest-*` plus a transient pod that wipes `/data/issues11k/_smoketest/` and `/data/_outbox/smoketest-*` from the PVC.

---

## 6. Production campaign — to launch after smoke passes

```bash
cd /home/ahosein/llm-labler
source venv/bin/activate
python scripts/nrp/submit.py --plan scripts/nrp/plan.yaml
```

This:
1. Submits Wave 0 (indexing), kubectl-waits for it to complete (~10 min real work + queue).
2. Submits all 58 Jobs of Waves 1-5 in parallel and returns.

Cluster scheduler runs as many in parallel as there are free A6000s. Expected total wall time: 12-24 hours depending on cluster availability.

Once Jobs start finishing (visible via `kubectl get jobs -l campaign=llm-labler-11k`), the user runs `bash scripts/nrp/sync.sh` (no cron — manual on demand) to pull tarballs back. Each tarball is untarred into `./fetched/<name>/`, `evaluate.py` is auto-run on the prediction CSVs, and the new directories merge into `./results/` at sibling `_v2/` paths (preserving the original broken Qwen-32B RAGTAG / debias outputs per paper-archival rules).

---

## 7. How to resume from any machine

You need three things:
1. **kubectl + kubelogin** for NRP OIDC auth.
2. **Your kubeconfig** (copy of `~/.kube/config` from BGSU machine).
3. **Git clone** of this repo at branch `11k-experiments`.

Then:

```bash
# Verify access
kubectl get pods -n bgsu-cs-heydarnoori    # namespace already set as default in kubeconfig

# See the current state of any in-flight Jobs
kubectl get jobs -l wave=smoke -o wide          # smoke test
kubectl get jobs -l campaign=llm-labler-11k     # production campaign

# Tail logs of a specific Job
kubectl logs -f job/<job-name>

# When ready to pull results back (any time, idempotent)
cd llm-labler && bash scripts/nrp/sync.sh
```

If the BGSU machine is offline or `fetched/.synced` is unavailable, sync.sh will simply re-pull tarballs that were already integrated locally — harmless duplication, no data loss.

---

## 8. Key commits

| SHA | Title |
|---|---|
| `b74eb2d` | Add NRP Kubernetes Jobs migration scripts (initial scaffolding) |
| `1ffc7aa` | Fix container image build for Ubuntu 24.04 + Python 3.12 (CUDA 12.6.3 + venv) |
| `d214fa0` | Bump warmup Job memory 4Gi → 16Gi (Qwen-32B download OOM fix) |
| `ef6bd30` | Add build-essential to image; fix sync.sh binary streaming + pod naming |
| `68a27e6` | Move deadline to pod-level + bump all wave budgets to 20h |

---

## 9. Gotchas worth remembering

- **Pod-level vs Job-level `activeDeadlineSeconds`**: Job-level counts queue time; pod-level only counts runtime. We use pod-level.
- **`llm_labeler.py` skips if output CSV exists.** When re-running on the same path, wipe outputs first.
- **`kubectl logs` corrupts binary streams.** Always base64-encode binary data inside the pod before reading via `kubectl logs`.
- **`pkill -f "<pattern>"` matches the bash script running it** if the pattern appears anywhere in the script's command line. Use a more specific pattern or kill by PID.
- **First image pull on a node is slow** (12-15 GB at variable network speed = 5-25 min depending on node). Subsequent pods on the same node are instant. The slow nodes we hit so far: `suncave-8` (UCSD), `ry-gpu-14.sdsc` (UCSD).
- **Tier S nodeAffinity restricts to RTX-3090/A5000.** When the cluster is busy, all 50+ matching nodes can be allocated and our pods sit in queue. Adding A40 to the pool is a safe escalation if needed (3 free nodes, doesn't compete with tier M).
