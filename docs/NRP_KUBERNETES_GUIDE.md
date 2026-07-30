# Running Jobs on NRP (Nautilus) — Field Guide

> **Audience:** Claude sessions on *any* project of ours that needs to run GPU
> workloads on the NRP (Nautilus Research Platform) Kubernetes cluster.
> This guide is distilled from a real multi-week campaign (LLM inference +
> LoRA fine-tuning of Qwen/Llama models) that ran ~hundreds of GPU-jobs on
> Nautilus. Every "lesson" and "gotcha" below cost us real debugging time —
> read it before you submit anything.
>
> **You do not need access to that project's repo to use this.** The patterns,
> manifests, and pitfalls are reproduced here in full.

---

## 0. TL;DR — the mental model

NRP is a **shared, free, heavily-contended** academic Kubernetes cluster. You
do not get a machine; you submit **Jobs** (batch workloads) that get scheduled
onto a GPU node *whenever one frees up*. Key consequences:

1. **You queue. Possibly for hours.** Plan for it. Never assume a job starts
   when you submit it.
2. **Pods are cattle, not pets.** A pod can be evicted/killed at any time.
   Everything must be **idempotent** and **resumable**, and all output must be
   written to a **PVC** (persistent volume) — local pod disk vanishes.
3. **No long-lived idle pods.** NRP policy forbids `sleep infinity` / idle GPU
   holding. Every pod must do continuous useful work and **exit when done**.
4. **The cluster is the boss.** You cannot SSH in, you cannot `scp`. You move
   data in/out via PVCs + transient pods + `kubectl logs`/`cp`.

If you internalize only one thing: **submit idempotent, self-resuming Jobs that
write to a PVC, and pull results out separately.**

---

## 1. Access & one-time setup

### Identity
- **Cluster / kube-context:** `nautilus`
- **Namespace:** `bgsu-cs-heydarnoori` — *always* pass `-n bgsu-cs-heydarnoori`.
  Forgetting the namespace is the #1 "why do I see no pods?" mistake.
- **Auth:** OIDC via the Nautilus portal. You need `kubectl` **and**
  `kubelogin` (a.k.a. `kubectl-oidc_login`) installed. First `kubectl` call
  opens a browser to authenticate; the token then caches.

### Resuming on a fresh machine
You need exactly three things:
1. `kubectl` + `kubelogin` installed.
2. `~/.kube/config` copied from a machine that already works. Then run any
   `kubectl get pods -n bgsu-cs-heydarnoori` to trigger the browser auth flow.
3. A git clone of the relevant project repo (for the manifests/scripts).

### Sanity check you're wired up
```bash
kubectl config current-context                  # -> nautilus
kubectl -n bgsu-cs-heydarnoori get pods          # should not error
kubectl -n bgsu-cs-heydarnoori get pvc           # see your persistent volumes
```

---

## 2. Persistent storage (PVCs) — where everything lives

Pod-local disk is **ephemeral** and dies with the pod. All durable state lives
on **PersistentVolumeClaims** backed by CephFS (`storageClassName: rook-cephfs`,
`ReadWriteMany` — so multiple pods can mount the same PVC at once).

From our campaign we ran **two** PVCs and the split is a good default pattern:

| PVC | Size / class | Purpose |
|-----|--------------|---------|
| `hf-cache-pvc` | 100Gi rook-cephfs RWX | **Model weight / asset cache.** Big, slow-to-download artifacts (HF model weights, embedding models). Populate **once**, reuse forever. |
| `results-pvc` | 50Gi rook-cephfs RWX | **Outputs.** Predictions, metrics, logs, plus an `_outbox/` subfolder of per-job tarballs staged for pickup. |

Why two? The weight cache is large, write-once, read-many, and you never want to
re-download 60GB of weights per job. Outputs are small, append-mostly, and you
pull them out constantly. Different lifecycles → different volumes.

### Creating a PVC
```yaml
# pvc-results.yaml
apiVersion: v1
kind: PersistentVolumeClaim
metadata:
  name: results-pvc
spec:
  storageClassName: rook-cephfs
  accessModes: [ReadWriteMany]      # RWX is what lets many pods share it
  resources:
    requests:
      storage: 50Gi
```
```bash
kubectl -n bgsu-cs-heydarnoori apply -f pvc-results.yaml
```

### ⚠️ PVC gotchas
- **`ReadWriteMany` (RWX) is essential** if more than one pod ever touches the
  volume concurrently (e.g. a job writing while you pull results). Default RWO
  will silently block the second mount and your pod hangs in `Pending`/
  `ContainerCreating`.
- **You cannot `kubectl cp` from a PVC that isn't mounted by a running pod.**
  To read/write a PVC you must have *some* pod mounting it. See §6 for the
  transient-pod pattern.
- **Quota is shared across the namespace.** Don't request 1Ti "just in case" —
  size to actual need; over-provisioning can block teammates.
- **CephFS is networked storage.** It's durable but not blazing fast for tiny
  random IO. Bulk sequential reads/writes (model weights, tarballs) are fine.

---

## 3. Container image discipline (the #1 source of pain)

Your job runs *your* code inside *your* container image. We host on GHCR
(`ghcr.io/<org>/<image>:<tag>`), public so the cluster can pull without a
pull-secret.

### THE GOLDEN RULE: **SHA-pin every image. Never use `:latest`.**

This bit us repeatedly and is the single most important operational rule:

> A node may have cached an older image under a mutable tag. With
> `imagePullPolicy: IfNotPresent` + `:latest`, the node happily reuses the
> **stale, broken** image and you debug a bug you already fixed. Pin the image
> to the **git commit SHA** (or content digest) so a new build = a new tag =
> a guaranteed fresh pull.

### The build → push → deploy loop (run after EVERY code change)
```bash
SHA=$(git rev-parse --short HEAD)
docker build -f scripts/nrp/Dockerfile -t ghcr.io/<org>/<image>:$SHA .
docker push ghcr.io/<org>/<image>:$SHA
# then update the image: field in your plan/manifest to :$SHA, commit it.
```

### ✅ ALWAYS verify the cluster actually pulled the new image
After pushing, spin up a one-shot pod that greps for the change you just made.
Skipping this step has burned us — the job "ran" but on old code.
```bash
kubectl -n bgsu-cs-heydarnoori run verify-img --rm -it --restart=Never \
  --image=ghcr.io/<org>/<image>:$SHA \
  --command -- grep -n "the_new_string_i_added" /workspace/<proj>/some_file.py
```

### Dockerfile lessons (CUDA + Python ML images)
Real bugs we hit building an Unsloth/transformers/triton image on
`nvidia/cuda:12.6.3-cudnn-runtime-ubuntu24.04`:

- **Triton JIT-compiles CUDA kernels at runtime** → the image needs a C
  compiler. Without `build-essential`, *every inference output* was the literal
  string `ERROR: Failed to find C compiler`. Install `build-essential`.
- **Triton's runtime compile does `#include <Python.h>`** → you need the Python
  dev headers. Install `python3-dev`. Symptom: runtime compile fails looking
  for `Python.h`.
- **Ubuntu 24.04 ships pip/setuptools as apt packages pip won't uninstall.**
  Side-step the whole mess with a venv:
  ```dockerfile
  RUN python3 -m venv /opt/venv
  ENV VIRTUAL_ENV=/opt/venv PATH="/opt/venv/bin:${PATH}"
  ```
- Set HF cache envs in the image so they point at the mounted PVC:
  ```dockerfile
  ENV HF_HOME=/workspace/hf_cache \
      TRANSFORMERS_CACHE=/workspace/hf_cache \
      HUGGINGFACE_HUB_CACHE=/workspace/hf_cache
  ```
- The full GPU ML image was ~12GB. First pull onto a cold node takes minutes;
  budget for it (see deadlines, §5).

---

## 4. Submitting a Job — the anatomy

A Nautilus GPU Job manifest, annotated. This is the skeleton; adapt the
`command`, resources, and node selector.

```yaml
apiVersion: batch/v1
kind: Job
metadata:
  name: my-job                      # must be unique; apply fails if it exists
  labels:
    campaign: my-campaign           # label everything so you can bulk-query/delete
spec:
  backoffLimit: 1                   # how many retries before the Job is "Failed"
  ttlSecondsAfterFinished: 604800   # auto-GC the Job object after 7 days
  template:
    spec:
      restartPolicy: Never
      # ⚠️ POD-level deadline. Clock starts when the container RUNS, not when
      # you submit. Queue time (could be hours) does NOT count against it.
      # This bounds runtime only — exactly what you want.
      activeDeadlineSeconds: 72000
      containers:
        - name: main
          image: ghcr.io/<org>/<image>:<SHA>   # SHA-pinned, always
          imagePullPolicy: IfNotPresent
          command: ["bash", "-c"]
          args:
            - |
              set -euo pipefail
              cd /workspace/<proj>
              echo "[start] $(date -Is) $(hostname)"
              python my_script.py --out /workspace/<proj>/results/...
              echo "[done] $(date -Is)"
          env:
            - name: PYTHONUNBUFFERED      # so logs stream live, not buffered
              value: "1"
            - name: HF_HOME
              value: /workspace/hf_cache
          resources:
            requests:                     # requests == limits is the safe default
              cpu: "8"
              memory: 64Gi
              nvidia.com/gpu: 1
            limits:
              cpu: "8"
              memory: 64Gi
              nvidia.com/gpu: 1
          volumeMounts:
            - name: hf-cache
              mountPath: /workspace/hf_cache
            - name: results
              mountPath: /workspace/<proj>/results
            - name: dshm                  # see gotcha below
              mountPath: /dev/shm
      affinity:                           # pin to GPU types you actually want
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
              - matchExpressions:
                  - key: nvidia.com/gpu.product
                    operator: In
                    values: [NVIDIA-L40, NVIDIA-L40S]
      volumes:
        - name: hf-cache
          persistentVolumeClaim: {claimName: hf-cache-pvc}
        - name: results
          persistentVolumeClaim: {claimName: results-pvc}
        - name: dshm
          emptyDir: {medium: Memory, sizeLimit: 8Gi}
```
```bash
kubectl -n bgsu-cs-heydarnoori apply -f my-job.yaml
```

### Resource sizing & GPU selection
- **GPUs are requested as `nvidia.com/gpu: 1`.** You select *which* GPU model
  via `nodeAffinity` on `nvidia.com/gpu.product`. Real values we used:
  - Small jobs (3B models): `NVIDIA-GeForce-RTX-3090`, `NVIDIA-RTX-A5000`
  - Big jobs (14B/32B 4-bit, LoRA FT): `NVIDIA-L40`, `NVIDIA-L40S`, A6000
- **Pick the smallest GPU that fits.** Smaller/older GPUs (3090, A5000) have far
  shorter queues than the contended big ones (A100, A6000, L40). A 32B 4-bit
  model needs ~23GB VRAM → fits on a 24GB card, but barely; lean to L40 (48GB)
  to avoid OOM.
- **Set `requests == limits`** for cpu/memory to avoid throttling/OOM-kill
  surprises. Don't over-request CPU — it doesn't speed up GPU work and lengthens
  your queue.

### ⚠️ Job-anatomy gotchas
- **`/dev/shm` is tiny by default** in containers (64MB). PyTorch DataLoaders
  and multi-process inference will crash with cryptic "bus error" / shared
  memory errors. Mount an in-memory `emptyDir` at `/dev/shm` (the `dshm` volume
  above) sized to a few GB.
- **POD-level vs JOB-level `activeDeadlineSeconds`.** Put it on the *pod
  template spec* (under `template.spec`), **not** the Job spec. Job-level counts
  queue time against your budget — so a job that queued 6h then ran 2h gets
  killed at an 8h limit before finishing. Pod-level only counts runtime.
- **`backoffLimit` + non-idempotent code = corruption.** If a retried pod
  re-runs from scratch and appends/half-writes, you get garbage. Make the work
  idempotent (skip-if-output-exists) before you allow retries.
- **`restartPolicy: Never`** for batch GPU work — let the Job controller make a
  fresh pod rather than restarting a dirty container in place.

---

## 5. Queue reality & deadlines

- **Per-job queues of 1–3 hours were normal** on the contended big-GPU pools.
  Multi-day if you submit many parallel big-GPU jobs and they all queue.
- **`activeDeadlineSeconds` is a safety bound, not a target.** Set it generously
  (we used 20h–100h for long campaigns) — its only job is to kill a hung/zombie
  pod, not to pace work. Remember it's pod-level so queue time is free.
- **A pod stuck in `Pending`** usually means: no node matches your
  `nodeAffinity`/resources right now (just wait), OR you asked for something
  impossible (too much memory, a GPU type that doesn't exist). Check:
  ```bash
  kubectl -n bgsu-cs-heydarnoori describe pod <pod> | sed -n '/Events/,$p'
  ```
  The Events section tells you *why* it isn't scheduling.

---

## 6. The mega-runner pattern (our most important strategic lesson)

**Problem:** We had ~100 small/medium "cells" of work. Submitting 100 parallel
Jobs meant 100 independent queue waits on contended GPUs → a campaign that was
mostly pods sitting in `Pending` for days.

**Solution — the mega-runner:** submit **one** Job that acquires **one** GPU
**once**, then loops over all cells *sequentially via `subprocess`* inside that
single pod. You pay the queue tax once, then hold the GPU and grind. This is
NRP-policy-compliant because the pod is doing continuous useful work the whole
time (no idle holding).

```python
# run_remaining_cells.py (runs INSIDE the pod)
for cell in enumerate_all_cells():
    if output_already_exists(cell):        # idempotent skip
        continue
    subprocess.run(cell.command, shell=True, cwd=REPO_ROOT)
    tar_outputs_to_outbox(cell)            # stage result for pickup
```

Properties that make this robust:
- **Idempotent skip:** each cell checks "does my output file already exist on
  the PVC?" and skips if so. A killed/restarted pod resumes exactly where it
  left off — no redo, no manual bookkeeping.
- **One queue wait** for the entire campaign instead of N.
- **Per-cell tarball to `_outbox/`** the moment a cell finishes, so results are
  retrievable incrementally — you don't wait for the whole campaign.

**When to use mega-runner vs parallel Jobs:**
- Many small cells, contended GPU → **mega-runner** (serialize, hold one GPU).
- Few large independent cells, GPU readily available → **parallel Jobs** (faster
  wall-clock when you can actually get the GPUs).

A hybrid template-driven submitter (Jinja2 → `kubectl apply`) is handy for the
parallel case; the mega-runner can *reuse the same builders* to enumerate cells
so there's no duplicated command-construction logic.

---

## 7. Getting data IN and OUT (no SSH/scp!)

You can't SSH into the cluster. Data movement is via PVCs + ephemeral pods.

### Populate a cache PVC once (warm-up Job)
Run a one-shot Job that downloads big assets onto the PVC, then exits:
```yaml
# job-warm-cache.yaml — downloads model weights to hf-cache-pvc, then exits
command: ["python", "-c"]
args:
  - |
    from huggingface_hub import snapshot_download
    for m in ["unsloth/Qwen2.5-32B-Instruct-bnb-4bit", "..."]:
        snapshot_download(repo_id=m, cache_dir="/workspace/hf_cache")
# (mounts hf-cache-pvc at /workspace/hf_cache; activeDeadlineSeconds ~3h)
```
After this runs once, every real job reads weights from the PVC — no
re-download. (This is the one place `:latest` is tolerable since it's a manual
one-off, but pinning is still safer.)

### Pull results OUT — the transient-pod pattern
Spin up a tiny `alpine` pod that mounts `results-pvc` **read-only**, have it
emit the file, capture via `kubectl logs`, then delete the pod. Key trick:
**`base64`-encode binary inside the pod** because `kubectl logs` mangles raw
binary; decode locally.

```bash
# 1. list the outbox
kubectl -n bgsu-cs-heydarnoori run lister --rm --restart=Never --image=alpine \
  --overrides='{"spec":{"volumes":[{"name":"r","persistentVolumeClaim":{"claimName":"results-pvc"}}],"containers":[{"name":"main","image":"alpine","command":["ls","/data/_outbox"],"volumeMounts":[{"name":"r","mountPath":"/data"}]}]}}' \
  -it 2>/dev/null

# 2. stream one tarball out, base64-safe
kubectl ... (alpine pod) command: ["base64","/data/_outbox/foo.tar.gz"]
  | base64 -d > ./fetched/foo.tar.gz
gzip -t ./fetched/foo.tar.gz        # ALWAYS verify integrity before trusting it
```

In practice, wrap this in a local `sync.sh` cron job (every 15 min) that:
1. lists `_outbox/` via a transient pod,
2. for each tarball not already pulled (track a `.synced` ledger), streams it
   out base64-safe, **verifies gzip integrity**, untars, processes, and deletes
   the pod.
- Use a **lock file (`flock`)** so overlapping cron ticks don't collide.
- Make pod names **k8s-valid**: lowercase alphanumeric + dashes, must start/end
  alphanumeric, ≤63 chars. Sanitize derived names or `apply` rejects them.
- The `.synced` ledger is **per-machine** — a fresh laptop re-pulls everything
  (harmless duplication, no data loss).

### `kubectl cp` (simpler, for interactive one-offs)
If you have a *running* pod mounting the PVC, `kubectl -n <ns> cp
<pod>:/path/file ./file` works. But for batch/automated pulls the transient-pod
+ base64 pattern is more robust (no long-lived pod required).

---

## 8. Observability & control cheat-sheet

```bash
NS=bgsu-cs-heydarnoori

# What's running / queued
kubectl -n $NS get jobs,pods
kubectl -n $NS get jobs -l campaign=my-campaign --sort-by=.metadata.creationTimestamp

# Why is my pod not scheduling? (read the Events section)
kubectl -n $NS describe pod <pod>

# Live logs
kubectl -n $NS logs -f job/<job-name>

# Filter to milestones only (skip chatty warnings)
kubectl -n $NS logs job/<job> | grep -E "^---|SKIP|OK in|FAILED|SUMMARY|done"

# Stop a job cleanly
kubectl -n $NS delete job <job-name>

# Resume after a fix: rebuild image, update SHA, re-apply.
# Idempotent skip means already-done work is not redone.
```

### ⚠️ Control gotchas
- **`pkill -f "submit.py"` self-immolation:** if the script calling `pkill` has
  the pattern in its *own* command line, `pkill` matches and kills itself. Kill
  by **PID**, not pattern, from inside a wrapper.
- **`kubectl logs` only shows the *current* pod.** If a Job made a new pod after
  a retry, you may be looking at the wrong one. Use `kubectl get pods` and target
  the specific pod, or `kubectl logs job/<name>` (shows one of them).
- **`kubectl wait --for=condition=complete job/<x>`** blocks until done — great
  for scripting a dependency (job B reads job A's output), but remember it also
  counts queue time on *your* wall clock even though the pod deadline doesn't.

---

## 9. Output hygiene (don't fill the PVC with junk)

- **HF `Trainer` saves checkpoints + a final adapter by default.** We were
  shipping 633MB tarballs of artifacts we never reloaded. Set
  `save_strategy="no"` (and skip saving the final adapter if you don't reload
  it). Result: 633MB → 113KB per cell. On a shared 50Gi PVC this matters.
- **Tar each cell's output to `_outbox/` and treat `results/` as archival.**
  When integrating pulled tarballs locally, `rsync -a --ignore-existing` so you
  never overwrite already-good data.
- **Be careful that your "eval/post-process" step writes to the right path.**
  Different job types had different output layouts (e.g. predictions under
  `predictions/` vs directly in a folder); a path assumption baked into the
  sync script silently mis-filed outputs. Detect the layout, don't assume.

---

## 10. The pre-flight checklist (run this every time)

Before submitting any real campaign:

- [ ] `kubectl config current-context` is `nautilus` and you're passing
      `-n bgsu-cs-heydarnoori`.
- [ ] Image is **SHA-pinned** to the commit you actually want, pushed to GHCR,
      and you **verified the pull** with a grep pod.
- [ ] Code is **idempotent** — re-running skips completed work (check for
      existing output, don't blindly recompute/append).
- [ ] All durable output goes to a **PVC mount**, not pod-local disk.
- [ ] `activeDeadlineSeconds` is on the **pod template**, set generously.
- [ ] Resources: smallest GPU that fits, `requests == limits`, `/dev/shm`
      mounted if you use DataLoaders / multiprocessing.
- [ ] Jobs are **labeled** (`campaign=...`) for bulk query/delete.
- [ ] You have a **pull path** (sync script / transient-pod) ready to retrieve
      results, with gzip integrity checks.
- [ ] For many small cells on contended GPUs: use the **mega-runner**, not N
      parallel Jobs.

---

## 11. Hall of fame — bugs that cost us real time

Each of these is a *class* of failure. If you hit something weird, scan here first.

| Symptom | Root cause | Fix |
|---|---|---|
| Every model output is `ERROR: Failed to find C compiler` | Triton JIT needs a C compiler at runtime | Add `build-essential` to the image |
| Runtime compile fails on missing `Python.h` | Triton includes `<Python.h>` | Add `python3-dev` |
| Bug you already fixed still happens in-cluster | Stale `:latest` image cached on the node | **SHA-pin** images; verify pull with a grep pod |
| Pod stuck `Pending` forever | No node matches affinity/resources, or impossible request | `describe pod` → read Events; relax affinity / shrink request |
| "Bus error" / shared-memory crash in PyTorch | `/dev/shm` default 64MB too small | Mount in-memory `emptyDir` at `/dev/shm` |
| Job killed mid-run despite generous limit | `activeDeadlineSeconds` at Job level counted queue time | Move it to **pod template** spec |
| Corrupted tarball after pull | `kubectl logs` mangled raw binary | `base64` in pod, decode locally, `gzip -t` verify |
| 600MB tarballs of nothing useful | HF Trainer default checkpoint saves | `save_strategy="no"`, skip adapter save |
| Outputs filed in wrong directory | Path layout assumption in post-process | Detect layout per job type, don't assume |
| Control script killed itself | `pkill -f pattern` matched its own cmdline | Kill by PID, not pattern |
| Campaign mostly `Pending` for days | N parallel Jobs = N independent queue waits | Mega-runner: one Job, one GPU, sequential cells |

---

*This guide is portable. Replace `<org>`, `<image>`, `<proj>`, model lists, and
GPU types with your project's specifics, but keep the patterns and the
checklist — they are the parts that were expensive to learn.*
