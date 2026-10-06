# Compute Resources: Machines, Clusters, Access, Gotchas

**Read this at the start of every session.** It is the cross-machine, cross-session source of
truth for where things run and how to reach them. Claude's auto-memory (`~/.claude/projects/...`)
is per-machine and does **not** travel with `git pull`; this file does.

Last verified: 2026-10-06 from the old local PC. OSC was checked live. bgsulab could not be
checked that day because the VPN was down, so its facts date from 2026-09-16 to 2026-09-25.

The other project that shares these resources is LinkAnchor (`my-la`). It uses the same OSC
account, NRP namespace and lab box. Its cluster docs are `my-la/infra/osc/README.md`,
`my-la/docs/project_documentation/SESSION_HANDOFF.md` ("KEY FACTS / GOTCHAS") and
`my-la/docs/project_documentation/reasoning-only-baselines.md` (NRP lessons).

---

## 0. First thing a session should do: check this machine's state

Run these four probes. Each one says whether a resource is wired up on the current machine.

```bash
git -C . remote -v | head -1                                      # repo cloned?
timeout 15 ssh -o BatchMode=yes -o ConnectTimeout=8 bgsulab hostname    # -> heydarnoori (needs VPN)
timeout 30 ssh -o BatchMode=yes -o ConnectTimeout=8 cardinal hostname 2>/dev/null  # -> cardinal-login0X...
kubectl config current-context 2>&1                               # -> nautilus
```

| Probe result | Meaning | Fix |
|---|---|---|
| `bgsulab`: `Could not resolve hostname bgsulab` | No ssh alias on this machine | §2.3 |
| `bgsulab`: `Connection timed out` | The alias exists but the VPN is down (the usual case) | Ask the user to run `bgsu-vpn` (§2.4). **Claude cannot do this**: it needs an interactive BGSU SSO + Duo login in a browser window |
| `bgsulab`: `Permission denied (publickey)` | The key is missing or not authorized | §2.2 |
| `cardinal`: `Permission denied` | No OSC key on this machine | §2.2 and §4.2 |
| `kubectl`: `command not found` / no `nautilus` context | NRP is not set up | §2.5 |
| `kubectl` hangs | OIDC token expired; it waits on a browser login | `kubectl oidc-login clean`, then rerun and finish the browser flow |

**If this is a new machine and anything above fails, tell the user what is missing and point
them to §2 before proposing any GPU work.** Never assume the VPN is up. Always probe it first.

---

## 1. The map

| Resource | What it is | Use it for | Reach it from |
|---|---|---|---|
| **Old local PC** | Ubuntu 20.04, GTX 1650 Max-Q 4 GB. Clone at `~/Desktop/my_projects/automated-issue-report-labeling` | Writing, docs, git, small CPU analyses. **No LLM inference** | — |
| **New laptop** (from 2026-10) | Set it up with §2 | Same role as the old PC unless it has a big GPU | — |
| **bgsulab** | BGSU lab box `heydarnoori`, RTX 4090 24 GB | All original experiments; **the only copy of `results/` (81 GB)** | Laptop, **through the BGSU VPN only** |
| **OSC Cardinal** | Slurm, H100-94GB (one GPU = a quarter node) | Fast or big single-GPU jobs (32B, new models) | Laptop, directly (no VPN) |
| **OSC Ascend** | Slurm, A100-40GB (`preemptible-nextgen`), A100-80GB (`quad`) | Many parallel 1-GPU shards | Submit from the Cardinal login node with `-M ascend` |
| **NRP / Nautilus** | Shared Kubernetes cluster, L40/L40S/A6000/3090/A5000 | 14B/32B LoRA FT (done in 2026-04/05); slow hedge only | Laptop or bgsulab via `kubectl` |
| **GitHub / GHCR** | `git@github.com:doctorhoseinpour/automated-issue-report-labeling.git`; image `ghcr.io/doctorhoseinpour/llm-labler:<sha>` | Code sync; NRP container images | Everywhere |

### Who can reach whom (the relay rule)

```
           VPN            direct ssh
bgsulab <------- laptop -----------> OSC (cardinal / ascend)
   |               |
   +--- kubectl ---+---> NRP
bgsulab  X  OSC      (they cannot reach each other in either direction)
```

**bgsulab and OSC cannot talk to each other.** Every bgsulab-to-OSC transfer is relayed through
the laptop with two `rsync`s, as `scripts/experiments/newllms/osc/pull_nm.sh` and
`scripts/experiments/rag_next/osc/sync_to_osc.sh` do. Stage the files in a scratch dir on the
laptop, never inside the repo.

### Where the data lives

| Data | Location | In git? |
|---|---|---|
| Code, paper sources, study notebooks (`docs/RAG_NEXT_STUDY.md`, `docs/NEWLLMS_STUDY.md`) | Repo | Yes |
| `issues11k.csv` (source pool) | Repo | Yes (a `.gitignore` exception) |
| Train/test splits, FAISS indices | Regenerated deterministically on first run | No |
| `results/issues11k/{agnostic,project_specific}/` (paper archival) | `bgsulab:~/llm-labler/results/` | No (gitignored, 81 GB) |
| RAG-next and new-LLM study outputs: test preds, evals, `test_eval_log.csv`, splits, logs (65 MB, big CSVs gzipped) | `exploration_results/` in the repo (copied 2026-10-06, SHA-256 verified; see its README) and the originals on bgsulab | **Yes** |
| RAG-next hidden-state features (2.5 GB), SetFit dev models (14 GB) | `bgsulab:~/llm-labler/results/issues11k/exploration/rag_next/{features,setfit_dev}/` | No |
| New-LLM study features (6.2 GB) and raw shard outputs (5.6 GB) | Both on `bgsulab:~/llm-labler/results/issues11k/exploration/newllms/{features,raw}/`; the raw shards also on OSC `~/nm/repo/results/issues11k/exploration/newllms/raw/` | No |
| Centered-retrieval probe preds | `bgsulab:~/center_probe_20260924/preds_centered/` | No |
| 32B decision-state features | Extracted on OSC under `/fs/ess/PCS0289/rag_next/repo/results/...`, copied to bgsulab | No |
| Qwen2.5-14B/32B bnb, Qwen3.5-9B, Gemma-4-12B-it, Ministral-3-8B weights | OSC `/fs/ess/PCS0289/rag_next/hf_cache` | No |
| NRP weights and outputs | PVCs `hf-cache-pvc`, `results-pvc` (namespace `bgsu-cs-heydarnoori`) | No |

`results/` is paper-archival. Never delete, move or rename anything in it; superseded runs go to
`archive/`. Copy only what a script needs: `rsync -av bgsulab:~/llm-labler/results/<path> <local scratch>`.

---

## 2. Setting up a new machine (laptop)

Do these in order. Steps 2.1 and 2.7 are all that paper writing needs. bgsulab needs 2.2–2.4,
OSC needs 2.2–2.3, NRP needs 2.5.

### 2.1 Repo

```bash
git clone git@github.com:doctorhoseinpour/automated-issue-report-labeling.git
cd automated-issue-report-labeling
git checkout encoder-baselines          # the SANER 2027 submission line (26eb20b, submitted 2026-09-25)
```

Branches to know about:

| Branch | What it holds |
|---|---|
| `encoder-baselines` | Active line. SANER 2027 submission (`26eb20b`), both post-deadline study notebooks and their code (`scripts/experiments/{rag_next,newllms}/`) |
| `saner-reframe` | Reframe sessions S1–S7 (`docs/reframe/`). **It diverges from `encoder-baselines`**: 27 commits are not in it and 6 are not in this one. Its old worktree was `~/Desktop/my_projects/saner-reframe` on the old PC |
| `main` | Old (pre-NRP). Do not base work on it |
| Tags | `saner-draft-2026-09-25` (draft before the reframe), `pre-nrp-migration` |

Clone to the **same absolute path as the old PC** if you can
(`~/Desktop/my_projects/automated-issue-report-labeling` under the same username). Claude's
memory folder name is derived from that path (see §2.8).

### 2.2 SSH keys

Three different services trust three keys on the old PC:

| Service | Key on the old PC | Authorized where |
|---|---|---|
| GitHub (`doctorhoseinpour`) | default key | GitHub → Settings → SSH keys |
| bgsulab (`ahosein@192.168.198.25`) | `~/.ssh/id_ed25519_other` | `bgsulab:~/.ssh/authorized_keys` |
| OSC (`alirezzzhp1378`) | `~/.ssh/id_ed25519` | `~/.ssh/authorized_keys` in the OSC home (shared by Cardinal and Ascend) |

Recommended: make a new key on the laptop and authorize it from the old PC, which still has access.

```bash
# on the laptop
ssh-keygen -t ed25519 -C "laptop-$(date +%Y%m)" -f ~/.ssh/id_ed25519
# copy ~/.ssh/id_ed25519.pub to the old PC, then on the old PC (VPN up for bgsulab):
cat laptop.pub | ssh bgsulab 'cat >> ~/.ssh/authorized_keys'
cat laptop.pub | ssh alirezzzhp1378@cardinal.osc.edu 'cat >> ~/.ssh/authorized_keys'
# and add it to GitHub in the browser
```

You can also copy the private keys from the old PC over a trusted channel and `chmod 600` them.
If neither machine is at hand, OSC also accepts password login and has a browser portal
(OnDemand, `ondemand.osc.edu`) with a shell, where you can paste the public key.

### 2.3 `~/.ssh/config`

```sshconfig
# BGSU CompSci lab machine (Dr. Heydarnoori's server). VPN first (`bgsu-vpn`).
# There is no DNS on that network, so the IP is the only way to reach it.
Host bgsulab
  HostName 192.168.198.25
  User ahosein
  IdentityFile ~/.ssh/id_ed25519          # whichever key is authorized there
  IdentitiesOnly yes
  ServerAliveInterval 30                   # the VPN drops idle connections; keeps
  ServerAliveCountMax 6                    # long sessions and the VS Code server alive
  # LocalForward 8888 localhost:8888       # remote Jupyter, if ever needed
  # ForwardAgent yes                       # off by default: root on the box could borrow keys

Host cardinal
  HostName cardinal.osc.edu
  User alirezzzhp1378
  IdentityFile ~/.ssh/id_ed25519

Host ascend
  HostName ascend.osc.edu
  User alirezzzhp1378
  IdentityFile ~/.ssh/id_ed25519
```

Gotchas:
- The old PC also has a `Host osc` alias pinned to the IP `192.148.247.176`. That IP is not one
  of Cardinal's current DNS addresses (`192.148.247.180–185`). Use hostnames, not IPs, for OSC.
- The repo's scripts hard-code `alirezzzhp1378@cardinal.osc.edu` rather than an alias, so they
  work without the `cardinal` alias but still need the key.
- The OSC login prints a long "authorized users only" banner. Scripts that parse ssh output
  should use `-o BatchMode=yes` and `2>/dev/null`.

### 2.4 BGSU VPN (needed for bgsulab only)

- Portal: **`csvpn.bgsu.edu`**. It is the CompSci GlobalProtect VPN (Palo Alto), not the general
  campus VPN.
- Auth is **SAML SSO** (login.bgsu.edu → Okta → Duo push). There is no GlobalProtect client
  for Linux on the portal, and a username/password `openconnect` login cannot work.
- The working recipe on the old PC is `openconnect` (GlobalProtect protocol) plus
  [`gp-saml-gui`](https://github.com/dlenski/gp-saml-gui). gp-saml-gui opens a small browser
  window for the BGSU + Duo login, grabs the cookie and hands it to `sudo openconnect`.

Install (Ubuntu 22.04/24.04; the old PC is 20.04 and uses `gir1.2-webkit2-4.0`):

```bash
sudo apt install openconnect python3-gi gir1.2-gtk-3.0 gir1.2-webkit2-4.1 pipx
pipx install --system-site-packages "git+https://github.com/dlenski/gp-saml-gui"
# (--system-site-packages lets it see the apt PyGObject; on 20.04 the old PC used `pip3 install --user`)
```

Then create `~/.local/bin/bgsu-vpn` (`chmod +x`). This is the exact script from the old PC:

```bash
#!/usr/bin/env bash
# Connect to the BGSU CompSci VPN (csvpn.bgsu.edu). Usage: bgsu-vpn ; disconnect: Ctrl-C.
set -euo pipefail
PORTAL=csvpn.bgsu.edu
export PATH="$HOME/.local/bin:$PATH"
# Gateway mode, not portal mode. csvpn.bgsu.edu is nominally the portal, but its SAML login
# returns a "prelogin-cookie" -- a gateway-interface credential. Sent as portal:prelogin-cookie
# it gets consumed by the portal step and the gateway login then fails with HTTP 512.
MODE=--gateway
for arg in "$@"; do
    case "$arg" in
        -g|--gateway|-p|--portal) MODE=""; break ;;   # caller chose a mode
    esac
done
# Caller args go BEFORE the server (gp-saml-gui options); args after it go to openconnect.
exec gp-saml-gui ${MODE:+"$MODE"} --sudo-openconnect "$@" "$PORTAL"
```

VPN gotchas:
- **Use gateway mode** (`--gateway`). Portal mode fails with **HTTP 512** (see the script comment).
- It asks for your sudo password (openconnect needs root for the tun device) after the browser login.
- Leave the terminal open. Ctrl-C disconnects. Idle connections drop, hence the ssh keep-alives.
- There is no DNS for lab hosts. Use `192.168.198.25` (the `bgsulab` alias).
- gp-saml-gui prints "Using WebKit2Gtk 4.0 (obsolete)" on 20.04. This is harmless.
- If gp-saml-gui will not install on a new distro, try the alternatives: openconnect ≥ 9 with
  `--protocol=gp` and its `--external-browser` SAML support, or the `yuezk/GlobalProtect-openconnect`
  GUI client. Neither has been tested against csvpn here, and both still need gateway mode.

### 2.5 NRP / Nautilus (`kubectl`)

You need `kubectl`, the OIDC plugin and a kubeconfig.

```bash
# kubectl: https://kubernetes.io/docs/tasks/tools/  (old PC: /usr/local/bin/kubectl)
# kubelogin plugin, installed as `kubectl-oidc_login` on PATH (old PC: /usr/local/bin/kubectl-oidc_login)
#   either: kubectl krew install oidc-login
#   or:     download the int128/kubelogin release binary, rename it to kubectl-oidc_login
# kubeconfig: copy ~/.kube/config from the old PC (or from bgsulab:~/.kube/config),
#   or download a fresh one from the NRP portal (nrp.ai docs, "Getting started").
kubectl -n bgsu-cs-heydarnoori get pods     # first call opens a browser for the OIDC login
```

- Context `nautilus`, user `oidc`, default namespace `bgsu-cs-heydarnoori`. Always pass `-n`
  anyway; a missing namespace is the classic "why do I see no pods" mistake.
- The kubeconfig holds an OIDC client config, not a long-lived token. Tokens are cached in
  `~/.kube/cache/` per machine, so a new machine always does one browser login.
- **An expired token makes `kubectl` hang** while it waits for a browser. Fix:
  `kubectl oidc-login clean`, rerun, finish the flow. In scripts, wrap `kubectl` in `timeout`.

### 2.6 OSC account

No setup beyond ssh (§2.2–2.3). The account is `alirezzzhp1378` on project **`PCS0289`**
(PI: Dr. Heydarnoori). Cardinal and Ascend share one `$HOME`.

### 2.7 LaTeX (paper builds)

`SANER2027/build.sh` runs `pdflatex → bibtex → pdflatex ×2` and then `pdfinfo` for the page
count. Install TeX Live (`texlive-latex-extra texlive-fonts-recommended texlive-science` or
`texlive-full`) and `poppler-utils`. `IEEEtran.cls`/`.bst` are vendored in `SANER2027/`, so
`texlive-publishers` is not needed. `latexmk` is not used. The old PC has TeX Live 2019.

### 2.8 Claude Code memory (does not travel with git)

Auto-memory lives in `~/.claude/projects/<slug>/memory/`. The slug is the clone's absolute path
with `/` and `_` replaced by `-`, for example
`-home-alireza-Desktop-my-projects-automated-issue-report-labeling`. To carry the memories over:

```bash
# on the old PC
tar czf claude-memory.tgz -C ~/.claude/projects \
  ./-home-alireza-Desktop-my-projects-automated-issue-report-labeling/memory \
  ./-home-alireza-Desktop-my-projects-my-la/memory
# on the laptop: untar into ~/.claude/projects/, renaming the folders if the clone path differs
```

The repo docs (this file, `CLAUDE.md`, `docs/*.md`) carry what a session needs without the
memories. The memories add user preferences and decisions (paper style, framing rules).

### 2.9 Python on the laptop

None is needed for writing. For a CPU-side analysis, use `python3 -m venv venv && venv/bin/pip
install -r requirements.txt`. Do not try GPU work on a laptop GPU with less than 24 GB.

---

## 3. bgsulab (the BGSU lab machine)

| | |
|---|---|
| Access | `ssh bgsulab` → `ahosein@192.168.198.25`, key auth, **VPN required** |
| Hostname | `heydarnoori` (Dr. Heydarnoori's server, shared with other lab members and projects) |
| OS / CUDA / Python | Ubuntu 24.04.2, CUDA 12.0 (`nvcc`), Python 3.12.3 |
| GPU | 1× RTX 4090, 24 GB |
| Disk | 1.9 TB NVMe, ~990 GB free (2026-09) |
| **Project path** | **`~/llm-labler`** (= `/home/ahosein/llm-labler`). Spelled **without the second "e"**, even though people say "llm-labeler" |
| Envs | `~/llm-labler/venv/` (Unsloth, FAISS, transformers 5.5.0, torch 2.10.0) and `~/llm-labler/venv-setfit/` (pinned SetFit, `requirements-setfit.txt`) |
| `results/` | 81 GB, **the only copy** |
| Other local-only dirs | `esem/`, `paper/main.pdf`, `canary_openai/`, `fetched/` (NRP pulls + `.synced` ledger), `logs/`, `unsloth_compiled_cache/`, `.faiss_cache/`, legacy `issues3k.csv`/`issues30k.csv`, `archive/` |
| NRP | `~/.kube/config` is here too; `scripts/nrp/sync.sh` was run from here by cron |

Typical loop:

```bash
ssh bgsulab
cd ~/llm-labler && git pull            # same remote; keep its branch in step with the laptop
source venv/bin/activate
tmux new -s <descriptive-name>          # anything over a few minutes runs in tmux or nohup
# ... run ...
# back on the laptop: bring small outputs over
rsync -av bgsulab:~/llm-labler/<path> <local path>
```

Gotchas:
1. **The 4090 is shared.** Other sessions and lab members use it. Check `nvidia-smi` before
   starting, and queue GPU jobs with `scripts/experiments/rag_next/gpu_queue.sh`, which waits for
   120 s of GPU idleness before each job and is idempotent through a done-file. Run it in tmux:
   `tmux new-session -d -s ragnext "bash ~/llm-labler/scripts/experiments/rag_next/gpu_queue.sh"`.
2. **Other projects' tmux sessions live here.** LinkAnchor runs GPT-4o-mini arms in tmux sessions
   named `lk-<run-id>`. Kill tmux sessions **by name only**. Never `tmux kill-server` and never
   `pkill python`.
3. 24 GB fits Qwen2.5-32B bnb-4bit inference (~23 GB) only barely. 32B jobs went to NRP/OSC.
   LoRA FT of 14B/32B does not fit; it ran on NRP.
4. The rag_next scripts import their siblings. Run them from `scripts/experiments/rag_next/` with
   `../../../venv/bin/python`. Outputs go to `results/issues11k/exploration/rag_next/`.
5. Unsloth's bare base model is not causal without padding (pitfall in `RAG_NEXT_STUDY.md` §4.3).
6. The box cannot reach OSC (§1, relay rule).
7. Its git working tree can hold work the laptop has not seen. Before relying on the remote,
   run `git status` there (it could not be checked on 2026-10-06).

---

## 4. OSC (Ohio Supercomputer Center): Cardinal + Ascend

### 4.1 Account, money, storage

| | |
|---|---|
| User / project | `alirezzzhp1378` / **`PCS0289`** (shared by the whole Heydarnoori lab) |
| Budget | `OSCusage` shows it. **$248.15 remaining on 2026-10-06** (about $500 on 2026-09-24). It is shared with lab members, so spend deliberately. A quarter-node H100 is about $0.11/h on Cardinal; CPU jobs cost cents |
| `$HOME` | `/users/PCS0289/alirezzzhp1378`. 500 GB / 1M-file quota (112 GB and 322k files used on 2026-10-06). Shared across clusters. `quota -s` shows usage |
| Project storage | `/fs/ess/PCS0289`. 2 TB, but a **200k-inode limit for the whole lab, at 97% (193,196) on 2026-10-06**. Only big, few-file data (HF weights) goes here. **A venv there fails with "Disk quota exceeded"** |
| Our dirs on `/fs/ess` | `rag_next/` (this project), `linkanchor/` (my-la), `lp2-ygg/`. Leave other lab members' dirs alone |

### 4.2 Access

- `ssh alirezzzhp1378@cardinal.osc.edu` (key auth, no VPN). Submit Ascend jobs **from Cardinal**
  with `sbatch -M ascend ...`, and query them with `squeue -M ascend`, `sacct -M ascend`.
  `ascend.osc.edu` also exists; the old PC never sshed to it directly.
- Login nodes are `cardinal-login0X.hpc.osc.edu`.

### 4.3 Capacity (measured 2026-09-24)

| Where | GPU | Limits / queue |
|---|---|---|
| Cardinal `debug` | H100-94GB | Max 2 running jobs × 1 GPU per user, 1 h. Starts fast |
| Cardinal `gpu` (regular) | H100-94GB | Walltime cap 7 days; **queue estimates of about 10 days** on 2026-09-24. Low fairshare means short jobs do **not** reliably backfill sooner |
| **Ascend `preemptible-nextgen`** | **A100-PCIE-40GB**, 2 per node | **Starts immediately, many parallel 1-GPU jobs** (QOS 96 GPUs per user), 1-day max. `PreemptMode=CANCEL`: jobs are **cancelled, not requeued**. The 31 + 26 newllms shards ran with no preemption |
| Ascend `debug-nextgen` | A100 | 2 jobs × 1 GPU, immediate. Good for smoke tests |
| Ascend `nextgen` / `quad` | A100-40GB / A100-80GB | 15 h+ queues |

`sbatch --test-only ...` prints the estimated start time without submitting.
`DefMemPerCPU` is about 4 GB, so request **12 CPUs to get about 48 GB RAM** (`--ntasks-per-node=12`).

### 4.4 Ready environments and data (on 2026-10-06)

| Path | What |
|---|---|
| `~/nm/venv-nm` | Portable venv on a **uv-managed CPython 3.12** (`~/nm/uv-python`, bootstrap uv in `~/nm/uvboot`). torch 2.10.0+cu128, transformers 5.17.0, bnb 0.50.2, accelerate 1.15.0, flash-linear-attention 0.5.2. Runs Qwen3.5 / Gemma 4 / Ministral 3 on **both** Cardinal and Ascend. Built by `scripts/experiments/newllms/osc/setup_nm.sbatch` |
| `/fs/ess/PCS0289/rag_next/venv` | Unsloth stack pinned to the lab venv (`scripts/experiments/rag_next/osc/requirements-osc.txt`). Built on `/apps/python/3.12`, so **Cardinal only** |
| `/fs/ess/PCS0289/rag_next/hf_cache` | `unsloth/Qwen2.5-{14B,32B}-Instruct-bnb-4bit`, `Qwen/Qwen3.5-9B`, `google/gemma-4-12B-it`, `mistralai/Ministral-3-8B-Instruct-2512-BF16` |
| `~/nm/repo` | Synced code + `results/issues11k/exploration/newllms/{raw,splits}` (5.7 GB) |
| `/fs/ess/PCS0289/rag_next/repo` | Synced code + rag_next splits/features (32B extraction) |
| Logs | `~/nm/logs/%x-%j.out`, `/fs/ess/PCS0289/rag_next/logs/` |

### 4.5 Workflow (driven from the laptop)

```bash
# 1. code (+ inputs relayed from bgsulab via a laptop scratch dir) -> OSC
bash scripts/experiments/newllms/osc/sync_nm_to_osc.sh <scratch-dir>
# 2. one-time CPU jobs: env + weights (never on the login node)
ssh cardinal 'sbatch ~/nm/repo/scripts/experiments/newllms/osc/setup_nm.sbatch'
ssh cardinal 'sbatch ~/nm/repo/scripts/experiments/newllms/osc/prefetch_nm.sbatch Qwen/Qwen3.5-9B'
# 3. smoke test on debug, then shard the real runs on Ascend preemptible
ssh cardinal 'sbatch -M ascend --partition=debug-nextgen --time=00:30:00 --job-name=nm-smoke \
   ~/nm/repo/scripts/experiments/newllms/osc/gpu_nm.sbatch smoke.py --tag qw35_9b'
bash scripts/experiments/newllms/osc/watch_nm.sh runs_main.txt   # relaunch loop, every 3 min, until all shards are done
# 4. pull OSC -> laptop scratch -> bgsulab (VPN up)
bash scripts/experiments/newllms/osc/pull_nm.sh <scratch-dir>
```

The pattern that makes preemptible Ascend safe has three parts. Each shard writes a
`done_XXofYY.json` marker. `launch_nm.sh` submits only shards with no marker and no
queued/running job of the same name, and gives up after 2 failures in a day. `watch_nm.sh` reruns
it from the laptop. Reuse this for any new sharded campaign.

### 4.6 OSC gotchas (each one cost real time)

1. **Login nodes kill any process past 20 CPU-minutes.** Installs, venv builds, model downloads
   and Apptainer pulls must run as CPU batch jobs (`setup_*.sbatch`, `prefetch_nm.sbatch`).
   Compute nodes have internet access.
2. **A stray pip `torch` in `~/.local/lib/python3.*`** shadows any venv or container torch. It
   cost LinkAnchor a day of zero-result runs (`undefined symbol: c10::ivalue::ConstantString::create`).
   Always `export PYTHONNOUSERSITE=1`. With Apptainer, also set `APPTAINERENV_PYTHONNOUSERSITE=1`
   and use `apptainer run --cleanenv`.
3. **A venv built on `/apps/python/3.12` exists on one cluster only.** For one venv that works
   on Cardinal and Ascend, use a uv-managed Python with `UV_PYTHON_INSTALL_DIR` under `$HOME`
   (see `setup_nm.sbatch`).
4. **Jobs sent from Cardinal to Ascend inherit Cardinal's Intel module env (`CC=.../icc`).**
   Triton kernels (e.g. flash-linear-attention) then fail to compile. Put this in the job script:
   `export CC=/usr/bin/gcc CXX=/usr/bin/g++; unset FC F77 F90 LD_LIBRARY_PATH`.
5. **`sbatch` exports the submitting shell's env by default (`--export=ALL`).** Pass per-job
   values with `--export=ALL,VAR=...`. A pending job's env cannot be read back, but its
   positional arguments can be: `sacct -j <id> -X --format=SubmitLine%200`.
6. **`/fs/ess` inode limit** (§4.1). Keep venvs, code and outputs with many small files in `$HOME`.
7. Keep the HF cache on `/fs/ess` and set `HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1` in GPU jobs,
   so they never spend GPU-hours downloading. Put the Triton cache in `$HOME` or `$TMPDIR` and
   `XDG_CACHE_HOME` in `$TMPDIR`.
8. Unsloth writes `unsloth_compiled_cache/` into the cwd, so `cd` into the repo copy first.
9. **Cross-cluster `--dependency` does not work** (Cardinal job → Ascend job). Poll for a marker
   file instead (LinkAnchor's `panel.sbatch` does this).
10. **Preemptible = cancelled.** No requeue, so no resume unless the code checkpoints. Shard small
    and use the idempotent relaunch loop (§4.5).
11. Apptainer (LinkAnchor only so far): put `APPTAINER_CACHEDIR` in `$TMPDIR`; `mksquashfs` sizes
    its buffers from the node's *physical* RAM, so a lean `--mem` gets OOM-killed during
    `apptainer pull` (the stage job uses 24 cores / 280 G); `/fs/scratch` and `$TMPDIR` need
    explicit `--bind`.
12. vLLM under Apptainer shares the host network namespace. Co-scheduled jobs on one node all
    bind `:8000`, and an orphaned server holds its port (`Errno 98`). vLLM also binds API port + 1.
    Pick a free port per job and space port seeds at least 2 apart. Never `pkill vllm`; it can
    kill a co-tenant's leg.
13. Cross-hardware check: 14B decision-state features from the H100 and the 4090 agree (cosine
    ≥ 0.9987, 100% prediction agreement on 90 issues), so OSC and bgsulab outputs can be mixed
    (`RAG_NEXT_STUDY.md` §4.9).

---

## 5. NRP (Nautilus)

Full field guide: [docs/NRP_KUBERNETES_GUIDE.md](NRP_KUBERNETES_GUIDE.md). Bug log:
[docs/NRP_MIGRATION_STATUS.md](NRP_MIGRATION_STATUS.md). Summary and cross-project rules:

| | |
|---|---|
| Context / namespace | `nautilus` / **`bgsu-cs-heydarnoori`**, shared with the whole lab and with LinkAnchor |
| Auth | OIDC (Authentik) through `kubectl oidc-login` (§2.5) |
| **Our PVCs** | `hf-cache-pvc` (100 Gi, ~60 GB of weights) and `results-pvc` (50 Gi, with `_outbox/`), both `rook-cephfs` RWX. **LinkAnchor's PVCs** are `linkanchor-data-pvc` and `linkanchor-hf-cache-pvc`. Each project touches only its own, and **PVCs are never deleted without asking** |
| Image | `ghcr.io/doctorhoseinpour/llm-labler:<sha>` (public on GHCR). The current pin in `scripts/nrp/plan.yaml` is `6570030`. Pushing needs a GitHub PAT with `write:packages` |
| GPUs | Premium GPUs (A100/H100/H200/GH200) have **quota 0**. The 48 GB L40/L40S/A6000 are heavily contended (1–3 h per-job queues; LinkAnchor once waited about 19 h). 24 GB 3090/A5000 queue short. L40 pods ran about 5× slower than OSC's H100 for LinkAnchor |
| Status | Idle since the 2026-05 FT campaign (last known; not re-checked on 2026-10-06) |

Rules and gotchas:
1. **Be a good tenant, or the lab risks eviction.** After any campaign, delete every Job, pod
   and ConfigMap we created and check that `kubectl -n bgsu-cs-heydarnoori get jobs,pods,cm`
   shows only `kube-root-ca.crt`. Delete by name or by our label, never broadly. No
   `sleep infinity` or idle GPU pods: every pod must do work and exit. Set
   `requests == limits` for CPU and memory, helper pods included.
2. **SHA-pin every image; never `:latest`.** After a push, verify the pull with a one-shot pod
   that greps for the new code. Each `plan.yaml` change means commit → rebuild → push → update
   the SHA. A stale image is the failure that has hit this project most.
3. `activeDeadlineSeconds` goes on the **pod template**; at Job level it counts queue time.
4. Mount an in-memory `emptyDir` at `/dev/shm`; the 64 MB default causes "bus error".
5. Images need `build-essential` and `python3-dev`, because Triton JIT-compiles at runtime.
6. For many small cells on a contended GPU, use the **mega-runner** (one Job, one GPU, cells run
   sequentially with an idempotent skip): `scripts/nrp/runners/run_remaining_cells.py`.
7. Pull results with a transient pod that `base64`s the tarball (`kubectl logs` mangles binary)
   and check it with `gzip -t`: `scripts/nrp/sync.sh`. Its cron entry ran on bgsulab:
   `*/15 * * * * cd /home/ahosein/llm-labler && bash scripts/nrp/sync.sh >> /tmp/llm-labler-sync.log 2>&1`.
   The `fetched/.synced` ledger is per machine, so a new machine re-pulls everything (harmless).
8. HF Trainer saves checkpoints by default (633 MB tarballs). Use `save_strategy="no"` and
   `--skip_save_adapter`.
9. From LinkAnchor: mounting the **same PVC twice in one pod deadlocks the kubelet** (stuck in
   `ContainerCreating` forever). `gp-argo.*` nodes are flaky, so delete the pod to reschedule.
   Helper pods can take about 6 min in `ContainerCreating`, so give `kubectl wait` 8 min.
   `kubectl create configmap` cannot mix `--from-env-file` with `--from-literal`.
10. `pkill -f <pattern>` from a script whose own command line matches the pattern kills itself.
    Kill by PID.

---

## 6. Choosing where to run

| Job | Best place | Why |
|---|---|---|
| Paper build, docs, git, small CSV analysis | Laptop | No GPU needed |
| Anything that reads `results/` | bgsulab | The only copy is there |
| ≤14B inference, encoder baselines (SetFit/RoBERTa), 3B/7B LoRA | bgsulab 4090 | Free, and the same hardware as the paper's numbers. Use the GPU queue |
| 32B inference/extraction, transformers ≥ 5.10 models (Qwen3.5, Gemma 4, Ministral 3) | OSC (Cardinal `debug` for one job, Ascend preemptible for shards) | Fast, starts quickly. Costs budget |
| 14B/32B LoRA FT | NRP (L40/A6000) or OSC H100 | 4090 too small |
| Timing/cost numbers for the paper | Note the hardware | Paper timings mix 4090, L40/A6000 and H100; ratios across them are only indicative |

---

## 7. Pre-flight checklist (before any GPU work)

- [ ] Probes in §0 pass for the resource you need. VPN up for bgsulab: ask the user.
- [ ] The target's git checkout matches the commit you mean to run (`git -C ~/llm-labler log -1` on bgsulab; re-sync the code to OSC).
- [ ] Outputs go to a new path, never over anything in `results/`.
- [ ] Shared hardware: GPU idle (bgsulab), `OSCusage` budget checked (OSC), our-labels-only cleanup planned (NRP).
- [ ] Long jobs run in tmux or nohup (bgsulab), as batch jobs (OSC), or as Jobs (NRP), never in a foreground ssh.
- [ ] Idempotent, resumable code (skip-if-output-exists, done markers) wherever jobs can be killed (Ascend preemptible, NRP evictions).
- [ ] Results come back to bgsulab `results/` (relayed through the laptop from OSC), so the single archival copy stays complete.
