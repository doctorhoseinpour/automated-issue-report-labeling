# Machines and Where Things Live

Last verified: 2026-09-16 (over `ssh bgsulab`).

The project is developed on two machines that share the same GitHub remote
(`git@github.com:doctorhoseinpour/automated-issue-report-labeling.git`).
Code and paper sources are synced through git. **Experiment outputs are not**:
`results/` is gitignored and exists only on the lab machine.

## 1. Local PC (this clone)

| | |
|---|---|
| Path | `~/Desktop/my_projects/automated-issue-report-labeling` |
| GPU | NVIDIA GTX 1650 Max-Q, 4 GB VRAM. **Cannot run any Qwen inference or fine-tuning.** |
| Python env | none set up yet (no `venv/`); create one from `requirements.txt` only if a CPU-side script is needed |
| `results/` | absent (gitignored). Copy specific CSVs from the lab machine with `scp`/`rsync` when an analysis script needs them |
| LaTeX | TeX Live 2019 (`pdflatex`, `bibtex`; no `latexmk`). `texlive-publishers` is not installed, so `IEEEtran.cls`/`.bst` are shipped inside `SANER2027/` |

Use this machine for: paper writing (`SANER2027/`, `paper/`), documentation,
offline analyses on small CSVs, git work.

## 2. BGSU lab machine (`bgsulab`)

| | |
|---|---|
| Access | `ssh bgsulab` (alias in `~/.ssh/config`: `192.168.198.25`, user `ahosein`, key auth). **Requires the BGSU VPN to be connected first.** The VPN drops idle connections, so keep-alives are configured in the ssh alias |
| Hostname | `heydarnoori` |
| OS / CUDA / Python | Ubuntu 24.04.2 LTS, CUDA 12.0 (`nvcc`), Python 3.12.3 |
| GPU | NVIDIA RTX 4090, 24 GB VRAM |
| Disk | 1.9 TB NVMe, ~990 GB free (Sep 2026) |
| **Project path** | **`/home/ahosein/llm-labler`** — note the directory is spelled `llm-labler` (no second "e"), not `llm-labeler` |
| Git | same remote; was on branch `encoder-baselines` at `36475bc` when checked |
| Python envs | `venv/` (main: Unsloth, FAISS, transformers) and `venv-setfit/` (pinned SetFit env, `requirements-setfit.txt`) |
| `results/` | **81 GB, the only copy.** `results/issues11k/{agnostic,project_specific}` is the paper's archival data; also holds `results/issues11k.zip`, legacy 3k/30k runs, `bragtag_margin_validation/`, `last_resort_llama/`, `vtag*/`, steering-vector experiments |
| Other local-only dirs | `esem/` (ESEM figure copies), `paper/` with a compiled `main.pdf`, `canary_openai/`, `fetched/` (raw GitHub fetches), `logs/`, `unsloth_compiled_cache/`, `.faiss_cache/`, legacy `issues3k.csv` / `issues30k.csv` |

Use this machine for: every GPU job (RAGTAG/VOTAG/zero-shot inference, LoRA
fine-tuning of 3B/7B, SetFit/RoBERTa baselines), regenerating paper
figures/tables from `results/` with `scripts/paper/*.py`.

Typical loop:

```bash
# on the lab machine
ssh bgsulab
cd ~/llm-labler && git pull
source venv/bin/activate
bash run_encoder_baselines.sh            # or any other driver
venv/bin/python scripts/paper/tab_method_comparison.py

# bring a generated figure/table back to the local clone
rsync -av bgsulab:~/llm-labler/paper/figures/ paper/figures/
```

Never delete or rename anything under `results/` on the lab machine; move
superseded runs to `archive/` (see `CLAUDE.md` conventions).

## 3. NRP (Nautilus) and OSC

Unchanged from `CLAUDE.md`: NRP namespace `bgsu-cs-heydarnoori` for 14B/32B
fine-tuning via `scripts/nrp/`, OSC Ascend as Slurm backup via
`run_server_11k.sh`. `kubectl` config for NRP lives on the lab machine
(`~/.kube`), not on the local PC.
