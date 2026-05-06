# Session Handoff — 2026-05-06

Single source of truth for the next session. Read this first.

This is a **state report**, not a narrative. It describes what is on disk, what is in flight, what conventions apply. Form your own analysis from `results/issues11k/` and the paper sections.

---

## 1. Where the paper stands

**Venue:** ESEM 2026, LIPIcs template, double-blind. Working title: *"RAGTAG: When Does Retrieval-Augmented Few-Shot Classification Match Fine-Tuning for GitHub Issue Triage?"* (placeholder, may change). Final deadline 2026-05-18.

**Drafted sections:**

| File | Status |
|---|---|
| [`paper/sections/03_approach.tex`](sections/03_approach.tex) | Done. RAGTAG, VOTAG, Fine-Tuning subsections written. |
| [`paper/sections/04_setup.tex`](sections/04_setup.tex) | Done except §4.5 hardware (placeholder pending kubectl describe). Significance-test methodology TBD ([TODO.md](TODO.md)). |
| [`paper/sections/05_evaluations.tex`](sections/05_evaluations.tex) | In progress. §5.1 *Retrieval Floor* prose has the VOTAG opener written. Two big result tables (PS giant, PA small) in place but currently report Qwen-7B per-class — needs regeneration to show all 4 models with F1 only (per locked discussion). §5.2 *Bug-Bias Diagnosis* exists as an empty-stub subsection (planning comments only). |

**Not yet drafted:** §1 Intro, §2 Related Work, §6 Discussion, §7 Threats, §8 Conclusion, §9 Data Availability.

LIPIcs `\input{}` order in [`paper/main.tex`](main.tex) is: 01_intro → 02_related → 03_approach → 04_setup → 05_evaluations → 06_discussion → 07_threats → 08_conclusion → 09_data_availability. Anonymous mode is on for double-blind.

---

## 2. Canonical results layout

`results/issues11k/` is the canonical paper-archival data store. After surgery on 2026-05-06, the directory naming is uniform across all four Qwen sizes — **no `_v2` suffixes anywhere**:

```
results/issues11k/
├── agnostic/
│   ├── neighbors/                     ← FAISS-derived neighbor CSVs (k=3, 9, 30)
│   ├── vtag/                          ← VOTAG predictions (model-independent)
│   └── <model_tag>/
│       ├── ragtag/predictions/        ← RAGTAG: preds_zero_shot.csv, preds_k{1,3,6,9}.csv (+ k=12, k=15 once campaigns finish)
│       ├── ragtag/evaluations/        ← matching eval_*.csv files
│       └── finetune_fixed/            ← FT predictions and evaluations (3-epoch)
└── project_specific/
    └── <project_tag>/
        ├── neighbors/
        ├── vtag/
        └── <model_tag>/
            ├── ragtag/                ← per-project RAGTAG
            ├── ragtag_debias_m3/      ← per-project Debiased RAGTAG (margin=3)
            └── finetune_fixed/        ← per-project FT
```

Active model_tags:
- `unsloth_Qwen2_5_3B_Instruct_bnb_4bit`
- `unsloth_Qwen2_5_7B_Instruct_bnb_4bit`
- `unsloth_Qwen2_5_14B_Instruct_bnb_4bit`
- `unsloth_Qwen2_5_32B_Instruct_bnb_4bit`

The 11 project tags: `ansible_ansible`, `bitcoin_bitcoin`, `dart-lang_sdk`, `dotnet_roslyn`, `facebook_react`, `flutter_flutter`, `kubernetes_kubernetes`, `microsoft_TypeScript`, `microsoft_vscode`, `opencv_opencv`, `tensorflow_tensorflow`.

Eval CSV path varies by how the cell was run:
- Local runs: `<model_tag>/finetune_fixed/eval_finetune_fixed.csv`
- NRP runs: `<model_tag>/finetune_fixed/evaluations/eval_finetune_fixed.csv`

Both layouts coexist in `results/issues11k/`. Analysis code should look in both locations.

---

## 3. The four canonicalization criteria

`results/issues11k/` was made canonical against four explicit criteria on 2026-05-06:

1. **All four method types present** for all four model sizes: zero-shot, VOTAG (model-independent), RAGTAG, fine-tune. ✓
2. **All fine-tunes are 3-epoch.** Confirmed: 3B/7B 3-epoch from local campaign on RTX 4090 (run via `run_3epoch_full_campaign.sh`); 14B/32B 3-epoch from NRP campaign synced 2026-05-06. The pre-3-epoch (1-epoch) data was wiped from `results/issues11k/` before the new campaign and is preserved at `results/issues11k_ft_1epoch_backup_20260504/` if needed for comparison. ✓
3. **No OOM-affected data in canonical paths.** The earlier Qwen-32B retrieval runs on the 4090 had a 22.8% invalid prediction rate at k=9 due to OOM. Those dirs were archived to `archive/oom_runs_20260422/` (131 MB) on 2026-05-06. The clean NRP-rerun versions (formerly in `_v2` dirs) were renamed to the standard names. ✓
4. **No conflicts when k=12/15 results land.** Plan.yaml waves 6/7 use the canonical (no-`_v2`) `output_subdir` paths so new NRP cells extend the canonical dirs alongside k=1,3,6,9. ✓

**14B retrieval runs are local, not NRP.** The paper text should not claim "all 14B retrieval results are from NRP." Only Qwen-32B retrieval and Qwen-14B Debias have NRP provenance. Qwen-14B RAGTAG (PA + PS) and Qwen-3B/7B retrieval ran locally; their invalid rates are normal (~5% range), no OOM signature.

---

## 4. Final 3-epoch fine-tune numbers (canonical)

Macro F1 macro across 3 epochs of LoRA fine-tuning (rank=16, α=16, lr=2e-4, paged AdamW 8-bit, max_seq_length=2048):

| Model | FT-PA (3,300 train) | FT-PS avg across 11 projects (300 train each) |
|---|---:|---:|
| Qwen-3B | 0.708 | 0.665 |
| Qwen-7B | 0.762 | 0.677 |
| Qwen-14B | 0.785 | 0.694 |
| Qwen-32B | 0.771 | 0.730 |

The 14B FT-PA being higher than 32B FT-PA is unexpected and may reflect single-seed variance (no multi-seed validation has been run). Multi-seed validation is flagged for §7 Threats.

---

## 5. In-flight campaigns

### Local: k=12/15 extension at ctx=8192

Started 2026-05-06 ~13:15 local. Driver: [`run_k12_k15_local_8k.sh`](../run_k12_k15_local_8k.sh).

For each of Qwen-3B, 7B, 14B, runs `python llm_labeler.py --top_ks "12,15"` for:
- RAGTAG agnostic (1 cell)
- RAGTAG project-specific (11 cells)
- Debiased RAGTAG project-specific (11 cells, `--debias_retrieval --debias_margin 3`)

Per-cell produces `preds_k12.csv` and `preds_k15.csv` (and matching eval CSVs via `--eval_dir`) in the canonical dirs. Idempotent skip on existing `preds_k15.csv`.

ETA: ~34 hours total (3B ~5h, 7B ~10h, 14B ~19h sequential).

### NRP: Qwen-32B k=12/15 at ctx=8192

Mega-runner Job submitted 2026-05-06 with image `:9506055`. Pod `mega-runner-x6z5z` is **Pending** at submission time, awaiting an L40 / L40S slot. Two waves in [`scripts/nrp/plan.yaml`](../scripts/nrp/plan.yaml):

- Wave 6 `qwen32b-ragtag-extended-8k`: 12 cells (PA + 11 PS)
- Wave 7 `qwen32b-debias-extended-8k`: 11 cells (PS only by design)

ETA when running: ~18-29 hours wall-clock.

### Local: DeBERTa-v3-large fine-tune (exploratory)

Started 2026-05-06 ~14:43 local. Driver: [`run_transformer_ft.py`](../run_transformer_ft.py).

`microsoft/deberta-v3-large` fine-tuned on the agnostic split (3,300 train) for 3 epochs at lr=2e-5, batch=8, max_seq_length=512. Outputs to `results/issues11k/agnostic/microsoft_deberta-v3-large/finetune_transformer/`. Auto-evaluates via `evaluate.py`.

**Status: exploratory only.** No commitment to including DeBERTa as a baseline in the paper. Decision will be made after seeing the F1 result.

ETA: ~15-20 minutes total.

---

## 6. Image lineage (NRP campaigns)

The Docker image `ghcr.io/doctorhoseinpour/llm-labler` is rebuilt and pushed when `plan.yaml` changes. Lineage:

| Tag | Date | Purpose |
|---|---|---|
| `:55ba8f1` | 2026-04-26 | Original NRP migration. Wave 1 RAGTAG-32B, Wave 3 14B-debias, Wave 4 32B-debias. |
| `:6570030` | 2026-05-04 | Bumped `num_train_epochs=1 → 3` in `fixed_fine-tune.py`. |
| `:7ecfc75` | 2026-05-04 | Wave-filtered plan.yaml ([0, 2, 5]) for the 3-epoch FT campaign. |
| `:3d1f3fa` | 2026-05-06 | plan.yaml waves 6/7 added for k=12/15 8K extension (still used `_v2` paths). |
| **`:9506055`** | **2026-05-06** | **Current.** plan.yaml updated to canonical (no-`_v2`) `output_subdir` after surgery. |

To verify the running pod's image: `kubectl -n bgsu-cs-heydarnoori get pod mega-runner-x6z5z -o jsonpath='{.spec.containers[0].image}'`.

---

## 7. Backups and archive

- `archive/oom_runs_20260422/` (131 MB) — Qwen-32B OOM-affected retrieval runs (Apr 22 timestamp) and the Apr 23 14B-debias-PS local runs that were superseded by the NRP rerun. Audit trail only; do not put back into canonical.
- `results/issues11k_ft_1epoch_backup_20260504/` (12 GB) — full snapshot of 1-epoch FT data before the May 4 wipe.
- `fetched/` (~204 MB) — sync.sh staging from NRP, both tarballs and extracted dirs. The extracted dirs duplicate canonical results; tarballs are a "what arrived from NRP" audit trail.
- `fetched_backup_20260505_112553/` — was deleted as cleanup; sync completed cleanly the day after.

---

## 8. Outstanding TODOs

See [paper/TODO.md](TODO.md) for the live list. High-level items:

- **Wait for the two in-flight 8K campaigns.** When both finish, regenerate the leaderboard tables to include k=12 and k=15 columns alongside k=1,3,6,9.
- **Bootstrap CIs + McNemar's significance tests.** Run after all k=12/15 cells land. ~50 lines of Python in `scripts/analysis/significance_tests.py`. Per-method × model × setting.
- **Hardware specs (§4.5).** Run `kubectl describe node <L40-node>` once the mega-runner schedules to capture exact spec.
- **§3.3 forward-pointer to debiasing.** Insert one-line cref to §5 once the §5 prose settles.
- **Phase 2 (16K context).** Optional follow-up to test k=12/15 at ctx=16384. Locked plan exists; defer the decision until 8K results show whether 16K is necessary.
- **Sections to draft** in order: §5 RQ-flow prose → §6 Discussion → §7 Threats → §8 Conclusion → §1 Intro → §2 Related Work.

---

## 9. File inventory

```
paper/
├── main.tex                  -- LIPIcs root, anonymous mode
├── refs.bib                  -- bibliography (user-managed)
├── sections/
│   ├── 01_intro.tex          -- stub
│   ├── 02_related.tex        -- stub
│   ├── 03_approach.tex       -- DONE
│   ├── 04_setup.tex          -- DONE except §4.5 hardware
│   ├── 05_evaluations.tex    -- partial; §5.1 Retrieval Floor prose started
│   ├── 06_discussion.tex     -- stub
│   ├── 07_threats.tex        -- stub
│   ├── 08_conclusion.tex     -- stub
│   └── 09_data_availability.tex  -- stub
├── figures/
│   ├── vtag_kcurve.pdf       -- VOTAG plateau figure (referenced from §5.1)
│   └── vtag_kcurve.png
├── TODO.md                   -- deferred items, live
├── SESSION_HANDOFF.md        -- this file
├── REVIEW_PROMPT.md          -- adversarial-reviewer template (may not exist)
├── CHANGELOG.md, LICENSE.md  -- LIPIcs template ship

docs/
├── NRP_MIGRATION_STATUS.md   -- steady-state cluster setup + bug history
└── *.pdf                     -- reference papers (Heo, Aracena, CAA, NoTrainingWheels)

results/
├── issues11k/                            -- canonical, paper-archival, never delete
└── issues11k_ft_1epoch_backup_20260504/  -- 1-epoch FT snapshot (audit only)

archive/
└── oom_runs_20260422/        -- archived OOM-affected and superseded retrieval runs

fetched/                      -- sync.sh staging + tarball audit trail

scripts/nrp/
├── plan.yaml                 -- waves 6/7 active for k=12/15 8K
├── manifests/job-mega-runner.yaml  -- mega-runner template
├── runners/run_remaining_cells.py
└── sync.sh                   -- pulls _outbox/ tarballs into local results/

# Top-level scripts (active)
run_k12_k15_local_8k.sh       -- in-flight local 8K extension campaign
run_transformer_ft.py         -- DeBERTa exploratory experiment
run_3epoch_full_campaign.sh   -- record of how 3-epoch FT was launched
llm_labeler.py                -- core RAGTAG inference (read-only reference)
fixed_fine-tune.py            -- core LoRA fine-tune (read-only reference)
vtag.py                       -- core VOTAG inference
build_11k_index.py            -- FAISS index builder
evaluate.py                   -- standard metric reporter
```

---

## 10. Conventions to follow

- **`results/issues11k/`** is paper-archival. Never delete or move data inside it during refactors. Preserve OOM-affected and superseded data via `archive/` only.
- **No `_v2` paths anywhere** — those were renamed on 2026-05-06.
- **Scripts write to `<dir>/predictions/preds_*.csv` and `<dir>/evaluations/eval_*.csv`** when invoked with `--eval_dir`. Some older scripts wrote eval CSVs directly into the FT root (`<dir>/eval_finetune_fixed.csv`); both layouts coexist on disk.
- **Image SHA discipline:** every plan.yaml change → rebuild image with new SHA → push → update mega-runner manifest. Verify the running pod's image before letting it run for hours.
- **Idempotency:** local shell drivers and the NRP runner skip cells whose `preds_*.csv` already exists. Safe to re-run after partial completion.
- **Splits are deterministic** — regenerated from the source CSVs on first run; safe to delete and re-create.
