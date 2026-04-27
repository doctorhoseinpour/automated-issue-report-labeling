#!/usr/bin/env python3
"""Mega-runner: process all remaining production cells sequentially in one Pod.

Reuses submit.py's WAVE_BUILDERS to enumerate cells and produce their
shell commands, then runs each via subprocess. After each cell, tars its
output dir into _outbox/<name>.tar.gz so sync.sh can pull it back.

Idempotent: a cell is skipped if any preds_*.csv already exists under its
output paths. Lets a Job pod restart resume cleanly without redoing work.

Per-cell errors are logged and the runner continues to the next cell;
final summary lists pass/fail. Exits 0 if all cells completed or skipped,
1 if any failed.
"""
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

REPO_ROOT = Path("/workspace/llm-labler")
SCRIPT_DIR = Path(__file__).resolve().parent
SUBMIT_DIR = SCRIPT_DIR.parent
PLAN_PATH = SUBMIT_DIR / "plan.yaml"
RESULTS_ROOT = REPO_ROOT / "results"
OUTBOX = RESULTS_ROOT / "_outbox"

sys.path.insert(0, str(SUBMIT_DIR))
import submit  # noqa: E402  — needs path tweak above


def cell_already_done(output_paths_str: str) -> bool:
    """True if any preds_*.csv exists anywhere under any of the cell's
    output paths. The output_paths field from submit.py builders is a
    space-separated list of paths relative to RESULTS_ROOT."""
    for rel in output_paths_str.split():
        full = RESULTS_ROOT / rel
        if not full.exists():
            continue
        for _ in full.rglob("preds_*.csv"):
            return True
    return False


def tar_outputs(outbox_name: str, output_paths_str: str) -> bool:
    """Tar each cell's output_paths into _outbox/<outbox_name>.tar.gz so
    sync.sh on the laptop can pull it. Mirrors the format the per-Job
    template uses."""
    OUTBOX.mkdir(parents=True, exist_ok=True)
    tarball = OUTBOX / f"{outbox_name}.tar.gz"
    paths = output_paths_str.split()
    cmd = ["tar", "czf", str(tarball), "-C", str(RESULTS_ROOT)] + paths
    rc = subprocess.run(cmd).returncode
    if rc != 0:
        print(f"    [tar] FAILED rc={rc} for {tarball}", flush=True)
        return False
    size_mb = tarball.stat().st_size / (1024 * 1024)
    print(f"    [tar] {tarball.name} ({size_mb:.1f} MB)", flush=True)
    return True


def run_cell(cell: dict) -> tuple[bool, str, float]:
    """Returns (ok, status, elapsed_s). status is one of
    'skipped' | 'completed' | 'failed_run' | 'failed_tar'."""
    name = cell["name"]
    cmd = cell["command"]
    out_paths = cell["output_paths"]
    outbox_name = cell["outbox_name"]

    if cell_already_done(out_paths):
        print(f"  [{name}] SKIP (preds already exist)", flush=True)
        return True, "skipped", 0.0

    print(f"  [{name}] running...", flush=True)
    print(f"    cmd: {cmd}", flush=True)
    t0 = time.time()
    rc = subprocess.run(cmd, shell=True, cwd=REPO_ROOT).returncode
    elapsed = time.time() - t0
    if rc != 0:
        print(f"  [{name}] FAILED rc={rc} after {elapsed:.0f}s", flush=True)
        return False, "failed_run", elapsed

    if not tar_outputs(outbox_name, out_paths):
        return False, "failed_tar", elapsed

    print(f"  [{name}] OK in {elapsed:.0f}s", flush=True)
    return True, "completed", elapsed


def enumerate_cells(plan: dict) -> list[dict]:
    """Build the ordered list of all cells from Waves 1-5 (Wave 0
    indexing is already done). Within each wave: project-specific cells
    first (short), agnostic cell last (long). Across waves: 1 → 5."""
    image = plan["image"]
    ordered = []
    for wave in plan["waves"]:
        if wave["id"] == 0:
            continue
        builder = submit.WAVE_BUILDERS.get(wave["type"])
        if builder is None:
            print(f"WARN: no builder for wave {wave['id']} type={wave['type']}",
                  flush=True)
            continue
        cells = builder(plan, image, wave)
        # agnostic-named cells go last so per-project (faster) cells run first
        cells.sort(key=lambda c: (1 if c["name"].endswith("-agnostic") else 0))
        ordered.extend(cells)
    return ordered


def main() -> int:
    plan = yaml.safe_load(PLAN_PATH.read_text())
    cells = enumerate_cells(plan)
    t_start = time.time()

    print("\n========== MEGA-RUNNER ==========", flush=True)
    print(f"Plan:        {PLAN_PATH}", flush=True)
    print(f"Image:       {plan['image']}", flush=True)
    print(f"Total cells: {len(cells)}", flush=True)
    print(f"Started:     {time.strftime('%Y-%m-%dT%H:%M:%S%z')}", flush=True)
    print("==================================\n", flush=True)

    results = []
    for i, cell in enumerate(cells, 1):
        print(f"\n--- [{i}/{len(cells)}] {cell['name']} ---", flush=True)
        ok, status, elapsed = run_cell(cell)
        results.append((cell["name"], ok, status, elapsed))

    total_elapsed = time.time() - t_start
    n_completed = sum(1 for _, ok, st, _ in results if ok and st == "completed")
    n_skipped = sum(1 for _, ok, st, _ in results if ok and st == "skipped")
    n_failed = sum(1 for _, ok, _, _ in results if not ok)

    print("\n========== SUMMARY ==========", flush=True)
    print(f"Total wall:  {total_elapsed / 3600:.2f} h", flush=True)
    print(f"Completed:   {n_completed}", flush=True)
    print(f"Skipped:     {n_skipped}", flush=True)
    print(f"Failed:      {n_failed}", flush=True)
    if n_failed > 0:
        print("\nFailed cells:", flush=True)
        for name, ok, status, _ in results:
            if not ok:
                print(f"  {name}: {status}", flush=True)
    print("=============================", flush=True)

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
