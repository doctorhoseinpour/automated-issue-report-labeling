#!/usr/bin/env python3
"""NRP campaign launcher for the 11k experiments.

Renders Kubernetes Job YAMLs from templates/job.yaml.j2 and applies them with
kubectl. Two main modes:

  --smoke-test          # 2 Jobs (indexing-mini + Llama-3B zero-shot)
  --plan plan.yaml      # Full campaign (Wave 0 first, then Waves 1-5 parallel)

Other flags:
  --dry-run             # Print rendered YAML to stdout, don't apply
  --waves 1,2,3         # Subset of waves to submit (default: all)
  --image TAG           # Override container image (default: from plan.yaml)
  --no-wait             # Don't kubectl wait for Wave 0 in plan mode

Idempotence: re-running submits the same Jobs, which kubectl apply will reject
if Job names already exist. Delete prior Jobs (or use --waves to skip done
ones) before retrying. The campaign is small enough that this is fine.
"""

import argparse
import os
import shlex
import subprocess
import sys
from pathlib import Path

import yaml
from jinja2 import Template

SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parent.parent
TEMPLATE_PATH = SCRIPT_DIR / "templates" / "job.yaml.j2"
DEFAULT_PLAN_PATH = SCRIPT_DIR / "plan.yaml"

CACHE_DIR_IN_POD = "/workspace/hf_cache"
RESULTS_ROOT = "results/issues11k"
TRAIN_CSV = "issues11k_train.csv"
TEST_CSV = "issues11k_test.csv"
DATASET = "issues11k.csv"
EMBED_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


# -----------------------------------------------------------------------------
# helpers
# -----------------------------------------------------------------------------

def k8s_safe(s: str) -> str:
    """Lowercase, dashes only — fits k8s name rules."""
    return s.lower().replace("_", "-").replace(".", "-").replace("/", "-")


def render(template_str: str, **vars) -> str:
    return Template(template_str).render(**vars)


def info(msg: str) -> None:
    """Status messages — always to stderr so stdout stays clean for pipes."""
    print(msg, file=sys.stderr)


def kubectl_apply(yaml_text: str, dry_run: bool) -> None:
    if dry_run:
        sys.stdout.write("---\n" + yaml_text + "\n")
        return
    p = subprocess.run(
        ["kubectl", "apply", "-f", "-"],
        input=yaml_text,
        text=True,
        capture_output=True,
    )
    sys.stdout.write(p.stdout)
    if p.returncode != 0:
        sys.stderr.write(p.stderr)
        raise SystemExit(f"kubectl apply failed (rc={p.returncode})")


def kubectl_wait(job_name: str, timeout_seconds: int) -> int:
    """Wait for a Job to reach Complete or Failed. Returns 0 on success."""
    print(f"[wait] kubectl wait --for=condition=complete job/{job_name} --timeout={timeout_seconds}s", file=sys.stderr)
    p = subprocess.run(
        ["kubectl", "wait", "--for=condition=complete",
         f"job/{job_name}", f"--timeout={timeout_seconds}s"],
    )
    return p.returncode


# -----------------------------------------------------------------------------
# Job builders — one per wave type
# -----------------------------------------------------------------------------

def base_render_args(plan: dict, tier: str, image: str, name: str, wave: str,
                     command: str, outbox: str, output_paths: str,
                     active_deadline_seconds: int = 43200) -> dict:
    tier_spec = plan["tiers"][tier]
    return dict(
        name=name,
        wave=wave,
        tier=tier,
        image=image,
        command=command,
        outbox_name=outbox,
        output_paths=output_paths,
        cpu=tier_spec["cpu"],
        mem=tier_spec["mem"],
        gpu_resources=tier_spec["gpu_resources"],
        node_selector_products=tier_spec.get("node_selector_products") or [],
        active_deadline_seconds=active_deadline_seconds,
    )


def build_indexing_job(plan: dict, image: str, wave: dict) -> dict:
    """Wave 0: single Job, agnostic + 11 projects, ~10 min."""
    name = "w0-indexing"
    cmd = "bash scripts/nrp/runners/run_indexing_all.sh"
    output_paths = "issues11k/agnostic/neighbors issues11k/project_specific"
    return base_render_args(
        plan, tier="M", image=image, name=name, wave="0",
        command=cmd, outbox="w0-indexing", output_paths=output_paths,
        active_deadline_seconds=wave.get("active_deadline_seconds", 3600),
    )


def build_ragtag_jobs(plan: dict, image: str, wave: dict) -> list[dict]:
    """Wave 1: Qwen-32B RAGTAG, agnostic + 11 project-specific."""
    model_short = wave["model"]
    model = plan["models"][model_short]
    out_subdir = wave["output_subdir"]
    top_ks = wave["top_ks"]
    ctx = wave["max_seq_length"]
    deadline = wave.get("active_deadline_seconds", 21600)

    jobs = []
    settings = wave["settings"]

    if "agnostic" in settings:
        out_dir = f"{RESULTS_ROOT}/agnostic/{model['tag']}/{out_subdir}/predictions"
        cmd = (
            f"python llm_labeler.py "
            f"--model {model['hf_id']} "
            f"--neighbors_dir {RESULTS_ROOT}/agnostic/neighbors "
            f"--top_ks \"{top_ks}\" "
            f"--output_dir {out_dir} "
            f"--max_seq_length {ctx} "
            f"--max_new_tokens 50 "
            f"--inference_batch_size 1 "
            f"--cache_dir {CACHE_DIR_IN_POD}"
        )
        name = f"w{wave['id']}-{model['short']}-ragtag-agnostic"
        outbox = name
        out_paths = f"issues11k/agnostic/{model['tag']}/{out_subdir}"
        jobs.append(base_render_args(
            plan, tier=model["tier"], image=image, name=name, wave=str(wave["id"]),
            command=cmd, outbox=outbox, output_paths=out_paths,
            active_deadline_seconds=deadline,
        ))

    if "project_specific" in settings:
        for proj in plan["projects"]:
            out_dir = f"{RESULTS_ROOT}/project_specific/{proj}/{model['tag']}/{out_subdir}/predictions"
            cmd = (
                f"python llm_labeler.py "
                f"--model {model['hf_id']} "
                f"--neighbors_dir {RESULTS_ROOT}/project_specific/{proj}/neighbors "
                f"--top_ks \"{top_ks}\" "
                f"--output_dir {out_dir} "
                f"--max_seq_length {ctx} "
                f"--max_new_tokens 50 "
                f"--inference_batch_size 1 "
                f"--cache_dir {CACHE_DIR_IN_POD}"
            )
            name = f"w{wave['id']}-{model['short']}-ragtag-{k8s_safe(proj)}"
            outbox = name
            out_paths = f"issues11k/project_specific/{proj}/{model['tag']}/{out_subdir}"
            jobs.append(base_render_args(
                plan, tier=model["tier"], image=image, name=name, wave=str(wave["id"]),
                command=cmd, outbox=outbox, output_paths=out_paths,
                active_deadline_seconds=deadline,
            ))
    return jobs


def build_debias_jobs(plan: dict, image: str, wave: dict) -> list[dict]:
    """Wave 3/4: debiased RAGTAG, project-specific only."""
    model_short = wave["model"]
    model = plan["models"][model_short]
    out_subdir = wave["output_subdir"]
    top_ks = wave["top_ks"]
    ctx = wave["max_seq_length"]
    margin = wave["debias_margin"]
    deadline = wave.get("active_deadline_seconds", 14400)

    jobs = []
    for proj in plan["projects"]:
        out_dir = f"{RESULTS_ROOT}/project_specific/{proj}/{model['tag']}/{out_subdir}/predictions"
        cmd = (
            f"python llm_labeler.py "
            f"--model {model['hf_id']} "
            f"--neighbors_dir {RESULTS_ROOT}/project_specific/{proj}/neighbors "
            f"--top_ks \"{top_ks}\" "
            f"--output_dir {out_dir} "
            f"--max_seq_length {ctx} "
            f"--max_new_tokens 50 "
            f"--inference_batch_size 1 "
            f"--debias_retrieval --debias_margin {margin} "
            f"--cache_dir {CACHE_DIR_IN_POD}"
        )
        name = f"w{wave['id']}-{model['short']}-debias-{k8s_safe(proj)}"
        out_paths = f"issues11k/project_specific/{proj}/{model['tag']}/{out_subdir}"
        jobs.append(base_render_args(
            plan, tier=model["tier"], image=image, name=name, wave=str(wave["id"]),
            command=cmd, outbox=name, output_paths=out_paths,
            active_deadline_seconds=deadline,
        ))
    return jobs


def build_finetune_jobs(plan: dict, image: str, wave: dict) -> list[dict]:
    """Wave 2/5: LoRA fine-tuning."""
    model_short = wave["model"]
    model = plan["models"][model_short]
    out_subdir = wave["output_subdir"]
    deadline = wave.get("active_deadline_seconds", 43200)

    jobs = []
    settings = wave["settings"]

    if "agnostic" in settings:
        ft_dir = f"{RESULTS_ROOT}/agnostic/{model['tag']}/{out_subdir}"
        cmd = (
            f"python fixed_fine-tune.py "
            f"--model {model['hf_id']} "
            f"--dataset {DATASET} "
            f"--train_csv {RESULTS_ROOT}/agnostic/neighbors/train_split.csv "
            f"--test_csv {RESULTS_ROOT}/agnostic/neighbors/test_split.csv "
            f"--max_seq_length 2048 "
            f"--max_new_tokens 50 "
            f"--inference_batch_size 1 "
            f"--output_dir {ft_dir} "
            f"--cache_dir {CACHE_DIR_IN_POD}"
        )
        name = f"w{wave['id']}-{model['short']}-ft-agnostic"
        out_paths = f"issues11k/agnostic/{model['tag']}/{out_subdir}"
        jobs.append(base_render_args(
            plan, tier=model["tier"], image=image, name=name, wave=str(wave["id"]),
            command=cmd, outbox=name, output_paths=out_paths,
            active_deadline_seconds=deadline,
        ))

    if "project_specific" in settings:
        for proj in plan["projects"]:
            ft_dir = f"{RESULTS_ROOT}/project_specific/{proj}/{model['tag']}/{out_subdir}"
            cmd = (
                f"python fixed_fine-tune.py "
                f"--model {model['hf_id']} "
                f"--dataset {DATASET} "
                f"--train_csv {RESULTS_ROOT}/project_specific/{proj}/neighbors/train_split.csv "
                f"--test_csv {RESULTS_ROOT}/project_specific/{proj}/neighbors/test_split.csv "
                f"--max_seq_length 2048 "
                f"--max_new_tokens 50 "
                f"--inference_batch_size 1 "
                f"--output_dir {ft_dir} "
                f"--cache_dir {CACHE_DIR_IN_POD}"
            )
            name = f"w{wave['id']}-{model['short']}-ft-{k8s_safe(proj)}"
            out_paths = f"issues11k/project_specific/{proj}/{model['tag']}/{out_subdir}"
            jobs.append(base_render_args(
                plan, tier=model["tier"], image=image, name=name, wave=str(wave["id"]),
                command=cmd, outbox=name, output_paths=out_paths,
                active_deadline_seconds=deadline,
            ))
    return jobs


WAVE_BUILDERS = {
    "indexing": lambda plan, image, w: [build_indexing_job(plan, image, w)],
    "ragtag": build_ragtag_jobs,
    "debias": build_debias_jobs,
    "finetune": build_finetune_jobs,
}


# -----------------------------------------------------------------------------
# Smoke test builder
# -----------------------------------------------------------------------------

def build_smoke_jobs(plan: dict, image: str) -> list[dict]:
    """Two Jobs: indexing-mini for ansible_ansible, then Llama-3B zero-shot."""
    proj_filter = "ansible/ansible"
    proj_tag = "ansible_ansible"
    out_root = f"{RESULTS_ROOT}/_smoketest"
    llama = plan["models"]["llama3b"]

    # 1. Indexing mini — uses tier S so it tests the smaller GPU pool too,
    #    matching where Llama-3B will land. build_11k_index.py needs GPU
    #    for the embedder regardless of tier.
    idx_cmd = (
        f"bash scripts/nrp/runners/run_indexing_one.sh "
        f"{shlex.quote(proj_filter)} {shlex.quote(proj_tag)} {shlex.quote(out_root)}"
    )
    idx_paths = f"issues11k/_smoketest/{proj_tag}/neighbors"
    idx_job = base_render_args(
        plan, tier="S", image=image, name="smoketest-indexing", wave="smoke",
        command=idx_cmd, outbox="smoketest-indexing", output_paths=idx_paths,
        active_deadline_seconds=1200,
    )

    # 2. Llama-3B zero-shot — tier S, ~5 min. This is the core "model load
    #    on PVC is fast" verification.
    pred_dir = f"{out_root}/{proj_tag}/{llama['tag']}/ragtag/predictions"
    label_cmd = (
        f"python llm_labeler.py "
        f"--model {llama['hf_id']} "
        f"--neighbors_dir {out_root}/{proj_tag}/neighbors "
        f"--top_ks \"0\" "
        f"--output_dir {pred_dir} "
        f"--max_seq_length 8192 "
        f"--max_new_tokens 50 "
        f"--inference_batch_size 1 "
        f"--cache_dir {CACHE_DIR_IN_POD}"
    )
    label_paths = f"issues11k/_smoketest/{proj_tag}/{llama['tag']}"
    label_job = base_render_args(
        plan, tier="S", image=image, name="smoketest-llama3b", wave="smoke",
        command=label_cmd, outbox="smoketest-llama3b", output_paths=label_paths,
        active_deadline_seconds=2400,  # 40 min budget for cold load + 300 issues
    )

    return [idx_job, label_job]


# -----------------------------------------------------------------------------
# main
# -----------------------------------------------------------------------------

def render_and_apply(jobs: list[dict], template_text: str, dry_run: bool) -> None:
    for j in jobs:
        rendered = render(template_text, **j)
        kubectl_apply(rendered, dry_run=dry_run)


def main():
    ap = argparse.ArgumentParser()
    grp = ap.add_mutually_exclusive_group(required=True)
    grp.add_argument("--smoke-test", action="store_true",
                     help="Submit the 2-Job smoke test (indexing-mini + Llama-3B zero-shot)")
    grp.add_argument("--plan", type=Path, default=None,
                     help="Submit the full campaign from this plan.yaml")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print rendered YAML to stdout instead of applying")
    ap.add_argument("--waves", type=str, default=None,
                    help="Comma-separated wave IDs to submit (default: all in the plan)")
    ap.add_argument("--image", type=str, default=None,
                    help="Override container image (default: plan.yaml's image field)")
    ap.add_argument("--no-wait", action="store_true",
                    help="Don't block on Wave 0 completion before submitting Waves 1+")
    args = ap.parse_args()

    template_text = TEMPLATE_PATH.read_text()
    plan_path = args.plan if args.plan else DEFAULT_PLAN_PATH
    plan = yaml.safe_load(plan_path.read_text())
    image = args.image or plan["image"]

    if args.smoke_test:
        jobs = build_smoke_jobs(plan, image)
        info(f"[smoke-test] rendering {len(jobs)} Job manifests, dry_run={args.dry_run}")
        render_and_apply(jobs, template_text, args.dry_run)
        if args.dry_run:
            return
        # Wait for both, in submission order (indexing first since llama3b reads its output)
        for j in jobs:
            rc = kubectl_wait(j["name"], j["active_deadline_seconds"] + 60)
            if rc != 0:
                info(f"[smoke-test] FAILED waiting on {j['name']} (rc={rc})")
                info(f"[smoke-test] Inspect with:  kubectl logs job/{j['name']}")
                raise SystemExit(rc)
        info("[smoke-test] both Jobs Complete. Now wait ≤15 min for sync.sh")
        info("[smoke-test] to pull tarballs and integrate into ./results/")
        return

    # Full campaign
    waves_filter = None
    if args.waves:
        waves_filter = {int(w) for w in args.waves.split(",")}

    wave0 = next((w for w in plan["waves"] if w["id"] == 0), None)
    other_waves = [w for w in plan["waves"] if w["id"] != 0]

    if waves_filter is None or 0 in waves_filter:
        if wave0 is None:
            raise SystemExit("plan.yaml has no wave 0 (indexing) — cannot proceed")
        info("[campaign] === Wave 0: indexing ===")
        jobs = WAVE_BUILDERS[wave0["type"]](plan, image, wave0)
        render_and_apply(jobs, template_text, args.dry_run)
        if not args.dry_run and not args.no_wait:
            for j in jobs:
                rc = kubectl_wait(j["name"], j["active_deadline_seconds"] + 60)
                if rc != 0:
                    raise SystemExit(f"Wave 0 failed (rc={rc}); aborting")
            info("[campaign] Wave 0 complete. Submitting Waves 1-5...")

    submitted = 0
    for w in other_waves:
        if waves_filter is not None and w["id"] not in waves_filter:
            continue
        builder = WAVE_BUILDERS.get(w["type"])
        if builder is None:
            info(f"[warn] no builder for wave type {w['type']!r}; skipping wave {w['id']}")
            continue
        jobs = builder(plan, image, w)
        info(f"[campaign] === Wave {w['id']} ({w['name']}): {len(jobs)} Jobs ===")
        render_and_apply(jobs, template_text, args.dry_run)
        submitted += len(jobs)

    if args.dry_run:
        return
    info(f"[campaign] submitted {submitted} Jobs across Waves 1-5.")
    info("[campaign] cron-driven sync.sh will pull results into ./results/ as Jobs finish.")
    info("[campaign] watch progress: kubectl get jobs -l campaign=llm-labler-11k --sort-by=.metadata.creationTimestamp")


if __name__ == "__main__":
    main()
