#!/usr/bin/env python3
"""Small CPU-only diagnostic, restricted to the verified historical TRAIN pool.

Requires numpy/scipy (available here in python3.9); no model downloads. This is
a lexical mechanism probe, not a competitor to the paper's neural methods.
Every reported score uses held-out rows *within* historical TRAIN. Template
discovery, vocabulary, IDF and regularization selection exclude that holdout.
"""
from __future__ import annotations

import argparse
import collections
import csv
import hashlib
import json
import math
import os
import re
import sys
from pathlib import Path

os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np
import scipy
from scipy import sparse
from scipy.sparse.linalg import lsqr

LABELS = ("bug", "feature", "question")
HTML = re.compile(r"<!--.*?-->", re.S)
WORD = re.compile(r"(?u)\b[a-z][a-z_]{1,}\b")
SPACE = re.compile(r"\s+")
BUG_FORM = re.compile(r"bug report|steps to reproduce|actual behavio[u]?r|expected behavio[u]?r", re.I)
ASK_TITLE = re.compile(r"\?|\b(how|why|question|help|is it possible|can i|can we)\b", re.I)


def identity_digest(rows):
    identities = sorted(json.dumps([r[k] for k in ("repo", "created_at", "labels", "title")],
                                   ensure_ascii=True, separators=(",", ":")) for r in rows)
    return hashlib.sha256("\n".join(identities).encode()).hexdigest()


def load_rows(path, verify_train=True):
    csv.field_size_limit(sys.maxsize)
    with path.open() as f:
        rows = list(csv.DictReader(f))
    for i, r in enumerate(rows):
        r["source_row"] = i
    if verify_train:
        assert len(rows) == 3300, "Must supply the original 3,300-row historical TRAIN CSV."
        assert set(collections.Counter((r["repo"], r["labels"]) for r in rows).values()) == {100}
        assert identity_digest(rows) == "f51394e647b57c15ab18335e88fff6c57af96e7969bd8cf2d7efe9d29504bbdd", \
            "TRAIN identities differ from the historical lab split; independently verify provenance."
    return rows


def reconstruct_train(source):
    """Exact oldest-100/project/label split, independently verified against lab.

    Sorted identities, row order and all bodies matched the original lab TRAIN
    on 2026-09-17. This avoids requiring a separately distributed split CSV.
    """
    full = load_rows(source, verify_train=False)
    assert len(full) == 6600
    groups = collections.defaultdict(list)
    for r in full:
        groups[r['repo'], r['labels']].append(r)
    rows = [r for key in sorted(groups) for r in sorted(groups[key], key=lambda r:r['created_at'])[:100]]
    assert identity_digest(rows) == "f51394e647b57c15ab18335e88fff6c57af96e7969bd8cf2d7efe9d29504bbdd"
    for i, r in enumerate(rows):
        r['source_row'] = i  # Row index in historical TRAIN, not the source pool.
    return rows


def line_key(line):
    return SPACE.sub(" ", line.strip().lower())


def learn_templates(rows):
    counts = collections.defaultdict(collections.Counter)
    for r in rows:
        lines = {line_key(x) for x in HTML.sub("", r["body"]).splitlines()}
        counts[r["repo"]].update(x for x in lines if 4 <= len(x) <= 250)
    return {repo: {line for line, n in c.items() if n >= 5} for repo, c in counts.items()}


def clean_body(r, templates):
    body = HTML.sub("", r["body"])
    repeated = templates.get(r["repo"], set())
    return "\n".join(line for line in body.splitlines() if line_key(line) not in repeated)


def tokens(r, view, templates):
    title = r["title"]
    if view == "title":
        text = title
    elif view == "template":
        kept = templates.get(r["repo"], set())
        text = "\n".join(x for x in r["body"].splitlines() if line_key(x) in kept)
    else:
        body = clean_body(r, templates) if view == "clean" else r["body"]
        text = title + " " + title + " " + body
    # Same bounded input policy in every content arm; URLs/number literals are
    # excluded by the tokenizer. Exact title/body text is not written to output.
    words = WORD.findall(text.lower())[:6000]
    return collections.Counter(words + [a + " " + b for a, b in zip(words, words[1:])])


class Vectorizer:
    def __init__(self, rows, view, templates):
        self.view, self.templates = view, templates
        counts = collections.Counter()
        for r in rows:
            counts.update(tokens(r, view, templates).keys())
        terms = sorted((w for w in counts if counts[w] >= 2), key=lambda w: (-counts[w], w))[:40000]
        self.vocab = {w: i for i, w in enumerate(terms)}
        self.idf = np.array([math.log((1 + len(rows)) / (1 + counts[w])) + 1 for w in terms])
        self.projects = {p: i for i, p in enumerate(sorted({r["repo"] for r in rows}))}

    def transform(self, rows, project=False):
        rr, cc, vv = [], [], []
        for i, r in enumerate(rows):
            vals = [(self.vocab[w], (1 + math.log(n)) * self.idf[self.vocab[w]])
                    for w, n in tokens(r, self.view, self.templates).items() if w in self.vocab]
            denom = math.sqrt(sum(v * v for _, v in vals)) or 1.0
            for j, v in vals:
                rr.append(i); cc.append(j); vv.append(v / denom)
        x = sparse.csr_matrix((vv, (rr, cc)), shape=(len(rows), len(self.vocab)))
        if project:
            # Shared weights plus equally regularized repo-specific deviations;
            # a sparse partial-pooling model, not separate small classifiers.
            coo = x.tocoo()
            pj = np.array([self.projects[r["repo"]] for r in rows])
            local = sparse.csr_matrix((coo.data, (coo.row, coo.col + pj[coo.row] * x.shape[1])),
                                      shape=(len(rows), len(self.projects) * x.shape[1]))
            x = sparse.hstack((x, local), format="csr")
        return sparse.hstack((x, np.ones((len(rows), 1))), format="csr")


def fit_predict(x, labels, queries, alpha):
    y = np.eye(3)[labels]
    weights = np.column_stack([lsqr(x, y[:, c], damp=math.sqrt(alpha), atol=1e-5,
                                   btol=1e-5, iter_lim=180)[0] for c in range(3)])
    return [np.asarray(q @ weights).argmax(axis=1) for q in queries]


def metrics(y, pred):
    cm = np.zeros((3, 3), dtype=int)
    np.add.at(cm, (y, pred), 1)
    f1 = [2 * cm[c, c] / max(1, cm[c, :].sum() + cm[:, c].sum()) for c in range(3)]
    return dict(macro_f1=float(np.mean(f1)), accuracy=float(np.trace(cm) / cm.sum()),
                per_class_f1=dict(zip(LABELS, map(float, f1))), confusion=cm.tolist(),
                question_to_bug=int(cm[2, 0]), bug_to_question=int(cm[0, 2]))


def split_rows(rows, seed):
    rng = np.random.RandomState(seed)
    groups = collections.defaultdict(list)
    for r in rows:
        groups[r["repo"], r["labels"]].append(r)
    fit, val, hold = [], [], []
    for key in sorted(groups):
        group = groups[key]
        order = rng.permutation(len(group))
        fit.extend(group[i] for i in order[:60])
        val.extend(group[i] for i in order[60:80])
        hold.extend(group[i] for i in order[80:])
    return fit, val, hold


def paired_intervals(predictions, seeds):
    """Exploratory paired percentile CIs, stratified by project and true label.

    A separate CI is computed per holdout. Overlapping seeds are never pooled.
    These are conditional on the fitted models; not estimates of training noise.
    """
    intervals = []
    for seed in seeds:
        base = {p['source_row']: p for p in predictions if p['seed']==seed and p['method']=='raw'}
        ids = sorted(base)
        y = np.array([LABELS.index(base[i]['gold']) for i in ids])
        raw = np.array([LABELS.index(base[i]['predicted']) for i in ids])
        strata = collections.defaultdict(list)
        for j, i in enumerate(ids):
            strata[base[i]['repo'], base[i]['gold']].append(j)
        for method in sorted({p['method'] for p in predictions} - {'raw'}):
            comparison = {p['source_row']: p for p in predictions if p['seed']==seed and p['method']==method}
            other = np.array([LABELS.index(comparison[i]['predicted']) for i in ids])
            rng = np.random.RandomState(seed + 10000)
            ds = []
            for _ in range(500):
                ix = np.concatenate([rng.choice(js, len(js), replace=True) for js in strata.values()])
                ds.append(metrics(y[ix],other[ix])['macro_f1']-metrics(y[ix],raw[ix])['macro_f1'])
            intervals.append(dict(seed=seed, comparison=method+' minus raw',
                                  macro_f1_delta=metrics(y,other)['macro_f1']-metrics(y,raw)['macro_f1'],
                                  percentile95=np.percentile(ds,[2.5,97.5]).tolist(),
                                  raw_errors_corrected=int(((raw!=y)&(other==y)).sum()),
                                  raw_correct_broken=int(((raw==y)&(other!=y)).sum())))
    return intervals


def diagnostics(rows):
    out = []
    for repo in sorted({r["repo"] for r in rows}):
        for label in LABELS:
            subset = [r for r in rows if r["repo"] == repo and r["labels"] == label]
            out.append(dict(repo=repo, label=label, n=len(subset),
                            bug_form_count=sum(bool(BUG_FORM.search(r["body"])) for r in subset),
                            asking_title_count=sum(bool(ASK_TITLE.search(r["title"])) for r in subset),
                            min_date=min(r["created_at"] for r in subset),
                            max_date=max(r["created_at"] for r in subset),
                            median_date=sorted(r["created_at"] for r in subset)[len(subset)//2],
                            over_384_words=sum(len((r['title']+' '+r['body']).split())>384 for r in subset),
                            over_1000_words=sum(len((r['title']+' '+r['body']).split())>1000 for r in subset)))
    groups = collections.defaultdict(list)
    for r in rows:
        normalized = SPACE.sub(" ", (r["title"] + " " + r["body"]).lower()).strip()
        groups[hashlib.sha256(normalized.encode()).hexdigest()].append(r)
    duplicated = [group for group in groups.values() if len(group) > 1]
    return dict(project_label_characteristics=out,
                rows=len(rows), label_counts=dict(collections.Counter(r['labels'] for r in rows)),
                duplicate_groups=len(duplicated), duplicate_rows=sum(map(len, duplicated)),
                conflicting_duplicate_groups=sum(len({r["labels"] for r in g}) > 1 for g in duplicated),
                warning="Regex flags are surface-form proxies, not manually adjudicated template misuse.")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-csv", type=Path,
                        help="Required for the pilot: exact historical 3,300-row TRAIN split, verified by identity digest.")
    parser.add_argument("--source", type=Path,
                        help="Reconstruct the independently verified historical TRAIN from issues11k.csv.")
    parser.add_argument("--audit-only", type=Path,
                        help="Descriptive corpus audit only; no model or holdout evaluation.")
    parser.add_argument("--output", type=Path, default=Path("docs/research/bug_question_probe"))
    parser.add_argument("--seeds", default="17,42,71")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.audit_only:
        rows = load_rows(args.audit_only, verify_train=False)
        report = diagnostics(rows)
        (args.output / "corpus_audit.json").write_text(json.dumps(report, indent=2) + "\n")
        print(json.dumps(report, indent=2))
        return
    if bool(args.train_csv) == bool(args.source):
        parser.error("Supply exactly one of --train-csv or --source (or use --audit-only).")
    rows = load_rows(args.train_csv) if args.train_csv else reconstruct_train(args.source)
    methods = [("title", "title", False), ("raw", "raw", False), ("clean", "clean", False),
               ("raw_project", "raw", True), ("clean_project", "clean", True),
               ("dual", "dual", False), ("template_only", "template", False)]
    results, predictions = [], []
    for seed in map(int, args.seeds.split(",")):
        fit, val, hold = split_rows(rows, seed)
        templates = learn_templates(fit)
        yf, yv, yh = [np.array([LABELS.index(r["labels"]) for r in part]) for part in (fit, val, hold)]
        for name, view, project in methods:
            vec = Vectorizer(fit, "raw" if view == "dual" else view, templates)
            xf, xv, xh = [vec.transform(part, project) for part in (fit, val, hold)]
            if view == "dual":
                clean_vec = Vectorizer(fit, "clean", templates)
                xc = [clean_vec.transform(part) for part in (fit, val, hold)]
                # Equal-weight independent raw and cleaned views, preserving
                # potentially useful template information for the classifier.
                xf, xv, xh = [sparse.hstack((raw, clean), format="csr") / math.sqrt(2)
                              for raw, clean in zip((xf, xv, xh), xc)]
            candidates = []
            for alpha in (0.3, 1.0, 3.0, 10.0):
                pv = fit_predict(xf, yf, [xv], alpha)[0]
                candidates.append((metrics(yv, pv)["macro_f1"], alpha))
            _, alpha = max(candidates, key=lambda x: (x[0], -x[1]))
            # Keep fitting pool fixed after model selection; no holdout leakage.
            pv, ph = fit_predict(xf, yf, [xv, xh], alpha)
            result = dict(seed=seed, method=name, alpha=alpha, validation=metrics(yv, pv),
                          holdout=metrics(yh, ph), vocabulary_size=len(vec.vocab))
            results.append(result)
            for r, pred in zip(hold, ph):
                predictions.append(dict(seed=seed, method=name, source_row=r["source_row"],
                                        repo=r["repo"], gold=r["labels"], predicted=LABELS[int(pred)]))
            print(seed, name, "alpha", alpha, "validation", round(result["validation"]["macro_f1"], 4),
                  "holdout", round(result["holdout"]["macro_f1"], 4), flush=True)
    summary = {}
    for name, _, _ in methods:
        rs = [r for r in results if r["method"] == name]
        summary[name] = dict(mean_holdout_macro_f1=float(np.mean([r["holdout"]["macro_f1"] for r in rs])),
                             holdout_macro_f1=[r["holdout"]["macro_f1"] for r in rs],
                             mean_question_f1=float(np.mean([r["holdout"]["per_class_f1"]["question"] for r in rs])))
    report = dict(protocol="Historical TRAIN only; nested per-project/label 60/20/20 random splits. Each holdout has 660 rows.",
                  train_digest=identity_digest(rows), official_test_used=False, fit_rows=1980, validation_rows=660,
                  holdout_rows=660, seeds=list(map(int,args.seeds.split(","))),
                  numpy_version=np.__version__, scipy_version=scipy.__version__,
                  diagnostics=diagnostics(rows), runs=results, summary=summary,
                  paired_intervals=paired_intervals(predictions,list(map(int,args.seeds.split(',')))),
                  limitations=["Lexical probe, not neural/LLM performance estimate.",
                               "Three overlapping random holdouts are sensitivity checks, not independent replications.",
                               "Cleaning removes repeated lines and HTML comments; does not identify intent or prove causality.",
                               "Source/train identity digest must match an independently verified historical split."])
    (args.output / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    with (args.output / "predictions.csv").open("w") as f:
        writer = csv.DictWriter(f, fieldnames=list(predictions[0]))
        writer.writeheader(); writer.writerows(predictions)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
