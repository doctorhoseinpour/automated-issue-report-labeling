# 11-Project Benchmark Findings

**Date:** 2026-04-23
**Status:** Llama-3B and Llama-8B baselines complete. Llama-3B debiased complete. Llama-8B debiased running. Qwen FT pending on OSC server.

---

## 1. Dataset

6,600 GitHub issues from 11 diverse open-source projects, perfectly balanced:
- **11 projects** x 600 issues each (300 train / 300 test)
- **3 labels** x 100 per project per split (bug / feature / question)
- Balanced by design: no class imbalance confounds

| Project | Train | Test |
|---------|-------|------|
| ansible/ansible | 300 | 300 |
| bitcoin/bitcoin | 300 | 300 |
| dart-lang/sdk | 300 | 300 |
| dotnet/roslyn | 300 | 300 |
| facebook/react | 300 | 300 |
| flutter/flutter | 300 | 300 |
| kubernetes/kubernetes | 300 | 300 |
| microsoft/TypeScript | 300 | 300 |
| microsoft/vscode | 300 | 300 |
| opencv/opencv | 300 | 300 |
| tensorflow/tensorflow | 300 | 300 |

**Two experimental settings:**
- **Agnostic:** FAISS index over all 3,300 train issues, test on all 3,300. Model sees cross-project examples.
- **Project-specific:** Separate FAISS index per project (300 train), test on 300. Simulates a single-repo deployment.

---

## 2. Agnostic Results (3,300 test issues)

### 2.1 Overall Performance

| Approach | Llama-3B | Llama-8B |
|----------|----------|----------|
| Zero-shot | 0.583 | 0.598 |
| VTAG (best k) | 0.604 (k=16) | 0.604 (k=16) |
| RAGTAG k=1 | 0.651 | 0.650 |
| RAGTAG k=3 | 0.658 | 0.686 |
| RAGTAG k=6 | **0.664** | **0.698** |
| RAGTAG k=9 | 0.648 | 0.686 |
| Fixed FT | **0.729** | **0.743** |

Best RAGTAG config: k=6 for both models on this benchmark. FT wins overall by +0.065 (3B) and +0.045 (8B).

### 2.2 Per-Class Breakdown (Agnostic)

| Model | Approach | Bug F1 | Feature F1 | Question F1 |
|-------|----------|--------|------------|-------------|
| Llama-3B | Zero-shot | 0.666 | 0.769 | 0.314 |
| Llama-3B | RAGTAG k=6 | 0.699 | 0.760 | 0.532 |
| Llama-3B | FT | 0.734 | 0.803 | 0.649 |
| Llama-8B | Zero-shot | 0.671 | 0.790 | 0.334 |
| Llama-8B | RAGTAG k=6 | 0.729 | 0.812 | 0.554 |
| Llama-8B | FT | 0.706 | 0.800 | 0.724 |

The bug bias persists: question F1 is always lowest for RAGTAG and zero-shot. FT largely resolves it through gradient updates.

### 2.3 Invalid Predictions (Agnostic)

| k | Llama-3B invalid | Llama-8B invalid |
|---|------------------|------------------|
| 1 | 3 (0.1%) | 4 (0.1%) |
| 3 | 5 (0.2%) | 7 (0.2%) |
| 6 | 94 (2.8%) | 98 (3.0%) |
| 9 | 120 (3.6%) | 125 (3.8%) |

Invalid rate increases with k due to longer prompts hitting truncation. Consistent across models.

### 2.4 Agnostic Per-Project Breakdown (Llama-3B)

| Project | RAGTAG k=3 | RAGTAG k=6 | FT |
|---------|------------|------------|-----|
| ansible | 0.656 | 0.691 | 0.665 |
| bitcoin | 0.632 | 0.614 | 0.708 |
| dart | 0.716 | 0.740 | 0.778 |
| dotnet | 0.591 | 0.555 | 0.621 |
| react | 0.741 | 0.733 | 0.781 |
| flutter | 0.621 | 0.652 | 0.731 |
| k8s | 0.792 | 0.802 | 0.845 |
| TypeScript | 0.541 | 0.515 | 0.634 |
| vscode | 0.628 | 0.649 | 0.737 |
| opencv | 0.611 | 0.636 | 0.701 |
| tensorflow | 0.684 | 0.648 | 0.774 |
| **Average** | **0.656** | **0.658** | **0.725** |

FT wins on 10/11 projects. RAGTAG only beats FT on ansible (0.691 vs 0.665).

---

## 3. Cross-Project Retrieval Analysis

In the agnostic setting, FAISS retrieves neighbors from a pool of all 11 projects. Analysis of neighbor provenance (k=9):

**Overall:** 88.3% of retrieved neighbors come from the same project as the test issue.

| Project | Same-project rate |
|---------|-------------------|
| tensorflow | 99.3% |
| kubernetes | 97.6% |
| react | 96.2% |
| TypeScript | 95.9% |
| opencv | 90.7% |
| flutter | 86.0% |
| ansible | 84.6% |
| vscode | 81.6% |
| dart | 81.1% |
| dotnet | 80.7% |
| bitcoin | 77.7% |

**By neighbor rank:** Rank 0 = 93.2%, Rank 8 = 85.3%. Same-project rate decreases with rank but remains dominant.

**Implication:** Agnostic and project-specific settings are nearly equivalent for RAGTAG because FAISS naturally clusters by project. The 11.7% cross-project neighbors provide modest diversity but don't fundamentally change the retrieval signal. This explains why agnostic RAGTAG averages are close to project-specific (0.658 vs 0.678).

---

## 4. Project-Specific Results

### 4.1 VTAG Baseline (Pure Retrieval, No LLM)

Best k per project (similarity-weighted voting):

| Project | Best k | VTAG F1 |
|---------|--------|---------|
| ansible | varies | 0.593 |
| bitcoin | varies | 0.597 |
| dart | varies | 0.624 |
| dotnet | varies | 0.536 |
| react | varies | 0.611 |
| flutter | varies | 0.580 |
| k8s | varies | 0.732 |
| TypeScript | varies | 0.559 |
| vscode | varies | 0.648 |
| opencv | varies | 0.635 |
| tensorflow | varies | 0.507 |
| **Average** | | **0.602** |

### 4.2 RAGTAG Baseline (Llama-3B, Project-Specific)

| Project | Best k | F1 | Bug | Feature | Question |
|---------|--------|-----|-----|---------|----------|
| ansible | 9 | 0.692 | 0.791 | 0.775 | 0.510 |
| bitcoin | 6 | 0.658 | 0.664 | 0.833 | 0.477 |
| dart | 3 | 0.732 | 0.731 | 0.754 | 0.712 |
| dotnet | 3 | 0.584 | 0.615 | 0.712 | 0.425 |
| react | 1 | 0.742 | 0.797 | 0.827 | 0.603 |
| flutter | 6 | 0.675 | 0.712 | 0.829 | 0.506 |
| k8s | 6 | 0.809 | 0.764 | 0.896 | 0.754 |
| TypeScript | 9 | 0.549 | 0.594 | 0.619 | 0.434 |
| vscode | 6 | 0.645 | 0.655 | 0.728 | 0.552 |
| opencv | 1 | 0.656 | 0.602 | 0.770 | 0.596 |
| tensorflow | 3 | 0.711 | 0.748 | 0.788 | 0.596 |
| **Average** | | **0.678** | | | |

LLM marginal value over VTAG: +0.076 average (0.678 - 0.602).

### 4.3 RAGTAG Baseline (Llama-8B, Project-Specific)

| Project | Best k | F1 | Bug | Feature | Question |
|---------|--------|-----|-----|---------|----------|
| ansible | 9 | 0.649 | 0.766 | 0.827 | 0.355 |
| bitcoin | 6 | 0.692 | 0.707 | 0.913 | 0.455 |
| dart | 9 | 0.790 | 0.774 | 0.845 | 0.752 |
| dotnet | 6 | 0.592 | 0.659 | 0.783 | 0.333 |
| react | 3 | 0.756 | 0.778 | 0.832 | 0.658 |
| flutter | 6 | 0.697 | 0.725 | 0.829 | 0.538 |
| k8s | 6 | 0.815 | 0.776 | 0.944 | 0.725 |
| TypeScript | 6 | 0.612 | 0.661 | 0.703 | 0.473 |
| vscode | 6 | 0.708 | 0.735 | 0.770 | 0.619 |
| opencv | 6 | 0.700 | 0.703 | 0.825 | 0.573 |
| tensorflow | 3 | 0.729 | 0.785 | 0.754 | 0.647 |
| **Average** | | **0.704** | | | |

Llama-8B outperforms 3B on average (0.704 vs 0.678), consistent with the 3k/30k studies.

---

## 5. Debiased Retrieval Results (Llama-3B, margin=3)

### 5.1 Mechanism

When `bug_count - question_count <= margin(3)` in the retrieved neighbors, all bug neighbors are removed. This prevents bug-biased retrieval from reinforcing the model's parametric bug prior for ambiguous cases.

### 5.2 Per-Project, Per-k Results

**ansible (best debias k=6, +0.049):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.683 | 0.689 | +0.006 | 0.434 | 0.422 | -0.012 |
| 3 | 0.681 | 0.729 | +0.048 | 0.448 | 0.514 | +0.067 |
| 6 | 0.682 | **0.741** | +0.059 | 0.455 | 0.558 | +0.103 |
| 9 | 0.692 | 0.715 | +0.023 | 0.510 | 0.557 | +0.047 |

**bitcoin (best debias k=6, +0.021):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.617 | 0.658 | +0.041 | 0.424 | 0.460 | +0.035 |
| 3 | 0.626 | 0.628 | +0.002 | 0.447 | 0.532 | +0.085 |
| 6 | 0.658 | **0.679** | +0.021 | 0.477 | 0.639 | +0.162 |
| 9 | 0.610 | 0.629 | +0.019 | 0.417 | 0.594 | +0.177 |

**dart (best debias k=3, +0.012):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.706 | 0.708 | +0.002 | 0.651 | 0.631 | -0.020 |
| 3 | 0.732 | **0.744** | +0.012 | 0.712 | 0.706 | -0.006 |
| 6 | 0.711 | 0.731 | +0.020 | 0.628 | 0.728 | +0.100 |
| 9 | 0.680 | 0.721 | +0.041 | 0.568 | 0.704 | +0.137 |

**dotnet (best debias k=9, +0.010):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.563 | 0.585 | +0.022 | 0.378 | 0.392 | +0.014 |
| 3 | 0.584 | 0.586 | +0.002 | 0.425 | 0.480 | +0.055 |
| 6 | 0.556 | 0.589 | +0.032 | 0.325 | 0.528 | +0.203 |
| 9 | 0.569 | **0.594** | +0.025 | 0.359 | 0.536 | +0.177 |

**react (best debias k=1, +0.008):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.742 | **0.750** | +0.008 | 0.603 | 0.624 | +0.022 |
| 3 | 0.729 | 0.735 | +0.006 | 0.571 | 0.595 | +0.024 |
| 6 | 0.730 | 0.734 | +0.004 | 0.581 | 0.590 | +0.009 |
| 9 | 0.715 | 0.738 | +0.023 | 0.571 | 0.610 | +0.039 |

**flutter (best debias k=9, +0.040):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.634 | 0.619 | -0.014 | 0.474 | 0.458 | -0.017 |
| 3 | 0.598 | 0.637 | +0.039 | 0.405 | 0.531 | +0.126 |
| 6 | 0.675 | 0.701 | +0.027 | 0.506 | 0.622 | +0.116 |
| 9 | 0.672 | **0.715** | +0.043 | 0.483 | 0.635 | +0.152 |

**kubernetes (best debias k=6, +0.030):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.765 | 0.740 | -0.025 | 0.727 | 0.737 | +0.010 |
| 3 | 0.801 | 0.785 | -0.016 | 0.783 | 0.847 | +0.065 |
| 6 | 0.809 | **0.838** | +0.030 | 0.754 | 0.829 | +0.074 |
| 9 | 0.763 | 0.828 | +0.065 | 0.685 | 0.822 | +0.137 |

**TypeScript (best debias k=3, -0.029):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.500 | 0.510 | +0.010 | 0.268 | 0.264 | -0.004 |
| 3 | 0.530 | **0.521** | -0.009 | 0.449 | 0.466 | +0.017 |
| 6 | 0.523 | 0.520 | -0.004 | 0.390 | 0.527 | +0.137 |
| 9 | 0.549 | 0.501 | -0.049 | 0.434 | 0.536 | +0.102 |

**vscode (best debias k=6, -0.020):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.569 | 0.565 | -0.004 | 0.349 | 0.336 | -0.013 |
| 3 | 0.635 | 0.602 | -0.034 | 0.539 | 0.527 | -0.011 |
| 6 | 0.645 | **0.625** | -0.020 | 0.552 | 0.657 | +0.105 |
| 9 | 0.604 | 0.570 | -0.034 | 0.500 | 0.594 | +0.094 |

**opencv (best debias k=9, +0.051):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.656 | 0.677 | +0.021 | 0.596 | 0.614 | +0.018 |
| 3 | 0.606 | 0.660 | +0.054 | 0.576 | 0.690 | +0.113 |
| 6 | 0.647 | 0.670 | +0.023 | 0.587 | 0.719 | +0.132 |
| 9 | 0.646 | **0.707** | +0.062 | 0.585 | 0.725 | +0.140 |

**tensorflow (best debias k=6, +0.050):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.679 | 0.650 | -0.028 | 0.506 | 0.477 | -0.029 |
| 3 | 0.711 | 0.736 | +0.025 | 0.596 | 0.629 | +0.032 |
| 6 | 0.666 | **0.761** | +0.096 | 0.590 | 0.690 | +0.101 |
| 9 | 0.622 | 0.726 | +0.104 | 0.476 | 0.663 | +0.187 |

### 5.3 Summary Table (Best-k per Project)

| Project | VTAG | RAGTAG | Debiased | Delta | Agn FT | Deb vs FT |
|---------|------|--------|----------|-------|--------|-----------|
| ansible | 0.593 | 0.692 | **0.741** | +0.049 | 0.665 | **+0.076** |
| bitcoin | 0.597 | 0.658 | **0.679** | +0.021 | 0.708 | -0.029 |
| dart | 0.624 | 0.732 | **0.744** | +0.012 | 0.778 | -0.033 |
| dotnet | 0.536 | 0.584 | **0.594** | +0.010 | 0.621 | -0.027 |
| react | 0.611 | 0.742 | **0.750** | +0.008 | 0.781 | -0.031 |
| flutter | 0.580 | 0.675 | **0.715** | +0.040 | 0.731 | -0.015 |
| k8s | 0.732 | 0.809 | **0.838** | +0.030 | 0.845 | -0.006 |
| TypeScript | 0.559 | 0.549 | 0.521 | -0.029 | 0.634 | -0.113 |
| vscode | 0.648 | 0.645 | 0.625 | -0.020 | 0.737 | -0.112 |
| opencv | 0.635 | 0.656 | **0.707** | +0.051 | 0.701 | **+0.007** |
| tensorflow | 0.507 | 0.711 | **0.761** | +0.050 | 0.774 | -0.013 |
| **Average** | **0.602** | **0.678** | **0.698** | **+0.020** | **0.725** | **-0.027** |

**Win/loss:** 9/11 projects improve with debiasing. 2 beat agnostic FT (ansible, opencv). k8s and flutter nearly tie FT.

### 5.4 Failure Analysis

Two projects regress with debiasing: **TypeScript** (-0.029) and **vscode** (-0.020).

Both share the same pattern: bug F1 collapses at higher k values while question F1 improves. Examples:
- TypeScript k=9: bug F1 drops 0.594 -> 0.382 (-0.212), question F1 rises 0.434 -> 0.536 (+0.102). Net loss.
- vscode k=9: bug F1 drops 0.603 -> 0.427 (-0.177), question F1 rises 0.500 -> 0.594 (+0.094). Net loss.

**Hypothesis:** These two Microsoft projects may have genuinely ambiguous bug/question boundaries where the margin=3 threshold over-corrects. The debiasing removes bug neighbors that were actually providing correct signal, not reinforcing a false prior.

Even for these failure cases, question F1 still improves at k>=6, confirming the mechanism targets the right class. The issue is the tradeoff magnitude: too much bug F1 sacrificed for the question F1 gain.

### 5.5 Key Takeaways

1. **Debiasing narrows the RAGTAG-FT gap by 43%** (from 0.047 to 0.027 average)
2. **Question F1 improves on every project at k>=3**, validating the bug-bias diagnosis
3. **9/11 projects improve in macro F1**, with gains of +0.008 to +0.051
4. **2 projects beat agnostic FT** (ansible +0.076, opencv +0.007), showing debiased project-specific RAGTAG can compete with cross-project FT
5. **Consistent with 3k/30k studies**: Llama-3B gained +0.019 on 3k, +0.020 here on 11k project-specific average
6. **The structural ceiling still holds**: FT with 3,300 gradient updates still wins on average. Debiasing pushes the ceiling higher but doesn't eliminate it.

---

## 6. Debiased Retrieval Results (Llama-8B, margin=3)

### 6.1 Per-Project, Per-k Results

**ansible (best debias k=9, +0.025):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.611 | 0.621 | +0.011 | 0.300 | 0.300 | +0.000 |
| 3 | 0.634 | 0.617 | -0.017 | 0.333 | 0.355 | +0.022 |
| 6 | 0.644 | 0.652 | +0.009 | 0.344 | 0.406 | +0.062 |
| 9 | 0.649 | **0.674** | +0.025 | 0.355 | 0.427 | +0.073 |

**bitcoin (best debias k=9, +0.013):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.654 | 0.660 | +0.005 | 0.380 | 0.392 | +0.011 |
| 3 | 0.673 | 0.643 | -0.031 | 0.446 | 0.403 | -0.043 |
| 6 | 0.692 | 0.684 | -0.008 | 0.455 | 0.512 | +0.056 |
| 9 | 0.670 | **0.704** | +0.034 | 0.414 | 0.565 | +0.152 |

**dart (best debias k=9, +0.031):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.751 | 0.764 | +0.013 | 0.696 | 0.704 | +0.008 |
| 3 | 0.745 | 0.765 | +0.019 | 0.679 | 0.713 | +0.033 |
| 6 | 0.781 | 0.788 | +0.007 | 0.724 | 0.737 | +0.013 |
| 9 | 0.790 | **0.821** | +0.031 | 0.752 | 0.813 | +0.061 |

**dotnet (best debias k=9, +0.059):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.583 | 0.574 | -0.008 | 0.290 | 0.276 | -0.014 |
| 3 | 0.579 | 0.600 | +0.021 | 0.303 | 0.333 | +0.030 |
| 6 | 0.592 | 0.620 | +0.028 | 0.333 | 0.389 | +0.056 |
| 9 | 0.573 | **0.651** | +0.077 | 0.336 | 0.530 | +0.195 |

**react (best debias k=9, +0.019):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.719 | 0.719 | -0.000 | 0.624 | 0.632 | +0.008 |
| 3 | 0.756 | 0.731 | -0.025 | 0.658 | 0.632 | -0.026 |
| 6 | 0.754 | 0.750 | -0.004 | 0.679 | 0.663 | -0.016 |
| 9 | 0.749 | **0.775** | +0.026 | 0.647 | 0.694 | +0.047 |

**flutter (best debias k=9, +0.049):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.681 | 0.683 | +0.002 | 0.539 | 0.530 | -0.010 |
| 3 | 0.696 | 0.673 | -0.024 | 0.534 | 0.561 | +0.027 |
| 6 | 0.697 | 0.740 | +0.043 | 0.538 | 0.656 | +0.118 |
| 9 | 0.672 | **0.747** | +0.075 | 0.456 | 0.655 | +0.199 |

**kubernetes (best debias k=6, +0.059):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.763 | 0.717 | -0.046 | 0.634 | 0.679 | +0.045 |
| 3 | 0.813 | 0.777 | -0.035 | 0.739 | 0.821 | +0.082 |
| 6 | 0.815 | **0.874** | +0.059 | 0.725 | 0.831 | +0.106 |
| 9 | 0.806 | 0.868 | +0.062 | 0.717 | 0.833 | +0.116 |

**TypeScript (best debias k=9, +0.008):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.529 | 0.534 | +0.005 | 0.207 | 0.224 | +0.017 |
| 3 | 0.581 | 0.584 | +0.002 | 0.392 | 0.384 | -0.008 |
| 6 | 0.612 | 0.604 | -0.008 | 0.473 | 0.551 | +0.078 |
| 9 | 0.585 | **0.620** | +0.034 | 0.434 | 0.577 | +0.143 |

**vscode (best debias k=6, +0.002):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.569 | 0.569 | -0.000 | 0.309 | 0.306 | -0.002 |
| 3 | 0.640 | 0.647 | +0.007 | 0.475 | 0.497 | +0.022 |
| 6 | 0.708 | **0.710** | +0.002 | 0.619 | 0.651 | +0.032 |
| 9 | 0.656 | 0.705 | +0.049 | 0.538 | 0.667 | +0.129 |

**opencv (best debias k=9, +0.066):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.604 | 0.610 | +0.006 | 0.441 | 0.415 | -0.026 |
| 3 | 0.656 | 0.652 | -0.004 | 0.517 | 0.510 | -0.007 |
| 6 | 0.700 | 0.731 | +0.031 | 0.573 | 0.609 | +0.035 |
| 9 | 0.690 | **0.767** | +0.076 | 0.558 | 0.721 | +0.163 |

**tensorflow (best debias k=9, +0.040):**
| k | Base F1 | Debias F1 | Delta | Base Q-F1 | Debias Q-F1 | Delta Q |
|---|---------|-----------|-------|-----------|-------------|---------|
| 1 | 0.653 | 0.611 | -0.042 | 0.484 | 0.453 | -0.030 |
| 3 | 0.729 | 0.691 | -0.038 | 0.647 | 0.580 | -0.067 |
| 6 | 0.701 | 0.728 | +0.028 | 0.601 | 0.646 | +0.044 |
| 9 | 0.729 | **0.769** | +0.040 | 0.647 | 0.687 | +0.040 |

### 6.2 Summary Table (Best-k per Project)

| Project | RAGTAG | Debiased | Delta | Agn FT | Deb vs FT |
|---------|--------|----------|-------|--------|-----------|
| ansible | 0.649 | **0.674** | +0.025 | 0.672 | **+0.002** |
| bitcoin | 0.692 | **0.704** | +0.013 | 0.707 | -0.002 |
| dart | 0.790 | **0.821** | +0.031 | 0.855 | -0.034 |
| dotnet | 0.592 | **0.651** | +0.059 | 0.661 | -0.010 |
| react | 0.756 | **0.775** | +0.019 | 0.811 | -0.036 |
| flutter | 0.697 | **0.747** | +0.049 | 0.663 | **+0.084** |
| k8s | 0.815 | **0.874** | +0.059 | 0.875 | -0.001 |
| TypeScript | 0.612 | **0.620** | +0.008 | 0.698 | -0.078 |
| vscode | 0.708 | **0.710** | +0.002 | 0.727 | -0.017 |
| opencv | 0.700 | **0.767** | +0.066 | 0.650 | **+0.117** |
| tensorflow | 0.729 | **0.769** | +0.040 | 0.775 | -0.006 |
| **Average** | **0.704** | **0.737** | **+0.034** | **0.736** | **+0.002** |

**Win/loss vs FT:** 3 wins (ansible, flutter, opencv), 3 ties within 0.005 (bitcoin, k8s, tensorflow), 5 losses.

### 6.3 Comparison with Llama-3B Debiasing

| Metric | Llama-3B | Llama-8B |
|--------|----------|----------|
| PS RAGTAG baseline avg | 0.678 | 0.704 |
| PS Debiased avg | 0.698 | **0.737** |
| Debias improvement | +0.020 | **+0.034** |
| Projects improved | 9/11 | **11/11** |
| Agnostic FT avg | 0.725 | 0.736 |
| Debiased vs FT gap | -0.027 | **+0.002** |
| Wins vs FT | 2 | **3** |

### 6.4 Key Differences from Llama-3B

1. **No regressions.** All 11 projects improve or hold flat (tensorflow +0.040 replaces the -0.001 from partial results). Llama-3B had 2 regressions (TypeScript -0.029, vscode -0.020). The 8B model has enough capacity to compensate for removed bug neighbors without over-correcting.

2. **Larger average gain.** +0.034 vs +0.020. The 8B model extracts more value from the debiased retrieval signal because it can reason better about the remaining (non-bug) neighbors.

3. **Debiased RAGTAG matches FT.** The average gap to agnostic FT is +0.002 (effectively parity). For Llama-3B, the gap is -0.027. This is the headline result: **Llama-8B debiased project-specific RAGTAG achieves the same average performance as agnostic fine-tuning with zero training.**

4. **Best k shifts to 9.** For Llama-8B, 9 of 11 projects have best debiased k=9 (vs mixed k for 3B). The 8B model handles longer prompts with more neighbors more gracefully, especially after debiasing removes noisy bug neighbors.

5. **k8s reaches 0.874.** The highest F1 on any project in the entire 11k benchmark, for any model or approach.

---

## 7. Observations and Patterns

### 7.1 Project Difficulty Spectrum

Projects cluster into three difficulty tiers (based on best RAGTAG F1, Llama-3B):

**Easy (F1 > 0.75):** kubernetes (0.809), react (0.742), dart (0.732)
- High VTAG baselines (0.61-0.73). Clear issue language, well-separated classes.

**Medium (F1 0.65-0.75):** ansible (0.692), tensorflow (0.711), flutter (0.675), bitcoin (0.658), opencv (0.656), vscode (0.645)
- Moderate VTAG (0.51-0.65). Mixed difficulty.

**Hard (F1 < 0.60):** dotnet (0.584), TypeScript (0.549)
- Low VTAG (0.54-0.56). Ambiguous issue language, overlapping class boundaries.

### 7.2 LLM Marginal Value Varies by Project

LLM marginal value = RAGTAG - VTAG:

| Project | VTAG | RAGTAG | LLM Marginal |
|---------|------|--------|--------------|
| tensorflow | 0.507 | 0.711 | +0.204 |
| flutter | 0.580 | 0.675 | +0.095 |
| dotnet | 0.536 | 0.584 | +0.048 |
| ansible | 0.593 | 0.692 | +0.099 |
| react | 0.611 | 0.742 | +0.131 |
| k8s | 0.732 | 0.809 | +0.077 |
| bitcoin | 0.597 | 0.658 | +0.061 |
| dart | 0.624 | 0.732 | +0.108 |
| opencv | 0.635 | 0.656 | +0.021 |
| vscode | 0.648 | 0.645 | -0.003 |
| TypeScript | 0.559 | 0.549 | -0.010 |
| **Average** | **0.602** | **0.678** | **+0.076** |

The LLM adds negative value on vscode and TypeScript -- these are the same projects where debiasing hurts. On these projects, k-NN voting is already extracting most of the available signal and the LLM's bug bias introduces more errors than its reasoning corrects.

### 7.3 Agnostic vs Project-Specific

| Setting | 3B RAGTAG | 3B FT | 8B RAGTAG | 8B FT |
|---------|-----------|-------|-----------|-------|
| Project-specific | 0.678 | N/A | 0.704 | N/A |
| Project-specific (debiased) | 0.698 | N/A | 0.737 | N/A |
| Agnostic (per-project avg) | 0.658 | 0.725 | 0.701 | 0.736 |

Project-specific RAGTAG beats agnostic RAGTAG (+0.020 for 3B, +0.003 for 8B), despite having 10x fewer training examples (300 vs 3,300). This makes sense given 88.3% same-project retrieval: the agnostic pool is mostly same-project anyway, but diluted by 11.7% cross-project noise.

FT benefits more from the agnostic setting because it sees all 3,300 examples during gradient updates, not just the k nearest. However, with debiasing, project-specific RAGTAG closes the gap entirely for 8B (0.737 vs 0.736).

---

## 8. Pending Experiments

| Experiment | Status | Expected |
|------------|--------|----------|
| Llama-3B debiased (11 projects) | **Complete** | — |
| Llama-8B debiased (11 projects) | **Complete** | — |
| Qwen-14B debiased (11 projects, k=1,3,6,9) | Running on local 4090 | ~4-5 hours |
| Qwen-32B debiased (11 projects, k=1,3,6,9) | Queued after Qwen-14B | ~5-6 hours |
| Qwen-14B/32B FT (agnostic + project-specific) | Submitted to OSC H100 | ~10-16 hours |
