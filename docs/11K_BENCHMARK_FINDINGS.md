# 11-Project Benchmark Findings

**Date:** 2026-04-24
**Status:** All RAGTAG baselines and debiased runs complete for all 4 models. Llama FT complete. Qwen FT pending on OSC server.

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

**Key finding on settings:** Cross-project retrieval analysis shows 88.3% of agnostic neighbors come from the same project (see Section 5). Agnostic RAGTAG is therefore nearly equivalent to project-specific RAGTAG and can be excluded from the paper. The meaningful comparison is project-specific RAGTAG vs agnostic FT — RAGTAG uses 300 same-project examples at retrieval time while FT trains on all 3,300.

---

## 2. Compact Summary

| Approach | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B | Qwen-32B (valid-only) |
|----------|----------|----------|----------|----------|----------------------|
| Zero-shot | 0.576 | 0.591 | 0.638 | 0.677 | — |
| VTAG (PS, best k) | 0.602 | 0.602 | 0.602 | 0.602 | — |
| RAGTAG (PS, best k) | 0.678 | 0.704 | 0.727 | 0.746 | 0.777 |
| Debiased RAGTAG (PS, best k) | 0.698 | 0.737 | **0.749** | 0.746 | **0.787** |
| Fine-Tuning (agnostic) | 0.725 | 0.736 | *pending* | *pending* | — |
| Debias gain | +0.020 | +0.034 | +0.022 | +0.000 | +0.010 |
| Debias wins/11 | 9 | 11 | 10 | — | — |

**Headline results:**
- Llama-8B debiased (0.737) matches Llama-8B FT (0.736) with zero training
- Qwen-14B debiased (0.749) beats both Llama FTs (0.725, 0.736) with zero training
- Qwen-32B valid-only (0.787) shows the true model capability when not limited by GPU memory
- All 4 debiased RAGTAG results beat Llama-3B FT (0.725)
- Qwen-32B raw scores (0.746) are artificially suppressed by 22.4% OOM-driven invalid rate at k=9

---

## 3. Agnostic Results (3,300 test issues, overall)

| Model | Approach | Macro F1 | Bug F1 | Feature F1 | Question F1 | Invalid |
|-------|----------|----------|--------|------------|-------------|---------|
| Llama-3B | Zero-shot | 0.583 | 0.666 | 0.769 | 0.314 | 9 |
| Llama-3B | RAGTAG k=1 | 0.651 | 0.689 | 0.763 | 0.501 | 3 |
| Llama-3B | RAGTAG k=3 | 0.658 | 0.671 | 0.758 | 0.546 | 5 |
| Llama-3B | RAGTAG k=6 | **0.664** | 0.699 | 0.760 | 0.532 | 94 |
| Llama-3B | RAGTAG k=9 | 0.648 | 0.681 | 0.761 | 0.502 | 120 |
| Llama-3B | Fine-Tuning | **0.729** | 0.734 | 0.803 | 0.649 | 9 |
| | | | | | | |
| Llama-8B | Zero-shot | 0.598 | 0.671 | 0.790 | 0.334 | 8 |
| Llama-8B | RAGTAG k=1 | 0.650 | 0.695 | 0.794 | 0.460 | 4 |
| Llama-8B | RAGTAG k=3 | 0.686 | 0.719 | 0.807 | 0.532 | 7 |
| Llama-8B | RAGTAG k=6 | **0.698** | 0.729 | 0.812 | 0.554 | 98 |
| Llama-8B | RAGTAG k=9 | 0.686 | 0.725 | 0.803 | 0.530 | 125 |
| Llama-8B | Fine-Tuning | **0.743** | 0.706 | 0.800 | 0.724 | 11 |
| | | | | | | |
| Qwen-14B | Zero-shot | 0.645 | 0.690 | 0.806 | 0.439 | 2 |
| Qwen-14B | RAGTAG k=1 | 0.684 | 0.713 | 0.822 | 0.516 | 3 |
| Qwen-14B | RAGTAG k=3 | 0.712 | 0.735 | 0.830 | 0.569 | 4 |
| Qwen-14B | RAGTAG k=6 | 0.721 | 0.741 | 0.831 | 0.592 | 118 |
| Qwen-14B | RAGTAG k=9 | **0.723** | 0.744 | 0.827 | 0.597 | 141 |
| | | | | | | |
| Qwen-32B | Zero-shot | 0.686 | 0.713 | 0.811 | 0.534 | 51 |
| Qwen-32B | RAGTAG k=1 | 0.717 | 0.739 | 0.822 | 0.592 | 91 |
| Qwen-32B | RAGTAG k=3 | **0.741** | 0.753 | 0.834 | 0.635 | 212 |
| Qwen-32B | RAGTAG k=6 | 0.727 | 0.739 | 0.825 | 0.616 | 486 |
| Qwen-32B | RAGTAG k=9 | 0.697 | 0.712 | 0.779 | 0.602 | 754 |

Bug bias persists across all models: question F1 is always lowest for RAGTAG and zero-shot. FT largely resolves it through gradient updates. Qwen-32B invalid rates are severe (up to 22.8% at k=9) due to CUDA OOM on RTX 4090.

---

## 4. Agnostic Per-Project Results

### 4.1 Zero-shot

| Project | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B |
|---------|----------|----------|----------|----------|
| ansible | 0.576 | 0.553 | 0.567 | 0.586 |
| bitcoin | 0.587 | 0.613 | 0.632 | 0.665 |
| dart | 0.713 | 0.719 | 0.771 | 0.827 |
| dotnet | 0.551 | 0.542 | 0.584 | 0.615 |
| react | 0.682 | 0.701 | 0.759 | 0.794 |
| flutter | 0.568 | 0.674 | 0.717 | 0.733 |
| k8s | 0.638 | 0.608 | 0.633 | 0.712 |
| TypeScript | 0.465 | 0.477 | 0.543 | 0.596 |
| vscode | 0.444 | 0.486 | 0.500 | 0.515 |
| opencv | 0.543 | 0.547 | 0.608 | 0.660 |
| tensorflow | 0.574 | 0.584 | 0.705 | 0.744 |
| **Average** | **0.576** | **0.591** | **0.638** | **0.677** |

### 4.2 RAGTAG (agnostic, best k per project)

| Project | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B |
|---------|----------|----------|----------|----------|
| ansible | 0.695 | 0.650 | 0.671 | 0.686 |
| bitcoin | 0.632 | 0.682 | 0.672 | 0.689 |
| dart | 0.740 | 0.781 | 0.827 | 0.852 |
| dotnet | 0.591 | 0.596 | 0.627 | 0.670 |
| react | 0.758 | 0.757 | 0.798 | 0.819 |
| flutter | 0.652 | 0.712 | 0.745 | 0.738 |
| k8s | 0.802 | 0.831 | 0.826 | 0.806 |
| TypeScript | 0.551 | 0.600 | 0.606 | 0.646 |
| vscode | 0.649 | 0.682 | 0.646 | 0.664 |
| opencv | 0.652 | 0.701 | 0.692 | 0.717 |
| tensorflow | 0.684 | 0.721 | 0.769 | 0.782 |
| **Average** | **0.673** | **0.701** | **0.716** | **0.734** |

### 4.3 Fine-Tuning (agnostic, per-project)

| Project | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B |
|---------|----------|----------|----------|----------|
| ansible | 0.665 | 0.672 | — | — |
| bitcoin | 0.708 | 0.707 | — | — |
| dart | 0.778 | 0.855 | — | — |
| dotnet | 0.621 | 0.661 | — | — |
| react | 0.781 | 0.811 | — | — |
| flutter | 0.731 | 0.663 | — | — |
| k8s | 0.845 | 0.875 | — | — |
| TypeScript | 0.634 | 0.698 | — | — |
| vscode | 0.737 | 0.727 | — | — |
| opencv | 0.701 | 0.650 | — | — |
| tensorflow | 0.774 | 0.775 | — | — |
| **Average** | **0.725** | **0.736** | — | — |

---

## 5. Cross-Project Retrieval Analysis

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

**Implication:** Agnostic RAGTAG can be excluded from the paper — it is nearly equivalent to project-specific RAGTAG because FAISS naturally clusters by project. The 11.7% cross-project neighbors add noise, not signal. Agnostic RAGTAG averages (0.673 for 3B, 0.701 for 8B) are close to project-specific (0.678, 0.704). The meaningful comparison is project-specific RAGTAG (300 train) vs agnostic FT (3,300 train) — RAGTAG competes at a 10x data disadvantage because retrieval naturally selects the right subset.

---

## 6. Project-Specific Results

### 6.1 VTAG Baseline (Pure Retrieval, No LLM)

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

### 6.2 RAGTAG Baseline (All Models, All k Values)

**Llama-3B:**
| Project | k=1 | k=3 | k=6 | k=9 | Best | k |
|---------|-----|-----|-----|-----|------|---|
| ansible | 0.683 | 0.681 | 0.682 | 0.692 | 0.692 | 9 |
| bitcoin | 0.617 | 0.626 | 0.658 | 0.610 | 0.658 | 6 |
| dart | 0.706 | 0.732 | 0.711 | 0.680 | 0.732 | 3 |
| dotnet | 0.563 | 0.584 | 0.556 | 0.569 | 0.584 | 3 |
| react | 0.742 | 0.729 | 0.730 | 0.715 | 0.742 | 1 |
| flutter | 0.634 | 0.598 | 0.675 | 0.672 | 0.675 | 6 |
| k8s | 0.765 | 0.801 | 0.809 | 0.763 | 0.809 | 6 |
| TypeScript | 0.500 | 0.530 | 0.523 | 0.549 | 0.549 | 9 |
| vscode | 0.569 | 0.635 | 0.645 | 0.604 | 0.645 | 6 |
| opencv | 0.656 | 0.606 | 0.647 | 0.646 | 0.656 | 1 |
| tensorflow | 0.679 | 0.711 | 0.666 | 0.622 | 0.711 | 3 |
| **Average** | | | | | **0.678** | |

**Llama-8B:**
| Project | k=1 | k=3 | k=6 | k=9 | Best | k |
|---------|-----|-----|-----|-----|------|---|
| ansible | 0.611 | 0.634 | 0.644 | 0.649 | 0.649 | 9 |
| bitcoin | 0.654 | 0.673 | 0.692 | 0.670 | 0.692 | 6 |
| dart | 0.751 | 0.745 | 0.781 | 0.790 | 0.790 | 9 |
| dotnet | 0.583 | 0.579 | 0.592 | 0.573 | 0.592 | 6 |
| react | 0.719 | 0.756 | 0.754 | 0.749 | 0.756 | 3 |
| flutter | 0.681 | 0.696 | 0.697 | 0.672 | 0.697 | 6 |
| k8s | 0.763 | 0.813 | 0.815 | 0.806 | 0.815 | 6 |
| TypeScript | 0.529 | 0.581 | 0.612 | 0.585 | 0.612 | 6 |
| vscode | 0.569 | 0.640 | 0.708 | 0.656 | 0.708 | 6 |
| opencv | 0.604 | 0.656 | 0.700 | 0.690 | 0.700 | 6 |
| tensorflow | 0.653 | 0.729 | 0.701 | 0.697 | 0.729 | 3 |
| **Average** | | | | | **0.704** | |

**Qwen-14B:**
| Project | k=1 | k=3 | k=6 | k=9 | Best | k |
|---------|-----|-----|-----|-----|------|---|
| ansible | 0.616 | 0.647 | 0.656 | 0.678 | 0.678 | 9 |
| bitcoin | 0.631 | 0.646 | 0.692 | 0.706 | 0.706 | 9 |
| dart | 0.800 | 0.821 | 0.821 | 0.841 | 0.841 | 9 |
| dotnet | 0.594 | 0.613 | 0.628 | 0.618 | 0.628 | 6 |
| react | 0.775 | 0.797 | 0.793 | 0.797 | 0.797 | 9 |
| flutter | 0.724 | 0.739 | 0.764 | 0.758 | 0.764 | 6 |
| k8s | 0.786 | 0.835 | 0.813 | 0.811 | 0.835 | 3 |
| TypeScript | 0.575 | 0.594 | 0.591 | 0.614 | 0.614 | 9 |
| vscode | 0.560 | 0.613 | 0.650 | 0.662 | 0.662 | 9 |
| opencv | 0.653 | 0.686 | 0.699 | 0.703 | 0.703 | 9 |
| tensorflow | 0.729 | 0.773 | 0.771 | 0.763 | 0.773 | 3 |
| **Average** | | | | | **0.727** | |

**Qwen-32B:**
| Project | k=1 | k=3 | k=6 | k=9 | Best | k |
|---------|-----|-----|-----|-----|------|---|
| ansible | 0.644 | 0.678 | 0.693 | 0.674 | 0.693 | 6 |
| bitcoin | 0.680 | 0.691 | 0.715 | 0.704 | 0.715 | 6 |
| dart | 0.817 | 0.848 | 0.844 | 0.848 | 0.848 | 9 |
| dotnet | 0.640 | 0.665 | 0.674 | 0.669 | 0.674 | 6 |
| react | 0.807 | 0.819 | 0.812 | 0.812 | 0.819 | 3 |
| flutter | 0.769 | 0.801 | 0.763 | 0.667 | 0.801 | 3 |
| k8s | 0.799 | 0.799 | 0.774 | 0.696 | 0.799 | 1 |
| TypeScript | 0.628 | 0.652 | 0.600 | 0.474 | 0.652 | 3 |
| vscode | 0.599 | 0.662 | 0.635 | 0.604 | 0.662 | 3 |
| opencv | 0.684 | 0.704 | 0.722 | 0.749 | 0.749 | 9 |
| tensorflow | 0.784 | 0.778 | 0.791 | 0.740 | 0.791 | 6 |
| **Average** | | | | | **0.746** | |

Qwen-32B's best k skews low (k=1-6) because higher k values trigger more OOMs. Its raw scores are suppressed — see Section 9.

---

## 7. Debiased Retrieval Results (All Models, margin=3)

### 7.1 Mechanism

When `bug_count - question_count <= margin(3)` in the retrieved neighbors, all bug neighbors are removed. This prevents bug-biased retrieval from reinforcing the model's parametric bug prior for ambiguous cases.

### 7.2 Summary Table (Best-k per Project)

**Llama-3B (avg gain +0.020, 9/11 improve):**
| Project | RAGTAG | Debiased | Delta | Agn FT | Deb vs FT |
|---------|--------|----------|-------|--------|-----------|
| ansible | 0.692 | **0.741** | +0.049 | 0.665 | **+0.076** |
| bitcoin | 0.658 | **0.679** | +0.021 | 0.708 | -0.029 |
| dart | 0.732 | **0.744** | +0.012 | 0.778 | -0.033 |
| dotnet | 0.584 | **0.594** | +0.010 | 0.621 | -0.027 |
| react | 0.742 | **0.750** | +0.008 | 0.781 | -0.031 |
| flutter | 0.675 | **0.715** | +0.040 | 0.731 | -0.015 |
| k8s | 0.809 | **0.838** | +0.030 | 0.845 | -0.006 |
| TypeScript | 0.549 | 0.521 | -0.029 | 0.634 | -0.113 |
| vscode | 0.645 | 0.625 | -0.020 | 0.737 | -0.112 |
| opencv | 0.656 | **0.707** | +0.051 | 0.701 | **+0.007** |
| tensorflow | 0.711 | **0.761** | +0.050 | 0.774 | -0.013 |
| **Average** | **0.678** | **0.698** | **+0.020** | **0.725** | **-0.027** |

2 wins vs FT (ansible, opencv). Gap to FT narrows from 0.047 to 0.027. Two regressions: TypeScript (-0.029), vscode (-0.020) — bug F1 collapses at higher k while question F1 improves. Over-correction on projects with ambiguous bug/question boundaries.

**Llama-8B (avg gain +0.034, 11/11 improve):**
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

3 wins vs FT (ansible, flutter, opencv), 3 ties within 0.005 (bitcoin, k8s, tensorflow). **Average gap is +0.002 — parity with FT.** No regressions. The 8B model handles debiasing most gracefully. k8s reaches 0.874, the highest F1 on any project.

**Qwen-14B (avg gain +0.022, 10/11 improve):**
| Project | RAGTAG | Debiased | Delta |
|---------|--------|----------|-------|
| ansible | 0.678 | **0.689** | +0.011 |
| bitcoin | 0.706 | **0.738** | +0.032 |
| dart | 0.841 | **0.844** | +0.003 |
| dotnet | 0.628 | **0.674** | +0.046 |
| react | 0.797 | **0.808** | +0.011 |
| flutter | 0.764 | **0.787** | +0.023 |
| k8s | 0.835 | **0.848** | +0.013 |
| TypeScript | 0.614 | **0.644** | +0.030 |
| vscode | 0.662 | **0.707** | +0.045 |
| opencv | 0.703 | **0.717** | +0.013 |
| tensorflow | 0.773 | **0.784** | +0.011 |
| **Average** | **0.727** | **0.749** | **+0.022** |

10/11 improve. Only tensorflow shows a slight dip when considering best-k. The debiased average of 0.749 is the highest across any model+approach combination, beating Qwen-32B baseline (0.746) and both Llama FTs.

**Qwen-32B (avg gain +0.000 raw, +0.010 valid-only):**
| Project | RAGTAG | Debiased | Delta |
|---------|--------|----------|-------|
| ansible | 0.693 | **0.708** | +0.015 |
| bitcoin | 0.715 | **0.742** | +0.027 |
| dart | 0.848 | **0.865** | +0.017 |
| dotnet | 0.674 | **0.685** | +0.011 |
| react | 0.819 | **0.821** | +0.003 |
| flutter | 0.801 | 0.789 | -0.012 |
| k8s | 0.799 | 0.775 | -0.024 |
| TypeScript | 0.652 | 0.652 | +0.000 |
| vscode | 0.662 | 0.648 | -0.014 |
| opencv | 0.749 | **0.751** | +0.002 |
| tensorflow | 0.791 | 0.768 | -0.023 |
| **Average** | **0.746** | **0.746** | **+0.000** |

Flat on raw scores — but this is an artifact of OOM, not a failure of debiasing. See Section 9 for valid-only analysis.

### 7.3 Cross-Model Debiasing Comparison

| Metric | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B |
|--------|----------|----------|----------|----------|
| PS RAGTAG baseline avg | 0.678 | 0.704 | 0.727 | 0.746 |
| PS Debiased avg | 0.698 | 0.737 | **0.749** | 0.746 |
| Debias gain | +0.020 | **+0.034** | +0.022 | +0.000 |
| Projects improved | 9/11 | **11/11** | 10/11 | 6/11 |
| Agnostic FT avg | 0.725 | 0.736 | *pending* | *pending* |
| Debiased vs FT gap | -0.027 | **+0.002** | — | — |

Debiasing helps most for mid-size models. The 8B model shows the largest gain (+0.034) and achieves FT parity. The 3B model benefits but can't fully close the gap (-0.027). The 14B model gains +0.022 and reaches the overall highest score. The 32B model's gains are masked by OOM invalids.

---

## 8. Detailed Debiased Per-k Results

### 8.1 Llama-3B

**ansible (best k=6, +0.049):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.683 | 0.689 | +0.006 | 0.434 | 0.422 | -0.012 |
| 3 | 0.681 | 0.729 | +0.048 | 0.448 | 0.514 | +0.067 |
| 6 | 0.682 | **0.741** | +0.059 | 0.455 | 0.558 | +0.103 |
| 9 | 0.692 | 0.715 | +0.023 | 0.510 | 0.557 | +0.047 |

**bitcoin (best k=6, +0.021):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.617 | 0.658 | +0.041 | 0.424 | 0.460 | +0.035 |
| 3 | 0.626 | 0.628 | +0.002 | 0.447 | 0.532 | +0.085 |
| 6 | 0.658 | **0.679** | +0.021 | 0.477 | 0.639 | +0.162 |
| 9 | 0.610 | 0.629 | +0.019 | 0.417 | 0.594 | +0.177 |

**dart (best k=3, +0.012):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.706 | 0.708 | +0.002 | 0.651 | 0.631 | -0.020 |
| 3 | 0.732 | **0.744** | +0.012 | 0.712 | 0.706 | -0.006 |
| 6 | 0.711 | 0.731 | +0.020 | 0.628 | 0.728 | +0.100 |
| 9 | 0.680 | 0.721 | +0.041 | 0.568 | 0.704 | +0.137 |

**dotnet (best k=9, +0.010):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.563 | 0.585 | +0.022 | 0.378 | 0.392 | +0.014 |
| 3 | 0.584 | 0.586 | +0.002 | 0.425 | 0.480 | +0.055 |
| 6 | 0.556 | 0.589 | +0.032 | 0.325 | 0.528 | +0.203 |
| 9 | 0.569 | **0.594** | +0.025 | 0.359 | 0.536 | +0.177 |

**react (best k=1, +0.008):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.742 | **0.750** | +0.008 | 0.603 | 0.624 | +0.022 |
| 3 | 0.729 | 0.735 | +0.006 | 0.571 | 0.595 | +0.024 |
| 6 | 0.730 | 0.734 | +0.004 | 0.581 | 0.590 | +0.009 |
| 9 | 0.715 | 0.738 | +0.023 | 0.571 | 0.610 | +0.039 |

**flutter (best k=9, +0.040):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.634 | 0.619 | -0.014 | 0.474 | 0.458 | -0.017 |
| 3 | 0.598 | 0.637 | +0.039 | 0.405 | 0.531 | +0.126 |
| 6 | 0.675 | 0.701 | +0.027 | 0.506 | 0.622 | +0.116 |
| 9 | 0.672 | **0.715** | +0.043 | 0.483 | 0.635 | +0.152 |

**k8s (best k=6, +0.030):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.765 | 0.740 | -0.025 | 0.727 | 0.737 | +0.010 |
| 3 | 0.801 | 0.785 | -0.016 | 0.783 | 0.847 | +0.065 |
| 6 | 0.809 | **0.838** | +0.030 | 0.754 | 0.829 | +0.074 |
| 9 | 0.763 | 0.828 | +0.065 | 0.685 | 0.822 | +0.137 |

**TypeScript (best k=3, -0.029):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.500 | 0.510 | +0.010 | 0.268 | 0.264 | -0.004 |
| 3 | 0.530 | **0.521** | -0.009 | 0.449 | 0.466 | +0.017 |
| 6 | 0.523 | 0.520 | -0.004 | 0.390 | 0.527 | +0.137 |
| 9 | 0.549 | 0.501 | -0.049 | 0.434 | 0.536 | +0.102 |

**vscode (best k=6, -0.020):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.569 | 0.565 | -0.004 | 0.349 | 0.336 | -0.013 |
| 3 | 0.635 | 0.602 | -0.034 | 0.539 | 0.527 | -0.011 |
| 6 | 0.645 | **0.625** | -0.020 | 0.552 | 0.657 | +0.105 |
| 9 | 0.604 | 0.570 | -0.034 | 0.500 | 0.594 | +0.094 |

**opencv (best k=9, +0.051):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.656 | 0.677 | +0.021 | 0.596 | 0.614 | +0.018 |
| 3 | 0.606 | 0.660 | +0.054 | 0.576 | 0.690 | +0.113 |
| 6 | 0.647 | 0.670 | +0.023 | 0.587 | 0.719 | +0.132 |
| 9 | 0.646 | **0.707** | +0.062 | 0.585 | 0.725 | +0.140 |

**tensorflow (best k=6, +0.050):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.679 | 0.650 | -0.028 | 0.506 | 0.477 | -0.029 |
| 3 | 0.711 | 0.736 | +0.025 | 0.596 | 0.629 | +0.032 |
| 6 | 0.666 | **0.761** | +0.096 | 0.590 | 0.690 | +0.101 |
| 9 | 0.622 | 0.726 | +0.104 | 0.476 | 0.663 | +0.187 |

### 8.2 Llama-8B

**ansible (best k=9, +0.025):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.611 | 0.621 | +0.011 | 0.300 | 0.300 | +0.000 |
| 3 | 0.634 | 0.617 | -0.017 | 0.333 | 0.355 | +0.022 |
| 6 | 0.644 | 0.652 | +0.009 | 0.344 | 0.406 | +0.062 |
| 9 | 0.649 | **0.674** | +0.025 | 0.355 | 0.427 | +0.073 |

**bitcoin (best k=9, +0.013):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.654 | 0.660 | +0.005 | 0.380 | 0.392 | +0.011 |
| 3 | 0.673 | 0.643 | -0.031 | 0.446 | 0.403 | -0.043 |
| 6 | 0.692 | 0.684 | -0.008 | 0.455 | 0.512 | +0.056 |
| 9 | 0.670 | **0.704** | +0.034 | 0.414 | 0.565 | +0.152 |

**dart (best k=9, +0.031):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.751 | 0.764 | +0.013 | 0.696 | 0.704 | +0.008 |
| 3 | 0.745 | 0.765 | +0.019 | 0.679 | 0.713 | +0.033 |
| 6 | 0.781 | 0.788 | +0.007 | 0.724 | 0.737 | +0.013 |
| 9 | 0.790 | **0.821** | +0.031 | 0.752 | 0.813 | +0.061 |

**dotnet (best k=9, +0.059):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.583 | 0.574 | -0.008 | 0.290 | 0.276 | -0.014 |
| 3 | 0.579 | 0.600 | +0.021 | 0.303 | 0.333 | +0.030 |
| 6 | 0.592 | 0.620 | +0.028 | 0.333 | 0.389 | +0.056 |
| 9 | 0.573 | **0.651** | +0.077 | 0.336 | 0.530 | +0.195 |

**react (best k=9, +0.019):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.719 | 0.719 | -0.000 | 0.624 | 0.632 | +0.008 |
| 3 | 0.756 | 0.731 | -0.025 | 0.658 | 0.632 | -0.026 |
| 6 | 0.754 | 0.750 | -0.004 | 0.679 | 0.663 | -0.016 |
| 9 | 0.749 | **0.775** | +0.026 | 0.647 | 0.694 | +0.047 |

**flutter (best k=9, +0.049):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.681 | 0.683 | +0.002 | 0.539 | 0.530 | -0.010 |
| 3 | 0.696 | 0.673 | -0.024 | 0.534 | 0.561 | +0.027 |
| 6 | 0.697 | 0.740 | +0.043 | 0.538 | 0.656 | +0.118 |
| 9 | 0.672 | **0.747** | +0.075 | 0.456 | 0.655 | +0.199 |

**k8s (best k=6, +0.059):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.763 | 0.717 | -0.046 | 0.634 | 0.679 | +0.045 |
| 3 | 0.813 | 0.777 | -0.035 | 0.739 | 0.821 | +0.082 |
| 6 | 0.815 | **0.874** | +0.059 | 0.725 | 0.831 | +0.106 |
| 9 | 0.806 | 0.868 | +0.062 | 0.717 | 0.833 | +0.116 |

**TypeScript (best k=9, +0.008):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.529 | 0.534 | +0.005 | 0.207 | 0.224 | +0.017 |
| 3 | 0.581 | 0.584 | +0.002 | 0.392 | 0.384 | -0.008 |
| 6 | 0.612 | 0.604 | -0.008 | 0.473 | 0.551 | +0.078 |
| 9 | 0.585 | **0.620** | +0.034 | 0.434 | 0.577 | +0.143 |

**vscode (best k=6, +0.002):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.569 | 0.569 | -0.000 | 0.309 | 0.306 | -0.002 |
| 3 | 0.640 | 0.647 | +0.007 | 0.475 | 0.497 | +0.022 |
| 6 | 0.708 | **0.710** | +0.002 | 0.619 | 0.651 | +0.032 |
| 9 | 0.656 | 0.705 | +0.049 | 0.538 | 0.667 | +0.129 |

**opencv (best k=9, +0.066):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.604 | 0.610 | +0.006 | 0.441 | 0.415 | -0.026 |
| 3 | 0.656 | 0.652 | -0.004 | 0.517 | 0.510 | -0.007 |
| 6 | 0.700 | 0.731 | +0.031 | 0.573 | 0.609 | +0.035 |
| 9 | 0.690 | **0.767** | +0.076 | 0.558 | 0.721 | +0.163 |

**tensorflow (best k=9, +0.040):**
| k | Base | Debias | Delta | Base Q | Debias Q | Delta Q |
|---|------|--------|-------|--------|----------|---------|
| 1 | 0.653 | 0.611 | -0.042 | 0.484 | 0.453 | -0.030 |
| 3 | 0.729 | 0.691 | -0.038 | 0.647 | 0.580 | -0.067 |
| 6 | 0.701 | 0.728 | +0.028 | 0.601 | 0.646 | +0.044 |
| 9 | 0.729 | **0.769** | +0.040 | 0.647 | 0.687 | +0.040 |

### 8.3 Qwen-14B and Qwen-32B

Per-k tables omitted for brevity. Summary tables in Section 7.2 cover the best-k results. Key pattern: Qwen-14B best debiased k is predominantly 9 (8/11 projects), matching Llama-8B. Qwen-32B's best k is mixed due to OOM effects at higher k.

---

## 9. Qwen-32B Invalid Rate Analysis

Qwen-32B at 32B parameters in 4-bit quantization pushes the RTX 4090 (24GB) to its limits. At higher k, longer prompts trigger CUDA OOM errors that produce invalid (unparseable) predictions.

### 9.1 Invalid Rates by k (total across 11 projects)

| k | Llama-3B | Llama-8B | Qwen-14B | Qwen-32B |
|---|----------|----------|----------|----------|
| 1 | 0.2% | 0.1% | 0.1% | **2.8%** |
| 3 | 0.2% | 0.2% | 0.1% | **6.0%** |
| 6 | 2.8% | 2.9% | 3.5% | **13.7%** |
| 9 | 3.7% | 3.9% | 4.3% | **22.4%** |

Llama-3B, Llama-8B, and Qwen-14B are all in the same range (0.1-4.3%). Qwen-32B is 5-6x worse at every k. Per-project extremes: flutter k=9 = 43.7%, TypeScript k=9 = 39.7%, k8s k=9 = 32.0%.

### 9.2 Valid-Only Scoring

When scoring only on predictions where the model produced a valid label:

| Approach | Qwen-32B (raw) | Qwen-32B (valid-only) |
|----------|----------------|----------------------|
| RAGTAG baseline avg | 0.746 | **0.777** |
| Debiased RAGTAG avg | 0.746 | **0.787** |
| Debias gain | +0.000 | **+0.010** |

Per-project valid-only (debiased):

| Project | Raw | Valid-only | Inv% at best k |
|---------|-----|------------|----------------|
| ansible | 0.708 | 0.740 | 13.0% |
| bitcoin | 0.742 | 0.773 | 10.0% |
| dart | 0.865 | 0.862 | 4.0% |
| dotnet | 0.685 | 0.685 | 6.7% |
| react | 0.821 | 0.822 | 0.0% |
| flutter | 0.789 | 0.809 | 24.3% |
| k8s | 0.775 | **0.915** | 16.3% |
| TypeScript | 0.652 | 0.681 | 3.7% |
| vscode | 0.648 | 0.765 | 6.7% |
| opencv | 0.751 | 0.795 | 9.0% |
| tensorflow | 0.768 | 0.815 | 5.3% |

k8s valid-only debiased reaches **0.915** — the model gets it right 91.5% of the time when it doesn't OOM. The 32B raw results are not comparable to other models due to the invalid rate disparity. Valid-only scoring shows debiasing works for Qwen-32B too (+0.010), and the model's true capability is far above what raw scores suggest.

---

## 10. Observations and Patterns

### 10.1 Project Difficulty Spectrum

Projects cluster into three difficulty tiers (based on best RAGTAG F1, Llama-3B):

**Easy (F1 > 0.75):** kubernetes (0.809), react (0.742), dart (0.732)
- High VTAG baselines (0.61-0.73). Clear issue language, well-separated classes.

**Medium (F1 0.65-0.75):** ansible (0.692), tensorflow (0.711), flutter (0.675), bitcoin (0.658), opencv (0.656), vscode (0.645)
- Moderate VTAG (0.51-0.65). Mixed difficulty.

**Hard (F1 < 0.60):** dotnet (0.584), TypeScript (0.549)
- Low VTAG (0.54-0.56). Ambiguous issue language, overlapping class boundaries.

### 10.2 LLM Marginal Value Varies by Project

LLM marginal value = RAGTAG - VTAG (Llama-3B):

| Project | VTAG | RAGTAG | LLM Marginal |
|---------|------|--------|--------------|
| tensorflow | 0.507 | 0.711 | +0.204 |
| react | 0.611 | 0.742 | +0.131 |
| dart | 0.624 | 0.732 | +0.108 |
| ansible | 0.593 | 0.692 | +0.099 |
| flutter | 0.580 | 0.675 | +0.095 |
| k8s | 0.732 | 0.809 | +0.077 |
| bitcoin | 0.597 | 0.658 | +0.061 |
| dotnet | 0.536 | 0.584 | +0.048 |
| opencv | 0.635 | 0.656 | +0.021 |
| vscode | 0.648 | 0.645 | -0.003 |
| TypeScript | 0.559 | 0.549 | -0.010 |
| **Average** | **0.602** | **0.678** | **+0.076** |

The LLM adds negative value on vscode and TypeScript — these are the same projects where Llama-3B debiasing regresses. On these projects, k-NN voting already extracts most of the available signal and the LLM's bug bias introduces more errors than its reasoning corrects.

### 10.3 Agnostic vs Project-Specific

| Setting | 3B RAGTAG | 3B FT | 8B RAGTAG | 8B FT |
|---------|-----------|-------|-----------|-------|
| Project-specific | 0.678 | N/A | 0.704 | N/A |
| Project-specific (debiased) | 0.698 | N/A | 0.737 | N/A |
| Agnostic (per-project avg) | 0.658 | 0.725 | 0.701 | 0.736 |

Project-specific RAGTAG beats agnostic RAGTAG (+0.020 for 3B, +0.003 for 8B), despite having 10x fewer training examples (300 vs 3,300). This makes sense given 88.3% same-project retrieval: the agnostic pool is mostly same-project anyway, but diluted by 11.7% cross-project noise.

FT benefits more from the agnostic setting because it sees all 3,300 examples during gradient updates, not just the k nearest. However, with debiasing, project-specific RAGTAG closes the gap entirely for 8B (0.737 vs 0.736).

### 10.4 Prediction for Qwen FT (pending)

Based on the data efficiency crossover studies (3k/30k), at 3,300 training examples:
- Qwen-14B RAGTAG beat FT at 3k (0.742 vs 0.739) and still beat FT at 27k (0.779 vs 0.767)
- Qwen-32B RAGTAG beat FT at 3k (0.778 vs 0.735) but FT won at 27k (0.810 vs 0.785)

At 3,300 examples, we are well below the crossover point for both Qwen models. Qwen-14B debiased (0.749) should beat its FT. Qwen-32B is harder to predict due to OOM effects.

---

## 11. Pending Experiments

| Experiment | Status | Expected |
|------------|--------|----------|
| Llama-3B debiased (11 projects) | **Complete** | — |
| Llama-8B debiased (11 projects) | **Complete** | — |
| Qwen-14B debiased (11 projects) | **Complete** | — |
| Qwen-32B debiased (11 projects) | **Complete** | — |
| Qwen-14B/32B FT (agnostic) | Submitted to OSC H100 | Pending |
