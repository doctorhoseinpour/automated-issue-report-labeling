# RQ2.1 — Master leaderboard

Best macro F1 per (model, approach, setting). For project-specific
settings, F1 is the per-project mean across 11 projects (macro-macro).
Agnostic F1 is computed on the pooled 3,300-issue test set directly.


### Qwen-3B

| Model | Approach | Setting | k | F1 macro | Acc | F1 bug | F1 feat | F1 q | Inv |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen-3B | zero_shot | agnostic | zero_shot | 0.6132 | 0.6258 | 0.6816 | 0.7227 | 0.4352 | 0.0024 |
| Qwen-3B | ragtag | agnostic | k3 | 0.6944 | 0.7006 | 0.7094 | 0.8018 | 0.5720 | 0.0024 |
| Qwen-3B | ragtag | project_specific | k3 | 0.6939 | 0.7030 | 0.7088 | 0.8021 | 0.5708 | 0.0021 |
| Qwen-3B | ragtag_debias | project_specific | k6 | 0.7089 | 0.7079 | 0.6815 | 0.8022 | 0.6431 | 0.0179 |
| Qwen-3B | ft | agnostic | finetune_fixed | 0.6520 | 0.6630 | 0.6989 | 0.7475 | 0.5098 | 0.0055 |
| Qwen-3B | ft | project_specific | finetune_fixed | 0.5754 | 0.5851 | 0.6479 | 0.6558 | 0.4225 | 0.0191 |

### Qwen-7B

| Model | Approach | Setting | k | F1 macro | Acc | F1 bug | F1 feat | F1 q | Inv |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen-7B | zero_shot | agnostic | zero_shot | 0.6621 | 0.6791 | 0.6959 | 0.7905 | 0.5000 | 0.0009 |
| Qwen-7B | ragtag | agnostic | k6 | 0.7122 | 0.7103 | 0.7287 | 0.8082 | 0.5996 | 0.0348 |
| Qwen-7B | ragtag | project_specific | k6 | 0.7142 | 0.7152 | 0.7323 | 0.8166 | 0.5938 | 0.0346 |
| Qwen-7B | ragtag_debias | project_specific | k6 | 0.7299 | 0.7342 | 0.7483 | 0.8138 | 0.6277 | 0.0161 |
| Qwen-7B | ft | agnostic | finetune_fixed | 0.7411 | 0.7412 | 0.7329 | 0.8263 | 0.6641 | 0.0012 |
| Qwen-7B | ft | project_specific | finetune_fixed | 0.5113 | 0.5267 | 0.5041 | 0.6299 | 0.4001 | 0.0106 |

### Qwen-14B

| Model | Approach | Setting | k | F1 macro | Acc | F1 bug | F1 feat | F1 q | Inv |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen-14B | zero_shot | agnostic | zero_shot | 0.6452 | 0.6752 | 0.6905 | 0.8057 | 0.4395 | 0.0006 |
| Qwen-14B | ragtag | agnostic | k9 | 0.7170 | 0.7109 | 0.7268 | 0.8272 | 0.5971 | 0.0427 |
| Qwen-14B | ragtag | project_specific | k9 | 0.7165 | 0.7142 | 0.7283 | 0.8330 | 0.5883 | 0.0430 |
| Qwen-14B | ragtag_debias | project_specific | k9 | 0.7418 | 0.7388 | 0.7477 | 0.8337 | 0.6441 | 0.0321 |
| Qwen-14B | ft | agnostic | finetune_fixed | 0.7154 | 0.7121 | 0.7027 | 0.7675 | 0.6759 | 0.0009 |
| Qwen-14B | ft | project_specific | finetune_fixed | 0.6366 | 0.6385 | 0.6922 | 0.6516 | 0.5660 | 0.0012 |

### Qwen-32B

| Model | Approach | Setting | k | F1 macro | Acc | F1 bug | F1 feat | F1 q | Inv |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| Qwen-32B | zero_shot | agnostic | zero_shot | 0.6876 | 0.7058 | 0.7150 | 0.8143 | 0.5335 | 0.0018 |
| Qwen-32B | ragtag | agnostic | k9 | 0.7594 | 0.7464 | 0.7577 | 0.8379 | 0.6825 | 0.0442 |
| Qwen-32B | ragtag | project_specific | k9 | 0.7579 | 0.7476 | 0.7545 | 0.8410 | 0.6782 | 0.0449 |
| Qwen-32B | ragtag_debias | project_specific | k9 | 0.7745 | 0.7664 | 0.7754 | 0.8405 | 0.7077 | 0.0330 |
| Qwen-32B | ft | agnostic | finetune_fixed | 0.7462 | 0.7552 | 0.7781 | 0.8138 | 0.6466 | 0.0000 |
| Qwen-32B | ft | project_specific | finetune_fixed | 0.7130 | 0.7200 | 0.6794 | 0.7972 | 0.6625 | 0.0006 |

### (no LLM)

| Model | Approach | Setting | k | F1 macro | Acc | F1 bug | F1 feat | F1 q | Inv |
|---|---|---|---|---:|---:|---:|---:|---:|---:|
| (no LLM) | vtag | agnostic | k16 | 0.6039 | 0.6061 | 0.6318 | 0.6294 | 0.5505 | 0.0000 |
| (no LLM) | vtag | project_specific | k7 | 0.5837 | 0.5961 | 0.6194 | 0.5981 | 0.5336 | 0.0000 |
| (no LLM) | vtag_debias | agnostic | k9 | 0.6048 | 0.6048 | 0.5892 | 0.6287 | 0.5966 | 0.0000 |
| (no LLM) | vtag_debias | project_specific | k9 | 0.5852 | 0.5967 | 0.5628 | 0.6075 | 0.5853 | 0.0000 |
