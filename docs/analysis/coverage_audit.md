# Coverage audit

| approach | setting | k | expected | actual |
|---|---|---|---:|---:|
| ragtag | agnostic | zero_shot | 4 | 4 |
| ragtag | agnostic | k1 | 4 | 4 |
| ragtag | agnostic | k3 | 4 | 4 |
| ragtag | agnostic | k6 | 4 | 4 |
| ragtag | agnostic | k9 | 4 | 4 |
| ft | agnostic | finetune_fixed | 4 | 4 |
| ragtag | project_specific | zero_shot | 44 | 11 ⚠ |
| ragtag | project_specific | k1 | 44 | 44 |
| ragtag | project_specific | k3 | 44 | 44 |
| ragtag | project_specific | k6 | 44 | 44 |
| ragtag | project_specific | k9 | 44 | 44 |
| ragtag_debias | project_specific | k1 | 44 | 44 |
| ragtag_debias | project_specific | k3 | 44 | 44 |
| ragtag_debias | project_specific | k6 | 44 | 44 |
| ragtag_debias | project_specific | k9 | 44 | 44 |
| ft | project_specific | finetune_fixed | 44 | 44 |
| vtag | agnostic | * | 22 | 22 |
| vtag | project_specific | * | 242 | 242 |
| vtag_debias | agnostic | * | 4 | 4 |
| vtag_debias | project_specific | * | 44 | 44 |

## Cells with invalid_rate > 0.10

| model | setting | project | approach | k | invalid_rate |
|---|---|---|---|---|---:|
| Qwen-14B | project_specific | flutter_flutter | ragtag | k9 | 0.1167 |
| Qwen-32B | project_specific | flutter_flutter | ragtag | k9 | 0.1167 |
| Qwen-3B | project_specific | flutter_flutter | ragtag | k9 | 0.1167 |
| Qwen-7B | project_specific | flutter_flutter | ragtag | k9 | 0.1167 |
