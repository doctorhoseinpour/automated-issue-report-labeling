# RQ2.9 — Bootstrap CIs + McNemar on headline pairs

| Pair | n | F1 A | F1 B | ΔF1 (A−B) | 95% CI | McNemar p | Sig |
|---|---:|---:|---:|---:|---|---:|:---:|
| H1: Qwen-3B zero-shot (ag) vs VTAG k=9 (ag) | 3300 | 0.6132 | 0.5984 | +0.0147 | [-0.0073, +0.0366] | 0.0251 | * |
| H2[Qwen-3B]: RAGTAG ag k3 vs VTAG ag k=9 | 3300 | 0.6944 | 0.5984 | +0.0960 | [+0.0778, +0.1151] | 7.4e-25 | *** |
| H2[Qwen-7B]: RAGTAG ag k6 vs VTAG ag k=9 | 3300 | 0.7122 | 0.5984 | +0.1138 | [+0.0942, +0.1350] | 6.86e-25 | *** |
| H2[Qwen-14B]: RAGTAG ag k9 vs VTAG ag k=9 | 3300 | 0.7170 | 0.5984 | +0.1186 | [+0.0982, +0.1394] | 2.02e-25 | *** |
| H2[Qwen-32B]: RAGTAG ag k9 vs VTAG ag k=9 | 3300 | 0.7594 | 0.5984 | +0.1609 | [+0.1415, +0.1806] | 2.45e-44 | *** |
| H3[Qwen-3B]: RAGTAG ag k3 vs FT ag | 3300 | 0.6944 | 0.6520 | +0.0424 | [+0.0251, +0.0602] | 1.3e-05 | *** |
| H3[Qwen-7B]: RAGTAG ag k6 vs FT ag | 3300 | 0.7122 | 0.7411 | -0.0289 | [-0.0436, -0.0139] | 4.45e-05 | *** |
| H3[Qwen-14B]: RAGTAG ag k9 vs FT ag | 3300 | 0.7170 | 0.7154 | +0.0016 | [-0.0161, +0.0191] | 0.922 | ns |
| H3[Qwen-32B]: RAGTAG ag k9 vs FT ag | 3300 | 0.7594 | 0.7462 | +0.0131 | [-0.0024, +0.0270] | 0.237 | ns |
| H4[Qwen-3B]: Debias ps k6 vs RAGTAG ps k3 | 3300 | 0.7137 | 0.6970 | +0.0167 | [+0.0021, +0.0312] | 0.539 | ns |
| H4[Qwen-7B]: Debias ps k6 vs RAGTAG ps k6 | 3300 | 0.7327 | 0.7175 | +0.0151 | [+0.0046, +0.0248] | 9.1e-05 | *** |
| H4[Qwen-14B]: Debias ps k9 vs RAGTAG ps k9 | 3300 | 0.7453 | 0.7208 | +0.0245 | [+0.0157, +0.0330] | 2.24e-09 | *** |
| H4[Qwen-32B]: Debias ps k9 vs RAGTAG ps k9 | 3300 | 0.7770 | 0.7613 | +0.0156 | [+0.0084, +0.0227] | 6.34e-07 | *** |
| H5[Qwen-3B]: Debias ps k6 vs FT ag | 3300 | 0.7137 | 0.6520 | +0.0616 | [+0.0442, +0.0804] | 4.35e-06 | *** |
| H5[Qwen-7B]: Debias ps k6 vs FT ag | 3300 | 0.7327 | 0.7411 | -0.0084 | [-0.0233, +0.0071] | 0.359 | ns |
| H5[Qwen-14B]: Debias ps k9 vs FT ag | 3300 | 0.7453 | 0.7154 | +0.0299 | [+0.0125, +0.0463] | 0.00343 | ** |
| H5[Qwen-32B]: Debias ps k9 vs FT ag | 3300 | 0.7770 | 0.7462 | +0.0308 | [+0.0141, +0.0453] | 0.139 | ns |
| H6: Qwen-7B FT ag vs Qwen-7B Debias ps k6 (the 7B anomaly) | 3300 | 0.7411 | 0.7327 | +0.0084 | [-0.0071, +0.0233] | 0.359 | ns |

Significance markers: `***` p<0.001, `**` p<0.01, `*` p<0.05, `ns` not significant.
