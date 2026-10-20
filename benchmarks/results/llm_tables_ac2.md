## Accuracy, per-state prompt  (intent / exact / exact after grounding filter)
| model | dev (31) | heldout (40) | fresh (29) | all 100 exact+grounded | turn p50 ms | turn p95 ms | prefill p50 ms |
|---|---|---|---|---|---|---|---|
| qwen3:0.6b | 0.58 / 0.10 / 0.32 | 0.45 / 0.12 / 0.30 | 0.41 / 0.17 / 0.21 | 0.28 | 537 | 881 | 50 |
| qwen3:1.7b | 0.87 / 0.45 / 0.81 | 0.72 / 0.35 / 0.60 | 0.76 / 0.41 / 0.59 | 0.66 | 1077 | 2826 | 227 |
| qwen3:4b-instruct-2507-q4_K_M | 0.97 / 0.55 / 0.81 | 0.93 / 0.57 / 0.82 | 0.93 / 0.52 / 0.83 | 0.82 | 1833 | 4391 | 307 |
| ministral-3:3b | 0.94 / 0.45 / 0.71 | 0.93 / 0.53 / 0.78 | 0.86 / 0.45 / 0.62 | 0.71 | 1765 | 3921 | 274 |

## Accuracy, static prompt + grammar-only state ('cached')  (intent / exact / exact after grounding filter)
| model | dev (31) | heldout (40) | fresh (29) | all 100 exact+grounded | turn p50 ms | turn p95 ms | prefill p50 ms |
|---|---|---|---|---|---|---|---|
| qwen3:0.6b | 0.45 / 0.10 / 0.35 | 0.40 / 0.10 / 0.25 | 0.38 / 0.10 / 0.24 | 0.28 | 724 | 1218 | 87 |
| qwen3:1.7b | 0.87 / 0.29 / 0.74 | 0.80 / 0.38 / 0.75 | 0.72 / 0.21 / 0.59 | 0.70 | 1270 | 2402 | 217 |
| qwen3:4b-instruct-2507-q4_K_M | 0.97 / 0.65 / 0.87 | 0.90 / 0.65 / 0.88 | 0.86 / 0.59 / 0.79 | 0.85 | 1929 | 3793 | 414 |
| ministral-3:3b | 0.94 / 0.61 / 0.81 | 0.78 / 0.60 / 0.75 | 0.86 / 0.52 / 0.69 | 0.75 | 1742 | 3449 | 298 |
