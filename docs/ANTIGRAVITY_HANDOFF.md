# Antigravity Handoff: Proposal v2 Full Kaggle Runs

Last updated: 2026-05-31 Europe/Istanbul

## Current Situation

Branch: `report-polish`

Goal: finish proposal-aligned full Kaggle pipeline and collect report-ready metrics/artifacts.

Kaggle username/prefix: `smoke78`

Important rule: push GPU notebooks with explicit T4 accelerator. Do not rely on default `enable_gpu=true`, because default GPU previously landed on P100 and failed with:

```text
AcceleratorError: CUDA error: no kernel image is available for execution on the device
```

Use:

```powershell
kaggle kernels push -p .kaggle_runs\full\<run-folder> --accelerator NvidiaTeslaT4
```

## Completed Runs

### 00 Folds

Ref:

```text
smoke78/perseptron-v2-00-folds-full
```

Status: `COMPLETE`

URL:

```text
https://www.kaggle.com/code/smoke78/perseptron-v2-00-folds-full
```

Outputs present:

- `proposal_v2_fold_splits.csv`
- `proposal_v2_fold_protocol_summary.json`

### 01 Tabular Only

Ref:

```text
smoke78/perseptron-v2-01-tabular-full
```

Status: `COMPLETE`

URL:

```text
https://www.kaggle.com/code/smoke78/perseptron-v2-01-tabular-full
```

Completed with explicit T4 after first default-GPU run failed.

Metrics from log:

| epoch | loss | AUC | accuracy |
| --- | ---: | ---: | ---: |
| 1 | 0.5634 | 0.7985 | 0.7305 |
| 2 | 0.5398 | 0.7938 | 0.7237 |
| 3 | 0.5315 | 0.7841 | 0.7150 |

Note: code currently saves final epoch, not best validation AUC. For reporting, record both final and best epoch. Consider adding best-checkpoint logic after the full pipeline finishes.

Outputs present:

- `tabular_only_fold0.pt`
- `fold0_tabular_only_context.json`
- `proposal_v2_classification_metrics.csv`

### 02 Image History

Ref:

```text
smoke78/perseptron-v2-02-image-history-full
```

Status: `COMPLETE`

URL:

```text
https://www.kaggle.com/code/smoke78/perseptron-v2-02-image-history-full
```

Metrics from log:

| epoch | loss | AUC | accuracy |
| --- | ---: | ---: | ---: |
| 1 | 0.6552 | 0.7219 | 0.6541 |
| 2 | 0.5949 | 0.7423 | 0.6708 |
| 3 | 0.5707 | 0.7480 | 0.6593 |

Outputs present:

- `image_history_fold0.pt`
- `fold0_image_history_context.json`
- `proposal_v2_classification_metrics.csv`

## Currently Running

### 03 Late Fusion

Ref:

```text
smoke78/perseptron-v2-03-late-fusion-full
```

Status at handoff: `RUNNING`

URL:

```text
https://www.kaggle.com/code/smoke78/perseptron-v2-03-late-fusion-full
```

Started with:

```powershell
kaggle kernels push -p .kaggle_runs\full\perseptron-v2-03-late-fusion-full --accelerator NvidiaTeslaT4
```

When the user says it finished, check:

```powershell
kaggle kernels status smoke78/perseptron-v2-03-late-fusion-full
kaggle kernels files smoke78/perseptron-v2-03-late-fusion-full
$env:PYTHONIOENCODING='utf-8'; kaggle kernels logs smoke78/perseptron-v2-03-late-fusion-full | Select-Object -Last 180
```

Expected outputs:

- `late_fusion_fold0.pt`
- `fold0_late_fusion_context.json`
- `proposal_v2_classification_metrics.csv`

Record all epoch metrics from logs.

## Next Runs

Run these only after the previous required source kernels are complete.

### 04 CNN Full

Folder:

```text
.kaggle_runs\full\perseptron-v2-04-cnn-full
```

Start:

```powershell
kaggle kernels push -p .kaggle_runs\full\perseptron-v2-04-cnn-full --accelerator NvidiaTeslaT4
```

Settings:

- `FAST_RUN=False`
- `MAX_TRAIN_POSITIVES=20_000`
- `MAX_VAL_POSITIVES=5_000`
- `EPOCHS=3`
- `TRAIN_BACKBONE=True`

Expected outputs:

- `image_only_effnet_cnn_fold0.pt`
- `proposal_v2_cnn_metrics.csv`

### 05 Ranking Full

Folder:

```text
.kaggle_runs\full\perseptron-v2-05-ranking-full
```

Start after 01, 02, 03 are complete:

```powershell
kaggle kernels push -p .kaggle_runs\full\perseptron-v2-05-ranking-full --accelerator NvidiaTeslaT4
```

Settings:

- `FAST_RUN=False`
- `CANDIDATE_LIMIT=5000`
- `VISUAL_NEIGHBORS=3000`
- `CO_PURCHASE_PER_ITEM=300`
- hybrid weights: `0.25`, `0.45`, `0.65`

Expected outputs:

- `proposal_v2_ranking_metrics.csv`
- `proposal_v2_cv_summary.md`
- merged `proposal_v2_classification_metrics.csv`

Important: ranking restore logic was patched to merge CSV outputs instead of overwriting them.

### 06 Explainability Full

Folder:

```text
.kaggle_runs\full\perseptron-v2-06-explainability-full
```

Start after 01 and 04 are complete:

```powershell
kaggle kernels push -p .kaggle_runs\full\perseptron-v2-06-explainability-full --accelerator NvidiaTeslaT4
```

Expected outputs:

- `proposal_v2_shap_summary.csv`
- `gradcam_examples/*.png`

## Download Outputs

Use local ignored folder:

```powershell
New-Item -ItemType Directory -Force -Path .kaggle_outputs\full\03-late-fusion | Out-Null
kaggle kernels output smoke78/perseptron-v2-03-late-fusion-full -p .kaggle_outputs\full\03-late-fusion
```

Repeat for later runs:

```powershell
kaggle kernels output smoke78/perseptron-v2-04-cnn-full -p .kaggle_outputs\full\04-cnn
kaggle kernels output smoke78/perseptron-v2-05-ranking-full -p .kaggle_outputs\full\05-ranking
kaggle kernels output smoke78/perseptron-v2-06-explainability-full -p .kaggle_outputs\full\06-explainability
```

If Windows encoding errors appear during log download but output files are listed as downloaded, inspect files manually; this happened before and the actual artifacts were fine.

## Interpretation Notes

- Smoke metrics are not report metrics.
- Full metrics should distinguish final epoch vs best validation AUC.
- `tabular_only` is currently strong; this is not necessarily a bug.
- `image_history` full improved over smoke and reached AUC `0.7480`.
- `late_fusion` must be judged after 03 full finishes.
- CNN is a separate image-only baseline, not part of `proposal_v2_classification_metrics.csv`; it writes `proposal_v2_cnn_metrics.csv`.
- `late_fusion_hybrid_w025/w045/w065` are ranking/reranking experiments, not main models.

## Useful URLs

- 00: https://www.kaggle.com/code/smoke78/perseptron-v2-00-folds-full
- 01: https://www.kaggle.com/code/smoke78/perseptron-v2-01-tabular-full
- 02: https://www.kaggle.com/code/smoke78/perseptron-v2-02-image-history-full
- 03: https://www.kaggle.com/code/smoke78/perseptron-v2-03-late-fusion-full
- 04: https://www.kaggle.com/code/smoke78/perseptron-v2-04-cnn-full
- 05: https://www.kaggle.com/code/smoke78/perseptron-v2-05-ranking-full
- 06: https://www.kaggle.com/code/smoke78/perseptron-v2-06-explainability-full
