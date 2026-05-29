# Kaggle Late Fusion Error Fix

## What failed

The late-fusion notebook completed epoch 1 training, then failed during validation with:

```text
NameError: name 'accuracy_score' is not defined
```

The cause was a missing import in `notebooks/kaggle/perseptron_late_fusion.ipynb`.

## Fixed locally

The notebook now imports both metrics before any long training starts:

```python
from sklearn.metrics import accuracy_score, roc_auc_score
```

A preflight check was also added after startup:

```python
missing_symbols = [name for name in ("accuracy_score", "roc_auc_score") if name not in globals()]
if missing_symbols:
    raise RuntimeError(f"Preflight failed. Missing imports before long training: {missing_symbols}")
if accuracy_score([0, 1], [0, 1]) != 1.0:
    raise RuntimeError("Preflight failed. accuracy_score smoke test returned an unexpected value.")
log("Preflight passed: sklearn metrics are available before long training.")
```

This makes the notebook fail in the first few seconds if the metric imports are broken, instead of failing after a long epoch.

## What to do on Kaggle

Use the updated local notebook:

```text
notebooks/kaggle/perseptron_late_fusion.ipynb
```

Upload/replace it on Kaggle and run again with the same datasets and accelerator.

In the early logs, confirm this appears before training:

```text
Preflight passed: sklearn metrics are available before long training.
```

## If the failed Kaggle kernel is still alive

If Kaggle still has the failed interactive session open with variables in memory, it may be possible to avoid rerunning epoch 1 by manually running:

```python
from sklearn.metrics import accuracy_score, roc_auc_score

metrics = evaluate_stream_model(
    model,
    validation_data,
    metadata,
    numeric_mean,
    numeric_std,
)
print(metrics)
```

Then continue carefully from the next epoch. If the Kaggle run was a detached Save Version / papermill run, assume the state is gone and rerun the fixed notebook from the start.
