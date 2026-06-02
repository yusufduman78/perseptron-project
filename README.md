# Perseptron Project

H&M Personalized Fashion Recommendations veri seti uzerinde proposal-aligned
multimodal fashion recommendation projesi.


## Project Goal

Temel hipotez:

> EfficientNet-B0 gorsel temsilleri ve tabular customer/article metadata
> sinyalleri late fusion ile birlestirildiginde, tabular-only ve image-only
> baseline'lara gore daha iyi onerme skoru uretir.

Final model ayrimi:

- Classification: `tabular_only`, `image_only_effnet_cnn`, `late_fusion`
- Ranking/demo: `tabular_only`, `image_history`, `late_fusion`
- `late_fusion_hybrid_*`: ana model degil, post-ranking/reranking deneyidir.

## Repository Layout

- `src/proposal_v2/`: reusable Python modules for data, models, training,
  ranking and explainability.
- `notebooks/kaggle/`: self-contained Kaggle notebooks, run in order.
- `demo/`: FastAPI backend and React frontend demo.
- `artifacts/final/`: selected final Kaggle checkpoints, metrics and
  explainability outputs.
- `data/README.md`: raw data policy. Raw H&M data is not committed.
- `docs/`: lightweight runbook docs only.

## Kaggle Notebook Order

1. `proposal_v2_00_folds.ipynb`
2. `proposal_v2_01_tabular_only.ipynb`
3. `proposal_v2_02_image_history.ipynb`
4. `proposal_v2_03_late_fusion.ipynb`
5. `proposal_v2_04_image_only_cnn.ipynb`
6. `proposal_v2_05_ranking_map12.ipynb`
7. `proposal_v2_06_explainability.ipynb`

Detailed smoke/full instructions are in `notebooks/kaggle/README.md` and
`docs/KAGGLE_V2_RUNBOOK.md`.

## Final Artifacts

Final Kaggle outputs are organized under `artifacts/final/`:

- `models/`: final fold-0 checkpoints used by ranking/demo/report evidence.
- `metrics/`: classification, CNN, ranking and CV summary outputs.
- `explainability/`: SHAP summary and Grad-CAM examples.

Large model/artifact files are tracked with Git LFS.

## Demo

Backend smoke:

```powershell
conda run -n perseptron python demo\backend\test_recommend.py --top-k 3 --candidate-limit 200
```

Backend:

```powershell
conda run -n perseptron python demo\backend\app.py
```

Frontend:

```powershell
cd demo\frontend
npm install
npm run dev -- --host 127.0.0.1
```

## Data Policy

The following are intentionally not committed:

- raw H&M competition data,
- full product image dataset,
- local Kaggle run caches,
- duplicated smoke outputs,
- final report/presentation deliverables.

Use `artifacts/final/` for selected final outputs that should be versioned.
