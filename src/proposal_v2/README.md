# src/proposal_v2

Proposal-aligned v2 deney pipeline'i.

Calisan moduller:

- `splits.py`: customer-level 5-fold split uretir.
- `train.py`: tabular-only, image-history ve multimodal late-fusion MLP modellerini egitir.
- `cnn_baseline.py`: EfficientNet-B0 image-only CNN baseline egitir.
- `ranking.py`: MAP@12 / Precision@10 / Recall@10 ranking evaluation calistirir.
- `explainability.py`: SHAP/permutation summary ve Grad-CAM ornekleri uretir.
- `aggregate_results.py`: fold sonuclarini rapor tablosuna toplar.

Kaggle akisi icin `docs/KAGGLE_V2_RUNBOOK.md` dosyasina bak.
