# START HERE - Perseptron

Bu branch final kod reposu olarak temizlenmistir.

Once sunlari oku:

1. `README.md`
2. `notebooks/kaggle/README.md`
3. `docs/KAGGLE_V2_RUNBOOK.md`
4. `artifacts/final/README.md`
5. `demo/README.md`

## Repo Mantigi

- Kod, demo ve Kaggle notebooklari GitHub'da kalir.
- Final rapor, sunum, proposal ve hoca teslim gorselleri GitHub kod reposunda
  tutulmaz.
- Bu teslim dosyalarinin yerel kopyasi:
  `C:\Users\Yusuf\Desktop\perseptron_submission_package`
- Final Kaggle checkpoint ve metrikleri `artifacts/final/` altinda Git LFS ile
  takip edilir.

## Ana Model Ayrimi

- Classification: `tabular_only`, `image_only_effnet_cnn`, `late_fusion`
- Ranking/demo: `tabular_only`, `image_history`, `late_fusion`
- `late_fusion_hybrid_*` ana model degil, reranking deneyidir.
