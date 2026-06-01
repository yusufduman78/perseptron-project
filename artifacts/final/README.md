# Final Kaggle Artifacts

Bu klasor final Kaggle full-run ciktilarinin temizlenmis ve isimlendirilmis
surumunu tutar. Dagitilabilir GitHub artifact alani burasidir; `.kaggle_outputs/`
local cache olarak kalir.

## models

- `tabular_only.pt`: `proposal_v2_01_tabular_only.ipynb`
- `image_history.pt`: `proposal_v2_02_image_history.ipynb`
- `late_fusion.pt`: `proposal_v2_03_late_fusion.ipynb`
- `image_only_effnet_cnn.pt`: `proposal_v2_04_image_only_cnn.ipynb`

## metrics

- `proposal_v2_classification_metrics.csv`: tabular/image-history/late-fusion
  classification metrics.
- `proposal_v2_cnn_metrics.csv`: image-only EfficientNet CNN metrics.
- `proposal_v2_ranking_metrics.csv`: MAP@12, Precision@10, Recall@10 and
  related ranking outputs.
- `proposal_v2_cv_summary.md`: final ranking summary used in report/savunma.

## explainability

- `proposal_v2_shap_summary.csv`: tabular explainability summary.
- `gradcam_examples/*.png`: image-only EfficientNet CNN Grad-CAM examples.

## Reproduce

Run Kaggle notebooks in this order:

1. `proposal_v2_00_folds.ipynb`
2. `proposal_v2_01_tabular_only.ipynb`
3. `proposal_v2_02_image_history.ipynb`
4. `proposal_v2_03_late_fusion.ipynb`
5. `proposal_v2_04_image_only_cnn.ipynb`
6. `proposal_v2_05_ranking_map12.ipynb`
7. `proposal_v2_06_explainability.ipynb`

Then copy selected outputs from Kaggle `models/proposal_v2/` and
`reports/proposal_v2/` into this folder with the names above.
