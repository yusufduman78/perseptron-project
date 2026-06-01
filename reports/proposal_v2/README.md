# Proposal V2 Final Artifacts

Bu klasör final Kaggle V2 koşularından seçilen teslim artifactlerini içerir. Büyük fold split dosyaları ve model checkpointleri Git’e eklenmez.

## Dosyalar

- `proposal_v2_classification_metrics.csv`: `tabular_only`, `image_history`, `late_fusion` classification sonuçları.
- `proposal_v2_cnn_metrics.csv`: `image_only_effnet_cnn` baseline sonucu.
- `proposal_v2_ranking_metrics.csv`: müşteri bazlı MAP@12/Precision@10/Recall@10 ranking çıktıları.
- `proposal_v2_cv_summary.md`: ranking summary.
- `proposal_v2_shap_summary.csv`: tabular explainability için SHAP feature summary.
- `gradcam_examples/*.png`: EfficientNet CNN Grad-CAM örnekleri.

## Final Metrik Özeti

- `tabular_only`: final AUC 0.7841, best AUC 0.7985.
- `image_only_effnet_cnn`: final AUC 0.7513, best AUC 0.7602.
- `image_history`: final AUC 0.7480.
- `late_fusion`: final AUC 0.8063, best AUC 0.8121.
- Ranking lideri: `late_fusion`, MAP@12 0.001262.
