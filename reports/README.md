# Reports Directory

Final teslim için ana dosyalar:

- `FINAL_REPORT.md`: Türkçe conference-paper formatındaki final rapor.
- `FINAL_REPORT_CONFERENCE_DRAFT.md`: aynı raporun conference draft kopyası.
- `PRESENTATION_OUTLINE.md`: 12 slaytlık final sunum akışı.
- `presentation_build/build_perseptron_presentation.mjs`: PPTX üretim scripti.
- `perseptron_multimodal_recommendation_presentation.pptx`: üretilen final sunum.
- `proposal_v2/`: final Kaggle V2 metrik ve explainability artifactleri.

## Kullanılacak Final Metrikler

| Model | Rol | Final AUC | Best AUC |
|---|---|---:|---:|
| `tabular_only` | Metadata baseline | 0.7841 | 0.7985 |
| `image_only_effnet_cnn` | Image-only baseline | 0.7513 | 0.7602 |
| `image_history` | Visual-history ranking baseline | 0.7480 | 0.7480 |
| `late_fusion` | Main multimodal model | 0.8063 | 0.8121 |

Ranking lideri ana modeller içinde `late_fusion` modelidir: MAP@12 0.001262. `late_fusion_hybrid_*` ana model değil, reranking/post-ranking deneyidir.

## Artifact Kaynakları

- Classification: `proposal_v2/proposal_v2_classification_metrics.csv`
- CNN baseline: `proposal_v2/proposal_v2_cnn_metrics.csv`
- Ranking: `proposal_v2/proposal_v2_ranking_metrics.csv`
- Ranking summary: `proposal_v2/proposal_v2_cv_summary.md`
- SHAP summary: `proposal_v2/proposal_v2_shap_summary.csv`
- Grad-CAM: `proposal_v2/gradcam_examples/`
