# Validation Protocol Notes

Final teslimde kullanılan sonuçlar Kaggle full-run V2 artifactlerine dayanır. Eski deney metrikleri superseded kabul edilmiştir.

## Protokol

- Split mantığı zaman temelli validation yaklaşımıdır.
- Classification hattında müşteri-ürün çiftleri pozitif/negatif örneklerle değerlendirilir.
- Ranking hattında aynı aday havuzu üzerinde `tabular_only`, `image_history` ve `late_fusion` skorları karşılaştırılır.
- Full 5-fold CV proposal’da hedeflenmişti; final teslimde runtime/GPU kotası nedeniyle tek final full-run validation protokolü kullanılmıştır.

## Final Classification Metrikleri

| Model | Final AUC | Best AUC | Accuracy |
|---|---:|---:|---:|
| `tabular_only` | 0.7841 | 0.7985 | 0.7150 |
| `image_only_effnet_cnn` | 0.7513 | 0.7602 | 0.6864 |
| `image_history` | 0.7480 | 0.7480 | 0.6593 |
| `late_fusion` | 0.8063 | 0.8121 | 0.7207 |

## Final Ranking Yorumu

Ana ranking lideri `late_fusion` modelidir: MAP@12 0.001262. `late_fusion_hybrid_*` varyantları ana model değil, post-ranking/reranking deneyidir.

## Rapor İçin Kullanım

Rapor ve sunumda eski metriklere dönülmemelidir. Ana hikaye şu şekilde kalmalıdır: görsel sinyal tek başına tabular baseline’ı geçmez, fakat tabular sinyalle late fusion içinde birleştiğinde en iyi sonuç elde edilir.
