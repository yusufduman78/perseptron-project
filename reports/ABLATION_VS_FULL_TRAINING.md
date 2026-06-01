# Ablation ve Full-run Sonuçlarının Final Yorumu

Bu dosya eski deney notlarının final V2 sonuçlarına göre temizlenmiş özetidir. Önceki sürümlerde raporlanan yüksek AUC değerleri artık final sonuç kabul edilmez. Final teslimde referans alınacak metrikler `reports/proposal_v2/` altındaki Kaggle full-run artifactlerinden gelir.

## Final Ana Karşılaştırma

| Model | Rol | Final AUC | Best AUC | Accuracy |
|---|---|---:|---:|---:|
| `tabular_only` | Metadata baseline | 0.7841 | 0.7985 | 0.7150 |
| `image_only_effnet_cnn` | Image-only baseline | 0.7513 | 0.7602 | 0.6864 |
| `late_fusion` | Multimodal main model | 0.8063 | 0.8121 | 0.7207 |

`image_history` ayrıca görsel geçmiş baseline’ı olarak takip edilir: AUC 0.7480, accuracy 0.6593. Bu model CNN değildir; müşteri geçmişi embedding profilini kullanır.

## Sonuç

Final yorum nettir: görsel bilgi tek başına tabular baseline’ı geçmemiştir, ancak tabular sinyalle late fusion içinde birleştiğinde en iyi sonucu vermiştir. Bu yüzden final raporun ana iddiası “görüntü tek başına en iyidir” değil, “görsel temsil tabular bağlamla birleştiğinde satın alma tahminini iyileştirir” şeklindedir.

Hybrid reranking deneyleri ana model değildir. Ranking sonuçlarında `late_fusion` MAP@12 0.001262 ile ana modeller arasında liderdir; `late_fusion_hybrid_*` varyantları sadece post-ranking deneyi olarak raporlanmalıdır.
