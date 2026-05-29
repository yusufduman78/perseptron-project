# models/final_kaggle

Bu klasör final Kaggle full-training deneylerinden gelen model checkpointlerini ve bazı sonuç CSV dosyalarını içerir.

## Dosyalar

- `tabular_only_streaming_full.pt`
  - Tabular-only MLP model checkpointi.
  - Yalnızca müşteri ve ürün metaverisi kullanır.
- `image_history_streaming_full.pt`
  - Image-history MLP model checkpointi.
  - EfficientNet-B0 ürün embeddingleri ve müşteri görsel geçmiş profili kullanır.
- `multimodal_fusion_streaming_full.pt`
  - Multimodal late-fusion model checkpointi.
  - Tabular branch ve görsel branch çıktısını birleştirir.
- `tabular_only_full_results.csv`
  - Tabular-only full-training sonucu.
- `image_history_full_results.csv`
  - Image-history full-training sonucu.

## Final Sonuçlar

Toplu sonuçlar `reports/kaggle_full_training_results.csv` dosyasına da aktarılmıştır.

| Model | AUC | Accuracy |
|---|---:|---:|
| Tabular-only MLP | 0.8550 | 0.7636 |
| Image-history MLP | 0.9237 | 0.8297 |
| Multimodal late fusion | 0.9341 | 0.8504 |

## Yorum

Bu sonuçlar multimodal hipotezi destekler. Görsel geçmiş tek başına güçlü bir sinyal üretmiştir; ancak tabular bilgiyle late fusion yapıldığında en yüksek performans elde edilmiştir.
