# Final Deney Karşılaştırması

Bu özet, final rapor ve sunumda kullanılacak deney ayrımını sabitler.

## Classification Karşılaştırması

| Model | Açıklama | Final AUC | Best AUC | Accuracy |
|---|---|---:|---:|---:|
| `tabular_only` | Müşteri ve ürün metadata baseline | 0.7841 | 0.7985 | 0.7150 |
| `image_only_effnet_cnn` | EfficientNet-B0 image-only baseline | 0.7513 | 0.7602 | 0.6864 |
| `late_fusion` | Metadata + görsel geçmiş multimodal model | 0.8063 | 0.8121 | 0.7207 |

Bu tablo proposal ile en doğrudan uyumlu ana model karşılaştırmasıdır. CNN görsel baseline’dır; demo öneri modeli değildir.

## Ranking Karşılaştırması

| Model | MAP@12 | Precision@10 | Recall@10 |
|---|---:|---:|---:|
| `late_fusion` | 0.001262 | 0.000889 | 0.003256 |
| `tabular_only` | 0.000713 | 0.000858 | 0.002702 |
| `image_history` | 0.000653 | 0.000764 | 0.002404 |

Ranking tarafında `image_history`, CNN yerine değil, müşterinin görsel geçmiş profilini kullanan pratik öneri baseline’ı olarak yer alır.

## Hybrid Deneyleri

| Model | MAP@12 | Not |
|---|---:|---|
| `late_fusion_hybrid_w025` | 0.000961 | Post-ranking/reranking |
| `late_fusion_hybrid_w045` | 0.000910 | Post-ranking/reranking |
| `late_fusion_hybrid_w065` | 0.000714 | Post-ranking/reranking |

Hybrid varyantlar ana model karşılaştırmasına yazılmamalıdır. Rapor ve sunumda yalnızca “heuristic reranking denendi, fakat ana late fusion modelini geçmedi” şeklinde destekleyici deney olarak kullanılmalıdır.

## Final Yorum

Tabular metadata güçlüdür, görüntü tek başına sınırlıdır, fakat geç müşteri görsel profiliyle birleştirilen late fusion model en iyi sonucu verir. Bu sonuç, projenin multimodal hipotezini kontrollü ve savunulabilir biçimde destekler.
