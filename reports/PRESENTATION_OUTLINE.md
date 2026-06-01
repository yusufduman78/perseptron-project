# Perseptron Final Sunum Outline

Final sunum 10-12 slaytlık profesyonel akışla ilerler. Ana metrikler V2 Kaggle full-run sonuçlarıdır.

## Slide 1 - Title

**Perseptron: H&M Multimodal Fashion Recommendation**
Alt mesaj: Metadata + product imagery + customer history.

## Slide 2 - Problem & Motivation

- Moda önerisi sadece kategori veya popülerlik problemi değildir.
- Renk, siluet ve müşterinin geçmiş görsel stili satın alma kararında etkilidir.
- Soru: Ürün görselleri tabular metadata’ya ek katkı sağlar mı?

## Slide 3 - Proposal Hypothesis

- Hipotez: Görsel temsil vektörleri tabular modelle birleştiğinde satın alma tahmini ve ranking kalitesi artar.
- Ana classification karşılaştırması: `tabular_only` vs `image_only_effnet_cnn` vs `late_fusion`.
- Ranking/demo karşılaştırması: `tabular_only` vs `image_history` vs `late_fusion`.

## Slide 4 - Data & Pipeline

- H&M customers, articles, transactions, product images.
- Time-based validation.
- EfficientNet-B0 image features.
- Customer visual-history profile.
- FastAPI + React demo.

## Slide 5 - Model Separation

- `tabular_only`: customer + article metadata MLP.
- `image_only_effnet_cnn`: product image CNN baseline and Grad-CAM source.
- `image_history`: customer visual-history ranking baseline; CNN değildir.
- `late_fusion`: tabular branch + visual branch + fusion head.

## Slide 6 - Classification Results

| Model | Final AUC | Best AUC | Accuracy |
|---|---:|---:|---:|
| `tabular_only` | 0.7841 | 0.7985 | 0.7150 |
| `image_only_effnet_cnn` | 0.7513 | 0.7602 | 0.6864 |
| `late_fusion` | 0.8063 | 0.8121 | 0.7207 |

Konuşma notu: Görsel tek başına en iyi değil; late fusion içinde anlamlı katkı veriyor.

## Slide 7 - Ranking Results

| Model | MAP@12 | Precision@10 | Recall@10 |
|---|---:|---:|---:|
| `late_fusion` | 0.001262 | 0.000889 | 0.003256 |
| `tabular_only` | 0.000713 | 0.000858 | 0.002702 |
| `image_history` | 0.000653 | 0.000764 | 0.002404 |

Konuşma notu: MAP@12 düşük ama aynı aday havuzunda late fusion lider.

## Slide 8 - Hybrid Reranking

- `late_fusion_hybrid_*` ana model değildir.
- Co-purchase ve heuristic sinyallerle post-ranking denemesidir.
- En iyi hybrid MAP@12 0.000961 ile ana late fusion sonucunun altında kalmıştır.

## Slide 9 - Explainability

- SHAP summary: `age`, `section_no`, `product_group_name`, `colour_group_code`, `FN` öne çıkıyor.
- Grad-CAM: image-only EfficientNet CNN görsellerinden üretilir.
- Demo chipleri: renk uyumu, kategori uyumu, görsel geçmiş, en yakın geçmiş ürün.

## Slide 10 - Demo Flow

- Hazır senaryo seç: Black Basics veya Denim Casual.
- Style bag geçmiş ürünleri gösterir.
- Üç demo modeli öneri üretir: `tabular_only`, `image_history`, `late_fusion`.
- Model insights: ortak ve unique öneriler.
- Final evidence: metrics + explainability.

## Slide 11 - Limitations

- Full 5-fold CV teslim süresi/GPU kotası nedeniyle final full-run validation’a indirildi.
- CNN demo recommender değil; image-only baseline ve Grad-CAM kaynağı.
- Ranking metrikleri aday havuzu recall ve kısa validation penceresinden etkileniyor.

## Slide 12 - Conclusion

- Tabular metadata güçlü baseline.
- Image-only tek başına sınırlı.
- Late fusion final AUC 0.8063, best AUC 0.8121 ve MAP@12 0.001262 ile final lider.
- Sonuç: Multimodal sinyal, tabular bağlamla birlikte kullanıldığında değer katıyor.
