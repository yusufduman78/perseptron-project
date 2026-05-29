# src/explainability

Bu klasör, model çıktılarının nedenlerini yorumlamak için kullanılan açıklanabilirlik scriptlerini içerir.

## Görsel Benzerlik Açıklaması

`visual_similarity_explanations.py`, EfficientNet-B0 ile çıkarılmış ürün embeddinglerini kullanarak bir aday ürünün müşterinin geçmişindeki hangi ürünlere görsel olarak benzediğini hesaplar.

Script şu çıktıları üretir:

- `reports/visual_similarity_examples.csv`
  - Her aday ürün için en benzer geçmiş ürünleri ve cosine similarity skorlarını içerir.
- `reports/visual_similarity_examples.html`
  - Aday ürün ve benzer geçmiş ürünleri görselleriyle birlikte gösteren okunabilir HTML raporu.

## Varsayılan Kullanım

```powershell
python src/explainability/visual_similarity_explanations.py
```

Varsayılan mod hızlı çalışmak için ilk 2 milyon transaction satırından örnek üretir.

## Full Transaction Geçmişiyle Çalıştırma

```powershell
python src/explainability/visual_similarity_explanations.py --full-transactions
```

Bu mod daha yavaştır; fakat final eğitimdeki veri evrenine daha yakındır.

## Submission Benzeri Dosyadan Aday Açıklama

Eğer `customer_id,prediction` kolonlarına sahip bir submission dosyası verilirse, her müşteri için ilk önerilen ürün açıklanır.

```powershell
python src/explainability/visual_similarity_explanations.py --recommendations-csv reports/submission.csv
```

## Rapor İçin Yorum

Bu analiz klasik Grad-CAM yerine embedding tabanlı açıklama üretir. Projedeki EfficientNet-B0 modeli doğrudan sınıflandırıcı olarak fine-tune edilmediği için Grad-CAM yerine görsel embedding uzayında en yakın geçmiş ürünleri göstermek bu mimariye daha uygundur.

## Tabular Feature Importance

`tabular_permutation_importance.py`, controlled tabular-only MLP checkpointini yükler ve validation örnekleri üzerinde permutation importance hesabı yapar. Her tabular özellik sırayla karıştırılır; AUC-ROC ve accuracy ne kadar düşüyorsa o özellik o kadar etkili kabul edilir.

```powershell
python src/explainability/tabular_permutation_importance.py
```

Üretilen çıktılar:

- `reports/tabular_permutation_importance.csv`
- `reports/tabular_feature_group_importance.csv`
