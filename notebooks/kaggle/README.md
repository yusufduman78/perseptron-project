# notebooks/kaggle

Bu klasör final Kaggle deney notebooklarını içerir.

## Notebooklar

- `tabular_only_fulltraining.ipynb`
  - Sadece tabular müşteri ve ürün metaverisiyle MLP eğitir.
  - Görsel embedding kullanmaz; bazı senaryolarda aynı ürün evrenini korumak için embedding id listesi referans alınabilir.
- `image_history_fulltraining_perseptron.ipynb`
  - EfficientNet-B0 ile çıkarılmış ürün embeddingleri ve müşteri görsel geçmiş profili üzerinden image-history baseline modelini eğitir.
  - Tabular metaveri kullanmaz.
- `perseptron_late_fusion.ipynb`
  - Tabular branch ve image-history branch'i late fusion ile birleştirir.
  - Projenin ana multimodal modelidir.

## Deney Düzeni

Üç notebook da full-training mantığına göre hazırlanmıştır. Büyük veri belleğe tek seferde alınmak yerine streaming/chunk yapısıyla eğitilir.

Ortak prensip:

1. Full pozitif müşteri-ürün çiftleri hazırlanır.
2. Negatif örnekler üretilir.
3. Model chunk chunk eğitilir.
4. Validation AUC ve accuracy takip edilir.
5. En iyi checkpoint `/kaggle/working` altında kaydedilir.

## Kullanılan Embedding Dataseti

Image-history ve late-fusion notebookları Kaggle üzerinde şu embedding datasetlerini kullanmıştır:

```text
/kaggle/input/datasets/smoke78/article-image-embeddings-popular-npy/article_image_embeddings_popular.npy
/kaggle/input/datasets/smoke78/article-image-embedding-ids-popular-csv/article_image_embedding_ids_popular.csv
```

Bu dosyaların yerel karşılığı:

```text
data/embeddings/final_kaggle/
```

## Final Çıktılar

Notebooklarda üretilen final checkpointlerin yerel karşılıkları:

- `models/final_kaggle/tabular_only_streaming_full.pt`
- `models/final_kaggle/image_history_streaming_full.pt`
- `models/final_kaggle/multimodal_fusion_streaming_full.pt`

Toplu sonuç tablosu:

- `reports/kaggle_full_training_results.csv`
