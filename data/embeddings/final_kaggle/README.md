# data/embeddings/final_kaggle

Bu klasör final Kaggle eğitimlerinde kullanılan gerçek full embedding cacheini içerir.

## Dosyalar

- `article_image_embeddings_popular.npy`
  - Shape: `(105100, 1280)`
  - EfficientNet-B0 ile çıkarılmış ürün görsel embeddingleri.
- `article_image_embedding_ids_popular.csv`
  - Satır sayısı: `105100`
  - `.npy` dosyasındaki her satırın hangi `article_id` değerine karşılık geldiğini gösterir.

## Kaggle Notebooklarla İlişkisi

Final Kaggle notebookları bu dosyaların Kaggle Dataset olarak yüklenmiş halini kullanmıştır:

```text
/kaggle/input/datasets/smoke78/article-image-embeddings-popular-npy/article_image_embeddings_popular.npy
/kaggle/input/datasets/smoke78/article-image-embedding-ids-popular-csv/article_image_embedding_ids_popular.csv
```

Yereldeki bu klasör, Kaggle'da kullanılan embedding setinin proje klasöründeki karşılığıdır.

## Projedeki Rolü

Image-history ve multimodal late-fusion modellerinde görsel modalitenin temel girdisi bu embeddinglerdir. Tabular-only model doğrudan embedding kullanmaz; fakat bazı tabular full-training senaryolarında aynı ürün evrenini korumak için embedding id listesi referans alınmıştır.

