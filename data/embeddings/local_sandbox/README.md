# data/embeddings/local_sandbox

Bu klasör eski local faz deneylerinde kullanılan embedding cacheini içerir.

## Dosyalar

- `article_image_embeddings.npy`
  - Shape: `(52833, 1280)`
  - Local/sandbox ürün evreni için çıkarılmış EfficientNet-B0 embeddingleri.
- `article_image_embedding_ids.csv`
  - Satır sayısı: `52833`
  - Embedding satırlarını `article_id` değerleriyle eşler.

## Neden Ayrı Tutuldu?

Final Kaggle deneylerinde 105,100 ürünlük daha geniş embedding seti kullanılmıştır. Bu klasördeki 52,833 ürünlük set ise local faz scriptlerinin tekrar çalıştırılabilmesi için saklanır.

## Kullanım Notu

`src/local_phases/` altındaki eski local scriptler bu klasörü kullanacak şekilde güncellenmiştir. Final rapor metrikleri için `data/embeddings/final_kaggle/` esas alınmalıdır.

