# src/kaggle

Bu klasör Kaggle ortamına yönelik pipeline kaynak kodunu içerir.

## Dosya

- `kaggle_multimodal_pipeline.py`
  - H&M veri seti üzerinde multimodal recommendation pipeline'ının script hali.

## Projedeki Rolü

Bu script, notebooklaştırmadan önceki Kaggle pipeline mantığını saklar. Veri yükleme, embedding cache kontrolü, müşteri görsel profili oluşturma, model eğitimi, aday ürün üretimi ve submission üretimi gibi adımları içerir.

## Final Notebooklarla İlişkisi

Final raporda kullanılacak ana deneyler `notebooks/kaggle/` altındaki düzenlenmiş notebooklardan gelir. Bu script daha çok kaynak kod referansı ve pipeline geçmişi olarak tutulur.

## Not

Kaggle pathleri Kaggle ortamına göre tasarlanmıştır. Local ortamda doğrudan çalıştırmak için H&M input klasörlerinin ve embedding dosyalarının pathleri uyarlanmalıdır.

