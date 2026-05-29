# src/local_phases

Bu klasör local geliştirme fazlarında kullanılan Python scriptlerini içerir.

## Faz Mantığı

Proje önce küçük local/sandbox veri üzerinde geliştirilmiş, sonra Kaggle full-training ortamına taşınmıştır. Bu klasördeki scriptler o local geliştirme akışını temsil eder.

## Önemli Scriptler

- `phase1_eda.py`
  - Ham H&M verisini okur, aktif müşteri örneklemesi yapar ve sandbox veri setlerini üretir.
- `phase2_tabular_mlp.py`
  - Tabular-only MLP baseline modelini local sandbox veri üzerinde eğitir.
- `phase2_image_baseline.py`
  - Erken görsel baseline denemesi.
- `phase2_prepare_image_index.py`
  - Yerel görsel dosyalarını `article_id` ile eşleyen indeks üretir.
- `phase3_extract_image_embeddings.py`
  - Yerel görsellerden EfficientNet-B0 embeddingleri çıkarır.
- `phase3_build_visual_features.py`
  - Müşteri görsel profili ve görsel benzerlik özellikleri üretir.
- `phase3_image_history_mlp.py`
  - Image-history MLP local baseline.
- `phase3_visual_tabular_mlp.py`
  - Görsel similarity özelliklerini tabular modele ekleyen ara deneme.
- `phase3_multimodal_fusion.py`
  - Local multimodal late-fusion denemesi.
- `phase3_controlled_ablation.py`
  - Aynı veri bölümü üzerinde tabular-only, image-history ve late-fusion kontrollü karşılaştırması.

## Final Sonuçlarla İlişkisi

Bu scriptler final Kaggle full-training sonucunun kendisi değildir. Ancak model ailesinin nasıl seçildiğini, hangi ara denemelerin yapıldığını ve ablation fikrinin nasıl kurulduğunu gösterir.

## Path Notu

Scriptler yeni klasör yapısına göre güncellenmiştir. Eski `sandbox_data/`, `embeddings/` ve root `images/` yolları yerine artık `data/...` altındaki klasörler kullanılır.

