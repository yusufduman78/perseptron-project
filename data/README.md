# data

Bu klasör proje boyunca kullanılan veri varlıklarını içerir. Ham Kaggle CSV dosyaları, ürün görselleri, EfficientNet-B0 ile çıkarılmış embedding cacheleri ve local deneylerde üretilmiş ara CSV dosyaları burada tutulur.

## Alt Klasörler

- `raw/`: Kaggle H&M veri setinden gelen ham CSV dosyaları.
- `images/`: H&M ürün fotoğrafları.
- `embeddings/`: Ürün görsellerinden çıkarılmış embedding dosyaları.
- `processed/`: Local sandbox deneyleri sırasında üretilmiş ara veri setleri.

## Projedeki Rolü

Bu klasör doğrudan model eğitiminin veri temelidir. Local faz scriptleri genellikle `data/processed/sandbox/` ve `data/embeddings/local_sandbox/` dosyalarını kullanır. Kaggle final deneyleri ise Kaggle ortamında aynı veri setini ve `data/embeddings/final_kaggle/` ile eşdeğer olan yüklenmiş embedding datasetlerini kullanmıştır.

## Dikkat Edilmesi Gerekenler

Bu klasör projenin en büyük kısmıdır. Özellikle `data/images/hm_images/` yaklaşık 29 GB yer kaplar. Eğer proje başka bir bilgisayara taşınacaksa önce model ve rapor dosyalarının mı, yoksa tüm ham verinin mi taşınacağına karar verilmelidir.

