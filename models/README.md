# models

Bu klasör eğitilmiş model checkpointlerini içerir.

## Alt Klasörler

- `final_kaggle/`: Final raporda kullanılacak Kaggle full-training model checkpointleri.
- `local_experiments/`: Local geliştirme ve sandbox aşamasında üretilmiş eski model checkpointleri.

## Projedeki Rolü

Model dosyaları `.pt` formatındadır ve PyTorch checkpointleri olarak kaydedilmiştir. Final değerlendirme için `final_kaggle/` altındaki checkpointler esas alınmalıdır.

## Arayüz İçin Önemi

İleride fake alışveriş sitesi veya demo arayüzü yapılırsa, model seçimi burada duran final checkpointler üzerinden yapılabilir:

- Tabular-only öneri
- Image-history öneri
- Late-fusion multimodal öneri

Canlı inference yapılmayacaksa, bu checkpointler yerine önceden üretilmiş öneri CSV'leri de kullanılabilir. Ancak model karşılaştırmalı bir demo hedeflenirse bu klasör kritik hale gelir.

