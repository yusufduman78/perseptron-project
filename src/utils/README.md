# src/utils

Bu klasör yardımcı scriptleri içerir.

## Dosya

- `download_images.py`
  - Local sandbox ürünleri için Kaggle'dan görsel indirme denemesinde kullanılan yardımcı script.

## Projedeki Rolü

Bu script final Kaggle full-training deneyinin parçası değildir. Local geliştirme sırasında ürün görsellerini yerel klasöre çekmek ve görsel pipeline'ı test etmek için kullanılmıştır.

## Kullanım Notu

Script yeni klasör yapısına göre güncellenmiştir:

- Sandbox article listesi: `data/processed/sandbox/sandbox_articles.csv`
- Görsel hedef klasörü: `data/images/hm_images/`

Kaggle CLI pathi kullanıcı ortamına özel olabilir. Farklı bir bilgisayarda çalıştırılacaksa `KAGGLE_EXE` değeri kontrol edilmelidir.

