# H&M Moda Urunleri Icin Proposal-Aligned Multimodal Oneri Sistemi

## Abstract

Bu rapor v2 deney sonuclari geldikten sonra MAP@12 merkezli olarak
doldurulacak. Ana hipotez, EfficientNet-B0 gorsel temsilleri ve MLP tabular
metadata branch'inin late fusion ile birlestirilmesinin tek modaliteli
baseline'lara gore daha iyi top-12 oneriler uretmesidir.

## 1. Introduction

H&M Personalized Fashion Recommendations problemi, musteri gecmisi, urun
metadata'si ve urun fotograflarini birlikte kullanan multimodal bir onerme
problemidir.

## 2. Related Work

Bu bolum proposal literaturu ile hizalanacak: multimodal recommender systems,
EfficientNet/CNN tabanli gorsel temsil, late fusion ve explainability.

## 3. Methods

V2 modelleri:

- Tabular-only MLP
- Image-only EfficientNet-B0 CNN baseline
- Image-history MLP
- Multimodal late fusion

## 4. Experimental Design

Ana validation duzeni:

- Customer-level 5-fold split
- MAP@12 ana metrik
- Precision@10, Recall@10, HitRate@12, CandidateRecall destekleyici metrik
- AUC-ROC ve accuracy yardimci classification metrikleri

## 5. Experimental Results

V2 Kaggle ciktilari geldikten sonra su tablolar doldurulacak:

- 5-fold MAP@12 mean/std ana tablo
- Fold bazli model karsilastirmasi
- Hybrid reranking destekleyici tablo
- CNN baseline ve classification metrikleri

## 6. Explainability

- SHAP: tabular branch / tabular-only feature etkileri; SHAP kutuphanesi
  calismazsa ayni tabloda permutation fallback raporlanir
- Grad-CAM: image-only EfficientNet-B0 CNN baseline
- Visual similarity: image-history ve late-fusion visual branch yorumu

## 7. Conclusions

Sonuc, v2 MAP@12 tablosuna gore yazilacak.
