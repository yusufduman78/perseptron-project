# Perseptron Projesine Başlangıç

Bu dosya, projeyi ilk kez açan bir insanın veya yeni bir Codex hesabının 5-10 dakika içinde doğru bağlama oturması için hazırlanmıştır. Ayrıntılı teknik hafıza için ana kaynak `PROJECT_HANDOFF.md` dosyasıdır.

## Yeni Codex İçin Başlangıç Promptu

Yeni hesapta veya yeni oturumda şunu yaz:

```text
C:\Users\Yusuf\Desktop\perseptron_project\START_HERE.md dosyasını oku. Sonra PROJECT_HANDOFF.md, docs/DOC_MAP.md ve docs/NEXT_STEPS.md dosyalarını okuyup projenin mevcut durumunu, ana sonuçlarını ve sıradaki uygulanabilir işleri özetle.
```

## Proje Nedir?

Perseptron, H&M Personalized Fashion Recommendation veri seti üzerinde geliştirilmiş bir multimodal moda öneri sistemi projesidir.

Ana hipotez:

> Ürün fotoğraflarından EfficientNet-B0 ile çıkarılan görsel embeddingler, müşterinin geçmiş satın alımlarından oluşturulan görsel tercih profili ve tabular müşteri/ürün metaverisi late fusion mimarisinde birleştirildiğinde, tek başına tabular veya tek başına görsel geçmiş kullanan modellere göre daha güçlü satın alma skoru üretir.

Ana bilimsel karşılaştırma üç model üzerinden yapılır:

| Model | Rol | Final AUC-ROC | Final Accuracy |
|---|---|---:|---:|
| Tabular-only MLP | Müşteri ve ürün metaverisi baseline | 0.8550 | 0.7636 |
| Image-history MLP | EfficientNet-B0 embedding + müşteri görsel geçmişi baseline | 0.9237 | 0.8297 |
| Multimodal late fusion | Tabular branch + visual branch ana model | 0.9354 | 0.8537 |

Ana sonuç: Late fusion en yüksek full-training performansını verdi; bu da multimodal hipotezi destekliyor.

## En Önemli Dosyalar

| Dosya | Ne için okunur? |
|---|---|
| `PROJECT_HANDOFF.md` | Projenin kapsamlı çalışma hafızası |
| `docs/DOC_MAP.md` | Hangi dosya/klasör ne işe yarıyor haritası |
| `docs/NEXT_STEPS.md` | Sıradaki yapılacak işler ve başarı kriterleri |
| `docs/DECISION_LOG.md` | Alınan kritik teknik kararların nedenleri |
| `reports/FINAL_REPORT.md` | Teslime yakın ana rapor taslağı |
| `reports/kaggle_full_training_results.csv` | Üç ana modelin final full-training sonuçları |
| `reports/ranking_metrics_validation_summary.md` | MAP@12 / Precision@10 / Recall@10 değerlendirmesi |
| `demo/README.md` | Demo backend/frontend çalıştırma bilgisi |

## Önerilen Okuma Sırası

Yeni bir hesap veya projeyi ilk kez gören bir kişi için:

1. `START_HERE.md`
2. `PROJECT_HANDOFF.md`
3. `docs/DOC_MAP.md`
4. `docs/NEXT_STEPS.md`
5. `docs/DECISION_LOG.md`
6. `reports/FINAL_REPORT.md`
7. `demo/README.md`

Sadece rapor/sunum odaklı hızlı okuma:

1. `README.md`
2. `reports/FINAL_REPORT.md`
3. `reports/EXPERIMENT_COMPARISON.md`
4. `reports/ABLATION_VS_FULL_TRAINING.md`
5. `reports/PRESENTATION_OUTLINE.md`

Sadece teknik devam için hızlı okuma:

1. `PROJECT_HANDOFF.md`
2. `docs/NEXT_STEPS.md`
3. `demo/backend/recommender.py`
4. `src/evaluation/ranking_metrics_validation.py`
5. `reports/kaggle_full_training_results.csv`

## Mevcut Durum

Tamamlanan ana işler:

- Ham H&M verisi ve görseller proje yapısına yerleştirildi.
- EfficientNet-B0 ile 105,100 ürün için 1280 boyutlu görsel embedding çıkarıldı.
- Tabular-only, image-history ve multimodal late fusion modelleri Kaggle üzerinde full-training düzeninde eğitildi.
- Final checkpointler `models/final_kaggle/` altına alındı.
- Final sonuç tablosu `reports/kaggle_full_training_results.csv` içine yazıldı.
- Controlled ablation ve erken faz deneyleri raporlandı.
- MAP@12 / Precision@10 / Recall@10 ranking evaluation eklendi.
- Görsel benzerlik açıklamaları ve tabular permutation importance üretildi.
- FastAPI backend ve React/Vite frontend demo hazırlandı.
- Final rapor ve sunum taslakları oluşturuldu.

## Son Teknik Nokta

Son konuşulan teknik konu `late_fusion_hybrid` idi.

Önemli ayrım:

- `late_fusion`: Eğitilmiş ana multimodal modeldir.
- `late_fusion_hybrid`: Eğitilmiş yeni bir model değildir; late fusion skorunu son dönem popülerliği, co-purchase, metadata benzerliği ve visual similarity sinyalleriyle yeniden sıralayan post-ranking/reranking deneyidir.

Bu yüzden raporda `late_fusion_hybrid` ana üç model karşılaştırmasına dördüncü model olarak sokulmamalı; pratik recommender sistemi iyileştirmesi olarak ayrı anlatılmalıdır.

## Sıradaki En Mantıklı İş

Öncelik sırası:

1. `reports/FINAL_REPORT.md` dosyasına ranking metrics ve `late_fusion_hybrid` açıklamasını kontrollü şekilde eklemek.
2. `reports/perseptron_multimodal_recommendation_presentation.pptx` sunumunu son metriklerle güncellemek.
3. Demo backend/frontend'i çalıştırıp smoke-test yapmak.
4. Rapor ve README dosyalarında Türkçe karakter/görünüm kontrolü yapmak.
5. Teslimden sonra istenirse hybrid sinyalleri eğitim feature'ı olarak yeni deney planına dönüştürmek.

Detaylı görev planı: `docs/NEXT_STEPS.md`

## Büyük Dosya Uyarısı

Projede büyük dosyalar var:

- `data/raw/transactions_train.csv`: yaklaşık 3.49 GB
- `data/images/hm_images/`: 105,100 görsel
- `data/embeddings/final_kaggle/article_image_embeddings_popular.npy`: yaklaşık 538 MB
- `demo/frontend/node_modules/`: taşınması gerekmeyen bağımlılık klasörü
- `reports/presentation_build/node_modules/`: taşınması gerekmeyen sunum build bağımlılık klasörü

Proje başka hesaba/bilgisayara taşınacaksa önce `PROJECT_HANDOFF.md`, `reports/`, `models/final_kaggle/`, `notebooks/kaggle/` ve gerekli veri/embedding dosyalarının taşındığından emin olunmalıdır.
