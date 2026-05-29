# H&M Moda Ürünleri İçin Multimodal Öneri Modeli

Bu proje, H&M Personalized Fashion Recommendation veri seti üzerinde müşteri-ürün önerisi üretmek için görsel ürün temsilleri ile tabular müşteri/ürün metaverisini birlikte kullanan bir multimodal öneri sistemi geliştirmeyi amaçlar.

Projenin ana hipotezi şudur: Ürün fotoğraflarından EfficientNet-B0 ile çıkarılan görsel embeddingler, müşterinin geçmiş alışverişlerinden oluşturulan görsel tercih profili ve tabular müşteri/ürün özellikleri late fusion mimarisinde birleştirildiğinde, sadece görsel veya sadece tabular modellere göre daha güçlü bir satın alma olasılığı skoru üretir.

## Yeni Başlayanlar İçin

Projeyi yeni bir Codex hesabı, yeni bir geliştirici veya projeyi ilk kez inceleyen biri devralacaksa önce `START_HERE.md` okunmalıdır. Bu dosya 5-10 dakikalık hızlı bağlam, önerilen okuma sırası, yeni Codex başlangıç promptu ve en yakın yapılacak işleri içerir.

Detaylı proje hafızası için canonical kaynak `PROJECT_HANDOFF.md` dosyasıdır. Dosya/klasör haritası için `docs/DOC_MAP.md`, sıradaki görevler için `docs/NEXT_STEPS.md`, kritik kararların gerekçeleri için `docs/DECISION_LOG.md` kullanılmalıdır.

## Kısa Özet

Projede üç ana model karşılaştırılmıştır:

| Model | Kullanılan veri | Kaggle full-training AUC | Accuracy |
|---|---|---:|---:|
| Tabular-only MLP | Müşteri ve ürün metaverisi | 0.8550 | 0.7636 |
| Image-history MLP | EfficientNet-B0 görsel embeddingleri ve müşteri görsel geçmişi | 0.9237 | 0.8297 |
| Multimodal late fusion | Tabular branch + image-history branch | 0.9354 | 0.8537 |

Sonuçlar hipotezi destekler: multimodal late fusion modeli, tabular-only ve image-history baseline modellerinden daha yüksek doğrulama performansı vermiştir.

## Önerilen İnceleme Sırası

Projeyi ilk kez inceleyen biri için en temiz okuma sırası:

1. `START_HERE.md`
2. `PROJECT_HANDOFF.md`
3. `docs/DOC_MAP.md`
4. `docs/NEXT_STEPS.md`
5. `docs/DECISION_LOG.md`
6. `reports/FINAL_REPORT.md`
7. `reports/kaggle_full_training_results.csv`
8. `notebooks/kaggle/README.md`
9. `models/final_kaggle/README.md`
10. `demo/README.md`

## Final Rapor

Teslim için ana rapor taslağı:

- `reports/FINAL_REPORT.md`

Bu dosya metodoloji, deney kurulumu, full-training sonuçları, controlled ablation yorumu, açıklanabilirlik analizi, sınırlılıklar ve sonuç bölümlerini tek yerde toplar.

Destekleyici rapor dosyaları:

- `reports/EXPERIMENT_COMPARISON.md`
- `reports/ABLATION_VS_FULL_TRAINING.md`
- `reports/VISUAL_EXPLAINABILITY.md`
- `reports/TABULAR_EXPLAINABILITY.md`
- `reports/PIPELINE_DIAGRAMS.md`
- `reports/PRESENTATION_OUTLINE.md`

## Final Deneyler

Final full-training deneyleri Kaggle üzerinde yapılmıştır. Bunun nedeni H&M veri setinin büyük olması, görsel embeddinglerin yüksek boyutlu olması ve full müşteri-ürün pozitif çiftlerinin local ortamda bellek açısından daha zor yönetilmesidir.

Kaggle notebookları:

- `notebooks/kaggle/tabular_only_fulltraining.ipynb`
- `notebooks/kaggle/image_history_fulltraining_perseptron.ipynb`
- `notebooks/kaggle/perseptron_late_fusion.ipynb`

Final model checkpointleri:

- `models/final_kaggle/tabular_only_streaming_full.pt`
- `models/final_kaggle/image_history_streaming_full.pt`
- `models/final_kaggle/multimodal_fusion_streaming_full.pt`

Toplu sonuç tablosu:

- `reports/kaggle_full_training_results.csv`

## Proje Mantığı

Veri seti dört ana bilgi türü içerir:

- `transactions_train.csv`: müşterilerin geçmiş satın alma hareketleri.
- `customers.csv`: müşteri metaverisi.
- `articles.csv`: ürün metaverisi.
- `images/`: ürün fotoğrafları.

Görsel tarafta ürün fotoğrafları doğrudan eğitim sırasında tekrar tekrar CNN'e sokulmamıştır. Bunun yerine EfficientNet-B0 pretrained modelinden 1280 boyutlu embeddingler çıkarılmıştır. Image-history model bu embeddinglerden müşteri görsel profili üretir. Late fusion model ise tabular branch ile image-history branch çıktısını birleştirir.

## Açıklanabilirlik

Projede iki açıklanabilirlik çıktısı hazırlanmıştır:

- Görsel benzerlik analizi:
  - `src/explainability/visual_similarity_explanations.py`
  - `reports/visual_similarity_examples.csv`
  - `reports/visual_similarity_examples.html`
- Tabular permutation importance:
  - `src/explainability/tabular_permutation_importance.py`
  - `reports/tabular_permutation_importance.csv`
  - `reports/tabular_feature_group_importance.csv`

Tabular importance analizi local controlled tabular model üzerinde yapılmıştır; Kaggle full-training modelinin birebir açıklaması değildir. SHAP, mevcut PyTorch embedding/preprocessing yapısı nedeniyle bu aşamada uygulanmamış ve ileriki çalışma olarak bırakılmıştır.

## Demo

Model checkpointlerini canlı kullanmak için FastAPI backend ve React/Vite frontend hazırlanmıştır:

- `demo/backend/recommender.py`
- `demo/backend/app.py`
- `demo/backend/test_recommend.py`
- `demo/frontend/`

Backend:

```powershell
python demo\backend\app.py
```

Frontend:

```powershell
cd demo\frontend
npm install
npm run dev -- --host 127.0.0.1
```

Arayüz `http://127.0.0.1:5173` adresinde açılır. Kullanıcı katalogdan geçmiş satın alma ürünleri seçebilir, basit müşteri metadata alanlarını değiştirebilir ve tabular-only, image-history, late-fusion modellerinin önerilerini yan yana görebilir.

Terminal testi:

```powershell
python demo\backend\test_recommend.py
```

## Klasör Yapısı

- `data/`: Ham veri, görseller, embedding cacheleri ve ara işlenmiş veriler.
- `models/`: Final Kaggle modelleri ve eski local deneme modelleri.
- `notebooks/`: Kaggle üzerinde çalıştırılan deney notebookları.
- `reports/`: Deney sonuç tabloları, final rapor ve açıklanabilirlik çıktıları.
- `src/`: Local geliştirme, Kaggle pipeline ve açıklanabilirlik scriptleri.
- `docs/`: Proposal ve proje dokümanları.
- `PROJECT_STRUCTURE.md`: Klasör yapısının kısa teknik özeti.

## Önemli Notlar

`data/images/hm_images/` klasörü yaklaşık 29 GB büyüklüğündedir ve 105 binden fazla ürün görseli içerir. Proje taşınırken veya paylaşılırken bu klasörün boyutu dikkate alınmalıdır.

`data/embeddings/final_kaggle/` klasörü final Kaggle eğitimlerinde kullanılan gerçek 105,100 ürünlük embedding setini içerir. `data/embeddings/local_sandbox/` ise daha önce local fazlarda kullanılan 52,833 ürünlük eski embedding cacheidir.

Bu proje bir Kaggle leaderboard çözümünden çok, ders projesi bağlamında multimodal recommendation hipotezini test eden deneysel bir sistemdir. Kaggle submission üretimi destekleyici bir dış test çıktısıdır; ana bilimsel karşılaştırma tabular-only, image-history ve late-fusion modellerinin aynı full-training düzeninde karşılaştırılmasıdır.
