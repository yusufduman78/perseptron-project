# reports

Bu klasör deney sonuç tablolarını, rapor taslaklarını ve açıklanabilirlik çıktılarını içerir.

## Dosyalar

- `kaggle_full_training_results.csv`
  - Final Kaggle full-training sonuçlarının toplu tablosu.
  - Rapor için ana sonuç dosyasıdır.
- `phase2_baseline_results.csv`
  - Local Phase 2 ve erken Phase 3 deneme sonuçları.
- `phase3_controlled_ablation_results.csv`
  - Local kontrollü ablation deney sonuçları.
- `ranking_metrics_validation.csv`
  - Zamana dayalı validation ayrımı üzerinde müşteri-model bazlı MAP@12, Precision@10 ve Recall@10 detayları.
- `ranking_metrics_validation_summary.md`
  - Ranking evaluation deneyinin kısa özeti ve model karşılaştırma tablosu.
- `FINAL_REPORT_DRAFT.md`
  - Final rapora doğrudan taşınabilecek Türkçe metodoloji, deney sonucu ve yorum taslağı.
- `FINAL_REPORT.md`
  - Parça rapor dosyalarından toparlanmış nihai Türkçe rapor taslağı.
- `EXPERIMENT_COMPARISON.md`
  - Üç ana modelin full-training ve controlled ablation sonuçlarını rapor dilinde karşılaştıran özet dosya.
- `ABLATION_VS_FULL_TRAINING.md`
  - Controlled ablation ile Kaggle full-training sonuçlarının neden aynı ölçekte yorumlanmaması gerektiğini açıklar.
- `VISUAL_EXPLAINABILITY.md`
  - Görsel benzerlik tabanlı açıklanabilirlik analizinin rapor metni.
- `TABULAR_EXPLAINABILITY.md`
  - Tabular permutation importance analizinin rapor metni.
- `PIPELINE_DIAGRAMS.md`
  - Rapor veya sunumda kullanılabilecek model, pipeline ve demo inference diyagramları.
- `PRESENTATION_OUTLINE.md`
  - Ders sunumu için 12 slaytlık Türkçe akış, konuşma notları ve kısa sunum alternatifi.
- `perseptron_multimodal_recommendation_presentation.pptx`
  - Presentations eklentisiyle üretilmiş, düzenlenebilir 12 slaytlık PowerPoint sunumu.
- `presentation_previews/`
  - Sunumun PNG önizlemeleri ve genel kontrol sayfası (`contact_sheet.png`).
- `presentation_build/`
  - Sunumu tekrar üretmek için kullanılan build scripti.

## Final Sonuç Dosyası

`kaggle_full_training_results.csv` üç ana modeli karşılaştırır:

- Tabular-only MLP
- Image-history MLP
- Multimodal late fusion

Bu dosya rapordaki ana performans tablosunun kaynağı olarak kullanılabilir.

`FINAL_REPORT_DRAFT.md` bu tablodaki sonuçları proje hipotezi, model mimarileri ve deney yorumu ile birlikte rapor metnine dönüştürür.

`EXPERIMENT_COMPARISON.md`, özellikle sonuçlar bölümünü yazarken kullanılacak kısa ve net karşılaştırma metnidir.

`ABLATION_VS_FULL_TRAINING.md`, sonuç tabloları arasında oluşabilecek skor farklarını akademik dille açıklamak için kullanılabilir.

## Explainability Çıktıları

`src/explainability/visual_similarity_explanations.py` çalıştırıldığında aşağıdaki dosyalar üretilir:

- `visual_similarity_examples.csv`
  - Aday ürün ile müşteri geçmişindeki en benzer ürünleri cosine similarity skorlarıyla listeler.
- `visual_similarity_examples.html`
  - Aynı açıklamaları ürün görselleriyle birlikte okunabilir HTML raporu olarak sunar.
- `tabular_permutation_importance.csv`
  - Tabular-only model için özellik bazlı permutation importance sonucu.
- `tabular_feature_group_importance.csv`
  - Tabular özelliklerin müşteri/ürün ve sayısal/kategorik gruplarına göre özet importance sonucu.

## Yorumlama

Local phase sonuçları geliştirme sürecini ve model tasarımının neden o yöne evrildiğini göstermek için faydalıdır. Final proje iddiası için ise Kaggle full-training sonuçları daha önemlidir.
