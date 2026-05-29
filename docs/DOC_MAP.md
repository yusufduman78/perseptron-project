# Perseptron Doküman ve Dosya Haritası

Bu dosya, yeni bir Codex hesabının veya projeyi ilk kez inceleyen birinin hangi dosyanın ne işe yaradığını hızlıca anlaması için hazırlanmıştır.

## İlk Okunacak Dosyalar

| Dosya | Rol |
|---|---|
| `START_HERE.md` | İlk giriş noktası; 5-10 dakikalık hızlı proje özeti |
| `PROJECT_HANDOFF.md` | En kapsamlı proje hafızası; yeni hesap için canonical bağlam |
| `docs/NEXT_STEPS.md` | Bundan sonra yapılacak işler ve öncelik sırası |
| `docs/DECISION_LOG.md` | Kritik teknik kararların gerekçeleri |
| `README.md` | İnsan okuyucu için proje ana sayfası |

## `reports/`

Raporlar, metrik tabloları, açıklanabilirlik çıktıları ve sunum burada tutulur.

| Dosya | Ne işe yarar? |
|---|---|
| `reports/FINAL_REPORT.md` | Ana final rapor taslağı |
| `reports/FINAL_REPORT_DRAFT.md` | Raporun daha taslak/çalışma metni |
| `reports/EXPERIMENT_COMPARISON.md` | Üç ana modelin sonuç karşılaştırması |
| `reports/ABLATION_VS_FULL_TRAINING.md` | Controlled ablation ile full-training farkını açıklar |
| `reports/kaggle_full_training_results.csv` | Ana final sonuç tablosu |
| `reports/phase2_baseline_results.csv` | Erken local phase sonuçları |
| `reports/phase3_controlled_ablation_results.csv` | Controlled ablation sonuçları |
| `reports/ranking_metrics_validation.csv` | Müşteri-model bazlı ranking metric detayları |
| `reports/ranking_metrics_validation_summary.md` | MAP@12, Precision@10, Recall@10 özeti |
| `reports/VISUAL_EXPLAINABILITY.md` | Görsel benzerlik açıklanabilirlik raporu |
| `reports/TABULAR_EXPLAINABILITY.md` | Tabular permutation importance raporu |
| `reports/visual_similarity_examples.csv` | Görsel açıklama örnekleri |
| `reports/visual_similarity_examples.html` | Görsellerle okunabilir açıklama raporu |
| `reports/tabular_permutation_importance.csv` | Tekil feature permutation importance |
| `reports/tabular_feature_group_importance.csv` | Feature group importance |
| `reports/PIPELINE_DIAGRAMS.md` | Mermaid pipeline/model akış diyagramları |
| `reports/PRESENTATION_OUTLINE.md` | Sunum akışı ve konuşma notları |
| `reports/perseptron_multimodal_recommendation_presentation.pptx` | PowerPoint sunumu |
| `reports/presentation_previews/` | Sunum görsel önizlemeleri |

En önemli rapor çıktısı:

```text
reports/kaggle_full_training_results.csv
```

Ana model sonuçları bu dosyadan alınmalıdır.

## `models/`

Model checkpointleri burada tutulur.

### `models/final_kaggle/`

Final ve raporda esas alınacak checkpointler:

| Dosya | Model |
|---|---|
| `tabular_only_streaming_full.pt` | Tabular-only MLP |
| `image_history_streaming_full.pt` | Image-history MLP |
| `multimodal_fusion_streaming_full.pt` | Multimodal late fusion |
| `tabular_only_full_results.csv` | Tabular-only sonuç özeti |
| `image_history_full_results.csv` | Image-history sonuç özeti |

Bu klasör demo backend tarafından da kullanılır.

### `models/local_experiments/`

Local phase ve controlled ablation modellerini içerir. Final iddia için destekleyici/arka plan niteliğindedir.

Örnekler:

- `tabular_mlp.pt`
- `image_history_mlp.pt`
- `multimodal_fusion.pt`
- `controlled_tabular_only_mlp.pt`
- `controlled_image_history_mlp.pt`
- `controlled_multimodal_late_fusion.pt`

## `notebooks/`

Kaggle ve arşiv notebookları burada tutulur.

### `notebooks/kaggle/`

Final eğitim notebookları:

| Notebook | Görev |
|---|---|
| `tabular_only_fulltraining.ipynb` | Sadece tabular model full-training |
| `image_history_fulltraining_perseptron.ipynb` | Image-history model full-training |
| `perseptron_late_fusion.ipynb` | Multimodal late fusion full-training |

Bu üç notebook ana eğitim sürecinin kaynağıdır.

### `notebooks/archive/`

Eski/generated notebooklar. Gerekmedikçe buradan devam edilmemelidir.

## `src/`

Kod kaynakları.

| Klasör | Rol |
|---|---|
| `src/local_phases/` | Erken local phase, ablation ve embedding scriptleri |
| `src/kaggle/` | Kaggle multimodal pipeline kaynak scripti |
| `src/explainability/` | Görsel açıklama ve permutation importance scriptleri |
| `src/evaluation/` | Ranking metrics validation scripti |
| `src/utils/` | Yardımcı scriptler |

Özellikle önemli dosyalar:

- `src/evaluation/ranking_metrics_validation.py`
- `src/explainability/visual_similarity_explanations.py`
- `src/explainability/tabular_permutation_importance.py`
- `src/kaggle/kaggle_multimodal_pipeline.py`

## `demo/`

Canlı model demo uygulaması.

| Klasör/Dosya | Rol |
|---|---|
| `demo/backend/app.py` | FastAPI uygulaması |
| `demo/backend/recommender.py` | Checkpointleri yükleyip üç modelden öneri üretir |
| `demo/backend/test_recommend.py` | Terminal smoke-test |
| `demo/frontend/` | React/Vite alışveriş sitesi benzeri arayüz |
| `demo/sample_recommendations.json` | Örnek çıktı |

Backend:

```powershell
python demo\backend\app.py
```

Frontend:

```powershell
cd demo\frontend
npm run dev -- --host 127.0.0.1
```

## `data/`

Ham veri, görseller, embedding cacheleri ve ara veriler.

| Klasör | Rol |
|---|---|
| `data/raw/` | H&M CSV dosyaları |
| `data/images/hm_images/` | 105,100 ürün görseli |
| `data/embeddings/final_kaggle/` | Final EfficientNet-B0 embedding cache |
| `data/embeddings/local_sandbox/` | Eski/local embedding cache |
| `data/processed/sandbox/` | Local phase ara CSV çıktıları |

Final embedding dosyaları:

```text
data/embeddings/final_kaggle/article_image_embeddings_popular.npy
data/embeddings/final_kaggle/article_image_embedding_ids_popular.csv
```

## Büyük ve Gereksiz Dolaşılmaması Gereken Klasörler

Yeni Codex veya insan okuyucu normalde bu klasörlere girmemelidir:

| Klasör | Neden dikkat? |
|---|---|
| `data/images/hm_images/` | 105,100 görsel; çok büyük liste üretir |
| `demo/frontend/node_modules/` | Frontend bağımlılıkları; dokümantasyon için gereksiz |
| `reports/presentation_build/node_modules/` | Sunum build bağımlılıkları; gereksiz |
| `data/raw/transactions_train.csv` | 3.49 GB; doğrudan açmak yerine scriptlerle okunmalı |

## En Sık Kullanılacak Bilgi Kaynakları

- Projenin kısa durumu: `START_HERE.md`
- Projenin tüm hafızası: `PROJECT_HANDOFF.md`
- Sonuç tablosu: `reports/kaggle_full_training_results.csv`
- Ranking sonuçları: `reports/ranking_metrics_validation_summary.md`
- Final rapor: `reports/FINAL_REPORT.md`
- Sıradaki işler: `docs/NEXT_STEPS.md`
- Karar gerekçeleri: `docs/DECISION_LOG.md`
