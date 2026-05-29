# Perseptron Sıradaki İşler

Bu dosya, projeye yeni devam edecek Codex hesabının veya bir geliştiricinin karar vermeden sıradaki işleri uygulayabilmesi için hazırlanmıştır.

## Güncel Kritik Not: 2026-05-30 Kaggle Rerun Sonrası

Teknik audit sırasında `notebooks/kaggle/perseptron_late_fusion.ipynb` içinde validation pozitifleri örneklendikten sonra index resetlenerek train tarafında yanlış satırların düşülme riski olduğu görüldü. Notebook düzeltildi ve artık `val_indices` doğrudan kullanılarak train/validation pozitif çiftleri ayrık oluşturuluyor.

Düzeltilmiş late-fusion notebooku Kaggle üzerinde yeniden çalıştırıldı. Yeni final late-fusion sonucu AUC-ROC `0.9341`, accuracy `0.8504` olarak kilitlendi ve checkpoint `models/final_kaggle/multimodal_fusion_streaming_full.pt` altına aktarıldı.

Güncel ilk sıra:

1. Final rapor, sunum taslağı ve demo metrik etiketlerinde yeni late-fusion sonucunun kaldığını doğrula.
2. Ranking sweep'i yeni checkpoint ile gerekirse yeniden üret.
3. Final raporu conference-paper formatında tamamla.
4. Sunumu final rapor hikayesine göre üret/güncelle.

Detaylı rerun notu: `docs/KAGGLE_LATE_FUSION_RERUN.md`

## Öncelik 1: Final Rapora Ranking Metrics ve Hybrid Açıklaması Eklemek

Amaç:

Final rapor, sonradan eklenen MAP@12 / Precision@10 / Recall@10 değerlendirmesini ve `late_fusion_hybrid` deneyini doğru konumlandırmalı.

Dokunulacak dosyalar:

- `reports/FINAL_REPORT.md`
- Gerekirse `reports/EXPERIMENT_COMPARISON.md`
- Gerekirse `reports/FINAL_REPORT_DRAFT.md`

Eklenecek içerik:

- `reports/ranking_metrics_validation_summary.md` tablosundan kısa sonuç özeti.
- Ranking evaluation'ın Kaggle public leaderboard ile birebir aynı şey olmadığı.
- CandidateRecall değerinin 0.475379 olduğu.
- HitRate@12'nin düşük kalmasının anlamı: doğru ürün aday havuzuna girse bile top-12 sıralamaya taşımak zor.
- `late_fusion_hybrid` açıklaması:
  - Eğitilmiş yeni model değildir.
  - Normal late fusion skoruna co-purchase, recency, metadata ve visual similarity sinyalleriyle reranking uygular.
  - Ana üç model karşılaştırmasına dördüncü model gibi konmamalıdır.

Başarı kriteri:

- Raporu okuyan kişi ana bilimsel karşılaştırmanın tabular-only, image-history ve multimodal late fusion arasında olduğunu anlar.
- Hybrid deneyin post-ranking stratejisi olduğunu anlar.
- Ranking metriklerinin destekleyici değerlendirme olduğu anlaşılır.

Risk/not:

- Hybrid sonucu daha yüksek olduğu için raporda yanlışlıkla "ana modelimiz hybrid" gibi sunulmamalı.
- Kaggle public score ile validation ranking metrikleri karıştırılmamalı.

## Öncelik 2: Sunumu Son Metriklerle Güncellemek

Amaç:

Sunum, final raporla aynı hikayeyi anlatmalı ve son ranking evaluation bilgisini içermeli.

Dokunulacak dosyalar:

- `reports/PRESENTATION_OUTLINE.md`
- `reports/perseptron_multimodal_recommendation_presentation.pptx`
- Gerekirse `reports/presentation_build/build_perseptron_presentation.mjs`

Eklenecek veya kontrol edilecek slayt içeriği:

- Üç ana modelin Kaggle full-training sonucu.
- Image-history modelin tabular-only'den güçlü çıkması.
- Late fusion modelin ana model olarak en iyi AUC/accuracy vermesi.
- Ranking evaluation'ın ayrı bir değerlendirme olduğu.
- Hybrid reranking'in ana model değil, pratik recommender sistemi iyileştirme deneyi olduğu.
- Demo arayüzü ekran görüntüsü veya kısa anlatımı.

Başarı kriteri:

- Sunum 8-12 dakika içinde anlatılabilir.
- Sunumdaki metrikler `reports/kaggle_full_training_results.csv` ile tutarlı olur.
- Hybrid model/deney ayrımı yanlış anlaşılmaz.

Risk/not:

- PowerPoint üretiminde görsel kayma olabilir; `reports/presentation_previews/` tekrar kontrol edilmeli.
- Sunum dosyası varsa doğrudan düzenlenebilir; yoksa build scripti üzerinden yeniden üretilebilir.

## Öncelik 3: Demo Smoke-Test Yapmak

Amaç:

Backend ve frontend'in final checkpointlerle çalıştığını doğrulamak.

Dokunulacak dosyalar:

- Normalde dosyaya dokunmadan test yapılır.
- Gerekirse:
  - `demo/backend/app.py`
  - `demo/backend/recommender.py`
  - `demo/frontend/src/App.jsx`
  - `demo/frontend/src/App.css`

Çalıştırma:

```powershell
python demo\backend\app.py
```

Ayrı terminal:

```powershell
cd demo\frontend
npm run dev -- --host 127.0.0.1
```

Kontrol listesi:

- `GET /api/health` yanıt veriyor.
- Katalog ürünleri geliyor.
- Ürün görselleri yükleniyor.
- Eksik görselde placeholder çalışıyor.
- Kullanıcı geçmiş ürün seçebiliyor.
- Customer metadata varsayılanları geliyor.
- `POST /api/recommend` üç model için öneri döndürüyor:
  - tabular-only
  - image-history
  - late-fusion
- Frontend önerileri yan yana okunabilir gösteriyor.

Başarı kriteri:

- Demo tarayıcıda açılır.
- En az bir geçmiş ürün seçilerek öneri alınır.
- Üç modelin önerileri görünür.

Risk/not:

- Kullanıcı frontend'i sonradan değiştirdi; UI son hali tekrar kontrol edilmeli.
- Görsel yüklenmiyorsa backend image route'u ve frontend `image_url` kullanımı kontrol edilmeli.
- `node_modules` eksikse `npm install` gerekir.

## Öncelik 4: Türkçe Karakter ve Rapor Dili Kontrolü

Amaç:

Markdown dosyaları Türkçe karakterlerle düzgün görünmeli ve ders dili Türkçe olduğu için rapor/sunum dili tutarlı olmalı.

Kontrol edilecek dosyalar:

- `README.md`
- `START_HERE.md`
- `PROJECT_HANDOFF.md`
- `docs/DOC_MAP.md`
- `docs/NEXT_STEPS.md`
- `docs/DECISION_LOG.md`
- `reports/FINAL_REPORT.md`
- `reports/EXPERIMENT_COMPARISON.md`
- `reports/ABLATION_VS_FULL_TRAINING.md`
- `reports/VISUAL_EXPLAINABILITY.md`
- `reports/TABULAR_EXPLAINABILITY.md`
- `reports/PRESENTATION_OUTLINE.md`

Başarı kriteri:

- VS Code veya Markdown preview'da Türkçe karakterler düzgün görünür.
- Ana terimler tutarlı kullanılır:
  - tabular-only
  - image-history
  - multimodal late fusion
  - post-ranking/reranking
  - EfficientNet-B0 embedding
  - visual similarity
  - permutation importance

Risk/not:

- PowerShell bazı UTF-8 dosyaları ekranda bozuk gösterebilir. Bu tek başına dosyanın bozuk olduğu anlamına gelmez.
- Kontrol VS Code/Markdown preview üzerinden yapılmalıdır.

## Öncelik 5: Opsiyonel Teknik İyileştirme Planı

Amaç:

Hybrid reranking'te işe yarayan sinyallerin eğitim sırasında feature olarak modele eklenmesini planlamak.

Bu teslim için zorunlu değildir. İleri çalışma olarak sunulabilir.

Olası yeni feature'lar:

- `co_purchase_score`
- `recent_popularity_score`
- `last_7d_popularity_rank`
- `last_30d_popularity_rank`
- `metadata_match_count`
- `visual_neighbor_rank`

Beklenen fayda:

- Model, co-purchase ve recency sinyalini sonradan heuristic olarak değil, eğitim sırasında öğrenir.
- MAP@12 / Precision@10 tarafında daha gerçekçi iyileşme ihtimali doğar.

Risk/not:

- Kaggle full-training notebooklarını yeniden düzenlemek gerekir.
- Büyük veri nedeniyle eğitim tekrar uzun sürebilir.
- Bu çalışma final teslim için değil, gelecek çalışma olarak daha uygun olabilir.

## Yeni Hesap İçin İlk Uygulama Önerisi

Yeni Codex hesabı devam edecekse ilk görev şu olmalı:

```text
START_HERE.md, PROJECT_HANDOFF.md ve docs/NEXT_STEPS.md dosyalarını oku. Öncelik 1'i uygula: reports/FINAL_REPORT.md dosyasına ranking metrics ve late_fusion_hybrid açıklamasını, ana üç model karşılaştırmasını bozmadan ekle.
```

## Tamamlanmış Sayılacak Durum

Bu onboarding/dokümantasyon setinden sonra proje teslim öncesi şu durumda olmalıdır:

- Yeni hesap proje bağlamını sohbet geçmişi olmadan anlayabilir.
- Final rapor güncel metrikleri içerir.
- Sunum final raporla tutarlıdır.
- Demo çalıştığı doğrulanmıştır.
- Hybrid/ranking açıklamaları yanlış konumlandırılmamıştır.
