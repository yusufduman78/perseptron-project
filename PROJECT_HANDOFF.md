# Perseptron Proje Devri ve Çalışma Hafızası

Son güncelleme: 2026-05-27  
Yerel proje dizini: `C:\Users\Yusuf\Desktop\perseptron_project`

Bu dosya, Codex sohbet geçmişine erişim olmadan projeye kaldığı yerden devam edebilmek için hazırlanmıştır. Yeni bir hesapta, yeni bir Codex oturumunda veya başka bir bilgisayarda projeyi açınca önce bu dosya okutulmalıdır.

## Hızlı Onboarding Katmanı

Bu dosya projenin canonical çalışma hafızasıdır; ancak yeni başlayan bir hesap veya insan okuyucu önce kısa dosyalardan bağlam almalıdır:

| Dosya | Rol |
|---|---|
| `START_HERE.md` | 5-10 dakikalık hızlı proje özeti, okuma sırası ve yeni Codex başlangıç promptu |
| `PROJECT_HANDOFF.md` | Ayrıntılı proje hafızası ve teknik bağlam |
| `docs/DOC_MAP.md` | Hangi dosya ve klasörün ne işe yaradığını gösteren harita |
| `docs/NEXT_STEPS.md` | Bundan sonra yapılacak işleri öncelik, başarı kriteri ve riskleriyle listeler |
| `docs/DECISION_LOG.md` | EfficientNet, Kaggle full-training, SHAP/Grad-CAM, hybrid reranking gibi kararların gerekçeleri |

Önerilen başlangıç mesajı:

```text
C:\Users\Yusuf\Desktop\perseptron_project\START_HERE.md dosyasını oku. Sonra PROJECT_HANDOFF.md, docs/DOC_MAP.md ve docs/NEXT_STEPS.md dosyalarını okuyup projenin mevcut durumunu, ana sonuçlarını ve sıradaki uygulanabilir işleri özetle.
```

## 1. Projenin Kısa Tanımı

Bu proje, H&M Personalized Fashion Recommendation veri seti üzerinde multimodal bir moda ürünü öneri sistemi geliştirmeyi amaçlar. Ana fikir, yalnızca müşteri ve ürün metaverisiyle öneri üretmek yerine ürün fotoğraflarından çıkarılan görsel temsilleri de kullanarak daha güçlü bir satın alma olasılığı modeli kurmaktır.

Projenin ana hipotezi:

> Ürün fotoğraflarından EfficientNet-B0 ile çıkarılan görsel embeddingler, müşterinin geçmiş satın alımlarından oluşturulan görsel tercih profili ve tabular müşteri/ürün özellikleri late fusion mimarisinde birleştirildiğinde, sadece tabular veya sadece görsel geçmiş kullanan modellere göre daha güçlü bir satın alma skoru üretir.

Bu nedenle üç ana model karşılaştırılmıştır:

| Model | Kısa açıklama | Kullanılan bilgi |
|---|---|---|
| Tabular-only MLP | Sadece müşteri ve ürün metaverisiyle çalışan baseline | `customers.csv`, `articles.csv`, işlemden türetilmiş sayısal özellikler |
| Image-history MLP | Ürün görsellerinden çıkarılan embeddingler ve müşteri görsel geçmişiyle çalışan görsel baseline | EfficientNet-B0 embeddingleri, müşteri geçmiş profil vektörü |
| Multimodal late fusion | Tabular branch ve image-history branch'i birleştiren ana model | Tabular özellikler + görsel embedding/profile özellikleri |

Sonradan ayrıca bir değerlendirme deneyi olarak `late_fusion_hybrid` eklendi. Bu yeni bir eğitimli model değildir; final late fusion skorunu co-purchase, recency, metadata benzerliği ve visual similarity sinyalleriyle yeniden sıralayan post-ranking/reranking denemesidir.

## 2. Proje Kapsamı ve Projenin Asıl İddiası

Bu proje bir Kaggle leaderboard çözümünü optimize etmekten çok, ders projesi bağlamında multimodal recommendation hipotezini test eder.

Asıl karşılaştırma:

1. Tabular-only model
2. Image-history model
3. Multimodal late-fusion model

Bu üç modelin aynı full-training düzeninde karşılaştırılması projenin ana bilimsel sonucudur.

Kaggle submission skorları yardımcı dış test olarak kullanıldı, fakat proje iddiasının merkezi Kaggle public score değildir. H&M yarışmasında leaderboard skorunu yükseltmek için candidate generation, co-purchase, trend/popularity, ensemble ve week-aware retrieval gibi ek yarışma stratejileri gerekir. Biz bunlardan bazılarını sonradan validation ranking evaluation içinde test ettik.

## 3. Veri Seti

Kullanılan veri seti: H&M Personalized Fashion Recommendations.

Yerel ham veri klasörü:

```text
data/raw/
```

Dosyalar:

| Dosya | Rol | Yaklaşık boyut |
|---|---|---:|
| `articles.csv` | Ürün metaverisi | 36 MB |
| `customers.csv` | Müşteri metaverisi | 207 MB |
| `transactions_train.csv` | Satın alma geçmişi | 3.49 GB |

Görseller:

```text
data/images/hm_images/
```

Bu klasörde 105,100 adet ürün görseli bulunuyor. Klasör yapısı H&M/Kaggle formatına benzer:

```text
data/images/hm_images/010/0108775015.jpg
data/images/hm_images/011/0110065001.jpg
...
```

Ürün görsel yolu şu mantıkla bulunur:

```python
padded = str(article_id).zfill(10)
path = data/images/hm_images/{padded[:3]}/{padded}.jpg
```

## 4. Embedding Mantığı

Ürün görselleri eğitim sırasında tekrar tekrar CNN'e sokulmadı. Bunun yerine EfficientNet-B0 pretrained modelinden ürün başına 1280 boyutlu görsel embedding çıkarıldı.

Önemli nokta:

> Image-only/Image-history eğitiminde CNN sıfırdan eğitilmedi. EfficientNet-B0 burada feature extractor olarak kullanıldı; çıkan embeddingler daha sonra MLP modellerine girdi oldu.

Final embedding cache:

```text
data/embeddings/final_kaggle/
```

Dosyalar:

| Dosya | Açıklama | Boyut |
|---|---|---:|
| `article_image_embeddings_popular.npy` | 105,100 x 1280 EfficientNet-B0 embedding matrisi | 538 MB |
| `article_image_embedding_ids_popular.csv` | Embedding satırlarının karşılık geldiği article_id listesi | 1.1 MB |

Bu dosyalar Kaggle'da dataset olarak yüklenmişti. Kaggle yolları şunlardı:

```text
/kaggle/input/datasets/smoke78/article-image-embeddings-popular-npy/article_image_embeddings_popular.npy
/kaggle/input/datasets/smoke78/article-image-embedding-ids-popular-csv/article_image_embedding_ids_popular.csv
```

Yerel karşılığı:

```text
data/embeddings/final_kaggle/article_image_embeddings_popular.npy
data/embeddings/final_kaggle/article_image_embedding_ids_popular.csv
```

Eski/local sandbox embedding cache:

```text
data/embeddings/local_sandbox/
```

Bu eski denemeler içindir. Final eğitim ve demo tarafında ana referans `final_kaggle` embedding setidir.

## 5. Model Mimarileri

### 5.1 Tabular-only MLP

Amaç: Görsel bilgi kullanmadan müşteri ve ürün metaverisiyle müşteri-ürün satın alma olasılığı tahmin etmek.

Girdi örnekleri:

- Müşteri özellikleri:
  - `FN`
  - `Active`
  - `age`
  - `club_member_status`
  - `fashion_news_frequency`
- Ürün özellikleri:
  - `product_code`
  - `product_type_no`
  - `department_no`
  - `index_code`
  - `section_no`
  - `garment_group_no`
  - `colour_group_code`
  - ürün adı/tipi/grubu gibi kategorik alanlar
- Satın alma geçmişinden türetilen:
  - `pair_purchase_count`

Model:

- Kategorik alanlar embedding layer ile temsil edilir.
- Sayısal alanlar normalize edilir.
- Kategorik embeddingler ve sayısal özellikler MLP'ye verilir.
- Çıktı, müşteri-ürün çiftinin satın alınma olasılığına karşılık gelen logit/skordur.

Final checkpoint:

```text
models/final_kaggle/tabular_only_streaming_full.pt
```

### 5.2 Image-history MLP

Amaç: Tabular metaveri kullanmadan, yalnızca ürün görsellerinden türetilen embeddingler ve müşterinin geçmiş görsel tercih profiliyle satın alma olasılığı tahmin etmek.

Görsel müşteri profili:

1. Müşterinin geçmişte satın aldığı ürünlerin EfficientNet-B0 embeddingleri alınır.
2. Bu embeddingler toplanır/ortalaması alınır.
3. Aday ürün geçmişte satın alınmışsa leakage'i azaltmak için pair count etkisi düşülür.
4. Müşteri görsel profili normalize edilir.

Image-history model girdileri:

- Aday ürün embeddingi
- Müşteri görsel profil embeddingi
- Aday ürün ile profil arasındaki cosine similarity
- Görsel geçmiş sayısı / history count

Model:

- `article_emb`
- `profile_emb`
- `visual_similarity`
- `visual_history_count`

birleştirilip MLP'ye verilir.

Final checkpoint:

```text
models/final_kaggle/image_history_streaming_full.pt
```

### 5.3 Multimodal Late Fusion

Amaç: Tabular branch ve görsel branch'i aynı modelde birleştirmek.

Mimari:

1. Tabular branch:
   - Sayısal müşteri/ürün özellikleri
   - Kategorik müşteri/ürün embeddingleri
   - MLP ile 128 boyutlu temsil
2. Visual branch:
   - Aday ürün embeddingi
   - Müşteri görsel geçmiş profili
   - Visual similarity
   - Visual history count
   - MLP ile 128 boyutlu temsil
3. Fusion head:
   - Tabular temsil + visual temsil concatenate edilir.
   - Son MLP satın alma logit/skoru üretir.

Final checkpoint:

```text
models/final_kaggle/multimodal_fusion_streaming_full.pt
```

Bu model projenin ana modelidir.

## 6. Eğitim Aşamaları ve Nasıl İlerlendi

### 6.1 İlk Lokal Fazlar

Başlangıçta local ortamda küçük/sandbox deneyleri yapıldı:

```text
src/local_phases/
```

Önemli scriptler:

| Script | Görev |
|---|---|
| `phase1_eda.py` | İlk veri keşfi / EDA |
| `phase2_tabular_mlp.py` | İlk tabular MLP baseline |
| `phase2_image_baseline.py` | İlk görsel baseline denemesi |
| `phase2_prepare_image_index.py` | Görsel index hazırlığı |
| `phase3_extract_image_embeddings.py` | EfficientNet-B0 embedding çıkarımı |
| `phase3_build_visual_features.py` | Visual similarity feature üretimi |
| `phase3_image_history_mlp.py` | Lokal image-history MLP |
| `phase3_multimodal_fusion.py` | Lokal multimodal fusion |
| `phase3_visual_tabular_mlp.py` | Görsel scalar feature + tabular MLP |
| `phase3_controlled_ablation.py` | Kontrollü ablation deneyi |

Local fazlar, yaklaşımın çalıştığını görmek ve model tasarımını belirlemek için kullanıldı. Final iddia için esas alınan sonuçlar Kaggle full-training sonuçlarıdır.

### 6.2 Kaggle'a Geçiş

Full veri ile eğitim local ortamda bellek açısından zor olduğu için Kaggle ortamına geçildi. Kaggle üzerinde:

- GPU olarak Tesla T4 kullanıldı.
- Büyük veri tek seferde belleğe alınmak yerine streaming/chunk mantığıyla işlendi.
- Full pozitif çiftler üzerinden eğitim yapıldı.
- Negatif örnekleme kullanıldı.
- Validation için 120,000 pozitif satır ayrıldı.
- Modeller 3 epoch eğitildi.

Kaggle notebookları:

```text
notebooks/kaggle/tabular_only_fulltraining.ipynb
notebooks/kaggle/image_history_fulltraining_perseptron.ipynb
notebooks/kaggle/perseptron_late_fusion.ipynb
```

Eski tek hücreli/yardımcı pipeline:

```text
notebooks/archive/kaggle_multimodal_pipeline_notebook.ipynb
src/kaggle/kaggle_multimodal_pipeline.py
```

## 7. Final Full-Training Sonuçları

Ana sonuç dosyası:

```text
reports/kaggle_full_training_results.csv
```

Sonuçlar:

| Model | Training setup | AUC-ROC | Accuracy | Full pozitif çift | Validation pozitif satır |
|---|---|---:|---:|---:|---:|
| Tabular-only MLP | streaming full, 3 epoch | 0.85498 | 0.76361 | 27,194,909 | 120,000 |
| Image-history MLP | streaming full, 3 epoch | 0.92373 | 0.82972 | 27,194,909 | 120,000 |
| Multimodal late fusion | streaming full, 3 epoch | 0.93540 | 0.85370 | 27,194,909 | 120,000 |

Yorum:

- Image-history modeli tabular-only'den belirgin şekilde daha güçlü çıktı.
- Late fusion modeli en yüksek AUC ve accuracy değerini verdi.
- Bu sonuç, ürün görsellerinden çıkarılan temsilin müşteri tercihlerini modellemede güçlü bir sinyal olduğunu ve tabular bilgiyle birleştirildiğinde performansı artırdığını destekler.

## 8. Controlled Ablation Sonuçları

Dosya:

```text
reports/phase3_controlled_ablation_results.csv
```

Sonuçlar:

| Model | Satır | Train | Validation | Epoch | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Tabular-only MLP | 1,014,202 | 811,361 | 202,841 | 3 | 0.72194 | 0.66581 |
| Image-history MLP | 1,014,202 | 811,361 | 202,841 | 3 | 0.78825 | 0.72614 |
| Multimodal late fusion | 1,014,202 | 811,361 | 202,841 | 3 | 0.96113 | 0.84111 |

Bu ablation sonuçları full-training ile birebir aynı ölçekte yorumlanmamalıdır.

Neden:

- Ürün ve müşteri evreni daha küçüktür.
- Validation dağılımı daha kolay olabilir.
- Negatif örnekleme farklı etki yaratabilir.
- Full training daha büyük, daha gerçekçi ve daha gürültülüdür.

Bu farkı açıklayan dosya:

```text
reports/ABLATION_VS_FULL_TRAINING.md
```

Rapor için doğru yaklaşım:

- Ana sonuç olarak Kaggle full-training tablosu kullanılmalı.
- Controlled ablation destekleyici deney olarak sunulmalı.
- Ablation sonuçları "hipotezin kontrollü ortamda da desteklendiğini" göstermek için kullanılmalı.

## 9. Phase 2 ve Erken Phase 3 Sonuçları

Dosya:

```text
reports/phase2_baseline_results.csv
```

Öne çıkanlar:

| Faz | Model | AUC-ROC | Accuracy |
|---|---|---:|---:|
| phase2 | tabular_mlp | 0.72857 | 0.66403 |
| phase2 | image_effnet_b0, 2,000 row | 0.49551 | 0.45750 |
| phase2 | image_effnet_b0, 20,000 row | 0.60344 | 0.55850 |
| phase3 | visual_similarity_tabular_mlp | 0.81632 | 0.73659 |
| phase3 | multimodal_late_fusion | 0.97010 | 0.90562 |
| phase3 | image_history_mlp | 0.79741 | 0.73916 |

Bu erken sonuçlar final iddianın ana kanıtı değildir; model tasarımının evrimini gösterir.

## 10. Ranking Evaluation: MAP@12 / Precision@10 / Recall@10

Proposal tarafında MAP@12 / Precision@10 / Recall@10 gibi ranking metrikleri de konuşulduğu için sonradan zaman tabanlı validation ranking değerlendirmesi eklendi.

Script:

```text
src/evaluation/ranking_metrics_validation.py
```

Çıktılar:

```text
reports/ranking_metrics_validation.csv
reports/ranking_metrics_validation_summary.md
```

Deney düzeni:

- Validation penceresi: son 7 gün
- Cutoff tarihi: 2020-09-15
- Değerlendirilen müşteri sayısı: 500
- Candidate limit: 5000
- Visual neighbors: 3000
- Co-purchase per item: 300
- Co-purchase max history/customer: 100
- Hybrid heuristic weight: 0.45
- Top-k: 12
- Precision/Recall k: 10

Metrikler:

| Model | MAP@12 | Precision@10 | Recall@10 | HitRate@12 | CandidateRecall |
|---|---:|---:|---:|---:|---:|
| tabular_only | 0.001171 | 0.001200 | 0.003933 | 0.018000 | 0.475379 |
| image_history | 0.003236 | 0.002400 | 0.007221 | 0.030000 | 0.475379 |
| late_fusion | 0.002427 | 0.002600 | 0.008488 | 0.032000 | 0.475379 |
| late_fusion_hybrid | 0.005316 | 0.003400 | 0.010305 | 0.032000 | 0.475379 |
| candidate_heuristic | 0.005322 | 0.003600 | 0.010971 | 0.032000 | 0.475379 |

Burada `late_fusion_hybrid` yeni eğitilmiş bir model değildir. Normal late fusion skorunun üstüne:

- son 7/30 gün popülerliği,
- co-purchase,
- metadata benzerliği,
- visual similarity

sinyalleriyle yapılan reranking deneyidir.

Yorum:

- CandidateRecall yaklaşık 0.475'e çıktı; yani doğru ürünlerin yaklaşık yüzde 47.5'i aday havuzuna girebiliyor.
- HitRate@12 düşük kaldı; yani doğru ürün aday havuzuna girse bile top 12'ye taşımak zor.
- Hybrid/rerank MAP@12'yi normal late fusion'a göre artırdı.
- Bu sonuç yarışma tipi recommender sistemlerinde retrieval + reranking aşamasının ne kadar önemli olduğunu gösterir.

Rapor dilinde dikkat:

- Ana üç model karşılaştırmasına `late_fusion_hybrid` aynı seviyede sokulmamalı.
- `late_fusion_hybrid`, "post-ranking iyileştirme / pratik recommender sistemi deneyi" olarak ayrı anlatılmalı.

## 11. Kaggle Submission Denemeleri

Kaggle submission tarafı ana bilimsel sonuç değil, yardımcı dış testti. Konuşma içinde kaydedilen skorlar:

| Deneme | Public Score | Private Score | Not |
|---|---:|---:|---|
| pre-fulltraining fast run öncesi | 0.00360 | 0.00387 | İlk submission |
| pre-fulltraining version 2 | 0.00372 | 0.00396 | Küçük iyileştirme |
| pre-fulltraining version 3 | 0.00510 | 0.00516 | En iyi görünen submission |
| pre-fulltraining version 4 | 0.00484 | 0.00488 | Bir önceki denemeden düşük |

Bu skorlar düşük göründü; bunun nedeni Kaggle yarışmasının esas başarısının model AUC'sinden çok doğru candidate generation, week-aware popularity, co-purchase ve retrieval stratejilerine bağlı olmasıdır.

Önemli ayrım:

- Eğitim AUC/accuracy: verilen pozitif/negatif çiftlerde modelin satın alma olasılığını ayırt etmesi.
- Kaggle MAP@12: her müşteri için doğru 12 ürünü geniş ürün evreni içinden sıralamak.

Bu yüzden AUC yüksek olsa bile Kaggle score düşük kalabilir.

## 12. Açıklanabilirlik Çalışmaları

Proposal'da açıklanabilirlik tarafı da vardı. İki ana çıktı hazırlandı.

### 12.1 Görsel Benzerlik Açıklamaları

Script:

```text
src/explainability/visual_similarity_explanations.py
```

Çıktılar:

```text
reports/visual_similarity_examples.csv
reports/visual_similarity_examples.html
reports/VISUAL_EXPLAINABILITY.md
```

Mantık:

1. Aday ürün embeddingi alınır.
2. Müşterinin geçmiş satın alımlarındaki ürün embeddingleri alınır.
3. Aday ürün ile geçmiş ürünler arasında cosine similarity hesaplanır.
4. En benzer geçmiş ürünler aday önerinin görsel açıklaması olarak gösterilir.

Bu Grad-CAM yerine tercih edildi çünkü EfficientNet-B0 final sınıflandırıcı olarak fine-tune edilmedi; sabit feature extractor olarak kullanıldı. Dolayısıyla modelin nihai skorunu açıklamak için embedding uzayındaki benzer ürünleri göstermek daha tutarlı oldu.

Örnek çıktı:

- Aday ürün: `0673677002`, Henry polo, Sweater, Black
- Benzer geçmiş ürün: `0666448006`, Janet sweater, Sweater, Blue
- Cosine similarity: 0.6237

### 12.2 Tabular Permutation Importance

Script:

```text
src/explainability/tabular_permutation_importance.py
```

Çıktılar:

```text
reports/tabular_permutation_importance.csv
reports/tabular_feature_group_importance.csv
reports/TABULAR_EXPLAINABILITY.md
```

Feature group importance:

| Feature group | Total AUC drop | Mean AUC drop | Feature count |
|---|---:|---:|---:|
| article_categorical | 0.15623 | 0.01302 | 12 |
| article_numeric | 0.01971 | 0.00197 | 10 |
| customer_numeric | 0.01522 | 0.00507 | 3 |
| customer_categorical | 0.00051 | 0.00026 | 2 |

En etkili tekil özelliklerden bazıları:

| Feature | AUC drop |
|---|---:|
| `section_name` | 0.04715 |
| `department_name` | 0.02359 |
| `product_type_name` | 0.01765 |
| `age` | 0.01522 |
| `product_code` | 0.01384 |
| `colour_group_name` | 0.01302 |

Yorum:

- Tabular modelde en güçlü sinyal ürün kategorik özelliklerinden geliyor.
- Müşteri kategorik alanları zayıf çıktı.
- `age` müşteri tarafında anlamlı sinyal verdi.

### 12.3 SHAP Durumu

SHAP proposal'da düşünülmüştü ama uygulanmadı.

Neden:

- PyTorch modeli categorical embedding layer'lar kullanıyor.
- Preprocessing + embedding + numeric normalization zinciri SHAP için doğrudan temiz bir tabular input yapısı vermiyor.
- Zaman ve proje kapsamı açısından permutation importance daha uygulanabilir ve raporlanabilir oldu.

SHAP ileriki çalışma olarak not edildi.

## 13. Demo Arayüzü

Demo iki parçadan oluşuyor:

```text
demo/backend/
demo/frontend/
```

Backend:

```text
demo/backend/app.py
demo/backend/recommender.py
demo/backend/test_recommend.py
```

Frontend:

```text
demo/frontend/
```

Teknik yapı:

- Backend: FastAPI
- Model inference: PyTorch checkpointleri
- Frontend: React + Vite
- Görseller: `data/images/hm_images/` klasöründen static olarak servis ediliyor
- Eksik görsel varsa placeholder SVG dönüyor

Backend çalıştırma:

```powershell
python demo\backend\app.py
```

Backend varsayılan adres:

```text
http://127.0.0.1:8000
```

Frontend çalıştırma:

```powershell
cd demo\frontend
npm install
npm run dev -- --host 127.0.0.1
```

Frontend varsayılan adres:

```text
http://127.0.0.1:5173
```

Terminal demo testi:

```powershell
python demo\backend\test_recommend.py
```

API endpointleri:

| Endpoint | Görev |
|---|---|
| `GET /api/health` | Backend sağlık kontrolü |
| `GET /api/customer-defaults` | Varsayılan müşteri metadata |
| `GET /api/catalog` | Popüler/katalog ürünleri |
| `POST /api/recommend` | Seçilen geçmiş ürünlere göre öneri üretimi |
| `GET /api/image-placeholder/{article_id}.svg` | Eksik görsel placeholder |
| `/images/...` | Ürün görsellerini static servis eder |

Demo ne yapıyor:

1. Kullanıcı katalogdan daha önce satın aldığı varsayılan ürünleri seçiyor.
2. İsterse müşteri metadata alanlarını değiştiriyor.
3. Backend üç modelden ayrı ayrı öneri döndürüyor:
   - tabular-only
   - image-history
   - late-fusion
4. Frontend bu önerileri alışveriş sitesi benzeri arayüzde gösteriyor.

Önemli not:

- Kullanıcı frontend tarafında arayüzü sonradan değiştirdi. Bu nedenle son UI hali tekrar görsel olarak smoke-test edilmelidir.
- Bazı görsellerin yüklenmemesi durumunda muhtemel nedenler:
  - article_id'nin görsel dosyası yoktur,
  - path eşleşmesi farklıdır,
  - frontend backend yerine doğrudan yanlış image path kullanıyordur,
  - backend static `/images` route'u çalışmıyordur,
  - frontend API base URL farklıdır.

## 14. Önemli Komutlar

### 14.1 Ranking Metrics

```powershell
python src\evaluation\ranking_metrics_validation.py --sample-customers 500 --candidate-limit 5000 --visual-neighbors 3000 --co-purchase-per-item 300 --co-purchase-max-history-per-customer 100 --hybrid-heuristic-weight 0.45 --output-csv reports\ranking_metrics_validation.csv --summary-md reports\ranking_metrics_validation_summary.md
```

Bu komut uzun sürebilir çünkü `transactions_train.csv` 31 milyondan fazla satır içerir.

### 14.2 Görsel Açıklanabilirlik

```powershell
python src\explainability\visual_similarity_explanations.py
```

Opsiyonel olarak recommendation CSV verilebilir:

```powershell
python src\explainability\visual_similarity_explanations.py --recommendations-csv path\to\submission.csv
```

Daha önce `reports\submission.csv` olmadığı için bu komut hata vermişti. Çünkü submission dosyası yerelde `reports/` altında yoktu. Kaggle submission dosyaları indirilmişse doğru path verilmelidir.

### 14.3 Tabular Permutation Importance

```powershell
python src\explainability\tabular_permutation_importance.py
```

### 14.4 Backend Demo

```powershell
python demo\backend\app.py
```

### 14.5 Frontend Demo

```powershell
cd demo\frontend
npm run dev -- --host 127.0.0.1
```

### 14.6 Frontend Build

```powershell
cd demo\frontend
npm run build
```

## 15. Klasör Yapısı

Önemli klasörler:

```text
perseptron_project/
├── data/
│   ├── raw/
│   ├── images/hm_images/
│   ├── embeddings/
│   │   ├── final_kaggle/
│   │   └── local_sandbox/
│   └── processed/sandbox/
├── models/
│   ├── final_kaggle/
│   └── local_experiments/
├── notebooks/
│   ├── kaggle/
│   └── archive/
├── reports/
├── src/
│   ├── evaluation/
│   ├── explainability/
│   ├── kaggle/
│   ├── local_phases/
│   └── utils/
├── demo/
│   ├── backend/
│   └── frontend/
└── docs/
```

### 15.1 `data/`

Projenin veri varlıkları.

Önemli alt klasörler:

- `data/raw/`: Ham H&M CSV dosyaları.
- `data/images/hm_images/`: 105,100 ürün görseli.
- `data/embeddings/final_kaggle/`: Final embedding cache.
- `data/embeddings/local_sandbox/`: Eski/local embedding cache.
- `data/processed/sandbox/`: Local fazlardan kalan ara CSV'ler.

### 15.2 `models/`

Model checkpointleri.

Final:

```text
models/final_kaggle/
```

Dosyalar:

- `tabular_only_streaming_full.pt`
- `image_history_streaming_full.pt`
- `multimodal_fusion_streaming_full.pt`
- `tabular_only_full_results.csv`
- `image_history_full_results.csv`

Local/erken deneyler:

```text
models/local_experiments/
```

Dosyalar:

- `tabular_mlp.pt`
- `image_effnet_b0.pt`
- `image_history_mlp.pt`
- `multimodal_fusion.pt`
- `visual_tabular_mlp.pt`
- `controlled_tabular_only_mlp.pt`
- `controlled_image_history_mlp.pt`
- `controlled_multimodal_late_fusion.pt`

### 15.3 `notebooks/`

Kaggle notebookları:

```text
notebooks/kaggle/tabular_only_fulltraining.ipynb
notebooks/kaggle/image_history_fulltraining_perseptron.ipynb
notebooks/kaggle/perseptron_late_fusion.ipynb
```

Archive:

```text
notebooks/archive/kaggle_multimodal_pipeline_notebook.ipynb
```

### 15.4 `reports/`

Rapor, metrik ve sunum çıktıları.

Önemli dosyalar:

| Dosya | Rol |
|---|---|
| `FINAL_REPORT.md` | Ana final rapor taslağı |
| `FINAL_REPORT_DRAFT.md` | Rapor taslak metni |
| `EXPERIMENT_COMPARISON.md` | Model sonuç karşılaştırması |
| `ABLATION_VS_FULL_TRAINING.md` | Ablation/full training farkı açıklaması |
| `PIPELINE_DIAGRAMS.md` | Mermaid pipeline/model diyagramları |
| `PRESENTATION_OUTLINE.md` | Sunum akışı |
| `perseptron_multimodal_recommendation_presentation.pptx` | PowerPoint sunumu |
| `kaggle_full_training_results.csv` | Ana final eğitim sonuç tablosu |
| `phase2_baseline_results.csv` | Erken deney sonuçları |
| `phase3_controlled_ablation_results.csv` | Controlled ablation sonuçları |
| `ranking_metrics_validation.csv` | Müşteri-model bazlı ranking detayları |
| `ranking_metrics_validation_summary.md` | Ranking değerlendirme özeti |
| `visual_similarity_examples.csv` | Görsel açıklama örnekleri |
| `visual_similarity_examples.html` | Görsel açıklama HTML raporu |
| `tabular_permutation_importance.csv` | Tabular feature importance |
| `tabular_feature_group_importance.csv` | Feature group importance |

### 15.5 `src/`

Kodların ana klasörü.

Alt klasörler:

- `src/local_phases/`: Yerel faz scriptleri.
- `src/kaggle/`: Kaggle pipeline kaynak scripti.
- `src/explainability/`: Açıklanabilirlik scriptleri.
- `src/evaluation/`: Ranking evaluation scripti.
- `src/utils/`: Yardımcı scriptler.

### 15.6 `demo/`

Canlı demo.

- `demo/backend/`: FastAPI + PyTorch inference.
- `demo/frontend/`: React/Vite alışveriş sitesi benzeri arayüz.

### 15.7 `docs/`

Proposal ve proje belgeleri.

Önemli dosya:

```text
docs/yusuf_duman_22118080056_proposal.docx
```

## 16. Proposal ile Eşleşme

Proposal'da amaçlanan genel çerçeve:

- H&M veri seti ile moda öneri sistemi.
- Görsel ürün temsilleri.
- Müşteri geçmişine göre görsel tercih profili.
- Tabular müşteri/ürün metadata.
- Multimodal late fusion.
- Baseline karşılaştırmaları.
- Metrikler.
- Açıklanabilirlik.

Tamamlananlar:

- Tabular-only baseline tamamlandı.
- Image-history/görsel baseline tamamlandı.
- Multimodal late fusion tamamlandı.
- EfficientNet-B0 embedding extraction tamamlandı.
- Kaggle full-training tamamlandı.
- Controlled ablation tamamlandı.
- MAP@12 / Precision@10 / Recall@10 validation evaluation eklendi.
- Görsel benzerlik açıklanabilirliği tamamlandı.
- Tabular permutation importance tamamlandı.
- Demo backend/frontend hazırlandı.
- Rapor ve sunum taslakları hazırlandı.

Eksik veya not edilenler:

- SHAP uygulanmadı, ileriki çalışma olarak kaldı.
- Grad-CAM uygulanmadı; yerine embedding-space visual similarity yapıldı.
- Kaggle leaderboard optimizasyonu sınırlı kaldı.
- UI kullanıcı tarafından sonradan değiştirildiği için final smoke-test gerekiyor.
- Hybrid reranking eğitimli model değil; raporda ayrı konumlandırılmalı.

## 17. Late Fusion Hybrid Nedir?

Son konuşmalarda sorulan önemli konu buydu.

`late_fusion_hybrid`, normal `late_fusion` modelinden farklıdır.

Normal late fusion:

- Eğitimli modeldir.
- Tabular branch + visual branch skorunu öğrenir.
- Checkpoint: `models/final_kaggle/multimodal_fusion_streaming_full.pt`

Late fusion hybrid:

- Yeni eğitilmiş bir model değildir.
- Mevcut late fusion skorunu alır.
- Üstüne heuristic/reranking skoru karıştırır.

Hybrid sinyaller:

- Son 7/30 gün popülerliği
- Co-purchase ilişkisi
- Metadata benzerliği
- Görsel embedding yakınlığı

Neden normal late fusion'dan yüksek çıktı?

Çünkü H&M gibi recommender problemlerinde doğru adayı bulmak ve top-k sıralamada yukarı taşımak sadece modelin pairwise AUC'sine bağlı değildir. Trend ve beraber satın alma sinyali çok güçlüdür. Hybrid, modelin öğrenmediği bu retrieval/ranking sinyallerini sonradan eklediği için MAP@12 ve Precision@10 tarafında daha iyi çıktı.

Raporda nasıl sunulmalı:

- Ana model karşılaştırmasına "dördüncü model" gibi eklenmemeli.
- "Post-ranking strategy" veya "hybrid reranking experiment" olarak ayrı anlatılmalı.

## 18. Nerede Kaldık?

Son durumda proje teknik olarak büyük ölçüde tamamlanmış durumda.

Tamamlanan ana işler:

1. Veri düzeni kuruldu.
2. Görseller indirildi/yerleştirildi.
3. EfficientNet-B0 embeddingleri çıkarıldı.
4. Local fazlar tamamlandı.
5. Kaggle full-training notebookları hazırlandı ve çalıştırıldı.
6. Tabular-only, image-history ve late-fusion modelleri eğitildi.
7. Final checkpointler projeye alındı.
8. Metrik tabloları oluşturuldu.
9. Explainability çıktıları üretildi.
10. Ranking evaluation eklendi.
11. Hybrid reranking denendi.
12. Demo backend/frontend hazırlandı.
13. Rapor ve sunum taslakları hazırlandı.

En son konuşulan teknik konu:

- `late_fusion_hybrid` nedir ve neden normal late fusion'dan yüksek çıktı?
- Cevap: Eğitimli yeni model değil, post-ranking/reranking deneyi; co-purchase/trend/metadata/visual similarity sinyalleriyle sıralamayı iyileştirdi.

## 19. Sıradaki Yapılacaklar

Önerilen mantıklı sıra:

### 19.1 Final raporu güncelle

`reports/FINAL_REPORT.md` içinde ranking evaluation ve hybrid reranking kısmı henüz son haliyle işlenmemiş olabilir. Eklenmeli:

- MAP@12 / Precision@10 / Recall@10 tablosu
- CandidateRecall yorumu
- HitRate@12'nin neden düşük kaldığı
- Late fusion hybrid'in yeni model değil post-ranking stratejisi olduğu
- Kaggle leaderboard skorlarının neden düşük kaldığı

### 19.2 Rapor dilini tekilleştir

Ders dili Türkçe. Tüm rapor ve sunum Türkçe olmalı.

Kontrol edilecekler:

- `README.md`
- `reports/FINAL_REPORT.md`
- `reports/EXPERIMENT_COMPARISON.md`
- `reports/ABLATION_VS_FULL_TRAINING.md`
- `reports/TABULAR_EXPLAINABILITY.md`
- `reports/VISUAL_EXPLAINABILITY.md`
- `reports/PRESENTATION_OUTLINE.md`

Bazı PowerShell çıktılarında Türkçe karakterler bozuk görünebilir. Dosyalar UTF-8 olarak okunmalı.

### 19.3 Sunumu final hale getir

Mevcut sunum:

```text
reports/perseptron_multimodal_recommendation_presentation.pptx
```

Önizlemeler:

```text
reports/presentation_previews/
```

Sunumda eklenebilecek son slaytlar:

- Ranking metrics validation
- Hybrid reranking notu
- Demo ekran görüntüsü
- Proje sınırlılıkları ve gelecek çalışma

### 19.4 Demo smoke-test

Backend:

```powershell
python demo\backend\app.py
```

Frontend:

```powershell
cd demo\frontend
npm run dev -- --host 127.0.0.1
```

Kontrol:

- Catalog ürünleri geliyor mu?
- Görseller yükleniyor mu?
- Ürün seçince history oluşuyor mu?
- Tabular-only önerileri geliyor mu?
- Image-history önerileri geliyor mu?
- Late-fusion önerileri geliyor mu?
- Eksik görselde placeholder çalışıyor mu?

### 19.5 Opsiyonel teknik iyileştirme

Hybrid reranking iyi çıktı. İleriki teknik iyileştirme:

Training feature olarak şu kolonları modellere eklemek:

- `co_purchase_score`
- `recent_popularity_score`
- `last_7d_popularity_rank`
- `last_30d_popularity_rank`
- `metadata_match_count`
- `visual_neighbor_rank`

Bu yapılırsa late fusion modeli bu sinyalleri sonradan heuristic olarak değil, eğitim sırasında öğrenebilir.

Bu teslim için şart değil; ek çalışma olarak sunulabilir.

## 20. Bilinen Riskler ve Dikkat Noktaları

### 20.1 Kaggle skorları düşük görünebilir

Bu normal. Çünkü:

- Kaggle MAP@12 candidate generation'a çok duyarlı.
- Model AUC'si yüksek olsa bile doğru ürünü aday havuzuna sokmak ayrı problemdir.
- Yarışma çözümleri genellikle güçlü popularity, co-visitation, week-aware retrieval ve ensemble kullanır.

### 20.2 Full-training ve ablation skorları doğrudan kıyaslanmamalı

Controlled ablation küçük/temiz veri evrenindedir. Full-training daha büyük ve gürültülüdür.

### 20.3 SHAP yok

Permutation importance var. SHAP yok. Bu raporda açıkça "ileri çalışma" olarak söylenmeli.

### 20.4 Grad-CAM yok

EfficientNet-B0 feature extractor olarak kullanıldığı için Grad-CAM yerine embedding-space visual similarity kullanıldı.

### 20.5 Demo frontend son halinin doğrulanması gerekebilir

Kullanıcı arayüzü sonradan değiştirildi. Son görsel kalite ve API bağlantısı tekrar kontrol edilmeli.

### 20.6 `node_modules` ve büyük veri klasörleri

Proje taşınırken dikkat:

- `data/images/hm_images/` çok büyük.
- `data/raw/transactions_train.csv` çok büyük.
- `demo/frontend/node_modules/` taşınmak zorunda değil; `npm install` ile tekrar kurulabilir.
- `reports/presentation_build/node_modules/` de taşınmak zorunda değil.

## 21. Yeni Codex Oturumunda İlk Yapılacaklar

Yeni bir Codex hesabı/oturumu açılırsa:

1. Bu dosyayı okut.
2. `README.md` ve `reports/FINAL_REPORT.md` okut.
3. `reports/kaggle_full_training_results.csv` ve `reports/ranking_metrics_validation_summary.md` okut.
4. Gerekirse demo backend dosyası `demo/backend/recommender.py` okut.

Başlangıç komutu:

```text
PROJECT_HANDOFF.md dosyasını oku, sonra bana projenin mevcut durumunu ve sıradaki en mantıklı 5 adımı söyle.
```

## 22. En Kısa Yönetici Özeti

Bu proje H&M moda öneri sistemi için üç model karşılaştırdı:

1. Tabular-only MLP
2. Image-history MLP
3. Multimodal late fusion

Ürün görselleri EfficientNet-B0 ile 1280 boyutlu embeddinglere çevrildi. Image-history modeli müşterinin geçmiş satın alımlarından görsel profil oluşturdu. Late fusion modeli tabular ve görsel branch'leri birleştirdi.

Final Kaggle full-training sonuçlarında late fusion en yüksek performansı verdi:

- Tabular-only: AUC 0.8550, Accuracy 0.7636
- Image-history: AUC 0.9237, Accuracy 0.8297
- Late fusion: AUC 0.9354, Accuracy 0.8537

Bu sonuç multimodal hipotezi destekliyor. Ek olarak ranking evaluation yapıldı; co-purchase ve trend sinyalleriyle hybrid reranking MAP@12'yi artırdı fakat bu yeni ana model değil, post-ranking deneyi olarak raporlanmalı.

Projede final rapor, sunum, explainability çıktıları ve demo arayüzü hazır. Kalan ana iş final rapor/sunum dilini son metriklerle güncellemek ve demo smoke-test yapmak.
