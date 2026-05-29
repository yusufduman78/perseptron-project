# Perseptron Karar Günlüğü

Bu dosya, projede alınan önemli teknik ve metodolojik kararları nedenleriyle birlikte açıklar. Amaç, yeni bir Codex hesabının veya projeyi değerlendiren bir kişinin "neden böyle yapılmış?" sorularına hızlı cevap vermektir.

## Karar 1: EfficientNet-B0 Feature Extractor Olarak Kullanıldı

Karar:

Ürün görselleri için EfficientNet-B0 pretrained modelinden 1280 boyutlu embedding çıkarıldı.

Neden:

- H&M veri setinde 105 binden fazla ürün görseli var; her eğitim epoch'unda görüntüleri tekrar CNN'den geçirmek çok maliyetli olurdu.
- Pretrained EfficientNet-B0 moda ürünlerinin renk, form, doku ve kategori sinyallerini temsil edebilecek güçlü bir genel görsel feature extractor sağlar.
- Embedding cache kullanmak Kaggle ve local ortamda eğitim süresini yönetilebilir hale getirdi.
- Bu yaklaşım image-history ve late fusion modellerine ortak ve sabit bir görsel temsil evreni sağladı.

Sonuç:

- `data/embeddings/final_kaggle/article_image_embeddings_popular.npy` üretildi.
- 105,100 ürün için 1280 boyutlu embedding kullanıldı.
- Final modeller aynı embedding evrenine göre eğitildi.

## Karar 2: Image-history Model CNN Değil, Embedding Tabanlı MLP Oldu

Karar:

Image-history baseline modeli görüntü piksellerini doğrudan CNN ile eğitmedi; EfficientNet-B0 embeddingleri ve müşteri görsel geçmiş profili üzerinden MLP eğitti.

Neden:

- Proje hipotezi, tek ürün görselinden kategori sınıflandırmak değil, müşterinin geçmişte aldığı ürünlere göre aday ürünü skorlamaktı.
- Müşteri görsel tercihi, geçmiş ürün embeddinglerinin ortalaması/toplamı üzerinden temsil edilebilir.
- CNN fine-tuning, veri ve GPU maliyetini artırırdı; öneri problemi için asıl ihtiyaç müşteri-ürün ilişkisini öğrenmekti.
- Embedding tabanlı MLP, tabular-only ve late fusion modelleriyle daha adil ve karşılaştırılabilir bir yapı sundu.

Sonuç:

- Image-history MLP, final full-training'de AUC 0.9237 ve accuracy 0.8297 aldı.
- Bu, görsel geçmiş sinyalinin tabular metaveriden güçlü olduğunu gösterdi.

## Karar 3: Ana Model Multimodal Late Fusion Olarak Tasarlandı

Karar:

Tabular branch ve visual branch ayrı temsiller üretti; bu temsiller fusion head içinde birleştirildi.

Neden:

- Tabular ve görsel veriler farklı yapıda modalitelerdir.
- Tabular metadata kategorik ve sayısal özelliklerden oluşur.
- Görsel taraf aday embeddingi, müşteri profil embeddingi ve cosine similarity gibi vektörel özelliklerden oluşur.
- Ayrı branch'ler her modalitenin kendi temsilini öğrenmesini sağlar.
- Son aşamada birleştirme, multimodal hipotezi doğrudan test eder.

Sonuç:

- Multimodal late fusion final full-training'de en yüksek performansı verdi:
  - AUC 0.9341
  - Accuracy 0.8504
  - Düzeltme sonrası Kaggle rerun 30 Mayıs 2026'da tamamlandı; validation pozitifleri train'den ayrık tutuldu.

## Karar 4: Full-training Kaggle Üzerinde Yapıldı

Karar:

Final eğitimler local yerine Kaggle üzerinde çalıştırıldı.

Neden:

- `transactions_train.csv` 31 milyondan fazla işlem içerir.
- Full pozitif müşteri-ürün çiftleri yaklaşık 27.2 milyon seviyesindedir.
- Local ortamda bellek ve runtime yönetimi zordu.
- Kaggle GPU/T4 ortamı, embedding cache ve streaming/chunk eğitim için daha uygundu.

Sonuç:

- Üç final notebook Kaggle üzerinde çalıştırıldı:
  - `tabular_only_fulltraining.ipynb`
  - `image_history_fulltraining_perseptron.ipynb`
  - `perseptron_late_fusion.ipynb`
- Final checkpointler `models/final_kaggle/` altına alındı.

## Karar 5: Streaming/Chunk Training Kullanıldı

Karar:

Büyük eğitim verisi tek seferde belleğe alınmadı; chunk/streaming mantığıyla işlendi.

Neden:

- Full pozitif çiftler ve negatif örnekler belleği kolayca aşabiliyordu.
- Kaggle working alanı ve RAM sınırları vardı.
- Chunk eğitim, tüm veri kapsamını korurken bellek kullanımını kontrol altında tuttu.

Sonuç:

- Full-training 3 epoch çalıştırılabildi.
- Her model aynı full pozitif çift evreninde karşılaştırıldı.

## Karar 6: Controlled Ablation Ayrı Yorumlandı

Karar:

Controlled ablation sonuçları ana sonuç tablosuyla aynı ölçekte yorumlanmadı.

Neden:

- Ablation daha küçük ve kontrollü bir veri evreninde çalıştı.
- Validation dağılımı full-training'den farklıydı.
- Negatif örnekleme etkisi daha belirgin olabilir.
- Full-training daha gerçekçi, daha büyük ve daha gürültülüdür.

Sonuç:

- Ablation destekleyici deney olarak raporlanmalı.
- Ana bilimsel iddia için `reports/kaggle_full_training_results.csv` esas alınmalı.

## Karar 7: SHAP Yerine Permutation Importance Kullanıldı

Karar:

Tabular açıklanabilirlik için SHAP yerine permutation importance uygulandı.

Neden:

- PyTorch modelinde categorical embedding layer'lar var.
- Preprocessing, categorical mapping, numeric normalization ve embedding layer zinciri SHAP'i doğrudan uygulamayı zorlaştırdı.
- Permutation importance, mevcut tabular modelin feature etkisini daha hızlı ve raporlanabilir biçimde verdi.
- Ders projesi kapsamında uygulanabilirlik ve güvenilir yorum önceliklendirildi.

Sonuç:

- `reports/tabular_permutation_importance.csv`
- `reports/tabular_feature_group_importance.csv`
- `reports/TABULAR_EXPLAINABILITY.md`

Ana bulgu:

- Ürün kategorik özellikleri en güçlü tabular feature grubu çıktı.
- `section_name`, `department_name`, `product_type_name`, `age`, `product_code` öne çıktı.

## Karar 8: Grad-CAM Yerine Visual Similarity Açıklaması Kullanıldı

Karar:

Görsel açıklanabilirlik için Grad-CAM yerine embedding uzayında en benzer geçmiş ürünleri gösteren visual similarity analizi yapıldı.

Neden:

- EfficientNet-B0 nihai öneri modelinin karar katmanı olarak fine-tune edilmedi.
- EfficientNet-B0 bu projede sabit feature extractor rolündedir.
- Grad-CAM, CNN'in görüntü içinde hangi bölgeye baktığını gösterebilir; fakat bu projede nihai skor müşteri geçmiş profili + aday embedding ilişkisine dayanır.
- Bu nedenle aday ürünün müşterinin geçmişindeki hangi ürünlere görsel olarak benzediğini göstermek model mantığıyla daha uyumludur.

Sonuç:

- `reports/visual_similarity_examples.csv`
- `reports/visual_similarity_examples.html`
- `reports/VISUAL_EXPLAINABILITY.md`

## Karar 9: Kaggle Public Score Ana Başarı Metriği Yapılmadı

Karar:

Kaggle submission skorları yardımcı dış test olarak görüldü; ana başarı metriği full-training validation AUC/accuracy ve model karşılaştırması oldu.

Neden:

- Kaggle MAP@12 skoru büyük ölçüde candidate generation ve retrieval stratejilerine bağlıdır.
- Pairwise satın alma tahmini yapan modelin AUC'si yüksek olsa bile doğru ürünü aday havuzuna sokmak ayrı bir problemdir.
- Yarışma çözümleri genellikle week-aware popularity, co-visitation, co-purchase, ensemble ve güçlü retrieval katmanları kullanır.
- Bu projenin amacı yarışma optimizasyonundan çok multimodal hipotezi test etmekti.

Sonuç:

- Kaggle submission skorları raporda dikkatli konumlandırılmalı.
- "Düşük Kaggle score = model başarısız" çıkarımı yapılmamalı.

## Karar 10: Ranking Metrics Sonradan Ayrı Değerlendirme Olarak Eklendi

Karar:

MAP@12 / Precision@10 / Recall@10 için time-based validation ranking evaluation scripti eklendi.

Neden:

- Proposal'da ranking metriklerinden bahsedilmişti.
- AUC/accuracy pairwise sınıflandırma başarısını ölçer; öneri sisteminde top-k sıralama kalitesi de ayrıca görülmelidir.
- Kaggle leaderboard ile birebir aynı olmasa da validation ranking evaluation model davranışını daha iyi açıklar.

Sonuç:

- `src/evaluation/ranking_metrics_validation.py`
- `reports/ranking_metrics_validation.csv`
- `reports/ranking_metrics_validation_summary.md`

Önemli bulgu:

- CandidateRecall 0.475379 seviyesine çıktı.
- HitRate@12 düşük kaldı; top-k sıralama hâlâ zor.

## Karar 11: `late_fusion_hybrid` Ana Model Değil, Reranking Deneyi Olarak Konumlandırıldı

Karar:

`late_fusion_hybrid` raporda ana dördüncü model olarak değil, post-ranking/reranking stratejisi olarak anlatılmalı.

Neden:

- Bu yöntem yeni bir checkpoint veya eğitimli model değildir.
- Normal late fusion skoruna heuristic sinyaller ekler:
  - recency/popularity
  - co-purchase
  - metadata match
  - visual similarity
- MAP@12'yi artırması, recommender sistemlerinde retrieval/reranking katmanının önemini gösterir.
- Ana bilimsel hipotez üç eğitimli modelin karşılaştırmasıdır.

Sonuç:

- Ana karşılaştırma:
  1. tabular-only
  2. image-history
  3. multimodal late fusion
- Hybrid ayrı başlıkta "post-ranking iyileştirme deneyi" olarak sunulmalı.

## Karar 12: Demo Backend Üç Modeli Aynı Arayüzde Sunacak Şekilde Tasarlandı

Karar:

Demo, kullanıcının geçmiş ürün seçip üç modelin önerilerini yan yana görebileceği şekilde tasarlandı.

Neden:

- Projenin farkı modellerin karşılaştırılmasıdır.
- İnsan okuyucu veya hoca, tabular-only, image-history ve late-fusion önerilerini tek arayüzde kıyaslayabilmelidir.
- Görsellerle desteklenen alışveriş sitesi benzeri demo, modelin kullanım senaryosunu somutlaştırır.

Sonuç:

- Backend: `demo/backend/app.py`
- Recommender: `demo/backend/recommender.py`
- Frontend: `demo/frontend/`

## Karar 13: Yeni Hesap İçin Katmanlı Dokümantasyon Seti Kuruldu

Karar:

Tek bir uzun dosya yerine `START_HERE.md`, `PROJECT_HANDOFF.md`, `docs/DOC_MAP.md`, `docs/NEXT_STEPS.md` ve `docs/DECISION_LOG.md` birlikte kullanılacak.

Neden:

- Yeni Codex hesabı önce hızlı bağlam, sonra detay, sonra görev listesi okumalı.
- İnsan okuyucu da dosya haritası ve karar gerekçeleriyle projeyi kolay anlamalı.
- Uzun handoff dosyası canonical hafıza olarak kalırken, kısa başlangıç dosyaları gezinmeyi kolaylaştırır.

Sonuç:

- `START_HERE.md`: hızlı başlangıç
- `PROJECT_HANDOFF.md`: kapsamlı hafıza
- `docs/DOC_MAP.md`: dosya haritası
- `docs/NEXT_STEPS.md`: sıradaki işler
- `docs/DECISION_LOG.md`: karar gerekçeleri
