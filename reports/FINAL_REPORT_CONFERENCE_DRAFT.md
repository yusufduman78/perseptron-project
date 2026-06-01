# Perseptron: H&M Veri Seti Üzerinde Multimodal Moda Öneri Sistemi

**Yusuf Duman**
Bilgisayar Mühendisliği Final Projesi
Teslim sürümü: 2026-06-01

## Abstract

Bu çalışma, H&M Personalized Fashion Recommendations veri seti üzerinde moda ürünleri için multimodal bir öneri sistemi geliştirmeyi hedefler. Temel hipotez, müşteri ve ürün metaverisiyle birlikte ürün görsellerinden çıkarılan temsil vektörlerinin satın alma olasılığı tahminini ve öneri sıralamasını iyileştireceğidir. Bu hipotezi test etmek için üç ana classification modeli karşılaştırılmıştır: yalnızca tabular metaveri kullanan MLP, EfficientNet-B0 tabanlı image-only CNN baseline ve tabular/görsel sinyalleri birleştiren late fusion MLP. Recommendation demo ve ranking değerlendirmesinde ise üç pratik öneri modeli kullanılmıştır: `tabular_only`, `image_history` ve `late_fusion`. `image_history`, CNN yerine geçmez; müşterinin geçmiş alışverişlerinden oluşturulan görsel stil profilini kullanan bir ranking baseline’ıdır. Final Kaggle full-run sonuçlarında `late_fusion` modeli final AUC 0.8063 ve en iyi epoch AUC 0.8121 ile ana classification karşılaştırmasında en yüksek performansa ulaşmıştır. Ranking tarafında da `late_fusion`, MAP@12 0.001262 ile en iyi ana model olmuştur. Ek olarak SHAP feature summary ve Grad-CAM görselleriyle model kararları açıklanabilir hale getirilmiştir.

## 1. Introduction

Moda öneri sistemleri klasik ürün öneri problemlerinden daha zordur; çünkü satın alma davranışı yalnızca kategori, fiyat veya yaş gibi metaveriyle açıklanamaz. Ürünün rengi, silueti, kumaş algısı, fotoğraf kompozisyonu ve müşterinin geçmişte tercih ettiği görsel stil de kararı etkiler. H&M Personalized Fashion Recommendations yarışması bu problemi büyük ölçekli bir perakende veri setiyle sunar: müşteri profilleri, ürün metaverisi, işlem geçmişi ve ürün görselleri aynı problem içinde yer alır.

Projenin ilk araştırma sorusu şudur: Ürün görsellerinden çıkarılan temsil vektörleri, yalnızca tabular metaveriye dayalı bir satın alma tahmin modeline anlamlı katkı sağlar mı? İkinci soru ise daha uygulamalıdır: Bu katkı, kullanıcıya gösterilebilecek bir demo öneri akışına dönüştürülebilir mi? Bu nedenle proje yalnızca tek bir metrik optimizasyonu olarak tasarlanmamış; eğitim pipeline’ı, ranking değerlendirmesi, açıklanabilirlik çıktıları ve çalışan backend/frontend demo birlikte ele alınmıştır.

Çalışmanın final sürümünde iki ayrı değerlendirme hattı korunmuştur. Birinci hat binary classification hattıdır: belirli müşteri-ürün çiftleri için satın alma olasılığı tahmin edilir ve AUC-ROC/accuracy ölçülür. İkinci hat ranking hattıdır: müşteri bazında aday ürünler sıralanır ve MAP@12, Precision@10, Recall@10 gibi öneri metrikleri hesaplanır. Bu ayrım önemlidir; çünkü classification modeli iyi bir skorlayıcı olsa bile, gerçek öneri sisteminde aday üretimi ve sıralama aşamaları ayrıca değerlendirilmelidir.

Projenin final hikayesi şu şekilde kilitlenmiştir. Ana model karşılaştırması `tabular_only`, `image_only_effnet_cnn` ve `late_fusion` üzerindedir. Demo ve ranking karşılaştırması ise `tabular_only`, `image_history` ve `late_fusion` üzerindedir. `late_fusion_hybrid_*` varyantları ana model değildir; co-purchase ve heuristic sinyallerle yapılan post-ranking/reranking deneyleri olarak raporlanmıştır.

## 2. Related Work

Öneri sistemleri literatüründe collaborative filtering, matrix factorization ve sequence-aware yöntemler uzun süredir temel yaklaşımlar arasındadır. Ancak moda alanında ürün görselleri davranış sinyali kadar kritik olabilir. Görsel öneri sistemlerinde CNN tabanlı temsil vektörleri ürün benzerliği, stil tamamlama ve cold-start senaryolarında sık kullanılır. ImageNet üzerinde önceden eğitilmiş EfficientNet gibi modern CNN mimarileri, sınırlı eğitim bütçesinde güçlü görsel özellik çıkarıcıları sağlar.

Bu proje iki araştırma çizgisinden etkilenmiştir. İlk çizgi, tabular ve categorical özelliklerin embedding katmanlarıyla MLP içinde modellenmesidir. Müşteri durumu, yaş, ürün tipi, renk grubu, departman ve garment group gibi alanlar, satın alma olasılığını açıklayan güçlü sinyaller taşır. İkinci çizgi ise görsel embeddinglerin müşteri geçmişiyle birleştirilmesidir. Bir müşterinin geçmiş ürün görsellerinin ortalaması, basit ama yorumlanabilir bir görsel stil profili üretir. Aday ürün embeddingi ile bu profil arasındaki benzerlik, tabular modele eklenebilecek doğal bir multimodal sinyaldir.

Önceki çalışmalarda multimodal öneri sistemleri genellikle metin, görüntü ve davranış verisini birlikte kullanır. Bu projede metin açıklaması yerine H&M ürün metaverisi ve ürün görselleri birleştirilmiştir. Amaç, çok karmaşık bir end-to-end sistem kurmak değil, teslim süresi içinde güvenilir şekilde eğitilebilen ve açıklanabilen bir multimodal late fusion mimarisi kurmaktır.

## 3. Methods

### 3.1 Veri Seti ve Ön İşleme

Kullanılan veri seti H&M Personalized Fashion Recommendations veri setidir. Temel tablolar `customers.csv`, `articles.csv` ve `transactions_train.csv` dosyalarıdır. Ürün görselleri H&M katalog fotoğraflarından oluşur. `articles.csv` içindeki ürün tipi, ürün grubu, renk grubu, departman, index, section ve garment group alanları article metadata olarak kullanılmıştır. `customers.csv` içindeki `FN`, `Active`, `age`, `club_member_status` ve `fashion_news_frequency` alanları müşteri metadata sinyali olarak alınmıştır.

Tabular özellikler iki grupta işlenmiştir. Sayısal alanlar eğitim splitinden hesaplanan ortalama ve standart sapma ile normalize edilmiştir. Kategorik alanlar ise training verisinde öğrenilen kategori haritalarıyla integer id’ye çevrilmiş ve model içinde embedding katmanlarına verilmiştir. Bilinmeyen kategoriler `__UNK__` id’sine düşürülmüştür. Bu yaklaşım, inference sırasında demo backend’in eğitimdeki encoding mantığıyla uyumlu kalmasını sağlamıştır.

Görsel tarafta iki farklı kullanım vardır. Image-only baseline için EfficientNet-B0 doğrudan ürün görselleri üzerinde binary sınıflandırıcı olarak fine-tune edilmiştir. Multimodal/ranking hattında ise ürün görsellerinden 1280 boyutlu EfficientNet embeddingleri kullanılmıştır. Her müşteri için geçmiş satın alma ürünlerinin embedding ortalaması görsel profil olarak hesaplanmış, aday ürün embeddingi ile bu profil arasındaki cosine benzerliği `visual_similarity` özelliği olarak modele verilmiştir. `visual_history_count` ise profilin kaç geçmiş üründen oluştuğunu gösteren ek bir sinyaldir.

Ön işleme aşamasında veri sızıntısını azaltmak için validation penceresinden sonraki alışverişler müşteri geçmişi oluşturulurken kullanılmamıştır. Bu karar özellikle görsel profil için önemlidir; çünkü validation döneminde satın alınan bir ürünün embeddingi müşteri profiline girerse ranking değerlendirmesi gerçekte olduğundan daha iyi görünebilir. Bu nedenle müşteri görsel profili yalnızca cutoff öncesi işlemlerden oluşturulmuştur. Aynı mantık, co-purchase ve aday üretimi için kullanılan yardımcı sinyallerde de korunmuştur.

Negatif örnekleme binary classification hattının kritik parçalarından biridir. Veri setinde pozitif örnekler gerçek satın alma çiftlerinden gelir; fakat satın alınmamış tüm müşteri-ürün kombinasyonlarını negatif kabul etmek pratik değildir. Bu nedenle her pozitif çift için katalogdan satın alınmamış adaylar örneklenmiş ve modellerin “satın alınır/satın alınmaz” ayrımını öğrenmesi sağlanmıştır. Bu tasarım AUC-ROC metriği için dengeli ve karşılaştırılabilir bir değerlendirme sağlar; ancak gerçek dünyadaki ürün sıralama kalitesini tek başına garanti etmediği için ayrıca ranking hattı kurulmuştur.

### 3.2 Model 1: Tabular-only MLP

`tabular_only` modeli müşteri ve ürün metaverisini kullanan baseline’dır. Sayısal özellikler normalize edilerek, kategorik özellikler ise embedding katmanlarından geçirilerek MLP’ye verilir. Bu model görsel bilgi içermez; bu nedenle multimodal katkıyı ölçmek için ana referans noktasıdır. Model, training sırasında binary cross entropy loss ile eğitilmiş ve AUC-ROC/accuracy ile değerlendirilmiştir.

Tabular-only baseline, görsel veri kullanılmadan perakende metadata’sının ne kadar bilgi taşıdığını gösterir. Final sonuçlarda bu model güçlü bir baseline olarak kalmış, ancak late fusion modelinin altında performans vermiştir. Bu durum, görsel sinyalin yalnız başına mucizevi olmadığını fakat doğru şekilde birleştirildiğinde tabular sinyale katkı sağlayabildiğini göstermektedir.

### 3.3 Model 2: Image-only EfficientNet CNN

`image_only_effnet_cnn` modeli ürün görsellerini doğrudan kullanan image-only baseline’dır. EfficientNet-B0 ImageNet ağırlıklarıyla başlatılmış, son sınıflandırıcı katman binary satın alma tahmini için değiştirilmiştir. Bu modelin amacı demo öneri listesi üretmek değil, proposal’daki görsel model hedefini karşılamak ve görüntünün tek başına ne kadar açıklayıcı olduğunu ölçmektir.

CNN modeli aynı zamanda Grad-CAM açıklanabilirlik çıktılarının kaynağıdır. Grad-CAM görselleri, CNN’in ürün fotoğrafında hangi bölgelere odaklandığını görselleştirmek için üretilmiştir. Final demoda CNN ayrı bir recommendation tab’i olarak gösterilmez; çünkü müşteri geçmişi, aday üretimi ve metadata bağlamı olmadan doğrudan ürün önerisi yapmak için uygun ana demo modeli değildir.

### 3.4 Model 3: Multimodal Late Fusion

`late_fusion` projenin ana multimodal modelidir. Bu model iki ayrı dal kullanır. Tabular dal, `tabular_only` modelindeki numeric ve categorical özellikleri işler. Görsel dal ise aday ürün embeddingi, müşteri görsel profil embeddingi, `visual_similarity` ve `visual_history_count` değerlerini işler. İki dalın latent temsilleri birleştirilir ve fusion head üzerinden satın alma olasılığı skoru üretilir.

Bu mimarinin tercih edilme nedeni pratik ve metodolojiktir. End-to-end CNN + tabular eğitim, Kaggle runtime ve GPU kotası açısından risklidir. Buna karşılık late fusion yaklaşımı, görsel bilgiyi önceden çıkarılmış embeddingler üzerinden kullanır; hem daha hızlı eğitilir hem de tabular modele göre ek katkısı daha net ölçülür. Final sonuçlarda late fusion modeli hem classification hem ranking tarafında ana modeller arasında en iyi performansı vermiştir.

Late fusion mimarisinde görsel dal yalnızca ürün embeddingini değil, müşteri geçmişinden gelen profil embeddingini de kullanır. Bu ayrım önemlidir: ürünün görsel temsili tek başına ürünün nasıl göründüğünü anlatır; müşteri profili ise bu görünümün kullanıcının geçmiş tercihleriyle ne kadar uyumlu olduğunu temsil eder. `visual_similarity` ve `visual_history_count` bu ilişkiyi sayısal olarak güçlendirir. Böylece model, örneğin aynı ürün kategorisindeki iki adaydan müşterinin geçmiş renk/siluet tercihine daha yakın olanı daha üst sıraya taşıyabilir.

Modelin tabular dalı ve görsel dalı ayrı tutulduğu için sonuçlar yorumlanabilir kalır. Eğer tüm özellikler tek bir büyük vektörde karıştırılsaydı, görsel sinyalin ne kadar katkı verdiğini ayırmak daha zor olurdu. Branch yapısı, raporda “metadata sinyali” ve “görsel geçmiş sinyali” ayrımını açık anlatmayı kolaylaştırmıştır. Bu tasarım, proje hedefinin yalnızca en yüksek skor değil, multimodal hipotezi ölçmek olduğunu da destekler.

### 3.5 Image-history Baseline ve Hybrid Reranking Ayrımı

`image_history`, CNN modeli değildir. Bu model, müşterinin geçmiş alışveriş görsellerinden oluşan stil profilini ve aday ürün embeddingini kullanır. Demo ve ranking hattında görsel geçmiş sinyalinin tek başına ne yaptığını görmek için kullanılmıştır. Classification tablosunda AUC değeri raporlanmış olsa da ana proposal karşılaştırmasındaki image-only baseline, EfficientNet CNN’dir.

`late_fusion_hybrid_*` varyantları ise ana model değildir. Bu varyantlar, late fusion skorunu co-purchase ve heuristic sinyallerle yeniden sıralayan post-ranking deneyleridir. Hybrid sonuçları raporda ayrı tutulmuştur; ana model karşılaştırmasına dahil edilmemiştir. Bu ayrım, projenin model iddiasını daha temiz ve savunulabilir hale getirir.

## 4. Experimental Design

Deneyler Kaggle ortamında çalıştırılmıştır. Notebook akışı birbirinden bağımsız adımlara ayrılmıştır: fold/split hazırlığı, tabular model eğitimi, image-history modeli, late fusion modeli, EfficientNet CNN baseline, ranking değerlendirmesi ve explainability çıktıları. Böylece her notebook kendi kodunu ve açıklamasını taşır; repo içindeki `src` dosyalarına bağımlı olmayan, sunumda gösterilebilir Kaggle çalışmaları elde edilmiştir.

Validation protokolü zaman temelli olarak kurulmuştur. Gelecekteki alışveriş davranışını taklit etmek için belirli bir cutoff sonrası işlemler validation tarafında tutulmuştur. Classification hattında müşteri-ürün çiftleri pozitif ve negatif örneklerle eğitilmiş; ranking hattında ise müşteri başına aday havuzu üretilerek top-k sıralama değerlendirilmiştir. Full 5-fold cross-validation proposal’da hedeflenmiş olsa da, final teslim süresi ve GPU/runtime kısıtları nedeniyle full 5-fold yerine kontrollü full-run validation sonuçları raporlanmıştır. Bu karar metodolojik bir sadeleştirmedir; model karşılaştırması aynı split ve aynı pipeline üzerinde tutulduğu için sonuçlar karşılaştırılabilir kalmıştır.

Ranking deneyinde aday havuzu popular ürünler, metadata benzerleri, görsel komşular ve co-purchase sinyalleriyle oluşturulmuştur. Ana modeller aynı aday havuzunu skorlamıştır. Değerlendirme metrikleri MAP@12, Precision@10, Recall@10, HitRate@12 ve candidate recall olarak hesaplanmıştır. Kaggle yarışmasının leaderboard metriği MAP@12 olduğu için final ranking yorumu bu metrik üzerinden yapılmıştır.

Aday üretimi ranking hattının performansını doğrudan etkiler. Eğer validation’da satın alınan ürün aday havuzuna hiç girmiyorsa, skorlayıcı model doğru ürünü üst sıraya taşıma şansı bulamaz. Bu nedenle ranking raporunda `candidate_recall` ayrıca takip edilmiştir. Final deneyde candidate recall tüm modeller için aynı kalmıştır; çünkü modeller aynı aday havuzunu skorlamıştır. Bu sayede MAP@12 farkı aday havuzundan değil, model skorlamasından kaynaklanan göreli fark olarak yorumlanabilir.

Hybrid reranking deneyleri bu noktada devreye girmiştir. Bu deneylerde late fusion skorunun üzerine co-purchase ve heuristic sinyaller eklenerek farklı ağırlıklar denenmiştir. Ancak final sonuçlarda hybrid varyantlar ana late fusion modelini geçmemiştir. Bu sonuç, daha fazla sinyal eklemenin her zaman daha iyi ranking anlamına gelmediğini gösterir. Özellikle heuristic sinyaller, kısa validation penceresinde bazı müşteriler için faydalı olabilirken genel ortalamada model skorunu bozabilir. Bu nedenle hybrid sonuçlar final hikayede “denendi fakat ana model olarak seçilmedi” şeklinde konumlandırılmıştır.

Explainability hattında iki yöntem kullanılmıştır. Tabular tarafında SHAP summary üretilmiş ve en etkili müşteri/ürün özellikleri raporlanmıştır. Görsel tarafta image-only EfficientNet CNN modeli için Grad-CAM örnekleri çıkarılmıştır. Demo arayüzünde ayrıca ürün önerilerine metadata uyumu, renk uyumu, görsel geçmiş benzerliği ve en yakın geçmiş ürün gibi hafif açıklama etiketleri eklenmiştir.

## 5. Experimental Results

### 5.1 Classification Sonuçları

Final full-run classification sonuçları aşağıdaki gibidir. Tablodaki final AUC değeri tamamlanan son epoch sonucunu, best AUC ise eğitim sırasında elde edilen en iyi validation AUC değerini gösterir.

| Model | Rol | Train rows | Validation rows | Final AUC | Best AUC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| `tabular_only` | Metadata baseline | 400,000 | 85,381 | 0.7841 | 0.7985 | 0.7150 |
| `image_only_effnet_cnn` | Image-only baseline | 39,843 | 9,965 | 0.7513 | 0.7602 | 0.6864 |
| `image_history` | Visual-history baseline | 399,191 | 79,501 | 0.7480 | 0.7480 | 0.6593 |
| `late_fusion` | Multimodal main model | 399,191 | 79,501 | 0.8063 | 0.8121 | 0.7207 |

Bu tablo iki önemli sonucu gösterir. Birincisi, tabular metadata güçlü bir sinyal taşır; `tabular_only` modeli AUC 0.7841 ile sağlam bir baseline’dır. İkincisi, görsel bilgi tek başına tabular modelin üstüne çıkmamıştır; `image_only_effnet_cnn` AUC 0.7513, `image_history` ise AUC 0.7480 elde etmiştir. Buna karşılık late fusion, tabular ve görsel geçmiş sinyallerini birlikte kullandığında final AUC 0.8063 ve best AUC 0.8121 ile en iyi classification sonucunu üretmiştir. Bu sonuç, projenin ana hipotezini destekler: görsel sinyal en güçlü etkisini tabular bağlamla birlikte kullanıldığında verir.

### 5.2 Ranking Sonuçları

Ranking değerlendirmesinde ana modeller ve hybrid reranking varyantları aynı tabloda raporlanmış, ancak yorum ikiye ayrılmıştır. Ana model karşılaştırması `tabular_only`, `image_history` ve `late_fusion` üzerindedir. Hybrid varyantlar yalnızca post-ranking deneyidir.

| Model | MAP@12 | Precision@10 | Recall@10 | HitRate@12 | Candidate Recall |
|---|---:|---:|---:|---:|---:|
| `late_fusion` | 0.001262 | 0.000889 | 0.003256 | 0.009906 | 0.241120 |
| `late_fusion_hybrid_w025` | 0.000961 | 0.000796 | 0.002871 | 0.009516 | 0.241120 |
| `late_fusion_hybrid_w045` | 0.000910 | 0.000733 | 0.002825 | 0.008424 | 0.241120 |
| `late_fusion_hybrid_w065` | 0.000714 | 0.000741 | 0.002418 | 0.008190 | 0.241120 |
| `tabular_only` | 0.000713 | 0.000858 | 0.002702 | 0.008658 | 0.241120 |
| `image_history` | 0.000653 | 0.000764 | 0.002404 | 0.008892 | 0.241120 |

Ranking metrikleri mutlak değer olarak düşük görünmektedir; bunun temel nedeni problem ölçeği ve aday üretim zorluğudur. H&M veri setinde çok geniş ürün kataloğu, seyrek müşteri davranışı ve kısa validation penceresi top-k isabet oranlarını düşürür. Bu nedenle yorum, mutlak MAP@12 büyüklüğünden çok aynı aday havuzu ve aynı split üzerindeki göreli model farkına dayanır. Bu bağlamda `late_fusion`, ana modeller içinde MAP@12 lideridir. Hybrid reranking varyantlarının late fusion ana modelini geçememesi de önemli bir sonuçtur: heuristic/co-purchase sinyali eklemek her zaman daha iyi sıralama üretmemiştir.

### 5.3 Explainability Sonuçları

Tabular explainability için üretilen SHAP summary’de en yüksek etkili özellikler arasında `age`, `section_no`, `product_group_name`, `colour_group_code`, `FN`, `garment_group_name` ve `index_group_name` yer almıştır. Bu sonuç beklenen alan bilgisiyle uyumludur: müşterinin yaşı ve fashion-news durumu gibi müşteri sinyalleri ile ürünün kategori/renk/section bilgileri satın alma tahmininde etkilidir.

Görsel explainability için EfficientNet CNN baseline üzerinde Grad-CAM örnekleri üretilmiştir. Grad-CAM çıktıları, modelin ürün fotoğrafında kıyafet gövdesi, renk blokları ve siluet bölgelerine odaklandığını göstermektedir. Bu, CNN’in arka plan yerine ürünün kendisinden sinyal almaya çalıştığına dair nitel bir kontrol sağlar. Demo arayüzündeki açıklama chip’leri ise production-grade explainability iddiası taşımaz; sunum sırasında öneri sonucunu daha anlaşılır kılmak için metadata/görsel komşuluk temelli hafif açıklamalardır.

Explainability çıktıları raporun model iddiasını desteklemek için kullanılmıştır. SHAP summary, tabular tarafta modelin mantıklı müşteri ve ürün alanlarına duyarlı olduğunu gösterir. Grad-CAM ise image-only CNN’in görsel olarak ürünün kendisine odaklanıp odaklanmadığını kontrol eder. Bu iki çıktı farklı modellerden geldiği için sunumda dikkatli ayrılmalıdır: SHAP tabular/metadata yorumudur, Grad-CAM CNN yorumudur, demo chip’leri ise recommendation inference sırasında üretilen kullanıcı-dostu açıklamalardır. Bu ayrım yanlış bir “Grad-CAM late fusion kararını doğrudan açıklıyor” iddiasını engeller.

## 6. Demo Application

Final demo iki parçadan oluşur: FastAPI backend ve React frontend. Backend V2 checkpoint formatını (`state_dict`) okur, `.kaggle_outputs/full/05-ranking` ve `.kaggle_outputs/full/06-explainability` artifactlerini kullanır, katalog ve öneri endpointleri sağlar. Frontend ise aydınlık fashion marketplace görünümünde çalışır; kullanıcı hazır senaryolardan birini seçebilir veya katalogdan geçmiş ürünler ekleyebilir. Ardından üç öneri modeli aynı aday havuzu üzerinde çalıştırılır: `tabular_only`, `image_history` ve `late_fusion`.

Backend endpointleri şunlardır: `/api/health`, `/api/catalog`, `/api/demo-scenarios`, `/api/recommend`, `/api/metrics` ve `/api/explainability`. `/api/recommend` yanıtında her modelin önerileri, model overlap bilgisi, unique öneriler, request context ve öneri bazında açıklama metadata’sı bulunur. `/api/metrics` final Kaggle metriklerini döndürür. `/api/explainability` ise SHAP top features ve Grad-CAM örneklerine erişim sağlar.

Backend tasarımında eğitim pipeline’ı ile demo inference ayrılmıştır. Demo uygulaması yeniden eğitim yapmaz; yalnızca final checkpointleri, metadata dosyalarını ve embeddingleri okur. Bu sayede sunum sırasında sistem daha hızlı ve daha güvenilir çalışır. Ayrıca `/api/health` endpointi checkpoint, embedding, article metadata, görsel klasör ve explainability artifactlerinin varlığını kontrol eder. Böylece demo başlamadan önce eksik dosya veya yanlış path problemi hızlıca görülebilir.

Frontend tarafında amaç teknik dashboard hissini azaltıp ürün odaklı bir fashion marketplace deneyimi üretmektir. Kullanıcı önce hazır bir senaryo seçebilir veya katalogdan ürün ekleyebilir. Seçilen ürünler `Style bag` alanında geçmiş alışveriş gibi tutulur. Öneri üretildiğinde model tab’leri arasında geçiş yapılır ve aynı müşteriye göre üç farklı modelin nasıl ayrıştığı görülebilir. Model insights paneli ortak ürünleri ve late fusion’a özel önerileri gösterir; bu bölüm sunum sırasında “multimodal model farklı ne yaptı?” sorusuna hızlı cevap verir.

Demo çalıştırma akışı:

```powershell
conda run -n perseptron python demo\backend\test_recommend.py --top-k 3 --candidate-limit 200
python demo\backend\app.py
cd demo\frontend
npm run dev -- --host 127.0.0.1
```

Sunum sırasında önerilen akış, önce `Black Basics` veya `Denim Casual` hazır senaryosunu seçmek, ardından önerileri üretip model tab’leri arasında gezmektir. Model insights paneli modellerin ortak ve ayrışan önerilerini gösterir. Final evidence alanı ise classification, ranking ve explainability sonuçlarını kısa biçimde demo içinde görünür kılar.

## 7. Limitations

Bu projede bazı proposal hedefleri final teslim koşullarına göre sadeleştirilmiştir. Full 5-fold cross-validation yerine tek final validation protokolü üzerinde full-run sonuçları raporlanmıştır. Bunun nedeni her fold için tabular, image-history, late fusion ve CNN eğitimlerinin toplamda çok yüksek Kaggle runtime ve GPU kotası gerektirmesidir. Bu sadeleştirme sonuçları geçersiz kılmaz; çünkü final karşılaştırmalar aynı split, aynı negative sampling mantığı ve aynı aday havuzu üzerinde yapılmıştır. Ancak daha uzun vadeli çalışmada full 5-fold CV daha güçlü istatistiksel güven sağlayacaktır.

İkinci sınırlılık image-only CNN tarafındadır. CNN modeli proposal’daki görsel baseline hedefini karşılar ve Grad-CAM üretir; fakat final recommendation demosunda ayrı model olarak kullanılmaz. Bunun nedeni CNN’in müşteri geçmişi ve metadata bağlamı olmadan doğrudan kişiselleştirilmiş öneri sıralaması için yeterli olmamasıdır. Demo için daha doğru görsel baseline, müşteri geçmişinden oluşturulan `image_history` modelidir.

Üçüncü sınırlılık ranking metriklerinin düşük mutlak değerleridir. Bu durum özellikle büyük katalog, seyrek müşteri davranışı, kısa validation penceresi ve aday havuzu recall limitinden kaynaklanır. Gelecekte candidate generation aşaması daha güçlü sequence/collaborative filtering yöntemleriyle geliştirilebilir. Bu çalışmanın ana katkısı, aday üretimini mükemmelleştirmekten çok aynı aday havuzu üzerinde multimodal skorlamanın etkisini göstermektir.

Dördüncü sınırlılık, final modellerin tek bir Kaggle runtime bütçesi içinde pratik olarak eğitilmiş olmasıdır. Daha uzun eğitim, daha büyük negative sampling, hiperparametre taraması ve farklı fusion mimarileri muhtemelen performansı değiştirebilir. Ancak teslim bağlamında önemli olan, üç ana modelin aynı çerçevede çalışması ve sonuçların açıkça raporlanmasıdır. Bu nedenle final sürümde yeni büyük deney açmak yerine mevcut V2 sonuçları kilitlenmiş ve rapor/demo bütünlüğüne odaklanılmıştır.

Gelecek çalışmada üç yönde iyileştirme yapılabilir. Birincisi, candidate generation aşaması sequence-aware veya collaborative filtering tabanlı bir modelle güçlendirilebilir. İkincisi, image-only CNN tarafında daha geniş veri ve daha uzun fine-tuning denenebilir. Üçüncüsü, late fusion yerine attention tabanlı veya gating mekanizmalı multimodal mimariler test edilebilir. Bu geliştirmeler mevcut projenin üzerine doğal olarak eklenebilir; çünkü veri işleme, checkpoint yönetimi, API ve demo yapısı modüler hale getirilmiştir.

## 8. Conclusions

Final sonuçlar, multimodal late fusion yaklaşımının H&M moda öneri probleminde tabular baseline’a göre anlamlı bir katkı sunduğunu göstermektedir. `tabular_only` modeli AUC 0.7841 ile güçlü bir baseline kurmuş, `image_only_effnet_cnn` AUC 0.7513 ile görsel sinyalin tek başına sınırlı kaldığını göstermiştir. Buna karşılık `late_fusion` final AUC 0.8063 ve best AUC 0.8121 ile en iyi classification sonucunu üretmiştir. Ranking tarafında da `late_fusion`, MAP@12 0.001262 ile ana modeller içinde liderdir.

Projenin en önemli çıktısı yalnızca metrik iyileşmesi değildir. Çalışma sonunda training notebookları, final artifactler, explainability çıktıları ve çalışan demo uygulaması aynı hikayede birleşmiştir. Demo, model sonuçlarını alışveriş sitesi benzeri bir arayüzde gösterir; model overlap ve açıklama chip’leri sayesinde teknik farklar sunum sırasında anlaşılır hale gelir. Bu nedenle final sistem, proposal’daki multimodal öneri hipotezini hem deneysel hem de uygulamalı olarak destekleyen bütünlüklü bir teslim çıktısıdır.

## Bibliography

[1] H&M Group, “H&M Personalized Fashion Recommendations,” Kaggle Competition Dataset, 2022.
[2] M. Tan and Q. Le, “EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks,” ICML, 2019.
[3] R. R. Selvaraju et al., “Grad-CAM: Visual Explanations from Deep Networks via Gradient-based Localization,” ICCV, 2017.
[4] S. M. Lundberg and S.-I. Lee, “A Unified Approach to Interpreting Model Predictions,” NeurIPS, 2017.
[5] Y. Koren, R. Bell, and C. Volinsky, “Matrix Factorization Techniques for Recommender Systems,” Computer, 2009.
[6] X. He et al., “Neural Collaborative Filtering,” WWW, 2017.
