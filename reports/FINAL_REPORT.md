# H&M Moda Ürünleri İçin Multimodal Öneri Sistemi

## 1. Giriş

Moda ürünleri için öneri sistemleri, klasik ürün öneri problemlerinden farklı olarak güçlü bir görsel bileşen içerir. Kullanıcıların tercihleri yalnızca ürün kategorisi, fiyat veya müşteri profili gibi tabular bilgilerle değil; ürünün rengi, kesimi, dokusu, görsel stili ve geçmişte satın alınan ürünlerle olan benzerliğiyle de ilişkilidir.

Bu projede H&M Personalized Fashion Recommendations veri seti üzerinde müşteri-ürün önerisi üretmek için tabular müşteri/ürün metaverisi ile ürün görsellerinden çıkarılan görsel temsiller birlikte kullanılmıştır. Projenin temel hipotezi şudur:

> Ürün fotoğraflarından EfficientNet-B0 ile çıkarılan görsel embeddingler, müşterinin geçmiş alışverişlerinden oluşturulan görsel tercih profili ve tabular müşteri/ürün özellikleri late fusion mimarisinde birleştirildiğinde, yalnızca tabular veya yalnızca görsel geçmiş kullanan modellere göre daha güçlü bir satın alma olasılığı skoru üretir.

Bu hipotezi test etmek için üç temel model karşılaştırılmıştır:

- Tabular-only MLP
- Image-history MLP
- Multimodal late fusion MLP

## 2. Problem Tanımı

Amaç, bir müşterinin belirli bir ürünü satın alma olasılığını skorlayabilen bir öneri modeli geliştirmektir. Model çıktısı doğrudan nihai bir sınıf etiketi olarak değil, müşteri-ürün çifti için satın alma olasılığına karşılık gelen bir skor olarak ele alınmıştır. Bu skorlar daha sonra aday ürünleri sıralamak ve öneri listesi üretmek için kullanılabilir.

Bu problem multimodal recommendation kapsamında değerlendirilmiştir. Çünkü veri seti aynı anda işlem geçmişi, müşteri metaverisi, ürün metaverisi ve ürün fotoğrafları içerir.

## 3. Veri Seti

Projede kullanılan ana veri kaynakları şunlardır:

| Dosya/Klasör | İçerik | Projedeki Rolü |
|---|---|---|
| `transactions_train.csv` | Müşteri satın alma geçmişi | Pozitif müşteri-ürün çiftleri, müşteri görsel profili ve aday üretimi |
| `customers.csv` | Müşteri metaverisi | Tabular model girdileri |
| `articles.csv` | Ürün metaverisi | Tabular model girdileri ve ürün açıklayıcı bilgileri |
| `images/` | Ürün fotoğrafları | EfficientNet-B0 ile görsel embedding çıkarımı |

Full-training aşamasında 27,194,909 pozitif müşteri-ürün çifti kullanılmıştır. Doğrulama için 120,000 pozitif satır ayrılmıştır. Negatif örnekler, müşterinin satın almadığı ürünlerden örneklenerek ikili sınıflandırma düzeni kurulmuştur.

## 4. Yöntem

Genel sistem akışı üç ayrı modelin aynı müşteri-ürün skorlama problemi üzerinde karşılaştırılmasına dayanır. İlk model yalnızca tabular müşteri/ürün metaverisini kullanır. İkinci model ürün fotoğraflarından çıkarılan embeddingler ve müşterinin geçmiş alışverişlerinden oluşturulan görsel tercih profilini kullanır. Üçüncü model ise bu iki bilgi kaynağını late fusion mimarisiyle birleştirir.

Rapor ve sunumda kullanılabilecek pipeline diyagramları ayrıca `reports/PIPELINE_DIAGRAMS.md` dosyasında hazırlanmıştır.

### 4.1 Görsel Temsil Çıkarımı

Ürün görselleri eğitim sırasında doğrudan CNN modeline tekrar tekrar verilmemiştir. Bunun yerine ImageNet üzerinde önceden eğitilmiş EfficientNet-B0 modeli özellik çıkarıcı olarak kullanılmıştır. Her ürün görseli için 1280 boyutlu bir embedding vektörü üretilmiştir.

Bu yaklaşımın temel avantajları şunlardır:

- Eğitim sırasında pahalı CNN forward işlemlerini tekrar etmeden görsel bilgiyi kullanmak.
- Ürün görsellerini tabular modeller ve fusion mimarisiyle uyumlu sabit boyutlu vektörlere dönüştürmek.
- Müşteri geçmişinden görsel tercih profili oluşturmayı kolaylaştırmak.

Final Kaggle embedding seti 105,100 ürün için çıkarılmıştır ve proje klasöründe `data/embeddings/final_kaggle/` altında saklanmıştır.

### 4.2 Müşteri Görsel Geçmişi

Image-history yaklaşımında her müşteri için geçmişte satın aldığı ürünlerin EfficientNet embeddingleri kullanılarak bir görsel tercih profili oluşturulmuştur. Bu profil müşterinin geçmiş alışverişlerinde hangi görsel özelliklere sahip ürünleri tercih ettiğini temsil eder.

Bir aday ürün değerlendirilirken şu görsel bilgiler kullanılmıştır:

- Aday ürünün kendi görsel embeddingi
- Müşterinin geçmiş ürünlerinden oluşturulan görsel profil embeddingi
- Aday ürün ile müşteri görsel profili arasındaki cosine similarity
- Müşterinin görsel geçmiş uzunluğu

### 4.3 Tabular-Only MLP

Tabular-only model yalnızca müşteri ve ürün metaverisini kullanır. Sayısal özellikler normalize edilmiş, kategorik özellikler embedding katmanlarıyla temsil edilmiştir. Sonrasında MLP katmanları müşteri-ürün çifti için satın alma skoru üretmiştir.

Bu model, görsel modalite kullanılmadığında elde edilen baseline performansı ölçmek için kullanılmıştır.

### 4.4 Image-History MLP

Image-history model tabular müşteri/ürün metaverisi kullanmadan görsel sinyale odaklanır. Model girdileri aday ürün embeddingi, müşteri görsel profil embeddingi, görsel benzerlik skoru ve görsel geçmiş sayısıdır.

Bu model, ürün fotoğrafları ve müşterinin görsel alışveriş geçmişinin tek başına ne kadar güçlü bir öneri sinyali taşıdığını ölçmek için tasarlanmıştır.

### 4.5 Multimodal Late Fusion

Late fusion model iki ayrı temsil üretir:

- Tabular branch: müşteri ve ürün metaverisini işler.
- Visual branch: aday ürün embeddingi, müşteri görsel profili ve görsel benzerlik sinyallerini işler.

Bu iki branch çıktısı birleştirilerek son fully connected katmanlara verilir. Böylece model hem yapısal metaveri sinyalini hem de görsel tercih sinyalini aynı skor fonksiyonunda kullanır. Bu mimari, projenin ana hipotezini test eden modeldir.

## 5. Deney Kurulumu

Final deneyler Kaggle GPU ortamında yürütülmüştür. Veri büyüklüğü nedeniyle streaming eğitim yaklaşımı kullanılmıştır. Tüm müşteri-ürün çiftlerini tek seferde belleğe almak yerine eğitim parçalar halinde yapılmıştır.

Genel deney ayarları:

| Ayar | Değer |
|---|---|
| Görsel embedding modeli | EfficientNet-B0 |
| Embedding boyutu | 1280 |
| Eğitim epoch sayısı | 3 |
| Pozitif örnekler | Geçmiş satın alma çiftleri |
| Negatif örnekler | Satın alınmamış ürünlerden örneklenen müşteri-ürün çiftleri |
| Ana metrikler | AUC-ROC, Accuracy |

AUC-ROC, modelin pozitif ve negatif müşteri-ürün çiftlerini ayırma başarısını ölçmek için ana metrik olarak kullanılmıştır. Accuracy ise destekleyici sınıflandırma metriği olarak raporlanmıştır.

## 6. Final Full-Training Sonuçları

Final sonuç olarak Kaggle full-training deneyleri esas alınmıştır.

| Model | Eğitim Düzeni | Pozitif Çift Sayısı | Validation Pozitif Satır | AUC-ROC | Accuracy |
|---|---|---:|---:|---:|---:|
| Tabular-only MLP | 3 epoch streaming full training | 27,194,909 | 120,000 | 0.8550 | 0.7636 |
| Image-history MLP | 3 epoch streaming full training | 27,194,909 | 120,000 | 0.9237 | 0.8297 |
| Multimodal late fusion | 3 epoch streaming full training | 27,194,909 | 120,000 | 0.9341 | 0.8504 |

Image-history modeli, tabular-only modele göre AUC-ROC metriğinde yaklaşık 0.0688 puan daha yüksek performans vermiştir. Bu sonuç, ürün görsellerinin ve müşterinin görsel satın alma geçmişinin güçlü bir kişiselleştirme sinyali taşıdığını göstermektedir.

Late fusion modeli ise hem tabular-only hem de image-history modelini geçmiştir. Late fusion modeli tabular-only modele göre AUC-ROC metriğinde yaklaşık 0.0791 puan, image-history modele göre ise yaklaşık 0.0104 puan daha yüksek performans göstermiştir. Accuracy metriğinde de en yüksek değer late fusion modelinde elde edilmiştir.

Bu sonuçlar, proposal aşamasında kurulan multimodal öneri hipotezini desteklemektedir.

## 7. Controlled Ablation Analizi

Local geliştirme sürecinde daha küçük ve kontrollü bir ablation deneyi de yapılmıştır. Bu deney, model mimarilerinin aynı örneklem üzerinde göreli davranışını görmek için kullanılmıştır.

| Model | Satır Sayısı | Train Satırı | Validation Satırı | Epoch | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Tabular-only MLP | 1,014,202 | 811,361 | 202,841 | 3 | 0.7219 | 0.6658 |
| Image-history MLP | 1,014,202 | 811,361 | 202,841 | 3 | 0.7883 | 0.7261 |
| Multimodal late fusion | 1,014,202 | 811,361 | 202,841 | 3 | 0.9611 | 0.8411 |

Controlled ablation ve full-training sonuçları birebir aynı ölçekte yorumlanmamalıdır. Ablation deneyi daha küçük ürün/müşteri evreninde ve daha kontrollü bir örneklem üzerinde yapılmıştır. Bu nedenle validasyon dağılımı full H&M problemine göre daha kolay ayrışmış olabilir.

Rapor açısından doğru yorum şudur:

- Controlled ablation ana sonuç değildir.
- Model tasarım kararını destekleyen yardımcı deneydir.
- Final performans iddiası Kaggle full-training tablosuna dayandırılmalıdır.

Her iki deney düzeninde de late fusion modelinin en güçlü model olması, multimodal yaklaşımın tutarlı biçimde avantaj sağladığını göstermektedir.

## 8. Açıklanabilirlik Analizi

Proposal'da açıklanabilirlik için SHAP ve Grad-CAM benzeri yaklaşımlar düşünülmüştür. Ancak projenin uygulanmış mimarisinde EfficientNet-B0 doğrudan sınıflandırıcı olarak fine-tune edilmemiş, sabit feature extractor olarak kullanılmıştır. Ayrıca tabular modelde kategorik alanların embedding katmanlarından geçmesi SHAP entegrasyonunu daha maliyetli hale getirmiştir.

Bu nedenle açıklanabilirlik iki pratik ve mimariye uygun analizle ele alınmıştır:

- Görsel benzerlik analizi
- Tabular permutation importance analizi

### 8.1 Görsel Benzerlik Analizi

Görsel benzerlik analizinde aday ürünün embedding vektörü ile müşterinin geçmişte satın aldığı ürünlerin embeddingleri karşılaştırılmıştır. Cosine similarity değeri en yüksek olan geçmiş ürünler, aday ürünün görsel açıklaması olarak raporlanmıştır.

Bu analiz şu çıktıları üretmiştir:

- `reports/visual_similarity_examples.csv`
- `reports/visual_similarity_examples.html`

Bu yaklaşım, “model bu ürünü önerdi çünkü müşterinin geçmişte aldığı şu ürünlerle görsel olarak benzer” şeklinde yorumlanabilir bir açıklama sağlar. Bu, moda öneri sistemleri için özellikle uygundur; çünkü kullanıcı tercihleri çoğu zaman ürün tipi, renk, kesim, stil ve görsel kompozisyon gibi görsel sinyallerle ilişkilidir.

Grad-CAM yerine bu yöntemin kullanılmasının nedeni, EfficientNet-B0 modelinin nihai karar modeli olarak değil, embedding extractor olarak kullanılmasıdır. Bu nedenle embedding uzayında en yakın geçmiş ürünleri göstermek mevcut mimariyle daha doğrudan uyumludur.

### 8.2 Tabular Feature Importance

Tabular açıklanabilirlik için permutation importance analizi yapılmıştır. Bu analiz controlled/local tabular-only MLP modeli üzerinde gerçekleştirilmiştir. Dolayısıyla Kaggle full-training tabular modelinin veya late fusion modelinin birebir açıklaması değildir; tabular branch'in genel olarak hangi özellik türlerinden sinyal aldığını göstermek için destekleyici analiz olarak değerlendirilmelidir.

Hızlı analiz için validation setinden 20,000 satırlık örneklem kullanılmıştır. Baseline performans yaklaşık olarak AUC-ROC 0.7254 ve accuracy 0.6677 olarak ölçülmüştür.

En yüksek AUC düşüşü oluşturan özellikler:

| Özellik | Tip | Grup | AUC Düşüşü |
|---|---|---|---:|
| `section_name` | kategorik | ürün kategorik | 0.0471 |
| `department_name` | kategorik | ürün kategorik | 0.0236 |
| `product_type_name` | kategorik | ürün kategorik | 0.0176 |
| `age` | sayısal | müşteri sayısal | 0.0152 |
| `product_code` | sayısal | ürün sayısal | 0.0138 |
| `colour_group_name` | kategorik | ürün kategorik | 0.0130 |

Grup bazında sonuçlar:

| Özellik Grubu | Toplam AUC Düşüşü | Ortalama AUC Düşüşü |
|---|---:|---:|
| Ürün kategorik özellikleri | 0.1562 | 0.0130 |
| Ürün sayısal özellikleri | 0.0197 | 0.0020 |
| Müşteri sayısal özellikleri | 0.0152 | 0.0051 |
| Müşteri kategorik özellikleri | 0.0005 | 0.0003 |

Bu sonuçlar, tabular modelin özellikle ürün kategorisi, departman, bölüm, ürün tipi, renk ve görünüm gibi ürün tanımlayıcı metaverilerinden yararlandığını göstermektedir. Müşteri tarafında ise `age` değişkeni anlamlı katkı sağlamıştır.

SHAP analizi bu aşamada uygulanmamıştır. Bunun nedeni PyTorch MLP yapısındaki embedding katmanları ve full-training pipeline'ındaki streaming preprocessing düzeninin SHAP entegrasyonunu daha karmaşık hale getirmesidir. SHAP analizi ileriki çalışma olarak değerlendirilebilir.

## 9. Demo Arayüzü

Projede final checkpointlerin yalnızca raporlanmasıyla yetinilmemiş, modellerin canlı olarak denenebileceği bir demo katmanı da hazırlanmıştır. Bu demo, fake alışveriş sitesi mantığında çalışır: kullanıcı katalogdan daha önce satın almış gibi ürünler seçer, basit müşteri profili alanlarını düzenler ve üç modelin önerilerini yan yana görür.

Demo iki parçadan oluşur:

- FastAPI backend: final Kaggle checkpointlerini, ürün metadatasını, embedding dosyalarını ve görsel dosyalarını yükler.
- React/Vite frontend: ürün katalogu, seçilen geçmiş ürünler, müşteri profili ve model önerilerini kullanıcıya gösterir.

Demo inference sürecinde backend önce seçilen geçmiş ürünlere göre aday ürün havuzu oluşturur. Daha sonra aynı aday havuzu tabular-only, image-history ve late-fusion modelleriyle ayrı ayrı skorlanır. Böylece kullanıcı aynı geçmiş alışveriş senaryosu için üç modelin farklı öneri davranışlarını karşılaştırabilir.

Bu arayüz projenin ana deney sonucunun yerine geçmez; ancak final modellerin gerçek bir ürün öneri uygulamasına nasıl bağlanabileceğini gösteren destekleyici bir çıktı olarak değerlendirilmelidir. Özellikle image-history ve late-fusion modellerinde seçilen geçmiş ürünler müşterinin görsel tercih profiline dönüştüğü için demo, projenin temel multimodal fikrini somutlaştırır.

Demo dosyaları:

- `demo/backend/app.py`
- `demo/backend/recommender.py`
- `demo/backend/test_recommend.py`
- `demo/frontend/`

## 10. Tartışma

Elde edilen sonuçlar, moda öneri sistemlerinde görsel temsilin güçlü bir kişiselleştirme sinyali olduğunu göstermektedir. Tabular-only model anlamlı performans üretmiş olsa da, image-history modelin belirgin biçimde daha yüksek performans vermesi görsel geçmiş bilgisinin bu veri setinde çok önemli olduğunu göstermektedir.

Late fusion modelinin en yüksek performansı vermesi, iki modalitenin birbirini tamamladığını göstermektedir. Görsel branch ürün stilini ve müşterinin görsel tercihlerini yakalarken, tabular branch ürün kategorisi, ürün tipi, renk, departman ve müşteri metaverisi gibi yapısal bilgileri modele taşımaktadır.

Bu proje Kaggle leaderboard optimizasyonundan çok, ders projesi bağlamında multimodal recommendation hipotezini test etmeye odaklanmıştır. Kaggle submission skorları destekleyici dış çıktı olarak görülebilir; ancak ana bilimsel karşılaştırma aynı full-training düzeninde eğitilen tabular-only, image-history ve late fusion modelleridir.

## 11. Sınırlılıklar

Bu çalışmanın bazı sınırlılıkları vardır:

- Final karşılaştırma AUC-ROC ve accuracy üzerinden yapılmıştır; MAP@12, Precision@10 ve Recall@10 ise ayrı time-based ranking evaluation olarak eklenmiştir. Bu değerlendirme Kaggle public leaderboard ile birebir aynı değildir.
- EfficientNet-B0 sabit feature extractor olarak kullanılmıştır; uçtan uca CNN fine-tuning yapılmamıştır.
- Negatif örnekleme yöntemi model başarısını etkileyebilir. Daha zor negatif örnekler ile ek deney yapılabilir.
- Tabular permutation importance analizi Kaggle full-training modeli yerine controlled/local tabular-only model üzerinde yapılmıştır.
- SHAP analizi bu aşamada uygulanmamış, permutation importance daha pratik açıklanabilirlik yöntemi olarak kullanılmıştır.

## 12. Gelecek Çalışmalar

Projenin sonraki adımlarında şu geliştirmeler yapılabilir:

- Yeni late-fusion checkpoint ile ranking evaluation sweep'inin tekrar üretilmesi ve candidate/reranking katmanlarının iyileştirilmesi.
- Kaggle full-training tabular modeli için ayrı feature importance veya SHAP analizi.
- Late fusion modelinin tüm karar mekanizmasını açıklayan daha kapsamlı multimodal interpretability analizi.
- Daha zor negatif örnekleme stratejileri.
- Demo arayüzünün daha kapsamlı kullanıcı senaryoları, ürün filtreleri ve görsel açıklama panelleriyle geliştirilmesi.

## 13. Sonuç

Bu projede H&M veri seti üzerinde tabular-only, image-history ve multimodal late fusion modelleri karşılaştırılmıştır. Final full-training sonuçlarında multimodal late fusion modeli AUC-ROC 0.9341 ve accuracy 0.8504 değerleriyle en yüksek performansı elde etmiştir.

Bu bulgu, ürün görsellerinden çıkarılan EfficientNet-B0 embeddinglerinin ve müşteri geçmişinden oluşturulan görsel tercih profilinin, tabular müşteri/ürün metaverisiyle birlikte kullanıldığında öneri performansını artırdığı hipotezini desteklemektedir. Görsel açıklanabilirlik analizi ve tabular feature importance sonuçları da modelin kullandığı sinyallerin yorumlanmasına katkı sağlamıştır.
