# H&M Moda Ürünleri İçin Multimodal Öneri Sistemi

## Özet

Bu projede H&M Personalized Fashion Recommendations veri seti üzerinde müşteri-ürün önerisi üretmek için tabular müşteri/ürün metaverisi ile ürün görsellerinden çıkarılan görsel temsiller birlikte kullanılmıştır. Projenin temel hipotezi, ürün fotoğraflarından EfficientNet-B0 ile çıkarılan görsel embeddinglerin ve müşteri geçmişinden oluşturulan görsel tercih profilinin, tabular metaveri ile late fusion mimarisinde birleştirildiğinde tek modaliteli modellere göre daha güçlü bir satın alma olasılığı skoru üreteceğidir.

Bu hipotezi test etmek için üç model ailesi karşılaştırılmıştır:

- Tabular-only MLP
- Image-history MLP
- Multimodal late fusion MLP

Final full-training deneyleri Kaggle ortamında gerçekleştirilmiştir. Sonuçlar, multimodal late fusion modelinin hem tabular-only hem de image-history modellerinden daha yüksek doğrulama performansı verdiğini göstermiştir.

## Problem Tanımı

Amaç, bir müşterinin belirli bir ürünü satın alma olasılığını skorlayabilen bir öneri modeli geliştirmektir. H&M veri seti işlem geçmişi, müşteri metaverisi, ürün metaverisi ve ürün fotoğrafları içerdiği için problem multimodal recommendation kapsamında ele alınmıştır.

Modelin çıktısı doğrudan bir sınıf etiketi olarak değil, müşteri-ürün çifti için bir satın alma skoru olarak yorumlanmıştır. Bu skorlar daha sonra aday ürünleri sıralamak ve öneri listesi üretmek için kullanılabilir.

## Veri Seti

Projede kullanılan ana dosyalar şunlardır:

| Dosya | İçerik | Kullanım Amacı |
|---|---|---|
| `transactions_train.csv` | Müşteri satın alma geçmişi | Pozitif müşteri-ürün çiftleri, müşteri görsel profili, aday üretimi |
| `customers.csv` | Müşteri metaverisi | Tabular branch girdileri |
| `articles.csv` | Ürün metaverisi | Tabular branch girdileri ve ürün açıklayıcı bilgileri |
| `images/` | Ürün fotoğrafları | EfficientNet-B0 görsel embedding çıkarımı |

Full-training aşamasında 27,194,909 pozitif müşteri-ürün çifti işlenmiştir. Doğrulama için 120,000 pozitif satır ayrılmıştır. Negatif örnekler, müşterinin satın almadığı ürünlerden örneklenerek ikili sınıflandırma düzeni kurulmuştur.

## Görsel Temsil Çıkarımı

Ürün görselleri eğitim sırasında doğrudan CNN'e tekrar tekrar verilmemiştir. Bunun yerine ImageNet üzerinde önceden eğitilmiş EfficientNet-B0 modeli özellik çıkarıcı olarak kullanılmıştır. Her ürün görseli için 1280 boyutlu bir embedding üretilmiştir.

Bu yaklaşımın iki nedeni vardır:

- Görsel bilgiyi eğitim sırasında daha düşük maliyetli vektör temsilleri olarak kullanmak.
- Tabular modeller ve late fusion mimarisiyle uyumlu, sabit boyutlu ürün temsilleri elde etmek.

Final Kaggle embedding seti 105,100 ürün için çıkarılmıştır ve proje içinde `data/embeddings/final_kaggle/` altında saklanmıştır.

## Müşteri Görsel Geçmişi

Image-history yaklaşımında her müşteri için geçmişte satın aldığı ürünlerin EfficientNet embeddingleri kullanılarak bir görsel tercih profili oluşturulmuştur. Bu profil, müşterinin önceki alışverişlerinde hangi görsel özelliklere sahip ürünleri tercih ettiğini temsil eder.

Bir aday ürün değerlendirilirken iki temel görsel bilgi kullanılmıştır:

- Aday ürünün kendi görsel embeddingi
- Müşterinin geçmiş satın alımlarından türetilen görsel profil embeddingi

Ayrıca aday ürün embeddingi ile müşteri görsel profili arasındaki cosine similarity değeri de modele yardımcı bir skaler özellik olarak verilmiştir.

## Model Mimarileri

### Tabular-Only MLP

Tabular-only model yalnızca müşteri ve ürün metaverisini kullanır. Bu modelde görsel embedding bulunmaz. Sayısal özellikler normalize edilmiş, kategorik özellikler embedding katmanları üzerinden temsil edilmiştir. Sonrasında MLP katmanları müşteri-ürün çifti için satın alma olasılığı skoru üretmiştir.

Bu model, görsel modalite kullanılmadığında elde edilen baseline performansı ölçmek için kullanılmıştır.

### Image-History MLP

Image-history model, tabular müşteri/ürün metaverisi kullanmadan görsel sinyale odaklanır. Model girdileri ürün embeddingi, müşteri görsel profil embeddingi, görsel benzerlik skoru ve geçmiş görsel etkileşim sayısı gibi görsel geçmiş tabanlı özelliklerden oluşur.

Bu model, ürün fotoğrafları ve müşterinin görsel alışveriş geçmişinin tek başına ne kadar güçlü bir öneri sinyali taşıdığını ölçmek için tasarlanmıştır.

### Multimodal Late Fusion

Late fusion model iki ayrı temsil üretir:

- Tabular branch: müşteri ve ürün metaverisini işler.
- Visual branch: aday ürün embeddingi, müşteri görsel profili ve görsel benzerlik sinyallerini işler.

Bu iki branch çıktısı birleştirilerek son MLP katmanlarına verilir. Böylece model hem metaveri sinyalini hem de görsel tercih sinyalini aynı skor fonksiyonunda kullanır.

Bu mimari, proposal'daki ana hipotezi test eden modeldir.

## Deney Kurulumu

Full-training deneylerinde veri büyüklüğü nedeniyle streaming eğitim yaklaşımı kullanılmıştır. Tüm müşteri-ürün çiftlerini tek seferde belleğe almak yerine eğitim parçalar halinde yürütülmüştür.

Genel deney düzeni:

- Eğitim ortamı: Kaggle GPU ortamı
- Görsel embedding modeli: EfficientNet-B0
- Embedding boyutu: 1280
- Eğitim epoch sayısı: 3
- Pozitif örnekler: geçmiş satın alma çiftleri
- Negatif örnekler: satın alınmamış ürünlerden örneklenen müşteri-ürün çiftleri
- Ana metrikler: AUC-ROC ve accuracy

AUC-ROC, modelin pozitif ve negatif müşteri-ürün çiftlerini ayırma başarısını ölçmek için ana metrik olarak kullanılmıştır. Accuracy ise destekleyici sınıflandırma metriği olarak raporlanmıştır.

## Final Full-Training Sonuçları

| Model | Eğitim Düzeni | AUC-ROC | Accuracy | Açıklama |
|---|---|---:|---:|---|
| Tabular-only MLP | streaming full, 3 epoch | 0.8550 | 0.7636 | Müşteri ve ürün metaverisi |
| Image-history MLP | streaming full, 3 epoch | 0.9237 | 0.8297 | EfficientNet-B0 embedding + müşteri görsel geçmişi |
| Multimodal late fusion | streaming full, 3 epoch | 0.9341 | 0.8504 | Tabular branch + image-history branch |

Sonuçlar proposal'daki ana hipotezi desteklemektedir. Late fusion modeli, tabular-only modele göre AUC-ROC metriğinde yaklaşık 0.0791 puan, image-history modele göre ise yaklaşık 0.0104 puan daha yüksek performans göstermiştir. Accuracy metriğinde de en yüksek değer late fusion modelinde elde edilmiştir.

Image-history modelinin tabular-only modelden belirgin şekilde güçlü çıkması, H&M veri setinde ürün görsellerinin ve müşterinin görsel alışveriş geçmişinin ciddi bir öneri sinyali taşıdığını göstermektedir. Late fusion modelinin image-history modelini de geçmesi ise tabular metaverinin görsel sinyale ek bilgi kattığını göstermektedir.

## Controlled Ablation Sonuçları

Local geliştirme sürecinde daha küçük ve kontrollü bir ortamda ablation deneyi de yapılmıştır. Bu deney, model mimarilerinin aynı örneklem üzerinde hızlı ve kontrollü biçimde karşılaştırılması için kullanılmıştır.

| Model | Satır Sayısı | AUC-ROC | Accuracy |
|---|---:|---:|---:|
| Tabular-only MLP | 1,014,202 | 0.7219 | 0.6658 |
| Image-history MLP | 1,014,202 | 0.7883 | 0.7261 |
| Multimodal late fusion | 1,014,202 | 0.9611 | 0.8411 |

Controlled ablation sonuçları final full-training sonuçlarıyla birebir aynı ölçekte yorumlanmamalıdır. Bu deney daha küçük ürün/müşteri evreninde, daha sınırlı örneklemle yapılmıştır. Bu nedenle özellikle AUC değerleri daha kolay veya daha keskin ayrışan bir validasyon dağılımı üretebilir. Final proje iddiası için ana sonuçlar Kaggle full-training tablosudur; controlled ablation ise mimari tercihin geliştirme sürecindeki destekleyici kanıtı olarak değerlendirilmelidir.

## Açıklanabilirlik Analizi Planı

Proposal'da yer alan açıklanabilirlik hedefi iki başlıkta ele alınabilir:

### Tabular Açıklanabilirlik

Tabular-only model ve late fusion modelinin tabular branch girdileri için SHAP veya permutation importance analizi uygulanabilir. Bu analiz, modelin satın alma skorunu üretirken hangi müşteri ve ürün metaverilerine daha fazla duyarlı olduğunu gösterebilir.

Örnek açıklanabilir özellik grupları:

- Ürün tipi
- Ürün grubu
- Renk bilgisi
- Fiyat
- Müşteri yaşı
- Müşteri üyelik ve haber alma bilgileri

### Görsel Açıklanabilirlik

Projede EfficientNet-B0 doğrudan sınıflandırıcı olarak fine-tune edilmediği için klasik Grad-CAM analizi sınırlı uygulanabilirliğe sahiptir. Bu nedenle görsel açıklanabilirlik için daha uygun yöntem, embedding uzayında benzer ürünleri göstermektir.

Bir öneri için şu açıklama üretilebilir:

- Önerilen ürünün görsel embeddingi alınır.
- Müşterinin geçmiş satın aldığı ürün embeddingleriyle cosine similarity hesaplanır.
- En benzer geçmiş ürünler gösterilir.

Bu yöntem, “model bu ürünü önerdi çünkü müşterinin geçmişte aldığı şu ürünlerle görsel olarak benzer” şeklinde kullanıcıya anlaşılır bir açıklama sunar.

## Tartışma

Elde edilen sonuçlar, moda öneri sistemlerinde görsel temsilin güçlü bir kişiselleştirme sinyali olduğunu göstermektedir. Tabular metaveri tek başına anlamlı performans üretse de, müşterinin geçmiş alışverişlerinden türetilen görsel profil daha güçlü bir ayrıştırıcı sinyal sağlamıştır.

Late fusion modelinin en iyi performansı vermesi, iki modalitenin birbirini tamamladığını göstermektedir. Görsel branch ürün stilini ve görsel benzerliği yakalarken, tabular branch ürün kategorisi, müşteri metaverisi ve ürün açıklayıcı alanlarından gelen yapısal bilgiyi modele taşımaktadır.

Bu çalışma Kaggle leaderboard optimizasyonundan çok, proposal'da tanımlanan multimodal recommendation hipotezini deneysel olarak test etmeye odaklanmıştır. Kaggle submission skorları destekleyici dış çıktı olarak görülebilir; ancak ana bilimsel karşılaştırma aynı eğitim düzeninde eğitilen tabular-only, image-history ve late-fusion modelleridir.

## Sınırlılıklar

Bu aşamadaki bazı sınırlılıklar şunlardır:

- Final karşılaştırma AUC-ROC ve accuracy üzerinden yapılmıştır; MAP@12 / Precision@10 / Recall@10 ise ayrı time-based ranking evaluation olarak eklenmiştir.
- EfficientNet-B0 görsel embeddingleri sabit özellik çıkarıcı olarak kullanılmıştır; uçtan uca CNN fine-tuning yapılmamıştır.
- Negatif örnekleme yöntemi model başarısını etkileyebilir; daha zor negatif örnekler ile ek deney yapılabilir.
- Kaggle leaderboard skoru doğrudan ana deney metriği olarak optimize edilmemiştir.

## Gelecek Çalışmalar

Projenin sonraki adımlarında şu geliştirmeler yapılabilir:

- Yeni late-fusion checkpoint ile ranking evaluation sweep'inin tekrar üretilmesi ve candidate/reranking katmanlarının iyileştirilmesi.
- SHAP veya permutation importance ile tabular açıklanabilirlik analizi.
- Embedding benzerliği üzerinden görsel öneri açıklamaları.
- Basit bir fake alışveriş arayüzü ile model önerilerinin görselleştirilmesi.
- Aday ürün üretiminin zaman bazlı popülerlik, müşteri geçmişi ve benzer ürün stratejileriyle güçlendirilmesi.

## Sonuç

Bu projede H&M veri seti üzerinde tabular-only, image-history ve multimodal late fusion modelleri karşılaştırılmıştır. Final full-training sonuçlarında multimodal late fusion modeli en yüksek AUC-ROC ve accuracy değerlerini elde etmiştir. Bu bulgu, ürün görsellerinden çıkarılan EfficientNet-B0 embeddinglerinin ve müşteri geçmişinden oluşturulan görsel profilin, tabular müşteri/ürün metaverisiyle birlikte kullanıldığında öneri performansını artırdığı hipotezini desteklemektedir.
