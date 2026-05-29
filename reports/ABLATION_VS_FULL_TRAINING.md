# Controlled Ablation ve Full Training Sonuçlarının Yorumlanması

Bu dosya, local controlled ablation deneyleri ile Kaggle full-training deneyleri arasındaki farkı açıklamak için hazırlanmıştır. Amaç, raporda iki farklı sonuç tablosu kullanılırken okuyucuda “neden skorlar birebir aynı ölçekte değil?” sorusunun oluşmasını engellemektir.

## Kısa Cevap

Controlled ablation ve full training aynı amaca hizmet etmez:

- Controlled ablation, model mimarilerini hızlı ve kontrollü biçimde karşılaştırmak için yapılmıştır.
- Full training, projenin nihai deney sonucu olarak büyük H&M veri ölçeğinde yapılmıştır.

Bu nedenle final proje iddiası için Kaggle full-training sonuçları esas alınmalıdır. Controlled ablation ise model seçimini ve multimodal yaklaşımın geliştirme sürecindeki etkisini destekleyen ek deney olarak sunulmalıdır.

## İki Deneyin Kapsam Farkı

| Özellik | Controlled Ablation | Kaggle Full Training |
|---|---|---|
| Amaç | Mimari karşılaştırmasını kontrollü ortamda görmek | Final model performansını büyük veri ölçeğinde ölçmek |
| Veri ölçeği | 1,014,202 satır | 27,194,909 pozitif çift |
| Ortam | Local/sandbox geliştirme düzeni | Kaggle GPU ortamı |
| Kullanım yeri | Destekleyici analiz | Ana deney sonucu |
| Yorumlama | Göreli model sıralaması için uygundur | Final performans iddiası için uygundur |

## Controlled Ablation Neden Farklı Skor Verebilir?

Controlled ablation daha küçük ve daha sınırlı bir örneklem üzerinde yapıldığı için skorları full training ile birebir karşılaştırmak doğru değildir. Bunun birkaç nedeni vardır:

### 1. Ürün ve müşteri evreni daha küçüktür

Controlled ablation deneyinde kullanılan ürün ve müşteri çeşitliliği full H&M evrenine göre daha sınırlıdır. Ürün evreni küçüldüğünde negatif ve pozitif örnekleri ayırmak daha kolay hale gelebilir. Bu durum özellikle AUC-ROC değerinin daha yüksek görünmesine neden olabilir.

### 2. Validasyon dağılımı daha kolay olabilir

Local ablation validasyonu, full training validasyonuna göre daha kontrollü bir örnekleme yapısından gelir. Bu validasyon setinde pozitif ve negatif çiftler daha belirgin biçimde ayrışıyorsa model daha yüksek AUC üretebilir. Bu, modelin full problemde aynı seviyede genelleyeceği anlamına gelmez.

### 3. Negatif örnekleme etkisi büyüktür

Öneri sistemlerinde negatif örnekler genellikle “müşterinin satın almadığı ürünler” olarak örneklenir. Eğer negatifler kolay örneklerden seçilirse modelin işi kolaylaşır. Daha zor negatifler, yani müşterinin gerçekten satın alma ihtimali olan ama satın almadığı benzer ürünler seçilirse skorlar daha gerçekçi ve daha düşük olabilir.

Controlled ablation deneylerinde negatif örnekleme düzeni model mimarilerini hızlı karşılaştırmak için yeterlidir; fakat final performans iddiası için full training sonucu daha güvenilirdir.

### 4. Full training daha gerçekçi ve daha gürültülüdür

Full H&M verisi çok daha fazla müşteri, ürün ve işlem çeşitliliği içerir. Bu ölçekte müşteri davranışları daha dağınık, ürün evreni daha geniş ve aday ayrımı daha zordur. Bu yüzden full-training skorları genellikle daha gerçekçi kabul edilmelidir.

## Sonuç Tablolarının Birlikte Yorumlanması

Controlled ablation sonuçları:

| Model | AUC-ROC | Accuracy |
|---|---:|---:|
| Tabular-only MLP | 0.7219 | 0.6658 |
| Image-history MLP | 0.7883 | 0.7261 |
| Multimodal late fusion | 0.9611 | 0.8411 |

Full-training sonuçları:

| Model | AUC-ROC | Accuracy |
|---|---:|---:|
| Tabular-only MLP | 0.8550 | 0.7636 |
| Image-history MLP | 0.9237 | 0.8297 |
| Multimodal late fusion | 0.9341 | 0.8504 |

İki tabloda da ortak sonuç şudur: multimodal late fusion modeli en güçlü modeldir. Bu nedenle controlled ablation, final bulguyu destekleyen bir ön/yan deney olarak değerlidir.

Ancak controlled ablation tablosundaki late fusion AUC değerinin 0.9611 olması, full-training AUC değerinin 0.9341 olmasıyla çelişmez. Çünkü iki deneyin veri ölçeği, örnekleme koşulları ve validasyon dağılımı aynı değildir.

## Raporda Nasıl Sunulmalı?

Raporun ana sonuç bölümünde önce Kaggle full-training tablosu verilmelidir. Bu tablo projenin final deney sonucudur.

Controlled ablation ise şu başlıklardan biri altında sunulabilir:

- “Destekleyici Ablation Deneyi”
- “Model Seçimini Destekleyen Kontrollü Deney”
- “Local Geliştirme ve Ablation Analizi”

Bu bölümde şu vurgu yapılmalıdır:

> Controlled ablation deneyleri, model mimarilerinin göreli davranışını görmek için kullanılmıştır. Final performans iddiası ise daha büyük veri ölçeğinde yapılan Kaggle full-training sonuçlarına dayandırılmıştır.

## Rapor İçin Hazır Metin

Controlled ablation deneyleri, tabular-only, image-history ve multimodal late fusion mimarilerini daha küçük ve kontrollü bir örneklem üzerinde karşılaştırmak amacıyla uygulanmıştır. Bu deneyler final performans iddiası için değil, model tasarım kararlarını desteklemek için kullanılmıştır. Ablation sonuçlarında multimodal late fusion modelinin en yüksek AUC-ROC değerine ulaşması, görsel ve tabular sinyallerin birlikte kullanımının model performansını artırabileceğini göstermiştir.

Buna karşılık final sonuçlar Kaggle full-training deneylerinden elde edilmiştir. Full-training düzeni daha geniş müşteri-ürün evrenini, daha fazla pozitif çifti ve daha gerçekçi veri çeşitliliğini içerdiği için nihai proje değerlendirmesinde ana referans olarak alınmıştır. Her iki deney düzeninde de late fusion modelinin en yüksek performansı vermesi, projenin temel multimodal öneri hipotezini desteklemektedir.

## Sonuç

Controlled ablation ve full training sonuçları birbirinin alternatifi değildir. Controlled ablation model tasarımını anlamaya yarayan kontrollü bir analizdir; full training ise final performans değerlendirmesidir. Rapor yazımında ana bilimsel iddia Kaggle full-training tablosuna dayandırılmalı, ablation sonuçları ise bu iddiayı destekleyen ek analiz olarak kullanılmalıdır.
