# Deney Sonuçları ve Model Karşılaştırması

Bu dosya, Perseptron projesindeki üç ana modelin deney sonuçlarını tek yerde karşılaştırmak için hazırlanmıştır. Buradaki ana amaç, proposal'daki multimodal öneri hipotezinin deneysel olarak desteklenip desteklenmediğini net biçimde göstermektir.

## Karşılaştırılan Modeller

Projede aynı temel problem için üç model eğitilmiştir:

| Model | Kullanılan Bilgi | Deneydeki Rolü |
|---|---|---|
| Tabular-only MLP | Müşteri ve ürün metaverisi | Görsel bilgi olmadan baseline |
| Image-history MLP | EfficientNet-B0 ürün embeddingleri ve müşteri görsel geçmişi | Sadece görsel geçmiş sinyalini ölçen baseline |
| Multimodal late fusion | Tabular branch + image-history branch | Proposal'daki ana multimodal model |

Bu üçlü karşılaştırma bilinçli olarak kurulmuştur. Tabular-only model yapısal metaverinin gücünü, image-history model görsel tercih sinyalinin gücünü, late fusion model ise iki sinyalin birlikte kullanıldığında oluşturduğu katkıyı ölçer.

## Ana Deney: Kaggle Full-Training

Final sonuç olarak Kaggle full-training deneyleri esas alınmalıdır. Bu deneylerde büyük H&M veri seti üzerinde streaming eğitim yapılmıştır.

| Model | Eğitim Düzeni | Pozitif Çift Sayısı | Validation Pozitif Satır | AUC-ROC | Accuracy |
|---|---|---:|---:|---:|---:|
| Tabular-only MLP | 3 epoch streaming full training | 27,194,909 | 120,000 | 0.8550 | 0.7636 |
| Image-history MLP | 3 epoch streaming full training | 27,194,909 | 120,000 | 0.9237 | 0.8297 |
| Multimodal late fusion | 3 epoch streaming full training | 27,194,909 | 120,000 | 0.9341 | 0.8504 |

## Ana Bulgular

### 1. Görsel geçmiş sinyali tabular metaveriden daha güçlü çıktı

Image-history MLP modeli, tabular-only MLP modelinden belirgin şekilde daha yüksek performans vermiştir.

| Karşılaştırma | AUC-ROC Farkı | Accuracy Farkı |
|---|---:|---:|
| Image-history - Tabular-only | +0.0688 | +0.0661 |

Bu sonuç, moda öneri probleminde ürün görsellerinin ve müşterinin geçmişte satın aldığı ürünlerin görsel karakterinin güçlü bir kişiselleştirme sinyali taşıdığını gösterir. Başka bir deyişle müşteri geçmişindeki stil, renk, form ve ürün görünümü bilgisi sadece ürün/müşteri metaverisine göre daha ayırt edici olmuştur.

### 2. Late fusion en yüksek performansı verdi

Multimodal late fusion modeli, iki tek modaliteli modeli de geçmiştir.

| Karşılaştırma | AUC-ROC Farkı | Accuracy Farkı |
|---|---:|---:|
| Late fusion - Tabular-only | +0.0791 | +0.0868 |
| Late fusion - Image-history | +0.0104 | +0.0207 |

Bu sonuç proposal'daki ana hipotezi destekler. Görsel geçmiş tek başına çok güçlü olsa da, tabular metaveri modele ek bilgi sağlamıştır. Late fusion modelinin image-history modelden daha iyi çıkması, müşteri/ürün metaverisinin görsel sinyale tamamlayıcı katkı yaptığını gösterir.

### 3. Ana sonuç Kaggle full-training tablosudur

Local controlled ablation deneyleri geliştirme sürecinde yararlıdır; ancak final proje iddiası için ana tablo Kaggle full-training tablosudur. Bunun nedeni full-training deneylerinin daha büyük ürün evrenini, daha fazla müşteri-ürün çiftini ve Kaggle ortamındaki gerçek veri ölçeğini kullanmasıdır.

## Destekleyici Deney: Controlled Ablation

Local geliştirme sırasında daha küçük ve kontrollü bir ablation deneyi yapılmıştır.

| Model | Satır Sayısı | Train Satırı | Validation Satırı | Epoch | AUC-ROC | Accuracy |
|---|---:|---:|---:|---:|---:|---:|
| Tabular-only MLP | 1,014,202 | 811,361 | 202,841 | 3 | 0.7219 | 0.6658 |
| Image-history MLP | 1,014,202 | 811,361 | 202,841 | 3 | 0.7883 | 0.7261 |
| Multimodal late fusion | 1,014,202 | 811,361 | 202,841 | 3 | 0.9611 | 0.8411 |

Bu tabloda late fusion AUC değerinin çok yüksek görünmesi, full-training sonucuyla doğrudan çelişmez. Controlled ablation daha küçük, daha sınırlı ve daha kontrollü bir veri evreninde yapılmıştır. Bu nedenle validasyon dağılımı full H&M problemine göre daha kolay ayrışmış olabilir.

Rapor yazarken bu deney şu şekilde konumlandırılmalıdır:

- Ana sonuç değildir.
- Model tasarım kararını destekleyen geliştirme deneyi olarak sunulmalıdır.
- Üç model arasındaki göreli sıralamanın late fusion lehine olduğunu gösterir.

## Rapor İçin Kısa Yorum

Final full-training sonuçları, multimodal late fusion yaklaşımının öneri problemi için en güçlü model olduğunu göstermektedir. Tabular-only model müşteri ve ürün metaverisinden anlamlı sinyal çıkarabilmiş, image-history model ürün görsellerinden ve müşteri görsel geçmişinden daha güçlü bir kişiselleştirme sinyali üretmiş, late fusion model ise bu iki bilgiyi birleştirerek en yüksek doğrulama performansına ulaşmıştır.

Bu bulgu, proposal'da kurulan “ürün görselleri ve tabular metaveri birlikte kullanıldığında öneri performansı artar” hipotezini desteklemektedir.

## Rapor İçin Kullanılacak Ana Cümle

Bu çalışmada elde edilen full-training sonuçlarına göre multimodal late fusion modeli AUC-ROC 0.9341 ve accuracy 0.8504 değerleriyle en yüksek performansı elde etmiştir. Bu sonuç, EfficientNet-B0 ile çıkarılan ürün görsel embeddinglerinin ve müşteri geçmişinden oluşturulan görsel tercih profilinin, tabular müşteri/ürün metaverisiyle birlikte kullanılmasının öneri modelinin ayırt edici gücünü artırdığını göstermektedir.
