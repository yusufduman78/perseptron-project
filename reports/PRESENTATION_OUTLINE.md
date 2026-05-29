# Perseptron Proje Sunumu Taslağı

Bu dosya, H&M multimodal öneri sistemi projesi için Türkçe sunum akışını içerir. Amaç, projeyi 8-10 dakikalık ders sunumunda anlaşılır, teknik olarak savunulabilir ve görsel olarak takip edilebilir şekilde anlatmaktır.

## Sunum Başlığı

**H&M Moda Ürünleri İçin Multimodal Öneri Sistemi**

Alt başlık:

**Tabular metadata, ürün görselleri ve müşteri görsel geçmişiyle late fusion tabanlı satın alma skorlama modeli**

---

## Slayt 1 - Kapak

**Başlık:** H&M Moda Ürünleri İçin Multimodal Öneri Sistemi

**Alt mesaj:** Ürün fotoğrafları ve tabular metaveri birlikte kullanıldığında öneri performansı artar mı?

**Görsel önerisi:**

- H&M ürün görsellerinden 3-4 örnek ürün kartı.
- Arka planda sade, moda katalogu hissi veren koyu/açık kontrastlı düzen.

**Konuşma notu:**

Bu projede H&M veri seti üzerinde müşteri-ürün önerisi problemi ele alındı. Temel soru, ürün görsellerinden çıkarılan temsilin müşteri ve ürün metaverisiyle birlikte kullanıldığında öneri başarısını artırıp artırmadığıdır.

---

## Slayt 2 - Problem ve Motivasyon

**Ana mesaj:** Moda önerisi sadece kategori veya müşteri bilgisiyle açıklanamaz; görsel stil önemli bir sinyaldir.

**İçerik:**

- Klasik öneri sistemleri çoğunlukla işlem geçmişi ve tabular özelliklere dayanır.
- Moda alanında renk, kesim, form, desen ve stil gibi görsel sinyaller belirleyicidir.
- Aynı ürün kategorisindeki iki ürün kullanıcı açısından tamamen farklı görünebilir.

**Görsel önerisi:**

- Aynı ürün tipinden farklı renk/stil örnekleri.
- Kısa bir ifade: “Metadata ürünü tarif eder; görsel embedding stili temsil eder.”

**Konuşma notu:**

H&M gibi moda veri setlerinde ürünün sadece “trousers” veya “shirt” olması yeterli değildir. Kullanıcının daha önce aldığı ürünlerin görsel karakteri yeni öneriler için güçlü bir ipucu verebilir.

---

## Slayt 3 - Hipotez

**Ana mesaj:** Görsel geçmiş + tabular metaveri birlikte kullanıldığında tek modaliteli modellere göre daha güçlü skor üretir.

**Hipotez metni:**

EfficientNet-B0 ile çıkarılan ürün görsel embeddingleri, müşterinin geçmiş alışverişlerinden oluşturulan görsel tercih profili ve tabular müşteri/ürün özellikleri late fusion mimarisinde birleştirildiğinde öneri performansı artar.

**Karşılaştırılan modeller:**

| Model | Rol |
|---|---|
| Tabular-only MLP | Görsel bilgi olmadan baseline |
| Image-history MLP | Sadece görsel geçmiş sinyali |
| Multimodal late fusion | Ana multimodal model |

**Konuşma notu:**

Deney tasarımı üç modeli bilinçli olarak karşılaştırıyor. Böylece tabular bilginin, görsel geçmişin ve ikisinin birleşiminin ayrı ayrı katkısı görülebiliyor.

---

## Slayt 4 - Veri Seti ve Kullanılan Alanlar

**Ana mesaj:** Veri seti işlem geçmişi, müşteri metaverisi, ürün metaverisi ve ürün görsellerini birlikte sunuyor.

**Tablo:**

| Veri | Projedeki rolü |
|---|---|
| `transactions_train.csv` | Pozitif müşteri-ürün çiftleri, müşteri geçmişi |
| `customers.csv` | Müşteri metaverisi |
| `articles.csv` | Ürün metaverisi |
| `images/` | EfficientNet-B0 embedding çıkarımı |

**Önemli sayılar:**

- 105,100 ürün için görsel embedding.
- 1280 boyutlu EfficientNet-B0 embedding.
- Full-training aşamasında 27,194,909 pozitif müşteri-ürün çifti.

**Konuşma notu:**

Bu ölçekteki veri local ortamda tek seferde belleğe alınamadığı için final eğitimler Kaggle üzerinde streaming yaklaşımla yapıldı.

---

## Slayt 5 - Görsel Embedding ve Müşteri Görsel Profili

**Ana mesaj:** Ürün görselleri doğrudan CNN ile her epoch tekrar işlenmedi; önce sabit embeddinglere dönüştürüldü.

**Akış:**

1. Ürün fotoğrafı alınır.
2. EfficientNet-B0 feature extractor’dan geçirilir.
3. 1280 boyutlu ürün embeddingi elde edilir.
4. Müşterinin geçmişte aldığı ürün embeddingleri ortalanarak görsel profil oluşturulur.
5. Aday ürün embeddingi ile müşteri profili karşılaştırılır.

**Görsel önerisi:**

`reports/PIPELINE_DIAGRAMS.md` içindeki genel pipeline veya image-history diyagramı.

**Konuşma notu:**

Burada CNN sınıflandırıcı olarak fine-tune edilmedi. EfficientNet-B0 sabit feature extractor olarak kullanıldı. Bu hem hesaplama maliyetini azalttı hem de görsel bilgiyi MLP modelleriyle kullanılabilir hale getirdi.

---

## Slayt 6 - Model Mimarileri

**Ana mesaj:** Üç model aynı problemi farklı bilgi kaynaklarıyla skorladı.

**Model özeti:**

| Model | Girdi | Çıktı |
|---|---|---|
| Tabular-only | Müşteri + ürün metaverisi | Satın alma skoru |
| Image-history | Aday embedding + müşteri görsel profili | Satın alma skoru |
| Late fusion | Tabular branch + visual branch | Satın alma skoru |

**Görsel önerisi:**

`reports/PIPELINE_DIAGRAMS.md` içindeki “Üç Modelin Girdi Farkı” diyagramı.

**Konuşma notu:**

Late fusion modelde iki ayrı branch var. Tabular branch metadata bilgisini, visual branch ise aday ürün ve müşteri görsel geçmişi bilgisini işliyor. Bu iki temsil son katmanlarda birleştiriliyor.

---

## Slayt 7 - Eğitim Kurulumu

**Ana mesaj:** Final deneyler full veri ölçeğinde, streaming eğitimle ve aynı doğrulama mantığında yürütüldü.

**İçerik:**

- Ortam: Kaggle GPU.
- Eğitim: 3 epoch streaming full training.
- Pozitifler: Gerçek satın alma çiftleri.
- Negatifler: Satın alınmamış ürünlerden örneklenen çiftler.
- Metrikler: AUC-ROC ve Accuracy.

**Tablo:**

| Ayar | Değer |
|---|---|
| Embedding modeli | EfficientNet-B0 |
| Embedding boyutu | 1280 |
| Epoch | 3 |
| Ana metrik | AUC-ROC |

**Konuşma notu:**

Burada amaç Kaggle leaderboard için maksimum MAP@12 almak değil, üç modelin aynı deney düzeninde karşılaştırılmasıydı. Bu yüzden ana değerlendirme AUC-ROC ve accuracy üzerinden yapıldı.

---

## Slayt 8 - Final Sonuçlar

**Ana mesaj:** Görsel geçmiş tabular baseline'dan güçlü çıktı; düzeltilmiş late-fusion rerun sonucunda multimodal model final tabloyu lider tamamladı.

**Tablo:**

| Model | AUC-ROC | Accuracy |
|---|---:|---:|
| Tabular-only MLP | 0.8550 | 0.7636 |
| Image-history MLP | 0.9237 | 0.8297 |
| Multimodal late fusion | 0.9341 | 0.8504 |

**Yorum:**

- Image-history, tabular-only modelden belirgin şekilde daha iyi.
- Late-fusion notebooku validation split düzeltmesi sonrası Kaggle'da yeniden çalıştırıldı.
- İki modalitenin birlikte kullanımı, image-history modeline göre AUC-ROC'ta 0.0104 puanlık ek katkı sağladı.

**Konuşma notu:**

Bu sonuçlar moda öneri probleminde görsel geçmişin çok güçlü bir sinyal olduğunu gösteriyor. Düzeltilmiş Kaggle rerun sonrasında late-fusion sonucu final metrik olarak kilitlendi.

---

## Slayt 9 - Açıklanabilirlik

**Ana mesaj:** Model davranışını yorumlamak için görsel benzerlik ve tabular permutation importance kullanıldı.

**Görsel açıklama:**

- Aday ürün embeddingi ile müşterinin geçmiş ürün embeddingleri karşılaştırıldı.
- En benzer geçmiş ürünler açıklama olarak raporlandı.

**Tabular açıklama:**

- Permutation importance ile hangi tabular özelliklerin daha etkili olduğu incelendi.
- En önemli sinyaller ürün kategorisi, departman, ürün tipi, yaş ve renk grubu çevresinde toplandı.

**SHAP notu:**

SHAP proposal aşamasında düşünülmüştü; fakat PyTorch embedding katmanları ve streaming preprocessing nedeniyle bu aşamada uygulanmadı. Bunun yerine daha pratik ve savunulabilir bir yöntem olarak permutation importance kullanıldı.

**Konuşma notu:**

Grad-CAM yerine görsel benzerlik analizi daha uygun oldu çünkü EfficientNet nihai karar modeli değil, embedding extractor olarak kullanıldı.

---

## Slayt 10 - Demo Sistemi

**Ana mesaj:** Final modeller canlı demo arayüzüne bağlandı.

**Demo akışı:**

1. Kullanıcı katalogdan geçmişte almış gibi ürün seçer.
2. Müşteri profilini düzenler.
3. Backend aynı aday havuzunu üç modelle skorlar.
4. Arayüz tabular-only, image-history ve late-fusion önerilerini yan yana gösterir.

**Teknik yapı:**

- Backend: FastAPI
- Frontend: React + Vite
- Modeller: Final Kaggle checkpointleri

**Konuşma notu:**

Demo, projenin bilimsel sonucunun yerine geçmiyor; ama final modellerin gerçek bir öneri arayüzüne nasıl bağlanabileceğini gösteriyor.

---

## Slayt 11 - Sınırlılıklar

**Ana mesaj:** Proje hipotezi desteklendi; ancak öneri sistemi açısından geliştirilebilecek alanlar var.

**Sınırlılıklar:**

- Ana full-training karşılaştırması AUC-ROC ve accuracy üzerinden yapıldı; MAP@12 / Precision@10 / Recall@10 ayrıca time-based ranking evaluation olarak eklendi.
- EfficientNet-B0 fine-tune edilmedi; sabit feature extractor olarak kullanıldı.
- Negatif örnekleme stratejisi performansı etkileyebilir.
- Tabular importance analizi full Kaggle modeli yerine controlled/local model üzerinde yapıldı.
- SHAP analizi ileriki çalışma olarak bırakıldı.

**Konuşma notu:**

Bu sınırlılıklar sonucu zayıflatmıyor; daha çok projenin sonraki aşamada nasıl genişletilebileceğini gösteriyor.

---

## Slayt 12 - Sonuç

**Ana mesaj:** Görsel temsil, moda önerilerinde güçlü bir kişiselleştirme sinyalidir.

**Sonuç cümlesi:**

Bu çalışmada EfficientNet-B0 görsel embeddingleri ve müşteri görsel geçmişinin tabular baseline'a göre belirgin performans artışı sağladığı gösterilmiştir. Düzeltilmiş late-fusion Kaggle rerun sonucu 0.9341 AUC-ROC ve 0.8504 accuracy ile final multimodal sonucu tamamlamıştır.

**Kapanış mesajı:**

Moda önerilerinde müşterinin ne aldığı kadar, aldığı ürünlerin nasıl göründüğü de önemlidir.

**Konuşma notu:**

Son slaytta ana fikre dön: görsel temsil, tabular veriyle birleştiğinde daha güçlü ve daha açıklanabilir bir öneri sistemi ortaya çıkıyor.

---

## Kısa Sunum Akışı

Zaman kısıtı varsa 12 slayt yerine şu 8 slaytlık kısa versiyon kullanılabilir:

1. Kapak
2. Problem ve hipotez
3. Veri seti
4. Görsel embedding ve müşteri profili
5. Üç model mimarisi
6. Final sonuçlar
7. Açıklanabilirlik ve demo
8. Sonuç

## Sunumda Vurgulanacak Ana Cümleler

- “Bu proje leaderboard optimizasyonundan çok multimodal öneri hipotezini test eder.”
- “Tabular-only model metadata baseline’ıdır.”
- “Image-history model, müşterinin geçmiş ürünlerinden görsel zevk profili çıkarır.”
- “Late fusion modeli iki modaliteyi birleştirir; final sonucu 0.9341 AUC-ROC ve 0.8504 accuracy ile kilitlendi.”
- “MAP@12 / Precision@10 / Recall@10 metrikleri ayrı time-based ranking evaluation olarak eklendi.”
- “Hybrid reranking ana model değil, post-ranking iyileştirme deneyidir.”
- “SHAP uygulanmadı; mevcut mimariye daha uygun olarak görsel benzerlik ve permutation importance kullanıldı.”
- “Demo, final Kaggle checkpointlerinin canlı inference arayüzüne bağlanmış halidir.”
