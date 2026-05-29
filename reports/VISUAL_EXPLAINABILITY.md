# Görsel Benzerlik Tabanlı Açıklanabilirlik Analizi

Bu analiz, önerilen veya aday gösterilen bir ürünün müşterinin geçmiş satın alımlarıyla görsel olarak nasıl ilişkilendiğini açıklamak için hazırlanmıştır. Projedeki EfficientNet-B0 modeli doğrudan sınıflandırıcı olarak fine-tune edilmediği için klasik Grad-CAM yerine embedding uzayında görsel benzerlik analizi kullanılmıştır.

## Yöntem

Her ürün için daha önce EfficientNet-B0 ile çıkarılmış 1280 boyutlu görsel embeddingler kullanılır. Bir aday ürün açıklanırken şu adımlar izlenir:

1. Aday ürünün embedding vektörü alınır.
2. Aynı müşterinin geçmişte satın aldığı ürünlerin embedding vektörleri alınır.
3. Aday ürün ile geçmiş ürünler arasında cosine similarity hesaplanır.
4. En yüksek benzerliğe sahip geçmiş ürünler aday ürünün görsel açıklaması olarak raporlanır.

Bu analiz modelin görsel branch mantığıyla uyumludur. Çünkü image-history ve late fusion modelleri de aday ürün embeddingini müşterinin geçmiş görsel profiliyle ilişkilendirerek skor üretmektedir.

## Üretilen Çıktılar

Script:

```powershell
python src/explainability/visual_similarity_explanations.py
```

Bu komut aşağıdaki çıktıları üretir:

- `reports/visual_similarity_examples.csv`
  - Aday ürün, müşterinin benzer geçmiş ürünleri ve cosine similarity skorları.
- `reports/visual_similarity_examples.html`
  - Aynı örneklerin ürün görselleriyle birlikte okunabilir HTML raporu.

## Rapor İçin Yorum

Görsel benzerlik analizi, öneri modelinin görsel sinyalini kullanıcı tarafından anlaşılabilir hale getirir. Bir ürünün önerilme nedeni yalnızca sayısal skor olarak değil, müşterinin geçmişte satın aldığı görsel olarak benzer ürünler üzerinden açıklanabilir. Örneğin aday ürün bir sandalet ise, müşterinin geçmişinde benzer form ve renkte sandaletlerin bulunması modelin görsel geçmiş sinyalini destekleyen bir açıklama sağlar.

Bu yaklaşım özellikle moda öneri sistemleri için uygundur. Çünkü kullanıcı tercihleri çoğu zaman ürün tipi, renk, kesim, stil ve görsel kompozisyon gibi doğrudan ürün görselinde bulunan sinyallerle ilişkilidir. EfficientNet-B0 embeddingleri bu görsel bilgiyi vektör uzayında temsil ettiği için, cosine similarity üzerinden en yakın geçmiş ürünleri göstermek modelin karar mantığını sezgisel olarak açıklanabilir hale getirir.

## Grad-CAM ile İlişkisi

Proposal'da Grad-CAM benzeri görsel açıklanabilirlik fikri yer almıştır. Ancak bu projede EfficientNet-B0 modeli son karar katmanıyla birlikte sınıflandırıcı olarak eğitilmemiş, sabit feature extractor olarak kullanılmıştır. Bu nedenle Grad-CAM üretmek teknik olarak mümkün olsa bile modelin nihai öneri skorunu doğrudan açıklamak açısından sınırlı kalır.

Bu projede daha uygun görsel açıklanabilirlik yaklaşımı, embedding uzayında müşteri geçmişindeki en benzer ürünleri göstermektir. Böylece açıklama doğrudan image-history ve late fusion modellerinin kullandığı görsel temsil mantığına dayanır.

## Sınırlılık

Bu analiz modelin nihai skorunu tüm bileşenleriyle açıklamaz. Özellikle late fusion modelinde tabular branch de karara katkı verir. Bu nedenle görsel benzerlik analizi sadece görsel modalitenin açıklamasıdır. Tabular katkıyı yorumlamak için ayrıca feature importance veya SHAP/permutation importance analizi eklenmelidir.
