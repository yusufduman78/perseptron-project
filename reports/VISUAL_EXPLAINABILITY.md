# Visual Explainability

Final V2 tesliminde görsel açıklanabilirlik iki seviyede ele alınır.

## 1. Demo Seviyesi: Visual Similarity

Demo önerilerinde `image_history` ve `late_fusion` modelleri, müşterinin geçmiş ürün görsellerinden oluşturulan görsel stil profilini kullanır. Bu nedenle öneri kartlarında şu açıklama sinyalleri gösterilir:

- aday ürünün geçmiş ürünlere görsel benzerliği,
- en yakın geçmiş ürün,
- ürün tipi/kategori/renk uyumu,
- görsel geçmiş etiketi.

Bu açıklamalar modelin tüm kararını matematiksel olarak açmaz; fakat sunum sırasında multimodal fikri görünür hale getirir.

## 2. CNN Seviyesi: Grad-CAM

Proposal’daki image-only görsel baseline final V2’de `image_only_effnet_cnn` olarak eğitilmiştir. Bu CNN demo recommender değildir; classification baseline ve Grad-CAM kaynağıdır.

Grad-CAM örnekleri şu klasörde tutulur:

```text
reports/proposal_v2/gradcam_examples/
```

Bu görseller, EfficientNet-B0 modelinin ürün fotoğrafında hangi bölgelere odaklandığını nitel olarak gösterir. Final raporda bu çıktı, “görsel model ürünün kendisinden sinyal alıyor mu?” sorusuna destekleyici açıklama olarak kullanılır.

## Kullanım Notu

Rapor ve sunumda Grad-CAM, late fusion kararını doğrudan açıklıyor gibi sunulmamalıdır. Grad-CAM image-only CNN baseline’a aittir. Late fusion için demo tarafındaki açıklama, embedding-space visual similarity ve metadata match chip’leri üzerinden yapılır.
