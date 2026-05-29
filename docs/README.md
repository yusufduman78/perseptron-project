# docs

Bu klasör proje dokümanlarını içerir.

## Dosyalar

- `yusuf_duman_22118080056_proposal.docx`: Projenin başlangıç proposal dokümanı.

## Proposal Özeti

Proposal'da ana hedef, H&M veri seti üzerinde görsel ürün bilgisi ile tabular müşteri/ürün bilgisini birleştiren multimodal bir öneri modeli kurmaktır.

Seçilen temel model aileleri:

- Görsel taraf: EfficientNet-B0 / CNN tabanlı ürün embeddingleri.
- Tabular taraf: MLP veya XGBoost baseline; final deneylerde MLP kullanılmıştır.
- Ana model: EfficientNet görsel embeddingleri + MLP tabular branch + late fusion.
- Yorumlanabilirlik hedefi: SHAP; alternatif olarak görsel taraf için Grad-CAM.

## Proje ile Uyum

Final deneyler proposal ile uyumludur. Üç ana karşılaştırma yapılmıştır:

- Tabular-only MLP
- Image-history MLP
- Multimodal late fusion

Bu karşılaştırma proposal'daki unimodal baseline ve multimodal hipotez yapısını doğrudan test eder.

