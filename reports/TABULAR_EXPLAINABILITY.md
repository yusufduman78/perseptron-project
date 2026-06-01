# Tabular Explainability

Final V2 tesliminde tabular açıklanabilirlik için SHAP feature summary üretilmiştir. Çıktı dosyası:

```text
reports/proposal_v2/proposal_v2_shap_summary.csv
```

## Öne Çıkan Özellikler

SHAP summary’de en etkili özellikler arasında şunlar yer alır:

- `age`
- `section_no`
- `product_group_name`
- `colour_group_code`
- `FN`
- `garment_group_name`
- `index_group_name`

Bu sonuç alan bilgisiyle uyumludur. Müşteri yaşı ve fashion-news durumu gibi müşteri sinyalleri ile ürün kategori/renk/section bilgileri satın alma tahmininde belirgin rol oynar.

## Rapor Kullanımı

SHAP çıktısı tabular/model metadata tarafını açıklamak için kullanılmalıdır. Görsel taraf için Grad-CAM ve visual similarity açıklamaları ayrı tutulur. Sunumda kısa mesaj şu olabilir:

> Metadata tarafında yaş ve ürün kategori/renk alanları öne çıkıyor; görsel tarafta CNN Grad-CAM ve demo visual-similarity chip’leri kullanıldı.
