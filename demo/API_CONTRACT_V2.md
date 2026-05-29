# Demo API Contract v2

V2 demo, Kaggle sonuclari geldikten sonra su minimum endpointlerle kurulacak.

## `GET /api/v2/health`

Doner:

- API status
- V2 checkpoint dosyalari var mi
- V2 ranking summary var mi
- device bilgisi

## `GET /api/v2/metrics`

Doner:

- MAP@12 / Precision@10 / Recall@10 ana tablo
- AUC-ROC / accuracy destekleyici tablo
- hybrid reranking destekleyici tablo

## `POST /api/v2/recommend`

Girdi:

- `history_article_ids`
- `customer_profile`
- `model_names`
- `top_k`

Cikti:

- tabular-only onerileri
- image-history onerileri
- late-fusion onerileri
- explanation chipleri

## `GET /api/v2/explainability`

Doner:

- SHAP/permutation feature summary
- Grad-CAM ornek dosya listesi
- visual similarity ornekleri
