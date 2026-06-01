# Demo API Contract V2

Final demo API sözleşmesi backend içinde `/api/*` endpointleriyle uygulanır.

## `GET /api/health`

Dönen bilgiler:

- API status
- CUDA/device bilgisi
- V2 checkpoint dosyaları var mı
- articles, transactions, embeddings ve image directory erişilebilir mi
- classification/ranking/explainability artifactleri var mı

## `GET /api/catalog`

Query:

- `limit`
- `q`

Çıktı:

- ürün metadata alanları
- `image_url`

## `GET /api/demo-scenarios`

Hazır demo senaryoları:

- Black Basics
- Dress Style
- Denim Casual
- Bright Knit

## `POST /api/recommend`

Girdi:

- `history_article_ids`
- `customer_profile`
- `top_k`
- `candidate_limit`

Çıktı:

- `models.tabular_only`
- `models.image_history`
- `models.late_fusion`
- `comparison`
- `request_context`
- öneri bazında explanation metadata

## `GET /api/metrics`

Çıktı:

- classification metrics
- image-only CNN metrics
- ranking leaderboard
- ranking summary markdown
- hybrid/reranking kapsam notu

## `GET /api/explainability`

Çıktı:

- SHAP top features
- Grad-CAM örnek listesi
- explainability yorum notları

## Not

`late_fusion_hybrid_*` ana demo modeli değildir. CNN de recommendation tab’i değildir; rapor/sunumda image-only baseline ve Grad-CAM kaynağı olarak kullanılır.
