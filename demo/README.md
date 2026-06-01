# Perseptron Demo

Final demo V2 Kaggle çıktılarına bağlı çalışan FastAPI + React uygulamasıdır.

## Modeller

Demo üç recommendation modelini gösterir:

- `tabular_only`
- `image_history`
- `late_fusion`

`image_only_effnet_cnn` demo tab’i değildir; rapor/sunumda image-only baseline ve Grad-CAM kaynağı olarak kullanılır. `late_fusion_hybrid_*` ise ana model değil, reranking/post-ranking deneyidir.

## Backend

```powershell
conda run -n perseptron python demo\backend\test_recommend.py --top-k 3 --candidate-limit 200
python demo\backend\app.py
```

Backend endpointleri:

- `GET /api/health`
- `GET /api/catalog`
- `GET /api/demo-scenarios`
- `POST /api/recommend`
- `GET /api/metrics`
- `GET /api/explainability`

## Frontend

```powershell
cd demo\frontend
npm install
npm run dev -- --host 127.0.0.1
```

Tarayıcı adresi:

```text
http://127.0.0.1:5173
```

## Beklenen Demo Akışı

Hazır senaryo seçilir, style bag geçmiş ürünlerle dolar, öneriler üretilir ve üç modelin sonuçları karşılaştırılır. Ekranda model overlap/unique öneriler, explanation chip’leri, final metrics ve explainability özeti görünür.
