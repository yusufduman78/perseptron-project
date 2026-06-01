# Perseptron React Frontend

Bu klasör final demo arayüzünün React + Vite uygulamasıdır. Arayüz aydınlık fashion marketplace düzeninde çalışır: kullanıcı geçmiş ürün seçer, hazır senaryo yükler ve üç öneri modelini karşılaştırır.

## Çalıştırma

Önce backend açık olmalıdır:

```powershell
python ..\backend\app.py
```

Frontend:

```powershell
npm install
npm run dev -- --host 127.0.0.1
```

Tarayıcı:

```text
http://127.0.0.1:5173
```

## API

Frontend şu backend adresini kullanır:

```text
http://127.0.0.1:8000
```

Endpointler:

- `GET /api/health`
- `GET /api/catalog`
- `GET /api/demo-scenarios`
- `POST /api/recommend`
- `GET /api/metrics`
- `GET /api/explainability`

## Not

UI içinde `tabular_only`, `image_history` ve `late_fusion` önerileri gösterilir. CNN ayrı recommendation modeli değildir; final rapor ve sunumda image-only baseline olarak kullanılır.
