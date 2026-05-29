# Perseptron React Frontend

Bu klasor, Perseptron demo arayuzunun React + Vite uygulamasidir. Arayuz kullanicinin urun katalogundan gecmis satin alma urunleri secmesini, basit musteri metadata alanlarini duzenlemesini ve uc final modelin onerilerini yan yana gormesini saglar.

## Calistirma

Once backendin acik oldugundan emin olun:

```powershell
python ..\backend\app.py
```

Frontend icin:

```powershell
npm install
npm run dev -- --host 127.0.0.1
```

Tarayicida:

```text
http://127.0.0.1:5173
```

## Beklenen API

Frontend su backend adresini kullanir:

```text
http://127.0.0.1:8000
```

Ana endpointler:

- `GET /api/catalog`: demo katalog urunlerini getirir.
- `GET /api/customer-defaults`: varsayilan musteri profili degerlerini getirir.
- `POST /api/recommend`: secili gecmis urunler ve musteri profili icin uc modelden oneri alir.

## Not

Bu arayuz bir Kaggle leaderboard arayuzu degildir. Ders projesindeki bilimsel karsilastirmayi daha anlasilir gostermek icin hazirlanmis interaktif bir demonstrasyondur.
