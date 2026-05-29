# Perseptron Demo

Bu klasor, Perseptron projesinin final model checkpointlerini canli kullanabilen demo katmanini icerir. Amac, kullanicinin gecmiste satin aldigi urunleri secip uc farkli modelin onerilerini yan yana gorebilmesidir.

Demo iki parcadan olusur:

- `backend/`: FastAPI tabanli inference servisi.
- `frontend/`: React + Vite tabanli fake alisveris arayuzu.

## Kullanilan Modeller

Backend ayni anda uc final Kaggle checkpointini yukler:

- `models/final_kaggle/tabular_only_streaming_full.pt`
- `models/final_kaggle/image_history_streaming_full.pt`
- `models/final_kaggle/multimodal_fusion_streaming_full.pt`

Bu sayede ayni gecmis urun listesi ve ayni musteri profili icin su uc cikti karsilastirilabilir:

- Tabular-only onerileri
- Image-history onerileri
- Late-fusion onerileri

## Backend Calistirma

Proje ana dizininden:

```powershell
python demo\backend\app.py
```

Backend varsayilan olarak su adreste acilir:

```text
http://127.0.0.1:8000
```

Hizli saglik kontrolu:

```powershell
Invoke-WebRequest http://127.0.0.1:8000/api/health
```

## Frontend Calistirma

Ayri bir terminalde:

```powershell
cd demo\frontend
npm install
npm run dev -- --host 127.0.0.1
```

Arayuz:

```text
http://127.0.0.1:5173
```

## Terminalden Model Testi

Web arayuzu olmadan uc modeli test etmek icin:

```powershell
python demo\backend\test_recommend.py
```

Ornek gecmis urunleri elle vermek icin:

```powershell
python demo\backend\test_recommend.py --history 0108775015 0108775044 0521269001 --age 24
```

JSON cikti almak icin:

```powershell
python demo\backend\test_recommend.py --json-output demo\sample_recommendations.json
```

## Demo Mantigi

Kullanici arayuzunde secilen urunler "gecmis satin alma" gibi ele alinir. Bu urunlerden image-history ve late-fusion modelleri icin bir gorsel tercih profili olusturulur. Tabular-only model ise gecmis urun embeddinglerini dogrudan kullanmaz; musteri metaverisi ve aday urun metaverisi uzerinden skor uretir.

Backend aday urun havuzunu populer urunler, secilen gecmis urunlere metadata olarak benzeyen urunler ve gorsel embedding yakinlariyla olusturur. Sonra uc model ayni aday havuzunu skorlar ve sonuclar frontendde yan yana gosterilir.

## Dogrulama

Bu demo final Kaggle checkpointleriyle test edilmistir. API testinde uc model de ayni istek icin oneriler dondurmustur. Ilk onerme istegi modelleri ve embeddingleri bellekte hazirladigi icin sonraki isteklere gore daha yavas olabilir.
