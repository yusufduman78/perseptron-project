# src

Bu klasör projenin Python kaynak kodlarını içerir.

## Alt Klasörler

- `local_phases/`: Local geliştirme, sandbox veri hazırlama ve ablation scriptleri.
- `kaggle/`: Kaggle için hazırlanmış pipeline scripti.
- `utils/`: Yardımcı scriptler.
- `explainability/`: Görsel benzerlik ve feature importance gibi açıklanabilirlik analizleri.
- `evaluation/`: Final modeller için MAP@12, Precision@10 ve Recall@10 gibi ranking metrikleri.

## Projedeki Rolü

Kaggle notebookları final deneyleri anlatır ve çalıştırır. `src/` klasörü ise bu deneylere gelmeden önce yapılan local geliştirme adımlarını, yardımcı pipeline kodlarını ve açıklanabilirlik analizlerini saklar.

## Çalıştırma Notu

Local scriptler yeni klasör yapısına göre güncellenmiştir:

- Sandbox verisi: `data/processed/sandbox/`
- Local embedding cache: `data/embeddings/local_sandbox/`
- Local model çıktıları: `models/local_experiments/`
- Rapor çıktıları: `reports/`

Final Kaggle sonuçlarını yeniden üretmek için öncelik `notebooks/kaggle/` olmalıdır.
