# data/embeddings

Bu klasör ürün görsellerinden çıkarılmış vektör temsillerini içerir. Embeddingler EfficientNet-B0 pretrained modeliyle çıkarılmıştır.

## Alt Klasörler

- `final_kaggle/`: Final Kaggle full-training deneylerinde kullanılan 105,100 ürünlük embedding seti.
- `local_sandbox/`: Daha önce local faz deneylerinde kullanılan 52,833 ürünlük embedding seti.

## Neden Embedding Kullanıldı?

Ürün fotoğraflarını her eğitim adımında CNN'den geçirmek çok pahalıdır. Bunun yerine EfficientNet-B0 ile her ürün için bir kez 1280 boyutlu embedding çıkarılmıştır. Sonraki MLP ve late-fusion eğitimleri bu embeddingleri kullanır.

Bu yaklaşım iki avantaj sağlar:

- Eğitim hızlanır.
- Görsel bilgi bütün modellerde tutarlı biçimde kullanılır.

## Önemli Ayrım

Final raporda ve Kaggle full-training sonuçlarında `final_kaggle/` altındaki embedding seti esas alınmalıdır. `local_sandbox/` eski local deneylerin tekrar çalıştırılabilmesi için saklanmıştır.

