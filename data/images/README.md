# data/images

Bu klasör H&M ürün fotoğraflarını barındırır.

## Alt Klasör

- `hm_images/`: Kaggle H&M veri setindeki ürün görsellerinin yerel kopyası.

## Projedeki Rolü

Ürün görselleri EfficientNet-B0 ile embedding üretmek için kullanılmıştır. Final Kaggle eğitimlerinde model doğrudan resimleri tekrar okumaz; önceden çıkarılmış embeddingleri kullanır. Ancak görseller, demo arayüzünde ürün kartlarını gerçek fotoğraflarla göstermek için değerlidir.

## Dikkat Edilmesi Gerekenler

Bu klasör büyük boyutludur. Tüm görsellerin saklanması arayüz ve görsel inceleme için faydalıdır; fakat sadece model metriklerini yeniden üretmek için her zaman gerekli değildir. Final model eğitimi için embedding dosyaları yeterlidir.

