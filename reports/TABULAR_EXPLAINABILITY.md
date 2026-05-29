# Tabular Feature Importance Analizi

Bu analiz, controlled/local tabular-only MLP modelinin müşteri ve ürün metaverisi içindeki hangi alanlara daha duyarlı olduğunu yorumlamak için hazırlanmıştır. SHAP doğrudan PyTorch MLP ve mevcut streaming eğitim düzenine bağlanabildiği halde daha maliyetli ve karmaşık olacağı için, bu aşamada daha pratik ve savunulabilir bir yöntem olan permutation importance kullanılmıştır.

Önemli not: Bu analiz Kaggle full-training tabular modelinin birebir açıklaması değildir. Local controlled ablation sırasında eğitilmiş tabular-only checkpoint üzerinde yapılmıştır. Bu nedenle raporda final performans iddiası için değil, tabular branch'in genel olarak hangi özellik türlerinden sinyal aldığını göstermek için destekleyici açıklanabilirlik analizi olarak kullanılmalıdır.

SHAP proposal aşamasında hedeflenen açıklanabilirlik yöntemlerinden biridir; ancak mevcut PyTorch MLP yapısında kategorik alanların embedding katmanlarından geçmesi ve full-training pipeline'ının streaming preprocessing düzeni SHAP entegrasyonunu daha maliyetli hale getirmiştir. Bu nedenle bu aşamada SHAP yerine permutation importance kullanılmıştır. SHAP analizi ileriki çalışma olarak ayrıca ele alınabilir.

## Yöntem

Permutation importance yönteminde validation verisi üzerinde önce modelin temel performansı ölçülür. Daha sonra her özellik tek tek karıştırılır ve modelin performansındaki düşüş hesaplanır. Bir özellik karıştırıldığında AUC-ROC belirgin şekilde düşüyorsa, model o özellikten daha fazla yararlanıyor demektir.

Bu analiz için controlled/local tabular-only MLP checkpointi kullanılmıştır:

```text
models/local_experiments/controlled_tabular_only_mlp.pt
```

Script:

```powershell
python src/explainability/tabular_permutation_importance.py
```

Üretilen dosyalar:

- `reports/tabular_permutation_importance.csv`
- `reports/tabular_feature_group_importance.csv`

## Örnek Çalıştırma Sonucu

Hızlı analiz için validation setinden 20,000 satırlık örneklem kullanılmıştır. Bu örneklemde baseline performans yaklaşık olarak şu şekildedir:

| Metrik | Değer |
|---|---:|
| AUC-ROC | 0.7254 |
| Accuracy | 0.6677 |

En yüksek AUC düşüşü oluşturan özellikler:

| Özellik | Tip | Grup | AUC Düşüşü |
|---|---|---|---:|
| `section_name` | kategorik | ürün kategorik | 0.0471 |
| `department_name` | kategorik | ürün kategorik | 0.0236 |
| `product_type_name` | kategorik | ürün kategorik | 0.0176 |
| `age` | sayısal | müşteri sayısal | 0.0152 |
| `product_code` | sayısal | ürün sayısal | 0.0138 |
| `colour_group_name` | kategorik | ürün kategorik | 0.0130 |

## Grup Bazlı Yorum

Özellik grupları toplam AUC düşüşüne göre incelendiğinde ürün kategorik özelliklerinin en güçlü tabular sinyal olduğu görülmektedir.

| Özellik Grubu | Toplam AUC Düşüşü | Ortalama AUC Düşüşü |
|---|---:|---:|
| Ürün kategorik özellikleri | 0.1562 | 0.0130 |
| Ürün sayısal özellikleri | 0.0197 | 0.0020 |
| Müşteri sayısal özellikleri | 0.0152 | 0.0051 |
| Müşteri kategorik özellikleri | 0.0005 | 0.0003 |

Bu sonuç, tabular-only modelin en çok ürünün bölüm, departman, ürün tipi, renk ve görünüm gibi ürün tanımlayıcı metaverilerinden yararlandığını göstermektedir. Müşteri tarafında ise en belirgin sinyal `age` değişkeninden gelmektedir. Müşteri kategorik alanlarının etkisi bu deneyde sınırlı kalmıştır.

## Rapor İçin Yorum

Tabular açıklanabilirlik analizi, modelin satın alma olasılığı skorunu üretirken özellikle ürün kategorisi ve ürün tanımlayıcı alanlarına duyarlı olduğunu göstermektedir. `section_name`, `department_name`, `product_type_name`, `colour_group_name` ve `garment_group_name` gibi alanlar karıştırıldığında AUC-ROC değerinde belirgin düşüşler oluşmuştur. Bu durum, tabular branch'in müşteriye önerilecek ürünleri ayırt ederken ürünün ait olduğu kategori, departman ve görsel/renk tanımları gibi yapısal bilgileri kullandığını göstermektedir.

Müşteri tarafında `age` değişkeni anlamlı bir katkı sağlamıştır. Buna karşılık `club_member_status` ve `fashion_news_frequency` gibi müşteri kategorik özelliklerinin etkisi daha sınırlı görünmektedir. Bu sonuç, H&M veri setinde ürün metaverisinin tabular model açısından müşteri üyelik bilgilerine göre daha güçlü bir ayırt edici sinyal taşıdığını düşündürmektedir.

## Sınırlılık

Bu analiz controlled tabular-only model üzerinde yapılmıştır. Dolayısıyla Kaggle full-training tabular modelinin veya late fusion modelinin tüm karar mekanizmasını açıklamaz. Late fusion modelinde görsel branch de skora katkı verir. Ancak tabular branch'in hangi özellik türlerinden yararlandığını anlamak için bu analiz raporda destekleyici açıklanabilirlik çıktısı olarak kullanılabilir.
