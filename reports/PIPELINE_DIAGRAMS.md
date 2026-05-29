# Pipeline ve Model Diyagramları

Bu dosya, final rapora veya sunuma eklenebilecek Mermaid diyagramlarını içerir. Diyagramların amacı projenin üç ana modelini, veri akışını ve demo inference sürecini görsel olarak özetlemektir.

## 1. Genel Proje Pipeline'ı

```mermaid
flowchart LR
    A["H&M CSV verileri"] --> B["transactions_train.csv"]
    A --> C["customers.csv"]
    A --> D["articles.csv"]
    E["Ürün görselleri"] --> F["EfficientNet-B0 feature extractor"]
    F --> G["1280 boyutlu ürün embeddingleri"]

    B --> H["Pozitif müşteri-ürün çiftleri"]
    H --> I["Negatif örnekleme"]
    C --> J["Müşteri metaverisi"]
    D --> K["Ürün metaverisi"]
    G --> L["Müşteri görsel profili"]

    J --> M["Tabular-only MLP"]
    K --> M

    G --> N["Image-history MLP"]
    L --> N

    J --> O["Late fusion tabular branch"]
    K --> O
    G --> P["Late fusion visual branch"]
    L --> P
    O --> Q["Fusion head"]
    P --> Q

    M --> R["Satın alma skoru"]
    N --> R
    Q --> R
```

## 2. Üç Modelin Girdi Farkı

```mermaid
flowchart TB
    subgraph T["Tabular-only MLP"]
        T1["Müşteri metaverisi"] --> T3["Tabular MLP"]
        T2["Ürün metaverisi"] --> T3
        T3 --> T4["Skor"]
    end

    subgraph I["Image-history MLP"]
        I1["Aday ürün embeddingi"] --> I4["Visual MLP"]
        I2["Müşteri görsel profili"] --> I4
        I3["Cosine similarity + geçmiş uzunluğu"] --> I4
        I4 --> I5["Skor"]
    end

    subgraph F["Multimodal late fusion"]
        F1["Müşteri + ürün metaverisi"] --> F3["Tabular branch"]
        F2["Aday embedding + görsel profil"] --> F4["Visual branch"]
        F3 --> F5["Fusion head"]
        F4 --> F5
        F5 --> F6["Skor"]
    end
```

## 3. Demo Inference Akışı

```mermaid
sequenceDiagram
    participant U as Kullanıcı
    participant UI as React demo arayüzü
    participant API as FastAPI backend
    participant M as Final Kaggle modelleri
    participant D as Veri ve embedding dosyaları

    U->>UI: Geçmiş ürünleri seçer
    U->>UI: Müşteri profilini düzenler
    UI->>API: /api/recommend isteği
    API->>D: Ürün metadata ve embeddingleri yükler
    API->>API: Aday ürün havuzu oluşturur
    API->>M: Tabular-only skorları
    API->>M: Image-history skorları
    API->>M: Late-fusion skorları
    M-->>API: Üç modelin sıralı önerileri
    API-->>UI: JSON öneri listeleri
    UI-->>U: Önerileri yan yana gösterir
```

## 4. Rapor İçin Kısa Açıklama

Bu diyagramlar, çalışmanın bir Kaggle leaderboard optimizasyonundan çok multimodal recommendation hipotezini test eden deneysel bir sistem olduğunu vurgular. Tabular-only ve image-history modelleri tek modaliteli baseline olarak konumlanırken, late fusion modeli iki sinyali birleştiren ana modeldir. Demo arayüzü ise final checkpointlerin gerçek inference ortamında nasıl kullanılabileceğini göstermektedir.
