import fs from "node:fs/promises";
import path from "node:path";

import {
  Presentation,
  PresentationFile,
  row,
  column,
  grid,
  panel,
  text,
  image,
  chart,
  rule,
  fill,
  fixed,
  hug,
  wrap,
  grow,
  fr,
} from "@oai/artifact-tool";

const projectRoot = "C:/Users/Yusuf/Desktop/perseptron_project";
const outDir = `${projectRoot}/reports`;
const previewDir = `${projectRoot}/reports/presentation_previews`;
const pptxPath = `${outDir}/perseptron_multimodal_recommendation_presentation.pptx`;

const W = 1920;
const H = 1080;

const C = {
  bg: "#090910",
  bg2: "#11111B",
  panel: "#171622",
  panel2: "#211A2C",
  text: "#F8F4EF",
  muted: "#B9B3C6",
  soft: "#7C748D",
  line: "#2B2938",
  pink: "#F472B6",
  pink2: "#F9A8D4",
  gold: "#FBBF24",
  teal: "#5EEAD4",
  blue: "#93C5FD",
  green: "#34D399",
  red: "#FCA5A5",
};

const img = {
  denim: `${projectRoot}/data/images/hm_images/070/0706016001.jpg`,
  tank: `${projectRoot}/data/images/hm_images/056/0565379001.jpg`,
  blackTop: `${projectRoot}/data/images/hm_images/075/0759871002.jpg`,
  trousers: `${projectRoot}/data/images/hm_images/039/0399256001.jpg`,
};

async function imageDataUrl(filePath) {
  const bytes = await fs.readFile(filePath);
  const ext = path.extname(filePath).toLowerCase();
  const mime = ext === ".png" ? "image/png" : "image/jpeg";
  return `data:${mime};base64,${bytes.toString("base64")}`;
}

const imgData = {
  denim: await imageDataUrl(img.denim),
  tank: await imageDataUrl(img.tank),
  blackTop: await imageDataUrl(img.blackTop),
  trousers: await imageDataUrl(img.trousers),
};

const deck = Presentation.create({
  slideSize: { width: W, height: H },
});
const slideRefs = [];

const titleStyle = { fontFace: "Georgia", fontSize: 68, bold: true, color: C.text };
const h2Style = { fontFace: "Aptos Display", fontSize: 46, bold: true, color: C.text };
const bodyStyle = { fontFace: "Aptos", fontSize: 27, color: C.muted };
const smallStyle = { fontFace: "Aptos", fontSize: 18, color: C.soft };
const labelStyle = { fontFace: "Aptos", fontSize: 16, bold: true, color: C.pink, letterSpacing: 1.6 };

function tx(value, options = {}) {
  return text(value, {
    width: options.width ?? fill,
    height: hug,
    style: options.style ?? bodyStyle,
    name: options.name,
  });
}

function root(child, fillColor = C.bg) {
  return panel(
    {
      name: "slide-bg",
      width: fill,
      height: fill,
      fill: fillColor,
      padding: { x: 72, y: 58 },
    },
    child,
  );
}

function add(slideNode) {
  const slide = deck.slides.add();
  slide.compose(slideNode, {
    frame: { left: 0, top: 0, width: W, height: H },
    baseUnit: 8,
  });
  slideRefs.push(slide);
  return slide;
}

function footer(n, label = "Perseptron · H&M Multimodal Recommendation") {
  return row(
    { width: fill, height: hug, align: "center", justify: "between" },
    [
      tx(label, { style: { ...smallStyle, fontSize: 15 } }),
      tx(String(n).padStart(2, "0"), {
        width: fixed(50),
        style: { ...smallStyle, fontSize: 15, bold: true, color: C.pink2 },
      }),
    ],
  );
}

function titleBlock(eyebrow, title, subtitle) {
  return column(
    { width: fill, height: hug, gap: 18 },
    [
      tx(eyebrow.toUpperCase(), { style: labelStyle }),
      tx(title, { style: h2Style }),
      subtitle ? tx(subtitle, { width: wrap(1220), style: bodyStyle }) : null,
    ].filter(Boolean),
  );
}

function bulletList(items, accent = C.pink) {
  return column(
    { width: fill, height: hug, gap: 18 },
    items.map((item) =>
      row(
        { width: fill, height: hug, gap: 14, align: "start" },
        [
          text("•", {
            width: fixed(26),
            height: hug,
            style: { fontFace: "Aptos", fontSize: 30, bold: true, color: accent },
          }),
          tx(item, { style: { ...bodyStyle, fontSize: 25, color: C.text } }),
        ],
      ),
    ),
  );
}

function stat(label, value, color = C.teal) {
  return column(
    { width: fill, height: hug, gap: 8 },
    [
      tx(label.toUpperCase(), { style: { ...smallStyle, bold: true, letterSpacing: 1.1 } }),
      tx(value, { style: { fontFace: "Aptos Display", fontSize: 42, bold: true, color } }),
    ],
  );
}

function softPanel(child, opts = {}) {
  return panel(
    {
      width: opts.width ?? fill,
      height: opts.height ?? hug,
      fill: opts.fill ?? C.panel,
      line: opts.line ?? { color: C.line, width: 1 },
      borderRadius: 20,
      padding: opts.padding ?? { x: 28, y: 24 },
    },
    child,
  );
}

function simpleTable(headers, rows, widths) {
  const tableRows = [
    row(
      { width: fill, height: hug, gap: 0 },
      headers.map((h, i) =>
        panel(
          { width: fixed(widths[i]), height: hug, fill: C.panel2, padding: { x: 14, y: 12 } },
          tx(h, { style: { ...smallStyle, bold: true, color: C.pink2 } }),
        ),
      ),
    ),
    ...rows.map((r, idx) =>
      row(
        { width: fill, height: hug, gap: 0 },
        r.map((cell, i) =>
          panel(
            {
              width: fixed(widths[i]),
              height: hug,
              fill: idx % 2 === 0 ? "#12121C" : "#171622",
              padding: { x: 14, y: 13 },
            },
            tx(cell, {
              style: {
                ...bodyStyle,
                fontSize: i === 0 ? 22 : 24,
                bold: i > 1,
                color: i > 1 ? C.teal : C.text,
              },
            }),
          ),
        ),
      ),
    ),
  ];
  return column({ width: hug, height: hug, gap: 2 }, tableRows);
}

function scoreBar(model, auc, accuracy, color) {
  const aucWidth = Math.round(auc * 520);
  const accWidth = Math.round(accuracy * 520);
  return column(
    { width: fixed(720), height: hug, gap: 8 },
    [
      row(
        { width: fill, height: hug, justify: "between", align: "center" },
        [
          tx(model, { style: { ...bodyStyle, fontSize: 24, color: C.text, bold: true } }),
          tx(`AUC ${auc.toFixed(4)} | Acc ${accuracy.toFixed(4)}`, {
            width: fixed(250),
            style: { ...smallStyle, fontSize: 18, color },
          }),
        ],
      ),
      panel(
        { width: fixed(560), height: fixed(13), fill: "#262333", borderRadius: 7, padding: 0 },
        panel({ width: fixed(aucWidth), height: fill, fill: color, borderRadius: 7 }),
      ),
      panel(
        { width: fixed(560), height: fixed(9), fill: "#1A1824", borderRadius: 5, padding: 0 },
        panel({ width: fixed(accWidth), height: fill, fill: C.soft, borderRadius: 5 }),
      ),
    ],
  );
}

function productImage(pathValue, width = 260, height = 360) {
  return panel(
    {
      width: fixed(width),
      height: fixed(height),
      fill: "#20202C",
      borderRadius: 24,
      padding: 0,
    },
    image({
      dataUrl: pathValue,
      width: fill,
      height: fill,
      fit: "cover",
      alt: "H&M product image",
    }),
  );
}

add(
  root(
    row(
      { width: fill, height: fill, gap: 44, align: "center" },
      [
        column(
          { width: grow(1.05), height: fill, gap: 26, justify: "center" },
          [
            tx("PERSEPTRON PROJE SUNUMU", { style: labelStyle }),
            tx("H&M Moda Ürünleri İçin Multimodal Öneri Sistemi", {
              width: wrap(920),
              style: { ...titleStyle, fontSize: 74, lineHeight: 0.95 },
            }),
            tx("Tabular metadata, ürün görselleri ve müşteri görsel geçmişiyle late fusion tabanlı satın alma skorlama modeli.", {
              width: wrap(850),
              style: { ...bodyStyle, fontSize: 29 },
            }),
            row(
              { width: fill, height: hug, gap: 18 },
              [
                stat("Ana sonuç", "Late fusion en yüksek AUC", C.teal),
                stat("Final AUC", "0.9341", C.pink2),
              ],
            ),
          ],
        ),
        row(
          { width: grow(0.95), height: fill, gap: 22, align: "center", justify: "center" },
          [
            productImage(imgData.denim, 250, 410),
            column(
              { width: fixed(260), height: hug, gap: 22 },
              [productImage(imgData.tank, 260, 330), productImage(imgData.blackTop, 260, 330)],
            ),
          ],
        ),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Problem", "Moda önerisi sadece kategori problemi değildir.", "Kullanıcının tercihleri ürün tipi kadar renk, kesim, form ve görsel stil ile de ilişkilidir."),
        row(
          { width: fill, height: fill, gap: 40, align: "center" },
          [
            column(
              { width: grow(1), height: hug, gap: 22 },
              [
                bulletList([
                  "Aynı kategori içindeki ürünler görsel olarak çok farklı olabilir.",
                  "Tabular metadata ürünü tarif eder; görsel embedding ürünün stilini temsil eder.",
                  "Müşterinin geçmiş alışverişindeki görsel karakter yeni öneriler için güçlü sinyaldir.",
                ], C.gold),
              ],
            ),
            row(
              { width: grow(0.9), height: hug, gap: 18, justify: "center" },
              [productImage(imgData.denim, 245, 350), productImage(imgData.trousers, 245, 350), productImage(imgData.tank, 245, 350)],
            ),
          ],
        ),
        footer(2),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Hipotez", "Görsel geçmiş ve tabular metadata birlikte daha güçlüdür.", "Deney tasarımı üç modeli aynı müşteri-ürün skorlama problemi üzerinde karşılaştırır."),
        grid(
          { width: fill, height: grow(1), columns: [fr(1), fr(1), fr(1)], columnGap: 24 },
          [
            softPanel(column({ width: fill, height: hug, gap: 14 }, [
              tx("Tabular-only MLP", { style: { ...h2Style, fontSize: 34 } }),
              tx("Müşteri ve ürün metaverisiyle görsel bilgi olmadan baseline.", { style: { ...bodyStyle, fontSize: 24 } }),
              tx("Rol: metadata sinyalini ölçmek", { style: { ...smallStyle, color: C.gold, bold: true } }),
            ])),
            softPanel(column({ width: fill, height: hug, gap: 14 }, [
              tx("Image-history MLP", { style: { ...h2Style, fontSize: 34 } }),
              tx("Ürün embeddingleri ve müşterinin görsel geçmiş profili.", { style: { ...bodyStyle, fontSize: 24 } }),
              tx("Rol: görsel tercih sinyalini ölçmek", { style: { ...smallStyle, color: C.teal, bold: true } }),
            ])),
            softPanel(column({ width: fill, height: hug, gap: 14 }, [
              tx("Late fusion", { style: { ...h2Style, fontSize: 34 } }),
              tx("Tabular branch ile visual branch çıktısını birleştiren ana model.", { style: { ...bodyStyle, fontSize: 24 } }),
              tx("Rol: multimodal hipotezi test etmek", { style: { ...smallStyle, color: C.pink2, bold: true } }),
            ]), { fill: "#241629", line: { color: "#6D3B71", width: 1 } }),
          ],
        ),
        footer(3),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Veri seti", "Dört veri kaynağı tek öneri probleminde birleşti.", "H&M veri seti işlem geçmişi, müşteri metaverisi, ürün metaverisi ve ürün fotoğraflarını birlikte sunar."),
        simpleTable(
          ["Veri", "Projedeki rolü"],
          [
            ["transactions_train.csv", "Pozitif müşteri-ürün çiftleri ve müşteri geçmişi"],
            ["customers.csv", "Müşteri metaverisi"],
            ["articles.csv", "Ürün metaverisi"],
            ["images/", "EfficientNet-B0 ile görsel embedding çıkarımı"],
          ],
          [430, 940],
        ),
        row(
          { width: fill, height: hug, gap: 28 },
          [
            stat("Görsel embedding", "105,100 ürün", C.pink2),
            stat("Embedding boyutu", "1280", C.teal),
            stat("Full pozitif çift", "27.19M", C.gold),
          ],
        ),
        footer(4),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Görsel temsil", "CNN her epoch tekrar çalıştırılmadı; ürünler embeddinge çevrildi.", "EfficientNet-B0 sabit feature extractor olarak kullanıldı ve her ürün için 1280 boyutlu temsil üretildi."),
        row(
          { width: fill, height: grow(1), gap: 26, align: "center" },
          [
            productImage(imgData.tank, 270, 380),
            column({ width: fixed(1010), height: hug, gap: 16 }, [
              softPanel(tx("Ürün fotoğrafı → EfficientNet-B0 → 1280 boyutlu ürün embeddingi", { style: { ...bodyStyle, fontSize: 31, color: C.text, bold: true } }), { fill: "#12121C" }),
              softPanel(tx("Müşterinin geçmiş ürün embeddingleri ortalanarak görsel tercih profili oluşturuldu.", { style: { ...bodyStyle, fontSize: 27 } }), { fill: "#151923" }),
              softPanel(tx("Aday ürün embeddingi ile müşteri profili arasındaki cosine similarity ek sinyal olarak kullanıldı.", { style: { ...bodyStyle, fontSize: 27 } }), { fill: "#1A1724" }),
            ]),
          ],
        ),
        footer(5),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 30 },
      [
        titleBlock("Model mimarileri", "Üç model aynı skoru farklı bilgi kaynaklarıyla üretir.", "Amaç, tabular ve görsel sinyallerin ayrı ayrı ve birlikte katkısını ölçmektir."),
        simpleTable(
          ["Model", "Girdi", "Çıktı"],
          [
            ["Tabular-only", "Müşteri + ürün metaverisi", "Satın alma skoru"],
            ["Image-history", "Aday embedding + müşteri görsel profili", "Satın alma skoru"],
            ["Late fusion", "Tabular branch + visual branch", "Satın alma skoru"],
          ],
          [360, 720, 330],
        ),
        softPanel(
          row({ width: fill, height: hug, gap: 22, align: "center" }, [
            tx("Late fusion", { width: fixed(210), style: { ...h2Style, fontSize: 32, color: C.pink2 } }),
            tx("metadata temsili", { width: fixed(330), style: { ...bodyStyle, fontSize: 25, color: C.text } }),
            tx("+", { width: fixed(40), style: { fontFace: "Aptos Display", fontSize: 42, bold: true, color: C.gold } }),
            tx("görsel tercih temsili", { width: fixed(380), style: { ...bodyStyle, fontSize: 25, color: C.text } }),
            tx("→ tek skor", { width: fixed(210), style: { ...bodyStyle, fontSize: 25, bold: true, color: C.teal } }),
          ]),
          { fill: "#241629", line: { color: "#6D3B71", width: 1 } },
        ),
        footer(6),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Eğitim kurulumu", "Final deneyler full veri ölçeğinde streaming eğitimle yapıldı.", "Büyük müşteri-ürün evreni tek seferde belleğe alınmadı; eğitim parçalara bölünerek yürütüldü."),
        grid(
          { width: fill, height: hug, columns: [fr(1), fr(1), fr(1), fr(1)], columnGap: 22 },
          [
            softPanel(stat("Ortam", "Kaggle GPU", C.blue)),
            softPanel(stat("Epoch", "3", C.pink2)),
            softPanel(stat("Pozitifler", "Gerçek satın alma", C.teal)),
            softPanel(stat("Metrikler", "AUC + Accuracy", C.gold)),
          ],
        ),
        bulletList([
          "Pozitif örnekler geçmiş satın alma çiftlerinden alındı.",
          "Negatif örnekler satın alınmamış müşteri-ürün çiftlerinden örneklendi.",
          "Ana amaç leaderboard optimizasyonu değil, üç modelin aynı düzende karşılaştırılmasıydı.",
        ], C.teal),
        footer(7),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 24 },
      [
        titleBlock("Final sonuçlar", "Late fusion en yüksek doğrulama performansını verdi.", "Görsel geçmiş tek başına güçlü; tabular bilgi eklendiğinde tamamlayıcı katkı sağlıyor."),
        row(
          { width: fill, height: grow(1), gap: 34, align: "center" },
          [
            column(
              { width: fixed(760), height: hug, gap: 28 },
              [
                scoreBar("Tabular-only MLP", 0.8550, 0.7636, C.blue),
                scoreBar("Image-history MLP", 0.9237, 0.8297, C.teal),
                scoreBar("Multimodal late fusion", 0.9341, 0.8504, C.pink2),
                tx("Kalın bar AUC-ROC, ince bar accuracy değerini gösterir.", {
                  style: { ...smallStyle, fontSize: 19, color: C.soft },
                }),
              ],
            ),
          ],
        ),
        tx("Ana bulgu: Multimodal late fusion modeli proposal hipotezini destekleyen en güçlü model oldu.", {
          style: { ...bodyStyle, fontSize: 27, color: C.text, bold: true },
        }),
        footer(8),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Controlled ablation", "Yardımcı deney tasarımı kararını destekledi.", "Ablation sonuçları full-training ile birebir aynı ölçekte yorumlanmadı; daha küçük ve kontrollü bir veri evreninde çalıştırıldı."),
        simpleTable(
          ["Model", "Satır", "AUC-ROC", "Accuracy"],
          [
            ["Tabular-only", "1.01M", "0.7219", "0.6658"],
            ["Image-history", "1.01M", "0.7883", "0.7261"],
            ["Late fusion", "1.01M", "0.9611", "0.8411"],
          ],
          [360, 230, 220, 220],
        ),
        bulletList([
          "Bu tablo ana sonuç değildir; geliştirme sürecindeki kontrollü karşılaştırmadır.",
          "Full-training performans iddiası Kaggle sonuç tablosuna dayandırıldı.",
          "İki düzende de göreli sıralama late fusion lehine gerçekleşti.",
        ], C.gold),
        footer(9),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Açıklanabilirlik", "SHAP yerine mimariye uygun iki pratik analiz kullanıldı.", "EfficientNet nihai karar modeli değil, embedding extractor olduğu için görsel açıklama embedding uzayında yapıldı."),
        grid(
          { width: fill, height: hug, columns: [fr(1), fr(1)], columnGap: 28 },
          [
            softPanel(column({ width: fill, height: hug, gap: 14 }, [
              tx("Görsel benzerlik", { style: { ...h2Style, fontSize: 34, color: C.teal } }),
              tx("Aday ürün embeddingi ile müşterinin geçmiş ürünleri cosine similarity üzerinden karşılaştırıldı.", { style: { ...bodyStyle, fontSize: 25 } }),
              tx("Çıktı: önerilen ürün, geçmişte alınan hangi ürünlere görsel olarak benziyor?", { style: { ...smallStyle, fontSize: 20, color: C.text } }),
            ])),
            softPanel(column({ width: fill, height: hug, gap: 14 }, [
              tx("Permutation importance", { style: { ...h2Style, fontSize: 34, color: C.pink2 } }),
              tx("Controlled tabular-only model üzerinde özellikler karıştırıldı ve AUC düşüşü ölçüldü.", { style: { ...bodyStyle, fontSize: 25 } }),
              tx("En güçlü sinyaller: section, department, product type, age, colour group.", { style: { ...smallStyle, fontSize: 20, color: C.text } }),
            ])),
          ],
        ),
        row(
          { width: fill, height: hug, gap: 18, align: "center" },
          [
            stat("Görsel açıklama", "cosine similarity", C.teal),
            stat("Tabular açıklama", "permutation", C.pink2),
            stat("SHAP durumu", "gelecek çalışma", C.gold),
          ],
        ),
        tx("SHAP, PyTorch embedding katmanları ve streaming preprocessing nedeniyle gelecek çalışma olarak bırakıldı.", {
          style: { ...bodyStyle, fontSize: 25, color: C.gold, bold: true },
        }),
        footer(10),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 34 },
      [
        titleBlock("Demo sistemi", "Final checkpointler canlı öneri arayüzüne bağlandı.", "Kullanıcı geçmişte almış gibi ürün seçer; backend aynı aday havuzunu üç modelle skorlar."),
        row(
          { width: fill, height: grow(1), gap: 30, align: "center" },
          [
            column({ width: grow(1), height: hug, gap: 18 }, [
              bulletList([
                "Frontend: React + Vite alışveriş arayüzü.",
                "Backend: FastAPI inference servisi.",
                "Modeller: final Kaggle checkpointleri.",
                "Çıktı: üç modelin önerileri yan yana.",
              ], C.pink2),
            ]),
            softPanel(column({ width: fill, height: hug, gap: 18 }, [
              tx("Demo inference akışı", { style: { ...h2Style, fontSize: 32 } }),
              tx("Ürün seçimi → müşteri profili → aday havuzu → üç model skoru → öneri listeleri", {
                style: { ...bodyStyle, fontSize: 27, color: C.text },
              }),
            ]), { width: fixed(670), fill: "#111827", line: { color: "#334155", width: 1 } }),
          ],
        ),
        footer(11),
      ],
    ),
  ),
);

add(
  root(
    column(
      { width: fill, height: fill, gap: 38, justify: "center" },
      [
        tx("Sonuç", { style: labelStyle }),
        tx("Moda önerilerinde müşterinin ne aldığı kadar, aldığı ürünlerin nasıl göründüğü de önemlidir.", {
          width: wrap(1500),
          style: { ...titleStyle, fontSize: 66, lineHeight: 1.02 },
        }),
        row(
          { width: fill, height: hug, gap: 28 },
          [
            stat("En iyi model", "Late fusion", C.pink2),
            stat("AUC-ROC", "0.9341", C.teal),
            stat("Accuracy", "0.8504", C.gold),
          ],
        ),
        tx("EfficientNet-B0 embeddingleri, müşteri görsel geçmişi ve tabular metaveri birlikte kullanıldığında öneri skorlama performansı arttı.", {
          width: wrap(1320),
          style: { ...bodyStyle, fontSize: 29, color: C.text },
        }),
        footer(12),
      ],
    ),
  ),
);

await fs.mkdir(previewDir, { recursive: true });

const pptxBlob = await PresentationFile.exportPptx(deck);
await pptxBlob.save(pptxPath);

for (let i = 0; i < slideRefs.length; i += 1) {
  const slide = slideRefs[i];
  const pngBlob = await deck.export({ slide, format: "png" });
  const bytes = Buffer.from(await pngBlob.arrayBuffer());
  const name = `slide_${String(i + 1).padStart(2, "0")}.png`;
  await fs.writeFile(path.join(previewDir, name), bytes);
}

const summary = {
  pptxPath,
  previewDir,
  slideCount: slideRefs.length,
};

await fs.writeFile(`${previewDir}/build_summary.json`, JSON.stringify(summary, null, 2), "utf8");
console.log(JSON.stringify(summary, null, 2));
