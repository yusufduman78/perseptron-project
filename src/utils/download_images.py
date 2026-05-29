"""
Faz 2 - AdÄ±m 1: Sandbox Ã¼rÃ¼nlerinden rastgele 2000 tanesinin gÃ¶rselini Kaggle'dan indir.
GÃ¶rseller: perseptron_project/images/ klasÃ¶rÃ¼ne kaydedilir.
"""
import pandas as pd
import numpy as np
import subprocess
import os
import sys
from pathlib import Path
from tqdm import tqdm

PROJECT_DIR = Path(__file__).resolve().parents[2]
SANDBOX_DIR = PROJECT_DIR / "data" / "processed" / "sandbox"
IMAGES_DIR = PROJECT_DIR / "data" / "images" / "hm_images"
KAGGLE_EXE = r"C:\Users\Yusuf\anaconda3\envs\perseptron\Scripts\kaggle.exe"
COMPETITION = "h-and-m-personalized-fashion-recommendations"
NUM_IMAGES = 2000

os.makedirs(IMAGES_DIR, exist_ok=True)

# 1. Sandbox Ã¼rÃ¼n ID'lerini oku
articles = pd.read_csv(SANDBOX_DIR / "sandbox_articles.csv", dtype={'article_id': str})
all_ids = articles['article_id'].values

# 2. Rastgele 2000 Ã¼rÃ¼n seÃ§
np.random.seed(42)
sampled_ids = np.random.choice(all_ids, size=min(NUM_IMAGES, len(all_ids)), replace=False)

print(f"Ä°ndirilecek gÃ¶rsel sayÄ±sÄ±: {len(sampled_ids)}")
print(f"Hedef klasÃ¶r: {IMAGES_DIR}")
print("-" * 50)

# 3. Her Ã¼rÃ¼n iÃ§in Kaggle'dan indir
success = 0
fail = 0
for i, article_id in enumerate(tqdm(sampled_ids, desc="GÃ¶rseller indiriliyor")):
    # H&M formatÄ±: article_id = "0108775015" â†’ klasÃ¶r = "010", dosya = "0108775015.jpg"
    padded_id = str(article_id).zfill(10)
    folder = padded_id[:3]
    kaggle_path = f"images/{folder}/{padded_id}.jpg"
    
    # Yerel klasÃ¶r yapÄ±sÄ±nÄ± oluÅŸtur
    local_folder = os.path.join(IMAGES_DIR, folder)
    os.makedirs(local_folder, exist_ok=True)
    
    local_file = os.path.join(local_folder, f"{padded_id}.jpg")
    
    # Zaten indirildiyse atla
    if os.path.exists(local_file):
        success += 1
        continue
    
    try:
        result = subprocess.run(
            [KAGGLE_EXE, "competitions", "download", "-c", COMPETITION, "-f", kaggle_path, "-p", local_folder, "-q"],
            capture_output=True, text=True, timeout=30
        )
        
        # Kaggle zip olarak indirebilir, kontrol et
        zip_file = os.path.join(local_folder, f"{padded_id}.jpg.zip")
        if os.path.exists(zip_file):
            import zipfile
            with zipfile.ZipFile(zip_file, 'r') as z:
                z.extractall(local_folder)
            os.remove(zip_file)
        
        if os.path.exists(local_file):
            success += 1
        else:
            fail += 1
    except Exception as e:
        fail += 1
    
    # Her 100 gÃ¶rselde durum raporu
    if (i + 1) % 100 == 0:
        print(f"  Ä°lerleme: {i+1}/{len(sampled_ids)} | BaÅŸarÄ±lÄ±: {success} | BaÅŸarÄ±sÄ±z: {fail}")

print(f"\n{'='*50}")
print(f"TAMAMLANDI! BaÅŸarÄ±lÄ±: {success} | BaÅŸarÄ±sÄ±z: {fail}")
print(f"GÃ¶rseller: {IMAGES_DIR}")

# 4. Ä°ndirilen ID'leri kaydet (hangi Ã¼rÃ¼nlerin gÃ¶rseli var bilmek iÃ§in)
downloaded_ids = []
for folder_name in os.listdir(IMAGES_DIR):
    folder_path = os.path.join(IMAGES_DIR, folder_name)
    if os.path.isdir(folder_path):
        for f in os.listdir(folder_path):
            if f.endswith('.jpg'):
                downloaded_ids.append(f.replace('.jpg', ''))

pd.DataFrame({'article_id': downloaded_ids}).to_csv(
    SANDBOX_DIR / "downloaded_image_ids.csv", index=False
)
print(f"Indirilen ID listesi kaydedildi: {SANDBOX_DIR / 'downloaded_image_ids.csv'}")

