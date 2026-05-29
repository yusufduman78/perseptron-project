import pandas as pd
import numpy as np
import os
import gc
from pathlib import Path

print("--- Faz 1: EDA ve Sampling BaÅŸlÄ±yor ---")

PROJECT_DIR = Path(__file__).resolve().parents[2]
articles_path = PROJECT_DIR / "data" / "raw" / "articles.csv"
customers_path = PROJECT_DIR / "data" / "raw" / "customers.csv"
transactions_path = PROJECT_DIR / "data" / "raw" / "transactions_train.csv"

print("1. Ham Veriler YÃ¼kleniyor (Bu adÄ±m veri boyutundan dolayÄ± RAM'de biraz yer kaplayacak ve 1-2 dakika sÃ¼recektir)...")
transactions = pd.read_csv(transactions_path, dtype={'article_id': str})
articles = pd.read_csv(articles_path, dtype={'article_id': str})
customers = pd.read_csv(customers_path)

print(f"\nOrijinal Ä°ÅŸlem SayÄ±sÄ±: {len(transactions):,}")
print(f"Orijinal MÃ¼ÅŸteri SayÄ±sÄ±: {len(customers):,}")
print(f"Orijinal ÃœrÃ¼n SayÄ±sÄ±: {len(articles):,}")

print("\n2. MÃ¼ÅŸteri Filtreleme ve Ã–rnekleme (10.000 KiÅŸilik Sandbox)...")
user_counts = transactions.groupby('customer_id').size()
active_users = user_counts[user_counts >= 5].index.tolist()

print(f"En az 5 Ã¼rÃ¼n alan 'Aktif' mÃ¼ÅŸteri sayÄ±sÄ±: {len(active_users):,}")

np.random.seed(42)
if len(active_users) > 10000:
    sampled_users = np.random.choice(active_users, 10000, replace=False)
else:
    sampled_users = active_users

sandbox_transactions = transactions[transactions['customer_id'].isin(sampled_users)].copy()
sandbox_customers = customers[customers['customer_id'].isin(sampled_users)].copy()

sampled_articles_ids = sandbox_transactions['article_id'].unique()
sandbox_articles = articles[articles['article_id'].isin(sampled_articles_ids)].copy()

print(f"\n--- Local Sandbox Veri DaÄŸÄ±lÄ±mÄ± ---")
print(f"Ã–rneklenmiÅŸ MÃ¼ÅŸteri SayÄ±sÄ±: {len(sandbox_customers):,}")
print(f"Ã–rneklenmiÅŸ Ä°ÅŸlem (SatÄ±ÅŸ) SayÄ±sÄ±: {len(sandbox_transactions):,}")
print(f"SatÄ±ÅŸa KarÄ±ÅŸmÄ±ÅŸ EtkileÅŸimli ÃœrÃ¼n SayÄ±sÄ±: {len(sandbox_articles):,}")

print("\nRAM Temizleniyor...")
del transactions
del customers
del articles
gc.collect()

print("\n3. Faz 1'in Kalbi: Negatif Ã–rneklem (Negative Sampling) Ãœretimi...")
# Her 1 pozitif iÅŸleme karÅŸÄ±lÄ±k modelin gÃ¶rebilmesi iÃ§in rastgele 2 negatif (alÄ±nmayan) Ã¼rÃ¼n Ã¼retiyoruz.
num_negatives_per_pos = 2
all_article_ids = sandbox_articles['article_id'].values

positive_df = sandbox_transactions[['customer_id', 'article_id']].copy()
positive_df['label'] = 1

neg_customer_ids = np.repeat(sandbox_transactions['customer_id'].values, num_negatives_per_pos)
neg_article_ids = np.random.choice(all_article_ids, size=len(neg_customer_ids))

negative_df = pd.DataFrame({
    'customer_id': neg_customer_ids,
    'article_id': neg_article_ids,
    'label': 0
})

final_dataset = pd.concat([positive_df, negative_df], ignore_index=True)
final_dataset = final_dataset.sample(frac=1, random_state=42).reset_index(drop=True)

print(f"Final Model EÄŸitim Veri Seti SatÄ±r SayÄ±sÄ±: {len(final_dataset):,}")
print(f"SÄ±nÄ±f DaÄŸÄ±lÄ±mÄ±:\n{final_dataset['label'].value_counts()}")

print("\n4. Veriler 'data/processed/sandbox' klasorune CSV olarak kaydediliyor...")
sandbox_dir = PROJECT_DIR / "data" / "processed" / "sandbox"
os.makedirs(sandbox_dir, exist_ok=True)

sandbox_customers.to_csv(os.path.join(sandbox_dir, "sandbox_customers.csv"), index=False)
sandbox_articles.to_csv(os.path.join(sandbox_dir, "sandbox_articles.csv"), index=False)
sandbox_transactions.to_csv(os.path.join(sandbox_dir, "sandbox_transactions.csv"), index=False)
final_dataset.to_csv(os.path.join(sandbox_dir, "sandbox_model_data.csv"), index=False)

print("\n[BASARILI] FAZ 1 TAMAMLADI! Tum altyapi data/processed/sandbox altinda kullanima hazirdir.")



