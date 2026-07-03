#!/usr/bin/env python3
"""
Fix: add human texts for news + social registers, and RAID extra human texts.
Also add encyclopedic AI from RAID extra.
"""
import os, re
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

RAID_REGISTER_MAP = {
    'reddit': 'social', 'peerread': 'academic', 'arxiv': 'academic',
    'wikihow': 'encyclopedic', 'wikipedia': 'encyclopedic',
    'news': 'news', 'book': 'creative', 'essay': 'creative',
    'reuter': 'news', 'yelp': 'social', 'code': 'encyclopedic',
}

def word_count(t): return len(str(t).split())
def map_reg(s):
    s = str(s).lower()
    for k,v in RAID_REGISTER_MAP.items():
        if k in s: return v
    return 'other'


def main():
    existing = pd.read_parquet(os.path.join(DATA_DIR, 'corpus_raw.parquet'))
    print(f"Existing: {len(existing)} rows")
    print(existing.groupby(['register','label']).size().to_string())

    rows = []

    # 1. RAID extra split -- model=='human', use 'generation' field
    print("\nLoading RAID extra human (generation field)...")
    try:
        ds = load_dataset("liamdugan/raid", split="extra")
        for row in tqdm(ds, desc="  RAID extra"):
            if row.get('model') != 'human':
                continue
            text = row.get('generation', '')
            if not text or len(text.strip()) < 50:
                continue
            domain = row.get('domain', 'unknown')
            reg = map_reg(domain)
            if reg == 'other':
                continue
            min_w = {'academic':100,'news':100,'social':30,'encyclopedic':80,'creative':80}.get(reg,50)
            if word_count(text) < min_w:
                continue
            rows.append({'text': text.strip(), 'label': 0, 'register': reg,
                         'model': 'human', 'source': f'raid_extra_{domain}'})
        print(f"  RAID extra human rows: {len(rows)}")
    except Exception as e:
        print(f"  RAID extra failed: {e}")

    # 2. News human: SetFit/bbc-news (parquet-based)
    print("Loading news human: SetFit/bbc-news...")
    news_count = 0
    try:
        ds = load_dataset("SetFit/bbc-news", split="train")
        for row in tqdm(ds, desc="  bbc-news"):
            text = row.get('text', '')
            if text and word_count(text) >= 80:
                rows.append({'text': text.strip(), 'label': 0, 'register': 'news',
                             'model': 'human', 'source': 'bbc_news'})
                news_count += 1
        print(f"  BBC news rows: {news_count}")
    except Exception as e:
        print(f"  bbc-news failed: {e}")

    # 3. News human: heegyu/news-category-dataset
    print("Loading news human: heegyu/news-category-dataset...")
    news_count2 = 0
    try:
        ds = load_dataset("heegyu/news-category-dataset", split="train")
        for row in tqdm(ds, desc="  news-category"):
            text = (row.get('headline','') + ' ' + row.get('short_description','')).strip()
            if text and word_count(text) >= 20:
                rows.append({'text': text.strip(), 'label': 0, 'register': 'news',
                             'model': 'human', 'source': 'news_category'})
                news_count2 += 1
                if news_count2 >= 50000:
                    break
        print(f"  news-category rows: {news_count2}")
    except Exception as e:
        print(f"  news-category failed: {e}")

    # 4. News human: fancyzhx/ag_news (correct namespace)
    print("Loading news human: fancyzhx/ag_news...")
    news_count3 = 0
    try:
        ds = load_dataset("fancyzhx/ag_news", split="train")
        for row in tqdm(ds, desc="  ag_news"):
            text = row.get('text', '')
            if text and word_count(text) >= 30:
                rows.append({'text': text.strip(), 'label': 0, 'register': 'news',
                             'model': 'human', 'source': 'ag_news'})
                news_count3 += 1
                if news_count3 >= 80000:
                    break
        print(f"  ag_news rows: {news_count3}")
    except Exception as e:
        print(f"  ag_news failed: {e}")

    # 5. Social human: mteb/mteb_tweets_sentiment
    print("Loading social human: tweet_eval...")
    social_count = 0
    try:
        ds = load_dataset("cardiffnlp/tweet_eval", "sentiment", split="train")
        for row in tqdm(ds, desc="  tweet_eval"):
            text = row.get('text', '')
            if text and word_count(text) >= 10:
                rows.append({'text': text.strip(), 'label': 0, 'register': 'social',
                             'model': 'human', 'source': 'tweet_eval'})
                social_count += 1
                if social_count >= 30000:
                    break
        print(f"  tweet_eval rows: {social_count}")
    except Exception as e:
        print(f"  tweet_eval failed: {e}")

    # 6. Social human: Yelp reviews
    print("Loading social human: Yelp reviews...")
    yelp_count = 0
    try:
        ds = load_dataset("Yelp/yelp_review_full", split="train", streaming=True)
        for row in tqdm(ds, desc="  yelp"):
            text = row.get('text', '')
            if text and word_count(text) >= 30:
                rows.append({'text': text.strip(), 'label': 0, 'register': 'social',
                             'model': 'human', 'source': 'yelp'})
                yelp_count += 1
                if yelp_count >= 80000:
                    break
        print(f"  yelp rows: {yelp_count}")
    except Exception as e:
        print(f"  yelp failed: {e}")

    # 7. Academic human boost: kiddothe2b/scientific_papers_acl
    print("Loading academic human: Cohere/wikipedia-22-12-en-embeddings style check...")
    acad_count = 0
    try:
        ds = load_dataset("allenai/s2orc", "20200705v1", split="train", streaming=True)
        for row in tqdm(ds, desc="  s2orc"):
            text = row.get('abstract', '')
            if text and word_count(text) >= 100:
                rows.append({'text': text.strip(), 'label': 0, 'register': 'academic',
                             'model': 'human', 'source': 's2orc'})
                acad_count += 1
                if acad_count >= 50000:
                    break
        print(f"  s2orc rows: {acad_count}")
    except Exception as e:
        print(f"  s2orc failed: {e}")

    new_df = pd.DataFrame(rows)
    print(f"\nNew human rows fetched: {len(new_df)}")
    if len(new_df) > 0:
        print(new_df.groupby(['register','label']).size().to_string())

    combined = pd.concat([existing, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['text'])
    combined = combined[combined['text'].str.len() > 30]

    print(f"\nFinal corpus: {len(combined)} rows")
    print(combined.groupby(['register','label']).size().to_string())

    combined.to_parquet(os.path.join(DATA_DIR, 'corpus_raw.parquet'), index=False)
    print("Saved corpus_raw.parquet")

    combined.groupby(['register','label','source']).size().reset_index(name='count').to_csv(
        os.path.join(DATA_DIR, 'corpus_summary.csv'), index=False)
    print("Saved corpus_summary.csv")


if __name__ == '__main__':
    main()
