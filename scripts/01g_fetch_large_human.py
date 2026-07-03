#!/usr/bin/env python3
"""
Fetch large-scale human texts and save each register to its own parquet
to avoid OOM when combining. Saves to data/human_<register>.parquet files.
Feature extraction script will load all shards.
"""
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

TARGET = 500_000
MIN_WORDS = {'news': 80, 'academic': 80, 'social': 30, 'creative': 80}
MAX_CHARS = 4000  # truncate stored text to save memory

np.random.seed(42)


def word_count(text):
    return len(str(text).split())


def existing_count(register):
    p = os.path.join(DATA_DIR, f'human_{register}.parquet')
    if os.path.exists(p):
        df = pd.read_parquet(p, columns=['text'])
        return len(df), set(df['text'].str[:200].tolist())
    return 0, set()


def save_register(register, rows):
    p = os.path.join(DATA_DIR, f'human_{register}.parquet')
    new_df = pd.DataFrame(rows)
    if os.path.exists(p):
        existing = pd.read_parquet(p)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['text'])
    else:
        combined = new_df.drop_duplicates(subset=['text'])
    combined.to_parquet(p, index=False)
    print(f"  Saved {len(combined)} rows to human_{register}.parquet")
    return combined


def fetch_ccnews(need, ex_set):
    rows = []
    print(f"  Fetching cc_news: need {need}")
    try:
        ds = load_dataset('stanford-oval/ccnews', split='train', streaming=True)
        for item in tqdm(ds, desc='ccnews'):
            text = str(item.get('plain_text', ''))
            if word_count(text) < MIN_WORDS['news']:
                continue
            key = text[:200]
            if key in ex_set:
                continue
            rows.append({'text': text[:MAX_CHARS], 'label': 0,
                         'register': 'news', 'source': 'ccnews'})
            ex_set.add(key)
            if len(rows) >= need:
                break
    except Exception as e:
        print(f"  FAILED: {e}")
    print(f"  Got {len(rows)} news rows")
    return rows


def fetch_arxiv(need, ex_set):
    rows = []
    print(f"  Fetching arxiv articles: need {need}")
    try:
        ds = load_dataset('ccdv/arxiv-summarization', split='train')
        for item in tqdm(ds, desc='arxiv'):
            text = str(item.get('article', ''))
            if word_count(text) < MIN_WORDS['academic']:
                continue
            key = text[:200]
            if key in ex_set:
                continue
            rows.append({'text': text[:MAX_CHARS], 'label': 0,
                         'register': 'academic', 'source': 'arxiv_article'})
            ex_set.add(key)
            if len(rows) >= need:
                break
    except Exception as e:
        print(f"  FAILED: {e}")
    print(f"  Got {len(rows)} academic rows")
    return rows


def fetch_yelp(need, ex_set):
    rows = []
    print(f"  Fetching yelp_polarity: need {need}")
    try:
        ds = load_dataset('fancyzhx/yelp_polarity', split='train')
        for item in tqdm(ds, desc='yelp'):
            text = str(item.get('text', ''))
            if word_count(text) < MIN_WORDS['social']:
                continue
            key = text[:200]
            if key in ex_set:
                continue
            rows.append({'text': text[:MAX_CHARS], 'label': 0,
                         'register': 'social', 'source': 'yelp_polarity'})
            ex_set.add(key)
            if len(rows) >= need:
                break
    except Exception as e:
        print(f"  FAILED: {e}")
    print(f"  Got {len(rows)} social rows")
    return rows


def also_check_raw(register):
    """Count human texts for register in corpus_raw.parquet."""
    raw_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    if not os.path.exists(raw_path):
        return 0, set()
    df = pd.read_parquet(raw_path, columns=['text', 'label', 'register'])
    mask = (df['label'] == 0) & (df['register'] == register)
    sub = df[mask]
    return len(sub), set(sub['text'].str[:200].tolist())


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for register, fetcher, raw_col in [
        ('news', fetch_ccnews, None),
        ('academic', fetch_arxiv, None),
        ('social', fetch_yelp, None),
    ]:
        print(f"\n=== {register.upper()} ===")
        # Count existing in shard file
        shard_count, ex_set = existing_count(register)
        # Also count in corpus_raw.parquet
        raw_count, raw_set = also_check_raw(register)
        ex_set |= raw_set
        total_existing = shard_count + raw_count
        print(f"  Existing: {shard_count} shard + {raw_count} raw = {total_existing} total")

        need = max(0, TARGET - total_existing)
        if need == 0:
            print(f"  Already at target, skipping")
            continue

        rows = fetcher(need, ex_set)
        if rows:
            save_register(register, rows)

    # Summary
    print("\n=== SUMMARY ===")
    raw_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    raw_df = pd.read_parquet(raw_path, columns=['label', 'register'])
    raw_human = raw_df[raw_df['label'] == 0].groupby('register').size()
    print("corpus_raw.parquet human:")
    print(raw_human.to_string())

    total = {}
    for register in ['news', 'academic', 'social', 'creative', 'encyclopedic']:
        p = os.path.join(DATA_DIR, f'human_{register}.parquet')
        shard = len(pd.read_parquet(p, columns=['text'])) if os.path.exists(p) else 0
        raw = raw_human.get(register, 0)
        total[register] = shard + raw
    print("\nTotal human per register (raw + shard):")
    for k, v in sorted(total.items()):
        print(f"  {k}: {v:,}")


if __name__ == '__main__':
    main()
