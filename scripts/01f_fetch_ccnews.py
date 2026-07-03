#!/usr/bin/env python3
"""
Fetch large-scale human news texts from stanford-oval/ccnews.
Also fetch more academic from ccdv/arxiv-summarization and
more social from reddit_tifu / webis-tldr-17.
Target: 500K human texts per register.
"""
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RAW_PATH = os.path.join(DATA_DIR, 'corpus_raw.parquet')

TARGET = 500_000
MIN_WORDS = {'news': 80, 'academic': 100, 'social': 30, 'creative': 80}


def word_count(text):
    return len(str(text).split())


def load_existing():
    if os.path.exists(RAW_PATH):
        df = pd.read_parquet(RAW_PATH)
        print(f"Existing corpus: {len(df)} rows")
        return df
    return pd.DataFrame(columns=['text', 'label', 'register', 'source'])


def existing_set(df):
    return set(df['text'].str[:200].tolist())


def fetch_register(ds_name, split, col, register, src_label, need, ex_set,
                   streaming=True, text_transform=None):
    rows = []
    print(f"\nFetching {register} from {ds_name}: need {need}")
    try:
        ds = load_dataset(ds_name, split=split, streaming=streaming)
        it = iter(ds) if streaming else iter(ds)
        for item in tqdm(it, desc=src_label):
            text = str(item.get(col, ''))
            if text_transform:
                text = text_transform(text)
            if word_count(text) < MIN_WORDS[register]:
                continue
            key = text[:200]
            if key in ex_set:
                continue
            rows.append({'text': text[:6000], 'label': 0,
                         'register': register, 'source': src_label})
            ex_set.add(key)
            if len(rows) >= need:
                break
    except Exception as e:
        print(f"  FAILED: {e}")
    print(f"  Got {len(rows)} rows")
    return rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    df = load_existing()
    ex = existing_set(df)

    human = df[df['label'] == 0]
    counts = human.groupby('register').size().to_dict()
    print(f"Current human counts: {counts}")

    all_new = []

    # --- NEWS: stanford-oval/ccnews ---
    need_news = max(0, TARGET - counts.get('news', 0))
    if need_news > 0:
        rows = fetch_register(
            'stanford-oval/ccnews', 'train', 'plain_text',
            'news', 'ccnews', need_news, ex, streaming=True
        )
        all_new.extend(rows)

    # --- ACADEMIC: ccdv/arxiv-summarization ---
    need_acad = max(0, TARGET - counts.get('academic', 0))
    if need_acad > 0:
        rows = fetch_register(
            'ccdv/arxiv-summarization', 'train', 'article',
            'academic', 'arxiv_article', need_acad, ex, streaming=False
        )
        all_new.extend(rows)

    # --- SOCIAL: reddit_tifu ---
    need_social = max(0, TARGET - counts.get('social', 0))
    if need_social > 0:
        rows = fetch_register(
            'reddit_tifu', 'train', 'documents',
            'social', 'reddit_tifu', need_social, ex, streaming=False
        )
        all_new.extend(rows)
        if len([r for r in all_new if r['register'] == 'social']) < need_social:
            still_need = need_social - len([r for r in all_new if r['register'] == 'social'])
            rows2 = fetch_register(
                'fancyzhx/yelp_polarity', 'train', 'text',
                'social', 'yelp_polarity', still_need, ex, streaming=False
            )
            all_new.extend(rows2)

    if not all_new:
        print("Nothing new fetched.")
        return

    new_df = pd.DataFrame(all_new)
    print("\nNew rows by register:")
    print(new_df.groupby('register').size().to_string())

    combined = pd.concat([df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['text'])
    combined.to_parquet(RAW_PATH, index=False)
    print(f"\nFinal corpus_raw.parquet: {len(combined)} rows")
    h = combined[combined['label'] == 0]
    print("Human counts by register:")
    print(h.groupby('register').size().to_string())


if __name__ == '__main__':
    main()
