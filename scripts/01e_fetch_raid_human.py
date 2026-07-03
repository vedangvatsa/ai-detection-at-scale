#!/usr/bin/env python3
"""
Fetch human texts from RAID train split (model='human').
These are the original source documents that RAID AI texts were generated from.
They are length-matched to the AI texts, resolving the document-length confound.
Domain -> register mapping:
  news      -> news
  reddit    -> social
  reviews   -> social
  abstracts -> academic
  wiki      -> encyclopedic
  books     -> creative
  poetry    -> creative
  recipes   -> creative (excluded: too domain-specific)
"""
import os
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RAW_PATH = os.path.join(DATA_DIR, 'corpus_raw.parquet')

DOMAIN_TO_REGISTER = {
    'news': 'news',
    'reddit': 'social',
    'reviews': 'social',
    'abstracts': 'academic',
    'wiki': 'encyclopedic',
    'books': 'creative',
    'poetry': 'creative',
}

MIN_WORDS = {'news': 50, 'social': 20, 'academic': 80, 'encyclopedic': 50, 'creative': 50}


def word_count(text):
    return len(str(text).split())


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    existing_df = pd.read_parquet(RAW_PATH) if os.path.exists(RAW_PATH) else pd.DataFrame()
    existing_set = set(existing_df['text'].str[:200].tolist()) if len(existing_df) else set()
    print(f"Existing corpus: {len(existing_df)} rows")

    print("Loading RAID train split...")
    ds = load_dataset('liamdugan/raid', split='train')
    df = ds.to_pandas()

    human = df[df['model'] == 'human'].copy()
    print(f"RAID train human rows: {len(human)}")
    print(human['domain'].value_counts().to_string())

    new_rows = []
    skipped_domain = 0
    skipped_len = 0
    skipped_dup = 0

    for _, row in tqdm(human.iterrows(), total=len(human), desc="Processing RAID human"):
        domain = str(row.get('domain', ''))
        register = DOMAIN_TO_REGISTER.get(domain)
        if register is None:
            skipped_domain += 1
            continue

        text = str(row.get('generation', '') or row.get('prompt', ''))
        if word_count(text) < MIN_WORDS.get(register, 30):
            skipped_len += 1
            continue

        key = text[:200]
        if key in existing_set:
            skipped_dup += 1
            continue

        new_rows.append({
            'text': text,
            'label': 0,
            'register': register,
            'source': f'raid_human_{domain}',
        })
        existing_set.add(key)

    print(f"\nSkipped: domain={skipped_domain}, len={skipped_len}, dup={skipped_dup}")
    print(f"New rows to add: {len(new_rows)}")

    if not new_rows:
        print("Nothing to add.")
        return

    new_df = pd.DataFrame(new_rows)
    print("\nNew rows by register:")
    print(new_df.groupby('register').size().to_string())

    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['text'])
    combined.to_parquet(RAW_PATH, index=False)

    print(f"\nFinal corpus_raw.parquet: {len(combined)} rows")
    human_final = combined[combined['label'] == 0]
    print("Human counts by register:")
    print(human_final.groupby('register').size().to_string())


if __name__ == '__main__':
    main()
