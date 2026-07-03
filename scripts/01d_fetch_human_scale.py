#!/usr/bin/env python3
"""
Fetch large-scale human baseline texts to reach ~500K per register.
Sources:
  - News: cc_news (CommonCrawl News) - millions of articles
  - Academic: ccdv/arxiv-summarization (abstract+article), togethercomputer/RedPajama-Data-1T-Sample academic
  - Social: reddit pushshift via webis-tldr-17 or reddit_tifu
  - Creative: bookcorpusopen or pg19
Appends to corpus_raw.parquet, deduplicating against existing texts.
"""
import os, sys
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RAW_PATH = os.path.join(DATA_DIR, 'corpus_raw.parquet')

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

TARGET_PER_REGISTER = 500_000
MIN_WORDS = {'news': 80, 'academic': 100, 'social': 30, 'creative': 80, 'encyclopedic': 80}

def word_count(text):
    return len(str(text).split())

def load_existing():
    if os.path.exists(RAW_PATH):
        df = pd.read_parquet(RAW_PATH)
        print(f"Existing corpus: {len(df)} rows")
        return df
    return pd.DataFrame(columns=['text', 'label', 'register', 'source'])

def existing_human_counts(df):
    human = df[df['label'] == 0]
    return human.groupby('register').size().to_dict()

def existing_texts_set(df):
    return set(df['text'].str[:200].tolist())

def fetch_news(need, existing_set):
    """cc_news dataset - CommonCrawl news articles."""
    rows = []
    print(f"\nFetching news: need {need} more texts")
    try:
        ds = load_dataset("cc_news", split="train", streaming=True, trust_remote_code=True)
        for item in tqdm(ds, desc="cc_news"):
            text = str(item.get('text', '') or item.get('description', ''))
            if word_count(text) < MIN_WORDS['news']:
                continue
            key = text[:200]
            if key in existing_set:
                continue
            rows.append({'text': text, 'label': 0, 'register': 'news', 'source': 'cc_news'})
            existing_set.add(key)
            if len(rows) >= need:
                break
    except Exception as e:
        print(f"  cc_news failed: {e}")
        # Fallback: try ag_news with lower threshold
        try:
            ds2 = load_dataset("ag_news", split="train", trust_remote_code=True)
            for item in tqdm(ds2, desc="ag_news fallback"):
                text = str(item.get('text', ''))
                if word_count(text) < 20:  # very low threshold for AG-News
                    continue
                key = text[:200]
                if key in existing_set:
                    continue
                rows.append({'text': text, 'label': 0, 'register': 'news', 'source': 'ag_news'})
                existing_set.add(key)
                if len(rows) >= need:
                    break
            print(f"  ag_news fallback: {len(rows)} rows")
        except Exception as e2:
            print(f"  ag_news fallback also failed: {e2}")
    print(f"  Got {len(rows)} news rows")
    return rows

def fetch_academic(need, existing_set):
    """Additional arXiv texts from RedPajama or pile-of-law."""
    rows = []
    print(f"\nFetching academic: need {need} more texts")
    sources_to_try = [
        ("togethercomputer/RedPajama-Data-1T-Sample", "train", "text", "redpajama_academic"),
        ("ccdv/arxiv-summarization", "train", "abstract", "arxiv_abstract"),
        ("ccdv/arxiv-summarization", "validation", "abstract", "arxiv_abstract"),
    ]
    for ds_name, split, col, src_label in sources_to_try:
        if len(rows) >= need:
            break
        try:
            if "RedPajama" in ds_name:
                ds = load_dataset(ds_name, split=split, streaming=True, trust_remote_code=True)
                for item in tqdm(ds, desc=src_label):
                    meta = item.get('meta', {})
                    if isinstance(meta, str):
                        import json
                        try:
                            meta = json.loads(meta)
                        except Exception:
                            meta = {}
                    if meta.get('redpajama_set_name', '') != 'RedPajamaArXiv':
                        continue
                    text = str(item.get(col, ''))
                    if word_count(text) < MIN_WORDS['academic']:
                        continue
                    key = text[:200]
                    if key in existing_set:
                        continue
                    rows.append({'text': text[:5000], 'label': 0, 'register': 'academic', 'source': src_label})
                    existing_set.add(key)
                    if len(rows) >= need:
                        break
            else:
                ds = load_dataset(ds_name, split=split, trust_remote_code=True)
                for item in tqdm(ds, desc=src_label):
                    text = str(item.get(col, ''))
                    if word_count(text) < MIN_WORDS['academic']:
                        continue
                    key = text[:200]
                    if key in existing_set:
                        continue
                    rows.append({'text': text, 'label': 0, 'register': 'academic', 'source': src_label})
                    existing_set.add(key)
                    if len(rows) >= need:
                        break
            print(f"  {src_label}: {len(rows)} total rows so far")
        except Exception as e:
            print(f"  {ds_name} failed: {e}")
    print(f"  Got {len(rows)} academic rows")
    return rows

def fetch_social(need, existing_set):
    """Reddit TIFU or webis-tldr for social register."""
    rows = []
    print(f"\nFetching social: need {need} more texts")
    sources_to_try = [
        ("reddit_tifu", "train", "documents", "reddit_tifu"),
        ("webis/tldr-17", "train", "content", "webis_tldr"),
        ("fancyzhx/yelp_polarity", "train", "text", "yelp_polarity"),
    ]
    for ds_name, split, col, src_label in sources_to_try:
        if len(rows) >= need:
            break
        try:
            ds = load_dataset(ds_name, split=split, trust_remote_code=True)
            for item in tqdm(ds, desc=src_label):
                text = str(item.get(col, ''))
                if word_count(text) < MIN_WORDS['social']:
                    continue
                key = text[:200]
                if key in existing_set:
                    continue
                rows.append({'text': text[:3000], 'label': 0, 'register': 'social', 'source': src_label})
                existing_set.add(key)
                if len(rows) >= need:
                    break
            print(f"  {src_label}: {len(rows)} total rows so far")
        except Exception as e:
            print(f"  {ds_name} failed: {e}")
    print(f"  Got {len(rows)} social rows")
    return rows

def fetch_creative(need, existing_set):
    """pg19 (Project Gutenberg) or bookcorpusopen for creative register."""
    rows = []
    print(f"\nFetching creative: need {need} more texts")
    sources_to_try = [
        ("pg19", "train", "text", "pg19"),
        ("bookcorpusopen", "plain_text", "text", "bookcorpus"),
    ]
    for ds_name, split, col, src_label in sources_to_try:
        if len(rows) >= need:
            break
        try:
            ds = load_dataset(ds_name, split=split, streaming=True, trust_remote_code=True)
            for item in tqdm(ds, desc=src_label):
                text = str(item.get(col, ''))
                # Chunk long books into ~500-word segments
                words = text.split()
                for i in range(0, len(words) - 400, 400):
                    chunk = ' '.join(words[i:i+500])
                    if word_count(chunk) < MIN_WORDS['creative']:
                        continue
                    key = chunk[:200]
                    if key in existing_set:
                        continue
                    rows.append({'text': chunk, 'label': 0, 'register': 'creative', 'source': src_label})
                    existing_set.add(key)
                    if len(rows) >= need:
                        break
                if len(rows) >= need:
                    break
            print(f"  {src_label}: {len(rows)} total rows so far")
        except Exception as e:
            print(f"  {ds_name} failed: {e}")
    print(f"  Got {len(rows)} creative rows")
    return rows

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    existing_df = load_existing()
    counts = existing_human_counts(existing_df)
    existing_set = existing_texts_set(existing_df)
    print(f"\nCurrent human counts: {counts}")

    all_new_rows = []

    for register, fetcher in [
        ('news', fetch_news),
        ('academic', fetch_academic),
        ('social', fetch_social),
        ('creative', fetch_creative),
    ]:
        current = counts.get(register, 0)
        need = max(0, TARGET_PER_REGISTER - current)
        if need == 0:
            print(f"\n{register}: already at {current}, skipping")
            continue
        print(f"\n{register}: have {current}, need {need} more to reach {TARGET_PER_REGISTER}")
        new_rows = fetcher(need, existing_set)
        all_new_rows.extend(new_rows)
        print(f"  Added {len(new_rows)} {register} rows")

    if not all_new_rows:
        print("\nNo new rows fetched.")
        return

    new_df = pd.DataFrame(all_new_rows)
    combined = pd.concat([existing_df, new_df], ignore_index=True)
    combined = combined.drop_duplicates(subset=['text'])
    combined.to_parquet(RAW_PATH, index=False)

    print(f"\nFinal corpus_raw.parquet: {len(combined)} rows")
    human = combined[combined['label'] == 0]
    print("Human counts by register:")
    print(human.groupby('register').size().to_string())

if __name__ == '__main__':
    main()
