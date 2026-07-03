#!/usr/bin/env python3
"""
Fetch definitively human texts from pre-ChatGPT sources.
OpenAlex API: papers published before 2022-11-01 (before ChatGPT launch).
The Pile subsets: academic, books, news — all scraped pre-2022.
Gutenberg: public domain books.

Saves to data/human_<register>_prellm.parquet per register.
Target: 500K per register.
"""
import os
import time
import json
import requests
import pandas as pd
import numpy as np
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

TARGET = 500_000
MIN_WORDS = {'news': 80, 'academic': 80, 'social': 40, 'creative': 100}
MAX_CHARS = 5000
CUTOFF_YEAR = 2022  # only texts published before ChatGPT (Nov 2022)


def word_count(text):
    return len(str(text).split())


def shard_path(register):
    return os.path.join(DATA_DIR, f'human_{register}_prellm.parquet')


def load_existing_shard(register):
    p = shard_path(register)
    if os.path.exists(p):
        df = pd.read_parquet(p)
        return len(df), set(df['text'].str[:200].tolist())
    return 0, set()


def save_rows(register, rows):
    if not rows:
        return 0
    p = shard_path(register)
    new_df = pd.DataFrame(rows)
    if os.path.exists(p):
        existing = pd.read_parquet(p)
        combined = pd.concat([existing, new_df], ignore_index=True)
        combined = combined.drop_duplicates(subset=['text'])
    else:
        combined = new_df.drop_duplicates(subset=['text'])
    combined.to_parquet(p, index=False)
    print(f"  Saved {len(combined)} total rows to human_{register}_prellm.parquet")
    return len(combined)


# ── OpenAlex academic ────────────────────────────────────────────────────────

def fetch_openalex_academic(need, ex_set):
    """
    OpenAlex REST API: abstracts of papers published 2010-2022.
    Endpoint: https://api.openalex.org/works
    Filter: has_abstract=true, publication_year:<2023, type=article
    Cursor pagination, polite pool (no key needed, add email for higher rate).
    """
    print(f"\n[OpenAlex] Fetching {need} academic abstracts (pre-2022)...")
    rows = []
    url = "https://api.openalex.org/works"
    params = {
        "filter": "has_abstract:true,publication_year:<2023,type:article",
        "select": "id,abstract_inverted_index,publication_year",
        "per-page": 200,
        "cursor": "*",
        "mailto": "research@example.com",  # polite pool
    }

    with tqdm(total=need, desc="openalex") as pbar:
        while len(rows) < need:
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(5)
                    continue
                if r.status_code != 200:
                    print(f"  HTTP {r.status_code}, stopping")
                    break
                data = r.json()
                results = data.get('results', [])
                if not results:
                    break

                for item in results:
                    aii = item.get('abstract_inverted_index')
                    if not aii:
                        continue
                    # Reconstruct abstract from inverted index
                    max_pos = max(pos for positions in aii.values() for pos in positions)
                    tokens = [''] * (max_pos + 1)
                    for token, positions in aii.items():
                        for pos in positions:
                            tokens[pos] = token
                    text = ' '.join(tokens).strip()
                    if word_count(text) < MIN_WORDS['academic']:
                        continue
                    key = text[:200]
                    if key in ex_set:
                        continue
                    rows.append({
                        'text': text[:MAX_CHARS],
                        'label': 0,
                        'register': 'academic',
                        'source': 'openalex',
                        'year': item.get('publication_year'),
                    })
                    ex_set.add(key)
                    pbar.update(1)
                    if len(rows) >= need:
                        break

                next_cursor = data.get('meta', {}).get('next_cursor')
                if not next_cursor:
                    break
                params['cursor'] = next_cursor
                time.sleep(0.1)  # polite rate limit

            except Exception as e:
                print(f"  Error: {e}, retrying in 5s...")
                time.sleep(5)

    print(f"  Got {len(rows)} OpenAlex academic rows")
    return rows


# ── The Pile (pre-2022 curated corpus) ──────────────────────────────────────

def fetch_pile_subset(subset_name, register, need, ex_set):
    """
    EleutherAI's The Pile — assembled pre-2022, all human text.
    Subsets: 'FreeLaw', 'PubMed Abstracts', 'ArXiv', 'Wikipedia (en)',
             'BookCorpus2', 'Gutenberg (PG-19)', 'OpenWebText2', 'Pile-CC'
    Use monology/pile-uncopyrighted or the raw parquet files.
    """
    print(f"\n[The Pile:{subset_name}] Fetching {need} {register} rows...")
    rows = []

    pile_datasets = [
        ('EleutherAI/pile', subset_name, 'text'),
        ('monology/pile-uncopyrighted', subset_name, 'text'),
    ]

    for ds_name, config, col in pile_datasets:
        if len(rows) >= need:
            break
        try:
            ds = load_dataset(ds_name, config, split='train', streaming=True,
                              trust_remote_code=False)
            for item in tqdm(ds, desc=f"{ds_name}/{config}"):
                text = str(item.get(col, ''))
                if word_count(text) < MIN_WORDS.get(register, 50):
                    continue
                key = text[:200]
                if key in ex_set:
                    continue
                rows.append({
                    'text': text[:MAX_CHARS],
                    'label': 0,
                    'register': register,
                    'source': f'pile_{config.lower().replace(" ", "_")}',
                    'year': None,
                })
                ex_set.add(key)
                if len(rows) >= need:
                    break
            print(f"  Got {len(rows)} rows from {ds_name}")
        except Exception as e:
            print(f"  {ds_name} failed: {e}")

    return rows


# ── S2ORC / Semantic Scholar ─────────────────────────────────────────────────

def fetch_s2orc_academic(need, ex_set):
    """allenai/s2orc — abstracts + full papers, all pre-2022 vintage."""
    print(f"\n[S2ORC] Fetching {need} academic rows...")
    rows = []
    sources = [
        ('allenai/peS2o', 'train', 'text'),
        ('leminda-ai/s2orc_tle', 'train', 'abstract'),
    ]
    for ds_name, split, col in sources:
        if len(rows) >= need:
            break
        try:
            ds = load_dataset(ds_name, split=split, streaming=True)
            for item in tqdm(ds, desc=ds_name):
                text = str(item.get(col, ''))
                if word_count(text) < MIN_WORDS['academic']:
                    continue
                key = text[:200]
                if key in ex_set:
                    continue
                rows.append({
                    'text': text[:MAX_CHARS],
                    'label': 0,
                    'register': 'academic',
                    'source': f's2orc_{ds_name.split("/")[-1]}',
                    'year': item.get('year'),
                })
                ex_set.add(key)
                if len(rows) >= need:
                    break
            print(f"  Got {len(rows)} rows from {ds_name}")
        except Exception as e:
            print(f"  {ds_name} failed: {e}")
    return rows


# ── RealNews / CC-News pre-2022 ──────────────────────────────────────────────

def fetch_realnews(need, ex_set):
    """
    cc_news via stanford-oval/ccnews (assembled 2016-2023).
    Filter to pre-2022 by checking publish_date field.
    """
    print(f"\n[CC-News pre-2022] Fetching {need} news rows...")
    rows = []
    try:
        ds = load_dataset('stanford-oval/ccnews', split='train', streaming=True)
        for item in tqdm(ds, desc='ccnews-pre2022'):
            # Filter to pre-ChatGPT: published_date or date field
            pub = str(item.get('published_date', '') or item.get('date', ''))
            if pub and pub[:4].isdigit() and int(pub[:4]) >= CUTOFF_YEAR:
                continue  # skip 2022+ articles to be safe
            text = str(item.get('plain_text', ''))
            if word_count(text) < MIN_WORDS['news']:
                continue
            key = text[:200]
            if key in ex_set:
                continue
            rows.append({
                'text': text[:MAX_CHARS],
                'label': 0,
                'register': 'news',
                'source': 'ccnews_pre2022',
                'year': int(pub[:4]) if pub and pub[:4].isdigit() else None,
            })
            ex_set.add(key)
            if len(rows) >= need:
                break
    except Exception as e:
        print(f"  CC-News failed: {e}")
    print(f"  Got {len(rows)} news rows")
    return rows


# ── Reddit pre-2022 (social) ─────────────────────────────────────────────────

def fetch_reddit_social(need, ex_set):
    """webis-tldr-17 and similar Reddit corpora scraped before 2020."""
    print(f"\n[Reddit/social pre-2022] Fetching {need} social rows...")
    rows = []
    sources = [
        ('ArmelR/the-pile-reddit', None, 'train', 'text'),
        ('sentence-transformers/reddit-title-body', None, 'train', 'body'),
        ('fancyzhx/yelp_polarity', None, 'train', 'text'),
    ]
    for ds_name, config, split, col in sources:
        if len(rows) >= need:
            break
        try:
            kw = {'split': split, 'streaming': True}
            if config:
                kw['name'] = config
            ds = load_dataset(ds_name, **kw)
            for item in tqdm(ds, desc=ds_name.split('/')[-1]):
                text = str(item.get(col, ''))
                if word_count(text) < MIN_WORDS['social']:
                    continue
                key = text[:200]
                if key in ex_set:
                    continue
                rows.append({
                    'text': text[:MAX_CHARS],
                    'label': 0,
                    'register': 'social',
                    'source': f'reddit_{ds_name.split("/")[-1]}',
                    'year': None,
                })
                ex_set.add(key)
                if len(rows) >= need:
                    break
            print(f"  Got {len(rows)} rows so far from {ds_name}")
        except Exception as e:
            print(f"  {ds_name} failed: {e}")
    print(f"  Got {len(rows)} social rows total")
    return rows


# ── Gutenberg creative ───────────────────────────────────────────────────────

def fetch_gutenberg_creative(need, ex_set):
    """pg19 (Project Gutenberg, pre-1919 books) — definitely human."""
    print(f"\n[Gutenberg] Fetching {need} creative rows...")
    rows = []
    try:
        ds = load_dataset('pg19', split='train', streaming=True)
        for item in tqdm(ds, desc='pg19'):
            text = str(item.get('text', ''))
            words = text.split()
            # Chunk into ~500-word segments
            for i in range(0, len(words) - 400, 400):
                chunk = ' '.join(words[i:i+500])
                if word_count(chunk) < MIN_WORDS['creative']:
                    continue
                key = chunk[:200]
                if key in ex_set:
                    continue
                rows.append({
                    'text': chunk,
                    'label': 0,
                    'register': 'creative',
                    'source': 'gutenberg_pg19',
                    'year': item.get('year'),
                })
                ex_set.add(key)
                if len(rows) >= need:
                    break
            if len(rows) >= need:
                break
    except Exception as e:
        print(f"  pg19 failed: {e}")
    print(f"  Got {len(rows)} creative rows")
    return rows


def also_existing_raw_count(register):
    """Count existing human texts for this register across all existing sources."""
    total = 0
    keys = set()
    raw_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    if os.path.exists(raw_path):
        df = pd.read_parquet(raw_path, columns=['text', 'label', 'register'])
        sub = df[(df['label'] == 0) & (df['register'] == register)]
        total += len(sub)
        keys |= set(sub['text'].str[:200].tolist())
    shard = os.path.join(DATA_DIR, f'human_{register}.parquet')
    if os.path.exists(shard):
        df2 = pd.read_parquet(shard, columns=['text'])
        total += len(df2)
        keys |= set(df2['text'].str[:200].tolist())
    return total, keys


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    tasks = [
        ('academic', [fetch_openalex_academic, fetch_s2orc_academic]),
        ('news',     [fetch_realnews]),
        ('social',   [fetch_reddit_social]),
        ('creative', [fetch_gutenberg_creative]),
    ]

    for register, fetchers in tasks:
        print(f"\n{'='*50}")
        print(f"REGISTER: {register.upper()}")
        prellm_count, prellm_set = load_existing_shard(register)
        raw_count, raw_keys = also_existing_raw_count(register)
        ex_set = prellm_set | raw_keys
        total_existing = prellm_count + raw_count
        print(f"  Existing: {prellm_count} prellm + {raw_count} raw = {total_existing}")

        need = max(0, TARGET - total_existing)
        if need == 0:
            print(f"  Already at {TARGET:,}, skipping")
            continue

        rows = []
        for fetcher in fetchers:
            if len(rows) >= need:
                break
            new = fetcher(need - len(rows), ex_set)
            rows.extend(new)

        if rows:
            save_rows(register, rows)
        else:
            print(f"  No rows fetched for {register}")

    # Final summary
    print(f"\n{'='*50}")
    print("FINAL SUMMARY")
    for register in ['academic', 'news', 'social', 'creative', 'encyclopedic']:
        total, _ = also_existing_raw_count(register)
        prellm, _ = load_existing_shard(register)
        print(f"  {register}: {total:,} raw/shard + {prellm:,} prellm = {total+prellm:,}")


if __name__ == '__main__':
    main()
