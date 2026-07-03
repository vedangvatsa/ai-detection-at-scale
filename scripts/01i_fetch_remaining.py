#!/usr/bin/env python3
"""
Fetch remaining human texts for academic and creative registers.
Academic: OpenAlex API (pre-2022 papers, definitively human).
Creative: pg19 Project Gutenberg books chunked to ~500 words.
Saves to human_academic2.parquet and human_creative.parquet.
"""
import os, time, requests
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
TARGET = 500_000
MAX_CHARS = 5000


def word_count(t):
    return len(str(t).split())


def load_ex_set(register):
    keys = set()
    for fname in [f'human_{register}.parquet', f'human_{register}2.parquet',
                  'corpus_raw.parquet']:
        p = os.path.join(DATA_DIR, fname)
        if not os.path.exists(p):
            continue
        try:
            if fname == 'corpus_raw.parquet':
                df = pd.read_parquet(p, columns=['text', 'label', 'register'])
                sub = df[(df['label'] == 0) & (df['register'] == register)]
                keys |= set(sub['text'].str[:200])
            else:
                df = pd.read_parquet(p, columns=['text'])
                keys |= set(df['text'].str[:200])
        except Exception:
            pass
    return keys


def existing_count(register):
    total = 0
    for fname in [f'human_{register}.parquet', f'human_{register}2.parquet',
                  'corpus_raw.parquet']:
        p = os.path.join(DATA_DIR, fname)
        if not os.path.exists(p):
            continue
        try:
            if fname == 'corpus_raw.parquet':
                df = pd.read_parquet(p, columns=['label', 'register'])
                total += ((df['label'] == 0) & (df['register'] == register)).sum()
            else:
                total += len(pd.read_parquet(p, columns=['text']))
        except Exception:
            pass
    return total


def save_shard(register, rows, suffix='2'):
    if not rows:
        return
    p = os.path.join(DATA_DIR, f'human_{register}{suffix}.parquet')
    new_df = pd.DataFrame(rows)
    if os.path.exists(p):
        old = pd.read_parquet(p)
        new_df = pd.concat([old, new_df], ignore_index=True).drop_duplicates(subset=['text'])
    new_df.to_parquet(p, index=False)
    print(f"  Saved {len(new_df):,} rows → human_{register}{suffix}.parquet")


# ── OpenAlex ─────────────────────────────────────────────────────────────────

def reconstruct_abstract(aii):
    if not aii:
        return ''
    try:
        max_pos = max(pos for positions in aii.values() for pos in positions)
        tokens = [''] * (max_pos + 1)
        for token, positions in aii.items():
            for pos in positions:
                if pos <= max_pos:
                    tokens[pos] = token
        return ' '.join(tokens).strip()
    except Exception:
        return ''


def fetch_openalex(need, ex_set):
    print(f"\n[OpenAlex] Fetching {need:,} academic abstracts (pre-2022)...")
    rows = []
    url = 'https://api.openalex.org/works'
    params = {
        'filter': 'has_abstract:true,publication_year:<2022,type:article',
        'select': 'id,abstract_inverted_index,publication_year,title',
        'per-page': 200,
        'cursor': '*',
        'mailto': 'research@example.com',
    }
    with tqdm(total=need, desc='openalex') as pbar:
        while len(rows) < need:
            try:
                r = requests.get(url, params=params, timeout=30)
                if r.status_code == 429:
                    time.sleep(10)
                    continue
                if r.status_code != 200:
                    print(f"  HTTP {r.status_code} — stopping")
                    break
                data = r.json()
                results = data.get('results', [])
                if not results:
                    break
                for item in results:
                    text = reconstruct_abstract(item.get('abstract_inverted_index'))
                    if word_count(text) < 80:
                        continue
                    key = text[:200]
                    if key in ex_set:
                        continue
                    rows.append({'text': text[:MAX_CHARS], 'label': 0,
                                 'register': 'academic', 'source': 'openalex'})
                    ex_set.add(key)
                    pbar.update(1)
                    if len(rows) >= need:
                        break
                cursor = data.get('meta', {}).get('next_cursor')
                if not cursor:
                    break
                params['cursor'] = cursor
                time.sleep(0.07)
            except Exception as e:
                print(f"  Error: {e} — retrying in 5s")
                time.sleep(5)
    print(f"  Got {len(rows):,} OpenAlex rows")
    return rows


# ── pg19 Gutenberg ────────────────────────────────────────────────────────────

def fetch_pg19(need, ex_set):
    print(f"\n[pg19] Fetching {need:,} creative chunks...")
    rows = []
    try:
        ds = load_dataset('pg19', split='train', streaming=True)
        for item in tqdm(ds, desc='pg19'):
            text = str(item.get('text', ''))
            words = text.split()
            for i in range(0, len(words) - 400, 400):
                chunk = ' '.join(words[i:i + 500])
                if word_count(chunk) < 100:
                    continue
                key = chunk[:200]
                if key in ex_set:
                    continue
                rows.append({'text': chunk, 'label': 0,
                             'register': 'creative', 'source': 'gutenberg_pg19'})
                ex_set.add(key)
                if len(rows) >= need:
                    break
            if len(rows) >= need:
                break
    except Exception as e:
        print(f"  pg19 failed: {e}")
    # Fallback: bookcorpusopen
    if len(rows) < need:
        try:
            ds2 = load_dataset('bookcorpusopen', 'plain_text', split='train', streaming=True)
            for item in tqdm(ds2, desc='bookcorpus'):
                text = str(item.get('text', ''))
                words = text.split()
                for i in range(0, len(words) - 400, 400):
                    chunk = ' '.join(words[i:i + 500])
                    if word_count(chunk) < 100:
                        continue
                    key = chunk[:200]
                    if key in ex_set:
                        continue
                    rows.append({'text': chunk, 'label': 0,
                                 'register': 'creative', 'source': 'bookcorpusopen'})
                    ex_set.add(key)
                    if len(rows) >= need:
                        break
                if len(rows) >= need:
                    break
        except Exception as e:
            print(f"  bookcorpusopen failed: {e}")
    print(f"  Got {len(rows):,} creative rows")
    return rows


def main():
    os.makedirs(DATA_DIR, exist_ok=True)

    for register, fetcher in [('academic', fetch_openalex), ('creative', fetch_pg19)]:
        count = existing_count(register)
        need = max(0, TARGET - count)
        print(f"\n{register}: existing={count:,}  need={need:,}")
        if need == 0:
            print("  Already at target.")
            continue
        ex_set = load_ex_set(register)
        rows = fetcher(need, ex_set)
        save_shard(register, rows)

    print("\n=== DONE ===")
    for register in ['academic', 'news', 'social', 'creative', 'encyclopedic']:
        print(f"  {register}: {existing_count(register):,}")


if __name__ == '__main__':
    main()
