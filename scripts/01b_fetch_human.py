#!/usr/bin/env python3
"""
Fetch human baseline texts.
Strategy:
  1. RAID 'extra' split contains original human source texts - use those
  2. Supplement with working HuggingFace datasets (parquet-based)
  3. Fall back to direct API for arXiv/PubMed
Output: appends human rows to data/corpus_raw.parquet
"""
import os, re, json, requests
import pandas as pd
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

RAID_REGISTER_MAP = {
    'reddit': 'social',
    'peerread': 'academic',
    'arxiv': 'academic',
    'wikihow': 'encyclopedic',
    'wikipedia': 'encyclopedic',
    'news': 'news',
    'book': 'creative',
    'essay': 'creative',
    'reuter': 'news',
    'yelp': 'social',
}

MIN_WORDS = {
    'academic': 100,
    'news': 150,
    'social': 30,
    'encyclopedic': 100,
    'creative': 100,
}


def word_count(text):
    return len(str(text).split())


def map_register(source_str):
    src = str(source_str).lower()
    for key, reg in RAID_REGISTER_MAP.items():
        if key in src:
            return reg
    return 'other'


def load_raid_human():
    """RAID extra split has human source texts."""
    print("Loading RAID human texts from 'extra' split...")
    rows = []
    try:
        ds = load_dataset("liamdugan/raid", split="extra")
        print(f"  RAID extra split: {len(ds)} rows")
        for row in tqdm(ds, desc="  RAID extra"):
            # Look for original/human text columns
            text = row.get('original', row.get('source_text', row.get('text', '')))
            if not text or not isinstance(text, str) or len(text.strip()) < 30:
                continue
            source = row.get('domain', row.get('source', row.get('dataset', 'unknown')))
            register = map_register(source)
            if register == 'other':
                continue
            min_w = MIN_WORDS.get(register, 30)
            if word_count(text) < min_w:
                continue
            rows.append({
                'text': text.strip(),
                'label': 0,
                'register': register,
                'model': 'human',
                'source': f'raid_human_{source}',
            })
        print(f"  RAID human rows: {len(rows)}")
    except Exception as e:
        print(f"  RAID extra failed: {e}")

    # Also check train split for human label
    try:
        ds = load_dataset("liamdugan/raid", split="train")
        human_count = 0
        for row in tqdm(ds, desc="  RAID train human"):
            model = row.get('model', '')
            if str(model).lower() not in ('human', 'original', ''):
                continue
            text = row.get('generation', row.get('text', ''))
            if not text or not isinstance(text, str):
                continue
            source = row.get('domain', row.get('source', 'unknown'))
            register = map_register(source)
            if register == 'other':
                continue
            min_w = MIN_WORDS.get(register, 30)
            if word_count(text) < min_w:
                continue
            rows.append({
                'text': text.strip(),
                'label': 0,
                'register': register,
                'model': 'human',
                'source': f'raid_train_human_{source}',
            })
            human_count += 1
        print(f"  RAID train human rows: {human_count}")
    except Exception as e:
        print(f"  RAID train human check failed: {e}")

    return pd.DataFrame(rows)


def load_working_hf_datasets():
    rows = []

    # Academic: allenai/peS2o (scientific papers, parquet-based)
    print("Loading academic: allenai/peS2o...")
    try:
        ds = load_dataset("allenai/peS2o", "v2", split="train", streaming=True)
        count = 0
        for row in tqdm(ds, desc="  peS2o"):
            text = row.get('text', '')
            if not text or word_count(text) < 100:
                continue
            # Take abstract portion (first 300 words)
            words = text.split()[:300]
            text = ' '.join(words)
            rows.append({'text': text.strip(), 'label': 0, 'register': 'academic',
                         'model': 'human', 'source': 'pes2o'})
            count += 1
            if count >= 100000:
                break
        print(f"  peS2o rows: {count}")
    except Exception as e:
        print(f"  peS2o failed: {e}")

    # Academic: scientific_papers (arXiv section)
    print("Loading academic: scientific_papers/arxiv...")
    try:
        ds = load_dataset("scientific_papers", "arxiv", split="train", streaming=True)
        count = 0
        for row in tqdm(ds, desc="  sci_papers arxiv"):
            text = row.get('abstract', '')
            if not text or word_count(text) < 100:
                continue
            rows.append({'text': text.strip(), 'label': 0, 'register': 'academic',
                         'model': 'human', 'source': 'scientific_papers_arxiv'})
            count += 1
            if count >= 100000:
                break
        print(f"  scientific_papers/arxiv rows: {count}")
    except Exception as e:
        print(f"  scientific_papers failed: {e}")
        # Try alternate
        try:
            ds = load_dataset("ccdv/arxiv-summarization", split="train", streaming=True)
            count = 0
            for row in tqdm(ds, desc="  arxiv summarization"):
                text = row.get('abstract', '')
                if not text or word_count(text) < 80:
                    continue
                rows.append({'text': text.strip(), 'label': 0, 'register': 'academic',
                             'model': 'human', 'source': 'arxiv_summarization'})
                count += 1
                if count >= 80000:
                    break
            print(f"  arxiv-summarization rows: {count}")
        except Exception as e2:
            print(f"  arxiv-summarization failed: {e2}")

    # News: cc_news (try parquet version)
    print("Loading news: cc_news...")
    try:
        ds = load_dataset("cc_news", split="train", streaming=True)
        count = 0
        for row in tqdm(ds, desc="  cc_news"):
            text = row.get('text', '')
            if not text or word_count(text) < 150:
                continue
            words = text.split()[:500]
            text = ' '.join(words)
            rows.append({'text': text.strip(), 'label': 0, 'register': 'news',
                         'model': 'human', 'source': 'cc_news'})
            count += 1
            if count >= 80000:
                break
        print(f"  cc_news rows: {count}")
    except Exception as e:
        print(f"  cc_news failed: {e}")
        # Try agnews
        try:
            ds = load_dataset("ag_news", split="train")
            count = 0
            for row in tqdm(ds, desc="  ag_news"):
                text = row.get('text', '')
                if not text or word_count(text) < 30:
                    continue
                rows.append({'text': text.strip(), 'label': 0, 'register': 'news',
                             'model': 'human', 'source': 'ag_news'})
                count += 1
                if count >= 50000:
                    break
            print(f"  ag_news rows: {count}")
        except Exception as e2:
            print(f"  ag_news failed: {e2}")

    # Social: reddit (via datasets that work)
    print("Loading social: reddit_eli5 / social...")
    try:
        ds = load_dataset("eli5_category", split="train", streaming=True)
        count = 0
        for row in tqdm(ds, desc="  eli5"):
            answers = row.get('answers', {})
            if isinstance(answers, dict):
                texts = answers.get('text', [])
            else:
                texts = []
            for text in texts:
                if text and word_count(text) >= 30:
                    rows.append({'text': text.strip(), 'label': 0, 'register': 'social',
                                 'model': 'human', 'source': 'eli5'})
                    count += 1
                    if count >= 80000:
                        break
            if count >= 80000:
                break
        print(f"  eli5 rows: {count}")
    except Exception as e:
        print(f"  eli5 failed: {e}")
        try:
            ds = load_dataset("rotten_tomatoes", split="train")
            count = 0
            for row in tqdm(ds, desc="  rotten_tomatoes"):
                text = row.get('text', '')
                if text and word_count(text) >= 20:
                    rows.append({'text': text.strip(), 'label': 0, 'register': 'social',
                                 'model': 'human', 'source': 'rotten_tomatoes'})
                    count += 1
            print(f"  rotten_tomatoes rows: {count}")
        except Exception as e2:
            print(f"  rotten_tomatoes failed: {e2}")

    # Encyclopedic: Wikipedia (parquet)
    print("Loading encyclopedic: wikipedia...")
    try:
        ds = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True)
        count = 0
        for row in tqdm(ds, desc="  wikipedia"):
            text = row.get('text', '')
            if not text or word_count(text) < 100:
                continue
            words = text.split()[:500]
            text = ' '.join(words)
            rows.append({'text': text.strip(), 'label': 0, 'register': 'encyclopedic',
                         'model': 'human', 'source': 'wikipedia'})
            count += 1
            if count >= 80000:
                break
        print(f"  wikipedia rows: {count}")
    except Exception as e:
        print(f"  wikimedia/wikipedia failed: {e}")
        try:
            ds = load_dataset("wikipedia", "20220301.en", split="train", streaming=True)
            count = 0
            for row in tqdm(ds, desc="  wikipedia 2022"):
                text = row.get('text', '')
                if not text or word_count(text) < 100:
                    continue
                words = text.split()[:500]
                text = ' '.join(words)
                rows.append({'text': text.strip(), 'label': 0, 'register': 'encyclopedic',
                             'model': 'human', 'source': 'wikipedia'})
                count += 1
                if count >= 80000:
                    break
            print(f"  wikipedia 2022 rows: {count}")
        except Exception as e2:
            print(f"  wikipedia 2022 failed: {e2}")

    # Creative: writing prompts
    print("Loading creative: writing_prompts...")
    try:
        ds = load_dataset("euclaise/writingprompts", split="train", streaming=True)
        count = 0
        for row in tqdm(ds, desc="  writingprompts"):
            text = row.get('story', row.get('text', ''))
            if not text or word_count(text) < 100:
                continue
            words = text.split()[:400]
            text = ' '.join(words)
            rows.append({'text': text.strip(), 'label': 0, 'register': 'creative',
                         'model': 'human', 'source': 'writingprompts'})
            count += 1
            if count >= 80000:
                break
        print(f"  writingprompts rows: {count}")
    except Exception as e:
        print(f"  writingprompts failed: {e}")
        try:
            ds = load_dataset("jtatman/writing-prompts-all-text", split="train", streaming=True)
            count = 0
            for row in tqdm(ds, desc="  writing_prompts_alt"):
                text = row.get('story', row.get('text', ''))
                if not text or word_count(text) < 100:
                    continue
                words = text.split()[:400]
                text = ' '.join(words)
                rows.append({'text': text.strip(), 'label': 0, 'register': 'creative',
                             'model': 'human', 'source': 'writingprompts_alt'})
                count += 1
                if count >= 50000:
                    break
            print(f"  writingprompts_alt rows: {count}")
        except Exception as e2:
            print(f"  writingprompts_alt failed: {e2}")

    df = pd.DataFrame(rows)
    print(f"  Total working HF human rows: {len(df)}")
    return df


def main():
    # Load existing AI corpus
    ai_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    if os.path.exists(ai_path):
        existing = pd.read_parquet(ai_path)
        print(f"Existing corpus: {len(existing)} rows (label 1: {sum(existing['label']==1)}, label 0: {sum(existing['label']==0)})")
    else:
        existing = pd.DataFrame()

    # Try RAID human texts
    raid_human = load_raid_human()

    # Try other working HF datasets
    hf_human = load_working_hf_datasets()

    # Combine all human
    all_human = pd.concat([raid_human, hf_human], ignore_index=True)
    all_human = all_human.drop_duplicates(subset=['text'])
    all_human = all_human[all_human['text'].str.len() > 50]
    print(f"\nTotal human texts: {len(all_human)}")
    if len(all_human) > 0:
        print(all_human.groupby(['register']).size().to_string())

    # Combine with existing AI corpus
    combined = pd.concat([existing, all_human], ignore_index=True)
    combined = combined.drop_duplicates(subset=['text'])
    print(f"\nFinal combined corpus: {len(combined)} rows")
    print(combined.groupby(['register', 'label']).size().to_string())

    out_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    combined.to_parquet(out_path, index=False)
    print(f"Saved to {out_path}")

    summary = combined.groupby(['register', 'label', 'source']).size().reset_index(name='count')
    summary.to_csv(os.path.join(DATA_DIR, 'corpus_summary.csv'), index=False)
    print("Updated corpus_summary.csv")


if __name__ == '__main__':
    main()
