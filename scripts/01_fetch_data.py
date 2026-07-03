#!/usr/bin/env python3
"""
Fetch RAID dataset (AI texts, 11 models) and human baselines from HuggingFace.
Outputs: data/raid_sample.parquet, data/human_sample.parquet
Each row: text, label (0=human, 1=AI), register, model (AI only), source
"""
import os, sys, json, re, random
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)

RANDOM_SEED = 42
random.seed(RANDOM_SEED)

# Register mapping for RAID sources
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
    return len(text.split())


def map_register(source_str):
    src = str(source_str).lower()
    for key, reg in RAID_REGISTER_MAP.items():
        if key in src:
            return reg
    return 'other'


def load_raid():
    print("Loading RAID dataset from HuggingFace...")
    try:
        ds = load_dataset("liamdugan/raid", split="train", trust_remote_code=True)
        print(f"  RAID train split: {len(ds)} rows")
    except Exception as e:
        print(f"  Error loading RAID train: {e}")
        try:
            ds = load_dataset("liamdugan/raid", trust_remote_code=True)
            split_name = list(ds.keys())[0]
            ds = ds[split_name]
            print(f"  RAID loaded via split '{split_name}': {len(ds)} rows")
        except Exception as e2:
            print(f"  Fatal: could not load RAID: {e2}")
            return pd.DataFrame()

    rows = []
    print("  Processing RAID rows...")
    for row in tqdm(ds):
        text = row.get('generation', row.get('text', row.get('passage', '')))
        if not text or not isinstance(text, str):
            continue
        source = row.get('domain', row.get('source', row.get('dataset', 'unknown')))
        model = row.get('model', 'unknown')
        register = map_register(source)
        if register == 'other':
            continue
        min_w = MIN_WORDS.get(register, 50)
        if word_count(text) < min_w:
            continue
        rows.append({
            'text': text.strip(),
            'label': 1,
            'register': register,
            'model': model,
            'source': f'raid_{source}',
        })
    df = pd.DataFrame(rows)
    print(f"  RAID AI texts kept: {len(df)}")
    return df


def load_human_baselines():
    rows = []

    # 1. Academic: PubMed (pre-2022 abstracts via HuggingFace)
    print("Loading academic human texts (PubMed via HuggingFace)...")
    try:
        ds = load_dataset("pubmed_qa", "pqa_labeled", split="train", trust_remote_code=True)
        for row in tqdm(ds, desc="  PubMed"):
            ctx = row.get('context', {})
            if isinstance(ctx, dict):
                texts = ctx.get('contexts', [])
                text = ' '.join(texts) if texts else ''
            else:
                text = str(ctx)
            if not text or word_count(text) < 100:
                continue
            rows.append({'text': text.strip(), 'label': 0, 'register': 'academic',
                         'model': 'human', 'source': 'pubmed_qa'})
        print(f"  PubMed rows added: {sum(1 for r in rows if r['source']=='pubmed_qa')}")
    except Exception as e:
        print(f"  PubMed failed: {e}")

    # 2. Academic: arXiv abstracts
    print("Loading academic human texts (arXiv)...")
    try:
        ds = load_dataset("arxiv_dataset", split="train", trust_remote_code=True)
        count = 0
        for row in tqdm(ds, desc="  arXiv"):
            text = row.get('abstract', '')
            if not text or word_count(text) < 100:
                continue
            # Only pre-2022
            upd = row.get('update_date', row.get('versions', ''))
            if isinstance(upd, str) and upd[:4].isdigit() and int(upd[:4]) >= 2022:
                continue
            rows.append({'text': text.strip(), 'label': 0, 'register': 'academic',
                         'model': 'human', 'source': 'arxiv'})
            count += 1
            if count >= 50000:
                break
        print(f"  arXiv rows added: {count}")
    except Exception as e:
        print(f"  arXiv failed: {e}. Trying alternate dataset name...")
        try:
            ds = load_dataset("tomasg25/scientific_lay_summarisation", split="train", trust_remote_code=True)
            count = 0
            for row in tqdm(ds, desc="  Sci summarisation"):
                text = row.get('abstract', row.get('text', ''))
                if not text or word_count(text) < 100:
                    continue
                rows.append({'text': text.strip(), 'label': 0, 'register': 'academic',
                             'model': 'human', 'source': 'sci_summarisation'})
                count += 1
                if count >= 30000:
                    break
            print(f"  Sci summarisation rows added: {count}")
        except Exception as e2:
            print(f"  arXiv alternate also failed: {e2}")

    # 3. News: CC-News via HuggingFace
    print("Loading news human texts (CC-News)...")
    try:
        ds = load_dataset("cc_news", split="train", trust_remote_code=True)
        count = 0
        for row in tqdm(ds, desc="  CC-News"):
            text = row.get('text', '')
            if not text or word_count(text) < 150:
                continue
            # Truncate very long articles to first 500 words
            words = text.split()[:500]
            text = ' '.join(words)
            rows.append({'text': text.strip(), 'label': 0, 'register': 'news',
                         'model': 'human', 'source': 'cc_news'})
            count += 1
            if count >= 50000:
                break
        print(f"  CC-News rows added: {count}")
    except Exception as e:
        print(f"  CC-News failed: {e}")
        try:
            ds = load_dataset("RealTimeData/bbc_news_alltime", split="train", trust_remote_code=True)
            count = 0
            for row in tqdm(ds, desc="  BBC News"):
                text = row.get('content', row.get('text', ''))
                if not text or word_count(text) < 150:
                    continue
                words = text.split()[:500]
                text = ' '.join(words)
                rows.append({'text': text.strip(), 'label': 0, 'register': 'news',
                             'model': 'human', 'source': 'bbc_news'})
                count += 1
                if count >= 30000:
                    break
            print(f"  BBC News rows added: {count}")
        except Exception as e2:
            print(f"  News alternate also failed: {e2}")

    # 4. Social: Reddit (via pushshift/HuggingFace)
    print("Loading social human texts (Reddit)...")
    try:
        ds = load_dataset("webis/tldr-17", split="train", trust_remote_code=True)
        count = 0
        for row in tqdm(ds, desc="  Reddit TLDR"):
            text = row.get('content', row.get('body', ''))
            if not text or word_count(text) < 30:
                continue
            rows.append({'text': text.strip(), 'label': 0, 'register': 'social',
                         'model': 'human', 'source': 'reddit_tldr'})
            count += 1
            if count >= 50000:
                break
        print(f"  Reddit rows added: {count}")
    except Exception as e:
        print(f"  Reddit failed: {e}")

    # 5. Encyclopedic: Wikipedia
    print("Loading encyclopedic human texts (Wikipedia)...")
    try:
        ds = load_dataset("wikipedia", "20220301.en", split="train", trust_remote_code=True)
        count = 0
        for row in tqdm(ds, desc="  Wikipedia"):
            text = row.get('text', '')
            if not text or word_count(text) < 100:
                continue
            # Take first 500 words of each article
            words = text.split()[:500]
            text = ' '.join(words)
            rows.append({'text': text.strip(), 'label': 0, 'register': 'encyclopedic',
                         'model': 'human', 'source': 'wikipedia'})
            count += 1
            if count >= 50000:
                break
        print(f"  Wikipedia rows added: {count}")
    except Exception as e:
        print(f"  Wikipedia failed: {e}")

    # 6. Creative: BookCorpus
    print("Loading creative human texts (BookCorpus)...")
    try:
        ds = load_dataset("bookcorpus", split="train", trust_remote_code=True)
        count = 0
        buf = []
        for row in tqdm(ds, desc="  BookCorpus"):
            sent = row.get('text', '')
            if not sent:
                continue
            buf.append(sent)
            if len(' '.join(buf).split()) >= 150:
                text = ' '.join(buf)
                rows.append({'text': text.strip(), 'label': 0, 'register': 'creative',
                             'model': 'human', 'source': 'bookcorpus'})
                buf = []
                count += 1
                if count >= 30000:
                    break
        print(f"  BookCorpus rows added: {count}")
    except Exception as e:
        print(f"  BookCorpus failed: {e}")

    df = pd.DataFrame(rows)
    print(f"  Total human rows: {len(df)}")
    return df


def main():
    ai_df = load_raid()
    human_df = load_human_baselines()

    # Also load M4 if available (supplementary AI)
    print("Trying supplementary AI datasets (M4, HC3)...")
    supp_rows = []
    try:
        ds = load_dataset("NicolaiSivesind/ChatGPT-Research-Abstracts", split="train", trust_remote_code=True)
        for row in tqdm(ds, desc="  ChatGPT abstracts"):
            text = row.get('generated_abstract', row.get('text', ''))
            if text and word_count(text) >= 100:
                supp_rows.append({'text': text.strip(), 'label': 1, 'register': 'academic',
                                  'model': 'gpt-3.5-turbo', 'source': 'chatgpt_abstracts'})
        print(f"  ChatGPT abstracts added: {len(supp_rows)}")
    except Exception as e:
        print(f"  ChatGPT abstracts failed: {e}")

    try:
        ds = load_dataset("Hello-SimpleAI/HC3", "all", split="train", trust_remote_code=True)
        count = 0
        for row in tqdm(ds, desc="  HC3"):
            for answer in row.get('chatgpt_answers', []):
                if answer and word_count(answer) >= 50:
                    supp_rows.append({'text': answer.strip(), 'label': 1, 'register': 'social',
                                      'model': 'gpt-3.5-turbo', 'source': 'hc3'})
                    count += 1
        print(f"  HC3 rows added: {count}")
    except Exception as e:
        print(f"  HC3 failed: {e}")

    if supp_rows:
        supp_df = pd.DataFrame(supp_rows)
        ai_df = pd.concat([ai_df, supp_df], ignore_index=True)

    # Combine
    all_df = pd.concat([ai_df, human_df], ignore_index=True)
    all_df = all_df.drop_duplicates(subset=['text'])
    all_df = all_df[all_df['text'].str.len() > 50]

    print(f"\nFinal dataset: {len(all_df)} texts")
    print(all_df.groupby(['register', 'label']).size().to_string())

    out_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    all_df.to_parquet(out_path, index=False)
    print(f"\nSaved to {out_path}")

    # Save summary
    summary = all_df.groupby(['register', 'label', 'source']).size().reset_index(name='count')
    summary.to_csv(os.path.join(DATA_DIR, 'corpus_summary.csv'), index=False)
    print("Saved corpus_summary.csv")


if __name__ == '__main__':
    main()
