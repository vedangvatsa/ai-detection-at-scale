#!/usr/bin/env python3
"""
Fast parallel feature extraction using multiprocessing.
Processes 2.3M texts in ~10-15 minutes using all CPU cores.
"""
import os, re, math, string, sys
import pandas as pd
import numpy as np
from collections import Counter
from multiprocessing import Pool, cpu_count

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

HEDGE_WORDS = frozenset({
    'may', 'might', 'could', 'possibly', 'perhaps', 'probably', 'likely',
    'appears', 'seems', 'suggests', 'indicates', 'tentatively', 'presumably',
    'arguably', 'approximately', 'generally', 'often', 'sometimes', 'tend',
    'tends', 'tended', 'appear', 'suggest', 'indicate', 'assume', 'assumes',
    'assumed', 'possible', 'potential', 'potentially', 'conceivably', 'loosely',
    'roughly', 'around', 'about', 'somewhat', 'relatively', 'seemingly',
})

BOOSTER_WORDS = frozenset({
    'clearly', 'obviously', 'undoubtedly', 'certainly', 'definitely',
    'demonstrates', 'demonstrate', 'proves', 'prove', 'establishes',
    'establish', 'confirms', 'confirm', 'shows', 'always', 'never',
    'absolutely', 'conclusively', 'evidently', 'indeed', 'strongly',
    'unambiguously', 'fundamentally', 'necessarily', 'undeniably',
})

CONNECTOR_WORDS = frozenset({
    'however', 'therefore', 'thus', 'hence', 'consequently', 'nevertheless',
    'furthermore', 'moreover', 'additionally', 'also', 'besides', 'likewise',
    'similarly', 'conversely', 'alternatively', 'subsequently', 'finally',
    'firstly', 'secondly', 'thirdly', 'lastly', 'meanwhile', 'nonetheless',
    'instead', 'otherwise', 'accordingly', 'specifically', 'notably',
    'importantly', 'significantly', 'overall', 'ultimately', 'essentially',
    'previously', 'subsequently', 'initially', 'additionally',
})

SELF_MENTION_WORDS = frozenset({'we', 'our', 'us', 'i', 'my', 'me', 'myself', 'ourselves'})

STOPWORDS = frozenset({
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'shall', 'can', 'of', 'in', 'to', 'for',
    'with', 'on', 'at', 'by', 'from', 'and', 'or', 'but', 'not', 'it',
    'its', 'this', 'that', 'these', 'those', 'as', 'if', 'so', 'than',
    'then', 'when', 'where', 'which', 'who', 'what', 'how', 'all', 'more',
    'most', 'any', 'each', 'no', 'into', 'up', 'out', 'about', 'through',
    'over', 'after', 'before', 'between', 'under', 'above', 'been', 'just',
    'also', 'both', 'very', 'too', 'only', 'even', 'back', 'there', 'here',
    'well', 'still', 'such', 'other', 'than', 'same', 'them', 'their',
    'they', 'he', 'she', 'him', 'his', 'her', 'his', 'its', 'our',
})

SENT_END_RE = re.compile(r'(?<=[.!?])\s+(?=[A-Z"\'(])')
WORD_RE = re.compile(r'\b[a-zA-Z]+\b')
CONNECTOR_OPENER_RE = {
    c: re.compile(r'^' + re.escape(c) + r'[\s,]', re.IGNORECASE)
    for c in CONNECTOR_WORDS
}
PUNCT_CHARS = frozenset('.,;:!?()[]{}"\'`-')


def mtld_forward(words, threshold=0.72):
    if len(words) < 10:
        return float(len(words))
    factor_count = 0.0
    token_count = 0
    types = set()
    for w in words:
        token_count += 1
        types.add(w)
        if len(types) / token_count <= threshold:
            factor_count += 1
            token_count = 0
            types = set()
    if token_count > 0:
        ttr = len(types) / token_count
        factor_count += (1.0 - ttr) / (1.0 - threshold)
    return len(words) / factor_count if factor_count > 0 else float(len(words))


def extract_row(args):
    text, label, register, model, source = args
    if not text or not isinstance(text, str) or len(text) < 30:
        return None

    words = WORD_RE.findall(text.lower())
    n_words = len(words)
    if n_words < 20:
        return None

    sents = SENT_END_RE.split(text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    if not sents:
        sents = [text.strip()]
    n_sents = len(sents)

    sent_lengths = []
    for s in sents:
        wc = len(WORD_RE.findall(s))
        if wc > 0:
            sent_lengths.append(wc)
    if not sent_lengths:
        return None

    mean_sl = float(np.mean(sent_lengths))
    sent_cv = (float(np.std(sent_lengths, ddof=1)) / mean_sl
               if len(sent_lengths) >= 2 and mean_sl > 0 else 0.0)

    # MTLD
    if n_words >= 50:
        mtld = (mtld_forward(words) + mtld_forward(words[::-1])) / 2.0
    else:
        return None  # skip short texts for MTLD stability

    wpm = n_words / 1000.0

    # Self-mention density
    self_count = sum(1 for w in words if w in SELF_MENTION_WORDS)
    self_density = self_count / max(wpm, 0.001)

    # Connector density
    conn_count = sum(1 for w in words if w in CONNECTOR_WORDS)
    conn_density = conn_count / max(wpm, 0.001)

    # Sentence-opener connector ratio
    opener_count = 0
    for s in sents:
        s_lower = s.lower().lstrip()
        for c in CONNECTOR_WORDS:
            pat = CONNECTOR_OPENER_RE[c]
            if pat.match(s_lower):
                opener_count += 1
                break
    opener_ratio = opener_count / max(n_sents, 1)

    # Hedge density
    hedge_count = sum(1 for w in words if w in HEDGE_WORDS)
    hedge_density = hedge_count / max(wpm, 0.001)

    # Booster density
    boost_count = sum(1 for w in words if w in BOOSTER_WORDS)
    boost_density = boost_count / max(wpm, 0.001)

    # Char n-gram entropy (trigrams)
    chars = re.sub(r'\s+', ' ', text.lower())
    if len(chars) >= 3:
        tg = [chars[i:i+3] for i in range(len(chars)-2)]
        tg_counts = Counter(tg)
        total = len(tg)
        char_entropy = -sum((c/total) * math.log2(c/total) for c in tg_counts.values())
    else:
        char_entropy = 0.0

    # Word repetition rate
    content = [w for w in words if w not in STOPWORDS and len(w) > 2]
    if content:
        wf = Counter(content)
        rep_rate = sum(1 for w in content if wf[w] > 1) / len(content)
    else:
        rep_rate = 0.0

    # Punctuation entropy
    puncts = [c for c in text if c in PUNCT_CHARS]
    if puncts:
        pf = Counter(puncts)
        total_p = len(puncts)
        punct_entropy = -sum((c/total_p) * math.log2(c/total_p) for c in pf.values())
    else:
        punct_entropy = 0.0

    return {
        'mean_sent_len': mean_sl,
        'sent_cv': sent_cv,
        'mtld': mtld,
        'self_mention_density': self_density,
        'connector_density': conn_density,
        'opener_ratio': opener_ratio,
        'hedge_density': hedge_density,
        'boost_density': boost_density,
        'char_entropy': char_entropy,
        'rep_rate': rep_rate,
        'punct_entropy': punct_entropy,
        'n_words': n_words,
        'n_sents': n_sents,
        'label': label,
        'register': register,
        'model': model,
        'source': source,
    }


def main():
    in_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found")
        sys.exit(1)

    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df)} texts from corpus_raw.parquet")

    # Also load per-register human shard files
    shard_dfs = []
    for register in ['news', 'academic', 'social', 'creative', 'encyclopedic']:
        for suffix in ['', '2']:
            fname = f'human_{register}{suffix}.parquet'
            sp = os.path.join(DATA_DIR, fname)
            if os.path.exists(sp):
                s = pd.read_parquet(sp)
                shard_dfs.append(s)
                print(f"  Loaded {len(s):,} texts from {fname}")

    if shard_dfs:
        shard_df = pd.concat(shard_dfs, ignore_index=True)
        # Add missing columns
        for col, default in [('label', 0), ('model', 'human_source'), ('source', 'shard')]:
            if col not in shard_df.columns:
                shard_df[col] = default
        df = pd.concat([df, shard_df], ignore_index=True)
        # Deduplicate on first 200 chars of text
        df['_key'] = df['text'].str[:200]
        df = df.drop_duplicates(subset=['_key']).drop(columns=['_key'])
        print(f"After merging shards + dedup: {len(df)} texts")

    if 'model' not in df.columns:
        df['model'] = 'unknown'
    df['model'] = df['model'].fillna('unknown')

    # Build args list
    args = list(zip(
        df['text'].tolist(),
        df['label'].tolist(),
        df['register'].tolist(),
        df['model'].tolist(),
        df['source'].tolist(),
    ))

    n_cores = max(1, cpu_count() - 1)
    print(f"Using {n_cores} cores for feature extraction...")

    CHUNK = 10000
    results = []
    total = len(args)

    with Pool(processes=n_cores) as pool:
        for i in range(0, total, CHUNK):
            chunk = args[i:i+CHUNK]
            chunk_results = pool.map(extract_row, chunk)
            results.extend(r for r in chunk_results if r is not None)
            pct = min(100, (i + CHUNK) / total * 100)
            print(f"  {pct:.1f}% done ({i+CHUNK}/{total}), kept so far: {len(results)}")

    print(f"Extracted features: {len(results)} / {total}")
    feat_df = pd.DataFrame(results)

    out_path = os.path.join(DATA_DIR, 'corpus_features.parquet')
    feat_df.to_parquet(out_path, index=False)
    print(f"Saved {len(feat_df)} rows to {out_path}")

    feat_cols = ['mean_sent_len', 'sent_cv', 'mtld', 'self_mention_density',
                 'connector_density', 'opener_ratio', 'hedge_density', 'boost_density',
                 'char_entropy', 'rep_rate', 'punct_entropy']
    print("\nFeature means by label:")
    print(feat_df.groupby('label')[feat_cols].mean().round(3).to_string())
    print("\nCounts by register x label:")
    print(feat_df.groupby(['register', 'label']).size().to_string())


if __name__ == '__main__':
    main()
