#!/usr/bin/env python3
"""
Extract 11 stylometric features from each text in corpus_raw.parquet.
Features: MTLD, sentence length CV, mean sentence length, self-mention density,
          sentence-opener connector ratio, connector density, hedge density,
          booster density, char ngram entropy, word repetition rate, punctuation entropy.
Output: data/corpus_features.parquet
"""
import os, re, math, string
import pandas as pd
import numpy as np
from tqdm import tqdm
from collections import Counter

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')

# ── Word lists ──────────────────────────────────────────────────────────────

HEDGE_WORDS = {
    'may', 'might', 'could', 'possibly', 'perhaps', 'probably', 'likely',
    'appears', 'seems', 'suggests', 'indicates', 'tentatively', 'presumably',
    'arguably', 'approximately', 'generally', 'often', 'sometimes', 'tend',
    'tends', 'tended', 'appear', 'suggest', 'indicate', 'assume', 'assumes',
    'assumed', 'appear to', 'seem to', 'likely to', 'possible', 'potential',
    'potentially', 'conceivably', 'presumably', 'loosely', 'roughly',
}

BOOSTER_WORDS = {
    'clearly', 'obviously', 'undoubtedly', 'certainly', 'definitely',
    'demonstrates', 'demonstrate', 'proves', 'prove', 'establishes',
    'establish', 'confirms', 'confirm', 'shows', 'always', 'never',
    'absolutely', 'conclusively', 'evidently', 'indeed', 'strongly',
    'unambiguously', 'fundamentally', 'necessarily', 'undeniably',
}

CONNECTOR_WORDS = {
    'however', 'therefore', 'thus', 'hence', 'consequently', 'nevertheless',
    'furthermore', 'moreover', 'additionally', 'also', 'besides', 'likewise',
    'similarly', 'conversely', 'alternatively', 'subsequently', 'finally',
    'first', 'second', 'third', 'firstly', 'secondly', 'thirdly', 'lastly',
    'meanwhile', 'nonetheless', 'instead', 'otherwise', 'accordingly',
    'for example', 'for instance', 'in contrast', 'in addition', 'in conclusion',
    'in summary', 'as a result', 'on the other hand', 'on the contrary',
}

SELF_MENTION_WORDS = {'we', 'our', 'us', 'i', 'my', 'me', 'myself', 'ourselves'}


# ── Tokenization ──────────────────────────────────────────────────────────

def tokenize_sentences(text):
    # Split on sentence-ending punctuation followed by space+capital or end
    sents = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'(])', text.strip())
    sents = [s.strip() for s in sents if s.strip()]
    return sents if sents else [text.strip()]


def tokenize_words(text):
    return re.findall(r'\b[a-zA-Z]+\b', text.lower())


# ── MTLD ──────────────────────────────────────────────────────────────────

def mtld_forward(words, threshold=0.72):
    """Single-pass MTLD calculation."""
    if len(words) < 10:
        return 0.0
    factor_count = 0.0
    token_count = 0
    types = set()
    start = 0
    for i, w in enumerate(words):
        token_count += 1
        types.add(w)
        ttr = len(types) / token_count
        if ttr <= threshold:
            factor_count += 1
            token_count = 0
            types = set()
            start = i + 1
    # Partial factor
    if token_count > 0:
        ttr = len(types) / token_count
        factor_count += (1.0 - ttr) / (1.0 - threshold)
    if factor_count == 0:
        return len(words)
    return len(words) / factor_count


def compute_mtld(words):
    if len(words) < 10:
        return np.nan
    forward = mtld_forward(words)
    backward = mtld_forward(list(reversed(words)))
    return (forward + backward) / 2.0


# ── Feature extraction ────────────────────────────────────────────────────

def extract_features(text):
    if not text or not isinstance(text, str):
        return None

    words = tokenize_words(text)
    if len(words) < 20:
        return None

    sents = tokenize_sentences(text)
    n_words = len(words)
    n_sents = len(sents)
    words_per_1000 = n_words / 1000.0

    # 1. Mean sentence length
    sent_lengths = [len(tokenize_words(s)) for s in sents if tokenize_words(s)]
    if not sent_lengths:
        return None
    mean_sent_len = np.mean(sent_lengths)

    # 2. Sentence length CV
    if len(sent_lengths) >= 2 and mean_sent_len > 0:
        sent_cv = np.std(sent_lengths, ddof=1) / mean_sent_len
    else:
        sent_cv = 0.0

    # 3. MTLD
    mtld = compute_mtld(words)

    # 4. Self-mention density (per 1000 words)
    self_count = sum(1 for w in words if w in SELF_MENTION_WORDS)
    self_density = self_count / max(words_per_1000, 0.001)

    # 5 & 6. Connector density + sentence-opener connector ratio
    text_lower = text.lower()
    conn_count = 0
    for cw in CONNECTOR_WORDS:
        conn_count += len(re.findall(r'\b' + re.escape(cw) + r'\b', text_lower))
    conn_density = conn_count / max(words_per_1000, 0.001)

    opener_count = 0
    for s in sents:
        s_lower = s.lower().strip()
        for cw in CONNECTOR_WORDS:
            if s_lower.startswith(cw + ' ') or s_lower.startswith(cw + ','):
                opener_count += 1
                break
    opener_ratio = opener_count / max(n_sents, 1)

    # 7. Hedge density
    hedge_count = sum(1 for w in words if w in HEDGE_WORDS)
    # Also check multi-word hedges
    for mw in ['appear to', 'seem to', 'likely to', 'for example', 'for instance']:
        hedge_count += len(re.findall(r'\b' + re.escape(mw) + r'\b', text_lower))
    hedge_density = hedge_count / max(words_per_1000, 0.001)

    # 8. Booster density
    boost_count = sum(1 for w in words if w in BOOSTER_WORDS)
    boost_density = boost_count / max(words_per_1000, 0.001)

    # 9. Character n-gram entropy (trigrams)
    chars = re.sub(r'\s+', ' ', text.lower())
    if len(chars) >= 3:
        trigrams = [chars[i:i+3] for i in range(len(chars)-2)]
        tg_counts = Counter(trigrams)
        total = sum(tg_counts.values())
        char_entropy = -sum((c/total) * math.log2(c/total) for c in tg_counts.values())
    else:
        char_entropy = 0.0

    # 10. Within-document word repetition rate
    # Ratio of words that appear more than once (excluding stopwords)
    stopwords = {'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been',
                 'being', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
                 'would', 'could', 'should', 'may', 'might', 'shall', 'can',
                 'of', 'in', 'to', 'for', 'with', 'on', 'at', 'by', 'from',
                 'and', 'or', 'but', 'not', 'it', 'its', 'this', 'that',
                 'these', 'those', 'as', 'if', 'so', 'than', 'then', 'when',
                 'where', 'which', 'who', 'what', 'how', 'all', 'more', 'most'}
    content_words = [w for w in words if w not in stopwords and len(w) > 2]
    if content_words:
        wf = Counter(content_words)
        repeated = sum(1 for w in content_words if wf[w] > 1)
        rep_rate = repeated / len(content_words)
    else:
        rep_rate = 0.0

    # 11. Punctuation entropy
    puncts = [c for c in text if c in '.,;:!?()[]{}"\'-']
    if puncts:
        pf = Counter(puncts)
        total_p = len(puncts)
        punct_entropy = -sum((c/total_p) * math.log2(c/total_p) for c in pf.values())
        punct_per_sent = total_p / max(n_sents, 1)
    else:
        punct_entropy = 0.0
        punct_per_sent = 0.0

    return {
        'mean_sent_len': mean_sent_len,
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
    }


def main():
    in_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run 01_fetch_data.py first.")
        return

    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df)} texts from corpus_raw.parquet")

    feature_rows = []
    failed = 0
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Extracting features"):
        feats = extract_features(row['text'])
        if feats is None:
            failed += 1
            continue
        feats['label'] = row['label']
        feats['register'] = row['register']
        feats['model'] = row['model']
        feats['source'] = row['source']
        feature_rows.append(feats)

    print(f"  Extracted: {len(feature_rows)}, Failed/skipped: {failed}")
    feat_df = pd.DataFrame(feature_rows)

    # Drop rows with NaN MTLD (texts too short)
    before = len(feat_df)
    feat_df = feat_df.dropna(subset=['mtld'])
    print(f"  Dropped {before - len(feat_df)} rows with NaN MTLD")

    out_path = os.path.join(DATA_DIR, 'corpus_features.parquet')
    feat_df.to_parquet(out_path, index=False)
    print(f"Saved {len(feat_df)} feature rows to {out_path}")

    # Quick sanity check
    print("\nFeature means by label:")
    feat_cols = ['mean_sent_len', 'sent_cv', 'mtld', 'self_mention_density',
                 'connector_density', 'opener_ratio', 'hedge_density', 'boost_density',
                 'char_entropy', 'rep_rate', 'punct_entropy']
    print(feat_df.groupby('label')[feat_cols].mean().round(3).to_string())
    print("\nCounts by register x label:")
    print(feat_df.groupby(['register', 'label']).size().to_string())


if __name__ == '__main__':
    main()
