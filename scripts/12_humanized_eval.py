#!/usr/bin/env python3
"""
Evaluate stylometric detector on humanized AI text.

"Humanized" AI text is AI-generated text that has been deliberately rewritten
to evade detection — either by prompting the model to write in a more human-
like style, or by post-processing with a paraphraser or style transfer tool.

This script:
1. Takes AI texts from the corpus and applies humanization transforms
2. Extracts features and runs the existing detector
3. Reports AUC degradation compared to clean AI text

Humanization strategies:
- Prompt-based: "Write this in a casual, conversational style with varied sentence length"
- Paraphrase via back-translation simulation (word reorder + synonym swap)
- Sentence splitting/merging
- Punctuation randomization
- Connector removal

Outputs:
  results/humanized_eval.csv
"""
import os
import sys
import json
import random
import re
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
MODELS_DIR = os.path.join(SCRIPT_DIR, '..', 'models')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_DIR)
from tool.feature_extractor import extract_feature_vector, ORIGINAL_FEATURE_COLS

FEATURE_COLS = ORIGINAL_FEATURE_COLS

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ── Humanization transforms ────────────────────────────────────────────────

CONNECTOR_WORDS = {
    'however', 'therefore', 'thus', 'hence', 'consequently', 'nevertheless',
    'furthermore', 'moreover', 'additionally', 'also', 'besides', 'likewise',
    'similarly', 'conversely', 'alternatively', 'subsequently', 'finally',
    'first', 'second', 'third', 'meanwhile', 'nonetheless', 'instead',
    'otherwise', 'accordingly',
}

SYNONYM_MAP = {
    'important': 'key', 'significant': 'notable', 'demonstrates': 'shows',
    'however': 'but', 'therefore': 'so', 'furthermore': 'also',
    'additionally': 'plus', 'consequently': 'as a result',
    'nevertheless': 'still', 'moreover': 'besides',
    'subsequently': 'then', 'alternatively': 'or',
    'utilize': 'use', 'commence': 'start', 'terminate': 'end',
    'facilitate': 'help', 'endeavor': 'try', 'ascertain': 'find out',
    'fundamental': 'basic', 'substantial': 'large', 'approximately': 'about',
    'sufficient': 'enough', 'numerous': 'many', 'additionally': 'also',
}


def remove_connectors(text):
    """Remove sentence-opening connectors to reduce opener_ratio."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for s in sents:
        s_lower = s.lower().strip()
        skip = False
        for cw in CONNECTOR_WORDS:
            if s_lower.startswith(cw + ' ') or s_lower.startswith(cw + ','):
                s = s[len(cw):].lstrip(', ').strip()
                s = s[0].upper() + s[1:] if s else s
                skip = True
                break
        result.append(s)
    return ' '.join(result)


def synonym_swap(text, swap_rate=0.3):
    """Replace formal words with casual synonyms."""
    words = text.split()
    for i, w in enumerate(words):
        w_lower = w.lower().strip('.,;:!?')
        if w_lower in SYNONYM_MAP and random.random() < swap_rate:
            replacement = SYNONYM_MAP[w_lower]
            if w[0].isupper():
                replacement = replacement[0].upper() + replacement[1:]
            words[i] = replacement
    return ' '.join(words)


def vary_sentence_length(text):
    """Split long sentences and merge short ones to increase sentence CV."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for s in sents:
        words = s.split()
        if len(words) > 25 and random.random() < 0.4:
            mid = len(words) // 2
            first = ' '.join(words[:mid]).rstrip(',;:') + '.'
            second = words[mid][0].upper() + ' '.join(words[mid+1:])
            result.append(first)
            result.append(second)
        else:
            result.append(s)

    # Merge some adjacent short sentences
    merged = []
    i = 0
    while i < len(result):
        if i + 1 < len(result) and len(result[i].split()) < 8 and len(result[i+1].split()) < 8 and random.random() < 0.3:
            merged.append(result[i].rstrip('.!?') + ', and ' + result[i+1].lower())
            i += 2
        else:
            merged.append(result[i])
            i += 1

    return ' '.join(merged)


def randomize_punctuation(text):
    """Randomly vary punctuation to increase punctuation entropy."""
    text = text.replace(', ', '; ') if random.random() < 0.2 else text
    text = text.replace('. ', '! ') if random.random() < 0.1 else text
    if random.random() < 0.15:
        text = text.replace('. ', '? ', 1)
    return text


def add_first_person(text, rate=0.15):
    """Insert first-person pronouns to increase self-mention density."""
    sents = re.split(r'(?<=[.!?])\s+', text)
    result = []
    for s in sents:
        if random.random() < rate:
            words = s.split()
            if len(words) > 3:
                insert_pos = min(3, len(words) - 1)
                words.insert(insert_pos, 'I')
                result.append(' '.join(words))
            else:
                result.append(s)
        else:
            result.append(s)
    return ' '.join(result)


def humanize_text(text, strategy='combined'):
    """Apply humanization transforms to AI text.

    Strategies:
        'remove_connectors' — only remove sentence-opening connectors
        'synonym_swap'      — only swap formal words for casual synonyms
        'vary_length'       — only vary sentence length
        'punctuation'       — only randomize punctuation
        'first_person'      — only add first-person pronouns
        'combined'          — apply all transforms
    """
    if strategy == 'remove_connectors':
        return remove_connectors(text)
    elif strategy == 'synonym_swap':
        return synonym_swap(text)
    elif strategy == 'vary_length':
        return vary_sentence_length(text)
    elif strategy == 'punctuation':
        return randomize_punctuation(text)
    elif strategy == 'first_person':
        return add_first_person(text)
    elif strategy == 'combined':
        text = remove_connectors(text)
        text = synonym_swap(text)
        text = vary_sentence_length(text)
        text = randomize_punctuation(text)
        text = add_first_person(text)
        return text
    else:
        raise ValueError(f"Unknown strategy: {strategy}")


def main():
    # Load models
    with open(os.path.join(MODELS_DIR, 'manifest.json')) as f:
        manifest = json.load(f)

    all_detector = joblib.load(os.path.join(MODELS_DIR, manifest['all_register_detector']))
    all_detector = all_detector['model'] if isinstance(all_detector, dict) else all_detector

    # Load features
    feat_path = os.path.join(DATA_DIR, 'corpus_features.parquet')
    if not os.path.exists(feat_path):
        print(f"ERROR: {feat_path} not found.")
        return

    df = pd.read_parquet(feat_path)
    print(f"Loaded {len(df)} texts from corpus_features.parquet")

    # We need raw text for humanization — load from raw parquet
    raw_path = os.path.join(DATA_DIR, 'corpus_raw.parquet')
    if not os.path.exists(raw_path):
        print("ERROR: Need raw text for humanization. corpus_raw.parquet not found.")
        return

    # Load only text + label from raw corpus (columnar read is fast)
    print("Loading text from raw corpus...")
    df_raw = pd.read_parquet(raw_path, columns=['text', 'label'])

    # Sample AI texts from raw corpus (we re-extract features after humanization)
    ai_available = df_raw[df_raw['label'] == 1].dropna(subset=['text'])
    ai_df = ai_available.sample(min(2000, len(ai_available)), random_state=RANDOM_SEED)

    # Sample human texts from features parquet (use pre-computed features)
    human_df = df[df['label'] == 0].dropna(subset=FEATURE_COLS).sample(
        min(2000, len(df[df['label'] == 0])), random_state=RANDOM_SEED
    )

    del df_raw

    print(f"AI texts: {len(ai_df)}, Human texts: {len(human_df)}")

    strategies = ['remove_connectors', 'synonym_swap', 'vary_length',
                  'punctuation', 'first_person', 'combined']
    results = []

    # Baseline: clean AI vs human
    clean_ai_feats = []
    for _, row in ai_df.iterrows():
        fv = extract_feature_vector(row['text'], feature_cols=FEATURE_COLS, extended=False)
        if fv is not None:
            clean_ai_feats.append(fv)

    human_feats = human_df[FEATURE_COLS].values
    y_human = np.zeros(len(human_feats))

    X_clean = np.array(clean_ai_feats)
    y_clean = np.ones(len(X_clean))

    X_all_clean = np.vstack([X_clean, human_feats])
    y_all_clean = np.concatenate([y_clean, y_human])
    y_proba_clean = all_detector.predict_proba(X_all_clean)[:, 1]
    auc_clean = roc_auc_score(y_all_clean, y_proba_clean)

    results.append({
        'strategy': 'clean (baseline)',
        'auc': auc_clean,
        'auc_delta': 0.0,
        'n_ai': len(X_clean),
        'n_human': len(human_feats),
    })
    print(f"\nClean baseline: AUC={auc_clean:.4f}")

    # Humanized variants
    for strategy in strategies:
        print(f"\nHumanizing with '{strategy}'...")
        humanized_feats = []
        for _, row in ai_df.iterrows():
            humanized = humanize_text(row['text'], strategy=strategy)
            fv = extract_feature_vector(humanized, feature_cols=FEATURE_COLS, extended=False)
            if fv is not None:
                humanized_feats.append(fv)

        if not humanized_feats:
            print(f"  No valid features extracted for {strategy}")
            continue

        X_humanized = np.array(humanized_feats)
        X_all = np.vstack([X_humanized, human_feats])
        y_all = np.concatenate([np.ones(len(X_humanized)), y_human])
        y_proba = all_detector.predict_proba(X_all)[:, 1]
        auc = roc_auc_score(y_all, y_proba)
        delta = auc - auc_clean

        results.append({
            'strategy': strategy,
            'auc': auc,
            'auc_delta': delta,
            'n_ai': len(X_humanized),
            'n_human': len(human_feats),
        })
        print(f"  AUC={auc:.4f} (delta={delta:+.4f})")

    # Save
    out_df = pd.DataFrame(results)
    out_path = os.path.join(RESULTS_DIR, 'humanized_eval.csv')
    out_df.to_csv(out_path, index=False)
    print(f"\nSaved to {out_path}")

    print("\n" + "=" * 60)
    print(f"{'Strategy':<25} {'AUC':>8} {'Delta':>8}")
    print("-" * 60)
    for _, row in out_df.iterrows():
        print(f"{row['strategy']:<25} {row['auc']:>8.4f} {row['auc_delta']:>+8.4f}")


if __name__ == '__main__':
    main()
