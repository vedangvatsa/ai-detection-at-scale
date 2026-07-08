#!/usr/bin/env python3
"""
Adversarial robustness evaluation: extract features from RAID texts with
adversarial attacks, then test the trained classifier on them.

RAID attack types: none, whitespace, upper_lower, synonym, 
perplexity_misspelling, paraphrase, number, insert_paragraphs, homoglyph

We fetch a sample of attacked AI texts from RAID, extract the same 11 features,
and evaluate the classifier trained on non-attacked texts.

Outputs:
  results/adversarial_results.csv
  results/adversarial_by_attack.csv
"""
import os, sys, time, warnings, re, math
import numpy as np
import pandas as pd
from collections import Counter
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score
from sklearn.model_selection import train_test_split
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
DATA_DIR = os.path.join(PROJECT_DIR, 'data')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

sys.path.insert(0, PROJECT_DIR)
from tool.feature_extractor import ORIGINAL_FEATURE_COLS

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_COLS = ORIGINAL_FEATURE_COLS

# Word lists from the feature extraction script
HEDGE_WORDS = set('may might maybe perhaps possibly probably likely appears seems suggests could would should about approximate approximately virtually virtually generally relatively somewhat rather quite nearly almost apparently presumably supposedly allegedly reportedly presumably'.split())
BOOSTER_WORDS = set('clearly obviously definitely certainly evidently indeed truly always never absolutely completely totally entirely unquestionably undeniably undoubtedly proves demonstrates establishes confirms shows reveals'.split())
CONNECTOR_WORDS = set('however therefore moreover furthermore nevertheless nonetheless thus hence consequently accordingly additionally also besides furthermore indeed instead meanwhile otherwise therefore thus rather alternatively subsequently subsequently finally first second third next then'.split())
SELF_MENTION_WORDS = set('i me my mine we us our ours myself ourselves'.split())
STOPWORDS = set('the a an and or but in on at to for of with by from as is are was were be been being have has had do does did will would could should may might must can this that these those it its they them their there here where when why how what which who whom whose'.split())

SENT_SPLIT = re.compile(r'[.!?]+(?:\s|$)|\n+')
WORD_SPLIT = re.compile(r"[a-zA-Z]+(?:'[a-z]+)?")


def tokenize_words(text):
    return [w.lower() for w in WORD_SPLIT.findall(text)]


def tokenize_sents(text):
    sents = SENT_SPLIT.split(text.strip())
    return [s.strip() for s in sents if len(s.strip()) > 0]


def compute_mtld(words, threshold=0.72):
    if len(words) < 10:
        return np.nan
    def factor_pass(w):
        types = set()
        n_factors = 0
        n_tokens = 0
        for word in w:
            types.add(word)
            n_tokens += 1
            ttr = len(types) / n_tokens
            if ttr <= threshold:
                n_factors += 1
                types = set()
                n_tokens = 0
        if n_tokens > 0:
            remainder_ttr = len(types) / n_tokens
            if remainder_ttr > threshold:
                n_factors += (1 - remainder_ttr) / (1 - threshold)
        return n_factors
    forward = factor_pass(words)
    backward = factor_pass(words[::-1])
    total = forward + backward
    if total == 0:
        return 0
    return len(words) / (total / 2)


def extract_features(text):
    if not text or len(text.strip()) < 50:
        return None
    words = tokenize_words(text)
    sents = tokenize_sents(text)
    if len(words) < 10 or len(sents) < 2:
        return None

    n_words = len(words)
    n_sents = len(sents)

    # MTLD
    mtld = compute_mtld(words)

    # Sentence lengths
    sent_lens = [len(tokenize_words(s)) for s in sents]
    mean_sent_len = np.mean(sent_lens)
    sent_cv = np.std(sent_lens) / mean_sent_len if mean_sent_len > 0 else 0

    # Self-mention density
    self_count = sum(1 for w in words if w in SELF_MENTION_WORDS)
    self_mention = self_count / n_words * 1000

    # Connector density
    conn_count = sum(1 for w in words if w in CONNECTOR_WORDS)
    connector_density = conn_count / n_words * 1000

    # Opener ratio
    openers = 0
    for s in sents:
        s_words = tokenize_words(s)
        if s_words and s_words[0] in CONNECTOR_WORDS:
            openers += 1
    opener_ratio = openers / n_sents if n_sents > 0 else 0

    # Hedge density
    hedge_count = sum(1 for w in words if w in HEDGE_WORDS)
    hedge_density = hedge_count / n_words * 1000

    # Booster density
    boost_count = sum(1 for w in words if w in BOOSTER_WORDS)
    boost_density = boost_count / n_words * 1000

    # Char entropy
    from collections import Counter
    trigrams = [text[i:i+3] for i in range(len(text) - 2)]
    if trigrams:
        counts = Counter(trigrams)
        total = len(trigrams)
        probs = [c / total for c in counts.values()]
        char_entropy = -sum(p * math.log2(p) for p in probs)
    else:
        char_entropy = 0

    # Repetition rate
    content_words = [w for w in words if w not in STOPWORDS]
    if content_words:
        unique_content = set(content_words)
        repeated = sum(1 for w in unique_content if content_words.count(w) > 1)
        rep_rate = repeated / len(unique_content)
    else:
        rep_rate = 0

    # Punctuation entropy
    puncts = [c for c in text if c in '.,;:!?()[]"\'-']
    if puncts:
        counts = Counter(puncts)
        total = len(puncts)
        probs = [c / total for c in counts.values()]
        punct_entropy = -sum(p * math.log2(p) for p in probs)
    else:
        punct_entropy = 0

    return {
        'mtld': mtld,
        'sent_cv': sent_cv,
        'self_mention_density': self_mention,
        'opener_ratio': opener_ratio,
        'connector_density': connector_density,
        'hedge_density': hedge_density,
        'mean_sent_len': mean_sent_len,
        'boost_density': boost_density,
        'char_entropy': char_entropy,
        'rep_rate': rep_rate,
        'punct_entropy': punct_entropy,
        'n_words': n_words,
        'n_sents': n_sents,
    }


def main():
    print("=== Adversarial Robustness Evaluation ===")

    # Load existing features for training
    in_path = os.path.join(DATA_DIR, 'corpus_features.parquet')
    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df)} existing feature rows for training")

    # Train classifier on non-attacked data (balanced sample)
    from sklearn.model_selection import StratifiedKFold
    parts = []
    for (reg, lab), grp in df.groupby(['register', 'label']):
        if len(grp) > 5000:
            parts.append(grp.sample(5000, random_state=RANDOM_SEED))
        else:
            parts.append(grp)
    df_sample = pd.concat(parts).sample(frac=1, random_state=RANDOM_SEED)
    df_clean = df_sample.dropna(subset=FEATURE_COLS)

    X_train = df_clean[FEATURE_COLS].values
    y_train = df_clean['label'].values

    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    print(f"Trained classifier on {len(X_train)} texts")

    # Now fetch adversarial texts from RAID
    print("\nFetching adversarial texts from RAID (liamdugan/raid)...")
    from datasets import load_dataset

    # We need texts with attack != 'none' and also attack == 'none' for comparison
    attack_data = {attack: [] for attack in [
        'none', 'whitespace', 'upper_lower', 'synonym',
        'perplexity_misspelling', 'paraphrase', 'number',
        'insert_paragraphs', 'homoglyph'
    ]}

    target_per_attack = 500  # texts per attack type
    domains_needed = {'abstracts': 'academic', 'wiki': 'encyclopedic',
                      'reddit': 'social', 'creative': 'creative',
                      'news': 'news'}

    ds = load_dataset('liamdugan/raid', split='train', streaming=True)

    n_scanned = 0
    for row in ds:
        attack = row['attack']
        domain = row['domain']
        model = row['model']
        text = row['generation']

        if attack not in attack_data:
            continue

        # Only take AI texts (skip human for adversarial - we want to test if attacked AI evades detection)
        if model == 'human':
            # But we need some human texts for comparison
            if attack == 'none' and len(attack_data['none']) < target_per_attack * 2:
                attack_data['none'].append({
                    'text': text, 'attack': attack, 'model': model,
                    'domain': domain, 'label': 0
                })
            continue

        if len(attack_data[attack]) < target_per_attack:
            attack_data[attack].append({
                'text': text, 'attack': attack, 'model': model,
                'domain': domain, 'label': 1
            })

        n_scanned += 1
        if n_scanned % 10000 == 0:
            counts = {k: len(v) for k, v in attack_data.items()}
            print(f"  Scanned {n_scanned}: {counts}")

        # Check if we have enough
        if all(len(v) >= target_per_attack for k, v in attack_data.items() if k != 'none'):
            if len(attack_data['none']) >= target_per_attack * 2:
                break

    print(f"\nCollected texts per attack type:")
    for attack, texts in attack_data.items():
        print(f"  {attack}: {len(texts)}")

    # Extract features from adversarial texts
    print("\nExtracting features from adversarial texts...")
    all_features = []
    for attack, texts in attack_data.items():
        for item in texts:
            feats = extract_features(item['text'])
            if feats:
                row = {
                    'attack': item['attack'],
                    'model': item['model'],
                    'domain': item['domain'],
                    'label': item['label'],
                    **feats,
                }
                all_features.append(row)

    adv_df = pd.DataFrame(all_features)
    print(f"Extracted features for {len(adv_df)} adversarial texts")

    # Evaluate classifier on each attack type
    print("\nEvaluating classifier on each attack type...")
    results = []

    # For each attack type, test AI texts against the classifier
    # We need human texts as the negative class
    human_feats = adv_df[adv_df['label'] == 0].dropna(subset=FEATURE_COLS)

    for attack in sorted(adv_df['attack'].unique()):
        ai_attack = adv_df[(adv_df['attack'] == attack) & (adv_df['label'] == 1)].dropna(subset=FEATURE_COLS)
        if len(ai_attack) < 10:
            continue

        # Combine with human texts
        combined = pd.concat([human_feats, ai_attack])
        X_test = combined[FEATURE_COLS].values
        y_test = combined['label'].values

        if len(np.unique(y_test)) < 2:
            continue

        probs = clf.predict_proba(X_test)[:, 1]
        preds = clf.predict(X_test)

        auc = roc_auc_score(y_test, probs) if len(np.unique(y_test)) > 1 else np.nan
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)

        # Also measure: what fraction of AI texts are still detected at threshold for 90% recall on clean data?
        # Get threshold from clean data
        clean_ai = adv_df[(adv_df['attack'] == 'none') & (adv_df['label'] == 1)].dropna(subset=FEATURE_COLS)
        if len(clean_ai) > 10:
            clean_combined = pd.concat([human_feats, clean_ai])
            X_clean = clean_combined[FEATURE_COLS].values
            y_clean = clean_combined['label'].values
            clean_probs = clf.predict_proba(X_clean)[:, 1]
            from sklearn.metrics import precision_recall_curve
            prec, rec, thresh = precision_recall_curve(y_clean, clean_probs)
            idx_90 = np.argmin(np.abs(rec - 0.90))
            t_90 = thresh[idx_90] if idx_90 < len(thresh) else 0.5
            # Apply this threshold to attacked texts
            attack_probs = clf.predict_proba(ai_attack[FEATURE_COLS].values)[:, 1]
            detection_rate = np.mean(attack_probs >= t_90)
        else:
            detection_rate = np.nan
            t_90 = np.nan

        results.append({
            'attack_type': attack,
            'n_ai_texts': len(ai_attack),
            'n_human_texts': len(human_feats),
            'auc': auc,
            'accuracy': acc,
            'f1': f1,
            'detection_rate_at_clean_90_recall': detection_rate,
            'clean_threshold_90': t_90,
        })
        print(f"  {attack:30s} AUC={auc:.3f} F1={f1:.3f} det_rate={detection_rate:.3f}")

    result_df = pd.DataFrame(results)
    result_df.to_csv(os.path.join(RESULTS_DIR, 'adversarial_results.csv'), index=False)
    print(f"\nSaved adversarial_results.csv ({len(result_df)} rows)")

    # Also save the extracted adversarial features for potential later use
    adv_df.to_csv(os.path.join(RESULTS_DIR, 'adversarial_features.csv'), index=False)
    print(f"Saved adversarial_features.csv ({len(adv_df)} rows)")

    print("\nAdversarial evaluation complete.")


if __name__ == '__main__':
    main()
