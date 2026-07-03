#!/usr/bin/env python3
"""
Length-deconfounded analysis: match human and AI texts by word-count bins
per register, then re-run effect sizes, classifiers, cross-domain AUC,
ablation, feature importance, and calibration.

Outputs:
  results/effect_sizes_matched.csv
  results/cross_domain_auc_matched.csv
  results/classifier_results_matched.csv
  results/ablation_matched.csv
  results/calibration_data_matched.csv
  results/feature_importance_matched.csv
"""
import os, warnings
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.metrics import (roc_auc_score, accuracy_score, precision_score,
                             recall_score, f1_score, brier_score_loss)
from sklearn.calibration import calibration_curve
from sklearn.inspection import permutation_importance
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, '..', 'data')
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

RANDOM_SEED = 42
np.random.seed(RANDOM_SEED)

FEATURE_COLS = [
    'mtld', 'sent_cv', 'self_mention_density', 'opener_ratio',
    'connector_density', 'hedge_density', 'mean_sent_len', 'boost_density',
    'char_entropy', 'rep_rate', 'punct_entropy',
]

FEATURE_LABELS = {
    'mtld': 'Lexical Diversity (MTLD)',
    'sent_cv': 'Sentence Length CV',
    'self_mention_density': 'Self-Mention Density',
    'opener_ratio': 'Sentence-Opener Connector Ratio',
    'connector_density': 'Connector Density',
    'hedge_density': 'Hedge Density',
    'mean_sent_len': 'Mean Sentence Length',
    'boost_density': 'Booster Density',
    'char_entropy': 'Char N-gram Entropy',
    'rep_rate': 'Word Repetition Rate',
    'punct_entropy': 'Punctuation Entropy',
}

REGISTERS = ['academic', 'news', 'social', 'encyclopedic', 'creative']


def cohens_d(g1, g2):
    n1, n2 = len(g1), len(g2)
    if n1 < 2 or n2 < 2:
        return np.nan
    s_pool = np.sqrt(((n1-1)*np.var(g1, ddof=1) + (n2-1)*np.var(g2, ddof=1)) / (n1+n2-2))
    if s_pool == 0:
        return 0.0
    return (np.mean(g1) - np.mean(g2)) / s_pool


def bootstrap_ci(vals, n_boot=500, ci=0.95):
    boots = [np.mean(np.random.choice(vals, size=len(vals), replace=True)) for _ in range(n_boot)]
    lo = np.percentile(boots, (1-ci)/2*100)
    hi = np.percentile(boots, (1+ci)/2*100)
    return lo, hi


def length_match(df, n_bins=20, seed=42):
    """Match human and AI texts by word-count quantile bins within each register.
    For each bin, sample min(human_count, ai_count) from each group."""
    matched_parts = []
    for reg in REGISTERS:
        sub = df[df['register'] == reg].copy()
        if sub['label'].nunique() < 2:
            continue
        human = sub[sub['label'] == 0]
        ai = sub[sub['label'] == 1]
        if len(human) < 10 or len(ai) < 10:
            continue

        # Create word-count bins on the combined distribution
        combined_words = sub['n_words'].values
        bins = np.quantile(combined_words, np.linspace(0, 1, n_bins+1))
        bins[0] = 0
        bins[-1] = np.inf
        # Make bins unique (in case of ties)
        bins = np.unique(bins)

        for i in range(len(bins)-1):
            lo, hi = bins[i], bins[i+1]
            h_bin = human[(human['n_words'] >= lo) & (human['n_words'] < hi)]
            a_bin = ai[(ai['n_words'] >= lo) & (ai['n_words'] < hi)]
            if len(h_bin) < 5 or len(a_bin) < 5:
                continue
            n = min(len(h_bin), len(a_bin))
            h_sample = h_bin.sample(n, random_state=seed, replace=False)
            a_sample = a_bin.sample(n, random_state=seed, replace=False)
            matched_parts.append(pd.concat([h_sample, a_sample]))

    if matched_parts:
        result = pd.concat(matched_parts).sample(frac=1, random_state=seed)
        return result
    return pd.DataFrame()


def compute_effect_sizes(df):
    rows = []
    for reg in REGISTERS + ['all']:
        sub = df if reg == 'all' else df[df['register'] == reg]
        if len(sub) < 50:
            continue
        human = sub[sub['label'] == 0]
        ai = sub[sub['label'] == 1]
        if len(human) < 10 or len(ai) < 10:
            continue
        for feat in FEATURE_COLS:
            h = human[feat].dropna().values
            a = ai[feat].dropna().values
            if len(h) < 5 or len(a) < 5:
                continue
            d = cohens_d(a, h)
            stat, pval = stats.mannwhitneyu(a, h, alternative='two-sided')
            rows.append({
                'register': reg,
                'feature': feat,
                'feature_label': FEATURE_LABELS[feat],
                'human_mean': np.mean(h),
                'human_sd': np.std(h, ddof=1),
                'ai_mean': np.mean(a),
                'ai_sd': np.std(a, ddof=1),
                'cohens_d': d,
                'mwu_pval': pval,
                'n_human': len(h),
                'n_ai': len(a),
            })
    return pd.DataFrame(rows)


def train_and_eval(X_train, y_train, X_test, y_test):
    clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    y_prob = clf.predict_proba(X_test)[:, 1]
    y_pred = clf.predict(X_test)
    return {
        'auc': roc_auc_score(y_test, y_prob) if len(np.unique(y_test)) > 1 else np.nan,
        'acc': accuracy_score(y_test, y_pred),
        'prec': precision_score(y_test, y_pred, zero_division=0),
        'rec': recall_score(y_test, y_pred, zero_division=0),
        'f1': f1_score(y_test, y_pred, zero_division=0),
    }, clf


def compute_cross_domain_auc(df):
    all_rows = []
    available = [r for r in REGISTERS if df[df['register']==r]['label'].nunique() == 2
                 and len(df[df['register']==r]) >= 100]

    for train_reg in available:
        train_df = df[df['register'] == train_reg].dropna(subset=FEATURE_COLS)
        X_train = train_df[FEATURE_COLS].values
        y_train = train_df['label'].values
        if len(np.unique(y_train)) < 2:
            continue
        for test_reg in available:
            test_df = df[df['register'] == test_reg].dropna(subset=FEATURE_COLS)
            X_test = test_df[FEATURE_COLS].values
            y_test = test_df['label'].values
            if len(np.unique(y_test)) < 2 or len(X_test) < 20:
                continue
            metrics, _ = train_and_eval(X_train, y_train, X_test, y_test)
            all_rows.append({
                'train_register': train_reg,
                'test_register': test_reg,
                'features': 'all_11',
                **metrics,
            })

    for feat in FEATURE_COLS:
        for train_reg in available:
            train_df = df[df['register'] == train_reg].dropna(subset=[feat])
            X_train = train_df[[feat]].values
            y_train = train_df['label'].values
            if len(np.unique(y_train)) < 2:
                continue
            for test_reg in available:
                test_df = df[df['register'] == test_reg].dropna(subset=[feat])
                X_test = test_df[[feat]].values
                y_test = test_df['label'].values
                if len(np.unique(y_test)) < 2 or len(X_test) < 20:
                    continue
                metrics, _ = train_and_eval(X_train, y_train, X_test, y_test)
                all_rows.append({
                    'train_register': train_reg,
                    'test_register': test_reg,
                    'features': feat,
                    **metrics,
                })

    return pd.DataFrame(all_rows)


def compute_classifier_results(df):
    rows = []
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)

    for reg in REGISTERS + ['all']:
        sub = df if reg == 'all' else df[df['register'] == reg]
        sub = sub.dropna(subset=FEATURE_COLS)
        if len(sub) < 100 or sub['label'].nunique() < 2:
            continue
        X = sub[FEATURE_COLS].values
        y = sub['label'].values

        aucs, accs, precs, recs, f1s = [], [], [], [], []
        for train_idx, test_idx in skf.split(X, y):
            X_tr, X_te = X[train_idx], X[test_idx]
            y_tr, y_te = y[train_idx], y[test_idx]
            m, _ = train_and_eval(X_tr, y_tr, X_te, y_te)
            if not np.isnan(m['auc']):
                aucs.append(m['auc'])
            accs.append(m['acc'])
            precs.append(m['prec'])
            recs.append(m['rec'])
            f1s.append(m['f1'])

        dummy = DummyClassifier(strategy='most_frequent')
        base_accs = cross_val_score(dummy, X, y, cv=skf, scoring='accuracy')

        rows.append({
            'register': reg,
            'n_texts': len(sub),
            'n_human': sum(y==0),
            'n_ai': sum(y==1),
            'auc_mean': np.mean(aucs) if aucs else np.nan,
            'auc_sd': np.std(aucs) if aucs else np.nan,
            'acc_mean': np.mean(accs),
            'acc_sd': np.std(accs),
            'prec_mean': np.mean(precs),
            'rec_mean': np.mean(recs),
            'f1_mean': np.mean(f1s),
            'baseline_acc': np.mean(base_accs),
        })

    return pd.DataFrame(rows)


def compute_ablation(df):
    sub = df.dropna(subset=FEATURE_COLS)
    if len(sub) < 100 or sub['label'].nunique() < 2:
        return pd.DataFrame()

    human = sub[sub['label']==0]
    ai = sub[sub['label']==1]
    feat_d = {}
    for feat in FEATURE_COLS:
        h = human[feat].dropna().values
        a = ai[feat].dropna().values
        if len(h) > 5 and len(a) > 5:
            feat_d[feat] = abs(cohens_d(a, h))
    sorted_feats = sorted(feat_d, key=feat_d.get, reverse=True)

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_SEED)
    rows = []
    for k in [1, 2, 4, 6, 11]:
        feats = sorted_feats[:min(k, len(sorted_feats))]
        X = sub[feats].values
        y = sub['label'].values
        aucs = []
        for tr, te in skf.split(X, y):
            clf = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
            clf.fit(X[tr], y[tr])
            prob = clf.predict_proba(X[te])[:,1]
            if len(np.unique(y[te])) > 1:
                aucs.append(roc_auc_score(y[te], prob))
        rows.append({
            'n_features': k,
            'features_used': ', '.join(feats),
            'auc_mean': np.mean(aucs) if aucs else np.nan,
            'auc_sd': np.std(aucs) if aucs else np.nan,
        })
    return pd.DataFrame(rows)


def compute_feature_importance(df):
    sub = df.dropna(subset=FEATURE_COLS)
    if len(sub) < 100 or sub['label'].nunique() < 2:
        return pd.DataFrame()

    X = sub[FEATURE_COLS].values
    y = sub['label'].values
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                               random_state=RANDOM_SEED, stratify=y)
    clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
    clf.fit(X_tr, y_tr)
    result = permutation_importance(clf, X_te, y_te, n_repeats=20,
                                    random_state=RANDOM_SEED, scoring='roc_auc', n_jobs=-1)
    rows = []
    for i, feat in enumerate(FEATURE_COLS):
        rows.append({
            'feature': feat,
            'feature_label': FEATURE_LABELS[feat],
            'importance_mean': result.importances_mean[i],
            'importance_sd': result.importances_std[i],
        })
    return pd.DataFrame(rows).sort_values('importance_mean', ascending=False)


def compute_calibration(df):
    sub = df.dropna(subset=FEATURE_COLS)
    if len(sub) < 100 or sub['label'].nunique() < 2:
        return pd.DataFrame()

    rows = []
    for reg in REGISTERS + ['all']:
        if reg == 'all':
            X = sub[FEATURE_COLS].values
            y = sub['label'].values
            X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2,
                                                       random_state=RANDOM_SEED, stratify=y)
            clf = RandomForestClassifier(n_estimators=200, random_state=RANDOM_SEED, n_jobs=-1)
            clf.fit(X_tr, y_tr)
            prob_sub = clf.predict_proba(X_te)[:,1]
            y_sub = y_te
        else:
            reg_sub = df[df['register']==reg].dropna(subset=FEATURE_COLS)
            if len(reg_sub) < 50 or reg_sub['label'].nunique() < 2:
                continue
            X_r = reg_sub[FEATURE_COLS].values
            y_r = reg_sub['label'].values
            X_tr_r, X_te_r, y_tr_r, y_te_r = train_test_split(
                X_r, y_r, test_size=0.2, random_state=RANDOM_SEED, stratify=y_r)
            clf_r = RandomForestClassifier(n_estimators=100, random_state=RANDOM_SEED, n_jobs=-1)
            clf_r.fit(X_tr_r, y_tr_r)
            prob_sub = clf_r.predict_proba(X_te_r)[:,1]
            y_sub = y_te_r

        try:
            frac_pos, mean_pred = calibration_curve(y_sub, prob_sub, n_bins=10)
            brier = brier_score_loss(y_sub, prob_sub)
            for fp, mp in zip(frac_pos, mean_pred):
                rows.append({'register': reg, 'mean_pred_prob': mp,
                             'frac_positive': fp, 'brier_score': brier})
        except Exception:
            pass

    return pd.DataFrame(rows)


def main():
    in_path = os.path.join(DATA_DIR, 'corpus_features.parquet')
    if not os.path.exists(in_path):
        print(f"ERROR: {in_path} not found. Run 02b_extract_features_fast.py first.")
        return

    df = pd.read_parquet(in_path)
    print(f"Loaded {len(df)} feature rows")

    print("\n=== Length-matching human and AI texts per register ===")
    df_matched = length_match(df, n_bins=20, seed=RANDOM_SEED)
    print(f"Length-matched dataset: {len(df_matched)} texts")
    print(df_matched.groupby(['register', 'label']).size().to_string())
    print(f"\nWord count distribution (matched):")
    print(df_matched.groupby('label')['n_words'].describe()[['mean','50%','min','max']].to_string())

    print("\n1. Computing effect sizes (length-matched)...")
    es_df = compute_effect_sizes(df_matched)
    es_df.to_csv(os.path.join(RESULTS_DIR, 'effect_sizes_matched.csv'), index=False)
    print(f"   Saved effect_sizes_matched.csv ({len(es_df)} rows)")

    print("\n2. Computing cross-domain AUC matrix (length-matched)...")
    cd_df = compute_cross_domain_auc(df_matched)
    cd_df.to_csv(os.path.join(RESULTS_DIR, 'cross_domain_auc_matched.csv'), index=False)
    print(f"   Saved cross_domain_auc_matched.csv ({len(cd_df)} rows)")

    print("\n3. Computing per-register classifier results (length-matched)...")
    clf_df = compute_classifier_results(df_matched)
    clf_df.to_csv(os.path.join(RESULTS_DIR, 'classifier_results_matched.csv'), index=False)
    print(f"   Saved classifier_results_matched.csv")
    print(clf_df[['register','n_texts','auc_mean','auc_sd','acc_mean','baseline_acc']].to_string())

    print("\n4. Computing ablation study (length-matched)...")
    abl_df = compute_ablation(df_matched)
    abl_df.to_csv(os.path.join(RESULTS_DIR, 'ablation_matched.csv'), index=False)
    print(f"   Saved ablation_matched.csv")
    print(abl_df.to_string())

    print("\n5. Computing permutation feature importance (length-matched)...")
    imp_df = compute_feature_importance(df_matched)
    imp_df.to_csv(os.path.join(RESULTS_DIR, 'feature_importance_matched.csv'), index=False)
    print(f"   Saved feature_importance_matched.csv")
    print(imp_df.to_string())

    print("\n6. Computing calibration data (length-matched)...")
    cal_df = compute_calibration(df_matched)
    cal_df.to_csv(os.path.join(RESULTS_DIR, 'calibration_data_matched.csv'), index=False)
    print(f"   Saved calibration_data_matched.csv ({len(cal_df)} rows)")

    print("\nAll length-matched analysis complete.")


if __name__ == '__main__':
    main()
