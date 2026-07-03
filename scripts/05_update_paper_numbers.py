#!/usr/bin/env python3
"""
Update paper markdown with actual numbers from analysis results.
Reads results CSVs and replaces placeholder tables in the paper.
"""
import os, re
import pandas as pd
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.join(SCRIPT_DIR, '..')
RESULTS_DIR = os.path.join(PROJECT_DIR, 'results')
PAPER_PATH = os.path.join(PROJECT_DIR, 'ai_detection_at_scale.md')

FEATURE_SHORT = {
    'mtld': 'MTLD',
    'sent_cv': 'Sentence CV',
    'self_mention_density': 'Self-Mention',
    'opener_ratio': 'Opener Ratio',
    'connector_density': 'Connector',
    'hedge_density': 'Hedge',
    'mean_sent_len': 'Mean Sent. Len.',
    'boost_density': 'Booster',
    'char_entropy': 'Char Entropy',
    'rep_rate': 'Repetition Rate',
    'punct_entropy': 'Punct. Entropy',
}

REGISTERS = ['academic', 'news', 'social', 'creative']
REG_LABELS = {'academic': 'Academic', 'news': 'News',
               'social': 'Social', 'creative': 'Creative'}


def format_d(d):
    if pd.isna(d):
        return 'n/a'
    return f'{d:.2f}'


def format_auc(a):
    if pd.isna(a):
        return 'n/a'
    return f'{a:.3f}'


def build_effect_table(es_df):
    """Build Table 2: Effect sizes by feature and register."""
    feat_order = ['mtld', 'sent_cv', 'char_entropy', 'punct_entropy',
                  'opener_ratio', 'connector_density', 'rep_rate',
                  'self_mention_density', 'hedge_density', 'mean_sent_len', 'boost_density']

    lines = ['| Feature | Academic | News | Social | Creative | Mean |d| |']
    lines.append('| --- | --- | --- | --- | --- | --- |')

    for feat in feat_order:
        sub = es_df[es_df['feature'] == feat]
        vals = []
        for reg in REGISTERS:
            row = sub[sub['register'] == reg]
            if len(row) > 0:
                vals.append(float(row['cohens_d'].iloc[0]))
            else:
                vals.append(np.nan)
        mean_abs = np.nanmean(np.abs(vals))
        label = FEATURE_SHORT.get(feat, feat)
        cell_vals = [format_d(v) for v in vals]
        lines.append(f'| {label} | {" | ".join(cell_vals)} | {format_d(mean_abs)} |')

    return '\n'.join(lines)


def build_cross_domain_table(cd_df):
    """Build Table 3: Cross-domain AUC matrix."""
    sub = cd_df[cd_df['features'] == 'all_11'].copy()
    if len(sub) == 0:
        return None

    avail = [r for r in REGISTERS if r in sub['train_register'].values]
    reg_labs = [REG_LABELS[r] for r in avail]

    lines = ['| Train / Test | ' + ' | '.join(reg_labs) + ' | Mean off-diag |']
    lines.append('| --- | ' + ' | '.join(['---'] * len(avail)) + ' | --- |')

    for tr in avail:
        row_vals = []
        off_diag = []
        for te in avail:
            cell = sub[(sub['train_register']==tr) & (sub['test_register']==te)]
            auc = float(cell['auc'].iloc[0]) if len(cell)>0 else np.nan
            if tr == te:
                row_vals.append(f'**{format_auc(auc)}**')
            else:
                row_vals.append(format_auc(auc))
                if not np.isnan(auc):
                    off_diag.append(auc)
        mean_od = np.mean(off_diag) if off_diag else np.nan
        lines.append(f'| {REG_LABELS[tr]} | ' + ' | '.join(row_vals) + f' | {format_auc(mean_od)} |')

    # Column means
    col_means = []
    for te in avail:
        vals = []
        for tr in avail:
            if tr != te:
                cell = sub[(sub['train_register']==tr) & (sub['test_register']==te)]
                if len(cell) > 0:
                    vals.append(float(cell['auc'].iloc[0]))
        col_means.append(format_auc(np.mean(vals)) if vals else 'n/a')

    # Overall off-diag mean
    all_off = []
    for tr in avail:
        for te in avail:
            if tr != te:
                cell = sub[(sub['train_register']==tr) & (sub['test_register']==te)]
                if len(cell) > 0:
                    all_off.append(float(cell['auc'].iloc[0]))
    overall = format_auc(np.mean(all_off)) if all_off else 'n/a'

    lines.append(f'| Mean off-diag | ' + ' | '.join(col_means) + f' | **{overall}** |')
    return '\n'.join(lines)


def build_classifier_table(clf_df):
    """Build Table 4: Per-register classifier results."""
    lines = ['| Register | N texts | AUC mean (SD) | Accuracy | F1 | Baseline acc. |']
    lines.append('| --- | --- | --- | --- | --- | --- |')

    reg_order = REGISTERS + ['all']
    for reg in reg_order:
        row = clf_df[clf_df['register'] == reg]
        if len(row) == 0:
            continue
        r = row.iloc[0]
        reg_label = REG_LABELS.get(reg, 'All registers')
        n = int(r['n_texts'])
        auc_str = f"{r['auc_mean']:.3f} ({r['auc_sd']:.3f})"
        acc = f"{r['acc_mean']:.3f}"
        f1 = f"{r['f1_mean']:.3f}"
        base = f"{r['baseline_acc']:.3f}"
        lines.append(f'| {reg_label} | {n:,} | {auc_str} | {acc} | {f1} | {base} |')

    return '\n'.join(lines)


def build_ablation_table(abl_df):
    """Build Table 5: Ablation results."""
    lines = ['| Features used | N features | AUC mean (SD) |']
    lines.append('| --- | --- | --- |')

    desc_map = {
        1: 'MTLD only',
        2: 'MTLD + Sentence CV',
        4: 'Top 4 (+ Char Entropy + Punct Entropy)',
        6: 'Top 6 (+ Opener Ratio + Rep Rate)',
        11: 'All 11',
    }

    for _, r in abl_df.iterrows():
        k = int(r['n_features'])
        desc = desc_map.get(k, f'Top {k}')
        auc_str = f"{r['auc_mean']:.3f} ({r['auc_sd']:.3f})"
        lines.append(f'| {desc} | {k} | {auc_str} |')

    return '\n'.join(lines)


def build_importance_table(imp_df):
    """Build Table 6: Feature importance."""
    lines = ['| Feature | Importance mean | Importance SD |']
    lines.append('| --- | --- | --- |')
    for _, r in imp_df.iterrows():
        label = FEATURE_SHORT.get(r['feature'], r['feature'])
        lines.append(f'| {label} | {r["importance_mean"]:.3f} | {r["importance_sd"]:.3f} |')
    return '\n'.join(lines)


def build_corpus_table(df):
    """Build Table 1: Corpus distribution."""
    lines = ['| Register | Human texts | AI texts | Total | AI models covered |']
    lines.append('| --- | --- | --- | --- | --- |')

    model_coverage = {
        'academic': 'GPT-3.5, GPT-4',
        'news': '11 models',
        'social': '11 models',
        'creative': '11 models',
        'encyclopedic': '(human only)',
    }

    reg_order = ['academic', 'news', 'social', 'encyclopedic', 'creative']
    for reg in reg_order:
        sub = df[df['register'] == reg]
        n_human = int(sum(sub['label'] == 0))
        n_ai = int(sum(sub['label'] == 1))
        total = n_human + n_ai
        models = model_coverage.get(reg, 'multiple')
        label = REG_LABELS.get(reg, reg.capitalize())
        lines.append(f'| {label} | {n_human:,} | {n_ai:,} | {total:,} | {models} |')

    return '\n'.join(lines)


def replace_table_in_paper(paper_text, table_num, new_table):
    """Replace Table N in paper with new_table content."""
    # Pattern: **Table N. ...** followed by table rows
    pattern = rf'(\*\*Table {table_num}\.[^*]*\*\*\n)\n\|[^\n]*\|.*?(?=\n\n|\Z)'
    match = re.search(pattern, paper_text, re.DOTALL)
    if match:
        replacement = match.group(1) + '\n' + new_table
        paper_text = paper_text[:match.start()] + replacement + paper_text[match.end():]
        print(f"  Replaced Table {table_num}")
    else:
        print(f"  Warning: Could not find Table {table_num} pattern in paper")
    return paper_text


def update_corpus_stats(paper_text, feat_df):
    """Update the corpus size mention in abstract and introduction."""
    n_total = len(feat_df)
    n_str = f'{n_total:,}'
    paper_text = re.sub(r'approximately 2\.1 million texts', f'approximately {n_str} texts', paper_text)
    paper_text = re.sub(r'approximately 2\.3 million texts', f'approximately {n_str} texts', paper_text)
    return paper_text


def main():
    print("Loading results...")

    # Load feature data for corpus table
    feat_path = os.path.join(DATA_DIR := os.path.join(PROJECT_DIR, 'data'), 'corpus_features.parquet')
    if os.path.exists(feat_path):
        feat_df = pd.read_parquet(feat_path)
        print(f"  Feature data: {len(feat_df)} rows")
    else:
        feat_df = None
        print("  Warning: corpus_features.parquet not found")

    es_path = os.path.join(RESULTS_DIR, 'effect_sizes.csv')
    cd_path = os.path.join(RESULTS_DIR, 'cross_domain_auc.csv')
    clf_path = os.path.join(RESULTS_DIR, 'classifier_results.csv')
    abl_path = os.path.join(RESULTS_DIR, 'ablation.csv')
    imp_path = os.path.join(RESULTS_DIR, 'feature_importance.csv')

    with open(PAPER_PATH, 'r', encoding='utf-8') as f:
        paper = f.read()

    # Update corpus table
    if feat_df is not None:
        table1 = build_corpus_table(feat_df)
        paper = replace_table_in_paper(paper, 1, table1)
        paper = update_corpus_stats(paper, feat_df)

    # Update effect sizes table
    if os.path.exists(es_path):
        es_df = pd.read_csv(es_path)
        table2 = build_effect_table(es_df)
        paper = replace_table_in_paper(paper, 2, table2)

    # Update cross-domain AUC table
    if os.path.exists(cd_path):
        cd_df = pd.read_csv(cd_path)
        table3 = build_cross_domain_table(cd_df)
        if table3:
            paper = replace_table_in_paper(paper, 3, table3)

    # Update classifier table
    if os.path.exists(clf_path):
        clf_df = pd.read_csv(clf_path)
        table4 = build_classifier_table(clf_df)
        paper = replace_table_in_paper(paper, 4, table4)

    # Update ablation table
    if os.path.exists(abl_path):
        abl_df = pd.read_csv(abl_path)
        table5 = build_ablation_table(abl_df)
        paper = replace_table_in_paper(paper, 5, table5)

    # Update importance table
    if os.path.exists(imp_path):
        imp_df = pd.read_csv(imp_path)
        table6 = build_importance_table(imp_df)
        paper = replace_table_in_paper(paper, 6, table6)

    # Write updated paper
    with open(PAPER_PATH, 'w', encoding='utf-8') as f:
        f.write(paper)
    print(f"\nUpdated {PAPER_PATH}")

    # Also print key numbers for manual verification
    if os.path.exists(clf_path):
        clf_df = pd.read_csv(clf_path)
        all_row = clf_df[clf_df['register'] == 'all']
        if len(all_row) > 0:
            print(f"\nKey numbers:")
            print(f"  All-register AUC: {all_row['auc_mean'].iloc[0]:.3f}")
        print("\nAll registers:")
        print(clf_df[['register','auc_mean','auc_sd','acc_mean','f1_mean']].to_string())

    if os.path.exists(cd_path):
        cd_df = pd.read_csv(cd_path)
        sub = cd_df[cd_df['features'] == 'all_11']
        diag = sub[sub['train_register'] == sub['test_register']]['auc'].mean()
        offdiag = sub[sub['train_register'] != sub['test_register']]['auc'].mean()
        print(f"\n  Cross-domain: diagonal mean AUC = {diag:.3f}, off-diagonal mean = {offdiag:.3f}")
        print(f"  Generalization cost: {diag - offdiag:.3f}")


if __name__ == '__main__':
    main()
