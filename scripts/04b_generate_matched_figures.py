#!/usr/bin/env python3
"""
Generate figures for the length-deconfounded (length-matched) analysis.
Figures:
  fig7_effect_heatmap_matched.png    - Cohen's d heatmap (length-matched)
  fig8_cross_domain_auc_matched.png  - 4x4 cross-domain AUC heatmap (length-matched)
  fig9_ablation_matched.png          - Ablation bar chart (length-matched)
  fig10_feature_importance_matched.png - Permutation importance bar chart (length-matched)
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

REGISTERS = ['academic', 'news', 'social', 'creative']
REG_LABELS = {
    'academic': 'Academic',
    'news': 'News',
    'social': 'Social',
    'creative': 'Creative',
}

FEATURE_SHORT = {
    'mtld': 'MTLD',
    'sent_cv': 'Sent. CV',
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

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 10,
    'axes.titlesize': 11,
    'axes.labelsize': 10,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.dpi': 150,
})


def fig7_effect_heatmap_matched(es_df):
    feat_order = ['mtld', 'sent_cv', 'char_entropy', 'rep_rate', 'punct_entropy',
                  'opener_ratio', 'connector_density', 'hedge_density',
                  'self_mention_density', 'mean_sent_len', 'boost_density']
    reg_order = ['all'] + REGISTERS

    pivot = es_df[es_df['register'].isin(reg_order)].pivot_table(
        index='feature', columns='register', values='cohens_d'
    )
    feat_order_avail = [f for f in feat_order if f in pivot.index]
    reg_order_avail = [r for r in reg_order if r in pivot.columns]
    pivot = pivot.reindex(index=feat_order_avail, columns=reg_order_avail)

    feat_labels = [FEATURE_SHORT.get(f, f) for f in pivot.index]
    reg_labels = ['All'] + [REG_LABELS.get(r, r) for r in pivot.columns if r != 'all']

    fig, ax = plt.subplots(figsize=(7, 5.5))
    vmax = max(abs(pivot.values[~np.isnan(pivot.values)].max()),
               abs(pivot.values[~np.isnan(pivot.values)].min())) if pivot.size > 0 else 2.0
    vmax = min(vmax, 2.5)

    sns.heatmap(pivot.values.astype(float),
                annot=True, fmt='.2f', cmap='RdBu_r',
                center=0, vmin=-vmax, vmax=vmax,
                xticklabels=reg_labels, yticklabels=feat_labels,
                linewidths=0.4, linecolor='#eeeeee',
                ax=ax, cbar_kws={'label': "Cohen's d (AI - Human)"})

    ax.set_xlabel('Register', labelpad=8)
    ax.set_ylabel('Feature', labelpad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title('Effect Sizes (Length-Matched)', fontsize=11, pad=10)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig7_effect_heatmap_matched.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig8_cross_domain_auc_matched(cd_df):
    sub = cd_df[cd_df['features'] == 'all_11'].copy()
    if len(sub) == 0:
        print("  Warning: no cross-domain data for all_11, skipping fig8")
        return

    avail_regs = [r for r in REGISTERS if r in sub['train_register'].values]

    pivot = sub.pivot_table(index='train_register', columns='test_register', values='auc')
    pivot = pivot.reindex(index=avail_regs, columns=avail_regs)

    reg_labels = [REG_LABELS.get(r, r) for r in avail_regs]

    fig, ax = plt.subplots(figsize=(6, 5))
    mask = np.isnan(pivot.values.astype(float))

    sns.heatmap(pivot.values.astype(float),
                annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0.5, vmax=1.0,
                xticklabels=reg_labels, yticklabels=reg_labels,
                linewidths=0.4, linecolor='#eeeeee',
                mask=mask,
                ax=ax, cbar_kws={'label': 'AUC-ROC'})

    ax.set_xlabel('Test Register', labelpad=8)
    ax.set_ylabel('Train Register', labelpad=8)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha='right')
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0)
    ax.set_title('Cross-Domain AUC (Length-Matched)', fontsize=11, pad=10)

    for i in range(len(avail_regs)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor='#222222', lw=2.0))

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig8_cross_domain_auc_matched.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig9_ablation_matched(abl_df):
    if len(abl_df) == 0:
        print("  Warning: no ablation data, skipping fig9")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(abl_df))
    bars = ax.bar(x, abl_df['auc_mean'], color='#4393c3', width=0.55,
                  yerr=abl_df['auc_sd'], capsize=4, error_kw={'linewidth': 1.2})

    for bar, val in zip(bars, abl_df['auc_mean']):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{val:.3f}', ha='center', va='bottom', fontsize=8.5)

    ax.set_xticks(x)
    ax.set_xticklabels([f'{int(n)}-feature{"s" if n>1 else ""}' for n in abl_df['n_features']],
                       rotation=15, ha='right')
    ax.set_ylim(0.5, 1.02)
    ax.axhline(0.5, color='#aaaaaa', linestyle='--', linewidth=1, label='Random baseline')
    ax.set_ylabel('AUC-ROC (5-fold CV)')
    ax.set_xlabel('Feature Set Size')
    ax.set_title('Ablation (Length-Matched)', fontsize=11, pad=10)
    ax.legend(loc='lower right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig9_ablation_matched.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig10_feature_importance_matched(imp_df):
    if len(imp_df) == 0:
        print("  Warning: no importance data, skipping fig10")
        return

    imp_df = imp_df.sort_values('importance_mean', ascending=True)
    labels = [FEATURE_SHORT.get(f, f) for f in imp_df['feature']]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#d6604d' if v > 0 else '#aaaaaa' for v in imp_df['importance_mean']]
    ax.barh(labels, imp_df['importance_mean'], xerr=imp_df['importance_sd'],
            color=colors, capsize=3, height=0.65, error_kw={'linewidth': 1.1})

    ax.axvline(0, color='#555555', linewidth=0.8)
    ax.set_xlabel('Mean Decrease in AUC (permutation importance)')
    ax.set_title('Feature Importance (Length-Matched)', fontsize=11, pad=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig10_feature_importance_matched.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def main():
    print("Generating length-matched figures...")

    es_path = os.path.join(RESULTS_DIR, 'effect_sizes_matched.csv')
    cd_path = os.path.join(RESULTS_DIR, 'cross_domain_auc_matched.csv')
    abl_path = os.path.join(RESULTS_DIR, 'ablation_matched.csv')
    imp_path = os.path.join(RESULTS_DIR, 'feature_importance_matched.csv')

    if os.path.exists(es_path):
        es_df = pd.read_csv(es_path)
        print("  Fig 7: Effect size heatmap (matched)")
        fig7_effect_heatmap_matched(es_df)
    else:
        print("  Skipping fig7: effect_sizes_matched.csv not found")

    if os.path.exists(cd_path):
        cd_df = pd.read_csv(cd_path)
        print("  Fig 8: Cross-domain AUC heatmap (matched)")
        fig8_cross_domain_auc_matched(cd_df)
    else:
        print("  Skipping fig8: cross_domain_auc_matched.csv not found")

    if os.path.exists(abl_path):
        abl_df = pd.read_csv(abl_path)
        print("  Fig 9: Ablation chart (matched)")
        fig9_ablation_matched(abl_df)
    else:
        print("  Skipping fig9: ablation_matched.csv not found")

    if os.path.exists(imp_path):
        imp_df = pd.read_csv(imp_path)
        print("  Fig 10: Feature importance (matched)")
        fig10_feature_importance_matched(imp_df)
    else:
        print("  Skipping fig10: feature_importance_matched.csv not found")

    print("Done. Figures saved to:", FIGURES_DIR)


if __name__ == '__main__':
    main()
