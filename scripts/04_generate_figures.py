#!/usr/bin/env python3
"""
Generate all 6 paper figures from analysis results.
Figures:
  fig1_pipeline.png          - Data pipeline diagram
  fig2_effect_heatmap.png    - Cohen's d heatmap (feature x register)
  fig3_cross_domain_auc.png  - 5x5 cross-domain AUC heatmap
  fig4_ablation.png          - Ablation bar chart
  fig5_calibration.png       - Calibration curves per register
  fig6_feature_importance.png - Permutation importance bar chart
"""
import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

REGISTERS = ['academic', 'news', 'social', 'encyclopedic', 'creative']
REG_LABELS = {
    'academic': 'Academic',
    'news': 'News',
    'social': 'Social',
    'encyclopedic': 'Encyclopedic',
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

PALETTE = {'human': '#2166ac', 'ai': '#d6604d'}

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


def fig1_pipeline():
    """Pipeline diagram using matplotlib patches."""
    fig, ax = plt.subplots(figsize=(10, 3.5))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 4)
    ax.axis('off')

    boxes = [
        (0.3, 1.5, 1.6, 1.0, 'Data Sources\n(RAID, PubMed,\nCC-News, Reddit,\nWikipedia)', '#d1e5f0'),
        (2.3, 1.5, 1.6, 1.0, 'Text\nFiltering\n(register +\nlength)', '#fddbc7'),
        (4.3, 1.5, 1.6, 1.0, 'Feature\nExtraction\n(11 features)', '#e0f3db'),
        (6.3, 1.5, 1.6, 1.0, 'Statistical\nAnalysis\n(Cohen\'s d,\ncross-domain)', '#fee8c8'),
        (8.3, 1.5, 1.6, 1.0, 'Classifier\n+\nAblation', '#f1b6da'),
    ]

    for x, y, w, h, label, color in boxes:
        rect = mpatches.FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.1",
                                       facecolor=color, edgecolor='#555555', linewidth=1.2)
        ax.add_patch(rect)
        ax.text(x + w/2, y + h/2, label, ha='center', va='center',
                fontsize=8.5, wrap=True, multialignment='center')

    # Arrows
    for i in range(len(boxes)-1):
        x1 = boxes[i][0] + boxes[i][2]
        x2 = boxes[i+1][0]
        y_mid = boxes[i][1] + boxes[i][3]/2
        ax.annotate('', xy=(x2, y_mid), xytext=(x1, y_mid),
                    arrowprops=dict(arrowstyle='->', color='#333333', lw=1.5))

    # Labels below boxes
    sub_labels = [
        '~10M texts\n5 registers\n11+ models',
        'Min length\nper register\nDeduplicate',
        'MTLD, CV,\nEntropy\netc.',
        'Effect sizes\nGeneralization\nprofiles',
        'Per-register\nAUC, F1\nCalibration',
    ]
    for i, (x, y, w, h, _, _) in enumerate(boxes):
        ax.text(x + w/2, y - 0.25, sub_labels[i], ha='center', va='top',
                fontsize=7.5, color='#444444', multialignment='center')

    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig1_pipeline.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig2_effect_heatmap(es_df):
    """Cohen's d heatmap: features x registers."""
    feat_order = ['mtld', 'sent_cv', 'char_entropy', 'rep_rate', 'punct_entropy',
                  'opener_ratio', 'connector_density', 'hedge_density',
                  'self_mention_density', 'mean_sent_len', 'boost_density']
    reg_order = ['all'] + REGISTERS

    pivot = es_df[es_df['register'].isin(reg_order)].pivot_table(
        index='feature', columns='register', values='cohens_d'
    )
    # Reorder
    feat_order_avail = [f for f in feat_order if f in pivot.index]
    reg_order_avail = [r for r in reg_order if r in pivot.columns]
    pivot = pivot.reindex(index=feat_order_avail, columns=reg_order_avail)

    feat_labels = [FEATURE_SHORT.get(f, f) for f in pivot.index]
    reg_labels = ['All' if r == 'all' else REG_LABELS.get(r, r) for r in pivot.columns]

    fig, ax = plt.subplots(figsize=(8, 5.5))
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
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig2_effect_heatmap.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig3_cross_domain_auc(cd_df):
    """5x5 cross-domain AUC heatmap for the full 11-feature model."""
    sub = cd_df[cd_df['features'] == 'all_11'].copy()
    if len(sub) == 0:
        print("  Warning: no cross-domain data for all_11, skipping fig3")
        return

    avail_regs = [r for r in REGISTERS if r in sub['train_register'].values
                  or r in sub['test_register'].values]

    pivot = sub.pivot_table(index='train_register', columns='test_register', values='auc')
    pivot = pivot.reindex(index=avail_regs, columns=avail_regs)

    reg_labels = [REG_LABELS.get(r, r) for r in avail_regs]

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
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

    # Diagonal annotation
    for i in range(len(avail_regs)):
        ax.add_patch(plt.Rectangle((i, i), 1, 1, fill=False,
                                   edgecolor='#222222', lw=2.0))

    ax.text(0.5, 1.04, 'Diagonal = within-register, Off-diagonal = cross-domain',
            transform=ax.transAxes, ha='center', fontsize=8, color='#555555')
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig3_cross_domain_auc.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig4_ablation(abl_df):
    """Ablation bar chart."""
    if len(abl_df) == 0:
        print("  Warning: no ablation data, skipping fig4")
        return

    fig, ax = plt.subplots(figsize=(6.5, 3.8))
    x = np.arange(len(abl_df))
    bars = ax.bar(x, abl_df['auc_mean'], color='#4393c3', width=0.55,
                  yerr=abl_df['auc_sd'], capsize=4, error_kw={'linewidth': 1.2})

    # Label each bar
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
    ax.legend(loc='lower right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig4_ablation.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig5_calibration(cal_df):
    """Calibration curves per register."""
    if len(cal_df) == 0:
        print("  Warning: no calibration data, skipping fig5")
        return

    avail_regs = ['all'] + [r for r in REGISTERS if r in cal_df['register'].values]
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(avail_regs)))

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([0, 1], [0, 1], 'k--', linewidth=1, label='Perfectly calibrated', alpha=0.6)

    for reg, color in zip(avail_regs, colors):
        sub = cal_df[cal_df['register'] == reg]
        if len(sub) == 0:
            continue
        label = 'All registers' if reg == 'all' else REG_LABELS.get(reg, reg)
        brier = sub['brier_score'].iloc[0]
        ax.plot(sub['mean_pred_prob'], sub['frac_positive'],
                'o-', color=color, label=f'{label} (Brier={brier:.3f})',
                markersize=5, linewidth=1.5)

    ax.set_xlabel('Mean Predicted Probability')
    ax.set_ylabel('Fraction of Positives (AI texts)')
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(loc='upper left', fontsize=8, framealpha=0.85)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig5_calibration.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig6_feature_importance(imp_df):
    """Permutation importance horizontal bar chart."""
    if len(imp_df) == 0:
        print("  Warning: no importance data, skipping fig6")
        return

    imp_df = imp_df.sort_values('importance_mean', ascending=True)
    labels = [FEATURE_SHORT.get(f, f) for f in imp_df['feature']]

    fig, ax = plt.subplots(figsize=(7, 5))
    colors = ['#d6604d' if v > 0 else '#aaaaaa' for v in imp_df['importance_mean']]
    ax.barh(labels, imp_df['importance_mean'], xerr=imp_df['importance_sd'],
            color=colors, capsize=3, height=0.65, error_kw={'linewidth': 1.1})

    ax.axvline(0, color='#555555', linewidth=0.8)
    ax.set_xlabel('Mean Decrease in AUC (permutation importance)')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig6_feature_importance.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def main():
    print("Generating figures...")

    print("  Fig 1: Pipeline diagram")
    fig1_pipeline()

    es_path = os.path.join(RESULTS_DIR, 'effect_sizes.csv')
    cd_path = os.path.join(RESULTS_DIR, 'cross_domain_auc.csv')
    abl_path = os.path.join(RESULTS_DIR, 'ablation.csv')
    cal_path = os.path.join(RESULTS_DIR, 'calibration_data.csv')
    imp_path = os.path.join(RESULTS_DIR, 'feature_importance.csv')

    if os.path.exists(es_path):
        es_df = pd.read_csv(es_path)
        print("  Fig 2: Effect size heatmap")
        fig2_effect_heatmap(es_df)
    else:
        print("  Skipping fig2: effect_sizes.csv not found")

    if os.path.exists(cd_path):
        cd_df = pd.read_csv(cd_path)
        print("  Fig 3: Cross-domain AUC heatmap")
        fig3_cross_domain_auc(cd_df)
    else:
        print("  Skipping fig3: cross_domain_auc.csv not found")

    if os.path.exists(abl_path):
        abl_df = pd.read_csv(abl_path)
        print("  Fig 4: Ablation chart")
        fig4_ablation(abl_df)
    else:
        print("  Skipping fig4: ablation.csv not found")

    if os.path.exists(cal_path):
        cal_df = pd.read_csv(cal_path)
        print("  Fig 5: Calibration curves")
        fig5_calibration(cal_df)
    else:
        print("  Skipping fig5: calibration_data.csv not found")

    if os.path.exists(imp_path):
        imp_df = pd.read_csv(imp_path)
        print("  Fig 6: Feature importance")
        fig6_feature_importance(imp_df)
    else:
        print("  Skipping fig6: feature_importance.csv not found")

    print("Done. Figures saved to:", FIGURES_DIR)


if __name__ == '__main__':
    main()
