#!/usr/bin/env python3
"""Generate figures for must-do analyses."""
import os, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_DIR = os.path.join(SCRIPT_DIR, '..', 'results')
FIGURES_DIR = os.path.join(RESULTS_DIR, 'figures')
os.makedirs(FIGURES_DIR, exist_ok=True)

plt.rcParams.update({
    'font.family': 'DejaVu Sans', 'font.size': 10,
    'axes.titlesize': 11, 'axes.labelsize': 10,
    'xtick.labelsize': 9, 'ytick.labelsize': 9,
    'legend.fontsize': 9, 'figure.dpi': 150,
})

REG_LABELS = {'academic': 'Academic', 'news': 'News', 'social': 'Social', 'creative': 'Creative'}

MODEL_LABELS = {
    'gpt4': 'GPT-4', 'gpt3': 'GPT-3', 'gpt-3.5-turbo': 'GPT-3.5',
    'chatgpt': 'ChatGPT', 'gpt2': 'GPT-2',
    'mistral': 'Mistral', 'mistral-chat': 'Mistral-Chat',
    'llama-chat': 'Llama-Chat', 'mpt': 'MPT', 'mpt-chat': 'MPT-Chat',
    'cohere': 'Cohere', 'cohere-chat': 'Cohere-Chat',
}


def fig_per_model():
    """Per-model AUC heatmap."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'per_model_auc.csv'))
    pivot = df.pivot_table(index='model', columns='register', values='auc')

    # Sort models by mean AUC
    pivot = pivot.reindex(index=pivot.mean(axis=1).sort_values(ascending=True).index)

    labels = [MODEL_LABELS.get(m, m) for m in pivot.index]
    reg_labels = [REG_LABELS.get(r, r) for r in pivot.columns]

    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(pivot.values.astype(float),
                annot=True, fmt='.3f', cmap='YlOrRd',
                vmin=0.5, vmax=1.0,
                xticklabels=reg_labels, yticklabels=labels,
                linewidths=0.4, linecolor='#eeeeee',
                ax=ax, cbar_kws={'label': 'AUC-ROC'})
    ax.set_xlabel('Register', labelpad=8)
    ax.set_ylabel('Model', labelpad=8)
    ax.set_title('Per-Model Detection AUC', fontsize=11, pad=10)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig11_per_model_auc.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig_adversarial():
    """Adversarial robustness bar chart."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'adversarial_results.csv'))
    df = df.sort_values('auc', ascending=True)

    fig, ax = plt.subplots(figsize=(7, 4))
    colors = ['#d6604d' if auc < 0.5 else '#4393c3' for auc in df['auc']]
    bars = ax.barh(df['attack_type'], df['auc'], color=colors, height=0.6)

    for bar, val in zip(bars, df['auc']):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height()/2,
                f'{val:.3f}', va='center', fontsize=9)

    ax.axvline(0.5, color='#aaaaaa', linestyle='--', linewidth=1, label='Random baseline')
    ax.set_xlim(0, 1.1)
    ax.set_xlabel('AUC-ROC')
    ax.set_title('Adversarial Robustness', fontsize=11, pad=10)
    ax.legend(loc='lower right', fontsize=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig12_adversarial.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig_deployment_pr():
    """Deployment precision at different ratios."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'deployment_pr.csv'))

    fig, ax = plt.subplots(figsize=(7, 4))
    ratios = ['1:1', '1:10', '1:100', '1:1000']
    regs = ['academic', 'news', 'social', 'creative', 'all']
    colors = plt.cm.tab10(np.linspace(0, 0.7, len(regs)))

    x = np.arange(len(ratios))
    width = 0.15

    for i, reg in enumerate(regs):
        sub = df[df['register'] == reg]
        if len(sub) == 0:
            continue
        vals = [sub[sub['deployment_ratio'] == r]['deployment_precision'].values[0]
                if len(sub[sub['deployment_ratio'] == r]) > 0 else 0
                for r in ratios]
        label = 'All registers' if reg == 'all' else REG_LABELS.get(reg, reg)
        ax.bar(x + i * width, vals, width, label=label, color=colors[i])

    ax.set_xticks(x + width * 2)
    ax.set_xticklabels(ratios)
    ax.set_ylabel('Deployment Precision (at 90% recall)')
    ax.set_xlabel('Human:AI Ratio')
    ax.set_title('Deployment Precision at Realistic Class Ratios', fontsize=11, pad=10)
    ax.legend(fontsize=8, loc='upper right')
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig13_deployment_pr.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def fig_lr_vs_rf():
    """LR vs RF comparison."""
    df = pd.read_csv(os.path.join(RESULTS_DIR, 'lr_vs_rf.csv'))

    fig, ax = plt.subplots(figsize=(6, 4))
    regs = df['register'].unique()
    x = np.arange(len(regs))
    width = 0.3

    rf_vals = df[df['classifier'] == 'RF']['auc_mean'].values
    lr_vals = df[df['classifier'] == 'LR']['auc_mean'].values

    bars1 = ax.bar(x - width/2, rf_vals, width, label='Random Forest', color='#4393c3')
    bars2 = ax.bar(x + width/2, lr_vals, width, label='Logistic Regression', color='#f4a582')

    for bars, vals in [(bars1, rf_vals), (bars2, lr_vals)]:
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                    f'{val:.3f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels([REG_LABELS.get(r, 'All') if r != 'all' else 'All' for r in regs])
    ax.set_ylabel('AUC-ROC (5-fold CV)')
    ax.set_ylim(0.5, 1.05)
    ax.set_title('Random Forest vs Logistic Regression', fontsize=11, pad=10)
    ax.legend(fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()
    out = os.path.join(FIGURES_DIR, 'fig14_lr_vs_rf.png')
    plt.savefig(out, bbox_inches='tight', dpi=150)
    plt.close()
    print(f"  Saved {out}")


def main():
    print("Generating must-do figures...")
    fig_per_model()
    fig_adversarial()
    fig_deployment_pr()
    fig_lr_vs_rf()
    print("Done.")


if __name__ == '__main__':
    main()
