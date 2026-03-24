#!/usr/bin/env python
"""
Plot 1: Overall Performance Heatmap
4 panels (one per pert_type), rows=noise, cols=grn_type x grn_size
Metric: mean Pearson Delta Corr across perturbations per file
"""

import os
import re
import numpy as np
import pandas as pd
import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from pathlib import Path

EVAL_DIR = Path('/mnt/polished-lake/home/nkondapaneni/state_runs/sergio_mini_incl_2000steps/eval')
OUT_DIR = Path('/mnt/polished-lake/home/nkondapaneni/state/simulate/sergio/results')
OUT_DIR.mkdir(exist_ok=True)

SCORES_CSV = OUT_DIR / 'scores_per_file.csv'


def parse_filename(fname):
    name = fname.replace('_simulated.h5ad', '')
    m = re.match(r'^(BA-VM|BA|ER)_size(\d+)_noise([a-z]+)_(.+)$', name)
    if not m:
        raise ValueError(f"Cannot parse: {fname}")
    grn_type, grn_size, noise_label, pert_type = m.groups()
    return grn_type, int(grn_size), noise_label, pert_type


def to_dense(x):
    if hasattr(x, 'toarray'):
        return x.toarray()
    return np.asarray(x)


def score_file(path):
    adata = ad.read_h5ad(path)
    ctrl_mask = adata.obs['gene'] == 'non-targeting'
    ctrl_true = to_dense(adata[ctrl_mask].X).mean(0)
    ctrl_pred = adata[ctrl_mask].obsm['X_hvg'].mean(0)

    pearson_rs = []
    for pert in adata.obs['gene'].unique():
        if pert == 'non-targeting':
            continue
        p_mask = adata.obs['gene'] == pert
        pert_true = to_dense(adata[p_mask].X).mean(0)
        pert_pred = adata[p_mask].obsm['X_hvg'].mean(0)
        true_delta = pert_true - ctrl_true
        pred_delta = pert_pred - ctrl_pred
        if true_delta.std() < 1e-8 or pred_delta.std() < 1e-8:
            continue
        r, _ = pearsonr(true_delta, pred_delta)
        pearson_rs.append(r)

    return np.mean(pearson_rs) if pearson_rs else np.nan


# --- Score all files (or load cached) ---
if SCORES_CSV.exists():
    print(f"Loading cached scores from {SCORES_CSV}")
    df = pd.read_csv(SCORES_CSV)
else:
    records = []
    files = sorted(f for f in os.listdir(EVAL_DIR) if f.endswith('_simulated.h5ad'))
    print(f"Scoring {len(files)} files...")
    for i, fname in enumerate(files):
        grn_type, grn_size, noise_label, pert_type = parse_filename(fname)
        score = score_file(EVAL_DIR / fname)
        records.append({
            'grn_type': grn_type,
            'grn_size': grn_size,
            'noise_label': noise_label,
            'pert_type': pert_type,
            'pearson_delta': score,
        })
        if (i + 1) % 20 == 0:
            print(f"  {i+1}/{len(files)} done")

    df = pd.DataFrame(records)
    df.to_csv(SCORES_CSV, index=False)
    print(f"Saved scores to {SCORES_CSV}")

print("\nScore summary:")
print(df.groupby(['grn_type', 'noise_label', 'pert_type'])['pearson_delta'].mean().unstack())

# --- Plot ---
PERT_TYPES = ['KO', 'KD_010', 'KD_050', 'KD_080']
NOISE_ORDER = ['none', 'low', 'high']
GRN_COLS = [
    ('ER', 10), ('ER', 50), ('ER', 100),
    ('BA', 10), ('BA', 50), ('BA', 100),
    ('BA-VM', 10), ('BA-VM', 50), ('BA-VM', 100),
]
COL_LABELS = [f'{g}\n{s}' for g, s in GRN_COLS]

# Determine color range from data
vmin = max(-0.1, df['pearson_delta'].min() - 0.05)
vmax = min(1.0, df['pearson_delta'].max() + 0.05)

fig, axes = plt.subplots(1, 4, figsize=(22, 3.5), sharey=True)
fig.suptitle('Mean Pearson Δ Correlation (predicted vs true perturbation effect)\n'
             'Trained: sergio_mini_incl_2000steps', fontsize=12, y=1.02)

for ax, pert_type in zip(axes, PERT_TYPES):
    sub = df[df['pert_type'] == pert_type]

    mat = np.full((len(NOISE_ORDER), len(GRN_COLS)), np.nan)
    for j, (gt, gs) in enumerate(GRN_COLS):
        for i, noise in enumerate(NOISE_ORDER):
            row = sub[(sub['grn_type'] == gt) & (sub['grn_size'] == gs) & (sub['noise_label'] == noise)]
            if len(row):
                mat[i, j] = row['pearson_delta'].iloc[0]

    im = ax.imshow(mat, vmin=vmin, vmax=vmax, cmap='RdYlGn', aspect='auto')
    ax.set_title(pert_type, fontsize=11, fontweight='bold')
    ax.set_xticks(range(len(GRN_COLS)))
    ax.set_xticklabels(COL_LABELS, fontsize=8)
    ax.set_yticks(range(len(NOISE_ORDER)))
    ax.set_yticklabels([f'noise\n{n}' for n in NOISE_ORDER], fontsize=9)

    # Vertical separators between grn_types
    for x in [2.5, 5.5]:
        ax.axvline(x, color='white', linewidth=2)

    for i in range(len(NOISE_ORDER)):
        for j in range(len(GRN_COLS)):
            v = mat[i, j]
            if not np.isnan(v):
                text_color = 'black' if 0.25 < v < 0.75 else 'white'
                ax.text(j, i, f'{v:.2f}', ha='center', va='center',
                        fontsize=8, color=text_color, fontweight='bold')

plt.colorbar(im, ax=axes[-1], label='Pearson Δ Corr', shrink=0.85, pad=0.02)
plt.tight_layout()

out_pdf = OUT_DIR / 'fig1_heatmap.pdf'
out_png = OUT_DIR / 'fig1_heatmap.png'
fig.savefig(out_pdf, bbox_inches='tight')
fig.savefig(out_png, dpi=150, bbox_inches='tight')
print(f"\nSaved: {out_png}")
print(f"Saved: {out_pdf}")
