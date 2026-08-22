"""
Definitive localCIDER analysis at MIN_LENGTH=5 for the thesis.

Reproduces the notebook's all-pairs Mann-Whitney + GLOBAL Benjamini-Hochberg cell,
computing everything from the persisted per-region descriptor CSVs (which are already
filter_short(min_length=5) + equal-count thresholds: LIPNet 0.612, AF 0.744).

Read-only on cider.py and source data. Writes only new figures/CSVs.
Figure annotations and the printed table come from THE SAME computed values.
"""
import os
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[1]          # repo root (scripts/ -> ..)
DATA = str(REPO / "data" / "processed")

# 8 descriptors used in the thesis (frac_bulky excluded, as in the notebook FEATURES list)
FEATURES = ['FCR', 'NCPR', 'kappa', 'Mean_hydropathy', 'PPII_propensity',
            'seq_complexity_shannon', 'frac_order_promoting', 'frac_aromatic']

FEAT_NAME = {
    'FCR': 'FCR', 'NCPR': 'NCPR', 'kappa': 'kappa',
    'Mean_hydropathy': 'mean hydropathy', 'PPII_propensity': 'PPII propensity',
    'seq_complexity_shannon': 'sequence complexity',
    'frac_order_promoting': 'order-promoting fraction',
    'frac_aromatic': 'aromatic fraction',
}

# colours as in the notebook violin cells
C_REF, C_TP, C_FP = '#4C72B0', '#55A868', '#C44E52'
COLOR = {'REF': C_REF, 'TP': C_TP, 'FP': C_FP}

# ---- load the persisted per-region descriptor tables (min_length=5) ----
ref   = pd.read_csv(os.path.join(DATA, "df_cider_ref1.csv"))
groupsets = {
    'LIPNet':    {'REF': ref,
                  'TP':  pd.read_csv(os.path.join(DATA, "df_cider_tp.csv")),
                  'FP':  pd.read_csv(os.path.join(DATA, "df_cider_fp.csv"))},
    'AlphaFold': {'REF': ref,
                  'TP':  pd.read_csv(os.path.join(DATA, "df_cider_af_tp.csv")),
                  'FP':  pd.read_csv(os.path.join(DATA, "df_cider_af_fp.csv"))},
}

print("=== group sizes (min_length=5, equal-count thr LIPNet 0.612 / AF 0.744) ===")
for method, g in groupsets.items():
    for grp in ['REF', 'TP', 'FP']:
        df = g[grp]
        assert df['length'].min() >= 5, f"{method} {grp} has length<5!"
    print(f"  {method:10s}  REF n={len(g['REF'])}  TP n={len(g['TP'])}  FP n={len(g['FP'])}")

# ---- 48 tests: 2 methods x 3 comparisons x 8 descriptors ----
# comparison (A, B): PS = P(A > B) = U_A / (nA*nB)
comparisons = [('TP', 'FP'), ('TP', 'REF'), ('FP', 'REF')]
rows = []
for method, g in groupsets.items():
    for A, B in comparisons:
        for f in FEATURES:
            a = g[A][f].dropna().values
            b = g[B][f].dropna().values
            U, p = mannwhitneyu(a, b, alternative='two-sided')
            PS = U / (len(a) * len(b))      # probability of superiority P(A>B)
            rows.append(dict(method=method, comparison=f"{A}-vs-{B}", descriptor=f,
                             A=A, B=B, nA=len(a), nB=len(b), p_raw=p, PS=PS))
res = pd.DataFrame(rows)
assert len(res) == 48, f"expected 48 tests, got {len(res)}"

# ---- GLOBAL Benjamini-Hochberg across all 48 ----
res = res.sort_values('p_raw').reset_index(drop=True)
m = len(res)
ranks = np.arange(1, m + 1)
p_adj = res['p_raw'].values * m / ranks
p_adj = np.minimum.accumulate(p_adj[::-1])[::-1]   # step-up monotonicity
res['p_adj'] = np.clip(p_adj, 0, 1.0)
res['sig'] = res['p_adj'] < 0.05
res['direction'] = np.where(res['PS'] > 0.5, f"{'A'}>{'B'}", f"{'B'}>{'A'}")
res['direction'] = [f"{r.A}>{r.B}" if r.PS > 0.5 else f"{r.B}>{r.A}" for r in res.itertuples()]

res.to_csv(os.path.join(DATA, "localcider_48tests_min5.csv"), index=False)

surv = res[res['sig']].sort_values(['method', 'comparison', 'descriptor']).reset_index(drop=True)
surv.to_csv(os.path.join(DATA, "localcider_significant_min5.csv"), index=False)

print("\n=== SIGNIFICANT effects under GLOBAL BH (FDR<0.05) — definitive min_length=5 ===")
hdr = f"{'method':10s} {'comparison':10s} {'descriptor':22s} {'dir':9s} {'PS':>6s} {'p_adj':>10s}  {'nA':>5s} {'nB':>5s}"
print(hdr); print('-' * len(hdr))
for r in surv.itertuples():
    print(f"{r.method:10s} {r.comparison:10s} {r.descriptor:22s} {r.direction:9s} "
          f"{r.PS:6.3f} {r.p_adj:10.2e}  {r.nA:5d} {r.nB:5d}")

# ---- figure helper ----
def panel(ax, method, A, B, feature, ps, padj, direction):
    dA = groupsets[method][A][feature].dropna().values
    dB = groupsets[method][B][feature].dropna().values
    arrays = [dA, dB]
    labels = [f"{('REF' if A=='REF' else method+' '+A)}\nn={len(dA)}",
              f"{('REF' if B=='REF' else method+' '+B)}\nn={len(dB)}"]
    cols = [COLOR[A], COLOR[B]]
    vp = ax.violinplot(arrays, showmedians=True, showextrema=False)
    for body, c in zip(vp['bodies'], cols):
        body.set_facecolor(c); body.set_edgecolor('black'); body.set_alpha(0.85)
    if 'cmedians' in vp:
        vp['cmedians'].set_color('black'); vp['cmedians'].set_linewidth(1.4)
    ax.set_xticks([1, 2]); ax.set_xticklabels(labels)
    ax.set_ylabel(feature)
    ax.set_title(f"{method} — {FEAT_NAME[feature]}\n{A} vs {B}: "
                 f"p_adj(global)={padj:.1e}, PS={ps:.2f} ({direction})", fontsize=10)

# ---- (a) all significant effects (grid layout) ----
all_eff = list(surv.itertuples())
ncol = 4
nrow = int(np.ceil(len(all_eff) / ncol))
fig, axes = plt.subplots(nrow, ncol, figsize=(5.2 * ncol, 4.6 * nrow))
axes = np.atleast_1d(axes).flatten()
for ax, r in zip(axes, all_eff):
    panel(ax, r.method, r.A, r.B, r.descriptor, r.PS, r.p_adj, r.direction)
for ax in axes[len(all_eff):]:
    ax.axis('off')
fig.tight_layout()
out_all = os.path.join(DATA, "cider_violins_significant_all.png")
fig.savefig(out_all, dpi=150, bbox_inches='tight'); plt.close(fig)

# ---- (b) LIPNet-only: order-promoting (TP vs REF) + aromatic (TP vs FP) ----
lip_eff = surv[(surv['method'] == 'LIPNet')].copy()
# enforce the two expected panels in a fixed order
order = [('LIPNet', 'frac_order_promoting', 'TP-vs-REF'),
         ('LIPNet', 'frac_aromatic', 'TP-vs-FP')]
lip_rows = []
for meth, feat, comp in order:
    sel = lip_eff[(lip_eff['descriptor'] == feat) & (lip_eff['comparison'] == comp)]
    if len(sel):
        lip_rows.append(sel.iloc[0])
fig, axes = plt.subplots(1, len(lip_rows), figsize=(5.2 * len(lip_rows), 4.6))
if len(lip_rows) == 1:
    axes = [axes]
for ax, r in zip(axes, lip_rows):
    panel(ax, r['method'], r['A'], r['B'], r['descriptor'], r['PS'], r['p_adj'], r['direction'])
fig.tight_layout()
out_lip = os.path.join(DATA, "cider_violins_LIPNet_only.png")
fig.savefig(out_lip, dpi=150, bbox_inches='tight'); plt.close(fig)

# ---- (c) AlphaFold-only: the two strongest false-positive depletions ----
af_order = [('AlphaFold', 'frac_aromatic', 'FP-vs-REF'),
            ('AlphaFold', 'frac_order_promoting', 'FP-vs-REF')]
af_rows = []
for meth, feat, comp in af_order:
    sel = surv[(surv['method'] == meth) & (surv['descriptor'] == feat) & (surv['comparison'] == comp)]
    if len(sel):
        af_rows.append(sel.iloc[0])
fig, axes = plt.subplots(1, len(af_rows), figsize=(5.2 * len(af_rows), 4.6))
axes = np.atleast_1d(axes).flatten()
for ax, r in zip(axes, af_rows):
    panel(ax, r['method'], r['A'], r['B'], r['descriptor'], r['PS'], r['p_adj'], r['direction'])
fig.tight_layout()
out_af = os.path.join(DATA, "cider_violins_AlphaFold_only.png")
fig.savefig(out_af, dpi=150, bbox_inches='tight'); plt.close(fig)

print("\n=== saved ===")
print(" full 48-test table :", os.path.join(DATA, "localcider_48tests_min5.csv"))
print(" survivors table    :", os.path.join(DATA, "localcider_significant_min5.csv"))
print(" figure (all)       :", out_all)
print(" figure (LIPNet)    :", out_lip)
print("\nFigure annotations and printed numbers come from the SAME computed values (one run).")
