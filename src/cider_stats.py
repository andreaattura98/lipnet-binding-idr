"""
cider_stats.py — Confronti CIDER a coppie (tutte le coppie) con FDR globale e per-famiglia.

Per ogni metodo (LIPNet, AlphaFold), ogni confronto a coppie (TP vs FP, TP vs REF, FP vs REF)
e ogni feature: Mann-Whitney U (two-sided), probabilita' di superiorita' (effect size, 0.5 = nessun
effetto) con direzione esplicita, n per gruppo (conteggio regioni).

FDR Benjamini-Hochberg applicata:
  - globalmente su tutti i test (un'unica correzione)
  - per-famiglia (method, comparison): 8 test ciascuna.

Read-only: non scrive nulla; il notebook salva il CSV.
"""
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu

COMPARISONS = [("TP", "FP"), ("TP", "REF"), ("FP", "REF")]
ALPHA = 0.05


def benjamini_hochberg(pvals):
    """BH-FDR. Ritorna array di p aggiustati (stesso ordine dell'input)."""
    p = np.asarray(pvals, float)
    n = len(p)
    order = np.argsort(p)
    ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]   # monotonia
    out = np.empty(n)
    out[order] = np.clip(adj, 0, 1)
    return out


def prob_superiority(x, y):
    """
    Probabilita' di superiorita' (common-language effect size) di x rispetto a y:
    P(X>Y) + 0.5*P(X=Y) = U_x / (n_x * n_y). 0.5 = nessun effetto.
    Ritorna (PS, U, p_two_sided).
    """
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    U, p = mannwhitneyu(x, y, alternative="two-sided")
    PS = U / (len(x) * len(y))
    return PS, U, p, len(x), len(y)


def compute_allpairs(groups, features):
    """
    groups: {method: {'REF':df, 'TP':df, 'FP':df}}.
    Ritorna un DataFrame tidy con le colonne richieste, FDR globale + per-famiglia.
    """
    rows = []
    for method, gd in groups.items():
        for g1, g2 in COMPARISONS:
            for feat in features:
                PS, U, p, n1, n2 = prob_superiority(gd[g1][feat], gd[g2][feat])
                if PS > 0.5:
                    direction = f"{g1}>{g2}"
                elif PS < 0.5:
                    direction = f"{g2}>{g1}"
                else:
                    direction = "tie"
                rows.append(dict(
                    method=method, comparison=f"{g1} vs {g2}", feature=feat,
                    n_group1=n1, n_group2=n2, p_raw=p,
                    prob_superiority=round(PS, 4), direction=direction,
                    _fam=f"{method}|{g1} vs {g2}"))
    df = pd.DataFrame(rows)

    # FDR globale (tutti i test)
    df["p_adj_global"] = benjamini_hochberg(df["p_raw"].values)
    # FDR per-famiglia (method, comparison)
    df["p_adj_family"] = np.nan
    for fam, idx in df.groupby("_fam").groups.items():
        df.loc[idx, "p_adj_family"] = benjamini_hochberg(df.loc[idx, "p_raw"].values)

    df["sig_global"] = df["p_adj_global"] < ALPHA
    df["sig_family"] = df["p_adj_family"] < ALPHA

    cols = ["method", "comparison", "feature", "n_group1", "n_group2",
            "p_raw", "p_adj_global", "p_adj_family", "sig_global", "sig_family",
            "prob_superiority", "direction"]
    return df[cols]


def summary_text(df, small_n=20):
    """Riassunto: quanti significativi (global / family) e quali; avvisi gruppi piccoli."""
    L = []
    ng = int(df["sig_global"].sum()); nf = int(df["sig_family"].sum())
    L.append(f"Test totali: {len(df)}  |  significativi FDR-globale: {ng}  |  significativi FDR-famiglia: {nf}  (alpha={ALPHA})")
    L.append("")
    L.append(f"Significativi sotto FDR GLOBALE ({ng}):")
    sg = df[df["sig_global"]].sort_values("p_adj_global")
    for _, r in sg.iterrows():
        L.append(f"  {r['method']:9s} {r['comparison']:10s} {r['feature']:24s} "
                 f"p_adj_global={r['p_adj_global']:.3g}  PS={r['prob_superiority']:.3f}  {r['direction']}")
    L.append("")
    L.append(f"Significativi sotto FDR per-FAMIGLIA ma NON sotto globale:")
    extra = df[(df["sig_family"]) & (~df["sig_global"])].sort_values("p_adj_family")
    if len(extra) == 0:
        L.append("  (nessuno)")
    for _, r in extra.iterrows():
        L.append(f"  {r['method']:9s} {r['comparison']:10s} {r['feature']:24s} "
                 f"p_adj_family={r['p_adj_family']:.3g}  PS={r['prob_superiority']:.3f}  {r['direction']}")
    L.append("")
    # avvisi gruppi piccoli
    small = df[(df["n_group1"] < small_n) | (df["n_group2"] < small_n)]
    L.append(f"Avviso gruppi piccoli (< {small_n} regioni) -> p-value inaffidabili:")
    if len(small) == 0:
        sizes = (df.groupby(["method", "comparison"])[["n_group1", "n_group2"]].first())
        mn = int(min(df["n_group1"].min(), df["n_group2"].min()))
        L.append(f"  Nessun gruppo < {small_n}. Gruppo piu' piccolo in tabella: n={mn} regioni.")
    else:
        for _, r in small[["method", "comparison", "n_group1", "n_group2"]].drop_duplicates().iterrows():
            L.append(f"  {r['method']:9s} {r['comparison']:10s} n_group1={r['n_group1']} n_group2={r['n_group2']}")
    return "\n".join(L)
