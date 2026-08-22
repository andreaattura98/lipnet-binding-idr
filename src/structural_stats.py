"""
structural_stats.py — Analisi strutturale (pLDDT/disorder, RSA, struttura secondaria)
per i gruppi della valutazione DisProt binding-IDR (allowed_100).

Read-only sugli input. Funzioni:
  - load_structural(...) : costruisce la tabella per-residuo allineando reference (0/1/-),
                           score LIPNet e AlphaFold-binding (.caid) e le quantita' strutturali
                           dalla TSV AlphaFold-disorder (lddt/rsa/ss). Verifica gli invarianti.
  - define_groups(df)    : maschere LIPNet_pred (lipnet>=0.612), REF_pos (ref==1), AF_pred (binding>=0.744).
  - continuous_tests(...): mediane per gruppo + Mann-Whitney U (two-sided), rank-biserial, direzione.
  - ss_tests(...)        : % helix/sheet/coil per gruppo + chi-quadro 3 classi + Cramer's V.
  - benjamini_hochberg(p): BH-FDR.
  - figure helper        : violini pLDDT/RSA e grouped bar SS.

Quantita' strutturali (TSV alphafold_disorder_pred_caid_data.tsv):
  pLDDT  = colonna 'lddt' (0-1)         disorder = 1 - lddt
  RSA    = colonna 'rsa'                SS (8-state DSSP) = colonna 'ss'
SS collapse: helix={H,G,I}, sheet={E,B}, coil={-,T,S}.
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, chi2_contingency

GROUP_COLORS = {"REF_pos": "#4C72B0", "LIPNet_pred": "#55A868", "AF_pred": "#DD8452"}
SS3_MAP = {**{k: "helix" for k in "HGI"}, **{k: "sheet" for k in "EB"},
           **{k: "coil" for k in ("-", "T", "S")}}
LIP_THR, AF_THR = 0.612, 0.744   # equal-count, convenzione >=


def benjamini_hochberg(pvals):
    p = np.asarray(pvals, float); n = len(p)
    order = np.argsort(p); ranked = p[order]
    adj = ranked * n / np.arange(1, n + 1)
    adj = np.minimum.accumulate(adj[::-1])[::-1]
    out = np.empty(n); out[order] = np.clip(adj, 0, 1)
    return out


def _read_caid_df(path, score_name):
    rows, cur = [], None
    for l in open(path):
        l = l.rstrip("\n")
        if l.startswith(">"):
            cur = l[1:].strip()
        elif l.strip():
            pr = l.split("\t")
            rows.append((cur, int(pr[0]), float(pr[2])))
    return pd.DataFrame(rows, columns=["disprot_id", "pos", score_name])


def _read_fasta3_long(path):
    L = [l.rstrip("\n") for l in open(path)]
    rows, i = [], 0
    while i < len(L):
        if L[i].startswith(">"):
            dp = L[i][1:].strip(); ann = L[i + 2]
            for k, ch in enumerate(ann):
                if ch in "01":
                    rows.append((dp, k + 1, int(ch)))
            i += 3
        else:
            i += 1
    return pd.DataFrame(rows, columns=["disprot_id", "pos", "ref"])


def load_structural(ref_path, lip_path, af_path, tsv_path, disprot_json):
    """Costruisce la tabella per-residuo e verifica gli invarianti. Ritorna (df, info)."""
    import ijson
    ref = _read_fasta3_long(ref_path)                         # disprot_id,pos,ref(0/1)
    lip = _read_caid_df(lip_path, "lipnet")
    af = _read_caid_df(af_path, "binding")

    dp2acc = {p["disprot_id"]: p["acc"]
              for p in ijson.items(open(disprot_json, encoding="utf-8"), "data.item")}
    ref["acc"] = ref["disprot_id"].map(dp2acc)

    tsv = (pd.read_csv(tsv_path, sep="\t")
           .rename(columns={"name": "acc", "pos": "pos"})[["acc", "pos", "lddt", "rsa", "ss"]])

    df = (ref.merge(lip, on=["disprot_id", "pos"], how="left")
             .merge(af, on=["disprot_id", "pos"], how="left")
             .merge(tsv, on=["acc", "pos"], how="left"))

    df["disorder"] = 1.0 - df["lddt"]
    df["ss3"] = df["ss"].map(SS3_MAP)

    info = dict(
        n_res=len(df), n_pos=int((df["ref"] == 1).sum()), n_neg=int((df["ref"] == 0).sum()),
        n_prot=df["disprot_id"].nunique(),
        nan_lddt=int(df["lddt"].isna().sum()), nan_rsa=int(df["rsa"].isna().sum()),
        nan_ss=int(df["ss"].isna().sum()),
        nan_lip=int(df["lipnet"].isna().sum()), nan_af=int(df["binding"].isna().sum()),
        ss3_unmapped=int(df["ss3"].isna().sum()),
    )
    return df, info


def define_groups(df):
    """Maschere dei 3 gruppi (convenzione >=, applicata inline sullo score grezzo)."""
    return {
        "LIPNet_pred": df["lipnet"] >= LIP_THR,
        "REF_pos":     df["ref"] == 1,
        "AF_pred":     df["binding"] >= AF_THR,
    }


PAIRS = [("LIPNet_pred", "REF_pos"), ("AF_pred", "REF_pos"), ("LIPNet_pred", "AF_pred")]


def _rank_biserial(x, y):
    """MWU two-sided + rank-biserial r = 2*P(X>Y)-1. Ritorna (p, r, PS, n1, n2)."""
    x = np.asarray(x, float); x = x[~np.isnan(x)]
    y = np.asarray(y, float); y = y[~np.isnan(y)]
    U, p = mannwhitneyu(x, y, alternative="two-sided")
    PS = U / (len(x) * len(y))
    return p, 2 * PS - 1, PS, len(x), len(y)


def continuous_tests(df, groups, variables=(("pLDDT", "lddt"), ("RSA", "rsa"))):
    """
    Ritorna (group_summary, pairwise). pLDDT/disorder = stesso test (disorder = 1-pLDDT).
    pairwise: una riga per (variable, pair); 'interpretation' segna le coppie con AF_pred
    come 'by-construction' (AF-binding e' definito da RSA+pLDDT -> circolare).
    """
    gsum = []
    for g, m in groups.items():
        sub = df[m]
        gsum.append(dict(group=g, n=int(m.sum()),
                         median_pLDDT=round(sub["lddt"].median(), 4),
                         median_disorder=round(sub["disorder"].median(), 4),
                         median_RSA=round(sub["rsa"].median(), 4)))
    group_summary = pd.DataFrame(gsum)

    rows = []
    for vname, col in variables:
        for g1, g2 in PAIRS:
            p, r, PS, n1, n2 = _rank_biserial(df[groups[g1]][col], df[groups[g2]][col])
            direction = f"{g1}>{g2}" if PS > 0.5 else (f"{g2}>{g1}" if PS < 0.5 else "tie")
            interp = "by-construction" if ("AF_pred" in (g1, g2)) else "informative"
            rows.append(dict(variable=vname, pair=f"{g1} vs {g2}", group1=g1, group2=g2,
                             n_group1=n1, n_group2=n2,
                             median_group1=round(float(df[groups[g1]][col].median()), 4),
                             median_group2=round(float(df[groups[g2]][col].median()), 4),
                             p_raw=p, rank_biserial=round(r, 4), direction=direction,
                             interpretation=interp))
    return group_summary, pd.DataFrame(rows)


def ss_tests(df, groups):
    """Ritorna (group_pct, pairwise). chi-quadro 3 classi (helix/sheet/coil) + Cramer's V."""
    classes = ["helix", "sheet", "coil"]
    gp = []
    counts = {}
    for g, m in groups.items():
        c = df[m]["ss3"].value_counts()
        n = int(m.sum())
        counts[g] = np.array([int(c.get(k, 0)) for k in classes])
        gp.append(dict(group=g, n=n,
                       pct_helix=round(100 * counts[g][0] / n, 2),
                       pct_sheet=round(100 * counts[g][1] / n, 2),
                       pct_coil=round(100 * counts[g][2] / n, 2)))
    group_pct = pd.DataFrame(gp)

    rows = []
    for g1, g2 in PAIRS:
        table = np.vstack([counts[g1], counts[g2]])
        chi2, p, dof, _ = chi2_contingency(table)
        N = table.sum()
        cramers_v = float(np.sqrt(chi2 / (N * (min(table.shape) - 1))))
        rows.append(dict(pair=f"{g1} vs {g2}", group1=g1, group2=g2,
                         n_group1=int(counts[g1].sum()), n_group2=int(counts[g2].sum()),
                         chi2=round(float(chi2), 3), dof=int(dof), p_raw=p,
                         cramers_v=round(cramers_v, 4)))
    return group_pct, pd.DataFrame(rows)


def apply_global_fdr(cont_pairwise, ss_pairwise):
    """BH-FDR su TUTTI i test di (A) continui + (B) SS insieme. Aggiunge p_adj e sig."""
    cont = cont_pairwise.copy(); ss = ss_pairwise.copy()
    cont["block"] = "continuous"; ss["block"] = "ss"
    allp = np.concatenate([cont["p_raw"].values, ss["p_raw"].values])
    padj = benjamini_hochberg(allp)
    nC = len(cont)
    cont["p_adj"] = padj[:nC]; ss["p_adj"] = padj[nC:]
    cont["sig_global"] = cont["p_adj"] < 0.05
    ss["sig_global"] = ss["p_adj"] < 0.05
    return cont, ss


# ----------------------------- figure -----------------------------
def violin_figure(df, groups, col, title, ylabel, outpath):
    import matplotlib.pyplot as plt
    order = ["LIPNet_pred", "REF_pos", "AF_pred"]
    data = [np.asarray(df[groups[g]][col].dropna(), float) for g in order]
    fig, ax = plt.subplots(figsize=(7, 5))
    vp = ax.violinplot(data, showmedians=True, showextrema=False)
    for body, g in zip(vp["bodies"], order):
        body.set_facecolor(GROUP_COLORS[g]); body.set_edgecolor("black"); body.set_alpha(0.85)
    if "cmedians" in vp:
        vp["cmedians"].set_color("black"); vp["cmedians"].set_linewidth(1.5)
    ax.set_xticks([1, 2, 3])
    ax.set_xticklabels([f"{g}\nn={len(d)}" for g, d in zip(order, data)], fontsize=10)
    ax.set_ylabel(ylabel, fontsize=11); ax.set_title(title, fontsize=11)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight")
    return fig


def ss_bar_figure(group_pct, outpath):
    import matplotlib.pyplot as plt
    order = ["LIPNet_pred", "REF_pos", "AF_pred"]
    gp = group_pct.set_index("group").loc[order]
    classes = ["pct_helix", "pct_sheet", "pct_coil"]; labels = ["helix", "sheet", "coil"]
    x = np.arange(len(classes)); w = 0.25
    fig, ax = plt.subplots(figsize=(8, 5))
    for i, g in enumerate(order):
        ax.bar(x + (i - 1) * w, gp.loc[g, classes].values, w,
               label=f"{g} (n={int(gp.loc[g,'n'])})", color=GROUP_COLORS[g], edgecolor="black")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=11)
    ax.set_ylabel("% residui", fontsize=11)
    ax.set_title("Struttura secondaria (3 classi) per gruppo", fontsize=11)
    ax.legend(fontsize=9)
    fig.tight_layout(); fig.savefig(outpath, dpi=150, bbox_inches="tight")
    return fig
