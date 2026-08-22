"""
delong_utils.py — Metriche threshold-free (AUC-ROC + IC DeLong, AUC-PR/APS) e test di DeLong
per il confronto fra due predittori per-residuo sulla stessa reference CAID (0/1/-).

Usato dal notebook Giugno per il task binding-IDR (reference allowed_100).
Tutto in sola lettura sugli input; nessun file viene modificato qui dentro.

Funzioni principali:
  - read_fasta3(path)        : legge la reference CAID a 3 righe -> {disprot_id: (seq, ann)}
  - read_caid(path)          : legge un .caid -> {disprot_id: {pos: score}} (score = colonna 3)
  - align_evaluable(...)     : allinea reference + 2 predittori sui residui con label 0/1
  - delong_test(y, s1, s2)   : AUC, varianza/cov DeLong, IC 95%, ΔAUC, z, p (bilaterale)
  - average_precision(y, s)  : AUC-PR (Average Precision Score), ties gestiti
  - threshold_free_compare() : orchestratore -> dict di risultati + DataFrame
  - format_report(result)    : stringa leggibile (tabella + riga DeLong + provenance)

DeLong: implementazione "fast" di Sun & Xu (2014), basata sui midrank.
"""
import numpy as np
from scipy import stats


# ----------------------------- parsing (read-only) -----------------------------
def read_fasta3(path):
    """Reference CAID a 3 righe: header, sequenza, annotazione 0/1/-."""
    L = [l.rstrip("\n") for l in open(path)]
    d, i = {}, 0
    while i < len(L):
        if L[i].startswith(">"):
            d[L[i][1:].strip()] = (L[i + 1], L[i + 2])
            i += 3
        else:
            i += 1
    return d


def read_caid(path):
    """Predizione .caid: header >ID poi righe 'pos<TAB>aa<TAB>score'. Ritorna {id: {pos: score}}."""
    d, cur = {}, None
    for l in open(path):
        l = l.rstrip("\n")
        if l.startswith(">"):
            cur = l[1:].strip()
            d[cur] = {}
        elif l.strip():
            pr = l.split("\t")
            d[cur][int(pr[0])] = float(pr[2])
    return d


def align_evaluable(ref, pred_a, pred_b):
    """
    Allinea reference + 2 predittori sui soli residui valutabili (label 0/1, '-' esclusi).
    Ritorna (y, sa, sb, coverage) con coverage = dict di diagnostica copertura.
    Solleva ValueError se un residuo valutabile non ha score in uno dei due predittori.
    """
    y, sa, sb = [], [], []
    cov_a = cov_b = both = ev = 0
    prot_a, prot_b = set(), set()
    for dp, (seq, ann) in ref.items():
        for k, ch in enumerate(ann):
            if ch in "01":
                ev += 1
                pos = k + 1
                ha = dp in pred_a and pos in pred_a[dp]
                hb = dp in pred_b and pos in pred_b[dp]
                if ha:
                    cov_a += 1; prot_a.add(dp)
                if hb:
                    cov_b += 1; prot_b.add(dp)
                if ha and hb:
                    both += 1
                    y.append(int(ch)); sa.append(pred_a[dp][pos]); sb.append(pred_b[dp][pos])
    coverage = dict(n_evaluable=ev, cov_a=cov_a, cov_b=cov_b, cov_both=both,
                    prot_ref=len(ref), prot_a=len(prot_a), prot_b=len(prot_b),
                    same_set=(ev == cov_a == cov_b == both))
    return np.array(y), np.array(sa, float), np.array(sb, float), coverage


# ----------------------------- DeLong (Sun & Xu 2014) -----------------------------
def _midrank(x):
    J = np.argsort(x)
    Z = x[J]
    N = len(x)
    T = np.zeros(N, float)
    i = 0
    while i < N:
        j = i
        while j < N and Z[j] == Z[i]:
            j += 1
        T[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    T2 = np.empty(N, float)
    T2[J] = T
    return T2


def _fast_delong(preds_sorted, m):
    """preds_sorted: [k, n] con gli m positivi nelle prime colonne. -> aucs[k], cov[k,k]."""
    n = preds_sorted.shape[1] - m
    pos = preds_sorted[:, :m]
    neg = preds_sorted[:, m:]
    k = preds_sorted.shape[0]
    tx = np.empty([k, m]); ty = np.empty([k, n]); tz = np.empty([k, m + n])
    for r in range(k):
        tx[r] = _midrank(pos[r]); ty[r] = _midrank(neg[r]); tz[r] = _midrank(preds_sorted[r])
    aucs = tz[:, :m].sum(axis=1) / m / n - (m + 1.0) / 2.0 / n
    v01 = (tz[:, :m] - tx) / n
    v10 = 1.0 - (tz[:, m:] - ty) / m
    cov = np.cov(v01) / m + np.cov(v10) / n
    return aucs, np.atleast_2d(cov)


def delong_test(y, s1, s2, alpha=0.05):
    """
    Confronto DeLong fra due AUC su residui appaiati.
    Ritorna dict: auc1, auc2, var1, var2, cov12, ci1, ci2, delta, z, p (bilaterale).
    """
    y = np.asarray(y)
    order = (-y).argsort(kind="mergesort")   # positivi (1) per primi, stabile
    m = int(y.sum())
    preds = np.vstack((s1, s2))[:, order]
    aucs, cov = _fast_delong(preds, m)
    a1, a2 = float(aucs[0]), float(aucs[1])
    v1, v2, c12 = float(cov[0, 0]), float(cov[1, 1]), float(cov[0, 1])
    zc = stats.norm.ppf(1 - alpha / 2)
    ci1 = (a1 - zc * np.sqrt(v1), a1 + zc * np.sqrt(v1))
    ci2 = (a2 - zc * np.sqrt(v2), a2 + zc * np.sqrt(v2))
    se = np.sqrt(v1 + v2 - 2 * c12)
    delta = a1 - a2
    z = delta / se if se > 0 else float("nan")
    p = 2 * stats.norm.sf(abs(z))
    return dict(auc1=a1, auc2=a2, var1=v1, var2=v2, cov12=c12,
                ci1=ci1, ci2=ci2, delta=delta, se=float(se), z=float(z), p=float(p))


def average_precision(y, score):
    """AUC-PR = Average Precision Score (somma dei salti di recall * precision); ties gestiti."""
    y = np.asarray(y)
    order = np.argsort(-score, kind="mergesort")
    yy = y[order]; ss = np.asarray(score)[order]
    tp = np.cumsum(yy)
    fp = np.cumsum(1 - yy)
    distinct = np.where(np.diff(ss))[0]
    idx = np.r_[distinct, len(ss) - 1]            # ultimo indice di ogni gruppo di score uguali
    precision = tp[idx] / (tp[idx] + fp[idx])
    recall = tp[idx] / yy.sum()
    recall_prev = np.r_[0.0, recall[:-1]]
    return float(np.sum((recall - recall_prev) * precision))


# ----------------------------- orchestratore -----------------------------
def threshold_free_compare(ref_path, lip_path, af_path,
                           names=("lipnet", "alphafold_binding")):
    """
    Calcola AUC-ROC (+IC 95% DeLong), AUC-PR (APS) per i due predittori sui residui
    valutabili (0/1) della reference, e il test di DeLong fra le due AUC-ROC.
    Ritorna un dict con tutti i numeri + una tabella pandas (chiave 'df').
    """
    import pandas as pd
    ref = read_fasta3(ref_path)
    pa = read_caid(lip_path)
    pb = read_caid(af_path)
    y, sa, sb, cov = align_evaluable(ref, pa, pb)
    if not cov["same_set"]:
        raise ValueError(f"Copertura NON identica fra i due predittori: {cov}. "
                         "Fermarsi e decidere come gestire (subset comune vs coperture separate).")

    n_pos = int((y == 1).sum()); n_neg = int((y == 0).sum()); n_res = len(y)
    dl = delong_test(y, sa, sb)
    aps_a = average_precision(y, sa)
    aps_b = average_precision(y, sb)

    df = pd.DataFrame([
        dict(predictor=names[0], auc_roc=round(dl["auc1"], 4),
             ci95_low=round(dl["ci1"][0], 4), ci95_high=round(dl["ci1"][1], 4),
             auc_pr_aps=round(aps_a, 4), n_residues=n_res, n_positives=n_pos,
             n_negatives=n_neg, n_proteins=cov["prot_ref"]),
        dict(predictor=names[1], auc_roc=round(dl["auc2"], 4),
             ci95_low=round(dl["ci2"][0], 4), ci95_high=round(dl["ci2"][1], 4),
             auc_pr_aps=round(aps_b, 4), n_residues=n_res, n_positives=n_pos,
             n_negatives=n_neg, n_proteins=cov["prot_ref"]),
    ])
    return dict(df=df, delong=dl, coverage=cov, names=names,
                n_res=n_res, n_pos=n_pos, n_neg=n_neg, n_prot=cov["prot_ref"],
                aps={names[0]: aps_a, names[1]: aps_b})


def format_report(result, ref_path, lip_path, af_path):
    """Stringa leggibile: tabella, riga DeLong, provenance, invarianti."""
    n = result["names"]; dl = result["delong"]; cov = result["coverage"]
    L = []
    L.append("Threshold-free metrics — DisProt binding-IDR (allowed_100, 322 proteine)")
    L.append("=" * 78)
    L.append(f"{'predictor':<20}{'AUC-ROC':>9}  {'95% CI (DeLong)':<20}{'AUC-PR(APS)':>12}{'n_res':>8}{'n_prot':>8}")
    for _, r in result["df"].iterrows():
        ci = f"[{r['ci95_low']:.4f}, {r['ci95_high']:.4f}]"
        L.append(f"{r['predictor']:<20}{r['auc_roc']:>9.4f}  {ci:<20}{r['auc_pr_aps']:>12.4f}"
                 f"{r['n_residues']:>8}{r['n_proteins']:>8}")
    L.append("")
    L.append(f"DeLong (AUC-ROC {n[0]} - {n[1]}, residui appaiati): "
             f"ΔAUC={dl['delta']:+.4f}  z={dl['z']:.4f}  p={dl['p']:.4g} (bilaterale)")
    L.append("")
    L.append("Provenance:")
    L.append(f"  reference (label 0/1/-): {ref_path}")
    L.append(f"  score {n[0]:<18}: {lip_path} (colonna 3)")
    L.append(f"  score {n[1]:<18}: {af_path} (colonna 3)")
    L.append(f"  AUC-ROC, IC 95%, ΔAUC/z/p : fast DeLong (Sun & Xu 2014) su {result['n_res']} residui appaiati")
    L.append(f"  AUC-PR                    : Average Precision Score")
    L.append("Coverage / invarianti:")
    L.append(f"  proteine reference={cov['prot_ref']}  coperte da {n[0]}={cov['prot_a']}  da {n[1]}={cov['prot_b']}")
    L.append(f"  residui valutabili={result['n_res']}  (con score {n[0]}={cov['cov_a']}, {n[1]}={cov['cov_b']}, entrambi={cov['cov_both']})")
    L.append(f"  stesso set residui per i due predittori: {cov['same_set']}")
    L.append(f"  n_positives={result['n_pos']} (atteso 23061: {'OK' if result['n_pos']==23061 else 'MISMATCH'})  "
             f"n_negatives={result['n_neg']}  totale={result['n_res']} (atteso 61496: {'OK' if result['n_res']==61496 else 'MISMATCH'})")
    return "\n".join(L)
