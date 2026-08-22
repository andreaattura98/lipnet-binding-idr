"""
reference.py — Build the CAID reference for binding region prediction.

The "reference" is the per-residue binary gold standard that CAID uses to evaluate predictors.
Each position receives: '-' (outside IDR), '0' (IDR non-binding), '1' (IDR binding).

Pipeline order:
  1. filter_by_term            → select biologically relevant proteins
  2. build_reference           → build the per-residue binary annotation
  3. filter_by_binding_content → remove degenerate proteins (too binding-rich)
  4. write_fasta               → serialise in the format expected by CAID
  5. plot_stats                → visual diagnostics of reference composition
"""

import numpy as np
import matplotlib.pyplot as plt


# ── 1. Filter by ontology term ────────────────────────────────────────────────

def filter_by_term(prot_dict, selected_terms):
    """
    Return the subset of prot_dict where at least one annotated region belongs
    to selected_terms (set of DisProt/GO/IDPO term_ids).

    Why: the DisProt dataset contains all IDR region types (disorder-to-order,
    display sites, binding, etc.). To build a binding-specific reference
    (e.g. D_O_Binding or D_O_D_D_Binding) only proteins with AT LEAST one
    region of that type are needed. Others are discarded as they do not
    contribute to the chosen prediction task.

    Equivalent to the old make_3rd_dict, renamed for clarity.
    """
    return {
        idx: prot
        for idx, prot in prot_dict.items()
        # any() short-circuits: stops at the first region that satisfies the condition
        if any(r.get('term_id') in selected_terms for r in prot.get('regions', {}).values())
    }


# ── 2. Build binary reference ─────────────────────────────────────────────────

def build_reference(prot_dict, selected_terms):
    """
    Build the per-residue annotation string for each protein.

    Encoding (compatible with CAID format):
      '-'  position outside any IDR region annotated in DisProt
      '0'  position inside an IDR region but not in selected_terms
      '1'  position inside an IDR region and in selected_terms (binding)

    Two-pass strategy — why:
      DisProt may have overlapping annotations on the same residue (e.g. a
      disorder-to-order region overlapping a binding region). The second pass
      ensures that binding always wins over generic IDR: a residue that is
      both '0' and '1' resolves to '1'. This is biologically correct because
      binding is a more specific annotation.

    Positions with '-' are excluded from the confusion matrix in label_confusion():
    CAID evaluates only positions with an explicit annotation (0 or 1).
    """
    reference = {}

    for idx, prot in prot_dict.items():
        seq_len = prot.get('length', 0)
        ref_seq = ['-'] * seq_len   # initialise everything as "outside IDR"

        # Pass 1: mark all IDR positions as non-binding
        # Iterate over ALL regions (not only those in selected_terms) because
        # generic IDR regions (disorder-to-order, display sites, etc.) must also
        # be labelled '0' to be included in the CAID evaluation.
        for region in prot.get('regions', {}).values():
            start, end = region.get('start'), region.get('end')
            for i in range(start - 1, end):   # -1: DisProt uses 1-based indices
                ref_seq[i] = '0'

        # Pass 2: overwrite with '1' only the binding positions
        # Done AFTER pass 1 so that overlaps resolve to '1'.
        for region in prot.get('regions', {}).values():
            if region.get('term_id') in selected_terms:
                start, end = region.get('start'), region.get('end')
                for i in range(start - 1, end):
                    ref_seq[i] = '1'

        reference[idx] = {
            'acc':          prot.get('acc'),          # UniProt accession
            'disprot_id':   prot.get('disprot_id'),   # DisProt ID (e.g. DP00001)
            'sequence':     prot.get('sequence'),     # amino acid sequence
            'ref_sequence': ''.join(ref_seq),         # binary string e.g. "---001110---"
        }

    return reference


# ── 3. Quality filter: removal by binding content ────────────────────────────

def filter_by_binding_content(ref_dict, threshold):
    """
    Remove proteins whose binding content exceeds the given threshold.

    Why this filter exists:
      Some DisProt proteins have almost all IDR residues annotated as binding.
      These are statistically degenerate for a binary prediction task: a naive
      predictor that always outputs '1' would achieve very high accuracy on them.
      Excluding them makes the benchmark more robust and representative.

    How the threshold works (percentage, 0–100):
      The fraction of IDR residues ('0' or '1') that are binding ('1') is computed.
      '-' residues (outside IDR) are ignored: they are not part of the evaluation.

      threshold=80  → remove proteins with >= 80% of IDR positions that are binding
                       (includes 100% since 100 >= 80)
      threshold=100 → remove ONLY proteins where ALL IDR positions are binding
                       (binding_frac >= 1.0, i.e. == 1.0, since 1.0 is the maximum)
      threshold=None → no filter, useful for all-in-all analysis

    Why >= and not == for threshold=100:
      Using >= is correct in both cases. When threshold=100, cutoff=1.0
      and binding_frac >= 1.0 is equivalent to binding_frac == 1.0
      because the fraction cannot exceed 1. No separate logic is needed.

    Prints how many proteins were removed for diagnostics.
    """
    if threshold is None:
        return ref_dict   # no filter requested

    cutoff = threshold / 100.0   # convert percentage to fraction (e.g. 80 → 0.8)
    filtered, removed = {}, []

    for idx, data in ref_dict.items():
        ref_seq = data.get('ref_sequence', '')

        # Consider only positions with an explicit IDR annotation (0 or 1).
        # '-' positions are outside IDR and do not enter the fraction calculation.
        valid = [c for c in ref_seq if c in '01']
        if not valid:
            continue   # protein with no IDR annotations → silently discarded

        binding_frac = valid.count('1') / len(valid)

        if binding_frac >= cutoff:
            removed.append(data['disprot_id'])
        else:
            filtered[idx] = data

    print(f"Removed {len(removed)} proteins with binding% >= {threshold}%")
    return filtered


# ── 4. Write FASTA for CAID ───────────────────────────────────────────────────

def write_fasta(ref_dict, output_path):
    """
    Write a FASTA file in the three-line format expected by CAID:
      > disprot_id
      AMINO_ACID_SEQUENCE
      BINARY_ANNOTATION_STRING

    Why three lines instead of standard FASTA (two):
      CAID requires each protein to have both sequence and annotation in the
      same file so the predictor can be aligned to the reference residue by residue.
      Format: header, sequence, annotation (characters '0'/'1'/'-').

    Raises ValueError if lengths do not match: this is a fatal error indicating
    a misalignment between the DisProt sequence and the region coordinates.
    """
    with open(output_path, 'w') as f:
        for prot in ref_dict.values():
            seq = prot.get('sequence', '')
            ref = prot.get('ref_sequence', '')
            if len(seq) != len(ref):
                # Should never happen if build_reference uses prot['length'] correctly.
                # If it does, there is a bug in the DisProt data (rare but possible).
                raise ValueError(
                    f"Length mismatch for {prot.get('disprot_id')}: "
                    f"sequence={len(seq)}, ref_sequence={len(ref)}"
                )
            f.write(f">{prot['disprot_id']}\n{seq}\n{ref}\n")


# ── 5. Reference statistics and plot ─────────────────────────────────────────

def plot_stats(ref_dict, show_hist=True, show_pie=True):
    """
    Visualise reference composition with two plots:
      - Histogram: distribution of % binding per protein
        (diagnostic: if skewed towards 0% or 100%, the filter did not work well)
      - Pie chart: global '1' vs '0' composition across all IDR residues
        (shows class imbalance, important for interpreting CAID metrics)

    Returns a DataFrame for further analysis:
      disprot_id, n_binding, n_idr, pct_binding, pct_idr
    """
    import pandas as pd

    rows = []
    for data in ref_dict.values():
        seq = data.get('ref_sequence', '')
        n1 = seq.count('1')    # binding residues
        n0 = seq.count('0')    # non-binding IDR residues
        denom = n0 + n1        # total IDR residues (excludes '-')
        pct1 = n1 / denom * 100 if denom > 0 else float('nan')
        rows.append({
            'disprot_id':  data.get('disprot_id'),
            'n_binding':   n1,
            'n_idr':       n0,
            'pct_binding': pct1,
            # pct1 == pct1 is the fastest way to test not isnan without importing math
            'pct_idr':     100 - pct1 if pct1 == pct1 else float('nan'),
        })
    df = pd.DataFrame(rows)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    if show_hist:
        axes[0].hist(df['pct_binding'].dropna(), bins=30, color='steelblue', alpha=0.7)
        axes[0].set_xlabel("% binding / IDR per protein")
        axes[0].set_ylabel("Number of proteins")
        axes[0].set_title("Binding content distribution")

    if show_pie:
        total1 = int(df['n_binding'].sum())
        total0 = int(df['n_idr'].sum())
        axes[1].pie(
            [total1, total0],
            labels=[f"binding ({total1})", f"IDR non-binding ({total0})"],
            colors=['salmon', 'lightgray'],
            autopct='%1.1f%%',
        )
        axes[1].set_title("Global reference composition")

    plt.tight_layout()
    plt.show()
    return df
