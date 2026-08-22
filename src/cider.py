"""
cider.py — Physicochemical analysis of IDR regions using LocalCIDER.

LocalCIDER (Das & Pappu, 2013) computes sequence properties relevant for IDRs:
charge, hydropathy, propensity to form specific structures, etc.
This module prepares the regions to analyse (from the reference or from LipNet predictions),
filters them, and computes the features. Also includes secondary structure calculation
from AlphaFold data (DSSP-based), used as a structural proxy.

External dependency: localcider (pip install localcider).
"""

import numpy as np
import pandas as pd
from localcider.sequenceParameters import SequenceParameters
from localcider.backend.sequence import SequenceException


# ── Build region dictionary from reference ────────────────────────────────────

def make_region_dict(prot_dict, allowed_terms):
    """
    Build a protein dictionary keeping only regions whose term_id is in allowed_terms.
    Used to prepare input for extract_regions() and compute_features() when analysing
    reference regions (e.g. D_O_Binding).

    Why this is separate from filter_by_term in reference.py:
      filter_by_term operates on the raw DisProt dictionary (with all regions) and
      filters PROTEINS. make_region_dict operates on the same dictionary but filters
      REGIONS within each protein, keeping only those of interest for LocalCIDER
      computation. The two operations serve different purposes.

    Equivalent to the old make_dic_for_df_localCIDER.
    """
    result = {}
    for idx, prot in prot_dict.items():
        regions = prot.get('regions', {})
        # Keep only regions with term_id in allowed_terms
        allowed = {k: v for k, v in regions.items() if v.get('term_id') in allowed_terms}
        if not allowed:
            continue   # protein has no regions of the target type → skip
        result[idx] = {
            'acc':             prot.get('acc'),
            'disprot_id':      prot.get('disprot_id'),
            'length':          prot.get('length'),
            'prot_sequence':   prot.get('sequence'),
            'Allowed_regions': allowed,   # only regions with the target term_id
        }
    return result


# ── Merge overlapping intervals (internal helper) ─────────────────────────────

def _merge_intervals(intervals):
    """
    Merge a list of (start, end) tuples into non-overlapping intervals.

    Why needed: DisProt may have two adjacent or partially overlapping regions of
    the same type (e.g. two consecutive D_O_Binding regions). Passing them separately
    to LocalCIDER would compute features on sub-sequences that are too short, losing
    the context of distributed charges. Merging resolves this.

    Algorithm: sort by start, then iterate and extend the last interval when the
    new one overlaps (start <= previous_end). O(n log n) due to sorting.
    """
    if not intervals:
        return []
    intervals = sorted(intervals, key=lambda x: x[0])
    merged = [list(intervals[0])]
    for start, end in intervals[1:]:
        if start <= merged[-1][1]:
            # Overlap or adjacency: extend current interval
            merged[-1][1] = max(merged[-1][1], end)
        else:
            merged.append([start, end])
    return [tuple(m) for m in merged]


# ── Extract region sequences from reference ───────────────────────────────────

def extract_regions(prot_dict):
    """
    For each protein in prot_dict (output of make_region_dict), merge overlapping
    allowed regions and return the corresponding sequences.

    Returns: {idx: [{acc, disprot_id, start, end, region_sequence}, ...]}.

    Why merge before extracting: see _merge_intervals.
    Slicing seq[s-1:e] because start/end are 1-based (DisProt) and Python is 0-based.

    Equivalent to the old get_region_sequence.
    """
    result = {}
    for idx, prot in prot_dict.items():
        seq       = prot['prot_sequence']
        intervals = [(r['start'], r['end']) for r in prot['Allowed_regions'].values()]
        merged    = _merge_intervals(intervals)
        result[idx] = [
            {
                'acc':             prot['acc'],
                'disprot_id':      prot['disprot_id'],
                'start':           s,
                'end':             e,
                'region_sequence': seq[s - 1:e],   # s-1: convert 1-based to 0-based
            }
            for s, e in merged
        ]
    return result


# ── Extract regions from LipNet predictions ───────────────────────────────────

def extract_predicted_regions(prot_dict):
    """
    Identify contiguous blocks of '1' in the lipnet_pred string of each protein
    and convert them into regions with start/end/sequence. Used to compute CIDER
    features on regions predicted as binding by LipNet.

    Why analyse regions instead of individual residues:
      LocalCIDER computes sequence properties that depend on local context
      (e.g. kappa, charge distribution). Analysing contiguous regions is
      biologically meaningful: a single IDR binding region has coherent
      physicochemical properties.

    Expected fields for each entry in prot_dict:
      acc, disprot_id, prot_sequence (str), lipnet_pred (binary string e.g. "0011110...")

    Edge handling: a region open at the end of the sequence (last character '1')
    is closed without waiting for a '0'.

    Equivalent to the old extract_regions_from_ones.
    """
    result = {}
    for idx, data in prot_dict.items():
        seq     = data['prot_sequence']
        pred    = data['lipnet_pred']
        regions = []
        in_region, start = False, None

        for i, char in enumerate(pred):
            if char == '1' and not in_region:
                # Start of a new positive block
                start, in_region = i, True
            elif char == '0' and in_region:
                # End of block: record the region
                regions.append({
                    'acc':             data['acc'],
                    'disprot_id':      data['disprot_id'],
                    'start':           start + 1,   # convert to 1-based for DisProt consistency
                    'end':             i,            # i is 0-based exclusive = 1-based end
                    'length':          i - start,
                    'region_sequence': seq[start:i],
                })
                in_region = False

        # Edge case: sequence ends with an open block
        if in_region:
            end = len(pred)
            regions.append({
                'acc':             data['acc'],
                'disprot_id':      data['disprot_id'],
                'start':           start + 1,
                'end':             end,
                'length':          end - start,
                'region_sequence': seq[start:end],
            })

        result[idx] = regions
    return result


# ── Filter: invalid sequences ─────────────────────────────────────────────────

def filter_invalid(region_dict):
    """
    Remove proteins that contain at least one region with non-standard amino acids
    that LocalCIDER cannot process (e.g. 'X', 'U', 'B').

    Why discard the entire protein and not just the problematic region:
      If a protein has a region with 'X', the overall sequence quality is likely low
      (e.g. incomplete sequence in the database). Keeping other regions from the same
      protein would introduce bias into the analysis.

    Returns (valid_dict, skipped_ids) to allow diagnostics on what was excluded.

    Equivalent to the old filter_invalid_proteins.
    """
    valid, skipped = {}, []
    for idx, regions in region_dict.items():
        skip = False
        for region in regions:
            try:
                SequenceParameters(region['region_sequence'])
            except SequenceException:
                skip = True
                break   # one invalid region is enough to discard the whole protein
        if skip:
            skipped.append(idx)
        else:
            valid[idx] = regions
    return valid, skipped


# ── Filter: short regions ─────────────────────────────────────────────────────

def filter_short(region_dict, min_length=3):
    """
    Remove regions shorter than min_length residues.
    Proteins left with no regions are dropped.

    Why 3 as default:
      LocalCIDER requires at least 2 residues to compute kappa (charge distribution).
      Below 3 residues, statistical features are not biologically meaningful
      (e.g. FCR on a single residue is 0 or 1, not a sequence property).

    Equivalent to the old filter_short_regions.
    """
    result = {}
    for idx, regions in region_dict.items():
        kept = [r for r in regions if len(r.get('region_sequence', '')) >= min_length]
        if kept:
            result[idx] = kept
    return result


# ── Normalised Shannon entropy (global compositional feature) ─────────────────

def _normalized_shannon_entropy(seq):
    """
    Normalised Shannon entropy over the full sequence (no sliding window).

    H = -sum(p_i * log2(p_i))  over all observed amino acids.
    H_max = log2(min(20, n))   — theoretical maximum for an alphabet of min(20,n) symbols.
    Returns H / H_max in [0, 1]: 0 = homopolymeric sequence, 1 = uniform composition.

    Why not WF windowed complexity: 25% of LIPNet regions have 5-9 aa and the WF
    measure with an adaptive blobLen produces entropies that are not comparable across
    different lengths. Global entropy is length-independent and consistent across the
    full dataset.
    """
    from collections import Counter
    seq = seq.upper()
    n = len(seq)
    if n <= 1:
        return 0.0
    counts = Counter(seq)
    probs = np.array([c / n for c in counts.values()])
    H = -np.sum(probs * np.log2(probs))
    H_max = np.log2(min(20, n))
    return float(H / H_max) if H_max > 0 else 0.0


# ── Compute LocalCIDER features ───────────────────────────────────────────────

def compute_features(region_dict):
    """
    Compute LocalCIDER physicochemical features for each region and return a
    DataFrame with one row per region.

    Features computed and their biological meaning:
      FCR                    — Fraction of Charged Residues: total charge density
                               (K, R, D, E). IDR binding regions tend to have high FCR.
      NCPR                   — Net Charge Per Residue: net charge (+/-). Positive =
                               K/R prevalence, negative = D/E prevalence.
      kappa                  — Spatial distribution of charges along the sequence.
                               kappa=0: well-mixed charges, kappa=1: separated +/- blocks.
                               Important for coacervate-forming propensity.
      Mean_hydropathy        — Mean hydropathy (normalised Kyte-Doolittle scale).
                               IDRs tend to have low hydropathy (polar sequences).
      PPII_propensity        — PPII helix propensity from LocalCIDER, Hilser scale
                               (Elam et al. 2013, Protein Sci.). f_PPII_chain: per-residue
                               weighted mean on the Hilser scale.
      seq_complexity_shannon — Normalised Shannon entropy in [0,1]. H/H_max with
                               H_max=log2(min(20,n)). Length-independent; comparable
                               across the full dataset.
      frac_aromatic          — Fraction of F, Y, W: true aromatic residues (pi ring).
                               Involved in pi-pi and cation-pi interactions in IDR binding.
      frac_bulky             — Fraction of F, I, L, V: bulky/hydrophobic residues,
                               proxy for the aliphatic-hydrophobic character of the region.
      frac_order_promoting   — Fraction of W, F, Y, I, L, V, C, N: order-promoting
                               residues per Uversky's classification (Uversky 2002).
                               High values indicate compositional tendency toward
                               folded/ordered conformations; low values are typical
                               of IDRs.

    Why skip the entire protein if one region is invalid (instead of only that region):
      Consistency with filter_invalid(). If a protein has already passed filter_invalid
      it should not have invalid regions. The check here is a safeguard.

    Equivalent to the old compute_lc_features_dataframe.
    """
    rows = []
    for idx, regions in region_dict.items():

        # Safeguard: skip protein if any region contains non-standard amino acids
        skip = False
        for region in regions:
            try:
                SequenceParameters(region['region_sequence'])
            except SequenceException:
                skip = True
                break
        if skip:
            continue

        for region in regions:
            seq = region['region_sequence']
            sp  = SequenceParameters(seq)   # LocalCIDER object for this sequence
            rows.append({
                'protein_id':      idx,
                'disprot_id':      region.get('disprot_id'),
                'start':           region['start'],
                'end':             region['end'],
                'length':          len(seq),
                'region_sequence': seq,
                # LocalCIDER native features
                'FCR':             sp.get_FCR(),
                'NCPR':            sp.get_NCPR(),
                'kappa':           sp.get_kappa(),
                'Mean_hydropathy': sp.get_mean_hydropathy(),
                # Additional LocalCIDER feature + manual compositional features
                'PPII_propensity':         sp.get_PPII_propensity(mode='hilser'),
                'seq_complexity_shannon':  _normalized_shannon_entropy(seq),
                'frac_aromatic':           sum(1 for aa in seq.upper() if aa in 'FYW') / len(seq),
                'frac_bulky':              sum(1 for aa in seq.upper() if aa in 'FILV') / len(seq),
                'frac_order_promoting':    sum(1 for aa in seq.upper() if aa in 'WFYILVCN') / len(seq),
            })

    return pd.DataFrame(rows)


# ── Secondary structure from AlphaFold (DSSP) ────────────────────────────────

def compute_ss_percentages(af_df, regions_dict, name_col='acc', pos_col='pos', ss_col='ss'):
    """
    Compute secondary structure composition for each region using DSSP annotations
    derived from AlphaFold PDB files.

    DSSP classification used (standard 8-state reduced to 3 classes):
      Helix: H (alpha), G (3-10), I (pi)
      Sheet: E (extended beta), B (beta bridge)
      Coil:  T (turn), S (bend), - (loop/irregular)

    Why 3 classes instead of 8:
      The 8-state DSSP classification is too fine-grained for IDR analysis, where
      structure is dynamic. The 3 main classes capture the relevant structural
      properties: IDR binding regions tend to have more coil and less helix
      than ordered regions.

    Why AlphaFold instead of experimental structures:
      Most proteins in DisProt lack complete experimental structures in PDB.
      AlphaFold provides predicted structures for nearly all UniProt proteins,
      making systematic analysis possible.
      AlphaFold pLDDT values are correlated with disorder: low pLDDT (< 50)
      in IDRs indicates high confidence of disorder.

    Raises ValueError if a protein or region is not found in af_df:
      This is a blocking error because it indicates a mismatch between the reference
      and AlphaFold data (protein not downloaded or removed).

    Equivalent to the old compute_ss_percentages in functions_for_1107.
    """
    HELIX = {'H', 'G', 'I'}
    SHEET = {'E', 'B'}
    # Coil = everything else, computed by subtraction (100 - helix - sheet)
    rows  = []
    af_grouped = af_df.groupby(name_col)

    for idx, regions in regions_dict.items():
        for region in regions:
            acc   = region['acc']
            start = region['start']
            end   = region['end']

            if acc not in af_grouped.groups:
                raise ValueError(f"Protein {acc} not found in AlphaFold DataFrame")

            # Extract region residues and verify all are present
            df_region = (
                af_grouped.get_group(acc)
                .query(f"{pos_col} >= @start and {pos_col} <= @end")
                .sort_values(pos_col)
            )

            expected = end - start + 1
            if len(df_region) != expected:
                # Can happen if AlphaFold skipped residues (rare)
                raise ValueError(
                    f"Incomplete region {acc} {start}-{end}: "
                    f"expected {expected}, found {len(df_region)}"
                )

            ss    = df_region[ss_col]
            n     = len(ss)
            pct_h = round(ss.isin(HELIX).sum() / n * 100)
            pct_s = round(ss.isin(SHEET).sum() / n * 100)
            # Coil by subtraction: avoids rounding errors from three separate round() calls
            pct_c = 100 - pct_h - pct_s

            rows.append({
                'protein_id': idx,
                'acc':        acc,
                'length':     n,
                '%Helix':     pct_h,
                '%Sheet':     pct_s,
                '%Coil':      pct_c,
            })

    return pd.DataFrame(rows)


# ── Utilities ─────────────────────────────────────────────────────────────────

def classify_region(length):
    """
    Classify a region by length into 3 categories:
      <= 30 aa:  'small_region'   (e.g. short linear motifs SLiMs)
      31-99 aa:  'medium_region'  (typical IDR length)
      >= 100 aa: 'large_region'   (extended IDRs, e.g. intrinsically disordered tails)

    Why these cutoffs:
      30 aa is the conventional threshold for SLiMs (Short Linear Motifs) in the literature.
      100 aa separates typical IDRs from very long regions that may have statistically
      different physicochemical properties due to their length.
    """
    if length <= 30:
        return 'small_region'
    elif length < 100:
        return 'medium_region'
    return 'large_region'
