"""
loader.py — Loading raw data: LipNet and AlphaFold.

This module centralises reading of heavy input files:
  - LipNet:     .caid file with per-residue scores (one file, simple parser)
  - AlphaFold:  three CAID files (.dat / .caid) + a TSV with ss/pLDDT/RSA,
                merged into a single AF_all DataFrame

Why a separate module instead of notebook cells:
  The original cells (12, 14, 15, 16) mixed exploratory code
  (PDB download, AlphaFold availability check) with operational code (parsing).
  Here parsing is separated and importable; exploratory code lives in the
  __main__ block — it never runs on import.

missing_proteins:
  List of proteins absent from AlphaFold, computed once with the __main__ block
  (HEAD requests to alphafold.ebi.ac.uk) and hardcoded here.
  If the DisProt dataset changes, re-run the __main__ block and update the list.
"""

import os
import pandas as pd


# ── Proteins absent from AlphaFold ───────────────────────────────────────────
# Computed with HEAD requests on alphafold.ebi.ac.uk — see __main__ block.
# Do not change unless the 1107-protein dataset changes.
missing_proteins = [
    'P03045','P03129','P03406','P68336','P03422','P04324','P16009','Q06253',
    'P12497','P04325','P25054','P27958','Q9WMX2','Q07097','Q98157','Q9IH62',
    'Q9IK92','O89339','Q9IK91','Q5UPJ7','Q38151','O89467','P24937','Q9Q8E9',
    'A4ZNR2','P69723','P03421','P04608','P59595','P04578','Q05127','P03315',
    'P27392','P68927','A5YV76','Q99IB8','Q80FJ1','A8CDV5','Q7X2A1','Q1PAB4',
    'Q03164','P12296','O92972','P03255','P03259','P03086','P12823','Q32ZE1',
    'Q98XH7','P03520','A4L7I2','A3RMR8','A4L7I4','P03050','Q9DBZ9','P03126',
    'P03070','E5LC01','P06492','Q67953','Q9NQC3-1','P03254','Q98325',
    'Q8N726-1','P68466','Q5UB51','P26554','P04486','Q98148','P22473',
    'A5HC98','P04616','P0DTC2','A1Z9S6','P59594','Q5QGG3','P03407','Q90VU7',
    'Q997F2','O55777','P0DTC3','Q9J0X9','Q967F4','P0DTC9','J9V8B5','Q87GF9',
    'Q9NR48','Q77M43','Q9WIK7','P03437','Q9QR71','Q8JUX6','P69718','P19554',
    'P12520','P69697','Q05320','P87666','Q91AU0','P04591','P20879','P22363',
    'F0TTD6','A0A024B7W1','P57104','P27913','P0C746','P42858','P03170',
    'P26663','P0DXN6',
]


# ── LipNet ────────────────────────────────────────────────────────────────────

def load_lipnet(path):
    """
    Read a LipNet file in .caid format and return a per-residue DataFrame.

    File format:
      >disprot_id
      position   amino_acid   score
      ...

    Columns of the returned DataFrame:
      disprot_id   (str)   — DisProt ID e.g. 'DP00001'
      position     (int)   — 1-based position
      aa           (str)   — single-letter amino acid
      lipnet_score (float) — raw LipNet score (0–1)

    Why this parser is here and not using disprot.load_predictor_file:
      load_predictor_file names the score 'disorder_rsa' (generic CAID name).
      Here 'lipnet_score' is more explicit and consistent with the rest of the
      notebook, where it is compared against threshold 0.494.
    """
    rows, current_id = [], None
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line.startswith(">"):
                current_id = line[1:]
            elif line:
                pos, aa, score = line.split()
                rows.append({
                    "disprot_id":   current_id,
                    "position":     int(pos),
                    "aa":           aa,
                    "lipnet_score": float(score),
                })
    df = pd.DataFrame(rows)
    print(f"LipNet: {df['disprot_id'].nunique()} proteins, {len(df)} residues")
    return df


# ── AlphaFold ─────────────────────────────────────────────────────────────────

def build_af_all(af_caid_dir, disprot_tsv, filtered_proteins):
    """
    Load and merge the 4 AlphaFold files into a single AF_all DataFrame.

    Parameters:
      af_caid_dir       (str)  — directory containing AlphaFold CAID files
      disprot_tsv       (str)  — path to DisProt TSV (for the acc <-> disprot_id map)
      filtered_proteins (dict) — dictionary of selected proteins
                                 (same format as filtered_proteins_lipnet)

    Expected files in af_caid_dir:
      alphafold_disorder_pred_caid_disorder.dat        — per-residue pLDDT
      alphafold_disorder_pred_caid_disorder-25.dat     — per-residue RSA
      alphafold_disorder_pred_caid_binding-25-0.581.caid — binding score
      alphafold_disorder_pred_caid_data.tsv            — ss, pLDDT, RSA aggregated

    Returns:
      AF_all               (DataFrame) — main per-residue DataFrame
      Uniprot_Names_filtered (DataFrame) — acc <-> disprot_id mapping

    Why merge 4 files:
      CAID produces a separate file for each score type (pLDDT, RSA, binding).
      The TSV adds secondary structure (ss) computed by DSSP on AlphaFold.
      The final merge aligns all values on the same residue (acc + residue_num + aa).

    Why rename 'disorder_rsa' immediately after loading:
      disprot.load_predictor_file uses 'disorder_rsa' as a generic score name.
      Renaming before the merge prevents the three files from being joined on the
      score column instead of only on protein_id + residue_num + aa.
    """
    import sys
    sys.path.insert(0, os.path.dirname(__file__))
    import disprot

    # 1. acc <-> disprot_id map for selected proteins
    df_disprot    = pd.read_csv(disprot_tsv, sep="\t")
    disprot_names = [v['disprot_id'] for v in filtered_proteins.values()]
    Uniprot_Names_filtered = (
        df_disprot[df_disprot['disprot_id'].isin(disprot_names)][['acc', 'disprot_id']]
        .drop_duplicates()
    )

    # 2. Load the three CAID score files
    file_plddt   = os.path.join(af_caid_dir, "alphafold_disorder_pred_caid_disorder.dat")
    file_rsa     = os.path.join(af_caid_dir, "alphafold_disorder_pred_caid_disorder-25.dat")
    file_binding = os.path.join(af_caid_dir, "alphafold_disorder_pred_caid_binding-25-0.581.caid")

    # Rename 'disorder_rsa' immediately to avoid merge collisions
    df_plddt   = disprot.load_predictor_file(file_plddt).rename(columns={"disorder_rsa": "disorder_plddt"})
    df_rsa     = disprot.load_predictor_file(file_rsa)    # keeps 'disorder_rsa'
    df_binding = disprot.load_predictor_file(file_binding).rename(columns={"disorder_rsa": "binding"})

    # 3. Load TSV with secondary structure
    file_tsv = os.path.join(af_caid_dir, "alphafold_disorder_pred_caid_data.tsv")
    AF_data  = (
        pd.read_csv(file_tsv, sep="\t")
        .rename(columns={"name": "acc", "pos": "residue_num"})
    )

    # 4. Merge the three score files on disprot_id + residue_num + aa
    # Note: load_predictor_file returns column 'disprot_id' (not 'protein_id').
    # For AF files, 'disprot_id' actually contains the UniProt acc (e.g. 'P04637')
    # because AlphaFold CAID files use acc as header (>P04637).
    # Rename to 'acc' before merging with Uniprot_Names_filtered.
    AF_scores = (
        df_binding
        .merge(df_rsa,   on=["disprot_id", "residue_num", "aa"])
        .merge(df_plddt, on=["disprot_id", "residue_num", "aa"])
        .rename(columns={"disprot_id": "acc"})
        .merge(Uniprot_Names_filtered, on="acc")
    )

    # 5. Merge with TSV (secondary structure and aggregated AF values)
    AF_all = pd.merge(AF_data, AF_scores, on=["acc", "residue_num", "aa"], how="inner")

    print(f"AlphaFold: {AF_all['acc'].nunique()} proteins, {len(AF_all)} residues")
    return AF_all, Uniprot_Names_filtered


# ── Exploratory code (run manually) ──────────────────────────────────────────
if __name__ == "__main__":
    # Run this block standalone to:
    #   1. Download PDB files from AlphaFold (already done, do not repeat)
    #   2. Recompute missing_proteins if the dataset changes
    # Requires: pip install requests

    import requests, os

    ALPHAFOLD_URL = "https://alphafold.ebi.ac.uk/files/AF-{}-F1-model_v6.pdb"

    # -- PDB download (already done) --
    # OUT_DIR = "data/raw/pdb_alphafold"  # (see scripts/download_alphafold_pdb.py)
    # os.makedirs(OUT_DIR, exist_ok=True)
    # for uid in uniprot_ids:
    #     url = ALPHAFOLD_URL.format(uid)
    #     outfile = os.path.join(OUT_DIR, f"{uid}.pdb")
    #     if os.path.exists(outfile):
    #         continue
    #     r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
    #     if r.status_code == 200:
    #         with open(outfile, "wb") as f:
    #             f.write(r.content)

    # -- Compute missing_proteins (already done) --
    # Requires: uniprot_ids = Uniprot_Names_filtered['acc'].dropna().unique()
    # computed_missing = []
    # for uid in uniprot_ids:
    #     url = ALPHAFOLD_URL.format(uid)
    #     r = requests.head(url, headers={"User-Agent": "Mozilla/5.0"})
    #     if r.status_code != 200:
    #         computed_missing.append(uid)
    # print(computed_missing)
    pass
