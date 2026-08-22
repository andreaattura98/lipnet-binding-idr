# Pipeline

End-to-end steps to reproduce the evaluation of **LIPNet** and **AlphaFold-binding**
on the DisProt binding-IDR reference. Each step assumes the previous one succeeded.
All paths are resolved relative to the repository root; see
[`../data/README.md`](../data/README.md) for how to obtain the inputs.

## Overview

```
DisProt release ──► protein selection ──► FASTA ──► LIPNet (GPU) ──►┐
                                                                     ├──► CAID scoring ──► analysis
AlphaFold DB ──► PDB download ──► AlphaFold-disorder (DSSP+pLDDT) ──►┘
```

## Step 0 — Protein selection
From the DisProt release, select proteins carrying IDPO structural-transition terms
and GO `binding` (GO:0005488) descendants, applying the length / training-set /
binding-content filters described in the thesis. Produces
`data/raw/filtered_proteins_lipnet.json` (one record per protein: `acc`,
`disprot_id`, `sequence`, `length`, `regions`, …).

## Step 1 — FASTA for LIPNet
```bash
python scripts/build_lipnet_fasta.py
```
Writes `data/raw/lipnet_fasta.fasta` (2-line records: `>DP-ID` + sequence).

## Step 2 — LIPNet inference (GPU)
LIPNet runs on a GPU. Compute the ProtT5 embeddings, then the CNN predictions:
```bash
# embeddings
python src/lipnet4_embed.py --embed_dir <DIR>/ --embedding_mode compute \
    --input_file data/raw/lipnet_fasta.fasta
# per-residue scores -> outputs/all_predictions.caid
python src/lipnet4.py --embed_dir <DIR>/ --embedding_mode load \
    --input_file data/raw/lipnet_fasta.fasta
```
Save the result as `data/raw/LIPNet_output/lipnet.caid`.

## Step 3 — Download AlphaFold structures
```bash
python scripts/download_alphafold_pdb.py
```
Downloads full-chain models (skips fragmented `F2+` entries) into
`data/raw/pdb_alphafold/`.

## Step 4 — AlphaFold-disorder (DSSP + pLDDT → binding score)
```bash
# needs mkdssp; here invoked through WSL (override the env vars for your setup)
WSL_PY=/path/to/env/bin/python WSL_DSSP=/path/to/mkdssp \
python scripts/run_alphafold_disorder.py
```
Runs [`alphafold_disorder.py`](https://github.com/BioComputingUP/AlphaFold-disorder),
remaps UniProt ACC → DisProt-ID, and writes
`data/raw/Alphafold_CAID_output/alphafold.caid`.

## Step 5 — CAID scoring
Score both predictors against the reference with the official CAID scorer
([BioComputingUP/CAID](https://github.com/BioComputingUP/CAID)):
```bash
python caid.py \
    -ref data/references/allowed_80.fasta \
    -pred data/raw/LIPNet_output/lipnet.caid \
    -pred data/raw/Alphafold_CAID_output/alphafold.caid \
    -out data/processed/Output_CAID_allowed80/
```
Produces Fmax, AUC-ROC, AUC-PR and the optimal thresholds per metric.

## Step 6 — Analysis notebooks
- [`../notebooks/evaluation.ipynb`](../notebooks/evaluation.ipynb) — main evaluation:
  loads predictions, computes ROC/PR, confusion matrices, and the localCIDER
  sequence-composition analysis.
- [`../notebooks/detailed_analysis.ipynb`](../notebooks/detailed_analysis.ipynb) —
  extended analysis (structural characterisation, per-group descriptors).

> The notebooks are shipped **with their rendered outputs** as a record of the
> analysis. Re-running them requires the external inputs listed in
> [`../data/README.md`](../data/README.md).

## Environment notes
- LIPNet inference needs a GPU (ProtT5 is large); everything else runs on CPU.
- `scripts/run_alphafold_disorder.py` calls `mkdssp` through WSL by default —
  configure `WSL_DISTRO`, `WSL_PY`, `WSL_DSSP` for your machine, or adapt it to a
  native DSSP install.
