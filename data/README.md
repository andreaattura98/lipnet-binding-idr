# Data

This repository ships only **small, redistributable reference files**. The large
inputs (protein structures, model weights, raw predictions) are **not versioned**
— they are downloaded or regenerated with the scripts in [`../scripts`](../scripts)
and the pipeline in [`../docs/PIPELINE.md`](../docs/PIPELINE.md).

## What is included

```
data/
└── references/
    ├── allowed_80.fasta     # DisProt binding-IDR reference (3-line CAID format), 80% coverage cutoff
    └── allowed_100.fasta    # same, 100% coverage cutoff
```

The 3-line CAID format is one record per protein: header, sequence, and a per-residue
label string (`1` = binding, `0` = disordered non-binding, `-` = not evaluated).

## What is NOT included (and how to get it)

| Item | Size | How to obtain |
|------|------|---------------|
| **ProtT5-XL-UniRef50** weights | ~2.5 GB (10+ GB uncompressed) | Auto-downloaded by `transformers` (`Rostlab/prot_t5_xl_uniref50`) or from the [ProtTrans repo](https://github.com/agemagician/ProtTrans). |
| **LIPNet weights** (`cnn_model.pth`) | ~a few MB | From the LIPNet repo: [BioComputingUP/LIPNet](https://github.com/BioComputingUP/LIPNet). |
| **DisProt release** (JSON/TSV) | ~tens of MB | [disprot.org/download](https://disprot.org/download) (release 2025_06 used here). |
| **AlphaFold PDB structures** | ~500 MB | `python scripts/download_alphafold_pdb.py` (pulls from the AlphaFold DB). |
| **Gene Ontology** (`go-basic.obo`) | ~150 MB | [geneontology.org/docs/download-ontology](https://geneontology.org/docs/download-ontology/) → place in `data/references/go-basic.obo`. |

Once downloaded, place raw inputs under `data/raw/` and generated outputs under
`data/processed/` (both git-ignored). Paths are resolved relative to the repo root,
so no absolute paths need editing.
