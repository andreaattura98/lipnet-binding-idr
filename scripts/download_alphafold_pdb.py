"""Scarica i PDB di AlphaFold-DB per le proteine elencate in filtered_proteins_lipnet.json.

Regola "solo sequenza intera": prima di scaricare F1, fa una HEAD su F2; se F2 esiste,
la proteina e' frammentata in AFDB e viene SKIPPATA (niente download). Se F2 ritorna 404
si scarica F1 (sequenza intera).

Output: <out_dir>/<ACC>.pdb (un file per proteina scaricata).
Idempotente: se <out_dir>/<ACC>.pdb esiste gia', salta.

CLI:
    python AF_pdb.py                            # path di default
    python AF_pdb.py -i IN -o OUT
"""
import argparse
import json
import sys
from pathlib import Path

import requests

REPO = Path(__file__).resolve().parents[1]          # repo root (scripts/ -> ..)
DEFAULT_IN  = REPO / "data" / "raw" / "filtered_proteins_lipnet.json"
DEFAULT_OUT = REPO / "data" / "raw" / "pdb_alphafold"

URL_TPL = "https://alphafold.ebi.ac.uk/files/AF-{acc}-F{frag}-model_v6.pdb"
UA = {"User-Agent": "Mozilla/5.0"}


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input",  type=Path, default=DEFAULT_IN,
                    help=f"JSON con record che contengono 'acc' (default: {DEFAULT_IN})")
    ap.add_argument("-o", "--output-dir", type=Path, default=DEFAULT_OUT,
                    help=f"cartella di output per i .pdb (default: {DEFAULT_OUT})")
    return ap.parse_args()


def load_accs(json_path: Path):
    """Ritorna la lista (ordinata, deduplicata) degli ACC nei record del JSON."""
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    records = obj.values() if isinstance(obj, dict) else obj
    seen = []
    seen_set = set()
    for rec in records:
        acc = rec.get("acc")
        if acc and acc not in seen_set:
            seen.append(acc)
            seen_set.add(acc)
    return seen


def main():
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"ERROR: input non trovato: {args.input}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    accs = load_accs(args.input)
    print(f"ACC trovati nel JSON: {len(accs)}")

    n_downloaded = n_skipped_exists = n_fragmented = n_missing = 0
    fragmented, missing = [], []

    for i, acc in enumerate(accs, 1):
        outfile = args.output_dir / f"{acc}.pdb"
        if outfile.exists():
            n_skipped_exists += 1
            print(f"[{i}/{len(accs)}] {acc}: gia' presente, skip")
            continue

        # pre-check: F2 esiste? -> frammentata, skip
        url_f2 = URL_TPL.format(acc=acc, frag=2)
        try:
            r2 = requests.head(url_f2, headers=UA, timeout=20, allow_redirects=True)
        except requests.RequestException as e:
            sys.exit(f"ERROR: rete fallita su HEAD {url_f2}: {e}")
        if r2.status_code == 200:
            n_fragmented += 1
            fragmented.append(acc)
            print(f"[{i}/{len(accs)}] {acc}: FRAMMENTATA (F2 esiste), skip")
            continue

        # scarica F1
        url_f1 = URL_TPL.format(acc=acc, frag=1)
        try:
            r1 = requests.get(url_f1, headers=UA, timeout=60)
        except requests.RequestException as e:
            sys.exit(f"ERROR: rete fallita su GET {url_f1}: {e}")
        if r1.status_code == 200:
            outfile.write_bytes(r1.content)
            n_downloaded += 1
            print(f"[{i}/{len(accs)}] {acc}: scaricato -> {outfile.name}")
        else:
            n_missing += 1
            missing.append(acc)
            print(f"[{i}/{len(accs)}] {acc}: NON in AFDB (HTTP {r1.status_code})")

    print()
    print("=" * 50)
    print(f"Totale ACC nel JSON     : {len(accs)}")
    print(f"  gia' presenti (skip)  : {n_skipped_exists}")
    print(f"  scaricati ora         : {n_downloaded}")
    print(f"  frammentate (skip)    : {n_fragmented}")
    print(f"  mancanti su AFDB      : {n_missing}")
    if fragmented:
        print(f"\nfragmented ({len(fragmented)}): {fragmented}")
    if missing:
        print(f"\nmissing on AFDB ({len(missing)}): {missing}")


if __name__ == "__main__":
    main()
