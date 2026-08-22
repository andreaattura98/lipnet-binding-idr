"""Genera il FASTA di input per LIPNet a partire da filtered_proteins_lipnet.json.

Formato di output: single-line FASTA a 2 righe per proteina:
    >DP00004
    MKTQRDGHSLGRWSLVLLLLGLVMPLAIIAQVLSYKEAVLRAIDGINQRSSDANLYRLLD...
Header = `disprot_id` del record; corpo = `sequence`. Nessuna label.

CLI:
    python lipnet_fasta.py                 # usa i path di default
    python lipnet_fasta.py -i IN -o OUT    # path custom
"""
import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]          # repo root (scripts/ -> ..)
DEFAULT_IN  = REPO / "data" / "raw" / "filtered_proteins_lipnet.json"
DEFAULT_OUT = REPO / "data" / "raw" / "lipnet_fasta.fasta"


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-i", "--input",  type=Path, default=DEFAULT_IN,
                    help=f"JSON di input (default: {DEFAULT_IN})")
    ap.add_argument("-o", "--output", type=Path, default=DEFAULT_OUT,
                    help=f"FASTA di output (default: {DEFAULT_OUT})")
    return ap.parse_args()


def load_records(json_path: Path):
    """Carica il JSON e ritorna una lista di record (dict). Supporta sia top-level
    dict (chiavi numeriche) che list."""
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        return list(obj.values())
    if isinstance(obj, list):
        return obj
    raise ValueError(f"top-level JSON inatteso: {type(obj).__name__}")


def main():
    args = parse_args()
    if not args.input.exists():
        sys.exit(f"ERROR: input non trovato: {args.input}")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    records = load_records(args.input)
    print(f"caricati {len(records)} record da {args.input}")

    written = 0
    seen = set()           # DP-ID gia' scritti (per duplicati)
    duplicates = []        # DP-ID duplicati incontrati (skippati)
    length_mismatch = []   # (dp_id, len(sequence), length dichiarata)

    with open(args.output, "w", encoding="utf-8", newline="\n") as out:
        for idx, rec in enumerate(records):
            dp = rec.get("disprot_id")
            seq = rec.get("sequence")
            if not dp or not seq:
                sys.exit(f"ERROR: record {idx} senza disprot_id o sequence: {rec.keys()}")
            if dp in seen:
                duplicates.append(dp)
                continue
            seen.add(dp)
            declared_len = rec.get("length")
            if declared_len is not None and declared_len != len(seq):
                length_mismatch.append((dp, len(seq), declared_len))
            out.write(f">{dp}\n{seq}\n")
            written += 1

    print(f"scritte {written} proteine in {args.output}")
    if duplicates:
        print(f"WARNING: {len(duplicates)} disprot_id duplicati (skippati, tenuto il primo): "
              f"{duplicates[:10]}{'...' if len(duplicates) > 10 else ''}")
    if length_mismatch:
        print(f"WARNING: {len(length_mismatch)} record con len(sequence) != length:")
        for dp, ns, nl in length_mismatch[:10]:
            print(f"   {dp}: sequence={ns}  length dichiarata={nl}")
        if len(length_mismatch) > 10:
            print(f"   ... ({len(length_mismatch)-10} altri)")


if __name__ == "__main__":
    main()
