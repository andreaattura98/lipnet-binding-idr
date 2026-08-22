"""Esegue alphafold_disorder.py (DSSP + pLDDT) sui PDB scaricati e produce un file
.caid con header DP-ID (pronto per CAID).

Flusso:
 1. Esegue ./AlphaFold-disorder/alphafold_disorder.py via WSL (env_afdisorder, mkdssp 3.x)
    con -f caid -> genera in --out-dir:
        out_data.tsv
        out_binding-25-0.581.dat   (header = UniProt ACC)
        out_disorder.dat
        out_disorder-25.dat
 2. Rimappa out_binding-25-0.581.dat sostituendo gli header >ACC con >DP-ID
    (mappa ACC->DP-ID letta da filtered_proteins_lipnet.json) e salva alphafold.caid
    nella stessa cartella.

Vincoli WSL/ambiente (configurabili via env var WSL_DISTRO / WSL_PY / WSL_DSSP):
 - WSL distro con mkdssp 3.x installato (default: Ubuntu)
 - python env conda con biopython/numpy/pandas
 - mkdssp 3.0.0

CLI:
    python AF_disorder.py                 # tutti i default
    python AF_disorder.py -p IN -o OUT -j JSON -s SCRIPT
"""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

BASE = Path(__file__).resolve().parents[1]           # repo root (scripts/ -> ..)
DEFAULT_PDB    = BASE / "data" / "raw" / "pdb_alphafold"
DEFAULT_OUT    = BASE / "data" / "raw" / "Alphafold_CAID_output"
DEFAULT_JSON   = BASE / "data" / "raw" / "filtered_proteins_lipnet.json"
DEFAULT_SCRIPT = BASE / "AlphaFold-disorder" / "alphafold_disorder.py"

# AlphaFold-disorder needs mkdssp, run here through WSL. Override these for your
# environment via the env vars below (see docs/PIPELINE.md).
WSL_DISTRO = os.environ.get("WSL_DISTRO", "Ubuntu")
WSL_PY     = os.environ.get("WSL_PY",   "/home/USER/miniforge3/envs/env_afdisorder/bin/python")
WSL_DSSP   = os.environ.get("WSL_DSSP", "/home/USER/miniforge3/envs/env_afdisorder/bin/mkdssp")


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("-p", "--pdb-dir", type=Path, default=DEFAULT_PDB,
                    help=f"cartella con i .pdb di input (default: {DEFAULT_PDB})")
    ap.add_argument("-o", "--out-dir", type=Path, default=DEFAULT_OUT,
                    help=f"cartella di output (default: {DEFAULT_OUT})")
    ap.add_argument("-j", "--json", type=Path, default=DEFAULT_JSON,
                    help=f"JSON con mappa ACC->disprot_id (default: {DEFAULT_JSON})")
    ap.add_argument("-s", "--af-script", type=Path, default=DEFAULT_SCRIPT,
                    help=f"path locale di alphafold_disorder.py (default: {DEFAULT_SCRIPT})")
    return ap.parse_args()


def to_wsl(p: Path) -> str:
    """Converte un path Windows in path WSL (/mnt/<lower-drive>/...)."""
    s = str(p).replace("\\", "/")
    if len(s) >= 3 and s[1:3] == ":/":
        return "/mnt/" + s[0].lower() + s[2:]
    return s


def run(cmd, check=True):
    cmd = [str(c) for c in cmd]
    print(">>", " ".join(cmd))
    p = subprocess.run(cmd, capture_output=True, text=True)
    if p.stdout: print("---- stdout ----\n" + p.stdout.rstrip())
    if p.stderr: print("---- stderr ----\n" + p.stderr.rstrip())
    print("exit code:", p.returncode)
    if check and p.returncode != 0:
        sys.exit(f"ERROR: comando fallito (exit {p.returncode})")
    return p


def load_acc_to_dp(json_path: Path):
    """ACC -> disprot_id. Conflitto se un ACC mappa a piu' DP-ID."""
    with open(json_path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    records = obj.values() if isinstance(obj, dict) else obj
    m = {}
    conflicts = {}
    for rec in records:
        acc = rec.get("acc")
        dp  = rec.get("disprot_id")
        if not acc or not dp:
            continue
        if acc in m and m[acc] != dp:
            conflicts.setdefault(acc, set()).update({m[acc], dp})
        else:
            m[acc] = dp
    if conflicts:
        print(f"WARNING: {len(conflicts)} ACC mappano a piu' DP-ID (tenuto il primo):")
        for acc, dps in list(conflicts.items())[:10]:
            print(f"   {acc}: {sorted(dps)}")
    return m


def remap_caid_dat(src: Path, dst: Path, acc2dp: dict):
    """Sostituisce le righe '>ACC' con '>DP-ID' nel .dat prodotto da alphafold_disorder.

    I blocchi senza mappa ACC->DP-ID vengono saltati (con log).
    """
    out_lines = []
    n_blocks = n_written = 0
    skipped = []
    skipping = False
    for raw in src.read_text().splitlines():
        if raw.startswith(">"):
            n_blocks += 1
            acc = raw[1:].strip()
            dp = acc2dp.get(acc)
            if dp is None:
                skipping = True
                skipped.append(acc)
                continue
            skipping = False
            n_written += 1
            out_lines.append(f">{dp}")
        else:
            if not skipping:
                out_lines.append(raw)
    dst.write_text("\n".join(out_lines) + ("\n" if out_lines else ""), newline="\n")
    print(f"rimappa: blocchi totali={n_blocks}  rimappati (DP-ID)={n_written}  "
          f"saltati (no mapping)={len(skipped)}")
    if skipped:
        print(f"  ACC senza mapping (primi 10): {skipped[:10]}")
    return n_written


def main():
    args = parse_args()
    for needed in (args.pdb_dir, args.json, args.af_script):
        if not needed.exists():
            sys.exit(f"ERROR: prerequisito mancante: {needed}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    # 1) lancia alphafold_disorder.py via WSL
    out_stem = args.out_dir / "out.tsv"   # lo script usa solo .stem; estensione ignorata
    bash_cmd = (
        f"{WSL_PY} {to_wsl(args.af_script)} "
        f"-i {to_wsl(args.pdb_dir)}/ -o {to_wsl(out_stem)} "
        f"-f caid -dssp {WSL_DSSP} -ll info"
    )
    run(["wsl", "-d", WSL_DISTRO, "--", "bash", "-lc", bash_cmd])

    raw_dat = args.out_dir / "out_binding-25-0.581.dat"
    if not raw_dat.exists():
        sys.exit(f"ERROR: output atteso mancante: {raw_dat}")
    print("\nfile prodotti da alphafold_disorder.py:")
    for f in sorted(args.out_dir.iterdir()):
        if f.name.startswith("out_"):
            print(f"  {f.name}  ({f.stat().st_size} B)")

    # 2) rimappa ACC -> DP-ID
    acc2dp = load_acc_to_dp(args.json)
    print(f"\nmappa ACC->DP-ID caricata da {args.json.name}: {len(acc2dp)} entries")
    out_caid = args.out_dir / "alphafold.caid"
    n = remap_caid_dat(raw_dat, out_caid, acc2dp)
    print(f"\nDONE: scritto {out_caid}  ({n} proteine con DP-ID)")


if __name__ == "__main__":
    main()
