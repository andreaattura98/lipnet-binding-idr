"""
disprot.py — Loading and pre-processing of DisProt data.

DisProt (https://disprot.org) is the reference database for intrinsically
disordered regions (IDRs). Each release is distributed as a JSON file containing
a list of proteins, each with metadata and a list of regions manually annotated
with ontological terms (GO, IDPO).

This module covers operations on raw DisProt data, before building the CAID
reference. Reference-building operations are in reference.py.
"""

import ijson
import pandas as pd


# ── JSON loading ──────────────────────────────────────────────────────────────

def load_json(path):
    """
    Stream-read a DisProt release JSON file and return a flat dictionary
    indexed by sequential integer.

    Why ijson and not json.load():
      The DisProt JSON file (release 2025_06) is over 200 MB. Loading it all
      into RAM with json.load() would require >1 GB for deserialisation.
      ijson reads the file as a stream, emitting one object at a time: peak
      RAM stays low regardless of file size.

    Structure of each entry in the returned dictionary:
      {
        'acc':        str,   # UniProt accession (e.g. 'P04637')
        'disprot_id': str,   # DisProt ID (e.g. 'DP00003')
        'name':       str,   # protein name
        'organism':   str,   # source organism
        'sequence':   str,   # full amino acid sequence (1-letter)
        'length':     int,   # sequence length
        'regions':    dict,  # {i: {start, end, term_id}} — 1-based indices
      }

    Regions without a 'start' field (e.g. composite regions) are silently ignored.
    """
    json_dic = {}
    with open(path, 'r', encoding='utf-8') as f:
        # 'data.item' is the ijson path matching elements of the "data" array in the JSON
        for idx, prot in enumerate(ijson.items(f, 'data.item')):
            regions = prot.get('regions', [])
            json_dic[idx] = {
                'acc':        prot.get('acc'),
                'disprot_id': prot.get('disprot_id'),
                'name':       prot.get('name'),
                'organism':   prot.get('organism'),
                'sequence':   prot.get('sequence'),
                'length':     prot.get('length'),
                'regions': {
                    i: {
                        'start':   r['start'],   # 1-based start position
                        'end':     r['end'],      # 1-based end position (inclusive)
                        'term_id': r.get('term_id'),  # e.g. 'IDPO:0000013' (D_O_Binding)
                    }
                    for i, r in enumerate(regions) if 'start' in r
                },
            }
    return json_dic


# ── Filter by ontological terms ───────────────────────────────────────────────

def filter_by_terms(json_dic, allowed_terms):
    """
    Return only proteins that have at least one region annotated with a
    term_id in allowed_terms.

    Why this step precedes filter_by_term in reference.py:
      This filter operates on the raw DisProt dataset (json_dic) and builds
      filtered_proteins_lipnet — the pool from which LipNet and AlphaFold
      scores are computed.
      filter_by_term in reference.py operates instead on filtered_proteins_REF
      (already stripped of training proteins and those missing from AlphaFold)
      to build the reference for the active configuration.
      These are two distinct filters on different protein sets.

    Equivalent to the old filter_proteins_by_terms.
    """
    return {
        idx: prot
        for idx, prot in json_dic.items()
        if any(r.get('term_id') in allowed_terms for r in prot.get('regions', {}).values())
    }


# ── Prepare input for LipNet ──────────────────────────────────────────────────

def to_fasta_dict(json_dic, fields=('disprot_id', 'sequence')):
    """
    Reduce each dictionary entry to only the specified fields.

    Why: LipNet takes as input a FASTA with only ID and sequence. Passing the
    full dictionary (with regions, organism, etc.) to write_lipnet_fasta would
    be unnecessary and wasteful. This function prepares a minimal version of
    the dictionary before FASTA writing.

    Default: keeps only disprot_id and sequence. Change fields if you need
    other metadata in the FASTA (e.g. for debugging).

    Equivalent to the old keep_only_fields.
    """
    return {
        idx: {key: prot[key] for key in fields if key in prot}
        for idx, prot in json_dic.items()
    }


def write_lipnet_fasta(fasta_dict, output_path):
    """
    Write a single-line FASTA file (one line per sequence, no 60/80-character wrapping)
    suitable as input for LipNet.

    Why single-line:
      LipNet reads sequences assuming each sequence fits on a single line.
      Multi-line FASTA (biological standard) would cause parsing errors in LipNet.
      The replace() removes any internal newlines that some DisProt exports may include.

    Equivalent to the old write_fasta_single_line.
    """
    with open(output_path, 'w') as f:
        for idx, prot in fasta_dict.items():
            prot_id  = prot.get('disprot_id', f'protein_{idx}')
            # strip internal newlines: guard against non-standard DisProt exports
            sequence = prot.get('sequence', '').replace('\n', '').replace('\r', '')
            f.write(f'>{prot_id}\n{sequence}\n')


# ── Load disorder predictor output ────────────────────────────────────────────

def load_predictor_file(file_path):
    """
    Read a predictor output file in CAID columnar format:
      > disprot_id
      residue_num  aa  score    (one row per residue, space-separated columns)

    This format is used by LipNet (.caid), AlphaFold disorder proxy,
    and other predictors evaluated in CAID.

    Rows with a column count != 3 are silently skipped: empty lines or comment
    lines are common in predictor output files.

    Returns a DataFrame with columns:
      disprot_id, residue_num, aa, disorder_rsa

    Equivalent to the old load_disorder_file (renamed because it loads any
    predictor, not only the disorder RSA proxy).
    """
    rows, current_id = [], None
    with open(file_path, 'r') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue   # empty line → skip
            if line.startswith('>'):
                current_id = line[1:]   # new header: update current protein
            else:
                parts = line.split()
                if len(parts) != 3:
                    continue   # malformed line → skip without interrupting
                res_num, aa, score = parts
                rows.append({
                    'disprot_id':   current_id,
                    'residue_num':  int(res_num),
                    'aa':           aa,
                    'disorder_rsa': float(score),
                })
    return pd.DataFrame(rows)
