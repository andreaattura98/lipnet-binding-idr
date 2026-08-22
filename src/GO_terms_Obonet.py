"""Estrazione dei figli del termine GO "binding" (GO:0005488) dalla Gene Ontology.

Espone:
    binding_children : set[str]   -> tutti i GO term discendenti is_a di "binding" (+ binding stesso)

Calcolato a livello di modulo per compatibilita' con il codice esistente:
    import GO_terms_Obonet
    GO_terms_Obonet.binding_children
"""

from pathlib import Path

import obonet
import networkx as nx

# go-basic.obo is a large ontology file, not shipped in the repo (see data/README.md
# for the download link). Default path is repo-relative; override with the env var.
import os
GO_BASIC_PATH = os.environ.get(
    "GO_BASIC_OBO",
    str(Path(__file__).resolve().parents[1] / "data" / "references" / "go-basic.obo"),
)
BINDING_GO_ID = "GO:0005488"


def get_binding_children(go_basic_path: str = GO_BASIC_PATH) -> set:
    """Carica l'OBO e ritorna il set dei discendenti is_a di 'binding' (+ binding)."""
    G = obonet.read_obo(go_basic_path)
    # tieni solo gli archi is_a (gli archi vanno figlio -> genitore)
    H = G.edge_subgraph(
        [(u, v, k) for u, v, k in G.edges(keys=True) if k == "is_a"]
    )
    # tutti i children = discendenti is_a di "binding" (+ binding stesso)
    return nx.ancestors(H, BINDING_GO_ID) | {BINDING_GO_ID}


# Calcolato all'import per compatibilita' con GO_terms_Obonet.binding_children.
# Se l'OBO non e' presente localmente, l'import non deve fallire: si usa un set vuoto
# e si emette un avviso (ricalcolabile con get_binding_children() dopo il download).
if Path(GO_BASIC_PATH).exists():
    binding_children = get_binding_children()
else:
    import warnings
    warnings.warn(
        f"go-basic.obo non trovato in {GO_BASIC_PATH}; binding_children = set() vuoto. "
        "Scaricare l'OBO (vedi data/README.md) e richiamare get_binding_children()."
    )
    binding_children = set()


if __name__ == "__main__":
    print(len(binding_children))
    print(sorted(binding_children))
