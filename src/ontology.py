"""
ontology.py — Termini GO/IDPO e set di annotazione per il benchmark CAID.

Questo modulo centralizza la definizione di tutti i termini ontologici usati
per costruire i reference CAID. Il notebook importa le variabili già pronte
senza dover definire nessun ID direttamente nelle celle.

Come sono stati ricavati i binding_children_ids:
  Estratti dal grafo GO usando obonet (vedere blocco __main__ in fondo).
  Il nodo radice è GO:0005488 ("binding"); i figli diretti sono tutti i
  tipi di binding presenti in go-basic.obo. L'estrazione è stata fatta
  una volta sola — i valori non cambiano finché non cambia la versione di GO.

Perché hardcoded e non caricati a runtime da go-basic.obo:
  obonet legge un file da ~50 MB e costruisce un grafo — circa 30 s.
  I termini estratti non cambiano tra un'esecuzione e l'altra.
  Se la versione GO cambia, si riesegue il blocco __main__ e si aggiornano
  i set qui sotto.
"""


# ── GO: figli diretti di GO:0005488 ("binding") ───────────────────────────────
# Ricavati con networkx.ancestors(graph, "GO:0005488") — vedere __main__.
binding_children_ids = {
    'GO:0003676', 'GO:0003682', 'GO:0003823', 'GO:0005515', 'GO:0005527',
    'GO:0005549', 'GO:0008289', 'GO:0015643', 'GO:0016597', 'GO:0019215',
    'GO:0030246', 'GO:0031409', 'GO:0033218', 'GO:0034617', 'GO:0035274',
    'GO:0035275', 'GO:0036094', 'GO:0042277', 'GO:0042562', 'GO:0043176',
    'GO:0043515', 'GO:0043546', 'GO:0044877', 'GO:0046812', 'GO:0046906',
    'GO:0050809', 'GO:0050840', 'GO:0051192', 'GO:0072341', 'GO:0097160',
    'GO:0097243', 'GO:0097367', 'GO:1901265', 'GO:1901338', 'GO:1901681',
    'GO:1902314', 'GO:1904483', 'GO:1905594', 'GO:1990300',
}

# ── IDPO: functional subcategories of IDR regions ────────────────────────────

# Structural transitions: IDR that binds by adopting structure (coupled folding)
disorder_to_order    = {"IDPO:0000011", "IDPO:0000012", "IDPO:0000013"} # DO: 0000011 (disorder to order), 0000012 (molten globule to order), 0000013 (pre-molten globule to order)

partial_folding      = {"IDPO:0000018", "IDPO:0000019", "IDPO:0000023"} # DD-extra: 0000018 (disorder→molten globule), 0000019 (disorder→pre-molten globule), 0000023 (pre-molten globule→molten globule)


# Reverse transitions: structure that becomes disordered upon binding (less common)
# These are not "binding" → labelled '0' in the reference (ZEROS)
order_to_disorder    = {"IDPO:0000014", "IDPO:0000015", "IDPO:0000016"}

# Fuzzy binding: IDR that binds while remaining disordered (e.g. nucleoporins)
# Included in D_O_D_D_Binding but not in D_O_Binding — key difference between the two configs
disorder_to_disorder = {
    "IDPO:0000018", "IDPO:0000019", "IDPO:0000020",
    "IDPO:0000021", "IDPO:0000022", "IDPO:0000023",
}

# Display sites: IDR regions with a presentation function
# Not binding in the strict sense → labelled '0' in the reference (ZEROS)
display_site         = {
    "IDPO:0000037", "IDPO:0000038", "IDPO:0000039",
    "IDPO:0000040", "IDPO:0000041", "IDPO:0000042",
    "IDPO:0000043", "IDPO:0000044", "IDPO:0000045",
    "IDPO:0000046", "IDPO:0000047", "IDPO:0000048",
}

# Self-binding: IDR that binds itself or identical copies of the protein
self_binding         = {
    "IDPO:0000056", "IDPO:0000057", "IDPO:0000058",
    "IDPO:0000059", "IDPO:0000060",
}

# Generico "disorder": IDR non ulteriormente classificata
DISORDER             = {"IDPO:0000002"}


# ── Set derivati ──────────────────────────────────────────────────────────────

# binding = all GO binding types + self_binding IDPO
# These residues are labelled '1' in the reference (positives)
binding = binding_children_ids | self_binding

# SELECTED_IDPO = all IDPO terms that bring a protein into the reference
# (some categories enter as '0', but are needed to build the IDR sequence)
SELECTED_IDPO = (
    disorder_to_order    |
    order_to_disorder    |
    disorder_to_disorder |
    display_site         |
    self_binding
)

positives = (disorder_to_order | partial_folding)


# allowed_terms = filter for disprot.filter_by_terms:
# keep only proteins that have at least one region in this set
allowed_terms = binding_children_ids | SELECTED_IDPO

# D_O_Binding: "baseline" annotation set — DOB_80 and DOB_100 configurations
# Residues labelled '1': GO binding + disorder_to_order (coupled folding)
D_O_Binding = binding | disorder_to_order

# D_O_D_D_Binding: "extended" annotation set — DDB_80 and DDB_100 configurations
# Adds disorder_to_disorder (fuzzy binding), which D_O_Binding ignores.
# Why: fuzzy complexes are biologically relevant for LipNet —
# excluding them could artificially penalise predictors that capture them.
D_O_D_D_Binding = binding | disorder_to_order | disorder_to_disorder

# ZEROS = terms that enter the reference but are labelled '0' (non-binding)
# Required for the two-pass in reference.build_reference: first all annotated
# IDRs → '0', then only binding → '1'. Without ZEROS, non-binding IDR
# positions would be lost from the final reference.
ZEROS = display_site | order_to_disorder | disorder_to_disorder | DISORDER


total_IDPO_terms = ['IDPO:0000031', 'IDPO:0000006', 'IDPO:0000038', 'IDPO:0000044', 'IDPO:0000011', 'IDPO:0000032', 'IDPO:0000048', 'IDPO:0000004', 'IDPO:0000022', 'IDPO:0000012', 'IDPO:0000016', 'IDPO:0000058', 'IDPO:0000003', 'IDPO:0000013', 'IDPO:0000040', 'IDPO:0000018', 'IDPO:0000039', 'IDPO:0000041', 'IDPO:0000057', 'IDPO:0000060', 'IDPO:0000030', 'IDPO:0000033', 'IDPO:0000059', 'IDPO:0000045', 'IDPO:0000002', 'IDPO:0000014', 'IDPO:0000037', 'IDPO:0000023', 'IDPO:0000043', 'IDPO:0000046', 'IDPO:0000019', 'IDPO:0000042']

ZEROS_GIUGNO = set(total_IDPO_terms) - allowed_terms


# ── Estrazione obonet (eseguire manualmente se cambia la versione GO) ─────────
if __name__ == "__main__":
    # Eseguire questo blocco standalone per ricalcolare binding_children_ids
    # da una nuova versione di go-basic.obo.
    # Richiede: pip install obonet networkx
    import obonet
    import networkx

    from pathlib import Path
    go_basic = str(Path(__file__).resolve().parents[1] / "data" / "references" / "go-basic.obo")
    graph = obonet.read_obo(go_basic)

    # Statistiche del grafo GO
    print(f"Nodi: {len(graph)}")
    print(f"Archi: {graph.number_of_edges()}")
    print(f"E un DAG: {networkx.is_directed_acyclic_graph(graph)}")

    # Mappa ID → nome leggibile
    id_to_name = {id_: data.get("name") for id_, data in graph.nodes(data=True)}
    name_to_id = {data["name"]: id_ for id_, data in graph.nodes(data=True) if "name" in data}

    print(f"GO:0005488 = '{id_to_name['GO:0005488']}'")

    # Trova i figli diretti di GO:0005488 ("binding")
    # graph.in_edges(node): archi che puntano VERSO il nodo = figli nel DAG GO
    node = name_to_id["binding"]
    computed_ids = set()
    for parent, child, key in graph.in_edges(node, keys=True):
        computed_ids.add(parent)

    print(f"Figli diretti di binding: {len(computed_ids)}")
    print(sorted(computed_ids))

    # Tutti i discendenti (sottotipi più specifici) di GO:0005488
    GO_specific_bindings = sorted(id_to_name[s] for s in networkx.ancestors(graph, "GO:0005488"))
    print(f"Discendenti totali: {len(GO_specific_bindings)}")
