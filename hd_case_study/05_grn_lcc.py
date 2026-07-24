#!/usr/bin/env python3
"""
05_grn_lcc.py   [run order: 05]

Take the LARGEST CONNECTED SUBCOMPONENT of the regulatory network.

Reads one or more edge-list files -- the netcontrol GRN now, and the 04 DNB edges
too when ready -- unions them, drops self-loops, and extracts the largest
WEAKLY-connected component. Edge direction is preserved in the output (netcontrol
edges are directed) for downstream GRN modelling.

Edge-list format: one edge per line, first two tokens split on  ; , tab or space.
Header rows (source/target/tf/... ) are skipped. Self-loop-only nodes are dropped
from the LCC (no relational information).

Inputs : auto-located Network-*.txt in the project root, plus any paths passed as
         args (e.g. the 04 DNB consensus edge list, once it exists).
Outputs (<root>/grn/):
  grn_lcc_edges.csv        source,target   (directed, LCC only)  <- the module substrate
  grn_lcc_nodes.txt        LCC node symbols
  grn_all_edges_clean.csv  all deduped non-self edges (every component)
  grn_lcc_summary.txt

RECORD: if it errors, fix and re-run.

Usage: python 05_grn_lcc.py [extra_edgelist ...]
"""
import csv, glob, os, re, sys

HEADERWORDS = {"source", "target", "tf", "gene", "edge", "from", "to",
               "node1", "node2", "regulator", "genea", "geneb"}
SPLIT = re.compile(r"[;,\t ]+")


def find_root():
    f = globals().get("__file__")
    return os.path.dirname(os.path.abspath(f)) if f else os.getcwd()


def parse_edges(path):
    out = []
    with open(path) as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            toks = [t for t in SPLIT.split(line) if t]
            if len(toks) < 2:
                continue
            a, b = toks[0], toks[1]
            if a.lower() in HEADERWORDS and b.lower() in HEADERWORDS:
                continue
            out.append((a, b))
    return out


def main():
    root = find_root()
    out_dir = os.path.join(root, "grn")
    os.makedirs(out_dir, exist_ok=True)

    inputs = sorted(glob.glob(os.path.join(root, "Network-*.txt")))
    inputs += [a for a in sys.argv[1:] if os.path.exists(a)]
    inputs = list(dict.fromkeys(inputs))
    if not inputs:
        sys.exit("No network files found. Put the netcontrol Network-*.txt in %s, "
                 "or pass edge-list paths as args." % root)

    all_edges, per_file = [], {}
    for p in inputs:
        e = parse_edges(p); per_file[p] = len(e); all_edges += e

    # drop self-loops, dedup directed
    seen, directed, selfloops = set(), [], 0
    for a, b in all_edges:
        if a == b:
            selfloops += 1; continue
        if (a, b) in seen:
            continue
        seen.add((a, b)); directed.append((a, b))
    nodes_all = set(x for e in all_edges for x in e)

    # union-find on the undirected projection -> weakly-connected components
    parent = {}
    def find(x):
        parent.setdefault(x, x)
        while parent[x] != x:
            parent[x] = parent[parent[x]]; x = parent[x]
        return x
    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry: parent[rx] = ry
    for a, b in directed:
        union(a, b)
    for n in nodes_all:
        find(n)                      # self-loop-only nodes become singletons

    comps = {}
    for n in parent:
        comps.setdefault(find(n), set()).add(n)
    comp_list = sorted(comps.values(), key=len, reverse=True)
    lcc = comp_list[0] if comp_list else set()

    lcc_edges = [(a, b) for a, b in directed if a in lcc and b in lcc]
    in_edges = set(x for e in directed for x in e)
    isolated = [n for n in nodes_all if n not in in_edges]
    singletons = sum(1 for c in comp_list if len(c) == 1)

    with open(os.path.join(out_dir, "grn_all_edges_clean.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["source", "target"]); w.writerows(directed)
    with open(os.path.join(out_dir, "grn_lcc_edges.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["source", "target"]); w.writerows(sorted(lcc_edges))
    with open(os.path.join(out_dir, "grn_lcc_nodes.txt"), "w") as fh:
        fh.write("\n".join(sorted(lcc)) + "\n")

    lines = []
    def S(*m):
        s = " ".join(str(x) for x in m); print(s); lines.append(s)
    S("inputs:")
    for p in inputs:
        S("  %-45s %d edges" % (os.path.basename(p), per_file[p]))
    S("")
    S("self-loops dropped      :", selfloops)
    S("unique directed edges   :", len(directed))
    S("total nodes             :", len(nodes_all))
    S("self-loop-only nodes    :", len(isolated), "(dropped from LCC)")
    S("weakly-conn components  :", len(comp_list),
      "| multi-node sizes(top10):", [len(c) for c in comp_list if len(c) > 1][:10],
      "| singletons:", singletons)
    S("")
    S("LARGEST CONNECTED SUBCOMPONENT: %d nodes, %d directed edges" % (len(lcc), len(lcc_edges)))
    S("  -> grn_lcc_edges.csv (directed) + grn_lcc_nodes.txt")
    with open(os.path.join(out_dir, "grn_lcc_summary.txt"), "w") as fh:
        fh.write("\n".join(lines) + "\n")
    S("outputs in: %s" % out_dir)


if __name__ == "__main__":
    main()
