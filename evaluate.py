"""Score the search engine against hand-made test queries.

eval/test_queries.csv has one row per query:
    query,relevant_ids,notes
    birthday cake,7;98,photo numbers from the app separated by ;

    python evaluate.py

Prints Recall@5, Recall@10 and MRR for each search method, so we can show that the
hybrid search beats each part on its own.
"""
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from search import SearchEngine

QUERIES = Path(__file__).parent / "eval" / "test_queries.csv"
METHODS = {
    "Keyword only (BM25)": ("keyword",),
    "Caption text only (MiniLM)": ("caption",),
    "Image only (CLIP)": ("image",),
    "Hybrid (all three, RRF)": ("image", "caption", "keyword"),
}


def load_queries():
    rows = []
    with open(QUERIES, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            # Photo numbers as shown in the app start at 1, the code counts from 0
            ids = {int(x) - 1 for x in r["relevant_ids"].replace(",", ";").split(";") if x.strip()}
            if r["query"].strip() and ids:
                rows.append((r["query"].strip(), ids))
    return rows


def score(engine, queries, use):
    r5 = r10 = mrr = 0.0
    for query, relevant in queries:
        hits, _ = engine.search(query, top=10, use=use, only_matches=False)
        ranked = [r["id"] for r, _, _ in hits]
        r5 += len(relevant & set(ranked[:5])) / len(relevant)
        r10 += len(relevant & set(ranked[:10])) / len(relevant)
        first = next((i for i, pid in enumerate(ranked) if pid in relevant), None)
        mrr += 1 / (first + 1) if first is not None else 0
    n = len(queries)
    return r5 / n, r10 / n, mrr / n


def main():
    queries = load_queries()
    if not queries:
        raise SystemExit(f"No queries with relevant_ids in {QUERIES}")
    engine = SearchEngine()
    print(f"{len(queries)} test queries\n")
    print(f"{'Method':30} {'Recall@5':>9} {'Recall@10':>10} {'MRR':>6}")
    results = {}
    for name, use in METHODS.items():
        r5, r10, mrr = score(engine, queries, use)
        results[name] = (r5, r10, mrr)
        print(f"{name:30} {r5:9.2f} {r10:10.2f} {mrr:6.2f}")
    chart(results, len(queries))


def chart(results, n):
    """Grouped bars: each search method on Recall@5, Recall@10 and MRR. Opens in VS Code."""
    names = list(results)
    x = np.arange(3)
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for k, name in enumerate(names):
        ax.bar(x + k * 0.2 - 0.3, results[name], 0.2, label=name)
    ax.set_xticks(x, ["Recall@5", "Recall@10", "MRR"])
    ax.set_ylim(0, 1)
    ax.set_title(f"Search quality on {n} test queries (higher is better)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    out = Path(__file__).parent / "outputs" / "search_quality.png"
    fig.savefig(out, dpi=120)
    print(f"\nChart saved: {out} (open it in VS Code)")


if __name__ == "__main__":
    main()
