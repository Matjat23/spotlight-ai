"""Score the search engine against hand-made test queries.

eval/test_queries.csv has one row per query:
    query,relevant_ids,notes
    birthday cake,6;97,ids of the right photos separated by ;

    python evaluate.py

Prints Recall@5, Recall@10 and MRR for each search method, so we can show that the
hybrid search beats each part on its own.
"""
import csv
from pathlib import Path

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
            ids = {int(x) for x in r["relevant_ids"].replace(",", ";").split(";") if x.strip()}
            if r["query"].strip() and ids:
                rows.append((r["query"].strip(), ids))
    return rows


def score(engine, queries, use):
    r5 = r10 = mrr = 0.0
    for query, relevant in queries:
        hits, _ = engine.search(query, top=10, use=use)
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
    for name, use in METHODS.items():
        r5, r10, mrr = score(engine, queries, use)
        print(f"{name:30} {r5:9.2f} {r10:10.2f} {mrr:6.2f}")


if __name__ == "__main__":
    main()
