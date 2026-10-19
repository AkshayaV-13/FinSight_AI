"""
test_retrieval.py

Retrieval testing stage of the Document Agent.

What this script does
----------------------
1. Loads the same local embedding model used for indexing
   (sentence-transformers/all-MiniLM-L6-v2)
2. Connects to the existing persistent ChromaDB database at vectordb/
3. Embeds a query, searches the "financial_documents" collection, and
   prints the most relevant chunks with their source metadata

This is a read-only test: it does not add, modify, or delete anything in
the collection.

Usage (PowerShell):
    python test_retrieval.py
        Runs a set of built-in sample queries.

    python test_retrieval.py "Tesla gross margin"
        Runs your own query instead.

    python test_retrieval.py "net income" --company Apple --top-k 3
        Restricts results to one company and controls how many results
        come back.
"""

import argparse
import sys
import textwrap
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
VECTORDB_DIR = PROJECT_ROOT / "vectordb"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "financial_documents"

SAMPLE_QUERIES = [
    "Amazon revenue 2025",
    "Apple net income",
    "Microsoft total debt",
    "Infosys operating margin",
    "Tesla cash flow from operating activities",
    "Reliance revenue trends",
    "TCS earnings per share",
]


def get_model():
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed.")
        print("Install it first with:")
        print("    pip install sentence-transformers")
        sys.exit(1)

    print(f"Loading embedding model: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.\n")
    return model


def get_collection():
    try:
        import chromadb
    except ImportError:
        print("chromadb is not installed.")
        print("Install it first with:")
        print("    pip install chromadb")
        sys.exit(1)

    if not VECTORDB_DIR.exists():
        print(f"No database found at {VECTORDB_DIR}")
        print("Run index_chromadb.py first.")
        sys.exit(1)

    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))

    try:
        return client.get_collection(name=COLLECTION_NAME)
    except Exception:
        print(f"Collection '{COLLECTION_NAME}' does not exist yet.")
        print("Run index_chromadb.py first.")
        sys.exit(1)


def build_filter(company=None, year=None):
    """Build an optional ChromaDB metadata filter from CLI arguments."""
    conditions = []
    if company:
        conditions.append({"company": company})
    if year:
        conditions.append({"fiscal_year": str(year)})

    if len(conditions) == 1:
        return conditions[0]
    if len(conditions) > 1:
        return {"$and": conditions}
    return None


def run_query(model, collection, query_text, top_k, where):
    """Embed the query and search the collection. Returns ChromaDB's raw result."""
    query_vector = model.encode(
        [query_text],
        normalize_embeddings=True,
        convert_to_numpy=True,
    )[0].tolist()

    return collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )


def show_results(query_text, result):
    print("=" * 78)
    print(f"QUERY: {query_text}")
    print("=" * 78)

    documents = result["documents"][0]
    metadatas = result["metadatas"][0]
    distances = result["distances"][0]

    if not documents:
        print("No results found.\n")
        return

    for rank, (text, metadata, distance) in enumerate(
        zip(documents, metadatas, distances), start=1
    ):
        # ChromaDB's cosine distance ranges 0 (identical) to 2 (opposite),
        # so similarity = 1 - distance is a rough closeness score, higher
        # is better, generally within roughly 0 to 1 for relevant results.
        similarity = round(1 - distance, 4)

        print(f"\n[{rank}] similarity {similarity}  (distance {round(distance, 4)})")
        print(f"    company     : {metadata.get('company')}")
        print(f"    fiscal_year : {metadata.get('fiscal_year')}")
        print(f"    source_file : {metadata.get('source_file')}")
        print(f"    page_number : {metadata.get('page_number')}")

        preview = " ".join(text.split())[:400]
        wrapped = textwrap.fill(preview, width=72, subsequent_indent=" " * 18)
        print(f"    text        : {wrapped}")
    print()


def main():
    parser = argparse.ArgumentParser(description="Semantic retrieval test for ChromaDB")
    parser.add_argument("query", nargs="?", help="a question to search for")
    parser.add_argument("--company", help="restrict results to one company, e.g. Apple")
    parser.add_argument("--year", help="restrict results to one fiscal year, e.g. 2025")
    parser.add_argument("--top-k", type=int, default=3, help="number of results per query")
    args = parser.parse_args()

    collection = get_collection()
    count = collection.count()
    print(f"Collection '{COLLECTION_NAME}' contains {count} chunk(s).\n")

    if count == 0:
        print("The collection is empty. Run index_chromadb.py first.")
        return

    model = get_model()
    where = build_filter(args.company, args.year)

    queries = [args.query] if args.query else SAMPLE_QUERIES

    for query_text in queries:
        result = run_query(model, collection, query_text, args.top_k, where)
        show_results(query_text, result)


if __name__ == "__main__":
    main()