import sys
from pathlib import Path
from collections import Counter

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import chromadb

client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "vectordb"))
col = client.get_collection("financial_documents")
count = col.count()
print(f"Total documents in ChromaDB: {count}")

data = col.get(include=["metadatas"])
metas = data["metadatas"]
companies = Counter(m.get("company") for m in metas)
print("Companies count:")
for c, cnt in companies.most_common():
    print(f"  '{c}': {cnt}")

pairs = Counter((m.get("company"), str(m.get("fiscal_year"))) for m in metas)
print("\nCompany-Year pairs:")
for (c, y), cnt in sorted(pairs.items()):
    print(f"  ({c}, {y}): {cnt}")

print("\nSample metadata keys:", list(metas[0].keys()) if metas else [])
