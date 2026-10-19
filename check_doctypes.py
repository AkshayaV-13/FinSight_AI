import sys
from pathlib import Path
from collections import Counter
import chromadb

PROJECT_ROOT = Path(__file__).resolve().parent
client = chromadb.PersistentClient(path=str(PROJECT_ROOT / "vectordb"))
col = client.get_collection("financial_documents")

data = col.get(include=["metadatas"])
doc_types = Counter(m.get("document_type") for m in data["metadatas"])
print("document_type values:")
for dt, cnt in doc_types.items():
    print(f"  {dt}: {cnt}")

files = Counter(m.get("source_file") for m in data["metadatas"])
print("\nsource_file values:")
for f, cnt in sorted(files.items()):
    print(f"  {f}: {cnt}")
