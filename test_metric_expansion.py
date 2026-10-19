import os
os.environ["HF_HUB_OFFLINE"] = "1"
os.environ["TRANSFORMERS_OFFLINE"] = "1"

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

print("Importing extraction_agent...", flush=True)
import extraction_agent

print("--- Testing Apple 2024 Balance Sheet Chunks ---", flush=True)
chunks = extraction_agent.retrieve_chunks("consolidated balance sheets total assets total liabilities shareholders equity",
                                          where=extraction_agent.build_filter("Apple", "2024"), top_k=3)
for c in chunks:
    print(f"Page {c['metadata'].get('page_number')} (sim: {c['similarity']}):", flush=True)
    print("  " + " ".join(c['text'].split())[:300], flush=True)

print("\n--- Testing TCS 2024 Profit & Loss Chunks ---", flush=True)
chunks = extraction_agent.retrieve_chunks("statement of profit and loss total income profit before tax",
                                          where=extraction_agent.build_filter("TCS", "2024"), top_k=3)
for c in chunks:
    print(f"Page {c['metadata'].get('page_number')} (sim: {c['similarity']}):", flush=True)
    print("  " + " ".join(c['text'].split())[:300], flush=True)
