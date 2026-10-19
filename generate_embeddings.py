"""
generate_embeddings.py

Embedding generation stage of the Document Agent.

What this script does
----------------------
1. Reads every *_chunks.json file in chunks/
2. Generates a 384-dimensional embedding for each chunk's text using the
   free local model sentence-transformers/all-MiniLM-L6-v2
3. Writes one *_embeddings.json file per chunk file into embeddings/,
   preserving chunk_id, text, and all existing metadata

What this script deliberately does NOT do
------------------------------------------
- It does not create or touch a ChromaDB collection. That is the next step.
- It does not modify anything in documents/ or chunks/.
- It does not call any paid API. Everything runs locally.

Duplicate protection
---------------------
Each output file already lists the chunk_ids it has embedded. On a second
run, chunk_ids already present in the matching output file are skipped, so
re-running this script after an interruption does not re-embed everything
from scratch and does not create duplicate entries.

Usage (PowerShell):
    python generate_embeddings.py
"""

import json
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
# Resolved relative to this file's own location, so the script works
# regardless of which folder you run it from.
PROJECT_ROOT = Path(__file__).resolve().parent
CHUNKS_DIR = PROJECT_ROOT / "chunks"
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
BATCH_SIZE = 64  # how many chunks are embedded at once; fine for a laptop CPU


def load_chunk_files():
    """Return a sorted list of every *_chunks.json file in chunks/."""
    files = sorted(CHUNKS_DIR.glob("*_chunks.json"))
    if not files:
        print(f"No chunk files found in {CHUNKS_DIR}")
        print("Run the chunking script first.")
        sys.exit(1)
    return files


def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path, data):
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def load_existing_output(output_path):
    """
    Load an embeddings file already produced by a previous run, if any.

    Returns (list_of_existing_records, set_of_already_embedded_chunk_ids).
    If the file does not exist yet, returns an empty list and empty set.
    """
    if not output_path.exists():
        return [], set()

    try:
        existing_records = load_json(output_path)
        existing_ids = {record["chunk_id"] for record in existing_records}
        return existing_records, existing_ids
    except (json.JSONDecodeError, KeyError) as exc:
        print(f"   [WARNING] Could not read existing {output_path.name} ({exc}). "
              f"Treating as empty and rewriting it.")
        return [], set()


def get_model():
    """Load the embedding model once. Fails with a clear message if missing."""
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        print("sentence-transformers is not installed.")
        print("Install it first with:")
        print("    pip install sentence-transformers")
        sys.exit(1)

    print(f"Loading embedding model: {MODEL_NAME}")
    print("(First run downloads the model, roughly 90 MB. Later runs are offline.)")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded.\n")
    return model


def process_chunk_file(model, chunk_path):
    """
    Embed every new chunk in one chunk file and write/update its
    matching embeddings/*_embeddings.json output file.

    Returns (embedded_count, skipped_count, error_count) for this file.
    """
    output_path = EMBEDDINGS_DIR / chunk_path.name.replace("_chunks.json", "_embeddings.json")

    try:
        chunks = load_json(chunk_path)
    except (json.JSONDecodeError, OSError) as exc:
        print(f"[ERROR] Could not read {chunk_path.name}: {exc}")
        return 0, 0, len(chunks) if isinstance(locals().get("chunks"), list) else 0

    if not chunks:
        print(f"{chunk_path.name}: 0 chunks, skipped")
        return 0, 0, 0

    existing_records, existing_ids = load_existing_output(output_path)

    # Work out which chunks still need embedding.
    pending = []
    skipped = 0
    error_count = 0

    for chunk in chunks:
        chunk_id = chunk.get("chunk_id")
        text = chunk.get("text")

        if not chunk_id or not text or not text.strip():
            error_count += 1
            continue  # malformed chunk entry; skip it rather than crash the run

        if chunk_id in existing_ids:
            skipped += 1
            continue

        pending.append(chunk)

    if error_count:
        print(f"   [WARNING] {error_count} chunk(s) in {chunk_path.name} "
              f"had a missing chunk_id or empty text and were skipped.")

    if not pending:
        print(f"{chunk_path.name:<55} -> already embedded ({skipped} chunks, 0 new)")
        return 0, skipped, error_count

    print(f"{chunk_path.name:<55} -> embedding {len(pending)} new chunk(s) "
          f"({skipped} already done)")

    texts = [chunk["text"] for chunk in pending]

    try:
        vectors = model.encode(
            texts,
            batch_size=BATCH_SIZE,
            show_progress_bar=True,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
    except Exception as exc:
        print(f"[ERROR] Embedding generation failed for {chunk_path.name}: {exc}")
        return 0, skipped, len(pending)

    new_records = []
    for chunk, vector in zip(pending, vectors):
        new_records.append({
            "chunk_id": chunk["chunk_id"],
            "text": chunk["text"],
            "metadata": chunk.get("metadata", {}),
            "embedding": vector.tolist(),
        })

    all_records = existing_records + new_records
    save_json(output_path, all_records)

    return len(new_records), skipped, error_count


def main():
    start_time = time.time()

    EMBEDDINGS_DIR.mkdir(parents=True, exist_ok=True)

    chunk_files = load_chunk_files()
    print(f"Found {len(chunk_files)} chunk file(s) in {CHUNKS_DIR}\n")

    model = get_model()

    total_embedded = 0
    total_skipped = 0
    total_errors = 0

    for i, chunk_path in enumerate(chunk_files, start=1):
        print(f"[{i}/{len(chunk_files)}] ", end="")
        embedded, skipped, errors = process_chunk_file(model, chunk_path)
        total_embedded += embedded
        total_skipped += skipped
        total_errors += errors

    elapsed = time.time() - start_time

    print("\n" + "=" * 60)
    print("EMBEDDING GENERATION COMPLETE")
    print("=" * 60)
    print(f"Newly embedded : {total_embedded}")
    print(f"Already done   : {total_skipped} (skipped)")
    print(f"Skipped/errors : {total_errors} (missing chunk_id or empty text)")
    print(f"Total chunks   : {total_embedded + total_skipped + total_errors}")
    print(f"Time taken     : {elapsed:.1f} seconds")
    print(f"Output folder  : {EMBEDDINGS_DIR}")


if __name__ == "__main__":
    main()