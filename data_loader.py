"""
data_loader.py

Automated dataset and ChromaDB synchronization from Hugging Face for FinSight AI.
Ensures that the 4 essential directories (vectordb, documents, chunks, embeddings)
are verified and downloaded from Hugging Face dataset 'Akshaya-V/finsight_ai_dataset'
on Streamlit Community Cloud or fresh environments without rebuilding embeddings.
"""

import logging
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config import CHUNKS_DIR, DOCUMENTS_DIR, EMBEDDINGS_DIR, VECTORDB_DIR

LOGGER = logging.getLogger("data_loader")
HF_DATASET_REPO = "Akshaya-V/finsight_ai_dataset"


def is_vectordb_ready():
    """Check if the persistent ChromaDB SQLite database and collection folder exist and are valid."""
    sqlite_file = VECTORDB_DIR / "chroma.sqlite3"
    if not sqlite_file.exists():
        return False
    try:
        # chroma.sqlite3 should be > 50MB
        if sqlite_file.stat().st_size < 10_000_000:
            return False
    except OSError:
        return False

    # Check for collection UUID folder
    subdirs = [p for p in VECTORDB_DIR.iterdir() if p.is_dir() and p.name != "__pycache__"]
    return len(subdirs) >= 1


def is_dataset_ready():
    """Check if all four directories exist and contain files."""
    if not is_vectordb_ready():
        return False

    for folder in (CHUNKS_DIR, DOCUMENTS_DIR, EMBEDDINGS_DIR):
        if not folder.exists():
            return False
        json_files = list(folder.glob("*.json"))
        if not json_files:
            return False

    return True


def ensure_data_loaded(force=False, progress_callback=None):
    """
    Download missing directories from Hugging Face dataset 'Akshaya-V/finsight_ai_dataset'.
    Does nothing if files are already present and valid (0s startup overhead).
    """
    if not force and is_dataset_ready():
        LOGGER.info("All datasets and ChromaDB vector store are already present and verified.")
        return True

    LOGGER.info("Ensuring datasets and ChromaDB vector store from %s...", HF_DATASET_REPO)
    if progress_callback:
        progress_callback("Connecting to Hugging Face dataset repository...")

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        LOGGER.error("huggingface_hub is not installed. Run: pip install huggingface_hub")
        raise RuntimeError("huggingface_hub is required to download FinSight AI datasets.")

    start_time = time.time()
    try:
        if progress_callback:
            progress_callback("Downloading ChromaDB vector database and document files (this runs once)...")

        # Snapshot download automatically handles Git LFS (e.g. chroma.sqlite3 ~102MB)
        download_path = snapshot_download(
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            allow_patterns=["vectordb/**", "chunks/**", "documents/**", "embeddings/**"],
            local_dir=str(PROJECT_ROOT),
            max_workers=4,
        )

        elapsed = time.time() - start_time
        LOGGER.info("Successfully synchronized dataset from HF in %.2f seconds.", elapsed)
        if progress_callback:
            progress_callback(f"Download complete in {elapsed:.1f}s. Initializing ChromaDB...")

        # Verify that vectordb is now ready
        if not is_vectordb_ready():
            raise FileNotFoundError(
                f"Download completed but {VECTORDB_DIR / 'chroma.sqlite3'} is missing or incomplete."
            )

        return True

    except Exception as exc:
        LOGGER.error("Failed to download dataset from Hugging Face: %s", exc, exc_info=True)
        raise RuntimeError(
            f"Failed to synchronize FinSight AI dataset from Hugging Face ({HF_DATASET_REPO}): {exc}"
        ) from exc


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    print(f"Checking data status for FinSight AI (Root: {PROJECT_ROOT})...")
    print(f"Vectordb ready: {is_vectordb_ready()}")
    print(f"Full dataset ready: {is_dataset_ready()}")
    if not is_dataset_ready():
        print("Triggering download from Hugging Face...")
        ensure_data_loaded()
        print("Data loaded successfully.")
    else:
        print("Data is already verified. No download needed.")
