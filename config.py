"""
config.py

Central settings for the project.

Every script imports its paths and numbers from here, so there is only one
place to change a setting and no absolute paths are hard-coded anywhere else.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Project folders
# ---------------------------------------------------------------------------
# PROJECT_ROOT is the folder that contains this file.
# Keeping config.py in the project root means every path below resolves
# correctly no matter which directory you run the scripts from.
PROJECT_ROOT = Path(__file__).resolve().parent

DATA_DIR = PROJECT_ROOT / "data" / "annual_reports"   # source PDFs
DOCUMENTS_DIR = PROJECT_ROOT / "documents"            # JSON from extract_pdf_text.py
CHUNKS_DIR = PROJECT_ROOT / "chunks"                  # chunked text (next step)
EMBEDDINGS_DIR = PROJECT_ROOT / "embeddings"          # reserved for later
VECTORDB_DIR = PROJECT_ROOT / "vectordb"              # reserved for later

# ---------------------------------------------------------------------------
# Chunking settings
# ---------------------------------------------------------------------------
# We measure chunks in WORDS because a word count is easy to check by hand.
# For English financial text, 1 word is roughly 1.3 tokens:
#   600 words -> about 780 tokens  (inside the 500-800 token target)
#   100 words -> about 130 tokens of overlap
CHUNK_WORDS = 600
CHUNK_OVERLAP_WORDS = 100

# Anything shorter than this is usually a page header, footer or stray line,
# so it is skipped rather than indexed as a useless chunk.
MIN_CHUNK_WORDS = 20

# ---------------------------------------------------------------------------
# Document metadata
# ---------------------------------------------------------------------------
# Every document collected so far is an annual report. When other document
# types are added later, this becomes the fallback rather than the only value.
DEFAULT_DOCUMENT_TYPE = "annual_report"


# ---------------------------------------------------------------------------
# Helper
# ---------------------------------------------------------------------------
def ensure_directories():
    """
    Create the working folders if they do not exist yet.

    mkdir(exist_ok=True) does nothing when a folder is already there, so this
    is safe to call at the start of any script and never deletes anything.
    """
    for folder in (DOCUMENTS_DIR, CHUNKS_DIR, EMBEDDINGS_DIR, VECTORDB_DIR):
        folder.mkdir(parents=True, exist_ok=True)


# Running "python config.py" directly prints the resolved paths, which is a
# quick way to confirm the folders are where you expect.
if __name__ == "__main__":
    print(f"PROJECT_ROOT  : {PROJECT_ROOT}")
    print(f"DOCUMENTS_DIR : {DOCUMENTS_DIR}  (exists: {DOCUMENTS_DIR.exists()})")
    print(f"CHUNKS_DIR    : {CHUNKS_DIR}  (exists: {CHUNKS_DIR.exists()})")
    print(f"EMBEDDINGS_DIR: {EMBEDDINGS_DIR}  (exists: {EMBEDDINGS_DIR.exists()})")
    print(f"VECTORDB_DIR  : {VECTORDB_DIR}  (exists: {VECTORDB_DIR.exists()})")