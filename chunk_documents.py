"""
Step A: Chunking.

Reads every JSON file in documents/ and splits each page's text into
overlapping chunks.

Important:
- Chunks are created page-by-page so page_number remains exact.
- Company and fiscal-year metadata are derived from the source filename.
- Metadata is kept clean and consistent for ChromaDB retrieval.
"""

import hashlib
import json
import re

from config import (
    CHUNK_OVERLAP_WORDS,
    CHUNK_WORDS,
    CHUNKS_DIR,
    DEFAULT_DOCUMENT_TYPE,
    DOCUMENTS_DIR,
    MIN_CHUNK_WORDS,
    ensure_directories,
)


# ---------------------------------------------------------------------------
# Metadata helpers
# ---------------------------------------------------------------------------

def detect_company_and_year(filename):
    """
    Detect company and fiscal year from known annual-report filename patterns.

    This is deliberately explicit because filenames such as:
        tsla-20251231.json
        RIL_IAR_2024.json
        Amazon.com, Inc 2025.json

    should not depend on guessing from report text.
    """

    name = filename.lower()

    # Tesla
    if "tsla-" in name or "tesla" in name:
        match = re.search(r"20\d{2}", name)
        return "Tesla", match.group(0) if match else "unknown"

    # Reliance Industries
    if "ril" in name or "reliance" in name:
        match = re.search(r"20\d{2}", name)
        return "Reliance Industries", match.group(0) if match else "unknown"

    # Tata Consultancy Services
    if "tcs" in name:
        match = re.search(r"20\d{2}", name)
        if match:
            return "TCS", match.group(0)
        return "TCS", "unknown"

    # Infosys
    if "infosys" in name:
        match = re.search(r"20\d{2}", name)
        return "Infosys", match.group(0) if match else "unknown"

    # Microsoft
    if "microsoft" in name or "msft" in name:
        match = re.search(r"20\d{2}", name)
        return "Microsoft", match.group(0) if match else "unknown"

    # Amazon
    if "amazon" in name:
        match = re.search(r"20\d{2}", name)
        return "Amazon", match.group(0) if match else "unknown"

    # Apple
    if "apple" in name:
        match = re.search(r"20\d{2}", name)
        return "Apple", match.group(0) if match else "unknown"

    # Generic fallback
    match = re.search(r"20\d{2}", name)

    company = re.sub(r"\.json$", "", filename, flags=re.IGNORECASE)
    company = re.sub(r"\.pdf$", "", company, flags=re.IGNORECASE)
    company = re.sub(r"20\d{2}", "", company)
    company = re.sub(r"\d{8}", "", company)
    company = company.replace("_", " ").replace("-", " ")
    company = re.sub(r"\s+", " ", company).strip()

    return company.title() if company else "Unknown", match.group(0) if match else "unknown"


def make_chunk_id(source_file, page_number, chunk_index):
    """
    Build a stable unique ID for each chunk.

    Running the script again on the same source document produces
    the same ID, preventing duplicate records during ChromaDB indexing.
    """

    raw = f"{source_file}|p{page_number}|c{chunk_index}"
    digest = hashlib.md5(raw.encode("utf-8")).hexdigest()[:12]

    return f"{digest}_p{page_number}_c{chunk_index}"


# ---------------------------------------------------------------------------
# Text processing
# ---------------------------------------------------------------------------

def normalise_whitespace(text):
    """Clean unnecessary spaces and blank lines."""

    text = text.replace("\xa0", " ")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()


def split_into_chunks(
    text,
    chunk_words=CHUNK_WORDS,
    overlap_words=CHUNK_OVERLAP_WORDS,
):
    """
    Split text into overlapping word windows.

    Example:
        chunk_words = 600
        overlap_words = 100

        Chunk 1 -> words 0-600
        Chunk 2 -> words 500-1100
        Chunk 3 -> words 1000-1600
    """

    words = text.split()

    if not words:
        return []

    step = max(1, chunk_words - overlap_words)

    chunks = []
    start = 0

    while start < len(words):

        window = words[start:start + chunk_words]

        chunks.append(" ".join(window))

        if start + chunk_words >= len(words):
            break

        start += step

    return chunks


# ---------------------------------------------------------------------------
# Document chunking
# ---------------------------------------------------------------------------

def chunk_document(doc, json_filename):
    """Convert one extracted JSON document into chunk dictionaries."""

    source_file = doc.get("source_file") or json_filename

    # Use the actual source PDF filename when available.
    company, fiscal_year = detect_company_and_year(source_file)

    # If source_file didn't contain useful metadata, also try JSON filename.
    if company == "Unknown" or fiscal_year == "unknown":
        fallback_company, fallback_year = detect_company_and_year(json_filename)

        if company == "Unknown":
            company = fallback_company

        if fiscal_year == "unknown":
            fiscal_year = fallback_year

    chunks = []

    for page in doc.get("pages", []):

        page_number = page.get("page_number")
        raw_text = page.get("text") or ""

        text = normalise_whitespace(raw_text)

        if not text:
            continue

        page_chunks = split_into_chunks(text)

        for chunk_index, chunk_text in enumerate(page_chunks):

            word_count = len(chunk_text.split())

            if word_count < MIN_CHUNK_WORDS:
                continue

            chunks.append(
                {
                    "chunk_id": make_chunk_id(
                        source_file,
                        page_number,
                        chunk_index,
                    ),

                    "text": chunk_text,

                    "metadata": {
                        "company": company,
                        "fiscal_year": fiscal_year,
                        "document_type": DEFAULT_DOCUMENT_TYPE,
                        "source_file": source_file,
                        "page_number": (
                            int(page_number)
                            if page_number is not None
                            else -1
                        ),
                        "chunk_index": chunk_index,
                        "word_count": word_count,
                    },
                }
            )

    return chunks


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():

    ensure_directories()

    json_files = sorted(DOCUMENTS_DIR.glob("*.json"))

    if not json_files:

        print(
            f"No JSON files found in {DOCUMENTS_DIR}. "
            "Run extract_pdf_text.py first."
        )

        return

    grand_total = 0

    for path in json_files:

        with open(path, "r", encoding="utf-8") as f:
            doc = json.load(f)

        if "pages" not in doc:

            print(
                f"[SKIPPED] {path.name}: no 'pages' key."
            )

            continue

        chunks = chunk_document(
            doc,
            path.stem,
        )

        grand_total += len(chunks)

        out_path = CHUNKS_DIR / f"{path.stem}_chunks.json"

        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(
                chunks,
                f,
                ensure_ascii=False,
                indent=2,
            )

        if chunks:

            meta = chunks[0]["metadata"]

            print(
                f"{path.name:<50} -> "
                f"{len(chunks):>5} chunks  "
                f"[{meta['company']} {meta['fiscal_year']}]"
            )

        else:

            print(
                f"{path.name:<50} -> "
                f"{0:>5} chunks  "
                "[No usable chunks]"
            )

    print()
    print(
        f"Done. {grand_total} chunks written to {CHUNKS_DIR}"
    )


if __name__ == "__main__":
    main()