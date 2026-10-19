"""
Document Agent - PDF Text Extraction
--------------------------------------
Extracts text page-by-page from a company's annual report PDF and saves
the result as structured JSON (page_number + text) inside documents/.

This is intentionally simple: no chunking, no embeddings, no vector DB.
Those come in later steps.
"""

import json
import os
import re
import sys
import pdfplumber

# ---- Configuration ---------------------------------------------------

# Folder that holds all annual report PDFs directly (no per-company subfolders yet).
INPUT_FOLDER = os.path.join("data", "annual_reports")

# Folder where the extracted JSON will be saved.
OUTPUT_FOLDER = "documents"

# Words that are almost never part of a company name — used to trim them
# off the front of a filename when guessing the company.
NON_COMPANY_KEYWORDS = {
    "annual", "report", "reports", "10k", "10-k", "form", "fy",
    "filing", "filings", "statement", "statements", "financial",
    "financials", "final", "draft", "v1", "v2",
}


def find_all_pdfs(folder_path):
    """
    Look inside `folder_path` and return the paths of every PDF found,
    sorted alphabetically for predictable, repeatable runs.
    """
    if not os.path.isdir(folder_path):
        raise FileNotFoundError(f"Input folder not found: {folder_path}")

    pdf_files = sorted(f for f in os.listdir(folder_path) if f.lower().endswith(".pdf"))

    if not pdf_files:
        raise FileNotFoundError(f"No PDF files found inside: {folder_path}")

    return [os.path.join(folder_path, f) for f in pdf_files]


def derive_company_name(filename):
    """
    Best-effort guess at the company name from a PDF filename, since there
    are no per-company folders anymore.

    Approach: split the filename on underscores/hyphens/spaces and keep the
    leading run of word tokens, stopping at the first number (usually a
    year) or a known non-company keyword (e.g. "annual", "report").

    If nothing usable is left after trimming, we don't guess further —
    we fall back to the original filename so we never invent a company
    name we're not reasonably confident about.
    """
    stem = os.path.splitext(filename)[0]
    tokens = re.split(r"[_\-\s]+", stem)

    company_tokens = []
    for token in tokens:
        if not token:
            continue
        if token.isdigit():
            break
        if token.lower() in NON_COMPANY_KEYWORDS:
            break
        company_tokens.append(token)

    if company_tokens:
        return " ".join(company_tokens).title()

    # Couldn't confidently derive a name - use the filename itself.
    return stem


def extract_text_by_page(pdf_path):
    """
    Extract text from every page of the PDF.
    Returns a list of dicts: [{"page_number": 1, "text": "..."}, ...]
    """
    pages_data = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"Opened PDF with {total_pages} pages: {pdf_path}")

        for i, page in enumerate(pdf.pages, start=1):
            try:
                text = page.extract_text()
            except Exception as e:
                # If extraction fails for some reason, don't crash the whole run.
                print(f"  Page {i}: extraction error ({e}), storing empty text.")
                text = None

            # extract_text() can return None for blank/image-only pages.
            if text is None:
                text = ""

            pages_data.append({
                "page_number": i,
                "text": text
            })

            print(f"  Page {i}/{total_pages} extracted ({len(text)} characters).")

    return pages_data


def save_as_json(pages_data, pdf_path, output_folder, company):
    """
    Save the extracted pages as a JSON file inside output_folder.
    The output filename is based on the source PDF's filename, which is
    already unique within the input folder, so different PDFs can't
    overwrite each other's JSON output.
    """
    os.makedirs(output_folder, exist_ok=True)

    pdf_filename = os.path.basename(pdf_path)
    base_name = os.path.splitext(pdf_filename)[0]
    output_path = os.path.join(output_folder, f"{base_name}.json")

    output_data = {
        "source_file": pdf_filename,
        "company": company,
        "total_pages": len(pages_data),
        "pages": pages_data
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    return output_path


def process_pdf(pdf_path, output_folder):
    """
    Run extraction + save for a single PDF. Used both in single-file mode
    and when looping over every PDF in the folder.
    """
    company = derive_company_name(os.path.basename(pdf_path))
    print(f"\nProcessing: {pdf_path}  (company guess: {company})")

    pages_data = extract_text_by_page(pdf_path)
    output_path = save_as_json(pages_data, pdf_path, output_folder, company)

    print(f"Extracted {len(pages_data)} pages.")
    print(f"Saved to: {output_path}")


def main():
    # Single-file mode still works, e.g.:
    #   python extract_pdf_text.py path/to/some.pdf
    if len(sys.argv) > 1:
        pdf_path = sys.argv[1]
        if not os.path.isfile(pdf_path):
            print(f"Error: file not found: {pdf_path}")
            sys.exit(1)

        process_pdf(pdf_path, OUTPUT_FOLDER)
        return

    # Default mode: process every PDF found directly inside INPUT_FOLDER.
    try:
        pdf_paths = find_all_pdfs(INPUT_FOLDER)
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print(f"Tip: place annual report PDFs inside '{INPUT_FOLDER}/'")
        print("     or run: python extract_pdf_text.py path/to/report.pdf")
        sys.exit(1)

    print(f"Found {len(pdf_paths)} PDF(s) in '{INPUT_FOLDER}':")
    for path in pdf_paths:
        print(f"  - {os.path.basename(path)}")

    for pdf_path in pdf_paths:
        process_pdf(pdf_path, OUTPUT_FOLDER)

    print(f"\nAll done. Processed {len(pdf_paths)} PDF(s).")


if __name__ == "__main__":
    main()