"""
extraction_agent.py

Agent 1: Extraction Agent.

v6 -- financial-context layer (replaces "first number after a phrase")
-----------------------------------------------------------------------
Root causes found in v5 (each verified by reading the code):

  A. Revenue "383300000000.0 billion USD". A scale word attached to the
     number ("$383.3 billion") was applied TWICE: the value was multiplied
     out to absolute dollars (383.3 * 1e9) and the unit was ALSO labeled
     "billion". Scales declared in a table header ("(in millions)") were
     labeled but never multiplied. Two incompatible conventions shared one
     value/unit pair, so Agent 2 correctly saw two different units.

  B. Net income "96995 million" with no currency. Currency was read ONLY from
     a "$" attached directly to the matched number. Only the SCALE was ever
     taken from the table/statement header, never the currency. A cell whose
     "$" was separated by table extraction therefore lost its currency.

  C. FY2023 EPS NOT_FOUND. v5 required the full phrase "diluted earnings per
     share" in the text and used a single query variant. Statement layouts
     that label rows just "Basic" / "Diluted" under an "Earnings per share:"
     heading could never match, and one query meant one chance to retrieve
     the page. (Whether this or a retrieval miss hit your real FY2023 chunk is
     answered by `--diagnose`, which prints the retrieved pages and every
     rejected candidate.)

  D. FY2023 Operating Margin NOT_FOUND. build_operating_margin refused to
     calculate when the two unit STRINGS differed ("billion USD" vs
     "million USD"). That guard fired because of bug A. A separate defect also
     showed up: FY2024 returned 17.3% because a regional-segment operating
     income was chosen instead of the consolidated figure.

What changed
------------
1. Every monetary value is normalized to ONE canonical unit, "million <CCY>".
   Each record keeps raw_value, reported_scale and scale_source (source
   fidelity) next to value / normalized_value / unit, and value and unit can
   never disagree. 383.3 billion -> 383300.0 "million USD"; a table cell
   391,035 under "(in millions)" -> 391035.0 "million USD".
2. Currency and scale come from evidence, in priority order: the number's own
   symbol/word, then the same row, then the statement/table header in the
   same chunk, then a cached statement probe of the same company/year. The
   evidence and its source are reported (currency_source / scale_source).
   Nothing is assumed from a company name, and USD is never the default.
3. Financial tables are read as rows of cells under year columns. The agent
   finds the nearest year header ("2023 2022 2021", or "September 30, 2023
   September 24, 2022 ..."), reads the contiguous cells after the row label,
   and picks the cell under the REQUESTED year. A header that does not
   contain the requested year rejects the row. With no header, multi-cell
   rows fall back to the first cell but are marked confidence "low".
4. EPS recognises "Diluted"/"Basic" row labels under an "earnings per share"
   heading, rejects rows that follow "shares used"/"weighted-average", needs
   a decimal point, and ranks diluted above basic.
5. Operating income rows labeled with a geographic/segment name are rejected
   (generic region words, not a company rule).
6. Retrieval uses several query variants per metric, a cached statement
   probe, and a second pass with different queries only when the first pass
   yields no candidate.
7. Operating margin is derived from normalized, unit-matched values and keeps
   source/page information for BOTH inputs.

Unchanged: sentence-transformers/all-MiniLM-L6-v2, ChromaDB (read-only
queries only), company/fiscal-year filtering, no LLM or paid API, no
invented values (no acceptable candidate -> not_found), and every existing
output field name.

Usage:
    python extraction_agent.py
    python extraction_agent.py --diagnose Apple 2023 eps
"""

import logging
import os
import re
import sys
import textwrap
from collections import Counter
from pathlib import Path

LOGGER = logging.getLogger("extraction_agent")

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent
VECTORDB_DIR = PROJECT_ROOT / "vectordb"

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "financial_documents"

DEFAULT_TOP_K = 3
NOT_FOUND = "not_found"
FOUND = "found"

CONFIDENCE_LEVELS = ["low", "medium", "high"]


# ---------------------------------------------------------------------------
# Metric definitions
# ---------------------------------------------------------------------------
# kind:                 currency | percent | eps
# query_variants:       several phrasings retrieved and merged
# fallback_query_variants: only used when the first pass finds no candidate
# phrase_tiers:         most specific label first (feeds the score)
# exclude_keywords:     rejected if found just before the label or between
#                       the label and the number
# exclude_between:      rejected only if found between label and number
# exclude_geo_label / exclude_segment_word: reject rows labeled by a
#                       region or segment (generic words, not company rules)
# penalize_repeated_label: a label repeated 3+ times in one chunk looks like
#                       a segment table, not a consolidated statement
# statement_markers:    headings that mark a real financial statement
INCOME_HEADINGS = [
    "consolidated statements of operations", "consolidated statement of operations",
    "consolidated statements of income", "consolidated income statement",
    "statements of operations", "statement of operations", "income statement",
    "statement of profit and loss", "statements of profit and loss",
]

METRIC_DEFINITIONS = {
    "revenue": {
        "label": "Revenue",
        "kind": "currency",
        "query_variants": [
            "total revenue net sales for the fiscal year",
            "consolidated statements of operations total net sales cost of sales",
        ],
        "fallback_query_variants": [
            "net sales by category products services total net sales",
            "statement of profit and loss revenue from operations total income",
        ],
        "phrase_tiers": [
            ["total net sales", "total revenues", "total revenue", "total net revenue",
             "revenue from operations"],
            ["net revenues", "net revenue", "net sales"],
            ["revenue"],
        ],
        "exclude_keywords": [
            "deferred", "unearned", "contract liabilit", "recogniz",
            "disaggregat", "by segment", "by product", "by category",
            "by region", "geographic", "source", "performance obligation",
            "subscription revenue", "product revenue", "service revenue",
            "segment revenue", "future revenue", "to be recognized",
        ],
        "exclude_between": ["per share", "per-share"],
        "exclude_geo_label": True,
        "statement_markers": INCOME_HEADINGS,
    },
    "net_income": {
        "label": "Net Income",
        "kind": "currency",
        "query_variants": [
            "net income profit for the year attributable to shareholders",
            "consolidated statements of operations net income earnings per share",
        ],
        "fallback_query_variants": [
            "net income before provision for income taxes provision net income",
            "statement of profit and loss profit for the year after tax",
        ],
        "phrase_tiers": [
            ["net income attributable", "net income"],
            ["net profit", "profit for the year", "profit attributable to",
             "net earnings", "profit after tax"],
        ],
        "exclude_keywords": [
            "noncontrolling interest", "non-controlling interest",
            "discontinued operations", "comprehensive income",
        ],
        "exclude_between": ["per share", "per-share"],
        "statement_markers": INCOME_HEADINGS,
    },
    "operating_income": {
        # Not one of the 7 target metrics itself; extracted the same way so
        # Operating Margin can be derived when no percentage is stated.
        "label": "Operating Income",
        "kind": "currency",
        "query_variants": [
            "operating income income from operations",
            "consolidated statements of operations total operating expenses operating income",
        ],
        "fallback_query_variants": [
            "operating income other income expense net income before provision for income taxes",
            "statement of profit and loss profit from operations",
        ],
        "phrase_tiers": [
            ["operating income", "income from operations"],
            ["operating profit", "profit from operations"],
        ],
        "exclude_keywords": ["total segment"],
        "exclude_between": ["per share", "per-share"],
        "exclude_geo_label": True,
        "exclude_segment_word": True,
        "penalize_repeated_label": True,
        "statement_markers": INCOME_HEADINGS,
    },
    "gross_margin": {
        "label": "Gross Margin",
        "kind": "percent",
        "query_variants": [
            "gross margin gross profit percentage",
            "total gross margin percentage products services",
        ],
        "fallback_query_variants": ["gross margin percentage change net sales cost of sales"],
        "phrase_tiers": [
            ["total gross margin percentage"],
            ["gross margin percentage", "gross profit margin", "gross margin"],
        ],
        "exclude_keywords": [],
        "exclude_label_words": ["products", "services", "segment"],
        "statement_markers": ["gross margin"],
    },
    "operating_margin": {
        "label": "Operating Margin",
        "kind": "percent",
        "query_variants": ["operating margin operating income percentage of revenue"],
        "fallback_query_variants": ["operating margin percentage"],
        "phrase_tiers": [["operating margin percentage", "operating margin"]],
        "exclude_keywords": [],
        "statement_markers": ["operating margin"],
    },
    "total_debt": {
        "label": "Total Debt",
        "kind": "currency",
        "query_variants": [
            "total debt long-term borrowings total borrowings",
            "notes to financial statements long-term debt borrowings note balance sheet",
        ],
        "fallback_query_variants": ["term debt commercial paper total term debt maturities"],
        "phrase_tiers": [
            ["total debt", "total borrowings", "total term debt"],
            ["total long-term debt", "long-term debt", "non-current borrowings", "term debt"],
        ],
        # "commercial paper" is deliberately NOT excluded: it is a real
        # component of debt for many companies.
        "exclude_keywords": [
            "deferred tax", "lease liabilit", "operating lease", "finance lease",
            "right-of-use", "pension", "warranty", "income tax payable",
            "contract liabilit", "total liabilit", "accounts payable",
        ],
        "exclude_between": ["per share", "per-share"],
        "statement_markers": [
            "consolidated balance sheet", "consolidated balance sheets",
            "balance sheet", "long-term debt", "borrowings", "note 9",
        ],
    },
    "cash_flow": {
        "label": "Cash Flow from Operating Activities",
        "kind": "currency",
        "query_variants": [
            "net cash provided by operating activities",
            "consolidated statements of cash flows operating activities net cash",
        ],
        "fallback_query_variants": [
            "cash generated by operating activities adjustments to reconcile net income",
            "net cash from operations operating activities",
        ],
        "phrase_tiers": [
            ["net cash provided by operating activities",
             "net cash generated by operating activities",
             "net cash generated from operating activities",
             "net cash from operating activities",
             "net cash from operations",
             "net cash provided by operations",
             "cash generated by operating activities"],
            ["cash flow from operating activities", "cash flows from operating activities",
             "cash generated from operations"],
        ],
        "exclude_keywords": ["investing activities", "financing activities", "free cash flow"],
        "exclude_between": ["per share", "per-share"],
        "statement_markers": [
            "consolidated statements of cash flows", "consolidated statement of cash flows",
            "statements of cash flows", "statement of cash flows", "cash flow statement",
            "cash flows statements", "cash flows statement",
        ],
    },
    "eps": {
        "label": "Earnings Per Share (EPS)",
        "kind": "eps",
        # Several phrasings on purpose: the EPS figure lives in the income
        # statement AND in an "earnings per share" note, laid out differently.
        "query_variants": [
            "diluted earnings per share basic EPS",
            "earnings per share note net income weighted-average basic diluted shares outstanding",
            "consolidated statements of operations earnings per share basic diluted shares used in computing",
        ],
        "fallback_query_variants": [
            "net income per share basic diluted computation of earnings per share",
            "earnings per equity share basic diluted statement of profit and loss",
        ],
        # Explicit, generic preference: diluted first, then basic.
        "phrase_tiers": [
            ["diluted earnings per share", "diluted eps", "earnings per share - diluted",
             "earnings per share (diluted)", "earnings per share, diluted",
             "diluted earnings per equity share",
             "diluted net income per share", "net income per share - diluted",
             "net income per share (diluted)", "net income per share, diluted"],
            ["basic earnings per share", "basic eps", "earnings per share - basic",
             "earnings per share (basic)", "earnings per share, basic",
             "basic earnings per equity share",
             "basic net income per share", "net income per share - basic",
             "net income per share (basic)", "net income per share, basic"],
        ],
        "exclude_keywords": [],
        "exclude_between": ["net income", "weighted", "shares used", "shares outstanding",
                            "numerator", "denominator"],
        "statement_markers": INCOME_HEADINGS + ["earnings per share", "net income per share"],
    },
    "total_income": {
        "label": "Total Income",
        "kind": "currency",
        "query_variants": [
            "total income statement of profit and loss revenue from operations other income",
            "consolidated statement of profit and loss total income",
        ],
        "fallback_query_variants": [
            "total revenue and other income statement of profit and loss",
        ],
        "phrase_tiers": [
            ["total income", "total revenue and other income"],
        ],
        "exclude_keywords": ["comprehensive income"],
        "exclude_between": ["per share", "per-share"],
        "statement_markers": INCOME_HEADINGS,
    },
    "profit_before_tax": {
        "label": "Profit Before Tax (PBT)",
        "kind": "currency",
        "query_variants": [
            "profit before tax income before provision for income taxes",
            "statement of profit and loss profit before tax",
            "consolidated statements of operations income before provision for income taxes",
        ],
        "fallback_query_variants": [
            "income before income taxes earnings before income taxes",
        ],
        "phrase_tiers": [
            ["profit before tax", "profit before taxes", "profit before income tax",
             "income before provision for income taxes", "income before income taxes",
             "earnings before income taxes", "income before taxes"],
        ],
        "exclude_keywords": [],
        "exclude_between": ["per share", "per-share"],
        "statement_markers": INCOME_HEADINGS,
    },
    "profit_after_tax": {
        "label": "Profit After Tax (PAT)",
        "kind": "currency",
        "query_variants": [
            "profit after tax profit for the year statement of profit and loss",
            "consolidated statements of operations net income profit after tax",
        ],
        "fallback_query_variants": [
            "net income consolidated statements of operations profit for the period",
        ],
        "phrase_tiers": [
            ["profit after tax", "profit for the year", "profit for the period",
             "net profit after tax", "net profit"],
            ["net income", "net earnings"],
        ],
        "exclude_keywords": ["noncontrolling interest", "non-controlling interest", "discontinued operations", "comprehensive income"],
        "exclude_between": ["per share", "per-share"],
        "statement_markers": INCOME_HEADINGS,
    },
    "ebitda": {
        "label": "EBITDA",
        "kind": "currency",
        "query_variants": [
            "earnings before interest tax depreciation and amortization EBITDA",
            "operating profit depreciation amortization EBITDA",
        ],
        "fallback_query_variants": [
            "adjusted ebitda operating ebitda",
        ],
        "phrase_tiers": [
            ["adjusted ebitda", "ebitda", "operating ebitda",
              "earnings before interest, tax, depreciation and amortization",
              "earnings before interest, taxes, depreciation and amortization"],
        ],
        "exclude_keywords": ["ebitda margin", "as a percentage"],
        "exclude_between": ["per share", "per-share", "%"],
        "statement_markers": INCOME_HEADINGS + ["ebitda", "non-gaap"],
    },
    "total_assets": {
        "label": "Total Assets",
        "kind": "currency",
        "query_variants": [
            "consolidated balance sheets total assets current assets non-current assets",
            "balance sheet total assets property plant and equipment",
        ],
        "fallback_query_variants": [
            "total non-current assets total current assets total assets",
        ],
        "phrase_tiers": [
            ["total assets"],
        ],
        "exclude_keywords": ["turnover", "total current assets", "total non-current assets"],
        "exclude_between": ["per share", "per-share"],
        "statement_markers": [
            "consolidated balance sheet", "consolidated balance sheets",
            "balance sheet", "balance sheets", "statement of financial position",
        ],
    },
    "total_liabilities": {
        "label": "Total Liabilities",
        "kind": "currency",
        "query_variants": [
            "consolidated balance sheets total liabilities current liabilities non-current liabilities",
            "balance sheet total liabilities",
        ],
        "fallback_query_variants": [
            "total non-current liabilities total current liabilities total liabilities",
        ],
        "phrase_tiers": [
            ["total liabilities"],
        ],
        "exclude_keywords": ["total current liabilities", "total non-current liabilities"],
        "exclude_between": ["per share", "per-share"],
        "statement_markers": [
            "consolidated balance sheet", "consolidated balance sheets",
            "balance sheet", "balance sheets", "statement of financial position",
        ],
    },
    "total_equity": {
        "label": "Total Equity",
        "kind": "currency",
        "query_variants": [
            "total shareholders equity total stockholders equity consolidated balance sheets",
            "total equity balance sheet equity share capital other equity",
        ],
        "fallback_query_variants": [
            "total shareholders' equity total stockholders' equity",
        ],
        "phrase_tiers": [
            ["total shareholders' equity", "total shareholders equity",
             "total stockholders' equity", "total stockholders equity",
             "total equity", "total share capital and other equity"],
        ],
        "exclude_keywords": [],
        "exclude_between": ["per share", "per-share"],
        "statement_markers": [
            "consolidated balance sheet", "consolidated balance sheets",
            "balance sheet", "balance sheets", "statement of financial position",
        ],
    },
    "cash_and_equivalents": {
        "label": "Cash and Cash Equivalents",
        "kind": "currency",
        "query_variants": [
            "cash and cash equivalents at end of year consolidated statements of cash flows",
            "cash and cash equivalents marketable securities consolidated balance sheets",
        ],
        "fallback_query_variants": [
            "cash and cash equivalents end of period balance sheet",
        ],
        "phrase_tiers": [
            ["cash and cash equivalents at end of year", "cash and cash equivalents at the end of the year",
             "cash and cash equivalents, end of year", "cash and cash equivalents end of period",
             "cash and cash equivalents", "cash and cash balances"],
        ],
        "exclude_keywords": ["restricted cash"],
        "exclude_between": ["per share", "per-share"],
        "statement_markers": [
            "consolidated statements of cash flows", "statements of cash flows",
            "consolidated balance sheet", "consolidated balance sheets", "balance sheet",
        ],
    },
    "investing_cash_flow": {
        "label": "Cash Flow from Investing Activities",
        "kind": "currency",
        "query_variants": [
            "net cash provided by used in investing activities consolidated statements of cash flows",
            "net cash used in investing activities capital expenditures",
        ],
        "fallback_query_variants": [
            "cash flows from investing activities purchase of property plant and equipment",
        ],
        "phrase_tiers": [
            ["net cash provided by (used in) investing activities",
             "net cash provided by/used in investing activities",
             "net cash used in investing activities",
             "net cash provided by investing activities",
             "net cash generated from investing activities",
             "cash flow from investing activities",
             "cash flows from investing activities",
             "net cash from investing activities"],
        ],
        "exclude_keywords": ["operating activities", "financing activities"],
        "exclude_between": ["per share", "per-share"],
        "statement_markers": [
            "consolidated statements of cash flows", "consolidated statement of cash flows",
            "statements of cash flows", "statement of cash flows", "cash flow statement",
        ],
    },
    "financing_cash_flow": {
        "label": "Cash Flow from Financing Activities",
        "kind": "currency",
        "query_variants": [
            "net cash provided by used in financing activities consolidated statements of cash flows",
            "net cash used in financing activities dividends paid debt repayments",
        ],
        "fallback_query_variants": [
            "cash flows from financing activities common stock repurchased dividends",
        ],
        "phrase_tiers": [
            ["net cash provided by (used in) financing activities",
             "net cash provided by/used in financing activities",
             "net cash used in financing activities",
             "net cash provided by financing activities",
             "net cash generated from financing activities",
             "cash flow from financing activities",
             "cash flows from financing activities",
             "net cash from financing activities"],
        ],
        "exclude_keywords": ["operating activities", "investing activities"],
        "exclude_between": ["per share", "per-share"],
        "statement_markers": [
            "consolidated statements of cash flows", "consolidated statement of cash flows",
            "statements of cash flows", "statement of cash flows", "cash flow statement",
        ],
    },
    "gross_profit": {
        "label": "Gross Profit",
        "kind": "currency",
        "query_variants": [
            "consolidated statements of operations gross profit total net sales cost of sales gross margin",
            "gross profit total net revenue cost of revenues",
        ],
        "fallback_query_variants": [
            "gross margin products services total gross margin",
            "gross profit revenue cost of goods sold",
        ],
        "phrase_tiers": [
            ["gross profit", "total gross profit", "gross income"],
            ["total gross margin", "gross margin"],
        ],
        "exclude_keywords": ["percentage", "%", "gross margin percentage", "gross margin %"],
        "exclude_between": ["per share", "per-share", "%"],
        "statement_markers": INCOME_HEADINGS,
    },
}

# Probed once per company/year to learn the statement currency and scale.
CONTEXT_PROBE_QUERIES = [
    "consolidated statements of operations in millions except per share amounts",
    "consolidated balance sheets statements of cash flows in millions U.S. dollars",
]
GENERIC_STATEMENT_MARKERS = INCOME_HEADINGS + [
    "consolidated balance sheet", "consolidated balance sheets",
    "statements of cash flows", "statement of cash flows", "cash flow statement",
]

GEO_LABEL_PATTERN = re.compile(
    r"(?:americas|europe|greater china|japan|asia[\s-]*pacific|north america|"
    r"emea|apac|middle east|africa|rest of (?:the )?world)\s*:", re.IGNORECASE)
SEGMENT_WORD_PATTERN = re.compile(r"\bsegments?\b", re.IGNORECASE)


# ---------------------------------------------------------------------------
# ChromaDB + embedding model access (read-only)
# ---------------------------------------------------------------------------
_model = None
_collection = None


def get_model():
    """Load the embedding model once and reuse it for every query."""
    global _model
    if _model is not None:
        return _model

    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        LOGGER.error("sentence-transformers is not installed.")
        raise RuntimeError("sentence-transformers is not installed. Run: pip install sentence-transformers")

    LOGGER.info(f"Loading embedding model: {MODEL_NAME}")
    try:
        _model = SentenceTransformer(MODEL_NAME, local_files_only=True)
    except Exception:
        _model = SentenceTransformer(MODEL_NAME)
    return _model


def get_collection():
    """Open the existing persistent ChromaDB collection (read-only use)."""
    global _collection
    if _collection is not None:
        return _collection

    try:
        import chromadb
    except ImportError:
        LOGGER.error("chromadb is not installed.")
        raise RuntimeError("chromadb is not installed. Run: pip install chromadb")

    if not (VECTORDB_DIR / "chroma.sqlite3").exists():
        try:
            import data_loader
            data_loader.ensure_data_loaded()
        except Exception as exc:
            LOGGER.warning("Automatic data loading attempt failed: %s", exc)

    if not VECTORDB_DIR.exists():
        msg = f"No database found at {VECTORDB_DIR}. Run index_chromadb.py first."
        LOGGER.error(msg)
        raise FileNotFoundError(msg)

    client = chromadb.PersistentClient(path=str(VECTORDB_DIR))
    try:
        _collection = client.get_collection(name=COLLECTION_NAME)
    except Exception as exc:
        msg = f"Collection '{COLLECTION_NAME}' does not exist at {VECTORDB_DIR}: {exc}"
        LOGGER.error(msg)
        raise RuntimeError(msg)

    return _collection


def build_filter(company=None, fiscal_year=None):
    """Build a ChromaDB metadata filter that scopes retrieval to one company/year."""
    conditions = []
    if company:
        conditions.append({"company": company})
    if fiscal_year:
        conditions.append({"fiscal_year": str(fiscal_year)})

    if len(conditions) == 1:
        return conditions[0]
    if len(conditions) > 1:
        return {"$and": conditions}
    return None


def retrieve_chunks(query_text, where=None, top_k=DEFAULT_TOP_K):
    """
    Embed a query and retrieve the top_k most similar chunks from ChromaDB.
    Returns a list of dicts: {text, metadata, similarity, distance}.
    """
    model = get_model()
    collection = get_collection()

    query_vector = model.encode(
        [query_text], normalize_embeddings=True, convert_to_numpy=True,
    )[0].tolist()

    raw = collection.query(
        query_embeddings=[query_vector],
        n_results=top_k,
        where=where,
        include=["documents", "metadatas", "distances"],
    )

    results = []
    for text, metadata, distance in zip(
        raw["documents"][0], raw["metadatas"][0], raw["distances"][0]
    ):
        results.append({
            "text": text, "metadata": metadata, "distance": distance,
            "similarity": round(1 - distance, 4),
        })
    return results


def retrieve_chunks_multi(query_variants, where=None, top_k=DEFAULT_TOP_K):
    """
    Run retrieve_chunks() once per query variant and merge, de-duplicating
    by chunk text and keeping each chunk's best similarity.
    """
    merged = {}
    for query_text in query_variants:
        for chunk in retrieve_chunks(query_text, where=where, top_k=top_k):
            key = chunk["text"]
            if key not in merged or chunk["similarity"] > merged[key]["similarity"]:
                merged[key] = chunk
    return sorted(merged.values(), key=lambda c: c["similarity"], reverse=True)


# ---------------------------------------------------------------------------
# Number, currency and scale primitives
# ---------------------------------------------------------------------------
CURRENCY = r"(?:US\$|\$|₹|Rs\.?|INR|USD)"
NUMBER = r"\(?-?\d{1,3}(?:,\d{2,3})*(?:\.\d+)?\)?|\(?-?\d+(?:\.\d+)?\)?"
SCALE = r"(?:million|billion|trillion|thousand|crore|lakh|mn|bn)"

# (?!\w) instead of \b so a number followed by "%" and then punctuation
# ("46.2%.") keeps its percent sign.
VALUE_PATTERN = re.compile(
    rf"({CURRENCY})?\s*({NUMBER})\s*(%)?\s*({SCALE})?(?!\w)", re.IGNORECASE,
)

CURRENCY_NAME_MAP = {
    "$": "USD", "us$": "USD", "usd": "USD",
    "₹": "INR", "rs": "INR", "rs.": "INR", "inr": "INR",
}

# Currency evidence in running text or table headers.
CURRENCY_TOKENS = [
    ("USD", re.compile(r"US\$|U\.S\.\s+dollars?|United\s+States\s+dollars?|\bUSD\b|\$", re.I)),
    ("INR", re.compile(r"₹|\bRs\.?(?=[\s\d])|\bINR\b|\bIndian\s+rupees?\b|\brupees?\b", re.I)),
]

SCALE_CANON = {
    "thousand": "thousand", "thousands": "thousand",
    "million": "million", "millions": "million", "mn": "million",
    "billion": "billion", "billions": "billion", "bn": "billion",
    "trillion": "trillion",
    "crore": "crore", "crores": "crore",
    "lakh": "lakh", "lakhs": "lakh", "lac": "lakh", "lacs": "lakh",
}
# Canonical monetary unit is "million <currency>".
SCALE_TO_MILLIONS = {
    "thousand": 0.001, "million": 1.0, "billion": 1000.0,
    "trillion": 1_000_000.0, "crore": 10.0, "lakh": 0.1,
}

# "in millions", "(millions)", "$ in millions", "Rs. in crore", "(₹ crore)".
TABLE_SCALE_PATTERN = re.compile(
    r"(?:\bin\s+|[₹$]\s*|\bRs\.?\s*|\bINR\s*|\bUSD\s*|\(\s*)"
    r"(thousands?|millions?|billions?|crores?|lakhs?|lacs?)\b", re.IGNORECASE)

YEAR_RE = re.compile(r"\b((?:19|20)\d{2})\b")


def parse_number(raw_text):
    """Convert a matched number string to a float. Parentheses mean negative."""
    text = raw_text.strip()
    negative = text.startswith("(") and text.endswith(")")
    text = text.strip("()").replace(",", "").replace("$", "").strip()
    try:
        value = float(text)
    except ValueError:
        return None
    return -value if negative else value


def looks_like_a_year(value, currency, scale, percent):
    """True for a bare 4-digit year (e.g. 2024), so it isn't read as an amount."""
    if currency or scale or percent:
        return False
    return value is not None and float(value).is_integer() and 1900 <= value <= 2100


def currency_symbol_to_name(symbol):
    """Turn a matched currency symbol into a code, e.g. '$' -> 'USD'."""
    if not symbol:
        return None
    return CURRENCY_NAME_MAP.get(symbol.strip().lower(), symbol.strip().upper())


def find_all_occurrences(text, phrase):
    """Every start index where `phrase` appears in `text` (case-insensitive)."""
    lower, needle = text.lower(), phrase.lower()
    starts, idx = [], lower.find(needle)
    while idx != -1:
        starts.append(idx)
        idx = lower.find(needle, idx + 1)
    return starts


def sentence_window(text, start_idx, length, max_chars=280):
    """Text following a position, cut at the end of the current sentence."""
    window_start = start_idx + length
    remainder = text[window_start:window_start + max_chars]
    end_of_sentence = re.search(r"[.!?](?:\s|$)", remainder)
    return remainder[:end_of_sentence.end()] if end_of_sentence else remainder


def count_currency_tokens(text):
    """Return (Counter of currency codes, first snippet per code)."""
    counts, snippets = Counter(), {}
    for code, pattern in CURRENCY_TOKENS:
        matches = list(pattern.finditer(text))
        if matches:
            counts[code] = len(matches)
            first = matches[0]
            start, end = max(0, first.start() - 30), min(len(text), first.end() + 30)
            snippets[code] = " ".join(text[start:end].split())
    return counts, snippets


def dominant_currency(counts, snippets):
    """A currency is only reported if it makes up >= 80% of the evidence."""
    if not counts:
        return None, None
    code, count = counts.most_common(1)[0]
    if count / sum(counts.values()) < 0.8:
        return None, None
    return code, snippets.get(code)


def detect_currency(text):
    """Currency evidenced inside one chunk: (code, evidence) or (None, None)."""
    counts, snippets = count_currency_tokens(text)
    return dominant_currency(counts, snippets)


def find_scale_mentions(text):
    """
    Scale declarations ("in millions", "(₹ in crore)") in a chunk. A mention
    that describes share counts ("number of shares ... in thousands") is
    skipped, because it does not apply to monetary amounts.
    """
    mentions = []
    for match in TABLE_SCALE_PATTERN.finditer(text):
        before = text[max(0, match.start() - 45):match.start()].lower()
        if "shares" in before or "per-share" in before or "per share" in before:
            continue
        scale = SCALE_CANON.get(match.group(1).lower())
        if scale is None:
            continue
        start, end = max(0, match.start() - 20), min(len(text), match.end() + 25)
        mentions.append({"pos": match.start(), "scale": scale,
                         "snippet": " ".join(text[start:end].split())})
    return mentions


def detect_table_scale(text):
    """First monetary scale declared in a chunk (canonical singular) or None."""
    mentions = find_scale_mentions(text)
    return mentions[0]["scale"] if mentions else None


def normalize_monetary(raw_value, reported_scale, currency_code):
    """
    Convert a monetary amount to the canonical unit "million <CCY>".

    The returned value and unit ALWAYS describe the same quantity:
        383.3 billion USD -> (383300.0, "million USD")
        391035 [table in millions] USD -> (391035.0, "million USD")
    If no scale is known the amount stays in absolute units and the unit says
    only the currency (or None), so it can never be mistaken for millions.
    """
    factor = SCALE_TO_MILLIONS.get(reported_scale)
    if factor is None:
        return raw_value, (currency_code or None)
    normalized = round(raw_value * factor, 6)
    return normalized, (f"million {currency_code}" if currency_code else "million")


# ---------------------------------------------------------------------------
# Table structure: year columns and row cells
# ---------------------------------------------------------------------------

def find_year_header_before(text, position, max_distance=3500):
    """
    Find the nearest year header above `position`, e.g. "2024 2023 2022" or
    "September 28, 2024 September 30, 2023 September 24, 2022".

    Returns (columns, evidence). `columns` lists the header in order; a
    "CHANGE" placeholder is inserted where the header has a change/percent
    column between two years, so cell positions stay aligned.
    Returns (None, None) if there is no consecutive-year header close enough.
    """
    segment = text[max(0, position - max_distance):position]
    runs, current = [], []
    for match in YEAR_RE.finditer(segment):
        item = (match.start(), match.end(), match.group(1))
        if current:
            gap = item[0] - current[-1][1]
            step = int(item[2]) - int(current[-1][2])
            consistent = len(current) < 2 or step == int(current[-1][2]) - int(current[-2][2])
            if gap <= 45 and abs(step) == 1 and consistent:
                current.append(item)
                continue
            runs.append(current)
            current = []
        current.append(item)
    if current:
        runs.append(current)

    for run in reversed(runs):
        if len(run) < 2:
            continue
        columns = []
        for index, (start, _end, year) in enumerate(run):
            if index > 0 and re.search(r"change|%|variance|growth",
                                       segment[run[index - 1][1]:start], re.I):
                columns.append("CHANGE")
            columns.append(year)
        evidence = " ".join(segment[run[0][0]:run[-1][1]].split())
        return columns, evidence
    return None, None


GAP_OK = re.compile(r"^[\s:$()\-–—]*$")


def parse_row_cells(text, begin, max_chars=320):
    """
    Read the contiguous numeric cells that follow a row label. Reading stops
    at the first word (the next row label), so one row's cells never mix
    with the next row's.
    """
    tail = text[begin:begin + max_chars]
    cells, cursor = [], 0
    for match in VALUE_PATTERN.finditer(tail):
        if not GAP_OK.match(tail[cursor:match.start()]):
            break
        currency, number_text, percent, scale = match.groups()
        cursor = match.end()
        value = parse_number(number_text)
        if value is None:
            continue
        cells.append({
            "number_text": number_text.strip(), "value": value,
            "currency_symbol": (currency or "").strip() or None,
            "percent": bool(percent),
            "scale_word": (scale or "").lower() or None,
            "start": begin + match.start(), "end": begin + match.end(),
        })
    return cells


def is_footnote_marker(cell):
    """A bare "(1)" next to a label is a footnote marker, not a data cell."""
    return (re.fullmatch(r"\(\d{1,2}\)", cell["number_text"]) is not None
            and not cell["currency_symbol"] and not cell["scale_word"] and not cell["percent"])


def select_column(cells, header_columns, requested_year):
    """
    Choose the cell under the requested fiscal year.
    Returns (column_source, cell, count_mismatch, rejection_reason).
    """
    if header_columns:
        years = [c for c in header_columns if c != "CHANGE"]
        if requested_year:
            if requested_year not in header_columns:
                return None, None, False, (
                    f"table header years {years} do not include requested FY{requested_year}")
            if len(cells) == len(header_columns):
                return "year_header", cells[header_columns.index(requested_year)], False, None
            if len(cells) == len(years) and len(years) != len(header_columns):
                # The header lists Change columns but this row has one cell per
                # year (e.g. a percentage row): align by year, not by position.
                return "year_header", cells[years.index(requested_year)], False, None
            index = header_columns.index(requested_year)
            if index >= len(cells):
                return None, None, False, (
                    f"row has {len(cells)} cell(s) but FY{requested_year} is column {index + 1}")
            return "year_header", cells[index], True, None
        return "first_column_no_year_requested", cells[0], False, None
    if len(cells) >= 2:
        return "assumed_first_column", cells[0], False, None
    return "single_cell", cells[0], False, None


def accept_cell(kind, cell, table_like):
    """Hard validity rules per metric kind. Returns (ok, rejection_reason)."""
    value = cell["value"]
    if kind == "percent":
        if cell["percent"]:
            return True, None
        return False, "value has no % sign, so it is not a stated percentage"
    if kind == "eps":
        if cell["percent"]:
            return False, "a percentage, not a per-share amount"
        if "." not in cell["number_text"]:
            return False, ("no decimal point: per-share values are reported like 6.13, "
                           "so this looks like a total or a share count")
        if not (0 < abs(value) < 1000):
            return False, "outside a plausible per-share range"
        return True, None
    if cell["percent"]:
        return False, "a percentage cell (for example a Change column), not a monetary amount"
    if looks_like_a_year(value, cell["currency_symbol"], cell["scale_word"], cell["percent"]):
        return False, "a bare year, not an amount"
    if not table_like and not (cell["currency_symbol"] or cell["scale_word"]
                               or "," in cell["number_text"]):
        return False, "bare small integer in running text (likely a note or page reference)"
    return True, None


def year_context_check(text, start, end, requested_year):
    """For running text: does the sentence tie this number to the requested year?"""
    if not requested_year:
        return "none"
    nearby = YEAR_RE.findall(text[end:end + 40]) + YEAR_RE.findall(text[max(0, start - 40):start])
    if not nearby:
        return "none"
    return "match" if requested_year in nearby else "conflict"


# ---------------------------------------------------------------------------
# Currency / scale resolution with evidence
# ---------------------------------------------------------------------------

def resolve_currency(cell, row_cells, chunk_ctx, doc_ctx):
    """Currency from the strongest available evidence: (code, source, evidence)."""
    if cell.get("currency_symbol"):
        return currency_symbol_to_name(cell["currency_symbol"]), "number_symbol", cell["currency_symbol"]
    for other in row_cells:
        if other.get("currency_symbol"):
            return (currency_symbol_to_name(other["currency_symbol"]), "row_symbol",
                    other["currency_symbol"])
    if chunk_ctx["currency"]:
        return chunk_ctx["currency"], "chunk_context", chunk_ctx["currency_evidence"]
    if doc_ctx and doc_ctx.get("currency"):
        return doc_ctx["currency"], "document_context", doc_ctx.get("currency_evidence")
    return None, None, None


def resolve_scale(cell, table_like, chunk_ctx, doc_ctx):
    """Scale from the strongest available evidence: (canonical, source, evidence)."""
    if cell.get("scale_word"):
        word = cell["scale_word"]
        return SCALE_CANON.get(word, word), "number", word
    mentions = chunk_ctx["scale_mentions"]
    if mentions:
        before = [m for m in mentions if m["pos"] <= cell["start"]]
        near = [m for m in mentions if abs(m["pos"] - cell["start"]) <= 150]
        chosen = (before[-1] if before else mentions[0]) if table_like else (near[0] if near else None)
        if chosen:
            return chosen["scale"], "table_header", chosen["snippet"]
    if table_like and doc_ctx and doc_ctx.get("scale"):
        return doc_ctx["scale"], "document_context", doc_ctx.get("scale_evidence")
    return None, None, None


def analyze_chunk_context(text, definition):
    """Statement heading, currency and scale evidence found in one chunk."""
    lower = text.lower()
    found = [(lower.find(m), m) for m in definition.get("statement_markers", []) if m in lower]
    heading = None
    if found:
        position = min(found)[0]
        heading = " ".join(text[position:position + 140].split())
    currency, evidence = detect_currency(text)
    return {"strong_marker": bool(found), "heading_snippet": heading,
            "currency": currency, "currency_evidence": evidence,
            "scale_mentions": find_scale_mentions(text)}


# ---------------------------------------------------------------------------
# Document-level context probe (cached per company / fiscal year)
# ---------------------------------------------------------------------------
_doc_context_cache = {}


def clear_context_cache():
    _doc_context_cache.clear()


def summarize_context(chunks):
    """Currency and scale evidenced by statement-like pages of one filing."""
    counts, snippets = Counter(), {}
    scale = scale_evidence = None
    for chunk in chunks:
        page = chunk["metadata"].get("page_number")
        chunk_counts, chunk_snippets = count_currency_tokens(chunk["text"])
        counts.update(chunk_counts)
        for code, snippet in chunk_snippets.items():
            snippets.setdefault(code, f"page {page}: {snippet}")
        lower = chunk["text"].lower()
        if scale is None and any(m in lower for m in GENERIC_STATEMENT_MARKERS):
            mentions = find_scale_mentions(chunk["text"])
            if mentions:
                scale = mentions[0]["scale"]
                scale_evidence = f"page {page}: {mentions[0]['snippet']}"
    currency, currency_evidence = dominant_currency(counts, snippets)
    return {"currency": currency, "currency_evidence": currency_evidence,
            "scale": scale, "scale_evidence": scale_evidence}


def get_document_context(company, fiscal_year, top_k=DEFAULT_TOP_K):
    """Probe the statement pages of one company/year once, then reuse the result."""
    key = (id(get_collection()), company, str(fiscal_year))
    if key not in _doc_context_cache:
        chunks = retrieve_chunks_multi(CONTEXT_PROBE_QUERIES,
                                       where=build_filter(company, fiscal_year), top_k=top_k)
        _doc_context_cache[key] = summarize_context(chunks)
    return _doc_context_cache[key]


def _safe_document_context(company, fiscal_year, top_k):
    """A failed probe only removes the fallback evidence; it never stops extraction."""
    try:
        return get_document_context(company, fiscal_year, top_k)
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Candidate generation and scoring
# ---------------------------------------------------------------------------

def find_eps_label_hits(text):
    """
    Row labels "Diluted" / "Basic" under an "earnings per share" heading (the
    income-statement layout). Rows that follow "shares used in computing" or
    "weighted-average" are share counts, not EPS, and are skipped.
    """
    hits = []
    heading = re.compile(
        r"(?:earnings|net\s+income|net\s+loss)\s+per\s+(?:common\s+|equity\s+|ordinary\s+)?share|\beps\b",
        re.I,
    )
    for found in heading.finditer(text):
        before = text[max(0, found.start() - 60):found.start()].lower()
        if any(w in before for w in ("shares used", "weighted", "number of shares")):
            continue
        region_start = found.end()
        region = text[region_start:region_start + 260]
        for label in re.finditer(r"\b(diluted|basic)\b", region, re.I):
            label_before = region[max(0, label.start() - 60):label.start()].lower()
            if any(w in label_before for w in ("weighted", "shares used",
                                               "shares outstanding", "number of shares")):
                continue
            word = label.group(1).lower()
            hits.append({
                "tier_index": 0 if word == "diluted" else 1,
                "phrase": f"{word} (row under 'earnings per share')",
                "start": region_start + label.start(), "end": region_start + label.end(),
            })
    return hits


def find_label_hits(text, definition):
    """Every place a metric label appears in a chunk, with its phrase tier."""
    hits = []
    for tier_index, tier in enumerate(definition["phrase_tiers"]):
        for phrase in tier:
            for start in find_all_occurrences(text, phrase):
                hits.append({"tier_index": tier_index, "phrase": phrase,
                             "start": start, "end": start + len(phrase)})
    if definition["kind"] == "eps":
        hits.extend(find_eps_label_hits(text))
    return hits


def first_match(keywords, text):
    for keyword in keywords:
        if keyword in text:
            return keyword
    return None


def build_evidence(text, label_start, cells_end, chunk_ctx, header_evidence):
    """Row text plus, when present, the statement heading and year columns."""
    row = " ".join(text[max(0, label_start - 50):min(len(text), cells_end + 35)].split())
    parts = []
    heading = chunk_ctx.get("heading_snippet")
    if heading and heading[:40] not in row:
        parts.append(heading[:140])
    if header_evidence and header_evidence not in row and not (heading and header_evidence in heading):
        parts.append(f"[columns: {header_evidence}]")
    parts.append(row)
    combined = " ... ".join(parts)
    return combined[:750] if len(combined) > 750 else combined


def downgrade(level):
    return CONFIDENCE_LEVELS[max(0, CONFIDENCE_LEVELS.index(level) - 1)]


def finalize_candidate(hit, chunk, chunk_index, definition, cell, row_cells, table_like,
                       column_source, mismatch, header_columns, header_evidence,
                       chunk_ctx, doc_ctx, mode, year_match, repeated_penalty):
    """Turn one validated cell into a scored, normalized candidate."""
    kind = definition["kind"]
    text, meta = chunk["text"], chunk["metadata"]
    warnings = []

    currency = currency_source = currency_evidence = None
    reported_scale = scale_source = scale_evidence = None

    if kind == "percent":
        normalized_value, normalized_unit = cell["value"], "percent"
    elif kind == "eps":
        currency, currency_source, currency_evidence = resolve_currency(cell, row_cells, chunk_ctx, doc_ctx)
        normalized_value = cell["value"]
        normalized_unit = f"{currency} per share" if currency else "per share"
        if currency is None:
            warnings.append("currency not determined from the document")
    else:
        currency, currency_source, currency_evidence = resolve_currency(cell, row_cells, chunk_ctx, doc_ctx)
        reported_scale, scale_source, scale_evidence = resolve_scale(cell, table_like, chunk_ctx, doc_ctx)
        normalized_value, normalized_unit = normalize_monetary(cell["value"], reported_scale, currency)
        if currency is None:
            warnings.append("currency not determined from the document")
        if reported_scale is None:
            warnings.append("scale not determined; value left in absolute units")

    if column_source == "year_header" and not mismatch:
        confidence = "high"
    elif mode == "prose" or column_source == "single_cell":
        confidence = "medium"
    else:
        confidence = "low"
        warnings.append("fiscal-year column assumed (no matching year header)"
                        if column_source == "assumed_first_column"
                        else "row cell count differs from the year header")
    if kind != "percent" and currency is None:
        confidence = downgrade(confidence)
    if kind == "currency" and reported_scale is None:
        confidence = downgrade(confidence)

    # --- score: structure and context, not just label proximity -----------
    score = (3 - min(hit["tier_index"], 3)) * 3          # more specific label wins
    if "total" in hit["phrase"].lower():
        score += 2
    score += 4 if chunk_ctx["strong_marker"] else 0       # real statement heading
    if mode == "row" and table_like:
        score += 3                                        # read from a table row
    if column_source == "year_header":
        score += 8 if not mismatch else 4                 # verified fiscal-year column
    if year_match == "match":
        score += 3                                        # prose names the requested year
    if cell.get("currency_symbol"):
        score += 1
    if reported_scale and scale_source in ("number", "table_header"):
        score += 1
    score -= repeated_penalty
    score -= max(0, cell["start"] - hit["end"]) / 80.0

    chunk_lower = text.lower()
    if "standalone" in chunk_lower and "consolidated" not in chunk_lower[:max(0, hit["start"])]:
        statement_type = "Standalone"
    elif "consolidated" in chunk_lower:
        statement_type = "Consolidated"
    elif any(k in str(meta.get("source_file", "")).lower() for k in ["apple", "amazon", "microsoft", "tesla"]):
        statement_type = "Consolidated"
    else:
        statement_type = "Consolidated"

    return {
        "value": normalized_value, "normalized_value": normalized_value,
        "normalized_unit": normalized_unit, "unit": normalized_unit,
        "raw_value": cell["value"], "raw_text": cell["number_text"],
        "reported_scale": reported_scale, "scale_source": scale_source,
        "scale_evidence": scale_evidence,
        "currency": currency, "currency_source": currency_source,
        "currency_evidence": currency_evidence,
        "matched_phrase": hit["phrase"], "tier_index": hit["tier_index"],
        "score": round(score, 2), "confidence": confidence, "warnings": warnings,
        "column_source": column_source, "header_columns": header_columns,
        "source_context": "table_row" if (mode == "row" and table_like) else mode,
        "evidence": build_evidence(text, hit["start"], cell["end"], chunk_ctx, header_evidence),
        "source_file": meta.get("source_file"), "page_number": meta.get("page_number"),
        "similarity": chunk["similarity"], "chunk_index": chunk_index,
        "cell_start": cell["start"],
        "statement_type": statement_type,
    }


def candidates_from_hit(hit, chunk, chunk_index, definition, requested_year, doc_ctx,
                        chunk_ctx, label_counts, trace):
    """All valid candidates for one label occurrence (empty list if rejected)."""
    kind = definition["kind"]
    text, meta = chunk["text"], chunk["metadata"]
    start, end, phrase = hit["start"], hit["end"], hit["phrase"]

    def reject(reason):
        if trace is not None:
            trace.append({
                "reason": reason, "phrase": phrase,
                "page": meta.get("page_number"), "source_file": meta.get("source_file"),
                "context": " ".join(text[max(0, start - 40):min(len(text), end + 90)].split()),
            })
        return []

    label_before = text[max(0, start - 40):start].lower()
    excluded = first_match(definition.get("exclude_keywords", []), label_before + " " + phrase.lower())
    if excluded:
        return reject(f"excluded context '{excluded}' just before the label")

    if definition.get("exclude_label_words") and "total" not in phrase.lower():
        label_word = first_match(definition["exclude_label_words"], text[max(0, start - 30):start].lower())
        if label_word:
            return reject(f"sub-line label '{label_word}' (not the company-wide figure)")

    before_90 = text[max(0, start - 90):start]
    if definition.get("exclude_geo_label") and GEO_LABEL_PATTERN.search(before_90):
        return reject("row sits under a geographic/segment label (not the consolidated figure)")
    if definition.get("exclude_segment_word") and SEGMENT_WORD_PATTERN.search(before_90):
        return reject("the word 'segment' precedes this row (segment table, not consolidated)")

    repeated_penalty = 4 if (definition.get("penalize_repeated_label")
                             and label_counts[phrase] >= 3) else 0
    header_columns, header_evidence = find_year_header_before(text, start)
    cells = [c for c in parse_row_cells(text, end) if not is_footnote_marker(c)]

    # ---- row mode: contiguous cells directly after the label --------------
    if cells:
        if looks_like_a_year(cells[0]["value"], cells[0]["currency_symbol"],
                             cells[0]["scale_word"], cells[0]["percent"]):
            return reject("first cell is a year, so this is a header row, not data")
        table_like = len(cells) >= 2 or header_columns is not None
        column_source, cell, mismatch, why = select_column(cells, header_columns, requested_year)
        if cell is None:
            return reject(why)
        ok, why = accept_cell(kind, cell, table_like)
        if not ok:
            return reject(f"cell {cell['number_text']!r}: {why}")
        return [finalize_candidate(
            hit, chunk, chunk_index, definition, cell, cells, table_like, column_source,
            mismatch, header_columns, header_evidence, chunk_ctx, doc_ctx,
            "row", "none", repeated_penalty)]

    # ---- prose mode: a number inside the same sentence --------------------
    window = sentence_window(text, start, len(phrase))
    candidates = []
    for match in VALUE_PATTERN.finditer(window):
        if match.start() > 200 or len(candidates) >= 2:
            break
        currency, number_text, percent, scale = match.groups()
        value = parse_number(number_text)
        if value is None:
            continue
        cell = {"number_text": number_text.strip(), "value": value,
                "currency_symbol": (currency or "").strip() or None,
                "percent": bool(percent), "scale_word": (scale or "").lower() or None,
                "start": end + match.start(), "end": end + match.end()}
        between = window[:match.start()].lower()

        excluded = first_match(definition.get("exclude_keywords", []), between)
        if excluded:
            return reject(f"excluded context '{excluded}' between label and number")
        if kind == "currency" or kind == "eps":
            blocked = first_match(definition.get("exclude_between", []), between)
            if blocked:
                return reject(f"'{blocked}' between label and number")
        if definition.get("exclude_label_words") and "total" not in phrase.lower():
            label_word = first_match(definition["exclude_label_words"], between)
            if label_word:
                return reject(f"sub-line label '{label_word}' between label and number")

        year_match = year_context_check(text, cell["start"], cell["end"], requested_year)
        if year_match == "conflict":
            reject(f"sentence ties {cell['number_text']!r} to a different year than FY{requested_year}")
            continue
        ok, why = accept_cell(kind, cell, False)
        if not ok:
            reject(f"cell {cell['number_text']!r}: {why}")
            continue
        candidates.append(finalize_candidate(
            hit, chunk, chunk_index, definition, cell, [cell], False, "prose_single_value",
            False, header_columns, header_evidence, chunk_ctx, doc_ctx,
            "prose", year_match, repeated_penalty))
    return candidates


def generate_candidates(chunks, definition, requested_year, doc_ctx=None, trace=None):
    """
    Score every plausible value for a metric across all retrieved chunks and
    return them best-first. Candidates at the same cell position are merged
    (the better-scoring label wins), so a value is never counted twice.
    """
    requested = str(requested_year) if requested_year else None
    by_cell = {}
    for chunk_index, chunk in enumerate(chunks):
        chunk_ctx = analyze_chunk_context(chunk["text"], definition)
        hits = find_label_hits(chunk["text"], definition)
        label_counts = Counter(h["phrase"] for h in hits)
        for hit in hits:
            for candidate in candidates_from_hit(hit, chunk, chunk_index, definition, requested,
                                                 doc_ctx, chunk_ctx, label_counts, trace):
                key = (candidate["source_file"], candidate["page_number"], candidate["cell_start"])
                if key not in by_cell or candidate["score"] > by_cell[key]["score"]:
                    by_cell[key] = candidate
    return sorted(by_cell.values(),
                  key=lambda c: (-c["score"], -CONFIDENCE_LEVELS.index(c["confidence"]),
                                 c["chunk_index"]))


# ---------------------------------------------------------------------------
# Building the structured metric record
# ---------------------------------------------------------------------------

def run_extraction_passes(definition, where, top_k, requested_year, doc_ctx, trace=None):
    """
    Pass 1 retrieves with the metric's query variants. Pass 2 runs ONLY if
    pass 1 produced no candidate: different query wording and a wider
    window, never a blind larger top_k on the same query.
    """
    queries = list(definition["query_variants"])
    chunks = retrieve_chunks_multi(queries, where=where, top_k=top_k)
    candidates = generate_candidates(chunks, definition, requested_year, doc_ctx, trace)
    passes = [{"pass": 1, "queries": queries, "top_k": top_k, "chunks": len(chunks)}]

    fallback = definition.get("fallback_query_variants")
    if not candidates and fallback:
        queries = queries + list(fallback)
        chunks = retrieve_chunks_multi(queries, where=where, top_k=top_k * 2)
        if trace is not None:
            trace.append({"reason": "pass 1 found no candidate; running pass 2 with extra queries",
                          "phrase": "-", "page": "-", "source_file": "-", "context": ""})
        candidates = generate_candidates(chunks, definition, requested_year, doc_ctx, trace)
        passes.append({"pass": 2, "queries": queries, "top_k": top_k * 2, "chunks": len(chunks)})
    return chunks, candidates, passes


def candidate_summary(candidate):
    keys = ["value", "unit", "raw_value", "reported_scale", "scale_source", "currency",
            "currency_source", "matched_phrase", "score", "confidence", "column_source",
            "source_file", "page_number", "evidence", "statement_type"]
    return {k: candidate.get(k) for k in keys}


def build_metric_record(metric_key, company, fiscal_year, top_k=DEFAULT_TOP_K,
                        doc_context=None, debug=False):
    """
    Retrieve evidence for one metric, score every candidate, and package the
    winner. Source file, page and evidence always belong to the winning
    candidate itself.
    """
    definition = METRIC_DEFINITIONS[metric_key]
    where = build_filter(company, fiscal_year)
    if doc_context is None:
        doc_context = _safe_document_context(company, fiscal_year, top_k)

    trace = [] if debug else None
    chunks, candidates, passes = run_extraction_passes(
        definition, where, top_k, fiscal_year, doc_context, trace)

    record = {
        "metric": definition["label"], "company": company, "fiscal_year": fiscal_year,
        "value": None, "unit": None, "currency": None, "scale": None,
        "source_file": None, "page_number": None, "evidence": None,
        "status": NOT_FOUND, "score": None, "derived": False, "calculation": None,
        "raw_value": None, "reported_scale": None, "scale_source": None,
        "normalized_value": None, "normalized_unit": None,
        "currency_source": None, "confidence": None, "source_context": None,
        "column_source": None, "warnings": [],
        "statement_type": "Consolidated",
        "candidates": [candidate_summary(c) for c in candidates[:5]],
        "retrieved_chunks": [
            {"text_preview": " ".join(c["text"].split())[:300],
             "source_file": c["metadata"].get("source_file"),
             "page_number": c["metadata"].get("page_number"),
             "similarity": c["similarity"]}
            for c in chunks
        ],
    }

    if candidates:
        best = candidates[0]
        record.update({
            "value": best["value"], "unit": best["unit"], "currency": best["currency"],
            "scale": best["reported_scale"],
            "source_file": best["source_file"], "page_number": best["page_number"],
            "evidence": best["evidence"], "status": FOUND, "score": best["score"],
            "raw_value": best["raw_value"], "reported_scale": best["reported_scale"],
            "scale_source": best["scale_source"],
            "normalized_value": best["normalized_value"],
            "normalized_unit": best["normalized_unit"],
            "currency_source": best["currency_source"], "confidence": best["confidence"],
            "source_context": best["source_context"], "column_source": best["column_source"],
            "warnings": best["warnings"],
            "statement_type": best.get("statement_type", "Consolidated"),
        })

    if debug:
        record["debug"] = {
            "query_variants": definition["query_variants"],
            "fallback_query_variants": definition.get("fallback_query_variants", []),
            "passes": passes, "document_context": doc_context,
            "retrieved": [
                {"source_file": c["metadata"].get("source_file"),
                 "page_number": c["metadata"].get("page_number"),
                 "similarity": c["similarity"],
                 "labels_present": sorted({p for tier in definition["phrase_tiers"] for p in tier
                                           if p.lower() in c["text"].lower()})}
                for c in chunks
            ],
            "all_candidates": candidates[:10], "rejected": trace,
        }
    return record


def build_operating_margin(company, fiscal_year, top_k=DEFAULT_TOP_K,
                           doc_context=None, debug=False):
    """
    Operating Margin: use a stated percentage if one exists. Otherwise derive
    Operating Income / Revenue * 100, only when both were found for the same
    company and fiscal year in the SAME canonical unit and currency. The
    result is marked derived and keeps source/page information for both inputs.
    """
    stated = build_metric_record("operating_margin", company, fiscal_year, top_k,
                                 doc_context=doc_context, debug=debug)
    if stated["status"] == FOUND:
        return stated

    income = build_metric_record("operating_income", company, fiscal_year, top_k,
                                 doc_context=doc_context, debug=debug)
    revenue = build_metric_record("revenue", company, fiscal_year, top_k,
                                  doc_context=doc_context, debug=debug)

    compatible = (
        income["status"] == FOUND and revenue["status"] == FOUND
        and revenue["normalized_value"] not in (None, 0)
        and income["normalized_unit"] == revenue["normalized_unit"]
        and income["currency"] == revenue["currency"]
    )
    if not compatible:
        reasons = []
        if income["status"] != FOUND:
            reasons.append("operating income not found")
        if revenue["status"] != FOUND:
            reasons.append("revenue not found")
        if income["status"] == FOUND and revenue["status"] == FOUND:
            reasons.append(f"units differ ({income['normalized_unit']} vs {revenue['normalized_unit']})")
        stated["warnings"] = ["not derived: " + "; ".join(reasons)]
        return stated

    margin = round((income["normalized_value"] / revenue["normalized_value"]) * 100, 2)
    sources = [
        {"metric": r["metric"], "value": r["value"], "unit": r["unit"],
         "source_file": r["source_file"], "page_number": r["page_number"]}
        for r in (income, revenue)
    ]
    confidence = min((income["confidence"], revenue["confidence"]),
                     key=CONFIDENCE_LEVELS.index)
    stated.update({
        "value": margin, "normalized_value": margin, "unit": "percent",
        "normalized_unit": "percent", "currency": None, "scale": None,
        "raw_value": margin,
        "source_file": income["source_file"], "page_number": income["page_number"],
        "evidence": (
            f"Calculated from Operating Income ({income['value']} {income['unit']}, "
            f"{income['source_file']} p.{income['page_number']}) and Revenue "
            f"({revenue['value']} {revenue['unit']}, {revenue['source_file']} "
            f"p.{revenue['page_number']})."),
        "status": FOUND, "derived": True,
        "calculation": "Operating Margin = Operating Income / Revenue * 100",
        "derived_from": sources, "confidence": confidence,
        "warnings": income["warnings"] + revenue["warnings"],
        "source_context": "derived",
        "statement_type": income.get("statement_type") or revenue.get("statement_type") or "Consolidated",
    })
    return stated


def build_net_margin(company, fiscal_year, top_k=DEFAULT_TOP_K,
                     doc_context=None, debug=False):
    """
    Net Profit Margin: derive from Net Income / Revenue * 100 when both are found.
    """
    net_inc = build_metric_record("net_income", company, fiscal_year, top_k,
                                 doc_context=doc_context, debug=debug)
    revenue = build_metric_record("revenue", company, fiscal_year, top_k,
                                  doc_context=doc_context, debug=debug)

    compatible = (
        net_inc["status"] == FOUND and revenue["status"] == FOUND
        and revenue["normalized_value"] not in (None, 0)
        and net_inc["normalized_unit"] == revenue["normalized_unit"]
        and net_inc["currency"] == revenue["currency"]
    )
    if not compatible:
        return {
            "metric": "Net Profit Margin", "company": company, "fiscal_year": fiscal_year,
            "value": None, "unit": "percent", "status": NOT_FOUND, "derived": True,
            "reason": "Net Income and Revenue must both be found with matching units to calculate Net Margin.",
            "source_file": None, "page_number": None, "evidence": None, "warnings": [],
            "statement_type": "Consolidated",
        }

    margin = round((net_inc["normalized_value"] / revenue["normalized_value"]) * 100, 2)
    sources = [
        {"metric": r["metric"], "value": r["value"], "unit": r["unit"],
         "source_file": r["source_file"], "page_number": r["page_number"]}
        for r in (net_inc, revenue)
    ]
    confidence = min((net_inc["confidence"], revenue["confidence"]),
                     key=CONFIDENCE_LEVELS.index)
    return {
        "metric": "Net Profit Margin", "company": company, "fiscal_year": fiscal_year,
        "value": margin, "normalized_value": margin, "unit": "percent",
        "normalized_unit": "percent", "currency": None, "scale": None,
        "raw_value": margin,
        "source_file": net_inc["source_file"], "page_number": net_inc["page_number"],
        "evidence": (
            f"Calculated from Net Income ({net_inc['value']} {net_inc['unit']}, "
            f"{net_inc['source_file']} p.{net_inc['page_number']}) and Revenue "
            f"({revenue['value']} {revenue['unit']}, {revenue['source_file']} "
            f"p.{revenue['page_number']})."),
        "status": FOUND, "derived": True,
        "calculation": "Net Profit Margin = Net Income / Revenue * 100",
        "derived_from": sources, "confidence": confidence,
        "warnings": net_inc["warnings"] + revenue["warnings"],
        "source_context": "derived",
        "statement_type": net_inc.get("statement_type", "Consolidated"),
    }


def extract_metrics(company, fiscal_year, metric_keys=None, top_k=DEFAULT_TOP_K):
    """
    Run extraction for a list of metrics (defaults to the 7 target metrics;
    "operating_income" is available on request). The statement currency/scale
    probe runs once and is shared by every metric.
    """
    default_keys = ["revenue", "net_income", "gross_margin", "operating_margin",
                    "total_debt", "cash_flow", "eps"]
    metric_keys = metric_keys or default_keys
    doc_context = _safe_document_context(company, fiscal_year, top_k)

    results = {}
    for key in metric_keys:
        if key not in METRIC_DEFINITIONS and key not in ("operating_margin", "net_margin"):
            print(f"[WARNING] Unknown metric '{key}', skipped.")
            continue
        if key == "operating_margin":
            results[key] = build_operating_margin(company, fiscal_year, top_k, doc_context=doc_context)
        elif key == "net_margin":
            results[key] = build_net_margin(company, fiscal_year, top_k, doc_context=doc_context)
        else:
            results[key] = build_metric_record(key, company, fiscal_year, top_k, doc_context=doc_context)
    return results


# ---------------------------------------------------------------------------
# Printing and diagnostics
# ---------------------------------------------------------------------------

def print_metric_report(results):
    for _key, record in results.items():
        print("=" * 78)
        print(f"METRIC: {record['metric']}  |  {record['company']} FY{record['fiscal_year']}")
        print("=" * 78)
        print(f"Status : {record['status'].upper()}"
              + ("  (DERIVED / CALCULATED)" if record.get("derived") else ""))
        if record["status"] == FOUND:
            print(f"Value  : {record['value']} ({record['unit']})   confidence: {record.get('confidence')}")
            if record.get("reported_scale") or record.get("raw_value") is not None:
                print(f"Raw    : {record.get('raw_value')} reported scale={record.get('reported_scale')} "
                      f"(scale from {record.get('scale_source')}), currency={record.get('currency')} "
                      f"(from {record.get('currency_source')})")
            if record.get("calculation"):
                print(f"Formula: {record['calculation']}")
            print(f"Source : {record['source_file']}  (page {record['page_number']})")
            print(f"Evidence: {record['evidence']}")
            for warning in record.get("warnings") or []:
                print(f"Warning: {warning}")
        else:
            print("Value  : not_found (no acceptable candidate in the retrieved evidence)")
            for warning in record.get("warnings") or []:
                print(f"Note   : {warning}")
        print()


def print_compact_summary(results_by_year):
    print("\n" + "=" * 78)
    print("COMPACT SUMMARY")
    print("=" * 78)
    for year, results in results_by_year.items():
        for key, record in results.items():
            if record["status"] == FOUND:
                print(f"FY{year} {key:<17} {record['value']} {record['unit']}  "
                      f"[p.{record['page_number']}, {record.get('confidence')}"
                      f"{', derived' if record.get('derived') else ''}]")
            else:
                print(f"FY{year} {key:<17} NOT_FOUND")


def diagnose_metric(company, fiscal_year, metric_key, top_k=DEFAULT_TOP_K):
    """Print everything needed to see WHY a metric was or was not extracted."""
    record = build_metric_record(metric_key, company, fiscal_year, top_k, debug=True)
    info = record["debug"]
    print("=" * 78)
    print(f"DIAGNOSE  metric={metric_key}  company={company}  FY{fiscal_year}")
    print("=" * 78)
    print("Query variants (pass 1):")
    for query in info["query_variants"]:
        print(f"  - {query}")
    for entry in info["passes"]:
        print(f"Pass {entry['pass']}: {len(entry['queries'])} queries, top_k={entry['top_k']} "
              f"-> {entry['chunks']} merged chunk(s)")
    context = info["document_context"] or {}
    print(f"Document context: currency={context.get('currency')} "
          f"({context.get('currency_evidence')}); scale={context.get('scale')} "
          f"({context.get('scale_evidence')})")
    print(f"\nRetrieved chunks ({len(info['retrieved'])}):")
    for chunk in info["retrieved"]:
        print(f"  {chunk['source_file']} p.{chunk['page_number']}  similarity={chunk['similarity']}  "
              f"labels present: {chunk['labels_present'] or 'none'}")
    print(f"\nCandidates considered ({len(info['all_candidates'])} shown, best first):")
    for number, c in enumerate(info["all_candidates"], start=1):
        print(f"  [{number}] value={c['value']} {c['unit']}  raw={c['raw_value']} "
              f"reported_scale={c['reported_scale']} (from {c['scale_source']})  "
              f"currency={c['currency']} (from {c['currency_source']})")
        print(f"      phrase='{c['matched_phrase']}' score={c['score']} confidence={c['confidence']} "
              f"column={c['column_source']} {c['source_file']} p.{c['page_number']}")
        print(f"      evidence: {c['evidence'][:200]}")
    print(f"\nRejected candidates ({len(info['rejected'])}):")
    for item in info["rejected"][:25]:
        print(f"  - p.{item['page']} '{item['phrase']}': {item['reason']}")
        if item["context"]:
            print(f"      ...{item['context'][:150]}")
    print(f"\nRESULT: {record['status'].upper()}  value={record['value']} {record['unit']}  "
          f"page={record['page_number']}  confidence={record['confidence']}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Extraction Agent")
    parser.add_argument("--diagnose", nargs=3, metavar=("COMPANY", "YEAR", "METRIC"),
                        help="print retrieval, candidates and rejections for one metric")
    parser.add_argument("--company", default="Apple")
    parser.add_argument("--years", nargs="+", default=["2023", "2024"])
    arguments = parser.parse_args()

    if arguments.diagnose:
        diagnose_metric(*arguments.diagnose)
    else:
        test_metrics = ["revenue", "net_income", "eps", "gross_margin",
                        "operating_income", "operating_margin"]
        all_results = {}
        for year in arguments.years:
            print(f"\nRunning Extraction Agent for {arguments.company}, FY{year}")
            all_results[year] = extract_metrics(arguments.company, year, metric_keys=test_metrics)
            print_metric_report(all_results[year])
        print_compact_summary(all_results)