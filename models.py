"""
models.py

Small shared helpers for reading the "status" field of an Agent 1
(extraction_agent.py) metric record. Imported by analysis_agent.py,
red_flag_agent.py and test_full_pipeline.py so that "is this metric
available?" has exactly one definition in the whole project.

This module deliberately does nothing else: no ChromaDB, no embedding
model, no PDF work, no financial calculations, no agent calls. It does not
change the Agent 1 record schema; it only reads one field from it.

Status values
-------------
extraction_agent.py writes exactly two statuses: "found" and "not_found"
(lowercase). Briefs and documentation for the later agents write them as
FOUND / NOT_FOUND. Both spellings are accepted; comparison ignores case and
surrounding spaces.

Anything that is not recognizably "found" (None, an empty string, a missing
key, or an unexpected word) is treated as NOT_FOUND. The safe default is
"not available", so a malformed record can never be mistaken for real data.
"""

FOUND = "FOUND"
NOT_FOUND = "NOT_FOUND"


def normalize_status(status):
    """
    Return "FOUND" or "NOT_FOUND" for any status value.

        "found", "FOUND", "Found", " found "  -> "FOUND"
        "not_found", "NOT_FOUND", None, "", anything else -> "NOT_FOUND"
    """
    if isinstance(status, str) and status.strip().lower() == "found":
        return FOUND
    return NOT_FOUND


def is_found(record):
    """
    True only if an Agent 1 metric record (a dict) has a found status.

    Returns False for None, for anything that is not a dict, and for a
    record with no "status" key. This checks the status only; whether the
    "value" is a usable number is validated separately by the agents that
    need to calculate with it (see validate_metric_record in
    analysis_agent.py).
    """
    if not isinstance(record, dict):
        return False
    return normalize_status(record.get("status")) == FOUND