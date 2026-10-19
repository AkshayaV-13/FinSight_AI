"""
multi_agent_orchestrator.py

Coordinates the agents sequentially:

    [optional] document processing / indexing   (injected hook, see below)
        -> Agent 1  extraction_agent.extract_metrics()   once per fiscal year
        -> Agent 2  analysis_agent.analyze_company()
        -> Agent 3  red_flag_agent.build_red_flags()
        -> ONE structured result

This module only coordinates: it calls each agent, hands the real output of
one stage to the next, checks that each stage returned the right kind of
object, records problems, and assembles the final result. It contains no
extraction, financial-calculation or red-flag logic, and it never mutates an
agent's output: the Agent 1 records inside the result are the very objects
Agent 2 and Agent 3 received, so source_file / page_number / evidence are
never lost.

Two kinds of problem are kept strictly apart
--------------------------------------------
PIPELINE ERROR   a stage crashed, stopped the process, or returned the wrong
                 kind of object. Listed in result["pipeline_errors"] and the
                 result status is "pipeline_error". Later stages that depend
                 on the failed one are NOT run, and are marked "not_run".
DATA LIMITATION  a metric was NOT_FOUND, a comparison could not be
                 calculated, or a red-flag category could not be evaluated.
                 This is normal for real filings. Listed in
                 result["limitations"]; the status stays "success".

Document upload
---------------
There is no per-document upload function in the project today: the existing
scripts (chunk_documents.main, generate_embeddings.main, index_chromadb.main)
are batch jobs over every file in their folders, and no UI or backend upload
event exists yet. So nothing here fakes an upload. The integration point is
the `ingest` argument of run_financial_analysis():

    result = run_financial_analysis("Apple", "2023", "2024", ingest=my_function)

`ingest` is any zero-argument callable. An upload handler would save the file,
then pass a function that processes and indexes it. The orchestrator runs it
FIRST, as stage "document_processing", and only then starts Agent 1. If no
`ingest` is given the stage is reported as "skipped" (documents assumed to be
indexed already). index_documents_with_existing_scripts() is an optional
ready-made `ingest` that runs the existing indexing scripts in order.

The agents are injectable (extractor / analyzer / red_flagger) so each stage
can be unit-tested in isolation. By default they are the real agents.

Usage:
    python multi_agent_orchestrator.py                  # Apple 2023 -> 2024
    python multi_agent_orchestrator.py Microsoft 2023 2024
"""

import sys
import time
from datetime import datetime, timezone

from models import is_found

ORCHESTRATOR_VERSION = "1.0"

STATUS_SUCCESS = "success"
STATUS_PIPELINE_ERROR = "pipeline_error"
STATUS_INVALID_INPUT = "invalid_input"

# Stage states
COMPLETED, FAILED, NOT_RUN, SKIPPED = "completed", "error", "not_run", "skipped"


# ---------------------------------------------------------------------------
# Small helpers
# ---------------------------------------------------------------------------

def _log(verbose, message):
    if verbose:
        print(f"[ORCHESTRATOR] {message}", flush=True)


def _error(stage, error_type, message, hint=None):
    return {"stage": stage, "error_type": error_type, "message": message, "hint": hint}


def _call_stage(stage, function, *args, **kwargs):
    """
    Run one stage. Returns (output, None) or (None, error_dict).
    SystemExit is caught on purpose: extraction_agent calls sys.exit(1) when a
    dependency, the vectordb/ folder or the collection is missing, and that
    would otherwise end the whole process without a structured error.
    """
    try:
        return function(*args, **kwargs), None
    except SystemExit as stop:
        return None, _error(
            stage, "system_exit",
            f"{stage} terminated the process (SystemExit {stop.code}).",
            hint="For Agent 1 this usually means a missing package, a missing vectordb/ "
                 "folder, or a missing 'financial_documents' collection. Read the message "
                 "printed just above.")
    except Exception as exc:  # any crash is a pipeline error, never a silent skip
        return None, _error(stage, "exception", f"{type(exc).__name__}: {exc}")


def _valid_year(value):
    return isinstance(value, str) and value.strip().isdigit() and len(value.strip()) == 4


def _validate_inputs(company, previous_year, current_year):
    problems = []
    if not isinstance(company, str) or not company.strip():
        problems.append("company must be a non-empty string")
    if not _valid_year(previous_year):
        problems.append("previous_fiscal_year must be a 4-digit year string such as '2023'")
    if not _valid_year(current_year):
        problems.append("current_fiscal_year must be a 4-digit year string such as '2024'")
    if not problems and previous_year == current_year:
        problems.append("previous_fiscal_year and current_fiscal_year must differ")
    if not problems and int(previous_year) > int(current_year):
        problems.append("previous_fiscal_year must be earlier than current_fiscal_year")
    return problems


def _agent1_limitations(previous_year, current_year, previous_data, current_data):
    """Valid data limitations from Agent 1 output (NOT pipeline errors)."""
    limitations = []
    for year, data in ((previous_year, previous_data), (current_year, current_data)):
        found_any = False
        for key, record in data.items():
            if is_found(record):
                found_any = True
            else:
                limitations.append({
                    "stage": "agent_1", "type": "metric_not_found", "metric": key,
                    "fiscal_year": year,
                    "message": f"Agent 1 returned NOT_FOUND for {key} in FY{year}.",
                })
        if data and not found_any:
            limitations.append({
                "stage": "agent_1", "type": "no_metrics_found_for_year", "metric": None,
                "fiscal_year": year,
                "message": (f"Agent 1 found no metric at all for FY{year}. This usually means "
                            f"no indexed chunks matched this company/fiscal-year filter, not "
                            f"that the report lacks the numbers."),
            })
    return limitations


def _agent2_limitations(analysis):
    limitations = []
    for key, comparison in (analysis.get("comparisons") or {}).items():
        if isinstance(comparison, dict) and comparison.get("status") != "calculated":
            limitations.append({
                "stage": "agent_2", "type": "comparison_not_calculated", "metric": key,
                "fiscal_year": None, "message": comparison.get("reason") or "not calculated",
            })
    return limitations


def _agent3_limitations(red_flags):
    return [{"stage": "agent_3", "type": "category_not_evaluated", "metric": None,
             "fiscal_year": None, "message": text}
            for text in (red_flags.get("data_limitations") or [])]


# ---------------------------------------------------------------------------
# Optional ready-made document-processing hook
# ---------------------------------------------------------------------------

def index_documents_with_existing_scripts():
    """
    Optional `ingest` function: runs the project's existing indexing scripts in
    order (chunking -> embeddings -> ChromaDB). It does not extract text from
    PDFs: extract_pdf_text.py must already have written documents/*.json.
    Those scripts are batch jobs that skip chunks already processed, so this
    is safe to repeat, but it works on every document, not just a new one.
    """
    import chunk_documents
    import generate_embeddings
    try:
        import index_chromadb as idx
    except ImportError:
        import index_cromadb as idx

    chunk_documents.main()
    generate_embeddings.main()
    idx.main()


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def run_financial_analysis(company, previous_fiscal_year, current_fiscal_year, *,
                           ingest=None, extractor=None, analyzer=None, red_flagger=None,
                           top_k=3, verbose=True):
    """
    Run document processing (optional), Agent 1, Agent 2 and Agent 3 for one
    company across two fiscal years and return ONE structured result.

    Never raises for an agent failure: failures are returned in
    result["pipeline_errors"] with result["status"] == "pipeline_error".
    """
    started = time.time()
    previous_year = previous_fiscal_year.strip() if isinstance(previous_fiscal_year, str) else previous_fiscal_year
    current_year = current_fiscal_year.strip() if isinstance(current_fiscal_year, str) else current_fiscal_year

    result = {
        "status": STATUS_SUCCESS,
        "company": company,
        "previous_fiscal_year": previous_year,
        "current_fiscal_year": current_year,
        "document_processing": {"status": NOT_RUN, "error": None},
        "agent_1": {"status": NOT_RUN, "previous": None, "current": None,
                    "limitations": [], "error": None},
        "agent_2": {"status": NOT_RUN, "result": None, "limitations": [], "error": None},
        "agent_3": {"status": NOT_RUN, "result": None, "limitations": [], "error": None},
        "limitations": [],
        "pipeline_errors": [],
        "pipeline_metadata": {
            "orchestrator_version": ORCHESTRATOR_VERSION,
            "started_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "finished_at": None, "duration_seconds": None,
            "stages_run": [],
            "handoffs": {"agent_1_to_agent_2": False, "agent_2_to_agent_3": False},
            "has_data_limitations": False,
        },
    }

    def finish(status):
        result["status"] = status
        result["limitations"] = (result["agent_1"]["limitations"] + result["agent_2"]["limitations"]
                                 + result["agent_3"]["limitations"])
        meta = result["pipeline_metadata"]
        meta["finished_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
        meta["duration_seconds"] = round(time.time() - started, 3)
        meta["has_data_limitations"] = bool(result["limitations"])
        return result

    def fail(stage_key, error):
        result[stage_key]["status"] = FAILED
        result[stage_key]["error"] = error
        result["pipeline_errors"].append(error)
        _log(verbose, f"{stage_key} FAILED: {error['message']}")

    _log(verbose, f"Starting pipeline for {company} FY{previous_year} -> FY{current_year}")

    problems = _validate_inputs(company, previous_year, current_year)
    if problems:
        result["pipeline_errors"].append(_error("input_validation", "invalid_input", "; ".join(problems)))
        _log(verbose, f"Invalid input: {'; '.join(problems)}")
        return finish(STATUS_INVALID_INPUT)

    # ---- document processing (optional hook) -----------------------------
    if ingest is None:
        result["document_processing"]["status"] = SKIPPED
        _log(verbose, "Document processing skipped (no ingest hook; documents assumed indexed)")
    else:
        _log(verbose, "Document processing started")
        _output, error = _call_stage("document_processing", ingest)
        if error:
            fail("document_processing", error)
            return finish(STATUS_PIPELINE_ERROR)
        result["document_processing"]["status"] = COMPLETED
        result["pipeline_metadata"]["stages_run"].append("document_processing")
        _log(verbose, "Document processing completed")

    # ---- Agent 1 ----------------------------------------------------------
    if extractor is None:
        import extraction_agent
        extractor = extraction_agent.extract_metrics

    _log(verbose, "Agent 1 started")
    previous_data, error = _call_stage("agent_1", extractor, company, previous_year)
    current_data = None
    if error is None:
        current_data, error = _call_stage("agent_1", extractor, company, current_year)
    if error is None:
        for year, data in ((previous_year, previous_data), (current_year, current_data)):
            if not isinstance(data, dict) or any(not isinstance(r, dict) for r in data.values()):
                error = _error("agent_1", "malformed_output",
                               f"Agent 1 returned an unexpected structure for FY{year} "
                               f"(expected a dict of metric records).")
                break
    if error:
        fail("agent_1", error)
        return finish(STATUS_PIPELINE_ERROR)

    result["agent_1"].update({
        "status": COMPLETED, "previous": previous_data, "current": current_data,
        "limitations": _agent1_limitations(previous_year, current_year, previous_data, current_data),
    })
    result["pipeline_metadata"]["stages_run"].append("agent_1")
    _log(verbose, f"Agent 1 completed ({len(result['agent_1']['limitations'])} data limitation(s))")

    # ---- Agent 2 ----------------------------------------------------------
    if analyzer is None:
        import analysis_agent
        analyzer = analysis_agent.analyze_company

    _log(verbose, "Agent 2 started")
    analysis, error = _call_stage("agent_2", analyzer, company, previous_year, current_year,
                                  previous_data, current_data)
    if error is None and (not isinstance(analysis, dict)
                          or not isinstance(analysis.get("comparisons"), dict)):
        error = _error("agent_2", "malformed_output",
                       "Agent 2 returned an unexpected structure (expected a dict with 'comparisons').")
    if error:
        fail("agent_2", error)
        return finish(STATUS_PIPELINE_ERROR)

    result["agent_2"].update({"status": COMPLETED, "result": analysis,
                              "limitations": _agent2_limitations(analysis)})
    result["pipeline_metadata"]["stages_run"].append("agent_2")
    result["pipeline_metadata"]["handoffs"]["agent_1_to_agent_2"] = True
    _log(verbose, f"Agent 2 completed ({len(result['agent_2']['limitations'])} comparison(s) not calculated)")

    # ---- Agent 3 ----------------------------------------------------------
    if red_flagger is None:
        import red_flag_agent
        red_flagger = red_flag_agent.build_red_flags

    _log(verbose, "Agent 3 started")
    red_flags, error = _call_stage("agent_3", red_flagger, analysis,
                                   current_data=current_data, previous_data=previous_data)
    if error is None and (not isinstance(red_flags, dict) or not isinstance(red_flags.get("flags"), list)):
        error = _error("agent_3", "malformed_output",
                       "Agent 3 returned an unexpected structure (expected a dict with a 'flags' list).")
    if error:
        fail("agent_3", error)
        return finish(STATUS_PIPELINE_ERROR)

    result["agent_3"].update({"status": COMPLETED, "result": red_flags,
                              "limitations": _agent3_limitations(red_flags)})
    result["pipeline_metadata"]["stages_run"].append("agent_3")
    result["pipeline_metadata"]["handoffs"]["agent_2_to_agent_3"] = True
    _log(verbose, f"Agent 3 completed ({len(red_flags['flags'])} flag(s))")

    _log(verbose, "Pipeline completed")
    return finish(STATUS_SUCCESS)


# ---------------------------------------------------------------------------
# Readable report (reuses the agents' own formatters; no logic duplicated)
# ---------------------------------------------------------------------------

def format_orchestration_report(result):
    lines = ["=" * 70, "MULTI-AGENT ANALYSIS RESULT", "",
             f"Company: {result['company']}",
             f"Fiscal years: FY{result['previous_fiscal_year']} -> FY{result['current_fiscal_year']}",
             f"Status: {result['status']}",
             f"Stages run: {', '.join(result['pipeline_metadata']['stages_run']) or '(none)'}", ""]

    if result["pipeline_errors"]:
        lines.append("PIPELINE ERRORS")
        for error in result["pipeline_errors"]:
            lines.append(f"  - [{error['stage']}] {error['error_type']}: {error['message']}")
            if error.get("hint"):
                lines.append(f"      hint: {error['hint']}")
        lines.append("")

    if result["agent_2"]["result"] is not None:
        import analysis_agent
        lines += [analysis_agent.format_report(result["agent_2"]["result"]), ""]
    if result["agent_3"]["result"] is not None:
        import red_flag_agent
        lines += [red_flag_agent.format_red_flag_report(result["agent_3"]["result"]), ""]

    lines.append("DATA LIMITATIONS (valid, not errors)")
    for item in result["limitations"] or [{"stage": "-", "message": "(none)"}]:
        lines.append(f"  - [{item['stage']}] {item['message']}")
    return "\n".join(lines)


if __name__ == "__main__":
    arguments = sys.argv[1:]
    if len(arguments) == 3:
        company_name, previous, current = arguments
    else:
        company_name, previous, current = "Apple", "2023", "2024"
    outcome = run_financial_analysis(company_name, previous, current)
    print()
    print(format_orchestration_report(outcome))
    sys.exit(0 if outcome["status"] == STATUS_SUCCESS else 1)