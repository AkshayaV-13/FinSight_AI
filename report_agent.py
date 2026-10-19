"""
report_agent.py

Report Agent (first version): turns the outputs of the existing agents into one
structured financial research report for a company across two fiscal years.

It assembles and formats. It does NOT extract, retrieve, calculate or score
anything itself:
  - values, sources, units, currency  <- Agent 1 records
  - changes (absolute, percent, percentage points) <- Agent 2's comparisons, copied
    as-is, never recomputed (this module contains no arithmetic on financial values)
  - red flags and their limitations   <- Agent 3's result, copied as-is
  - optional verbatim explanation excerpts <- the Research Agent's result
Agents 1-3 are reached through multi_agent_orchestrator.run_financial_analysis();
a caller that already has that result can pass it in as `pipeline_result`.

Report structure (a dict; sections can be added later by extending SECTION_ORDER)
  report_title, company, fiscal_years, status ("complete" | "partial" | "error"),
  executive_summary {text, statements[]}, key_financials [rows], red_flags [flags],
  red_flag_summary, red_flags_evaluated, evidence_excerpts [], sources [],
  limitations [], errors [], section_order [], metadata {}

Traceability
  Each key-financials row holds, for each year, the value, unit, currency, basis
  (REPORTED / DERIVED), source_file, page_number, evidence and confidence exactly as
  Agent 1 returned them. A derived value (operating margin; net profit margin) keeps
  its formula and the sources of BOTH inputs. Every executive-summary statement lists
  the sources it rests on. A missing value stays missing, with the reason.

Wording is neutral and descriptive. The report never gives advice, never ranks, and
never reads "no red flags" as "financially healthy".

The module does not depend on any UI. From Python:
    report = generate_report("Apple", "2023", "2024")
    print(format_report_markdown(report))
"""

from models import is_found

REPORT_ROWS = [
    ("revenue", "Revenue"), ("net_income", "Net Income"), ("eps", "EPS"),
    ("gross_margin", "Gross Margin"), ("operating_margin", "Operating Margin"),
    ("net_profit_margin", "Net Profit Margin"),
]
MARGIN_KEYS = {"gross_margin", "operating_margin", "net_profit_margin"}
SECTION_ORDER = ["executive_summary", "key_financials", "red_flags", "evidence_excerpts",
                 "sources", "limitations"]
NET_PROFIT_MARGIN_FORMULA = "Net Profit Margin = Net Income / Revenue * 100 (calculated by Agent 2)"
DISCLAIMER = ("This report is descriptive research based only on the indexed annual reports. "
              "It is not investment advice and does not rate or rank companies.")
INSUFFICIENT = "Insufficient evidence was found in the indexed annual reports for this item."


# ---------------------------------------------------------------------------
# 1. Data collection (reuses the existing pipeline; nothing is extracted here)
# ---------------------------------------------------------------------------

def collect_report_inputs(company, previous_year, current_year, pipeline_result=None, runner=None):
    """Return the orchestrator result for this company and years."""
    if pipeline_result is not None:
        return pipeline_result
    if runner is None:
        import multi_agent_orchestrator
        runner = multi_agent_orchestrator.run_financial_analysis
    return runner(company, previous_year, current_year, verbose=False)


# ---------------------------------------------------------------------------
# 2. Key financials
# ---------------------------------------------------------------------------

def missing_cell(year, reason):
    return {"fiscal_year": year, "status": "NOT_FOUND", "value": None, "unit": None,
            "currency": None, "basis": None, "source_file": None, "page_number": None,
            "evidence": None, "confidence": None, "warnings": [], "calculation": None,
            "derived_from": None, "reason": reason}


def cell_from_record(record, year):
    """One year's value for one metric, copied from an Agent 1 record."""
    if not is_found(record) or record.get("value") is None:
        return missing_cell(year, "NOT_FOUND in the indexed document")
    return {
        "fiscal_year": year, "status": "FOUND", "value": record["value"],
        "unit": record.get("unit"), "currency": record.get("currency"),
        "basis": "DERIVED" if record.get("derived") else "REPORTED",
        "source_file": record.get("source_file"), "page_number": record.get("page_number"),
        "evidence": record.get("evidence"), "confidence": record.get("confidence"),
        "warnings": list(record.get("warnings") or []), "calculation": record.get("calculation"),
        "derived_from": record.get("derived_from"), "reason": None,
    }


def net_profit_margin_cell(year, value, net_income_cell, revenue_cell):
    """
    Net profit margin has no Agent 1 record. Its per-year value is taken from the Agent 2
    comparison, and its sources are those of the two inputs it was calculated from.
    """
    inputs = [{"metric": label, "value": cell["value"], "unit": cell["unit"],
               "source_file": cell["source_file"], "page_number": cell["page_number"]}
              for label, cell in (("Net Income", net_income_cell), ("Revenue", revenue_cell))
              if cell["status"] == "FOUND"]
    return {
        "fiscal_year": year, "status": "FOUND", "value": value, "unit": "percent", "currency": None,
        "basis": "DERIVED", "source_file": inputs[0]["source_file"] if inputs else None,
        "page_number": inputs[0]["page_number"] if inputs else None,
        "evidence": "Calculated by Agent 2 from the Net Income and Revenue values in this table.",
        "confidence": None, "warnings": [], "calculation": NET_PROFIT_MARGIN_FORMULA,
        "derived_from": inputs, "reason": None,
    }


def change_from_comparison(comparison):
    """Agent 2's comparison, copied. Nothing is recalculated here."""
    if not comparison:
        return {"status": "not_calculated", "absolute_change": None, "percentage_change": None,
                "margin_change_points": None, "direction": "unknown", "basis": "CALCULATED",
                "reason": "no comparison was provided by Agent 2"}
    return {"status": comparison.get("status"), "absolute_change": comparison.get("change"),
            "percentage_change": comparison.get("percentage_change"),
            "margin_change_points": comparison.get("margin_change_points"),
            "direction": comparison.get("direction"), "basis": "CALCULATED",
            "reason": comparison.get("reason")}


def eps_label(previous_record, current_record):
    """Say honestly which EPS Agent 1 found (diluted is its preference)."""
    import comparison_agent
    kinds = {comparison_agent.eps_type_of(r) for r in (previous_record, current_record) if is_found(r)}
    if "basic" in kinds:
        return "EPS (basic)"
    return "Diluted EPS" if kinds == {"diluted"} else "EPS (diluted preferred)"


def build_key_financials(pipeline, previous_year, current_year):
    previous = pipeline["agent_1"]["previous"] or {}
    current = pipeline["agent_1"]["current"] or {}
    comparisons = (pipeline["agent_2"]["result"] or {}).get("comparisons") or {}
    cells = {}
    rows = []
    for key, label in REPORT_ROWS:
        comparison = comparisons.get(key)
        if key == "net_profit_margin":
            ni = cells.get(("net_income", "prev")), cells.get(("net_income", "curr"))
            rev = cells.get(("revenue", "prev")), cells.get(("revenue", "curr"))
            if comparison and comparison.get("status") == "calculated":
                prev_cell = net_profit_margin_cell(previous_year, comparison["previous_value"], ni[0], rev[0])
                curr_cell = net_profit_margin_cell(current_year, comparison["current_value"], ni[1], rev[1])
            else:
                reason = (comparison or {}).get("reason") or "not calculated by Agent 2"
                prev_cell, curr_cell = missing_cell(previous_year, reason), missing_cell(current_year, reason)
        else:
            prev_cell = cell_from_record(previous.get(key), previous_year)
            curr_cell = cell_from_record(current.get(key), current_year)
            cells[(key, "prev")], cells[(key, "curr")] = prev_cell, curr_cell
        if key == "eps":
            label = eps_label(previous.get("eps"), current.get("eps"))
        rows.append({"metric_key": key, "metric": label, "is_margin": key in MARGIN_KEYS,
                     "previous": prev_cell, "current": curr_cell,
                     "change": change_from_comparison(comparison)})
    return rows


# ---------------------------------------------------------------------------
# 3. Number and text helpers
# ---------------------------------------------------------------------------

def format_number(value):
    return f"{value:,.2f}".rstrip("0").rstrip(".")


def format_cell_value(cell):
    if cell["status"] != "FOUND":
        return "Insufficient evidence"
    text = format_number(cell["value"])
    if cell["unit"] == "percent":
        return f"{text}%"
    return f"{text} {cell['unit']}".strip() if cell["unit"] else text


def format_change(row):
    change = row["change"]
    if change["status"] != "calculated":
        return "Not available"
    if row["is_margin"]:
        points = change["margin_change_points"]
        return f"{points:+g} percentage points"
    percent = change["percentage_change"]
    return f"{percent:+g}%" if percent is not None else "Not available"


def cell_sources(cell):
    """The sources a cell rests on (both inputs for a derived value)."""
    if cell["status"] != "FOUND":
        return []
    if cell.get("derived_from"):
        return [{"source_file": p.get("source_file"), "page_number": p.get("page_number"),
                 "fiscal_year": cell["fiscal_year"]} for p in cell["derived_from"]]
    return [{"source_file": cell["source_file"], "page_number": cell["page_number"],
             "fiscal_year": cell["fiscal_year"]}]


# ---------------------------------------------------------------------------
# 4. Executive summary
# ---------------------------------------------------------------------------

def movement_statement(row, previous_year, current_year):
    label, change = row["metric"], row["change"]
    prev, curr = row["previous"], row["current"]
    sources = cell_sources(prev) + cell_sources(curr)
    if change["status"] != "calculated":
        return {"text": f"{label} could not be compared between FY{previous_year} and FY{current_year}: "
                        f"{change['reason'] or 'a required value was not found'}.",
                "basis": "LIMITATION", "metric": row["metric_key"], "sources": sources}
    derived = prev["basis"] == "DERIVED" or curr["basis"] == "DERIVED"
    tag = " (derived from reported figures)" if derived else ""
    if row["is_margin"]:
        points = change["margin_change_points"]
        if change["direction"] == "unchanged":
            text = f"{label} was unchanged at {format_cell_value(curr)} in both years{tag}."
        else:
            text = (f"{label} {change['direction']} by {format_number(abs(points))} percentage points, "
                    f"from {format_cell_value(prev)} in FY{previous_year} to {format_cell_value(curr)} "
                    f"in FY{current_year}{tag}.")
    else:
        percent = change["percentage_change"]
        if change["direction"] == "unchanged":
            text = f"{label} was unchanged at {format_cell_value(curr)} in both years."
        elif percent is None:
            text = (f"{label} went from {format_cell_value(prev)} in FY{previous_year} to "
                    f"{format_cell_value(curr)} in FY{current_year}.")
        else:
            text = (f"{label} {change['direction']} by {format_number(abs(percent))}%, from "
                    f"{format_cell_value(prev)} in FY{previous_year} to {format_cell_value(curr)} "
                    f"in FY{current_year}{tag}.")
    return {"text": text, "basis": "CALCULATED", "metric": row["metric_key"], "sources": sources}


def red_flag_statement(flags, evaluated, limitation_count):
    if not evaluated:
        return {"text": "The red-flag screening did not run, so no warning signals are reported.",
                "basis": "LIMITATION", "metric": None, "sources": []}
    sources = [{"source_file": f["source_file"], "page_number": f["page_number"],
                "fiscal_year": None} for f in flags if f.get("source_file")]
    if flags:
        text = (f"Agent 3's screening reported {len(flags)} potential warning signal(s): "
                + " ".join(f["description"] for f in flags))
    else:
        text = ("No configured screening thresholds were crossed for the metrics Agent 3 could evaluate. "
                "This is not an assessment of the company's financial condition.")
    if limitation_count:
        text += f" {limitation_count} screening categor{'y' if limitation_count == 1 else 'ies'} could not be evaluated because of missing data."
    return {"text": text, "basis": "RED_FLAG_SCREEN", "metric": None, "sources": sources}


def build_executive_summary(company, previous_year, current_year, rows, flags, evaluated,
                            red_flag_limit_count, limitations):
    statements = [movement_statement(row, previous_year, current_year) for row in rows]
    statements.append(red_flag_statement(flags, evaluated, red_flag_limit_count))
    if limitations:
        statements.append({"text": f"{len(limitations)} data limitation(s) apply to this report; see the "
                                   f"Limitations section. Missing values are shown as missing, not estimated.",
                           "basis": "LIMITATION", "metric": None, "sources": []})
    opening = (f"This report summarizes {company}'s FY{previous_year} and FY{current_year} results using "
               f"only the indexed annual reports.")
    return {"text": " ".join([opening] + [s["text"] for s in statements]), "statements": statements}


# ---------------------------------------------------------------------------
# 5. Sources and limitations
# ---------------------------------------------------------------------------

def collect_sources(company, rows, flags):
    merged = {}

    def add(file, page, year, used_for):
        if file is None or page is None:
            return
        entry = merged.setdefault((file, page), {"company": company, "fiscal_year": year,
                                                 "source_file": file, "page_number": page, "used_for": []})
        if used_for not in entry["used_for"]:
            entry["used_for"].append(used_for)

    for row in rows:
        for cell in (row["previous"], row["current"]):
            for source in cell_sources(cell):
                add(source["source_file"], source["page_number"], cell["fiscal_year"],
                    f"{row['metric']} FY{cell['fiscal_year']}")
    for flag in flags:
        add(flag.get("source_file"), flag.get("page_number"), None, f"red flag: {flag['type']}")
    return list(merged.values())


def collect_limitations(pipeline, rows, research_result):
    limitations = [{"stage": item.get("stage"), "type": item.get("type"), "message": item.get("message")}
                   for item in pipeline.get("limitations", [])]
    for row in rows:
        for cell in (row["previous"], row["current"]):
            if cell["status"] == "FOUND" and (cell.get("confidence") == "low" or cell.get("warnings")):
                limitations.append({
                    "stage": "agent_1", "type": "low_confidence",
                    "message": (f"{row['metric']} FY{cell['fiscal_year']}: Agent 1 reported "
                                f"{cell.get('confidence') or 'unstated'} confidence"
                                + (f" ({'; '.join(cell['warnings'])})" if cell["warnings"] else "")
                                + ". Treat this figure with caution and check the cited page.")})
    if research_result:
        for item in research_result.get("limitations", []):
            if item.get("type") in ("insufficient_evidence", "no_document"):
                limitations.append({"stage": "research_agent", "type": item["type"], "message": item["message"]})
    seen, unique = set(), []
    for item in limitations:
        if item["message"] not in seen:
            seen.add(item["message"])
            unique.append(item)
    return unique


def collect_evidence_excerpts(research_result):
    """Verbatim excerpts the Research Agent found (never rewritten here)."""
    if not research_result:
        return []
    return [{k: e.get(k) for k in ("id", "company", "fiscal_year", "topic", "statement",
                                   "source_file", "page_number")}
            for e in research_result.get("explanations", []) if e.get("status") == "FOUND"]


# ---------------------------------------------------------------------------
# 6. Main entry point
# ---------------------------------------------------------------------------

def generate_report(company, previous_fiscal_year, current_fiscal_year, *, pipeline_result=None,
                    research_result=None, runner=None):
    """Return a structured report (a dict). Never raises for missing data or a pipeline error."""
    py, cy = str(previous_fiscal_year), str(current_fiscal_year)
    report = {
        "report_title": f"{company} Financial Research Report: FY{py} vs FY{cy}",
        "company": company, "fiscal_years": [py, cy], "status": "error",
        "executive_summary": None, "key_financials": [], "red_flags": [], "red_flag_summary": None,
        "red_flags_evaluated": False, "evidence_excerpts": [], "sources": [], "limitations": [],
        "errors": [], "section_order": list(SECTION_ORDER),
        "metadata": {"generated_from": ["Agent 1", "Agent 2", "Agent 3"]
                     + (["Research Agent"] if research_result else []), "pipeline_status": None},
    }
    try:
        pipeline = collect_report_inputs(company, py, cy, pipeline_result, runner)
    except (Exception, SystemExit) as problem:
        report["errors"].append({"stage": "report_inputs", "message":
                                 f"the analysis pipeline could not be run ({type(problem).__name__})"})
        return report

    report["metadata"]["pipeline_status"] = pipeline.get("status")
    if pipeline.get("status") != "success":
        report["errors"] = [{"stage": e.get("stage"), "message": e.get("message")}
                            for e in pipeline.get("pipeline_errors", [])]
        return report

    rows = build_key_financials(pipeline, py, cy)
    agent3 = pipeline["agent_3"]["result"] or {}
    flags = [dict(f) for f in agent3.get("flags", [])]
    evaluated = pipeline["agent_3"]["status"] == "completed"
    red_flag_limitations = agent3.get("data_limitations", [])
    limitations = collect_limitations(pipeline, rows, research_result)

    report.update({
        "key_financials": rows, "red_flags": flags, "red_flag_summary": agent3.get("summary"),
        "red_flags_evaluated": evaluated,
        "evidence_excerpts": collect_evidence_excerpts(research_result),
        "limitations": limitations,
        "sources": collect_sources(company, rows, flags),
        "executive_summary": build_executive_summary(company, py, cy, rows, flags, evaluated,
                                                     len(red_flag_limitations), limitations),
    })
    report["status"] = "partial" if limitations else "complete"
    return report


# ---------------------------------------------------------------------------
# 7. Formatting
# ---------------------------------------------------------------------------

def citation_lines(report):
    lines = []
    for row in report["key_financials"]:
        for cell in (row["previous"], row["current"]):
            if cell["status"] != "FOUND":
                continue
            where = f"{cell['source_file']}, p. {cell['page_number']}"
            lines.append((row["metric"], cell["fiscal_year"], cell["basis"], where, cell["evidence"],
                          cell.get("calculation"), cell.get("derived_from")))
    return lines


def format_report_markdown(report):
    out = [f"# {report['report_title']}", f"*Status: {report['status']}*", ""]
    if report["status"] == "error":
        out.append("This report could not be generated.")
        out += [f"- {e['message']}" for e in report["errors"]]
        return "\n".join(out)

    out += ["## Executive Summary", report["executive_summary"]["text"], "", "## Key Financials"]
    py, cy = report["fiscal_years"]
    out += [f"| Metric | FY{py} | FY{cy} | Change | Basis |", "|---|---|---|---|---|"]
    for row in report["key_financials"]:
        basis = "Derived" if "DERIVED" in (row["previous"]["basis"], row["current"]["basis"]) else "Reported"
        out.append(f"| {row['metric']} | {format_cell_value(row['previous'])} | "
                   f"{format_cell_value(row['current'])} | {format_change(row)} | {basis} |")
    out += ["", "**Citations**"]
    for metric, year, basis, where, evidence, calculation, derived_from in citation_lines(report):
        line = f"- {metric} FY{year} ({basis.title()}): {where}"
        if evidence:
            line += f' - evidence: "{evidence}"'
        out.append(line)
        if calculation:
            parts = "; ".join(f"{p['metric']} {format_number(p['value'])} ({p['source_file']}, p. {p['page_number']})"
                              for p in derived_from or [])
            out.append(f"  - Derived using: {calculation}" + (f". Inputs: {parts}" if parts else ""))

    out += ["", "## Red Flags / Potential Warning Signals"]
    if not report["red_flags_evaluated"]:
        out.append("The red-flag screening did not run.")
    elif report["red_flags"]:
        for flag in report["red_flags"]:
            src = f" ({flag['source_file']}, p. {flag['page_number']})" if flag.get("source_file") else ""
            out.append(f"- [{flag['severity']}] {flag['description']}{src}")
    else:
        out.append("No configured screening thresholds were crossed for the metrics that could be evaluated. "
                   "This is not an assessment of the company's financial condition.")
    if report["red_flag_summary"]:
        out.append(f"\n{report['red_flag_summary']}")

    if report["evidence_excerpts"]:
        out += ["", "## Evidence Excerpts (verbatim from the annual reports)"]
        out += [f'- {e["topic"]} ({e["company"]} FY{e["fiscal_year"]}): "{e["statement"]}" '
                f'({e["source_file"]}, p. {e["page_number"]})' for e in report["evidence_excerpts"]]

    out += ["", "## Sources"]
    out += [f"- {s['source_file']}, p. {s['page_number']} (used for: {'; '.join(s['used_for'])})"
            for s in report["sources"]] or ["- (none)"]
    out += ["", "## Limitations"]
    out += [f"- {l['message']}" for l in report["limitations"]] or ["- None identified in the available data."]
    out += ["", f"> {DISCLAIMER}"]
    return "\n".join(out)


def format_report_text(report):
    """Plain-text version of the markdown report."""
    return format_report_markdown(report).replace("**", "").replace("## ", "").replace("# ", "")


if __name__ == "__main__":
    import sys
    company_name, previous, current = ("Apple", "2023", "2024") if len(sys.argv) != 4 else sys.argv[1:4]
    outcome = generate_report(company_name, previous, current)
    print(format_report_markdown(outcome))
    sys.exit(0 if outcome["status"] in ("complete", "partial") else 1)