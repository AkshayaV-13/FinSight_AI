"""
chat_interface.py

The conversational layer between a user and the existing agents. It contains no
retrieval, extraction, comparison or financial calculation:

    user question
      -> ChatSession.ask()
           resolve follow-ups from the session context (companies / years / metrics)
           route: report request -> Report Agent
                  anything else  -> Research Agent
                                    (which hands multi-company questions to the Comparison
                                     Agent and uses Agent 1 / Agent 2 for values and changes)
      -> render_answer() / answer_to_markdown()   structured result -> readable markdown
      -> ChatSession.generate_report()            Report Agent (Agents 1-3) on request

Everything the interface displays is built here as markdown, so the web page
(app.py) only shows strings and the behavior can be tested without Streamlit.

Display rules
-------------
* Every value is shown with metric, company, fiscal year, unit and currency, and
  with its source as "<source_file>, p. <page>" plus the stored evidence text.
* Reported values and derived values are labeled differently; a derived value shows
  its formula and the source of each input.
* A missing value is shown as "Insufficient evidence was found in the indexed annual
  reports to answer this part." Never as 0, N/A or an estimate.
* A value Agent 1 flagged as low confidence is shown with a warning.
* A technical failure is logged with its details, and the user sees only a generic
  message. No exception text, class name or stack trace reaches the page.

Session memory is just the last successfully understood companies / fiscal years /
metrics, which is enough for follow-ups such as "What about net income?".
"""

import logging
import re
from datetime import datetime, timezone

import comparison_agent
import report_agent
import research_agent

LOGGER = logging.getLogger("research_chat")

GENERIC_ERROR = "Sorry, I couldn't complete the research request. Please try again."
REPORT_ERROR = "Sorry, I couldn't generate the report. Please try again."
INSUFFICIENT_TEXT = "Insufficient evidence was found in the indexed annual reports to answer this part."
EVIDENCE_PREVIEW = 320

EXAMPLE_QUESTIONS = [
    "Compare Apple's FY2023 and FY2024 revenue, net income, diluted EPS and operating margin. "
    "Explain the changes using only the indexed annual reports.",
    "Compare Apple, Microsoft and Amazon for FY2024 revenue, net income and operating margin.",
    "What financial risks are mentioned in Microsoft's annual report for FY2024?",
]

REPORT_REQUEST = re.compile(r"\b(generate|create|produce|prepare|write|make)\b[^.?!]*\breport\b", re.I)
METRIC_PHRASES = {
    "revenue": "revenue", "net_income": "net income", "eps": "diluted EPS",
    "gross_margin": "gross margin", "operating_margin": "operating margin",
    "operating_income": "operating income", "total_debt": "total debt",
    "cash_flow": "operating cash flow",
}

# Tests (or an embedding application) can set default constructor arguments here.
DEFAULTS = {}


def create_session():
    return ChatSession(**DEFAULTS)


# ---------------------------------------------------------------------------
# Follow-up resolution
# ---------------------------------------------------------------------------

def join_words(items):
    items = list(items)
    if len(items) <= 1:
        return "".join(items)
    return ", ".join(items[:-1]) + " and " + items[-1]


def compose_question(companies, years, metrics, wants_explanation):
    """A self-contained question built from resolved parts (parses back to the same parts)."""
    year_text = join_words(f"FY{y}" for y in years)
    metric_text = join_words(METRIC_PHRASES.get(m, m) for m in metrics)
    if len(companies) > 1:
        text = f"Compare {join_words(companies)} for {year_text} {metric_text}."
    elif len(years) > 1:
        text = f"Compare {companies[0]}'s {year_text} {metric_text}."
    else:
        text = f"What was {companies[0]}'s {year_text} {metric_text}?"
    if wants_explanation:
        text += " Explain the changes using only the indexed annual reports."
    return text


def resolve_question(question, context, catalog):
    """
    Fill in whatever a follow-up leaves out (company, fiscal years, metrics) from the
    previous question. Returns (effective_question, plan, interpreted_note_or_None).
    A question that already names everything is passed through unchanged.
    """
    plan = research_agent.parse_question(question, catalog)
    if not context or plan.get("unsupported_metric"):
        return question, plan, None

    named_metrics = [] if plan["assumptions"] else plan["metrics"]  # assumptions = defaults were guessed
    companies = plan["companies"] or context["companies"]
    years = plan["fiscal_years"] or context["fiscal_years"]
    metrics = named_metrics or context["metrics"]
    inherited = [name for name, new, old in (("company", plan["companies"], context["companies"]),
                                             ("fiscal years", plan["fiscal_years"], context["fiscal_years"]),
                                             ("metrics", named_metrics, context["metrics"]))
                 if not new and old]
    if not inherited or not companies:
        return question, plan, None
    if plan["wants_risks"]:
        effective = (f"{question.strip().rstrip('?.')} for {join_words(companies)}"
                     + (f" {join_words('FY' + y for y in years)}" if years else "") + ".")
    elif not metrics or not years:
        return question, plan, None
    else:
        effective = compose_question(companies, years, metrics, plan["wants_explanation"])
    return effective, research_agent.parse_question(effective, catalog), (
        f"Interpreted as: {effective} (kept from your previous question: {', '.join(inherited)})")


# ---------------------------------------------------------------------------
# Rendering a research result as readable markdown
# ---------------------------------------------------------------------------

def md_cell(text):
    return " ".join(str(text).split()).replace("|", "/")


def citation(item):
    return f"{item['source_file']}, p. {item['page_number']}"


def shorten(text, limit=EVIDENCE_PREVIEW):
    text = " ".join((text or "").split())
    return text if len(text) <= limit else text[:limit].rstrip() + " ..."


def cell_text(finding):
    """One table cell: value with unit, or the insufficient-evidence wording."""
    if finding["status"] != "FOUND":
        return "Insufficient evidence"
    text = report_agent.format_cell_value(finding)
    if finding.get("basis") == "DERIVED":
        text += " (derived)"
    if finding.get("confidence") == "low":
        text += " (low confidence)"
    return text


def build_tables(result):
    plan, findings = result["plan"], result["findings"]
    if not findings:
        return []
    companies, years = plan["companies"], plan["fiscal_years"]
    lookup = {(f["company"], f["fiscal_year"], f["metric"]): f for f in findings}
    metrics = [m for m in plan["metrics"] if any(k[2] == m for k in lookup)]
    label = {f["metric"]: f["metric_label"] for f in findings}
    tables = []
    if len(companies) > 1:
        for year in years:
            rows = [[company] + [cell_text(lookup[(company, year, m)]) if (company, year, m) in lookup
                                 else "Insufficient evidence" for m in metrics] for company in companies]
            tables.append({"title": f"FY{year}", "columns": ["Company"] + [label[m] for m in metrics],
                           "rows": rows})
    else:
        company = companies[0]
        changes = {(c["metric"], c["current_year"]): c for c in result["comparisons"]
                   if c["kind"] == "year_over_year"}
        columns = ["Metric"] + [f"FY{y}" for y in years] + (["Change"] if len(years) > 1 else [])
        rows = []
        for metric in metrics:
            row = [label[metric]] + [cell_text(lookup[(company, y, metric)]) if (company, y, metric) in lookup
                                     else "Insufficient evidence" for y in years]
            if len(years) > 1:
                change = changes.get((metric, years[-1]))
                row.append(report_agent.format_change({"is_margin": change["is_margin"], "change": change})
                           if change else "Not available")
            rows.append(row)
        tables.append({"title": company, "columns": columns, "rows": rows})
    return tables


def finding_lines(finding):
    """The detail block for one finding: value, basis, derivation, source and evidence."""
    head = f"{finding['company']} FY{finding['fiscal_year']} {finding['metric_label']}"
    if finding["status"] != "FOUND":
        return [f"- **{head}**: {INSUFFICIENT_TEXT}"]
    basis = "Derived" if finding["basis"] == "DERIVED" else "Reported"
    currency = f" | Currency: {finding['currency']}" if finding.get("currency") else ""
    statement = f" | Statement: {finding['statement_type']}" if finding.get("statement_type") else ""
    lines = [f"- **{head}**: {report_agent.format_cell_value(finding)}{currency}{statement} - *{basis}*"]
    if finding["basis"] == "DERIVED":
        inputs = "; ".join(f"{p['metric']} {report_agent.format_number(p['value'])} {p['unit']} ({citation(p)})"
                           for p in finding.get("derived_from") or [])
        lines.append(f"  - Derived using: {finding.get('calculation')}. Inputs: {inputs}")
    lines.append(f"  - Source: {citation(finding)}")
    if finding.get("evidence"):
        lines.append(f"  - Evidence: \"{shorten(finding['evidence'])}\"")
    if finding.get("confidence") == "low" or finding.get("warnings"):
        lines.append("  - Warning: this value has low confidence"
                     + (f" ({'; '.join(finding['warnings'])})" if finding.get("warnings") else "")
                     + ". Please check the cited page.")
    return lines


def headline_for(result):
    status = result["status"]
    if status == "success":
        return "Here is what the indexed annual reports show."
    if status == "partial":
        return ("Here is what the indexed annual reports show. Some parts could not be answered "
                "from the indexed documents.")
    if status == "insufficient_evidence":
        for l in result.get("limitations") or []:
            if l.get("type") in ("unsupported_metric", "unindexed_company", "unindexed_fiscal_year"):
                return l.get("message")
        return INSUFFICIENT_TEXT
    if status == "needs_clarification":
        for l in result.get("limitations") or []:
            if l.get("type") in ("unindexed_company", "unindexed_fiscal_year", "no_fiscal_year", "no_company"):
                return l.get("message")
        return "I need a little more detail to answer that."
    return GENERIC_ERROR


def render_answer(result, interpreted=None):
    """Structured research result -> display blocks (no markdown yet)."""
    if result["status"] == "error":
        return {"status": "error", "headline": GENERIC_ERROR, "interpreted": interpreted, "tables": [],
                "findings": [], "changes": [], "comparisons": [], "not_compared": [], "evidence": [],
                "insufficient": [], "limitations": [], "sources": [], "notes": []}
    findings = result["findings"]
    changes = [c["statement"] for c in result["comparisons"] if c["kind"] == "year_over_year"]
    relations, not_compared = [], []
    for c in result["comparisons"]:
        if c["kind"] != "cross_company":
            continue
        relations += [r["statement"] for r in c["relations"]]
        if c["incompatibility"]:
            not_compared.append(f"FY{c['fiscal_year']} {c['metric_label']}: not compared across "
                                f"{', '.join(g for g in c['incompatibility']['groups'])} - "
                                f"{c['incompatibility']['reason']}.")
    evidence, insufficient = [], []
    for e in result["explanations"]:
        if e["status"] == "FOUND":
            evidence.append({"topic": e["topic"], "company": e["company"], "fiscal_year": e["fiscal_year"],
                             "statement": e["statement"], "citation": citation(e)})
        else:
            insufficient.append(f"{e['company']} - {e['topic']}: {INSUFFICIENT_TEXT}")
    technical = bool(result["errors"])
    headline = headline_for(result)
    limitations = [l["message"] for l in result["limitations"]
                   if l["type"] not in ("note", "missing_data", "insufficient_evidence")
                   and l["message"] != headline]
    if technical:
        limitations.append("Some data could not be retrieved because of a technical problem.")
    return {
        "status": result["status"], "headline": headline, "interpreted": interpreted,
        "tables": build_tables(result), "findings": findings, "changes": changes,
        "comparisons": relations, "not_compared": not_compared, "evidence": evidence,
        "insufficient": insufficient, "limitations": limitations,
        "sources": [f"{citation(s)} (used for: {', '.join(s['used_for'])})" for s in result["sources"]],
        "notes": [l["message"] for l in result["limitations"] if l["type"] == "note"],
    }


def answer_to_markdown(answer):
    out = [f"**{answer['headline']}**"]
    if answer["interpreted"]:
        out.append(f"*{answer['interpreted']}*")
    for table in answer["tables"]:
        out += ["", f"**{table['title']}**", "| " + " | ".join(md_cell(c) for c in table["columns"]) + " |",
                "|" + "---|" * len(table["columns"])]
        out += ["| " + " | ".join(md_cell(c) for c in row) + " |" for row in table["rows"]]
    if answer["findings"]:
        out += ["", "**Details and sources**"]
        for finding in answer["findings"]:
            out += finding_lines(finding)
    if answer["changes"]:
        out += ["", "**Changes (calculated from the values above)**"] + [f"- {s}" for s in answer["changes"]]
    if answer["comparisons"] or answer["not_compared"]:
        out += ["", "**Company comparison (calculated from the values above)**"]
        out += [f"- {s}" for s in answer["comparisons"] + answer["not_compared"]]
    if answer["evidence"]:
        out += ["", "**Evidence from the annual reports (verbatim)**"]
        out += [f"- {e['topic']} ({e['company']} FY{e['fiscal_year']}): \"{shorten(e['statement'])}\" - Source: {e['citation']}"
                for e in answer["evidence"]]
    if answer["insufficient"]:
        out += ["", "**Not answerable from the indexed documents**"] + [f"- {s}" for s in answer["insufficient"]]
    if answer["limitations"] or answer["notes"]:
        out += ["", "**Limitations**"] + [f"- {s}" for s in answer["limitations"] + answer["notes"]]
    if answer["sources"]:
        out += ["", "**Sources**"] + [f"- {s}" for s in answer["sources"]]
    out += ["", "*Descriptive research from the indexed annual reports only; not investment advice.*"]
    return "\n".join(out)


# ---------------------------------------------------------------------------
# The session
# ---------------------------------------------------------------------------

class ChatSession:
    """One user's conversation. Holds history and the last understood context only."""

    def __init__(self, research_fn=None, report_fn=None, catalog_loader=None, logger=None):
        self.research_fn = research_fn or research_agent.run_research
        self.report_fn = report_fn or report_agent.generate_report
        self.catalog_loader = catalog_loader or comparison_agent.load_index_catalog
        self.logger = logger or LOGGER
        self.history = []
        self.context = None
        self.last_research = None
        self._catalog = None
        self._catalog_tried = False

    # -- housekeeping ------------------------------------------------------
    def reset(self):
        self.history, self.context, self.last_research = [], None, None

    def _add(self, role, markdown, **extra):
        turn = {"role": role, "markdown": markdown, "content": markdown,
                "timestamp": datetime.now(timezone.utc).isoformat(timespec="seconds"),
                "kind": extra.pop("kind", "message"), "route": None, "answer": None,
                "interpreted": None, "report_available": None, **extra}
        self.history.append(turn)
        return turn

    def _catalog_for_research(self):
        """Read the index catalog once per session; a failure is logged, not shown."""
        if not self._catalog_tried:
            self._catalog_tried = True
            try:
                self._catalog = self.catalog_loader()
            except (Exception, SystemExit):
                self.logger.error("index catalog could not be loaded", exc_info=True)
        return self._catalog

    # -- main entry --------------------------------------------------------
    def ask(self, question):
        question = (question or "").strip()
        if not question:
            return self._add("assistant", "Please type a question about the indexed annual reports.",
                             kind="message")
        self._add("user", question, kind="question")

        catalog = self._catalog_for_research()
        effective, plan, interpreted = resolve_question(question, self.context, catalog)

        if REPORT_REQUEST.search(question):
            return self._handle_report_request(plan)

        route = "research+comparison" if len(plan["companies"]) >= 2 else "research"
        try:
            kwargs = {"catalog": catalog} if catalog is not None else {}
            result = self.research_fn(effective, **kwargs)
        except (Exception, SystemExit):
            self.logger.error("research agent failed for question %r", question, exc_info=True)
            return self._add("assistant", GENERIC_ERROR, kind="error", route=route)

        if result.get("errors"):
            self.logger.error("research agent reported technical errors: %s", result["errors"])
        answer = render_answer(result, interpreted)
        kind = "error" if result["status"] == "error" else "research"
        turn = self._add("assistant", answer_to_markdown(answer), kind=kind, route=route,
                         answer=answer, interpreted=interpreted)
        if result["status"] != "error":
            self.last_research = result
            if result["status"] != "needs_clarification":
                self.context = {"companies": list(plan["companies"]),
                                "fiscal_years": list(plan["fiscal_years"]), "metrics": list(plan["metrics"])}
            if len(plan["companies"]) == 1 and len(plan["fiscal_years"]) >= 2:
                turn["report_available"] = {"company": plan["companies"][0],
                                            "previous_year": plan["fiscal_years"][0],
                                            "current_year": plan["fiscal_years"][-1]}
        return turn

    # -- reports -----------------------------------------------------------
    def _handle_report_request(self, plan):
        context = self.context or {}
        companies = plan["companies"] or context.get("companies", [])
        years = plan["fiscal_years"] or context.get("fiscal_years", [])
        if len(companies) != 1:
            return self._add("assistant", "Reports cover one company at a time. Please name a single company, "
                                          "for example: \"Generate a report for Apple FY2023 vs FY2024\".",
                             kind="message", route="report")
        if len(years) < 2:
            return self._add("assistant", "A report compares two fiscal years. Please name both years, for "
                                          "example FY2023 and FY2024.", kind="message", route="report")
        return self.generate_report(companies[0], years[0], years[-1])

    def generate_report(self, company=None, previous_year=None, current_year=None):
        context = self.context or {}
        company = company or (context.get("companies") or [None])[0]
        years = context.get("fiscal_years") or []
        previous_year = previous_year or (years[0] if len(years) >= 2 else None)
        current_year = current_year or (years[-1] if len(years) >= 2 else None)
        if not (company and previous_year and current_year):
            return self._add("assistant", "Please ask about one company and two fiscal years first.",
                             kind="message", route="report")
        try:
            report = self.report_fn(company, previous_year, current_year, research_result=self.last_research)
        except (Exception, SystemExit):
            self.logger.error("report agent failed for %s %s-%s", company, previous_year, current_year,
                              exc_info=True)
            return self._add("assistant", REPORT_ERROR, kind="error", route="report")
        if report.get("status") == "error":
            self.logger.error("report could not be generated: %s", report.get("errors"))
            return self._add("assistant", REPORT_ERROR, kind="error", route="report")
        return self._add("assistant", report_agent.format_report_markdown(report), kind="report",
                         route="report", answer=report)