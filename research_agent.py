"""
research_agent.py

Research Agent: answers multi-part financial research questions using ONLY the
indexed annual reports.

    question
      -> parse_question()   companies, fiscal years, metrics, intent (deterministic rules)
      -> decompose()        a list of sub-questions
      -> Agent 1            extraction_agent.extract_metrics()  -> REPORTED / DERIVED values
      -> Agent 2            analysis_agent.compare_metric()     -> year-over-year changes
      -> Comparison Agent   comparison_agent.build_comparison() -> cross-company comparison
      -> evidence search    sentences quoted verbatim from the existing ChromaDB
      -> ONE structured result  (format_research_report() turns it into text)

Reuse, not duplication
----------------------
Values come from Agent 1 and calculations from Agent 2; nothing is extracted or
calculated here. Retrieval goes through extraction_agent.retrieve_chunks_multi,
which uses Agent 1's single cached embedding model and collection connection, so
a multi-part question loads the model once, not once per sub-question.

Grounding rules
---------------
* Every finding keeps company, fiscal_year, metric, value, unit, currency,
  source_file, page_number and evidence.
* REPORTED = a value Agent 1 read directly from a report. DERIVED = a value
  calculated from reported values (for example operating margin = operating
  income / revenue * 100); it keeps the formula and BOTH input sources. Changes
  between years are always labeled DERIVED (calculated), with the findings they
  came from.
* A missing value is NOT_FOUND with a stated reason. It is never filled with
  zero, an estimate or outside knowledge.
* Explanations are verbatim excerpts from the indexed reports, each with its
  source page. This agent never writes a business reason of its own. If no
  retrieved sentence both discusses the metric and gives an explanatory cue
  (for example "due to", "primarily"), the explanation is reported as
  INSUFFICIENT_EVIDENCE.
* No web search, external API, LLM, advice, ranking or prediction.

Question understanding is rule-based, so unusual phrasing can be missed. Whatever
the agent assumed (for example default metrics when none were named) is listed
in plan["assumptions"], and anything it could not resolve is a limitation.

Usage:
    python research_agent.py "Compare Apple's FY2023 and FY2024 revenue, net income, diluted EPS and operating margin. Explain the changes using only the indexed annual reports."
"""

import logging
import re
import sys

import analysis_agent
import comparison_agent
from comparison_agent import METRIC_LABELS, company_key, make_view, normalized_record_copy, pick_stored_name

LOGGER = logging.getLogger("research_agent")

DEFAULT_RESEARCH_METRICS = ["revenue", "net_income", "eps", "operating_margin"]
DEFAULT_COMPANY_NAMES = ["Apple", "Microsoft", "Tesla", "Amazon", "Reliance Industries", "Infosys", "TCS"]

METRIC_PATTERNS = [
    ("operating_margin", r"operating\s+margins?"),
    ("gross_margin", r"gross\s+margins?"),
    ("net_margin", r"net\s+(?:profit\s+)?margins?"),
    ("operating_income", r"operating\s+income|income\s+from\s+operations|operating\s+profit"),
    ("profit_before_tax", r"profit\s+before\s+tax(?:es)?|profit\s+before\s+income\s+tax\b|pbt\b|income\s+before\s+(?:provision\s+for\s+)?income\s+taxes|earnings\s+before\s+(?:income\s+)?taxes"),
    ("profit_after_tax", r"profit\s+after\s+tax\b|pat\b"),
    ("net_income", r"net\s+income|net\s+profit|net\s+earnings"),
    ("ebitda", r"\bebitda\b|earnings\s+before\s+interest,?\s+tax(?:es)?,?\s+depreciation"),
    ("total_income", r"total\s+income|total\s+revenue\s+and\s+other\s+income"),
    ("eps", r"(?:diluted\s+|basic\s+)?(?:eps\b|earnings\s+per\s+(?:equity\s+)?share)"),
    ("total_assets", r"total\s+assets?"),
    ("total_liabilities", r"total\s+liabilities?"),
    ("total_equity", r"total\s+(?:shareholders'?|stockholders'?\s+)?equity|shareholders'?\s+equity|stockholders'?\s+equity|total\s+equity"),
    ("cash_and_equivalents", r"cash\s+and\s+cash\s+equivalents?|cash\s+equivalents?"),
    ("investing_cash_flow", r"investing\s+cash\s+flow|cash\s+flow(?:s)?\s+from\s+investing|cash\s+(?:used\s+in|provided\s+by)\s+investing"),
    ("financing_cash_flow", r"financing\s+cash\s+flow|cash\s+flow(?:s)?\s+from\s+financing|cash\s+(?:used\s+in|provided\s+by)\s+financing"),
    ("gross_profit", r"gross\s+profit|gross\s+income"),
    ("revenue", r"revenues?|revenue\s+from\s+operations|net\s+sales|total\s+sales|(?<!\binventory\s)(?<!\basset\s)(?<!\breceivable\s)(?<!\breceivables\s)(?<!\baccounts\sreceivable\s)(?<!\bcapital\s)\bturnover\b"),
    ("total_debt", r"total\s+debt|borrowings|\bdebt\b"),
    ("cash_flow", r"operating\s+cash\s+flow|cash\s+flow(?:s)?(?:\s+from\s+operati\w+)?|cash\s+generated\s+by\s+operati\w+"),
]
FISCAL_YEAR_PATTERN = re.compile(r"\b(?:fy\s?)?((?:19|20)\d{2})\b|\bfy\s?(\d{2})\b", re.IGNORECASE)
EXPLAIN_PATTERN = re.compile(r"\b(explain|why|reasons?|drivers?|drove|factors?|caus(?:e|ed|es)|due to|"
                             r"attribut\w+|what affected|what influenced|contribut\w+)\b", re.I)
CHANGE_PATTERN = re.compile(r"\b(compare|comparison|change[sd]?|increase[sd]?|decrease[sd]?|grew|growth|"
                            r"declin\w+|trend\w*|versus|vs\.?|difference|between)\b", re.I)
RISK_PATTERN = re.compile(r"\b(risks?|uncertaint\w+|exposures?|threats?)\b", re.I)
UNSUPPORTED_METRIC_PATTERN = re.compile(
    r"\b(inventory\s+turnover|asset\s+turnover|receivables?\s+turnover|turnover\s+ratio|"
    r"working\s+capital|current\s+ratio|quick\s+ratio|debt\s+to\s+equity|roe|roa|roce|"
    r"return\s+on\s+equity|return\s+on\s+assets|return\s+on\s+capital|ebit\b|"
    r"free\s+cash\s+flow|fcf\b|capex|capital\s+expenditures?|goodwill|intangible\s+assets|"
    r"market\s+cap\w*|pe\s+ratio|p/e\s+ratio|price\s+to\s+earnings|dividend\s+yield|"
    r"book\s+value|shares?\s+outstanding|beta)\b",
    re.I
)

EXPLANATION_CUES = ["due to", "driven by", "primarily", "attributable", "as a result", "because",
                    "resulted from", "result of", "partially offset", "higher", "lower", "increase",
                    "decrease", "declined", "growth", "favorable", "unfavorable", "compared to",
                    "compared with", "impact"]
METRIC_TOPICS = {
    "revenue": ["net sales", "revenue"],
    "net_income": ["net income", "income tax", "provision for income taxes"],
    "eps": ["earnings per share", "per share"],
    "gross_margin": ["gross margin"],
    "operating_margin": ["operating income", "operating margin", "operating expenses", "gross margin"],
    "operating_income": ["operating income", "operating expenses"],
    "total_debt": ["debt", "borrowings", "commercial paper"],
    "cash_flow": ["cash generated", "operating activities", "cash flow"],
    "total_income": ["total income", "revenue from operations", "other income"],
    "profit_before_tax": ["profit before tax", "income before taxes", "provision for income taxes"],
    "profit_after_tax": ["profit after tax", "net profit", "profit for the year"],
    "ebitda": ["ebitda", "depreciation", "amortization"],
    "total_assets": ["total assets", "assets", "property, plant and equipment"],
    "total_liabilities": ["total liabilities", "liabilities"],
    "total_equity": ["total equity", "shareholders' equity", "stockholders' equity"],
    "cash_and_equivalents": ["cash and cash equivalents", "cash and cash balances"],
    "investing_cash_flow": ["investing activities", "capital expenditures"],
    "financing_cash_flow": ["financing activities", "dividends"],
    "gross_profit": ["gross profit", "cost of sales", "net sales"],
    "net_margin": ["net profit margin", "net margin"],
}
RISK_TOPIC_TERMS = ["risk", "adverse", "adversely", "uncertain"]
RISK_CUES = ["could", "may", "adversely", "might", "subject to", "exposure", "harm"]
SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z$\"'(])")


# ---------------------------------------------------------------------------
# 1. Question understanding
# ---------------------------------------------------------------------------

def parse_question(question, catalog=None):
    """
    Identify companies, fiscal years, metrics and intent in a natural-language
    question. Company names are matched against the index catalog when one is
    available (so what is searched is what is actually indexed), otherwise
    against the project's company list.
    """
    text = question or ""
    tokens = re.findall(r"[a-z0-9]+", text.lower())
    token_set = set(tokens)
    assumptions, notes = [], []

    if catalog:
        known = {}
        for stored in {name for (name, _year) in catalog["counts"]}:
            known.setdefault(company_key(stored), stored)
    else:
        known = {company_key(name): name for name in DEFAULT_COMPANY_NAMES}

    found = []
    for key, name in known.items():
        if key and key <= token_set:
            position = min(tokens.index(t) for t in key)
            typed = " ".join(m.group(0) for t in sorted(key, key=tokens.index)
                             for m in [re.search(rf"\b{re.escape(t)}\b", text, re.I)] if m)
            found.append((position, typed or name))

    # Also detect aliases and unindexed companies/entities
    for alias, canonical in comparison_agent.COMPANY_ALIASES.items():
        alias_toks = set(re.findall(r"[a-z0-9]+", alias.lower()))
        if alias_toks and alias_toks <= token_set:
            if not any(company_key(name) == company_key(canonical) for _pos, name in found):
                pos = min(tokens.index(t) for t in alias_toks)
                found.append((pos, canonical))

    if not found:
        # Heuristic entity detection for unindexed companies (e.g. "Nestle", "Google", "Tata Motors")
        m = re.search(r"\b(?:for|of|in|at|about)\s+([A-Z][a-zA-Z0-9&.\-]+(?:\s+[A-Z][a-zA-Z0-9&.\-]+)*)\b", text)
        if not m:
            m = re.search(r"\b([A-Z][a-zA-Z0-9&.\-]+(?:\s+[A-Z][a-zA-Z0-9&.\-]+)*)'s\b", text)
        if m:
            cand = m.group(1).strip()
            non_company = {"fy", "fiscal", "year", "annual", "report", "the", "what", "how", "why", "q1", "q2", "q3", "q4"}
            if cand.lower() not in non_company and not any(p[0] in cand.lower() for p in METRIC_PATTERNS):
                found.append((m.start(), cand))

    companies = [name for _pos, name in sorted(found)]

    years = []
    for match in FISCAL_YEAR_PATTERN.finditer(text):
        year = match.group(1) or f"20{match.group(2)}"
        if year not in years:
            years.append(year)
    years.sort()

    spans = []
    for key, pattern in METRIC_PATTERNS:
        for match in re.finditer(pattern, text, re.I):
            spans.append((match.start(), match.end(), key))
    spans.sort(key=lambda s: (s[0], -(s[1] - s[0])))
    metrics, last_end = [], -1
    for start, end, key in spans:
        if start >= last_end and key not in metrics:
            metrics.append(key)
        last_end = max(last_end, end)

    has_unsupported = bool(UNSUPPORTED_METRIC_PATTERN.search(text))

    wants_risks = bool(RISK_PATTERN.search(text))
    is_specific_metric_query = (
        bool(re.search(r"\b(?:what\s+(?:was|is|were)|how\s+much\b|find\b|show\s+me\b|tell\s+me\s+about)\b", text, re.I))
        and not bool(re.search(r"\b(compare|comparison|overview|summary|perform\w*|financial\s+results)\b", text, re.I))
        and len(companies) <= 1
        and len(years) <= 1
    )
    unsupported_metric = False

    if has_unsupported:
        metrics = []
        unsupported_metric = True
    elif not metrics and not wants_risks:
        if is_specific_metric_query:
            metrics = []
            unsupported_metric = True
        else:
            metrics = list(DEFAULT_RESEARCH_METRICS)
            assumptions.append("No specific metric was named, so the default core metrics were used: "
                               + ", ".join(METRIC_LABELS[m] for m in metrics))
    if "eps" in metrics and re.search(r"\bbasic\b", text, re.I):
        notes.append("basic EPS requested: Agent 1 extracts diluted EPS when available and does not "
                     "report basic EPS separately")

    return {
        "companies": companies, "fiscal_years": years, "metrics": metrics,
        "wants_explanation": bool(EXPLAIN_PATTERN.search(text)),
        "wants_change": len(years) >= 2 and bool(CHANGE_PATTERN.search(text)),
        "wants_comparison": len(companies) >= 2,
        "wants_risks": wants_risks, "assumptions": assumptions, "notes": notes,
        "unsupported_metric": unsupported_metric,
        "wants_standalone": bool(re.search(r"\bstandalone\b", text, re.I)),
        "wants_consolidated": bool(re.search(r"\bconsolidated\b", text, re.I)),
    }


def decompose(plan):
    """Split a parsed question into sub-questions (data only; nothing is retrieved here)."""
    subs = []

    def add(kind, text, **fields):
        subs.append({"id": f"Q{len(subs) + 1}", "type": kind, "text": text, "status": "pending",
                     "finding_ids": [], "comparison_ids": [], "explanation_ids": [], **fields})

    companies, years, metrics = plan["companies"], plan["fiscal_years"], plan["metrics"]
    for company in companies:
        for metric in metrics:
            for year in years:
                add("metric_value", f"What was {company}'s FY{year} {METRIC_LABELS[metric]}?",
                    company=company, fiscal_year=year, metric=metric)
    if plan["wants_change"]:
        for company in companies:
            for metric in metrics:
                add("metric_change", f"How did {company}'s {METRIC_LABELS[metric]} change from "
                    f"FY{years[0]} to FY{years[-1]}?", company=company, fiscal_year=years[-1], metric=metric)
    if plan["wants_comparison"]:
        for year in years:
            for metric in metrics:
                add("company_comparison", f"How do {', '.join(companies)} compare on FY{year} "
                    f"{METRIC_LABELS[metric]}?", company=None, fiscal_year=year, metric=metric)
    if plan["wants_explanation"]:
        for company in companies:
            for metric in metrics:
                add("explanation", f"What evidence in the {company} "
                    f"{'FY' + years[-1] + ' ' if years else ''}annual report explains the change in "
                    f"{METRIC_LABELS[metric]}?", company=company,
                    fiscal_year=years[-1] if years else None, metric=metric)
    if plan["wants_risks"]:
        for company in companies:
            add("risk", f"What financial risks does the {company} annual report describe?",
                company=company, fiscal_year=years[-1] if years else None, metric=None)
    return subs


# ---------------------------------------------------------------------------
# 2. Evidence: verbatim sentences from the indexed reports
# ---------------------------------------------------------------------------

def split_sentences(text):
    return [s.strip() for s in SENTENCE_SPLIT.split(" ".join((text or "").split())) if s.strip()]


def score_sentence(sentence, topic_terms, cue_terms):
    """Return (score, topics, cues) or None if the sentence is not usable evidence."""
    lower = sentence.lower()
    if len(sentence) < 40 or len(sentence) > 600:
        return None
    digits = sum(ch.isdigit() for ch in sentence)
    if digits / max(1, len(sentence)) > 0.25:
        return None  # looks like a table row, not an explanation
    topics = [t for t in topic_terms if t in lower]
    cues = [c for c in cue_terms if c in lower]
    if not topics or not cues:
        return None
    return len(topics) * 2 + len(cues), topics, cues


def find_evidence(chunks, topic_terms, cue_terms, limit):
    """Best verbatim sentences across retrieved chunks, each with its own source."""
    scored, seen = [], set()
    for chunk in chunks:
        meta = chunk["metadata"]
        for sentence in split_sentences(chunk["text"]):
            result = score_sentence(sentence, topic_terms, cue_terms)
            if result is None or sentence in seen:
                continue
            seen.add(sentence)
            score, topics, cues = result
            scored.append((score + chunk.get("similarity", 0), {
                "statement": sentence, "source_file": meta.get("source_file"),
                "page_number": meta.get("page_number"), "company": meta.get("company"),
                "fiscal_year": str(meta.get("fiscal_year")), "similarity": chunk.get("similarity"),
                "matched_topic_terms": topics, "matched_cue_terms": cues}))
    scored.sort(key=lambda item: -item[0])
    return [item for _score, item in scored[:limit]]


# ---------------------------------------------------------------------------
# 3. Wording of calculated changes (neutral, no advice)
# ---------------------------------------------------------------------------

def describe_change(change, company, label):
    years = f"FY{change['previous_year']} to FY{change['current_year']}"
    span = f"FY{change['previous_year']} and FY{change['current_year']}"
    if change["status"] != "calculated":
        return f"{company} {label}: change not calculated ({change['reason']})."
    if change["is_margin"]:
        points = change["margin_change_points"]
        if points == 0:
            return f"{company}'s {label} was unchanged from {years} at {change['current_value']}%."
        return (f"{company}'s {label} {change['direction']} by {abs(points)} percentage points from "
                f"{change['previous_value']}% to {change['current_value']}% between {span}.")
    unit = f" {change['unit']}" if change.get("unit") else ""
    percent = change["percentage_change"]
    if percent is None or change["direction"] == "unchanged":
        return (f"{company}'s {label} was unchanged from {years} at "
                f"{change['current_value']:,}{unit}." if change["direction"] == "unchanged" else
                f"{company}'s {label} went from {change['previous_value']:,}{unit} to "
                f"{change['current_value']:,}{unit} between {span}.")
    return (f"{company}'s {label} {change['direction']} by {abs(percent)}% from "
            f"{change['previous_value']:,}{unit} to {change['current_value']:,}{unit} between {span}.")


# ---------------------------------------------------------------------------
# 4. Main entry point
# ---------------------------------------------------------------------------

def run_research(question, *, extractor=None, retriever=None, catalog=None, use_catalog=True,
                 top_k=3, max_evidence=3):
    """
    Answer a research question from the indexed reports. Returns structured data:
    status, query, plan, subquestions, findings, comparisons, explanations, sources,
    limitations, errors, pipeline_metadata. Never raises for missing data or an
    agent failure; those appear in `limitations` / `errors`.

    extractor : callable(company, fiscal_year, metric_keys=..., top_k=...) (default Agent 1)
    retriever : callable(queries, where=..., top_k=...) (default Agent 1's retrieval)
    catalog   : index catalog; read from the existing collection when None and use_catalog
    """
    result = {"status": "success", "query": question, "plan": None, "subquestions": [],
              "findings": [], "comparisons": [], "explanations": [], "sources": [],
              "limitations": [], "errors": [],
              "pipeline_metadata": {"agent_1_calls": 0, "retrieval_calls": 0,
                                    "reused": {"agent_1": True, "agent_2": True, "comparison_agent": True}}}
    meta = result["pipeline_metadata"]

    def limit(kind, message, **fields):
        result["limitations"].append({"type": kind, "message": message, **fields})

    if catalog is None and use_catalog:
        try:
            catalog = comparison_agent.load_index_catalog()
        except (Exception, SystemExit) as problem:
            result["errors"].append({"stage": "index_catalog", "error_type":
                                     "system_exit" if isinstance(problem, SystemExit) else "exception",
                                     "message": f"the index could not be read: {type(problem).__name__}"})
            result["status"] = "error"
            return result

    plan = parse_question(question, catalog)
    result["plan"] = plan
    LOGGER.info("run_research: question=%r | plan companies=%s, years=%s, metrics=%s",
                question, plan["companies"], plan["fiscal_years"], plan["metrics"])
    for note in plan["notes"]:
        limit("note", note)

    available_companies = sorted({name for (name, _y) in catalog["counts"]}) if catalog else DEFAULT_COMPANY_NAMES

    if not plan["companies"]:
        limit("no_company", f"No company matching the indexed documents was found in the question. "
                            f"Currently indexed companies are: {', '.join(available_companies)}.")
        result["status"] = "needs_clarification"
        return result

    unindexed_in_query = [c for c in plan["companies"]
                          if not any(company_key(c) == company_key(stored) for stored in available_companies)]
    if unindexed_in_query:
        bad_co = unindexed_in_query[0]
        limit("unindexed_company",
              f"Company '{bad_co}' is not available in the indexed annual reports. "
              f"Currently indexed companies are: {', '.join(available_companies)}.",
              company=bad_co)
        result["status"] = "needs_clarification"
        return result

    if plan.get("unsupported_metric") or (not plan["metrics"] and not plan["wants_risks"]):
        limit("unsupported_metric", "Insufficient evidence: the requested financial metric could not be extracted from the annual reports. "
                                    "FinSight AI extracts revenue, net income, EPS, gross margin, operating margin, "
                                    "operating income, total debt, cash flow, total income, profit before tax, "
                                    "profit after tax, EBITDA, total assets, total liabilities, total equity, "
                                    "cash and cash equivalents, and gross profit.")
        result["status"] = "insufficient_evidence"
        return result
    if plan["metrics"] and not plan["fiscal_years"]:
        co = plan["companies"][0] if plan["companies"] else None
        avail = sorted({yr for (name, yr) in catalog["counts"]
                        if company_key(name) == company_key(co)}) if catalog and co else []
        year_hint = f" Available fiscal years for {co}: {', '.join('FY' + y for y in avail)}." if avail else ""
        limit("no_fiscal_year", f"No fiscal year was specified{' for ' + co if co else ''}. "
                                f"A fiscal year is required to look up metric values.{year_hint}")
        result["status"] = "needs_clarification"
        return result
    if plan["wants_change"] is False and len(plan["fiscal_years"]) == 1 and CHANGE_PATTERN.search(question or ""):
        limit("single_year", f"Only FY{plan['fiscal_years'][0]} was named, so no year-over-year change "
                             f"was calculated; the report's own discussion is searched instead.")

    subs = decompose(plan)
    result["subquestions"] = subs
    companies, years, metrics = plan["companies"], plan["fiscal_years"], plan["metrics"]

    # ---- Agent 1: values -------------------------------------------------
    if extractor is None and metrics:
        import extraction_agent
        extractor = extraction_agent.extract_metrics

    raw, stored_for, findings = {}, {}, {}
    for company in companies:
        for year in years if metrics else []:
            stored, reason = (company, None)
            if catalog:
                stored, reason = pick_stored_name(company, year, catalog)
            if stored is None:
                if reason == "company_not_indexed":
                    message = f"Company '{company}' is not available in the indexed annual reports. Currently indexed companies are: {', '.join(available_companies)}."
                    limit("unindexed_company", message, company=company, fiscal_year=year)
                elif reason == "fiscal_year_not_indexed":
                    available_years = sorted({yr for (name, yr) in catalog["counts"]
                                              if company_key(name) == company_key(company)}) if catalog else []
                    message = f"{company} FY{year} is not available in the indexed annual reports. Available fiscal years for {company}: {', '.join('FY' + y for y in available_years)}."
                    limit("unindexed_fiscal_year", message, company=company, fiscal_year=year)
                else:
                    message = f"no indexed document for FY{year}"
                    limit(reason or "missing_document", f"{company}: {message}", company=company, fiscal_year=year)
                raw[(company, year)] = {}
                stored_for[(company, year)] = company
                for metric in metrics:
                    findings[(company, year, metric)] = make_view(company, year, metric, None, missing_reason=message)
                continue
            stored_for[(company, year)] = stored
            try:
                meta["agent_1_calls"] += 1
                raw[(company, year)] = extractor(stored, year, metric_keys=metrics, top_k=top_k)
            except (Exception, SystemExit) as problem:
                result["errors"].append({
                    "stage": "agent_1", "company": company, "fiscal_year": year,
                    "error_type": "system_exit" if isinstance(problem, SystemExit) else "exception",
                    "message": f"Agent 1 failed for {company} FY{year}: {type(problem).__name__}: {problem}"})
                raw[(company, year)] = {}
                for metric in metrics:
                    findings[(company, year, metric)] = make_view(
                        company, year, metric, None, missing_reason="Agent 1 failed for this company and year")
                continue
            for metric in metrics:
                findings[(company, year, metric)] = make_view(
                    company, year, metric, (raw[(company, year)] or {}).get(metric), indexed_name=stored)

    finding_ids = {}
    for sub in subs:
        if sub["type"] != "metric_value":
            continue
        key = (sub["company"], sub["fiscal_year"], sub["metric"])
        if key not in finding_ids:
            view = dict(findings[key])
            view["id"] = f"F{len(result['findings']) + 1}"
            view["subquestion_ids"] = []
            result["findings"].append(view)
            finding_ids[key] = view["id"]
            if view["status"] != "FOUND":
                limit("missing_data", f"{key[0]} FY{key[1]} {view['metric_label']}: NOT_FOUND "
                                      f"({view['reason']}). Insufficient evidence; no value was estimated.",
                      company=key[0], fiscal_year=key[1], metric=key[2])
            elif view["metric"] == "eps" and view["eps_type"] == "basic":
                limit("eps_basic", f"{key[0]} FY{key[1]}: Agent 1 only found a basic EPS figure.",
                      company=key[0], fiscal_year=key[1], metric="eps")
        sub["finding_ids"].append(finding_ids[key])
        next(f for f in result["findings"] if f["id"] == finding_ids[key])["subquestion_ids"].append(sub["id"])
        sub["status"] = "answered" if findings[key]["status"] == "FOUND" else "insufficient_evidence"

    # ---- Agent 2: year-over-year changes --------------------------------
    change_ids = {}
    for sub in subs:
        if sub["type"] != "metric_change":
            continue
        company, metric = sub["company"], sub["metric"]
        sub["status"] = "answered"
        for previous, current in zip(years, years[1:]):
            prev_raw = (raw.get((company, previous)) or {}).get(metric)
            curr_raw = (raw.get((company, current)) or {}).get(metric)
            change = analysis_agent.compare_metric(
                metric, normalized_record_copy(prev_raw) if prev_raw else None,
                normalized_record_copy(curr_raw) if curr_raw else None)
            label = METRIC_LABELS[metric]
            inputs = [finding_ids.get((company, previous, metric)), finding_ids.get((company, current, metric))]
            entry = {
                "id": f"C{len(result['comparisons']) + 1}", "kind": "year_over_year", "basis": "DERIVED",
                "company": company, "metric": metric, "metric_label": label,
                "previous_year": previous, "current_year": current, "status": change["status"],
                "previous_value": change["previous_value"], "current_value": change["current_value"],
                "unit": change["unit"], "change": change["change"],
                "percentage_change": change["percentage_change"],
                "margin_change_points": change["margin_change_points"],
                "direction": change["direction"], "is_margin": change["is_margin"],
                "reason": change["reason"], "input_finding_ids": [i for i in inputs if i],
                "calculation": ("difference in percentage points = current - previous" if change["is_margin"]
                                else "percentage change = (current - previous) / previous * 100"),
                "previous_source": comparison_agent._source_of(findings[(company, previous, metric)]),
                "current_source": comparison_agent._source_of(findings[(company, current, metric)]),
            }
            entry["statement"] = describe_change(
                {**change, "previous_year": previous, "current_year": current}, company, label)
            result["comparisons"].append(entry)
            sub["comparison_ids"].append(entry["id"])
            if change["status"] != "calculated":
                sub["status"] = "insufficient_evidence"
                limit("change_not_calculated", entry["statement"], company=company, metric=metric)
            change_ids[(company, metric, previous, current)] = entry["id"]

    # ---- Comparison Agent: cross-company --------------------------------
    if plan["wants_comparison"] and metrics:
        cross = comparison_agent.build_comparison(
            companies, years, metrics, records=raw, indexed_names=stored_for, resolve_names=False)
        for comparison in cross["comparisons"]:
            entry = {"id": f"C{len(result['comparisons']) + 1}", "kind": "cross_company", **comparison}
            result["comparisons"].append(entry)
            for sub in subs:
                if (sub["type"] == "company_comparison" and sub["fiscal_year"] == comparison["fiscal_year"]
                        and sub["metric"] == comparison["metric"]):
                    sub["comparison_ids"].append(entry["id"])
                    sub["status"] = "answered" if comparison["relations"] else "not_comparable"
        for item in cross["limitations"]:
            if item["type"] == "incompatible_values":
                limit("incompatible_values", item["message"], fiscal_year=item["fiscal_year"], metric=item["metric"])
        for note in cross["notes"]:
            limit("note", note)

    # ---- Evidence-based explanations ------------------------------------
    if retriever is None and any(s["type"] in ("explanation", "risk") for s in subs):
        import extraction_agent
        retriever = extraction_agent.retrieve_chunks_multi
    if any(s["type"] in ("explanation", "risk") for s in subs):
        import extraction_agent
        for sub in subs:
            if sub["type"] not in ("explanation", "risk"):
                continue
            company, year, metric = sub["company"], sub["fiscal_year"], sub["metric"]
            if catalog:
                stored, reason = (pick_stored_name(company, year, catalog) if year
                                  else (_any_year_name(company, catalog), None))
            else:
                stored, reason = company, None
            if stored is None:
                sub["status"] = "insufficient_evidence"
                limit("no_document", f"{company}: no indexed document available to search for evidence.",
                      company=company)
                continue
            if sub["type"] == "risk":
                queries = ["risk factors that could adversely affect our business financial condition and results of operations",
                           "financial risks foreign currency interest rate credit liquidity"]
                topics, cues, label = RISK_TOPIC_TERMS, RISK_CUES, "financial risks"
            else:
                label = METRIC_LABELS[metric]
                topics, cues = METRIC_TOPICS.get(metric, [label.lower()]), EXPLANATION_CUES
                queries = [f"{label} increased decreased compared to prior year primarily due to",
                           f"management's discussion and analysis {topics[0]} change"]
            meta["retrieval_calls"] += 1
            try:
                chunks = retriever(queries, where=extraction_agent.build_filter(stored, year), top_k=top_k)
            except (Exception, SystemExit) as problem:
                result["errors"].append({"stage": "retrieval", "company": company, "fiscal_year": year,
                                         "error_type": "exception",
                                         "message": f"evidence retrieval failed: {type(problem).__name__}: {problem}"})
                sub["status"] = "insufficient_evidence"
                continue
            evidence = find_evidence(chunks, topics, cues, max_evidence)
            if not evidence:
                entry = {"id": f"E{len(result['explanations']) + 1}", "kind": "INSUFFICIENT_EVIDENCE",
                         "status": "INSUFFICIENT_EVIDENCE", "company": company, "fiscal_year": year,
                         "metric": metric, "topic": label,
                         "message": f"The retrieved {company} report text does not contain an explanatory "
                                    f"statement about {label}; no reason is given."}
                result["explanations"].append(entry)
                sub["explanation_ids"].append(entry["id"])
                sub["status"] = "insufficient_evidence"
                limit("insufficient_evidence", entry["message"], company=company, metric=metric)
                continue
            for item in evidence:
                entry = {"id": f"E{len(result['explanations']) + 1}", "kind": "EVIDENCE_EXCERPT",
                         "status": "FOUND", "metric": metric, "topic": label, **item}
                result["explanations"].append(entry)
                sub["explanation_ids"].append(entry["id"])
            sub["status"] = "answered"

    # ---- sources ----------------------------------------------------------
    merged = {}

    def add_source(source_file, page, company, year, used_for):
        if source_file is None or page is None:
            return
        row = merged.setdefault((source_file, page), {"company": company, "fiscal_year": year,
                                                      "source_file": source_file, "page_number": page,
                                                      "used_for": []})
        row["used_for"].append(used_for)

    for finding in result["findings"]:
        if finding["status"] == "FOUND":
            add_source(finding["source_file"], finding["page_number"], finding["company"],
                       finding["fiscal_year"], finding["id"])
            for part in finding["derived_from"] or []:
                add_source(part.get("source_file"), part.get("page_number"), finding["company"],
                           finding["fiscal_year"], f"{finding['id']} (derivation input: {part.get('metric')})")
    for entry in result["explanations"]:
        if entry["status"] == "FOUND":
            add_source(entry["source_file"], entry["page_number"], entry["company"],
                       entry["fiscal_year"], entry["id"])
    result["sources"] = list(merged.values())

    # ---- status -----------------------------------------------------------
    found_anything = any(f["status"] == "FOUND" for f in result["findings"]) or \
        any(e["status"] == "FOUND" for e in result["explanations"])
    if result["errors"] and not found_anything:
        result["status"] = "error"
    elif not found_anything:
        result["status"] = "insufficient_evidence"
    elif result["errors"] or any(l["type"] in ("missing_data", "insufficient_evidence", "change_not_calculated",
                                               "company_not_indexed", "fiscal_year_not_indexed")
                                 for l in result["limitations"]):
        result["status"] = "partial"
    return result


def _any_year_name(company, catalog):
    key = company_key(company)
    totals = {}
    for (stored, _year), count in catalog["counts"].items():
        if company_key(stored) == key:
            totals[stored] = totals.get(stored, 0) + count
    return max(totals, key=totals.get) if totals else None


# ---------------------------------------------------------------------------
# 5. Readable output
# ---------------------------------------------------------------------------

def format_research_report(result):
    lines = ["=" * 78, "RESEARCH RESULT", f"Question: {result['query']}", f"Status: {result['status']}"]
    plan = result["plan"]
    if plan:
        lines.append(f"Understood as: companies={plan['companies']} fiscal_years={plan['fiscal_years']} "
                     f"metrics={plan['metrics']}")
        lines += [f"Assumption: {a}" for a in plan["assumptions"]]
    lines.append("=" * 78)

    if result["findings"]:
        lines += ["", "FINDINGS (from the indexed annual reports)"]
        for f in result["findings"]:
            if f["status"] == "FOUND":
                value = (f"{f['value']:,.2f}".rstrip("0").rstrip(".") + ("%" if f["unit"] == "percent" else
                                                                         f" {f['unit'] or ''}"))
                lines.append(f"  [{f['id']}] {f['company']} FY{f['fiscal_year']} {f['metric_label']}: "
                             f"{value.strip()}  <{f['basis']}>  source: {f['source_file']} p.{f['page_number']}")
                if f["basis"] == "DERIVED":
                    lines.append(f"        {f['calculation']}; inputs: " + "; ".join(
                        f"{p['metric']} {p['value']} {p['unit']} ({p['source_file']} p.{p['page_number']})"
                        for p in f["derived_from"] or []))
            else:
                lines.append(f"  [{f['id']}] {f['company']} FY{f['fiscal_year']} {f['metric_label']}: "
                             f"NOT_FOUND - insufficient evidence ({f['reason']})")

    if result["comparisons"]:
        lines += ["", "COMPARISONS (calculated from the findings above)"]
        for c in result["comparisons"]:
            if c["kind"] == "year_over_year":
                lines.append(f"  [{c['id']}] <DERIVED> {c['statement']}")
            else:
                lines += [f"  [{c['id']}] <DERIVED> {r['statement']}" for r in c["relations"]]
                if c["incompatibility"]:
                    lines.append(f"  [{c['id']}] FY{c['fiscal_year']} {c['metric_label']}: not compared across "
                                 f"groups - {c['incompatibility']['reason']}")

    if result["explanations"]:
        lines += ["", "EVIDENCE (verbatim excerpts; no reasons are supplied beyond what the report states)"]
        for e in result["explanations"]:
            if e["status"] == "FOUND":
                lines.append(f"  [{e['id']}] {e['company']} FY{e['fiscal_year']} - {e['topic']}: "
                             f"\"{e['statement']}\"  ({e['source_file']} p.{e['page_number']})")
            else:
                lines.append(f"  [{e['id']}] INSUFFICIENT EVIDENCE - {e['message']}")

    if result["limitations"]:
        lines += ["", "LIMITATIONS"] + [f"  - {l['message']}" for l in result["limitations"]]
    if result["errors"]:
        lines += ["", "ERRORS"] + [f"  - [{e['stage']}] {e['message']}" for e in result["errors"]]
    if result["sources"]:
        lines += ["", "SOURCES"] + [f"  - {s['company']} FY{s['fiscal_year']}: {s['source_file']} "
                                     f"p.{s['page_number']}  ({', '.join(s['used_for'])})" for s in result["sources"]]
    return "\n".join(lines)


DEMO_QUESTION = ("Compare Apple's FY2023 and FY2024 revenue, net income, diluted EPS and operating margin. "
                 "Explain the changes using only the indexed annual reports.")

if __name__ == "__main__":
    question_text = " ".join(sys.argv[1:]) or DEMO_QUESTION
    outcome = run_research(question_text)
    print(format_research_report(outcome))
    sys.exit(0 if outcome["status"] in ("success", "partial") else 1)