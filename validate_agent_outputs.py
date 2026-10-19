"""
validate_agent_outputs.py

Validation layer for the multi-agent pipeline. This is NOT production code:
nothing here is imported by the agents or the orchestrator, and no
assertions live inside the agents.

Every validator returns a list of issues:
    {"severity": "FAIL" | "WARN", "check": str, "metric": str | None, "message": str}
An empty list (or only WARN items) means the check passed. FAIL items are never
hidden; the caller prints them.

Three kinds of validation
-------------------------
STRUCTURAL         right fields, valid statuses, no malformed records, no
                   value attached to a NOT_FOUND record, stages all present.
FINANCIAL ACCURACY Agent 1 values compared with the seed-document expectations
                   below, using explicit tolerances. The expected numbers live
                   ONLY in this file (the validation layer); they must never be
                   copied into extraction logic.
SOURCE FAITHFULNESS The cited evidence is checked against the value, not just
                   for existence: the value must appear in the evidence, the
                   evidence must carry the right label and fiscal-year
                   column, the scale arithmetic must be consistent, EPS must
                   be the diluted figure, and a derived margin must recompute
                   from its two cited inputs.

Scope note: these checks verify that a metric is SUPPORTED BY THE EVIDENCE THE
AGENT STORED. They do not re-read the PDF, so they cannot detect a case where
the stored evidence snippet itself came from the wrong place but happens to be
internally consistent. The seed expectations (taken from the annual reports)
cover that gap for the Apple documents.
"""

import re

from models import normalize_status

# ---------------------------------------------------------------------------
# Schema constants (taken from the actual agents)
# ---------------------------------------------------------------------------
TARGET_METRICS = ["revenue", "net_income", "gross_margin", "operating_margin",
                  "total_debt", "cash_flow", "eps"]
RECORD_FIELDS = ["metric", "company", "fiscal_year", "value", "unit", "currency",
                 "source_file", "page_number", "evidence", "status", "derived"]
AGENT2_METRICS = ["revenue", "net_income", "eps", "gross_margin", "operating_margin",
                  "net_profit_margin"]
MARGIN_KEYS = {"gross_margin", "operating_margin", "net_profit_margin"}
VALID_DIRECTIONS = {"increased", "decreased", "unchanged", "unknown"}
VALID_SEVERITIES = {"informational", "moderate", "high"}
FLAG_TYPES = {"declining_revenue", "declining_net_income", "declining_eps",
              "margin_compression", "rising_debt", "negative_cash_flow",
              "declining_cash_flow", "debt_rising_profitability_declining",
              "unusual_pattern"}
FLAG_FIELDS = ["type", "severity", "metric", "previous_value", "current_value",
               "change_percent", "change_points", "fiscal_years", "description",
               "source_file", "page_number", "evidence"]
FLAG_METRIC_TO_KEY = {"Revenue": "revenue", "Net Income": "net_income", "EPS": "eps",
                      "Gross Margin": "gross_margin", "Operating Margin": "operating_margin",
                      "Total Debt": "total_debt",
                      "Cash Flow from Operating Activities": "cash_flow"}
AGENT2_LABELS = {"revenue": "Revenue", "net_income": "Net Income", "eps": "EPS",
                 "gross_margin": "Gross Margin", "operating_margin": "Operating Margin",
                 "net_profit_margin": "Net Profit Margin"}

# Independent copy of the scale table on purpose: the check is only useful if
# it does not borrow the agent's own table.
SCALE_FACTORS_TO_MILLIONS = {"thousand": 0.001, "million": 1.0, "billion": 1000.0,
                             "trillion": 1_000_000.0, "crore": 10.0, "lakh": 0.1}

# Words that must appear in the evidence for each metric to be "about" it.
LABEL_KEYWORDS = {
    "revenue": ["net sales", "revenue"],
    "net_income": ["net income", "net profit", "profit for the year", "net earnings",
                   "profit after tax"],
    "operating_income": ["operating income", "income from operations", "operating profit",
                         "profit from operations"],
    "gross_margin": ["gross margin", "gross profit"],
    "operating_margin": ["operating margin"],
    "eps": ["earnings per share", "eps", "diluted", "basic"],
    "total_debt": ["debt", "borrowings"],
    "cash_flow": ["operating activities", "operations"],
}

GEO_SEGMENT_IN_EVIDENCE = re.compile(
    r"(?:americas|europe|greater china|japan|asia[\s-]*pacific)\s*:", re.I)

RECOMMENDATION_PATTERN = re.compile(
    r"\b(buy|sell|should invest|recommend\w*|best|worst|winner|loser|"
    r"good investment|bad investment|avoid investing|overvalued|undervalued|"
    r"fraud\w*|bankrupt\w*)\b", re.IGNORECASE)

# ---------------------------------------------------------------------------
# Seed expectations (validation layer ONLY; taken from the Apple annual reports)
# ---------------------------------------------------------------------------
SEED_EXPECTATIONS = {
    ("Apple", "2023"): {
        "source_file_contains": ["apple", "2023"],
        "revenue": {"value": 383285.0, "unit": "million USD", "rel_tol": 0.001},
        "net_income": {"value": 96995.0, "unit": "million USD", "rel_tol": 0.001},
        "eps": {"value": 6.13, "unit": "USD per share", "abs_tol": 0.005},
        "gross_margin": {"value": 44.1, "unit": "percent", "abs_tol": 0.05},
        "operating_income": {"value": 114301.0, "unit": "million USD", "rel_tol": 0.001},
        "operating_margin": {"value": 29.82, "unit": "percent", "abs_tol": 0.1},
    },
    ("Apple", "2024"): {
        "source_file_contains": ["apple", "2024"],
        "revenue": {"value": 391035.0, "unit": "million USD", "rel_tol": 0.001},
        "net_income": {"value": 93736.0, "unit": "million USD", "rel_tol": 0.001},
        "eps": {"value": 6.08, "unit": "USD per share", "abs_tol": 0.005},
        "gross_margin": {"value": 46.2, "unit": "percent", "abs_tol": 0.05},
        "operating_income": {"value": 123216.0, "unit": "million USD", "rel_tol": 0.001},
        "operating_margin": {"value": 31.51, "unit": "percent", "abs_tol": 0.1},
    },
}


def get_seed(company, fiscal_year):
    return SEED_EXPECTATIONS.get((company, str(fiscal_year)))


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _issue(severity, check, metric, message):
    return {"severity": severity, "check": check, "metric": metric, "message": message}


def failures(issues):
    return [i for i in issues if i["severity"] == "FAIL"]


def _is_number(value):
    return isinstance(value, (int, float)) and not isinstance(value, bool) and value == value


def _numbers_in(text):
    """Every number in a text, commas removed (works for 1,53,670 style grouping)."""
    numbers = []
    for token in re.findall(r"\d[\d,]*\.?\d*", text or ""):
        try:
            numbers.append(float(token.replace(",", "")))
        except ValueError:
            pass
    return numbers


def _close(a, b, rel_tol=0.0, abs_tol=0.0):
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b)))


# ---------------------------------------------------------------------------
# 1. STRUCTURAL VALIDATION
# ---------------------------------------------------------------------------

def validate_agent1_records(data, company, fiscal_year):
    """Structure of one fiscal year of Agent 1 output."""
    issues = []
    if not isinstance(data, dict):
        return [_issue("FAIL", "agent1.structure", None, "Agent 1 output is not a dict.")]

    for key in TARGET_METRICS:
        record = data.get(key)
        if not isinstance(record, dict):
            issues.append(_issue("FAIL", "agent1.structure", key, "no record dict for this metric"))
            continue
        missing = [f for f in RECORD_FIELDS if f not in record]
        if missing:
            issues.append(_issue("FAIL", "agent1.structure", key, f"missing fields {missing}"))
            continue
        if str(record["status"]).lower() not in ("found", "not_found"):
            issues.append(_issue("FAIL", "agent1.status", key, f"invalid status {record['status']!r}"))
            continue
        if str(record["company"]).lower() != company.lower():
            issues.append(_issue("FAIL", "agent1.identity", key,
                                 f"company {record['company']!r} != requested {company!r}"))
        if str(record["fiscal_year"]) != str(fiscal_year):
            issues.append(_issue("FAIL", "agent1.identity", key,
                                 f"fiscal_year {record['fiscal_year']!r} != requested {fiscal_year!r}"))

        if normalize_status(record["status"]) == "FOUND":
            if not _is_number(record["value"]):
                issues.append(_issue("FAIL", "agent1.found", key, "FOUND record has no numeric value"))
            if not (isinstance(record["source_file"], str) and record["source_file"].strip()):
                issues.append(_issue("FAIL", "agent1.found", key, "FOUND record has no source_file"))
            page = record["page_number"]
            if isinstance(page, bool) or not isinstance(page, int) or page < 1:
                issues.append(_issue("FAIL", "agent1.found", key, f"invalid page_number {page!r}"))
            if not (isinstance(record["evidence"], str) and record["evidence"].strip()):
                issues.append(_issue("FAIL", "agent1.found", key, "FOUND record has no evidence"))
            if not isinstance(record["derived"], bool):
                issues.append(_issue("FAIL", "agent1.found", key, "'derived' is not a bool"))
            if record["unit"] is None:
                issues.append(_issue("WARN", "agent1.found", key, "FOUND record has no unit"))
        elif record["value"] is not None:
            issues.append(_issue("FAIL", "agent1.not_found", key,
                                 f"NOT_FOUND record carries a value ({record['value']!r}); possible fabrication"))
    return issues


def validate_agent2_result(analysis, company, previous_year, current_year):
    issues = []
    if not isinstance(analysis, dict):
        return [_issue("FAIL", "agent2.structure", None, "Agent 2 output is not a dict.")]
    for key in ("company", "previous_year", "current_year", "comparisons", "summary"):
        if key not in analysis:
            issues.append(_issue("FAIL", "agent2.structure", None, f"missing '{key}'"))
    if issues:
        return issues
    if analysis["company"] != company or analysis["previous_year"] != previous_year \
            or analysis["current_year"] != current_year:
        issues.append(_issue("FAIL", "agent2.identity", None,
                             "company/years in Agent 2 output differ from the request"))
    if not isinstance(analysis["summary"], str):
        issues.append(_issue("FAIL", "agent2.structure", None, "summary is not text"))

    comparisons = analysis["comparisons"]
    from analysis_agent import calculate_margin_change, calculate_percentage_change
    for key in AGENT2_METRICS:
        comparison = comparisons.get(key)
        if not isinstance(comparison, dict):
            issues.append(_issue("FAIL", "agent2.structure", key, "no comparison for this metric"))
            continue
        status = comparison.get("status")
        if status not in ("calculated", "not_calculated"):
            issues.append(_issue("FAIL", "agent2.status", key, f"invalid status {status!r}"))
            continue
        if status == "not_calculated":
            if not (isinstance(comparison.get("reason"), str) and comparison["reason"].strip()):
                issues.append(_issue("FAIL", "agent2.missing_data", key, "not calculated but no reason given"))
            if comparison.get("percentage_change") is not None or comparison.get("margin_change_points") is not None:
                issues.append(_issue("FAIL", "agent2.missing_data", key,
                                     "a change was reported although the metric was not calculated"))
            continue
        prev, curr = comparison.get("previous_value"), comparison.get("current_value")
        if not (_is_number(prev) and _is_number(curr)):
            issues.append(_issue("FAIL", "agent2.calculated", key, "calculated without numeric values"))
            continue
        if comparison.get("direction") not in VALID_DIRECTIONS:
            issues.append(_issue("FAIL", "agent2.calculated", key, "invalid direction"))
        if key in MARGIN_KEYS:
            expected, actual = calculate_margin_change(curr, prev), comparison.get("margin_change_points")
        else:
            expected, actual = calculate_percentage_change(curr, prev), comparison.get("percentage_change")
        if expected != actual:
            issues.append(_issue("FAIL", "agent2.arithmetic", key,
                                 f"reported change {actual} but recomputed {expected}"))
    return issues


def validate_agent3_result(red, company):
    issues = []
    if not isinstance(red, dict):
        return [_issue("FAIL", "agent3.structure", None, "Agent 3 output is not a dict.")]
    for key in ("company", "fiscal_years", "flags", "summary", "data_limitations"):
        if key not in red:
            issues.append(_issue("FAIL", "agent3.structure", None, f"missing '{key}'"))
    if issues:
        return issues
    if red["company"] != company:
        issues.append(_issue("FAIL", "agent3.identity", None, "company differs from the request"))
    if not isinstance(red["data_limitations"], list) or \
            any(not isinstance(t, str) for t in red["data_limitations"]):
        issues.append(_issue("FAIL", "agent3.limitations", None, "data_limitations must be a list of text"))
    for number, flag in enumerate(red["flags"], start=1):
        tag = f"flag #{number}"
        missing = [f for f in FLAG_FIELDS if f not in flag]
        if missing:
            issues.append(_issue("FAIL", "agent3.flag", tag, f"missing fields {missing}"))
            continue
        if flag["severity"] not in VALID_SEVERITIES:
            issues.append(_issue("FAIL", "agent3.flag", tag, f"invalid severity {flag['severity']!r}"))
        if flag["type"] not in FLAG_TYPES:
            issues.append(_issue("FAIL", "agent3.flag", tag, f"unknown type {flag['type']!r}"))
        if not (isinstance(flag["fiscal_years"], list) and len(flag["fiscal_years"]) == 2):
            issues.append(_issue("FAIL", "agent3.flag", tag, "fiscal_years must list two years"))
        if not (isinstance(flag["description"], str) and flag["description"].strip()):
            issues.append(_issue("FAIL", "agent3.flag", tag, "empty description"))
        if flag["source_file"] and flag["page_number"] is None:
            issues.append(_issue("FAIL", "agent3.flag", tag, "source_file without page_number"))
    return issues


def validate_orchestration_result(result):
    issues = []
    required = ["status", "company", "previous_fiscal_year", "current_fiscal_year",
                "document_processing", "agent_1", "agent_2", "agent_3", "limitations",
                "pipeline_errors", "pipeline_metadata"]
    missing = [k for k in required if k not in result]
    if missing:
        return [_issue("FAIL", "orchestrator.structure", None, f"missing keys {missing}")]

    status = result["status"]
    if status not in ("success", "pipeline_error", "invalid_input"):
        issues.append(_issue("FAIL", "orchestrator.status", None, f"unknown status {status!r}"))

    stages = ["agent_1", "agent_2", "agent_3"]
    states = {s: result[s]["status"] for s in stages}
    if status == "success":
        for stage in stages:
            if states[stage] != "completed":
                issues.append(_issue("FAIL", "orchestrator.stages", stage,
                                     f"status is success but {stage} is {states[stage]!r} (silent skip)"))
        if result["pipeline_errors"]:
            issues.append(_issue("FAIL", "orchestrator.stages", None, "success with pipeline_errors"))
        handoffs = result["pipeline_metadata"]["handoffs"]
        if not (handoffs["agent_1_to_agent_2"] and handoffs["agent_2_to_agent_3"]):
            issues.append(_issue("FAIL", "orchestrator.handoff", None, "a handoff flag is False"))
        if result["agent_1"]["previous"] is None or result["agent_1"]["current"] is None:
            issues.append(_issue("FAIL", "orchestrator.stages", "agent_1", "raw Agent 1 records not preserved"))
        if result["agent_2"]["result"] is None or result["agent_3"]["result"] is None:
            issues.append(_issue("FAIL", "orchestrator.stages", None, "stage output missing from result"))
    elif status == "pipeline_error":
        if not result["pipeline_errors"]:
            issues.append(_issue("FAIL", "orchestrator.errors", None, "pipeline_error without an error entry"))
        failed = [s for s in ["document_processing"] + stages if result[s]["status"] == "error"]
        if not failed:
            issues.append(_issue("FAIL", "orchestrator.errors", None, "no stage is marked as failed"))
        else:
            order = ["document_processing"] + stages
            for later in order[order.index(failed[0]) + 1:]:
                if result[later]["status"] not in ("not_run", "skipped"):
                    issues.append(_issue("FAIL", "orchestrator.stages", later,
                                         "a stage after the failed one still ran"))

    combined = result["agent_1"]["limitations"] + result["agent_2"]["limitations"] + result["agent_3"]["limitations"]
    if combined != result["limitations"]:
        issues.append(_issue("FAIL", "orchestrator.limitations", None,
                             "combined limitations differ from the per-stage lists"))
    return issues


# ---------------------------------------------------------------------------
# 2. FINANCIAL ACCURACY VALIDATION
# ---------------------------------------------------------------------------

def validate_financial_accuracy(data, company, fiscal_year, operating_income_record=None):
    """
    Compare Agent 1 values with the seed expectations for this company/year.
    A NOT_FOUND for a metric the seed expects is reported as a FAIL: the point
    is accuracy, so a missing value is not a pass. Companies with no seed are
    reported as WARN (nothing to compare against), never as a pass.
    """
    seed = get_seed(company, fiscal_year)
    if seed is None:
        return [_issue("WARN", "accuracy.no_seed", None,
                       f"no seed expectations for {company} FY{fiscal_year}; accuracy not checked")]

    issues = []
    for key in ("revenue", "net_income", "eps", "gross_margin", "operating_margin"):
        expected = seed[key]
        record = data.get(key) if isinstance(data, dict) else None
        if not record or normalize_status(record.get("status")) != "FOUND":
            issues.append(_issue("FAIL", "accuracy.value", key,
                                 f"expected {expected['value']} {expected['unit']} but Agent 1 returned NOT_FOUND"))
            continue
        actual = record["value"]
        if not _close(actual, expected["value"], expected.get("rel_tol", 0.0), expected.get("abs_tol", 0.0)):
            issues.append(_issue("FAIL", "accuracy.value", key,
                                 f"expected {expected['value']} but got {actual}"))
        if record.get("unit") != expected["unit"]:
            issues.append(_issue("FAIL", "accuracy.unit", key,
                                 f"expected unit {expected['unit']!r} but got {record.get('unit')!r}"))

    # Operating income is not one of the 7 default metrics. Use a directly
    # fetched record if supplied, otherwise the input cited by a derived margin.
    expected = seed["operating_income"]
    income_value = income_unit = None
    if operating_income_record and normalize_status(operating_income_record.get("status")) == "FOUND":
        income_value, income_unit = operating_income_record["value"], operating_income_record.get("unit")
    else:
        margin = (data or {}).get("operating_margin") or {}
        for source in margin.get("derived_from") or []:
            if source.get("metric") == "Operating Income":
                income_value, income_unit = source.get("value"), source.get("unit")
    if income_value is None:
        issues.append(_issue("WARN", "accuracy.operating_income", "operating_income",
                             "operating income not verifiable (margin was stated directly, not derived)"))
    else:
        if not _close(income_value, expected["value"], expected["rel_tol"]):
            issues.append(_issue("FAIL", "accuracy.value", "operating_income",
                                 f"expected {expected['value']} but got {income_value}"))
        if income_unit != expected["unit"]:
            issues.append(_issue("FAIL", "accuracy.unit", "operating_income",
                                 f"expected unit {expected['unit']!r} but got {income_unit!r}"))
    return issues


# ---------------------------------------------------------------------------
# 3. SOURCE FAITHFULNESS VALIDATION
# ---------------------------------------------------------------------------

def _eps_label_before_value(evidence, raw_value):
    """Which EPS label ('diluted' / 'basic' / 'unknown') sits just before the value?"""
    lower, token, saw_basic = evidence.lower(), f"{raw_value:.2f}", False
    for match in re.finditer(re.escape(token), evidence):
        before = lower[max(0, match.start() - 120):match.start()]
        diluted_at, basic_at = before.rfind("diluted"), before.rfind("basic")
        if diluted_at > basic_at:
            return "diluted"
        if basic_at > diluted_at:
            saw_basic = True
    return "basic" if saw_basic else "unknown"


def validate_record_faithfulness(key, record, company, fiscal_year, seed=None):
    """Does the stored evidence actually support this FOUND, non-derived record?"""
    issues = []
    evidence = record.get("evidence") or ""
    lower = evidence.lower()

    if str(record.get("company", "")).lower() != company.lower():
        issues.append(_issue("FAIL", "faithful.company", key, "record company differs from the request"))
    if str(record.get("fiscal_year")) != str(fiscal_year):
        issues.append(_issue("FAIL", "faithful.year", key, "record fiscal_year differs from the request"))

    if seed and seed.get("source_file_contains"):
        source = (record.get("source_file") or "").lower()
        absent = [t for t in seed["source_file_contains"] if t not in source]
        if absent:
            issues.append(_issue("FAIL", "faithful.source_file", key,
                                 f"source_file {record.get('source_file')!r} lacks {absent}"))

    if not any(word in lower for word in LABEL_KEYWORDS.get(key, [])):
        issues.append(_issue("FAIL", "faithful.label", key,
                             f"evidence does not mention {LABEL_KEYWORDS.get(key)}"))

    raw = record.get("raw_value", record.get("value"))
    if not _is_number(raw) or not any(_close(n, abs(raw), 1e-9, 1e-6) for n in _numbers_in(evidence)):
        issues.append(_issue("FAIL", "faithful.value", key,
                             f"the value {raw} does not appear in the cited evidence"))

    # Scale arithmetic: value and unit must describe the same quantity. If the
    # unit names a scale ("million USD", "billion USD"), then
    #   value == raw x factor(reported scale) / factor(unit scale).
    # This catches e.g. 383300000000.0 labeled "billion USD", where the scale
    # was applied to the number AND repeated in the label.
    unit = record.get("unit") or ""
    unit_scale = unit.split()[0] if unit.split() else None
    if key in ("revenue", "net_income", "total_debt", "cash_flow") and unit_scale in SCALE_FACTORS_TO_MILLIONS:
        reported = record.get("reported_scale")
        if reported not in SCALE_FACTORS_TO_MILLIONS:
            issues.append(_issue("FAIL", "faithful.unit", key,
                                 f"unit is '{unit}' but the reported scale is {reported!r}"))
        else:
            expected_value = raw * SCALE_FACTORS_TO_MILLIONS[reported] / SCALE_FACTORS_TO_MILLIONS[unit_scale]
            if not _close(record["value"], expected_value, 1e-6):
                issues.append(_issue("FAIL", "faithful.unit", key,
                                     f"value {record['value']} does not match raw {raw} {reported} "
                                     f"expressed in '{unit_scale}' ({expected_value})"))

    if record.get("column_source") == "year_header" and str(fiscal_year) not in evidence:
        issues.append(_issue("FAIL", "faithful.year_evidence", key,
                             f"read from a year column but FY{fiscal_year} is not in the evidence"))
    elif record.get("column_source") not in ("year_header", None) and str(fiscal_year) not in evidence:
        issues.append(_issue("WARN", "faithful.year_evidence", key,
                             "the evidence snippet does not name the fiscal year"))

    if key == "revenue":
        if "deferred revenue" in lower or "contract liabilit" in lower or "performance obligation" in lower:
            issues.append(_issue("FAIL", "faithful.context", key, "evidence is about deferred revenue / obligations"))
        if GEO_SEGMENT_IN_EVIDENCE.search(evidence):
            issues.append(_issue("FAIL", "faithful.context", key, "evidence is a regional/segment row"))
    if key == "operating_income" and GEO_SEGMENT_IN_EVIDENCE.search(evidence):
        issues.append(_issue("FAIL", "faithful.context", key, "evidence is a regional/segment row"))

    if key == "eps" and _is_number(raw):
        label = _eps_label_before_value(evidence, raw)
        if label == "basic":
            issues.append(_issue("FAIL", "faithful.eps_label", key,
                                 "evidence shows BASIC EPS where diluted is the preferred figure"))
        elif label == "unknown":
            issues.append(_issue("WARN", "faithful.eps_label", key,
                                 "could not tell from the evidence whether this is basic or diluted"))
    return issues


def validate_derived_margin(record, data):
    """A derived operating margin must recompute from its two cited inputs."""
    issues = []
    key = "operating_margin"
    if record.get("derived") is not True:
        return issues
    if "Operating Income / Revenue" not in str(record.get("calculation")):
        issues.append(_issue("FAIL", "derived.formula", key, "calculation text is not Operating Income / Revenue"))
    sources = record.get("derived_from") or []
    by_metric = {s.get("metric"): s for s in sources}
    if set(by_metric) != {"Operating Income", "Revenue"}:
        return issues + [_issue("FAIL", "derived.inputs", key, f"derived_from has {sorted(by_metric)}")]

    income, revenue = by_metric["Operating Income"], by_metric["Revenue"]
    if not (_is_number(income.get("value")) and _is_number(revenue.get("value")) and revenue["value"] != 0):
        return issues + [_issue("FAIL", "derived.inputs", key, "inputs are not usable numbers")]
    if income.get("unit") != revenue.get("unit"):
        issues.append(_issue("FAIL", "derived.units", key,
                             f"input units differ ({income.get('unit')} vs {revenue.get('unit')})"))
    expected = round(income["value"] / revenue["value"] * 100, 2)
    if not _close(record["value"], expected, 0, 0.011):
        issues.append(_issue("FAIL", "derived.arithmetic", key,
                             f"reported {record['value']} but {income['value']}/{revenue['value']}*100 = {expected}"))
    if income.get("source_file") != revenue.get("source_file"):
        issues.append(_issue("FAIL", "derived.same_filing", key, "inputs come from different documents"))
    revenue_record = (data or {}).get("revenue") or {}
    if revenue_record.get("value") != revenue["value"]:
        issues.append(_issue("FAIL", "derived.consistency", key,
                             "the revenue input differs from Agent 1's own revenue record"))
    if record.get("source_file") != income.get("source_file") or record.get("page_number") != income.get("page_number"):
        issues.append(_issue("FAIL", "derived.source", key, "record source differs from the operating-income input"))
    for page in (income.get("page_number"), revenue.get("page_number")):
        if f"p.{page}" not in (record.get("evidence") or ""):
            issues.append(_issue("FAIL", "derived.evidence", key, f"evidence does not cite page {page}"))
    return issues


def validate_source_faithfulness(data, company, fiscal_year, extra_records=None):
    """Faithfulness of every FOUND record for one fiscal year."""
    seed = get_seed(company, fiscal_year)
    issues = []
    records = dict(data or {})
    records.update(extra_records or {})
    for key, record in records.items():
        if not isinstance(record, dict) or normalize_status(record.get("status")) != "FOUND":
            continue
        if record.get("derived") is True:
            issues += validate_derived_margin(record, data)
        else:
            issues += validate_record_faithfulness(key, record, company, fiscal_year, seed)
    return issues


# ---------------------------------------------------------------------------
# Cross-stage checks: handoffs, source survival, wording
# ---------------------------------------------------------------------------

def validate_handoffs(result):
    """Prove each agent worked from the previous agent's ACTUAL output."""
    issues = []
    previous = result["agent_1"]["previous"] or {}
    current = result["agent_1"]["current"] or {}
    analysis = (result["agent_2"]["result"] or {})
    red = (result["agent_3"]["result"] or {})
    comparisons = analysis.get("comparisons") or {}

    # Agent 1 -> Agent 2: every calculated comparison uses Agent 1's own values.
    for key in ("revenue", "net_income", "eps", "gross_margin", "operating_margin"):
        comparison = comparisons.get(key) or {}
        if comparison.get("status") != "calculated":
            continue
        if comparison["previous_value"] != (previous.get(key) or {}).get("value") \
                or comparison["current_value"] != (current.get(key) or {}).get("value"):
            issues.append(_issue("FAIL", "handoff.agent1_to_agent2", key,
                                 "Agent 2's values differ from the Agent 1 records it should have received"))

    # Agent 2 -> Agent 3.
    if red.get("fiscal_years") != [result["previous_fiscal_year"], result["current_fiscal_year"]]:
        issues.append(_issue("FAIL", "handoff.agent2_to_agent3", None, "Agent 3 fiscal_years differ from Agent 2"))
    flag_to_key = {"declining_revenue": "revenue", "declining_net_income": "net_income",
                   "declining_eps": "eps"}
    for flag in red.get("flags") or []:
        key = flag_to_key.get(flag["type"])
        if key is None and flag["type"] == "margin_compression":
            key = FLAG_METRIC_TO_KEY.get(flag["metric"], "net_profit_margin"
                                         if flag["metric"] == "Net Profit Margin" else None)
        comparison = comparisons.get(key) if key else None
        if comparison and comparison.get("status") == "calculated":
            if flag["previous_value"] != comparison["previous_value"] \
                    or flag["current_value"] != comparison["current_value"]:
                issues.append(_issue("FAIL", "handoff.agent2_to_agent3", key,
                                     f"{flag['type']} flag values differ from Agent 2's comparison"))
        elif key:
            issues.append(_issue("FAIL", "handoff.agent2_to_agent3", key,
                                 f"{flag['type']} flag raised although Agent 2 did not calculate it"))

    limitations = red.get("data_limitations") or []
    for key, comparison in comparisons.items():
        label = AGENT2_LABELS.get(key)
        if comparison.get("status") != "calculated" and not any(t.startswith(label) for t in limitations):
            issues.append(_issue("FAIL", "handoff.limitations", key,
                                 "Agent 2 could not calculate this but Agent 3 recorded no limitation"))

    # Source metadata survives to the end of the pipeline.
    for flag in red.get("flags") or []:
        key = FLAG_METRIC_TO_KEY.get(flag["metric"])
        record = current.get(key) if key else None
        if record and normalize_status(record.get("status")) == "FOUND" and flag["type"] != "unusual_pattern":
            if flag["source_file"] != record.get("source_file") or flag["page_number"] != record.get("page_number"):
                issues.append(_issue("FAIL", "handoff.source_survival", key,
                                     f"{flag['type']} flag lost or changed the Agent 1 source"))
    return issues


def collect_generated_text(result):
    """Text written by Agent 2 / Agent 3 (never the raw report evidence)."""
    texts = []
    analysis, red = result["agent_2"]["result"], result["agent_3"]["result"]
    if analysis:
        import analysis_agent
        texts += [analysis.get("summary", ""), analysis_agent.format_report(analysis)]
    if red:
        import red_flag_agent
        texts += [red.get("summary", ""), red_flag_agent.format_red_flag_report(red)]
        texts += [flag["description"] for flag in red["flags"]]
    return texts


def validate_no_recommendations(texts):
    hits = sorted({m.group(0).lower() for text in texts for m in RECOMMENDATION_PATTERN.finditer(text)})
    if hits:
        return [_issue("FAIL", "wording.recommendation", None, f"advice/ranking wording found: {hits}")]
    return []


# ---------------------------------------------------------------------------
# One call that runs every check on an orchestration result
# ---------------------------------------------------------------------------

def validate_orchestration_output(result, operating_income_records=None):
    """
    Run all validators on an orchestration result. Returns {category: [issues]}.
    operating_income_records: optional {fiscal_year: Agent-1 record} fetched
    directly (used only to check the seed operating-income expectation).
    """
    company = result["company"]
    previous_year, current_year = result["previous_fiscal_year"], result["current_fiscal_year"]
    operating_income_records = operating_income_records or {}
    categories = {"orchestration": validate_orchestration_result(result)}

    if result["status"] != "success":
        return categories

    previous = result["agent_1"]["previous"]
    current = result["agent_1"]["current"]
    categories["agent_1_structure"] = (validate_agent1_records(previous, company, previous_year)
                                       + validate_agent1_records(current, company, current_year))
    categories["agent_2_structure"] = validate_agent2_result(
        result["agent_2"]["result"], company, previous_year, current_year)
    categories["agent_3_structure"] = validate_agent3_result(result["agent_3"]["result"], company)
    categories["handoffs"] = validate_handoffs(result)
    categories["financial_accuracy"] = (
        validate_financial_accuracy(previous, company, previous_year, operating_income_records.get(previous_year))
        + validate_financial_accuracy(current, company, current_year, operating_income_records.get(current_year)))
    extra_prev = {"operating_income": operating_income_records[previous_year]} \
        if previous_year in operating_income_records else None
    extra_curr = {"operating_income": operating_income_records[current_year]} \
        if current_year in operating_income_records else None
    categories["source_faithfulness"] = (
        validate_source_faithfulness(previous, company, previous_year, extra_prev)
        + validate_source_faithfulness(current, company, current_year, extra_curr))
    categories["no_recommendations"] = validate_no_recommendations(collect_generated_text(result))
    return categories


def print_category_summary(categories, indent="  "):
    for name, issues in categories.items():
        bad = failures(issues)
        print(f"{indent}{name}: {'PASS' if not bad else 'FAIL'}"
              + (f" ({len(issues) - len(bad)} warning(s))" if issues and not bad else ""))
        for item in issues:
            print(f"{indent}   [{item['severity']}] {item['check']}"
                  f"{' / ' + str(item['metric']) if item['metric'] else ''}: {item['message']}")


if __name__ == "__main__":
    # REAL validation: runs the actual pipeline on your indexed documents.
    import sys
    import extraction_agent
    import multi_agent_orchestrator

    company_name, previous, current = ("Apple", "2023", "2024") if len(sys.argv) != 4 else sys.argv[1:4]
    outcome = multi_agent_orchestrator.run_financial_analysis(company_name, previous, current)
    print()
    if outcome["status"] != "success":
        for error in outcome["pipeline_errors"]:
            print(f"PIPELINE ERROR [{error['stage']}] {error['message']}")
            if error.get("hint"):
                print(f"   hint: {error['hint']}")
        sys.exit(1)

    income_records = {y: extraction_agent.build_metric_record("operating_income", company_name, y)
                      for y in (previous, current)}
    report = validate_orchestration_output(outcome, income_records)
    print("[VALIDATION] Results on REAL indexed data")
    print_category_summary(report)
    sys.exit(0 if not any(failures(v) for v in report.values()) else 1)