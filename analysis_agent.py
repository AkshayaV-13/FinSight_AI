"""
analysis_agent.py

Agent 2: Financial Analysis + Comparison Agent.

Consumes structured metric records produced by Agent 1 (extraction_agent.py)
and performs year-over-year and cross-company financial analysis. This file
never retrieves from ChromaDB and never re-implements extraction -- it only
imports extract_metrics() from extraction_agent.py for the optional "live"
convenience wrapper; the core functions accept already-fetched Agent 1
output so they can be tested and reasoned about independently.

Pipeline:
    Agent 1 output (dict of metric_key -> record)
        -> validate_metric_record()          per-metric input validation
        -> calculate_absolute_change() /
           calculate_percentage_change() /
           calculate_margin_change()          pure calculation functions
        -> compare_metric()                   wraps the above, handles
                                               NOT_FOUND and currency/unit
                                               safety for one metric
        -> analyze_company()                  runs compare_metric() across
                                               Revenue/Net Income/EPS/
                                               Gross Margin/Operating Margin
                                               for two fiscal years
        -> compare_companies()                same metric, same fiscal
                                               year, across companies
        -> format_report()                    renders the structured
                                               result as readable text

Descriptive only: this file never produces buy/sell/invest recommendations
or rankings of companies as "best"/"worst". It describes what the numbers
were and how they changed.

Currency/unit safety: a comparison is only computed when both records have
the identical 'currency' and 'unit' strings from Agent 1 (e.g. both
"million USD"). No currency conversion is ever invented. A mismatch is
reported as "could not be compared", not silently ignored.
"""

from collections import Counter

from models import is_found, normalize_status


# ---------------------------------------------------------------------------
# Metric configuration
# ---------------------------------------------------------------------------
METRIC_LABELS = {
    "revenue": "Revenue",
    "net_income": "Net Income",
    "eps": "EPS",
    "gross_margin": "Gross Margin",
    "operating_margin": "Operating Margin",
    "operating_income": "Operating Income",
    "total_debt": "Total Debt",
    "cash_flow": "Cash Flow from Operating Activities",
    "total_income": "Total Income",
    "profit_before_tax": "Profit Before Tax (PBT)",
    "profit_after_tax": "Profit After Tax (PAT)",
    "ebitda": "EBITDA",
    "total_assets": "Total Assets",
    "total_liabilities": "Total Liabilities",
    "total_equity": "Total Equity",
    "cash_and_equivalents": "Cash and Cash Equivalents",
    "investing_cash_flow": "Cash Flow from Investing Activities",
    "financing_cash_flow": "Cash Flow from Financing Activities",
    "gross_profit": "Gross Profit",
    "net_margin": "Net Profit Margin",
}

# Margins are compared in PERCENTAGE POINTS (46.2% -> 47.0% = +0.8 points),
# never as a "percentage change of a percentage", which is a different and
# much more confusing number.
MARGIN_METRICS = {"gross_margin", "operating_margin", "net_margin"}

PROFITABILITY_METRICS = ["revenue", "net_income", "eps", "gross_margin", "operating_margin"]


# ---------------------------------------------------------------------------
# 1. Input validation
# ---------------------------------------------------------------------------

def validate_metric_record(record):
    """
    Check whether an Agent 1 record can be used in a calculation.

    Returns (is_valid: bool, reason: str or None). `reason` is a short
    machine-readable code ("missing", "not_found", "invalid_value") used
    to build a human-readable message elsewhere; it is None when valid.
    """
    if record is None:
        return False, "missing"

    if not is_found(record):
        return False, "not_found"

    value = record.get("value")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False, "invalid_value"
    if isinstance(value, float) and value != value:  # NaN check, no math import needed
        return False, "invalid_value"

    return True, None


def units_compatible(record_a, record_b):
    """
    True only if both records share the identical currency AND unit string
    from Agent 1 (e.g. both "million USD"). This is intentionally strict:
    "million USD" vs "billion USD", or USD vs INR, are both refused rather
    than silently compared or converted.
    """
    return (
        record_a.get("currency") == record_b.get("currency")
        and record_a.get("unit") == record_b.get("unit")
    )


# ---------------------------------------------------------------------------
# 2. Pure calculation functions (kept separate per the "don't put everything
#    in one giant function" requirement)
# ---------------------------------------------------------------------------

def calculate_absolute_change(current_value, previous_value):
    """
    current - previous. None if either input is missing. Rounded to 6
    decimal places purely to remove IEEE-754 floating-point noise (e.g.
    6.08 - 6.13 otherwise prints as -0.04999999999999982); financial
    figures from Agent 1 are already clean to at most a few decimals, so
    this never discards real precision.
    """
    if current_value is None or previous_value is None:
        return None
    return round(current_value - previous_value, 6)


def calculate_percentage_change(current_value, previous_value):
    """((current - previous) / previous) * 100. None if previous is 0/missing."""
    if current_value is None or previous_value is None:
        return None
    if previous_value == 0:
        return None  # never divide by zero
    return round(((current_value - previous_value) / previous_value) * 100, 2)


def calculate_margin_change(current_percent, previous_percent):
    """
    PERCENTAGE-POINT change for margins: 46.2 -> 47.0 returns +0.8, not a
    percentage-of-a-percentage calculation. None if either input is missing.
    """
    if current_percent is None or previous_percent is None:
        return None
    return round(current_percent - previous_percent, 2)


def direction_from_change(change):
    """'increased' / 'decreased' / 'unchanged' / 'unknown' (if change is None)."""
    if change is None:
        return "unknown"
    if change > 0:
        return "increased"
    if change < 0:
        return "decreased"
    return "unchanged"


def calculate_net_profit_margin(net_income_record, revenue_record):
    """
    Net Profit Margin = Net Income / Revenue * 100.

    Only calculated when both inputs are valid, found, and share the same
    currency/unit. Always marked derived=True; never overwrites an
    Agent-1-provided Gross/Operating Margin, since this is a distinct
    metric Agent 1 does not currently extract on its own.
    """
    base = {"metric": "Net Profit Margin", "derived": True, "status": "not_calculated",
            "value": None, "unit": "percent", "reason": None}

    ni_valid, ni_reason = validate_metric_record(net_income_record)
    rev_valid, rev_reason = validate_metric_record(revenue_record)

    if not ni_valid or not rev_valid:
        base["reason"] = ("Net Profit Margin could not be calculated because "
                           "Net Income and/or Revenue was not available.")
        return base

    if not units_compatible(net_income_record, revenue_record):
        base["reason"] = (
            f"Net Profit Margin could not be calculated: Net Income "
            f"({net_income_record.get('unit')}) and Revenue "
            f"({revenue_record.get('unit')}) are not in compatible units."
        )
        return base

    revenue_value = revenue_record["value"]
    if revenue_value == 0:
        base["reason"] = "Net Profit Margin could not be calculated: Revenue is zero."
        return base

    margin = round((net_income_record["value"] / revenue_value) * 100, 2)
    base.update({
        "value": margin,
        "status": "calculated",
        "calculation": "Net Profit Margin = Net Income / Revenue * 100",
        "company": net_income_record.get("company"),
        "fiscal_year": net_income_record.get("fiscal_year"),
    })
    return base


# ---------------------------------------------------------------------------
# 3-4. Year-over-year / margin / profitability comparison for one metric
# ---------------------------------------------------------------------------

def compare_metric(metric_key, previous_record, current_record):
    """
    Compare one metric across two fiscal years. Handles NOT_FOUND and
    currency/unit incompatibility by returning a "not_calculated" result
    with a human-readable reason, rather than raising or fabricating a
    number.
    """
    label = METRIC_LABELS.get(metric_key, metric_key)
    is_margin = metric_key in MARGIN_METRICS

    result = {
        "metric": label, "metric_key": metric_key, "is_margin": is_margin,
        "previous_value": None, "current_value": None,
        "unit": None, "currency": None,
        "change": None, "percentage_change": None, "margin_change_points": None,
        "direction": "unknown", "status": "not_calculated", "reason": None,
        "derived": False,
    }

    curr_valid, curr_reason = validate_metric_record(current_record)
    if not curr_valid:
        result["reason"] = (
            f"{label} could not be analyzed because the required metric was "
            f"not available for the current fiscal year."
            if curr_reason == "not_found" else
            f"{label} could not be analyzed: current-year value is {curr_reason}."
        )
        return result

    prev_valid, prev_reason = validate_metric_record(previous_record)
    if not prev_valid:
        result["current_value"] = current_record.get("value")
        result["unit"] = current_record.get("unit")
        result["currency"] = current_record.get("currency")
        result["reason"] = (
            f"{label} could not be analyzed because the required metric was "
            f"not available for the previous fiscal year."
            if prev_reason == "not_found" else
            f"{label} could not be analyzed: previous-year value is {prev_reason}."
        )
        return result

    if not units_compatible(previous_record, current_record):
        result["previous_value"] = previous_record.get("value")
        result["current_value"] = current_record.get("value")
        result["reason"] = (
            f"{label} could not be compared: incompatible units/currency "
            f"(previous: {previous_record.get('unit')}, current: {current_record.get('unit')})."
        )
        return result

    previous_value = previous_record["value"]
    current_value = current_record["value"]

    result.update({
        "previous_value": previous_value,
        "current_value": current_value,
        "unit": current_record.get("unit"),
        "currency": current_record.get("currency"),
        "derived": bool(current_record.get("derived") or previous_record.get("derived")),
        "status": "calculated",
    })

    if is_margin:
        change = calculate_margin_change(current_value, previous_value)
        result["margin_change_points"] = change
        result["direction"] = direction_from_change(change)
    else:
        result["change"] = calculate_absolute_change(current_value, previous_value)
        result["percentage_change"] = calculate_percentage_change(current_value, previous_value)
        result["direction"] = direction_from_change(result["change"])

    return result


def build_summary(comparisons):
    """
    Deterministic, factual, one-sentence-per-metric summary. No LLM, no
    recommendations -- just a plain-language restatement of the numbers
    already computed by compare_metric().
    """
    order = PROFITABILITY_METRICS + ["net_profit_margin"]
    lines = []

    for key in order:
        c = comparisons.get(key)
        if not c or c["status"] != "calculated":
            continue
        label = c["metric"]
        if c["is_margin"]:
            points = c["margin_change_points"]
            lines.append(
                f"{label} {c['direction']} by {abs(points)} percentage points "
                f"({c['previous_value']}% to {c['current_value']}%)."
            )
        else:
            pct = c["percentage_change"]
            unit_suffix = f" {c['unit']}" if c["unit"] else ""
            pct_text = f" ({abs(pct)}%)" if pct is not None else ""
            lines.append(
                f"{label} {c['direction']}{pct_text} from {c['previous_value']}{unit_suffix} "
                f"to {c['current_value']}{unit_suffix}."
            )

    if not lines:
        return "Insufficient data was available to summarize year-over-year changes."
    return " ".join(lines)


def analyze_company(company, previous_year, current_year, previous_data, current_data):
    """
    Core Agent 2 entry point. previous_data / current_data are dicts of
    metric_key -> Agent 1 record, exactly as returned by
    extraction_agent.extract_metrics(). Never calls Agent 1 itself.
    """
    comparisons = {}
    for key in PROFITABILITY_METRICS:
        comparisons[key] = compare_metric(key, previous_data.get(key), current_data.get(key))

    net_margin_current = calculate_net_profit_margin(
        current_data.get("net_income"), current_data.get("revenue")
    )
    net_margin_previous = calculate_net_profit_margin(
        previous_data.get("net_income"), previous_data.get("revenue")
    )

    if net_margin_current["status"] == "calculated" and net_margin_previous["status"] == "calculated":
        points = calculate_margin_change(net_margin_current["value"], net_margin_previous["value"])
        comparisons["net_profit_margin"] = {
            "metric": "Net Profit Margin", "metric_key": "net_profit_margin", "is_margin": True,
            "previous_value": net_margin_previous["value"], "current_value": net_margin_current["value"],
            "unit": "percent", "currency": None,
            "change": None, "percentage_change": None,
            "margin_change_points": points, "direction": direction_from_change(points),
            "status": "calculated", "derived": True, "reason": None,
        }
    else:
        comparisons["net_profit_margin"] = {
            "metric": "Net Profit Margin", "metric_key": "net_profit_margin", "is_margin": True,
            "previous_value": None, "current_value": None,
            "unit": "percent", "currency": None,
            "change": None, "percentage_change": None, "margin_change_points": None,
            "direction": "unknown", "status": "not_calculated", "derived": True,
            "reason": net_margin_current.get("reason") or net_margin_previous.get("reason"),
        }

    return {
        "company": company,
        "previous_year": previous_year,
        "current_year": current_year,
        "comparisons": comparisons,
        "summary": build_summary(comparisons),
    }


def analyze_company_live(company, previous_year, current_year, top_k=3):
    """
    Convenience wrapper: calls Agent 1's extract_metrics() for both years
    and feeds the result into analyze_company(). This is the only function
    in this file that imports extraction_agent -- everything else works on
    plain dicts so it can be tested without a live ChromaDB connection.
    """
    from extraction_agent import extract_metrics

    previous_data = extract_metrics(company, previous_year, top_k=top_k)
    current_data = extract_metrics(company, current_year, top_k=top_k)
    return analyze_company(company, previous_year, current_year, previous_data, current_data)


# ---------------------------------------------------------------------------
# 5. Multi-company comparison
# ---------------------------------------------------------------------------

def compare_companies(metric_key, companies_data, fiscal_year=None):
    """
    Compare one metric across multiple companies for one fiscal year.

    companies_data: dict of company_name -> Agent 1 record for that one
    metric/company/year (caller fetches these however it likes, typically
    one extract_metrics() call per company).

    Companies whose value is missing, or whose currency/unit doesn't match
    the majority of the group, are excluded from the comparable table and
    listed separately with a reason -- never converted or guessed.
    """
    label = METRIC_LABELS.get(metric_key, metric_key)
    excluded = []
    valid_records = {}

    for company, record in companies_data.items():
        is_valid, reason = validate_metric_record(record)
        if not is_valid:
            excluded.append({"company": company, "reason": f"{label} not available ({reason})."})
            continue
        valid_records[company] = record

    if not valid_records:
        return {
            "metric": label, "fiscal_year": fiscal_year, "rows": [], "excluded": excluded,
            "note": "No company had a usable value for this metric.",
        }

    # The comparable group is whichever (unit, currency) combination the
    # majority of companies share; anything else is excluded rather than
    # forced into the same table.
    signature_counts = Counter(
        (record.get("unit"), record.get("currency")) for record in valid_records.values()
    )
    comparable_signature = signature_counts.most_common(1)[0][0]

    rows = []
    for company, record in valid_records.items():
        signature = (record.get("unit"), record.get("currency"))
        if signature == comparable_signature:
            rows.append({
                "company": company, "value": record["value"],
                "unit": record.get("unit"), "currency": record.get("currency"),
                "source_file": record.get("source_file"), "page_number": record.get("page_number"),
            })
        else:
            excluded.append({
                "company": company,
                "reason": (
                    f"{label} unit/currency ({record.get('unit')}) is not comparable to the "
                    f"majority group ({comparable_signature[0]}) without a supplied conversion."
                ),
            })

    rows.sort(key=lambda r: r["company"])
    return {
        "metric": label, "fiscal_year": fiscal_year,
        "unit": comparable_signature[0], "currency": comparable_signature[1],
        "rows": rows, "excluded": excluded,
    }


# ---------------------------------------------------------------------------
# 9. Output formatting
# ---------------------------------------------------------------------------

def format_report(analysis_result):
    """Render an analyze_company() result as the requested readable text block."""
    comps = analysis_result["comparisons"]
    lines = [
        "=" * 60, "FINANCIAL ANALYSIS", "",
        f"Company: {analysis_result['company']}",
        f"Comparison: FY{analysis_result['previous_year']} -> FY{analysis_result['current_year']}",
        "",
    ]

    for key in PROFITABILITY_METRICS + ["net_profit_margin"]:
        c = comps.get(key)
        if not c:
            continue
        lines.append(c["metric"])
        if c["status"] != "calculated":
            lines.append(f"  {c.get('reason', 'Not available.')}")
            lines.append("")
            continue

        if c["is_margin"]:
            lines.append(f"Previous: {c['previous_value']}%")
            lines.append(f"Current: {c['current_value']}%")
            points = c["margin_change_points"]
            sign = "+" if points is not None and points >= 0 else ""
            lines.append(f"Change: {sign}{points} percentage points"
                         + ("  (derived)" if c.get("derived") else ""))
        else:
            unit_suffix = f" {c['unit']}" if c["unit"] else ""
            lines.append(f"Previous: {c['previous_value']}{unit_suffix}")
            lines.append(f"Current: {c['current_value']}{unit_suffix}")
            change_sign = "+" if c["change"] is not None and c["change"] >= 0 else ""
            lines.append(f"Change: {change_sign}{c['change']}{unit_suffix}")
            pct = c["percentage_change"]
            if pct is not None:
                pct_sign = "+" if pct >= 0 else ""
                lines.append(f"Percentage Change: {pct_sign}{pct}%")
            else:
                lines.append("Percentage Change: N/A")
            lines.append(f"Direction: {c['direction'].capitalize()}"
                         + ("  (derived)" if c.get("derived") else ""))
        lines.append("")

    lines.append("=" * 60)
    lines.append("SUMMARY")
    lines.append("")
    lines.append(analysis_result["summary"])
    lines.append("")
    lines.append("This analysis is descriptive only and does not constitute investment advice.")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Built-in CLI test (mock data; see test_analysis_agent.py for the full suite)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    def _mock_record(metric_label, value, unit, currency, year, status="found", derived=False):
        return {
            "metric": metric_label, "company": "Apple", "fiscal_year": year,
            "value": value, "unit": unit, "currency": currency,
            "source_file": "Apple_2024_Annual_report.pdf", "page_number": 32,
            "evidence": "(mock evidence)", "status": status, "derived": derived,
        }

    previous_data = {
        "revenue": _mock_record("Revenue", 383285.0, "million USD", "USD", "2023"),
        "net_income": _mock_record("Net Income", 96995.0, "million USD", "USD", "2023"),
        "eps": _mock_record("EPS", 6.13, "USD per share", "USD", "2023"),
        "gross_margin": _mock_record("Gross Margin", 44.1, "percent", None, "2023"),
        "operating_margin": _mock_record("Operating Margin", 29.98, "percent", None, "2023"),
    }
    current_data = {
        "revenue": _mock_record("Revenue", 391035.0, "million USD", "USD", "2024"),
        "net_income": _mock_record("Net Income", 93736.0, "million USD", "USD", "2024"),
        "eps": _mock_record("EPS", 6.08, "USD per share", "USD", "2024"),
        "gross_margin": _mock_record("Gross Margin", 46.2, "percent", None, "2024"),
        "operating_margin": _mock_record("Operating Margin", 31.51, "percent", None, "2024", derived=True),
    }

    print("[MOCK DATA] Apple FY2023 -> FY2024\n")
    result = analyze_company("Apple", "2023", "2024", previous_data, current_data)
    print(format_report(result))