"""
red_flag_agent.py

Agent 3: Red Flag Agent.

Screens the structured output of Agent 2 (analysis_agent.py) for
potentially important numerical patterns, and states the evidence behind
each one. It is descriptive only: it never gives investment advice, never
rates a company's overall risk, and never uses words like fraud, buy, sell,
best or worst. A "high" severity means only that one specific numerical
pattern was far past its configured screening threshold.

What this file does NOT do (by design):
  - no ChromaDB queries, no embedding model, no PDF work
  - no calls to Agent 1's retrieval/extraction functions
  - no web search, no LLM API
  - no modification of extraction_agent.py or analysis_agent.py

Inputs
------
1. analysis_result (required): the dict returned by
   analysis_agent.analyze_company(). Actual shape, as inspected:
       {"company", "previous_year", "current_year", "summary",
        "comparisons": {metric_key: {"metric", "previous_value",
            "current_value", "percentage_change", "margin_change_points",
            "direction", "status", "reason", "unit", "currency", ...}}}
   where metric_key is one of revenue, net_income, eps, gross_margin,
   operating_margin, net_profit_margin.

2. current_data / previous_data (optional): the raw Agent 1 dicts
   (metric_key -> record) that were fed into Agent 2. They are needed for
   two reasons, both discovered by inspecting Agent 2:
     a) Agent 2's comparisons do not carry source_file / page_number, so
        Agent 1's records are the only place that information exists.
     b) Agent 2 does not compare Total Debt or Cash Flow across years. For
        those two metrics only, this file applies the same percentage-
        change formula directly to the Agent 1 records (same currency/unit
        safety rule as Agent 2). If the records are not supplied, those
        categories are reported as data limitations rather than guessed.

Thresholds
----------
RED_FLAG_THRESHOLDS below are SCREENING thresholds: initial, adjustable
values chosen to keep the output readable. They are not universal
definitions of financial risk. A 6% revenue drop can be routine for a
cyclical business and serious for another. Pass your own dict to
build_red_flags(thresholds=...) to change them; any keys you omit fall
back to these defaults. All values are positive magnitudes.
"""

from collections import Counter

from models import is_found


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
RED_FLAG_THRESHOLDS = {
    # A decline of at least this many percent triggers the flag.
    "revenue_decline_pct": 5.0,
    "net_income_decline_pct": 5.0,
    "eps_decline_pct": 5.0,
    "cash_flow_decline_pct": 10.0,
    # An increase of at least this many percent triggers the flag.
    "debt_increase_pct": 15.0,
    # A margin decline of at least this many PERCENTAGE POINTS triggers it.
    "margin_decline_points": 2.0,
    # Minimum movement for a metric to count as "moving" in a combined
    # pattern (stops tiny changes being described as an unusual pattern).
    "unusual_pattern_min_pct": 3.0,
    "unusual_pattern_min_points": 1.0,
    # A flag whose magnitude is at least this multiple of its threshold is
    # labeled "high"; otherwise a triggered flag is "moderate".
    "high_severity_multiple": 2.0,
}

SEVERITY_INFORMATIONAL = "informational"
SEVERITY_MODERATE = "moderate"
SEVERITY_HIGH = "high"

MARGIN_LABELS = [
    ("gross_margin", "Gross Margin"),
    ("operating_margin", "Operating Margin"),
    ("net_profit_margin", "Net Profit Margin"),
]


# ---------------------------------------------------------------------------
# Validation and small helpers
# ---------------------------------------------------------------------------

def validate_analysis_result(analysis_result):
    """
    Structural check on an Agent 2 result. Returns (is_valid, problems).
    Does not check the numbers themselves; each detector does that.
    """
    if not isinstance(analysis_result, dict):
        return False, ["analysis_result is not a dict."]

    problems = []
    for key in ("company", "previous_year", "current_year"):
        if not analysis_result.get(key):
            problems.append(f"analysis_result is missing '{key}'.")
    if not isinstance(analysis_result.get("comparisons"), dict):
        problems.append("analysis_result is missing a 'comparisons' dict.")

    return len(problems) == 0, problems


def get_comparison(analysis_result, metric_key):
    """Return one metric's Agent 2 comparison dict, or None."""
    return (analysis_result.get("comparisons") or {}).get(metric_key)


def comparison_is_calculated(comparison):
    """True if Agent 2 actually calculated this metric's change."""
    return bool(comparison) and comparison.get("status") == "calculated"


def get_fiscal_years(analysis_result):
    return [analysis_result["previous_year"], analysis_result["current_year"]]


def classify_severity(magnitude, threshold, thresholds):
    """
    Map how far a value is past its threshold to a severity label. Only
    called for values that already crossed the threshold.
        magnitude >= high_severity_multiple * threshold -> high
        otherwise                                        -> moderate
    """
    if threshold <= 0:
        return SEVERITY_INFORMATIONAL
    if abs(magnitude) >= thresholds["high_severity_multiple"] * threshold:
        return SEVERITY_HIGH
    return SEVERITY_MODERATE


def get_source_info(metric_key, current_data):
    """
    Source details for a metric from the raw Agent 1 records, if supplied.
    Agent 2's comparisons do not include these, so without current_data the
    source fields stay None (nothing is invented).
    """
    record = (current_data or {}).get(metric_key)
    if not record:
        return {"source_file": None, "page_number": None, "evidence": None}
    return {
        "source_file": record.get("source_file"),
        "page_number": record.get("page_number"),
        "evidence": record.get("evidence"),
    }


def make_flag(flag_type, severity, metric, previous_value, current_value,
              fiscal_years, description, change_percent=None, change_points=None,
              source=None):
    """Build one flag dict in the agreed structure."""
    source = source or {"source_file": None, "page_number": None, "evidence": None}
    return {
        "type": flag_type,
        "severity": severity,
        "metric": metric,
        "previous_value": previous_value,
        "current_value": current_value,
        "change_percent": change_percent,
        "change_points": change_points,
        "fiscal_years": fiscal_years,
        "description": description,
        "source_file": source.get("source_file"),
        "page_number": source.get("page_number"),
        "evidence": source.get("evidence"),
    }


def compute_simple_change(metric_key, current_data, previous_data):
    """
    Percentage change for one metric, computed from raw Agent 1 records.

    Used ONLY for total_debt and cash_flow, which Agent 2 does not compare.
    Same formula and same currency/unit safety rule as Agent 2: returns None
    if either record is missing, NOT_FOUND, non-numeric, has a zero previous
    value, or uses a different currency/unit from the other year.
    """
    current = (current_data or {}).get(metric_key)
    previous = (previous_data or {}).get(metric_key)
    if not is_found(current) or not is_found(previous):
        return None

    current_value, previous_value = current.get("value"), previous.get("value")
    for value in (current_value, previous_value):
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            return None
    if previous_value == 0:
        return None
    if (current.get("unit") != previous.get("unit")
            or current.get("currency") != previous.get("currency")):
        return None

    return {
        "previous_value": previous_value,
        "current_value": current_value,
        "percentage_change": round(((current_value - previous_value) / previous_value) * 100, 2),
        "unit": current.get("unit"),
        "source_file": current.get("source_file"),
        "page_number": current.get("page_number"),
        "evidence": current.get("evidence"),
    }


def _unit_text(unit):
    return f" {unit}" if unit else ""


# ---------------------------------------------------------------------------
# Detectors (each returns a list of zero or more flags)
# ---------------------------------------------------------------------------

def _detect_percent_decline(analysis_result, thresholds, current_data,
                            metric_key, label, flag_type, threshold_key):
    """Shared logic for revenue / net income / EPS declines."""
    comparison = get_comparison(analysis_result, metric_key)
    if not comparison_is_calculated(comparison):
        return []

    change = comparison.get("percentage_change")
    threshold = thresholds[threshold_key]
    if change is None or change > -threshold:
        return []  # no decline, or smaller than the screening threshold

    years = get_fiscal_years(analysis_result)
    description = (
        f"{label} declined {abs(change)}% from {comparison['previous_value']} to "
        f"{comparison['current_value']}{_unit_text(comparison.get('unit'))} between "
        f"FY{years[0]} and FY{years[1]} (screening threshold: {threshold}% decline)."
    )
    return [make_flag(
        flag_type, classify_severity(change, threshold, thresholds), label,
        comparison["previous_value"], comparison["current_value"], years,
        description, change_percent=change,
        source=get_source_info(metric_key, current_data),
    )]


def detect_revenue_decline(analysis_result, thresholds, current_data=None, previous_data=None):
    return _detect_percent_decline(analysis_result, thresholds, current_data,
                                   "revenue", "Revenue", "declining_revenue",
                                   "revenue_decline_pct")


def detect_net_income_decline(analysis_result, thresholds, current_data=None, previous_data=None):
    return _detect_percent_decline(analysis_result, thresholds, current_data,
                                   "net_income", "Net Income", "declining_net_income",
                                   "net_income_decline_pct")


def detect_eps_decline(analysis_result, thresholds, current_data=None, previous_data=None):
    return _detect_percent_decline(analysis_result, thresholds, current_data,
                                   "eps", "EPS", "declining_eps",
                                   "eps_decline_pct")


def detect_margin_compression(analysis_result, thresholds, current_data=None, previous_data=None):
    """Gross, operating and net profit margin declines, in percentage points."""
    flags = []
    threshold = thresholds["margin_decline_points"]
    years = get_fiscal_years(analysis_result)

    for metric_key, label in MARGIN_LABELS:
        comparison = get_comparison(analysis_result, metric_key)
        if not comparison_is_calculated(comparison):
            continue
        points = comparison.get("margin_change_points")
        if points is None or points > -threshold:
            continue

        derived_note = " (derived value)" if comparison.get("derived") else ""
        description = (
            f"{label} declined {abs(points)} percentage points, from "
            f"{comparison['previous_value']}% to {comparison['current_value']}%{derived_note}, "
            f"between FY{years[0]} and FY{years[1]} "
            f"(screening threshold: {threshold} percentage points)."
        )
        flags.append(make_flag(
            "margin_compression", classify_severity(points, threshold, thresholds), label,
            comparison["previous_value"], comparison["current_value"], years,
            description, change_points=points,
            source=get_source_info(metric_key, current_data),
        ))
    return flags


def detect_rising_debt(analysis_result, thresholds, current_data=None, previous_data=None):
    """Total Debt increase. Needs the raw Agent 1 records (see module docstring)."""
    change = compute_simple_change("total_debt", current_data, previous_data)
    if change is None:
        return []

    percent = change["percentage_change"]
    threshold = thresholds["debt_increase_pct"]
    if percent < threshold:
        return []

    years = get_fiscal_years(analysis_result)
    description = (
        f"Total Debt increased {percent}% from {change['previous_value']} to "
        f"{change['current_value']}{_unit_text(change['unit'])} between FY{years[0]} and "
        f"FY{years[1]} (screening threshold: {threshold}% increase)."
    )
    return [make_flag(
        "rising_debt", classify_severity(percent, threshold, thresholds), "Total Debt",
        change["previous_value"], change["current_value"], years, description,
        change_percent=percent, source=change,
    )]


def detect_cash_flow_warning(analysis_result, thresholds, current_data=None, previous_data=None):
    """
    Two checks, both based strictly on available values:
      - operating cash flow is negative in the current year
      - operating cash flow declined by at least the configured percentage
    """
    flags = []
    years = get_fiscal_years(analysis_result)

    current = (current_data or {}).get("cash_flow")
    if is_found(current) and isinstance(current.get("value"), (int, float)) \
            and not isinstance(current.get("value"), bool) and current["value"] < 0:
        description = (
            f"Cash Flow from Operating Activities was negative "
            f"({current['value']}{_unit_text(current.get('unit'))}) in FY{years[1]}."
        )
        flags.append(make_flag(
            "negative_cash_flow", SEVERITY_HIGH, "Cash Flow from Operating Activities",
            None, current["value"], years, description,
            source=get_source_info("cash_flow", current_data),
        ))

    change = compute_simple_change("cash_flow", current_data, previous_data)
    if change is not None:
        threshold = thresholds["cash_flow_decline_pct"]
        if change["percentage_change"] <= -threshold:
            description = (
                f"Cash Flow from Operating Activities declined "
                f"{abs(change['percentage_change'])}% from {change['previous_value']} to "
                f"{change['current_value']}{_unit_text(change['unit'])} between FY{years[0]} "
                f"and FY{years[1]} (screening threshold: {threshold}% decline)."
            )
            flags.append(make_flag(
                "declining_cash_flow",
                classify_severity(change["percentage_change"], threshold, thresholds),
                "Cash Flow from Operating Activities",
                change["previous_value"], change["current_value"], years, description,
                change_percent=change["percentage_change"], source=change,
            ))
    return flags


def detect_debt_vs_profitability(analysis_result, thresholds, current_data=None, previous_data=None):
    """
    Category 7: Total Debt rising while Net Income falls. This also covers
    the "Net Income decreasing while Debt increases" combination from the
    unusual-pattern list, so detect_unusual_patterns() does not repeat it.
    """
    debt = compute_simple_change("total_debt", current_data, previous_data)
    net_income = get_comparison(analysis_result, "net_income")
    if debt is None or not comparison_is_calculated(net_income):
        return []

    net_income_change = net_income.get("percentage_change")
    min_pct = thresholds["unusual_pattern_min_pct"]
    if net_income_change is None:
        return []
    if not (debt["percentage_change"] >= min_pct and net_income_change <= -min_pct):
        return []

    crossed_full_threshold = (
        debt["percentage_change"] >= thresholds["debt_increase_pct"]
        or net_income_change <= -thresholds["net_income_decline_pct"]
    )
    severity = SEVERITY_MODERATE if crossed_full_threshold else SEVERITY_INFORMATIONAL
    years = get_fiscal_years(analysis_result)
    description = (
        f"Total Debt increased {debt['percentage_change']}% while Net Income declined "
        f"{abs(net_income_change)}% between FY{years[0]} and FY{years[1]}."
    )
    return [make_flag(
        "debt_rising_profitability_declining", severity, "Total Debt vs Net Income",
        None, None, years, description, source=debt,
    )]


def detect_unusual_patterns(analysis_result, thresholds, current_data=None, previous_data=None):
    """
    Category 8: metrics moving in opposite directions. These describe a
    numerical combination only; they do not imply a cause or wrongdoing.
    """
    years = get_fiscal_years(analysis_result)
    min_pct = thresholds["unusual_pattern_min_pct"]
    min_points = thresholds["unusual_pattern_min_points"]

    def value_of(metric_key, field):
        comparison = get_comparison(analysis_result, metric_key)
        return comparison.get(field) if comparison_is_calculated(comparison) else None

    revenue = value_of("revenue", "percentage_change")
    net_income = value_of("net_income", "percentage_change")
    eps = value_of("eps", "percentage_change")
    operating_margin = value_of("operating_margin", "margin_change_points")

    patterns = []
    if revenue is not None and net_income is not None \
            and revenue >= min_pct and net_income <= -min_pct:
        patterns.append((
            "Revenue increased while Net Income decreased", "Revenue / Net Income",
            f"Revenue changed {revenue:+}% while Net Income changed {net_income:+}%",
        ))
    if revenue is not None and operating_margin is not None \
            and revenue >= min_pct and operating_margin <= -min_points:
        patterns.append((
            "Revenue increased while Operating Margin decreased", "Revenue / Operating Margin",
            f"Revenue changed {revenue:+}% while Operating Margin changed "
            f"{operating_margin:+} percentage points",
        ))
    if eps is not None and revenue is not None \
            and eps <= -min_pct and revenue >= min_pct:
        patterns.append((
            "EPS decreased while Revenue increased", "EPS / Revenue",
            f"EPS changed {eps:+}% while Revenue changed {revenue:+}%",
        ))

    return [
        make_flag(
            "unusual_pattern", SEVERITY_INFORMATIONAL, metric, None, None, years,
            f"{title} between FY{years[0]} and FY{years[1]} ({detail}).",
        )
        for title, metric, detail in patterns
    ]


# ---------------------------------------------------------------------------
# Data limitations, summary, and the main entry point
# ---------------------------------------------------------------------------

def collect_data_limitations(analysis_result, current_data, previous_data):
    """
    List every red-flag category that could not be evaluated, and why, so a
    missing metric is never mistaken for "nothing wrong".
    """
    limitations = []

    labels = [("revenue", "Revenue"), ("net_income", "Net Income"), ("eps", "EPS")] + MARGIN_LABELS
    for metric_key, label in labels:
        comparison = get_comparison(analysis_result, metric_key)
        if not comparison_is_calculated(comparison):
            reason = (comparison or {}).get("reason") or "no comparison was provided by Agent 2."
            limitations.append(f"{label} could not be evaluated: {reason}")

    for metric_key, label in [("total_debt", "Total Debt"), ("cash_flow", "Cash Flow")]:
        if compute_simple_change(metric_key, current_data, previous_data) is None:
            if not current_data or not previous_data:
                limitations.append(
                    f"{label} was not evaluated: the raw Agent 1 records for both fiscal "
                    f"years were not supplied (Agent 2 does not compare {label})."
                )
            else:
                limitations.append(
                    f"{label} could not be evaluated: the value is missing, NOT_FOUND, zero "
                    f"in the previous year, or reported in different units/currency across years."
                )

    return limitations


def build_summary_text(analysis_result, flags, data_limitations):
    """One factual paragraph. Reports counts only; no overall risk rating."""
    company = analysis_result.get("company")
    years = get_fiscal_years(analysis_result)

    if flags:
        counts = Counter(flag["severity"] for flag in flags)
        breakdown = ", ".join(f"{count} {severity}" for severity, count in counts.items())
        text = (f"{len(flags)} potential warning signal(s) identified for {company} between "
                f"FY{years[0]} and FY{years[1]} ({breakdown}). Each is a screening flag for one "
                f"numerical pattern, not an overall risk rating.")
    else:
        text = (f"No configured screening thresholds were crossed for {company} between "
                f"FY{years[0]} and FY{years[1]} using the metrics that were available.")

    if data_limitations:
        text += (f" {len(data_limitations)} category(ies) could not be evaluated because of "
                 f"missing or incompatible data, so the absence of a flag there is not a finding.")
    return text


def build_red_flags(analysis_result, thresholds=None, current_data=None, previous_data=None):
    """
    Main entry point. Runs every detector against an Agent 2 result and
    returns structured data:
        {"company", "fiscal_years", "flags", "summary", "data_limitations"}
    Never raises for missing data; problems are reported in data_limitations.
    """
    is_valid, problems = validate_analysis_result(analysis_result)
    if not is_valid:
        return {
            "company": (analysis_result or {}).get("company") if isinstance(analysis_result, dict) else None,
            "fiscal_years": [],
            "flags": [],
            "summary": "The red-flag screen could not run because the analysis input was invalid.",
            "data_limitations": problems,
        }

    active_thresholds = dict(RED_FLAG_THRESHOLDS)
    active_thresholds.update(thresholds or {})

    detectors = [
        detect_revenue_decline,
        detect_net_income_decline,
        detect_eps_decline,
        detect_margin_compression,
        detect_rising_debt,
        detect_cash_flow_warning,
        detect_debt_vs_profitability,
        detect_unusual_patterns,
    ]
    flags = []
    for detector in detectors:
        flags.extend(detector(analysis_result, active_thresholds, current_data, previous_data))

    data_limitations = collect_data_limitations(analysis_result, current_data, previous_data)

    return {
        "company": analysis_result["company"],
        "fiscal_years": get_fiscal_years(analysis_result),
        "flags": flags,
        "summary": build_summary_text(analysis_result, flags, data_limitations),
        "data_limitations": data_limitations,
    }


def format_red_flag_report(result):
    """Render a build_red_flags() result as readable text."""
    years = result.get("fiscal_years") or ["?", "?"]
    lines = [
        "=" * 60, "RED FLAG ANALYSIS", "",
        f"Company: {result.get('company')}",
        f"Fiscal Years: FY{years[0]} -> FY{years[1]}", "",
    ]

    flags = result.get("flags", [])
    if not flags:
        lines.append("No screening thresholds were crossed for the evaluated metrics.")
        lines.append("")
    for index, flag in enumerate(flags, start=1):
        lines.append(f"[{index}] {flag['type']}  (severity: {flag['severity']})")
        lines.append(f"    {flag['description']}")
        if flag.get("source_file"):
            lines.append(f"    Source: {flag['source_file']} (page {flag.get('page_number')})")
        lines.append("")

    lines.append("-" * 60)
    lines.append("SUMMARY")
    lines.append(result.get("summary", ""))

    if result.get("data_limitations"):
        lines.append("")
        lines.append("DATA LIMITATIONS")
        for limitation in result["data_limitations"]:
            lines.append(f"  - {limitation}")

    lines.append("")
    lines.append("These are descriptive screening flags based on configurable thresholds.")
    lines.append("They are not investment advice and not an overall risk rating.")
    return "\n".join(lines)