"""
test_red_flag_agent.py

Tests for Agent 3 (red_flag_agent.py).

How the inputs are built
------------------------
Every test builds hand-made Agent-1-shaped records (MOCK data, with round
numbers so results can be checked by eye) and passes them through the REAL
analysis_agent.analyze_company(). The Red Flag Agent therefore receives the
genuine Agent 2 output structure, not a structure invented for the tests.

None of these tests touch ChromaDB, the embedding model, or your real
annual-report data. Test 8 additionally checks that neither chromadb nor
sentence_transformers gets imported while the Red Flag Agent runs.

Run:
    python test_red_flag_agent.py
"""

import sys

from analysis_agent import analyze_company
from red_flag_agent import (
    RED_FLAG_THRESHOLDS, build_red_flags, format_red_flag_report,
)


# ---------------------------------------------------------------------------
# Helpers: build Agent 1 style records and a real Agent 2 result
# ---------------------------------------------------------------------------

def record(metric, value, unit, currency, year, status="found",
           source_file=None, page_number=None, derived=False):
    return {
        "metric": metric, "company": "TestCo", "fiscal_year": year,
        "value": value, "unit": unit, "currency": currency,
        "source_file": source_file, "page_number": page_number,
        "evidence": "(mock evidence text)" if source_file else None,
        "status": status, "derived": derived,
    }


def build_year_data(year, revenue=None, net_income=None, eps=None,
                    gross_margin=None, operating_margin=None, debt=None,
                    cash_flow=None, source_file=None, page_number=None):
    """Turn plain numbers into an Agent 1 style dict; None means NOT_FOUND."""
    def rec(metric, value, unit, currency):
        if value is None:
            return record(metric, None, None, None, year, status="not_found")
        return record(metric, value, unit, currency, year,
                      source_file=source_file, page_number=page_number)

    return {
        "revenue": rec("Revenue", revenue, "million USD", "USD"),
        "net_income": rec("Net Income", net_income, "million USD", "USD"),
        "eps": rec("EPS", eps, "USD per share", "USD"),
        "gross_margin": rec("Gross Margin", gross_margin, "percent", None),
        "operating_margin": rec("Operating Margin", operating_margin, "percent", None),
        "total_debt": rec("Total Debt", debt, "million USD", "USD"),
        "cash_flow": rec("Cash Flow from Operating Activities", cash_flow, "million USD", "USD"),
    }


def run_screen(previous, current, thresholds=None, pass_raw_records=True):
    """previous/current: dicts of kwargs for build_year_data."""
    previous_data = build_year_data("2023", **previous)
    current_data = build_year_data("2024", **current)
    analysis = analyze_company("TestCo", "2023", "2024", previous_data, current_data)
    if pass_raw_records:
        return build_red_flags(analysis, thresholds=thresholds,
                               current_data=current_data, previous_data=previous_data)
    return build_red_flags(analysis, thresholds=thresholds)


def flags_of_type(result, flag_type):
    return [flag for flag in result["flags"] if flag["type"] == flag_type]


def check(label, condition):
    print(f"  [{'PASS' if condition else 'FAIL'}] {label}")
    return bool(condition)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_1_revenue_decline():
    print("\nTEST 1 -- Revenue decline (MOCK: Revenue 100 -> 90)")
    result = run_screen({"revenue": 100.0}, {"revenue": 90.0})
    flags = flags_of_type(result, "declining_revenue")
    print(f"  Flags found: {len(flags)}" + (f" | {flags[0]['description']}" if flags else ""))
    ok1 = check("a declining_revenue flag was raised", len(flags) == 1)
    ok2 = check("change_percent == -10.0 (hand-calculated)",
                bool(flags) and flags[0]["change_percent"] == -10.0)
    ok3 = check("previous/current values are 100 and 90",
                bool(flags) and flags[0]["previous_value"] == 100.0 and flags[0]["current_value"] == 90.0)
    ok4 = check("fiscal years are ['2023', '2024']",
                bool(flags) and flags[0]["fiscal_years"] == ["2023", "2024"])
    return all([ok1, ok2, ok3, ok4])


def test_2_net_income_decline():
    print("\nTEST 2 -- Net Income decline (MOCK: Net Income 100 -> 70)")
    result = run_screen({"net_income": 100.0}, {"net_income": 70.0})
    flags = flags_of_type(result, "declining_net_income")
    print(f"  Flags found: {len(flags)}" + (f" | {flags[0]['description']}" if flags else ""))
    ok1 = check("a declining_net_income flag was raised", len(flags) == 1)
    ok2 = check("change_percent == -30.0 (hand-calculated)",
                bool(flags) and flags[0]["change_percent"] == -30.0)
    return all([ok1, ok2])


def test_3_margin_compression():
    print("\nTEST 3 -- Margin compression (MOCK: Operating Margin 25% -> 20%)")
    result = run_screen({"operating_margin": 25.0}, {"operating_margin": 20.0})
    flags = flags_of_type(result, "margin_compression")
    print(f"  Flags found: {len(flags)}" + (f" | {flags[0]['description']}" if flags else ""))
    ok1 = check("a margin_compression flag was raised", len(flags) == 1)
    ok2 = check("flag is for Operating Margin", bool(flags) and flags[0]["metric"] == "Operating Margin")
    ok3 = check("change_points == -5.0 (percentage POINTS, hand-calculated)",
                bool(flags) and flags[0]["change_points"] == -5.0)
    ok4 = check("change_percent is not used for a margin", bool(flags) and flags[0]["change_percent"] is None)
    return all([ok1, ok2, ok3, ok4])


def test_4_rising_debt():
    print("\nTEST 4 -- Rising debt (MOCK: Total Debt 100 -> 130)")
    result = run_screen({"debt": 100.0}, {"debt": 130.0})
    flags = flags_of_type(result, "rising_debt")
    print(f"  Flags found: {len(flags)}" + (f" | {flags[0]['description']}" if flags else ""))
    ok1 = check("a rising_debt flag was raised", len(flags) == 1)
    ok2 = check("change_percent == 30.0 (hand-calculated)",
                bool(flags) and flags[0]["change_percent"] == 30.0)
    return all([ok1, ok2])


def test_5_unusual_pattern():
    print("\nTEST 5 -- Revenue up but Net Income down (MOCK: Revenue +10%, Net Income -15%)")
    result = run_screen({"revenue": 1000.0, "net_income": 200.0},
                        {"revenue": 1100.0, "net_income": 170.0})
    flags = flags_of_type(result, "unusual_pattern")
    print(f"  Flags found: {len(flags)}" + (f" | {flags[0]['description']}" if flags else ""))
    ok1 = check("an unusual_pattern flag was raised", len(flags) >= 1)
    ok2 = check("it describes Revenue up / Net Income down",
                any("Revenue increased while Net Income decreased" in f["description"] for f in flags))
    ok3 = check("it does not use accusatory wording",
                not any(word in f["description"].lower()
                        for f in flags for word in ("fraud", "manipulat", "wrongdoing", "misconduct")))
    return all([ok1, ok2, ok3])


def test_6_no_false_flag():
    print("\nTEST 6 -- No false flag (MOCK: Revenue +10%, Net Income +15%, margins up)")
    previous = {"revenue": 1000.0, "net_income": 100.0, "eps": 1.0,
                "gross_margin": 40.0, "operating_margin": 20.0, "debt": 100.0, "cash_flow": 150.0}
    current = {"revenue": 1100.0, "net_income": 115.0, "eps": 1.1,
               "gross_margin": 41.0, "operating_margin": 21.0, "debt": 100.0, "cash_flow": 160.0}
    result = run_screen(previous, current)
    print(f"  Flags found: {len(result['flags'])}")
    ok1 = check("no flags at all were raised", len(result["flags"]) == 0)
    ok2 = check("summary states no thresholds were crossed",
                "No configured screening thresholds were crossed" in result["summary"])
    return all([ok1, ok2])


def test_7_missing_data():
    print("\nTEST 7 -- Missing data (MOCK: every metric NOT_FOUND)")
    result = run_screen({}, {})
    print(f"  Flags found: {len(result['flags'])}, data limitations: {len(result['data_limitations'])}")
    for limitation in result["data_limitations"][:3]:
        print(f"    - {limitation}")
    ok1 = check("no flags were fabricated", len(result["flags"]) == 0)
    ok2 = check("data limitations were reported", len(result["data_limitations"]) >= 7)
    ok3 = check("Revenue limitation mentions it could not be evaluated",
                any(l.startswith("Revenue could not be evaluated") for l in result["data_limitations"]))
    ok4 = check("Total Debt and Cash Flow are listed as not evaluated",
                any("Total Debt" in l for l in result["data_limitations"])
                and any("Cash Flow" in l for l in result["data_limitations"]))
    ok5 = check("summary warns that no flag is not the same as a finding",
                "not a finding" in result["summary"])

    # A partly-missing case: previous-year revenue missing must not be calculated.
    partial = run_screen({"net_income": 100.0}, {"revenue": 90.0, "net_income": 70.0})
    ok6 = check("partial data: no revenue flag when previous revenue is NOT_FOUND",
                len(flags_of_type(partial, "declining_revenue")) == 0)
    ok7 = check("partial data: the available Net Income decline is still flagged",
                len(flags_of_type(partial, "declining_net_income")) == 1)
    return all([ok1, ok2, ok3, ok4, ok5, ok6, ok7])


def test_8_agent2_integration():
    print("\nTEST 8 -- Agent 2 integration (real analyze_company() output, no ChromaDB)")
    modules_before = {"chromadb", "sentence_transformers"} & set(sys.modules)

    previous_data = build_year_data("2023", revenue=383285.0, net_income=96995.0, eps=6.13,
                                    gross_margin=44.1, operating_margin=29.98,
                                    debt=111088.0, cash_flow=110543.0)
    current_data = build_year_data("2024", revenue=391035.0, net_income=93736.0, eps=6.08,
                                   gross_margin=46.2, operating_margin=31.51,
                                   debt=106629.0, cash_flow=118254.0)
    analysis = analyze_company("TestCo", "2023", "2024", previous_data, current_data)

    print(f"  Agent 2 result keys: {sorted(analysis.keys())}")
    print(f"  Agent 2 comparison keys: {sorted(analysis['comparisons'].keys())}")

    result = build_red_flags(analysis, current_data=current_data, previous_data=previous_data)
    report = format_red_flag_report(result)
    print(f"  Red Flag Agent returned {len(result['flags'])} flag(s)")

    modules_after = {"chromadb", "sentence_transformers"} & set(sys.modules)
    ok1 = check("result has company, fiscal_years, flags, summary, data_limitations",
                all(k in result for k in ("company", "fiscal_years", "flags", "summary", "data_limitations")))
    ok2 = check("company and years carried over from Agent 2",
                result["company"] == "TestCo" and result["fiscal_years"] == ["2023", "2024"])
    ok3 = check("chromadb / sentence_transformers were NOT imported by the run",
                modules_after == modules_before)
    ok4 = check("a printable report was produced", "RED FLAG ANALYSIS" in report)
    ok5 = check("net income drop (-3.36%) is below the 5% threshold, so it is not flagged",
                len(flags_of_type(result, "declining_net_income")) == 0)
    ok6 = check("the report says it is not investment advice", "not investment advice" in report)

    # Threshold configurability: a stricter threshold now catches the same drop.
    strict = build_red_flags(analysis, thresholds={"net_income_decline_pct": 3.0},
                             current_data=current_data, previous_data=previous_data)
    ok7 = check("a custom threshold (3%) makes the same -3.36% drop flag",
                len(flags_of_type(strict, "declining_net_income")) == 1)
    ok8 = check("default thresholds dict was not mutated by the override",
                RED_FLAG_THRESHOLDS["net_income_decline_pct"] == 5.0)

    print("\n  --- Sample report (Apple-like FY2023 -> FY2024, MOCK numbers) ---")
    for line in report.splitlines()[:14]:
        print("  " + line)
    return all([ok1, ok2, ok3, ok4, ok5, ok6, ok7, ok8])


def test_9_source_preservation():
    print("\nTEST 9 -- Source preservation (MOCK records carrying source_file / page_number)")
    previous_data = build_year_data("2023", revenue=100.0, debt=100.0)
    current_data = build_year_data("2024", revenue=90.0, debt=130.0,
                                   source_file="TestCo_2024_Annual_report.pdf", page_number=32)
    analysis = analyze_company("TestCo", "2023", "2024", previous_data, current_data)

    with_records = build_red_flags(analysis, current_data=current_data, previous_data=previous_data)
    revenue_flag = flags_of_type(with_records, "declining_revenue")[0]
    debt_flag = flags_of_type(with_records, "rising_debt")[0]
    print(f"  Revenue flag source: {revenue_flag['source_file']} p.{revenue_flag['page_number']}")
    print(f"  Debt flag source:    {debt_flag['source_file']} p.{debt_flag['page_number']}")

    ok1 = check("revenue flag keeps source_file", revenue_flag["source_file"] == "TestCo_2024_Annual_report.pdf")
    ok2 = check("revenue flag keeps page_number", revenue_flag["page_number"] == 32)
    ok3 = check("debt flag keeps source_file and page_number",
                debt_flag["source_file"] == "TestCo_2024_Annual_report.pdf" and debt_flag["page_number"] == 32)
    ok4 = check("evidence text is carried through", revenue_flag["evidence"] == "(mock evidence text)")

    # Without raw Agent 1 records, the source fields stay None: nothing invented.
    without_records = build_red_flags(analysis)
    bare_flag = flags_of_type(without_records, "declining_revenue")[0]
    ok5 = check("without raw records, source fields are None (nothing invented)",
                bare_flag["source_file"] is None and bare_flag["page_number"] is None)
    return all([ok1, ok2, ok3, ok4, ok5])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("1. Revenue decline", test_1_revenue_decline),
        ("2. Net Income decline", test_2_net_income_decline),
        ("3. Margin compression", test_3_margin_compression),
        ("4. Rising debt", test_4_rising_debt),
        ("5. Revenue up / Net Income down", test_5_unusual_pattern),
        ("6. No false flag", test_6_no_false_flag),
        ("7. Missing data", test_7_missing_data),
        ("8. Agent 2 integration", test_8_agent2_integration),
        ("9. Source preservation", test_9_source_preservation),
    ]

    outcomes = []
    for name, test_function in tests:
        try:
            passed = test_function()
        except Exception as error:  # a crash is a failure, but keep running the rest
            print(f"  [ERROR] {name} raised {type(error).__name__}: {error}")
            passed = False
        outcomes.append((name, passed))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in outcomes:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")
    print(f"\n{sum(1 for _, p in outcomes if p)}/{len(outcomes)} tests passed.")