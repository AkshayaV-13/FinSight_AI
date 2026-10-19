"""
test_analysis_agent.py

Dedicated test suite for Agent 2 (analysis_agent.py).

Tests 1-6 use mock Agent-1-shaped records built by hand in this file --
labeled MOCK throughout, and the numbers are checked by hand in each
test's comment so you can verify them independently of the code.

Test 7 calls the REAL extraction_agent.extract_metrics() function (the
actual Agent 1 code path executes) but against a SIMULATED ChromaDB
backend, not your live vectordb/. This chat environment has no access to
your real database, so this test cannot be a true end-to-end run against
your real 4,995 chunks -- that must be run on your machine. Test 7 is
labeled accordingly; do not read it as proof against your real data.

Run:
    python test_analysis_agent.py
"""

import sys
import types

from analysis_agent import (
    analyze_company, compare_metric, compare_companies,
    calculate_percentage_change, calculate_margin_change,
    calculate_net_profit_margin, format_report,
)


def mock_record(metric, value, unit, currency, company, year, status="found", derived=False):
    return {
        "metric": metric, "company": company, "fiscal_year": year,
        "value": value, "unit": unit, "currency": currency,
        "source_file": f"{company}_{year}.pdf", "page_number": 30,
        "evidence": "(mock evidence)", "status": status, "derived": derived,
    }


def check(label, condition):
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {label}")
    return condition


# ---------------------------------------------------------------------------
# Test 1 -- Year-over-year comparison (Apple FY2023 -> FY2024), MOCK data
# ---------------------------------------------------------------------------

def test_1_year_over_year():
    print("\nTEST 1 -- Year-over-year comparison (MOCK data: Apple FY2023 -> FY2024)")

    previous = {
        "revenue": mock_record("Revenue", 383285.0, "million USD", "USD", "Apple", "2023"),
        "net_income": mock_record("Net Income", 96995.0, "million USD", "USD", "Apple", "2023"),
        "eps": mock_record("EPS", 6.13, "USD per share", "USD", "Apple", "2023"),
        "gross_margin": mock_record("Gross Margin", 44.1, "percent", None, "Apple", "2023"),
        "operating_margin": mock_record("Operating Margin", 29.98, "percent", None, "Apple", "2023"),
    }
    current = {
        "revenue": mock_record("Revenue", 391035.0, "million USD", "USD", "Apple", "2024"),
        "net_income": mock_record("Net Income", 93736.0, "million USD", "USD", "Apple", "2024"),
        "eps": mock_record("EPS", 6.08, "USD per share", "USD", "Apple", "2024"),
        "gross_margin": mock_record("Gross Margin", 46.2, "percent", None, "Apple", "2024"),
        "operating_margin": mock_record("Operating Margin", 31.51, "percent", None, "Apple", "2024"),
    }

    print("  Input from Agent 1: Revenue 383,285 -> 391,035 million USD")

    result = analyze_company("Apple", "2023", "2024", previous, current)
    rev = result["comparisons"]["revenue"]
    ni = result["comparisons"]["net_income"]

    # Manual check: ((391035 - 383285) / 383285) * 100 = 2.0223...  -> 2.02
    ok1 = check("Revenue % change == 2.02 (hand-calculated)", rev["percentage_change"] == 2.02)
    ok2 = check("Revenue direction == increased", rev["direction"] == "increased")
    # Manual check: ((93736 - 96995) / 96995) * 100 = -3.3584...  -> -3.36
    ok3 = check("Net Income % change == -3.36 (hand-calculated)", ni["percentage_change"] == -3.36)
    ok4 = check("Net Income direction == decreased", ni["direction"] == "decreased")

    return all([ok1, ok2, ok3, ok4])


# ---------------------------------------------------------------------------
# Test 2 -- Missing metric (status NOT_FOUND) must not be used in a calc
# ---------------------------------------------------------------------------

def test_2_missing_metric():
    print("\nTEST 2 -- Missing metric (MOCK data: status NOT_FOUND)")

    previous_missing = mock_record("Total Debt", None, None, None, "Apple", "2023", status="not_found")
    current_found = mock_record("Total Debt", 106629.0, "million USD", "USD", "Apple", "2024")

    print("  Input from Agent 1: previous-year Total Debt status = NOT_FOUND")

    result = compare_metric("total_debt", previous_missing, current_found)
    print(f"  Result: status={result['status']}, reason='{result['reason']}'")

    ok1 = check("status == not_calculated", result["status"] == "not_calculated")
    ok2 = check("no percentage_change was fabricated", result["percentage_change"] is None)
    ok3 = check("reason mentions previous fiscal year", "previous fiscal year" in (result["reason"] or ""))
    return all([ok1, ok2, ok3])


# ---------------------------------------------------------------------------
# Test 3 -- Margin comparison: percentage-POINT change, not percentage change
# ---------------------------------------------------------------------------

def test_3_margin_points():
    print("\nTEST 3 -- Margin percentage-point calculation (MOCK data)")

    previous = mock_record("Gross Margin", 46.2, "percent", None, "Apple", "2023")
    current = mock_record("Gross Margin", 47.0, "percent", None, "Apple", "2024")

    print("  Input from Agent 1: Gross Margin 46.2% -> 47.0%")

    result = compare_metric("gross_margin", previous, current)
    print(f"  Result: margin_change_points={result['margin_change_points']}")

    # Manual check: 47.0 - 46.2 = 0.8 percentage points (NOT (47-46.2)/46.2*100 = 1.73%)
    ok1 = check("margin_change_points == 0.8 (points, not %)", result["margin_change_points"] == 0.8)
    ok2 = check("percentage_change field is untouched (None) for a margin",
                result["percentage_change"] is None)
    ok3 = check("direction == increased", result["direction"] == "increased")
    return all([ok1, ok2, ok3])


# ---------------------------------------------------------------------------
# Test 4 -- Currency safety: USD vs INR must not be diffed directly
# ---------------------------------------------------------------------------

def test_4_currency_safety():
    print("\nTEST 4 -- Currency safety (MOCK data: USD vs INR)")

    usd_record = mock_record("EPS", 6.08, "USD per share", "USD", "Apple", "2024")
    inr_record = mock_record("EPS", 45.2, "INR per share", "INR", "Infosys", "2024")

    print("  Input: Apple EPS = 6.08 USD/share, Infosys EPS = 45.2 INR/share")

    result = compare_metric("eps", usd_record, inr_record)
    print(f"  Result: status={result['status']}, reason='{result['reason']}'")

    ok1 = check("status == not_calculated (currencies differ)", result["status"] == "not_calculated")
    ok2 = check("no change/percentage_change was computed",
                result["change"] is None and result["percentage_change"] is None)
    ok3 = check("reason mentions incompatible units/currency", "incompatible" in (result["reason"] or ""))

    # Also confirm the INR value itself was never silently relabeled USD.
    ok4 = check("INR record's currency field is untouched", inr_record["currency"] == "INR")
    return all([ok1, ok2, ok3, ok4])


# ---------------------------------------------------------------------------
# Test 5 -- Multi-company comparison
# ---------------------------------------------------------------------------

def test_5_multi_company():
    print("\nTEST 5 -- Multi-company comparison (MOCK data: Apple, Microsoft, Amazon)")

    companies_data = {
        "Apple": mock_record("Revenue", 391035.0, "million USD", "USD", "Apple", "2024"),
        "Microsoft": mock_record("Revenue", 245122.0, "million USD", "USD", "Microsoft", "2024"),
        "Amazon": mock_record("Revenue", 574785.0, "million USD", "USD", "Amazon", "2024"),
    }
    print("  Input: Apple/Microsoft/Amazon Revenue, all million USD, FY2024")

    result = compare_companies("revenue", companies_data, fiscal_year="2024")
    print(f"  Result: {len(result['rows'])} comparable companies, {len(result['excluded'])} excluded")
    for row in result["rows"]:
        print(f"    {row['company']}: {row['value']} {row['unit']}")

    ok1 = check("all 3 companies are in the comparable group", len(result["rows"]) == 3)
    ok2 = check("no companies excluded", len(result["excluded"]) == 0)
    ok3 = check("no ranking/winner language present",
                "best" not in str(result).lower() and "winner" not in str(result).lower())
    return all([ok1, ok2, ok3])


# ---------------------------------------------------------------------------
# Test 6 -- Derived metric (Net Profit Margin from Revenue + Net Income)
# ---------------------------------------------------------------------------

def test_6_derived_metric():
    print("\nTEST 6 -- Derived metric: Net Profit Margin (MOCK data)")

    net_income = mock_record("Net Income", 93736.0, "million USD", "USD", "Apple", "2024")
    revenue = mock_record("Revenue", 391035.0, "million USD", "USD", "Apple", "2024")

    print("  Input: Net Income = 93,736M, Revenue = 391,035M (both million USD)")

    result = calculate_net_profit_margin(net_income, revenue)
    print(f"  Result: value={result['value']}%, derived={result['derived']}")

    # Manual check: (93736 / 391035) * 100 = 23.9700...  -> 23.97
    ok1 = check("Net Profit Margin == 23.97 (hand-calculated)", result["value"] == 23.97)
    ok2 = check("derived flag == True", result["derived"] is True)
    ok3 = check("status == calculated", result["status"] == "calculated")
    return all([ok1, ok2, ok3])


# ---------------------------------------------------------------------------
# Test 7 -- Agent 1 integration (REAL extract_metrics() code path,
#           SIMULATED ChromaDB backend -- see module docstring)
# ---------------------------------------------------------------------------

def test_7_agent1_integration():
    print("\nTEST 7 -- Agent 1 integration")
    print("  NOTE: this calls the REAL extraction_agent.extract_metrics() function,")
    print("  but against a SIMULATED ChromaDB backend built in this test file, not")
    print("  your live vectordb/. This environment has no access to your real")
    print("  database. Run this same test on your machine to validate against the")
    print("  actual 4,995 indexed chunks.")

    class FakeModel:
        def encode(self, texts, normalize_embeddings, convert_to_numpy):
            import numpy as np
            return np.array([[0.1, 0.2, 0.3]])

    st_mod = types.ModuleType("sentence_transformers")
    st_mod.SentenceTransformer = lambda name: FakeModel()
    sys.modules["sentence_transformers"] = st_mod

    statement_text = (
        "Apple Inc. CONSOLIDATED STATEMENTS OF OPERATIONS (In millions, except "
        "per share amounts) 2024 2023 2022 "
        "Total net sales $ 391,035 $ 383,285 $ 394,328 "
        "Net income $ 93,736 $ 96,995 $ 99,803 "
        "Basic earnings per share $ 6.13 $ 6.16 $ 6.15 "
        "Diluted earnings per share $ 6.08 $ 6.13 $ 6.11."
    )

    def make_collection(text, meta):
        class FakeCollection:
            def query(self, query_embeddings, n_results, where, include):
                return {"documents": [[text]], "metadatas": [[meta]], "distances": [[0.1]]}
        return FakeCollection()

    class FakeClient:
        def __init__(self, path):
            self._collection = make_collection(
                statement_text,
                {"company": "Apple", "fiscal_year": "2024",
                 "source_file": "Apple_2024_Annual_report.pdf", "page_number": 32},
            )
        def get_collection(self, name):
            return self._collection

    chroma_mod = types.ModuleType("chromadb")
    chroma_mod.PersistentClient = lambda path: FakeClient(path)
    sys.modules["chromadb"] = chroma_mod

    import extraction_agent as ea
    ea._collection = None
    ea._model = None
    ea.VECTORDB_DIR = ea.PROJECT_ROOT  # bypass the exists() check trivially

    current_data = ea.extract_metrics(
        "Apple", "2024", metric_keys=["revenue", "net_income", "eps"], top_k=1
    )
    print(f"  Agent 1 returned: revenue={current_data['revenue']['value']}, "
          f"net_income={current_data['net_income']['value']}, eps={current_data['eps']['value']}")

    from analysis_agent import validate_metric_record
    ok1 = check("Agent 1 revenue record validates in Agent 2",
                validate_metric_record(current_data["revenue"])[0])
    ok2 = check("Agent 1 revenue value == 391035.0 (real Agent 1 code path)",
                current_data["revenue"]["value"] == 391035.0)
    ok3 = check("Agent 1 net_income value == 93736.0", current_data["net_income"]["value"] == 93736.0)
    return all([ok1, ok2, ok3])


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    tests = [
        ("1. Year-over-year comparison", test_1_year_over_year),
        ("2. Missing metric handling", test_2_missing_metric),
        ("3. Margin percentage-point calculation", test_3_margin_points),
        ("4. Currency safety", test_4_currency_safety),
        ("5. Multi-company comparison", test_5_multi_company),
        ("6. Derived metric (Net Profit Margin)", test_6_derived_metric),
        ("7. Agent 1 integration (simulated backend)", test_7_agent1_integration),
    ]

    results = []
    for name, fn in tests:
        try:
            passed = fn()
        except Exception as exc:
            print(f"  [ERROR] {name} raised: {exc}")
            passed = False
        results.append((name, passed))

    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    for name, passed in results:
        print(f"  [{'PASS' if passed else 'FAIL'}] {name}")

    total_passed = sum(1 for _, p in results if p)
    print(f"\n{total_passed}/{len(results)} tests passed.")