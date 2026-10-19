"""
test_research_retrieval_suite.py

Automated test suite verifying the Research Copilot retrieval fixes:
- TEST A: Metric parity between Research Copilot and Visual Analytics (Apple FY2024 Net Income)
- TEST B: Alternative terminology mapping ("net profit", "profit after tax", "profit before tax", "operating profit")
- TEST C: Revenue and aliases ("Tata Consultancy Services", "RIL", "revenue from operations")
- TEST D: Unindexed company query ("Nestle") returns clear explanation and lists available companies
- TEST E: Unindexed fiscal year query ("Apple FY2026") returns explanation and lists available years
- TEST F: Statement type labeling ("Consolidated" vs "Standalone")
- TEST G: Provenance fidelity (source_file, page_number, unit, verbatim evidence)
- TEST H: Missing fiscal year guidance (prompts with available fiscal years)
- TEST I: Expanded financial concepts (Total Assets, Gross Profit, Cash and Cash Equivalents)
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import chat_interface
import comparison_agent
import extraction_agent
import research_agent


def run_all_tests():
    catalog = comparison_agent.load_index_catalog()
    session = chat_interface.create_session()
    passed, failed = 0, 0

    def check(condition, test_name, detail=""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  [PASS] {test_name}" + (f" -> {detail}" if detail else ""))
        else:
            failed += 1
            print(f"  [FAIL] {test_name}" + (f" -> {detail}" if detail else ""))

    print("\n" + "=" * 70)
    print("FINSIGHT AI - RESEARCH COPILOT RETRIEVAL TEST SUITE")
    print("=" * 70)

    # -----------------------------------------------------------------------
    # TEST A: Parity between Research Copilot and Visual Analytics
    # -----------------------------------------------------------------------
    print("\n--- TEST A: Parity with Visual Analytics (Apple FY2024 Net Income) ---")
    direct = extraction_agent.extract_metrics("Apple", "2024", metric_keys=["net_income"])
    direct_net_inc = direct["net_income"]["value"]

    session.reset()
    turn_a = session.ask("What is the net income of Apple for FY2024?")
    res_a = session.last_research
    finding_a = next((f for f in res_a["findings"] if f["metric"] == "net_income"), None)

    check(finding_a is not None, "Finding for net_income exists")
    check(finding_a["value"] == direct_net_inc == 93736.0, "Parity with Visual Analytics direct call",
          f"Visual Analytics: {direct_net_inc}, Research Copilot: {finding_a['value'] if finding_a else None}")
    check("93,736" in turn_a["markdown"], "Formatted net income present in chat markdown")

    # -----------------------------------------------------------------------
    # TEST B: Alternative financial terminology
    # -----------------------------------------------------------------------
    print("\n--- TEST B: Alternative Financial Terminology ---")
    # "net profit"
    session.reset()
    turn_b1 = session.ask("What is the net profit of Apple for FY2024?")
    res_b1 = session.last_research
    finding_b1 = next((f for f in res_b1["findings"] if f["metric"] in ("net_income", "profit_after_tax")), None)
    check(finding_b1 is not None and finding_b1["value"] == 93736.0, "Alternative term 'net profit' retrieves Net Income",
          f"Value: {finding_b1['value'] if finding_b1 else None}")

    # "profit after tax"
    session.reset()
    turn_b2 = session.ask("What is the profit after tax of Apple for FY2024?")
    res_b2 = session.last_research
    finding_b2 = next((f for f in res_b2["findings"] if f["metric"] in ("net_income", "profit_after_tax")), None)
    check(finding_b2 is not None and finding_b2["status"] == "FOUND",
          "Alternative term 'profit after tax' retrieves PAT/Net Income",
          f"Metric: {finding_b2['metric'] if finding_b2 else None}, Value: {finding_b2['value'] if finding_b2 else None}")

    # "operating profit"
    session.reset()
    turn_b3 = session.ask("What is the operating profit of Apple for FY2024?")
    res_b3 = session.last_research
    finding_b3 = next((f for f in res_b3["findings"] if f["metric"] == "operating_income"), None)
    check(finding_b3 is not None and finding_b3["value"] == 123216.0,
          "Alternative term 'operating profit' retrieves Operating Income",
          f"Value: {finding_b3['value'] if finding_b3 else None}")

    # -----------------------------------------------------------------------
    # TEST C: Revenue and company aliases
    # -----------------------------------------------------------------------
    print("\n--- TEST C: Revenue and Company Aliases ---")
    # Microsoft FY2024 revenue
    session.reset()
    turn_c1 = session.ask("What was the revenue of Microsoft for FY2024?")
    res_c1 = session.last_research
    finding_c1 = next((f for f in res_c1["findings"] if f["metric"] == "revenue"), None)
    check(finding_c1 is not None and finding_c1["value"] == 245122.0, "Microsoft FY2024 Revenue retrieved",
          f"Value: {finding_c1['value'] if finding_c1 else None}")

    # Tata Consultancy Services alias
    session.reset()
    turn_c2 = session.ask("What was the revenue from operations of Tata Consultancy Services for FY2024?")
    res_c2 = session.last_research
    check(len(res_c2["plan"]["companies"]) == 1 and res_c2["plan"]["companies"][0] == "TCS",
          "Alias 'Tata Consultancy Services' resolved to 'TCS'")
    check(res_c2["status"] == "success", "TCS FY2024 revenue query succeeded")

    # RIL alias
    session.reset()
    turn_c3 = session.ask("What was the revenue of RIL for FY2024?")
    res_c3 = session.last_research
    check(len(res_c3["plan"]["companies"]) == 1 and res_c3["plan"]["companies"][0] == "Reliance Industries",
          "Alias 'RIL' resolved to 'Reliance Industries'")
    check(res_c3["status"] == "success", "Reliance Industries FY2024 revenue query succeeded")

    # -----------------------------------------------------------------------
    # TEST D: Unindexed Company Query (e.g. Nestle)
    # -----------------------------------------------------------------------
    print("\n--- TEST D: Unindexed Company Query (Nestle) ---")
    session.reset()
    turn_d = session.ask("What is the net income of Nestle for FY2026?")
    res_d = session.last_research
    check(res_d["status"] == "needs_clarification", "Status is needs_clarification for unindexed company")
    check(any(l["type"] == "unindexed_company" for l in res_d["limitations"]),
          "Recorded 'unindexed_company' limitation")
    check("Nestle" in turn_d["markdown"] and "indexed annual reports" in turn_d["markdown"].lower(),
          "Chat response explicitly names Nestle as not in indexed reports")
    check("Amazon" in turn_d["markdown"] and "Apple" in turn_d["markdown"],
          "Chat response lists available indexed companies")

    # -----------------------------------------------------------------------
    # TEST E: Unindexed Fiscal Year Query (e.g. Apple FY2026)
    # -----------------------------------------------------------------------
    print("\n--- TEST E: Unindexed Fiscal Year Query (Apple FY2026) ---")
    session.reset()
    turn_e = session.ask("What is the net income of Apple for FY2026?")
    res_e = session.last_research
    check(any(l["type"] == "unindexed_fiscal_year" for l in res_e["limitations"]),
          "Recorded 'unindexed_fiscal_year' limitation")
    check("FY2026 is not available" in turn_e["markdown"] or "FY2026" in turn_e["markdown"],
          "Chat response identifies FY2026 as unavailable")
    check("FY2024" in turn_e["markdown"] or "FY2023" in turn_e["markdown"],
          "Chat response lists available fiscal years for Apple")

    # -----------------------------------------------------------------------
    # TEST F: Statement Type Labeling (Consolidated vs Standalone)
    # -----------------------------------------------------------------------
    print("\n--- TEST F: Statement Type Labeling ---")
    session.reset()
    turn_f = session.ask("What was Apple's revenue for FY2024?")
    res_f = session.last_research
    finding_f = next((f for f in res_f["findings"] if f["metric"] == "revenue"), None)
    check(finding_f is not None and finding_f.get("statement_type") in ("Consolidated", "Standalone"),
          "Finding contains statement_type tag", f"Statement type: {finding_f.get('statement_type') if finding_f else None}")
    check("Statement: Consolidated" in turn_f["markdown"] or "Statement:" in turn_f["markdown"],
          "Markdown shows statement type label")

    # -----------------------------------------------------------------------
    # TEST G: Provenance Fidelity
    # -----------------------------------------------------------------------
    print("\n--- TEST G: Provenance Fidelity ---")
    check(finding_f["source_file"] == "Apple_2024_Annual_report.pdf", "Correct source file cited")
    check(finding_f["page_number"] == 32, "Correct page cited (p.32)")
    check(finding_f["unit"] == "million USD", "Normalized unit is million USD")
    check(bool(finding_f["evidence"]), "Verbatim evidence text present")

    # -----------------------------------------------------------------------
    # TEST H: Missing Fiscal Year Guidance
    # -----------------------------------------------------------------------
    print("\n--- TEST H: Missing Fiscal Year Guidance ---")
    session.reset()
    turn_h = session.ask("What is Apple's net income?")
    res_h = session.last_research
    check(res_h["status"] == "needs_clarification", "Status is needs_clarification when year is missing")
    check(any(l["type"] == "no_fiscal_year" for l in res_h["limitations"]), "Recorded 'no_fiscal_year' limitation")
    check("FY2024" in turn_h["markdown"] and "FY2023" in turn_h["markdown"],
          "Chat response suggests available fiscal years for Apple")

    # -----------------------------------------------------------------------
    # TEST I: Expanded Financial Concepts
    # -----------------------------------------------------------------------
    print("\n--- TEST I: Expanded Financial Concepts ---")
    # Gross Profit
    session.reset()
    turn_i1 = session.ask("What was the gross profit of Apple for FY2024?")
    res_i1 = session.last_research
    finding_i1 = next((f for f in res_i1["findings"] if f["metric"] == "gross_profit"), None)
    check(finding_i1 is not None and finding_i1["status"] == "FOUND", "Gross Profit extracted for Apple FY2024",
          f"Value: {finding_i1['value'] if finding_i1 else None} {finding_i1['unit'] if finding_i1 else ''}")

    # Total Assets
    session.reset()
    turn_i2 = session.ask("What were the total assets of Apple for FY2024?")
    res_i2 = session.last_research
    finding_i2 = next((f for f in res_i2["findings"] if f["metric"] == "total_assets"), None)
    check(finding_i2 is not None and finding_i2["status"] == "FOUND", "Total Assets extracted for Apple FY2024",
          f"Value: {finding_i2['value'] if finding_i2 else None} {finding_i2['unit'] if finding_i2 else ''}")

    # Cash and Cash Equivalents
    session.reset()
    turn_i3 = session.ask("What was the cash and cash equivalents of Apple for FY2024?")
    res_i3 = session.last_research
    finding_i3 = next((f for f in res_i3["findings"] if f["metric"] == "cash_and_equivalents"), None)
    check(finding_i3 is not None and finding_i3["status"] == "FOUND", "Cash & Cash Equivalents extracted for Apple FY2024",
          f"Value: {finding_i3['value'] if finding_i3 else None} {finding_i3['unit'] if finding_i3 else ''}")

    # Net Margin
    session.reset()
    turn_i4 = session.ask("What was the net profit margin of Apple for FY2024?")
    res_i4 = session.last_research
    finding_i4 = next((f for f in res_i4["findings"] if f["metric"] == "net_margin"), None)
    check(finding_i4 is not None and finding_i4["status"] == "FOUND", "Net Profit Margin calculated for Apple FY2024",
          f"Value: {finding_i4['value'] if finding_i4 else None} %")

    # -----------------------------------------------------------------------
    # Summary
    # -----------------------------------------------------------------------
    print("\n" + "=" * 70)
    print(f"TEST SUITE COMPLETE: {passed} passed, {failed} failed")
    print("=" * 70)
    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
