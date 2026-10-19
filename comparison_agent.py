"""
comparison_agent.py

Comparison Agent: compares financial metrics across companies, fiscal years
and metrics, and returns structured side-by-side data with provenance.

It reuses what already exists instead of rebuilding it:
  - Agent 1  extraction_agent.extract_metrics()  supplies every value, with
             source_file / page_number / evidence / status. Nothing is extracted
             or retrieved here.
  - Agent 2  analysis_agent supplies the calculations: calculate_absolute_change,
             calculate_percentage_change, calculate_margin_change,
             validate_metric_record, and compare_metric() for year-over-year.
  Agent 2's own compare_companies() is not used, because it silently drops any
  company whose unit differs from the majority. This agent has to say WHY a
  value was not compared, so it reuses Agent 2's calculation functions and adds
  the explicit compatibility reporting itself.

Rules enforced
--------------
* Missing values stay NOT_FOUND. They are never replaced by zero.
* Fiscal years are never mixed: a record whose fiscal_year differs from the
  requested year is rejected, and every comparison is within ONE fiscal-year
  label. Labels are each annual report's own; fiscal periods can end in
  different months, so the result carries a note saying so.
* Companies are never mixed: a record whose company differs from the one
  requested is rejected.
* Currencies are never converted or silently compared. Different currencies
  (USD vs INR), an undetermined currency, or a scale that could not be
  normalized are reported as incompatible, with the reason. Percent metrics
  (margins) have no currency and can be compared across currencies.
* Compatible scales ARE normalized: "billion USD" and "thousand USD" are
  converted to "million USD" using Agent 1's own scale table.
* No ranking and no recommendation. A pairwise statement only says one company
  reported a higher or lower figure than another in the selected data.

Company names: the index can store a company under a slightly different string
than the one a person types (for example "Amazon.com Inc"). resolve names
through the index catalog (read-only metadata listing of the existing
collection) so "Amazon" finds the right chunks. If the catalog cannot be read
the names are used as typed and a limitation says so.

Usage:
    python comparison_agent.py                       # Apple Microsoft Amazon, FY2024
    python comparison_agent.py Apple Microsoft --years 2023 2024
"""

import re
import sys
from collections import Counter

import analysis_agent
from extraction_agent import SCALE_TO_MILLIONS
from models import normalize_status

DEFAULT_METRICS = ["revenue", "net_income", "eps", "operating_margin"]
SUPPORTED_METRICS = [
    "revenue", "net_income", "eps", "gross_margin", "operating_margin",
    "operating_income", "total_debt", "cash_flow",
    "total_income", "profit_before_tax", "profit_after_tax", "ebitda",
    "total_assets", "total_liabilities", "total_equity", "cash_and_equivalents",
    "investing_cash_flow", "financing_cash_flow", "gross_profit", "net_margin",
]
METRIC_LABELS = {
    "revenue": "Revenue", "net_income": "Net Income", "eps": "EPS (diluted preferred)",
    "gross_margin": "Gross Margin", "operating_margin": "Operating Margin",
    "operating_income": "Operating Income", "total_debt": "Total Debt",
    "cash_flow": "Cash Flow from Operating Activities",
    "total_income": "Total Income", "profit_before_tax": "Profit Before Tax (PBT)",
    "profit_after_tax": "Profit After Tax (PAT)", "ebitda": "EBITDA",
    "total_assets": "Total Assets", "total_liabilities": "Total Liabilities",
    "total_equity": "Total Equity", "cash_and_equivalents": "Cash and Cash Equivalents",
    "investing_cash_flow": "Cash Flow from Investing Activities",
    "financing_cash_flow": "Cash Flow from Financing Activities",
    "gross_profit": "Gross Profit", "net_margin": "Net Profit Margin",
}
FISCAL_YEAR_NOTE = ("Fiscal-year labels are each company's own annual-report label. Fiscal periods "
                    "can end in different months, so the same label is not guaranteed to cover "
                    "the same calendar period for every company.")
PER_SHARE_NOTE = ("Per-share amounts depend on each company's share count; the figures are shown "
                  "side by side, not interpreted.")


# ---------------------------------------------------------------------------
# Index catalog: map a typed company name to the name stored in the index
# ---------------------------------------------------------------------------
NAME_STOPWORDS = {"inc", "incorporated", "corp", "corporation", "ltd", "limited", "co", "company",
                  "plc", "the", "com", "industries", "integrated", "annual", "report", "comp", "and"}

COMPANY_ALIASES = {
    "tata consultancy services": "TCS",
    "tata consultancy": "TCS",
    "tcs": "TCS",
    "reliance": "Reliance Industries",
    "reliance industries": "Reliance Industries",
    "ril": "Reliance Industries",
    "microsoft": "Microsoft",
    "msft": "Microsoft",
    "amazon": "Amazon",
    "amzn": "Amazon",
    "tesla": "Tesla",
    "tsla": "Tesla",
    "infosys": "Infosys",
    "infy": "Infosys",
    "apple": "Apple",
    "aapl": "Apple",
    "nestle": "Nestle",
}


def company_key(name):
    """Comparable identity of a company name: resolves aliases and ignores legal suffixes/digits."""
    norm = (name or "").strip().lower()
    for alias, canonical in COMPANY_ALIASES.items():
        if norm == alias or norm == alias.lower():
            name = canonical
            break
    tokens = [t for t in re.findall(r"[a-z0-9]+", (name or "").lower())
              if t not in NAME_STOPWORDS and not t.isdigit()]
    return frozenset(tokens)


def load_index_catalog():
    """
    Read which (company, fiscal_year) pairs exist in the EXISTING collection.
    Read-only metadata listing; nothing is queried, written or re-indexed.
    """
    import extraction_agent
    collection = extraction_agent.get_collection()
    metadatas = collection.get(include=["metadatas"])["metadatas"]
    counts = Counter((m.get("company"), str(m.get("fiscal_year")))
                     for m in metadatas if m and m.get("company"))
    return {"counts": counts}


def pick_stored_name(name, fiscal_year, catalog):
    """Return (stored_name, None) or (None, reason) for a typed company name and fiscal year."""
    norm = (name or "").strip().lower()
    canonical = COMPANY_ALIASES.get(norm, name)
    key = company_key(canonical)
    stored_names = {stored for (stored, _year) in catalog["counts"]
                    if (company_key(stored) == key if key else stored.lower() == canonical.lower())}
    if not stored_names:
        return None, "company_not_indexed"
    with_year = {s: catalog["counts"][(s, str(fiscal_year))] for s in stored_names
                 if catalog["counts"].get((s, str(fiscal_year)))}
    if not with_year:
        return None, "fiscal_year_not_indexed"
    return max(with_year, key=with_year.get), None


# ---------------------------------------------------------------------------
# Record views and unit normalization
# ---------------------------------------------------------------------------

def normalize_monetary_unit(value, unit):
    """
    Convert "billion USD" / "thousand INR" style units to "million <CCY>" using
    Agent 1's scale table. Returns (value, unit, converted_from_or_None).
    Percent, per-share and unscaled units are returned unchanged.
    """
    tokens = (unit or "").split()
    if tokens and tokens[0] in SCALE_TO_MILLIONS and tokens[0] != "million":
        converted = round(value * SCALE_TO_MILLIONS[tokens[0]], 6)
        new_unit = " ".join(["million"] + tokens[1:])
        return converted, new_unit, f"{value} {unit}"
    return value, unit, None


def eps_type_of(record):
    """Report honestly whether Agent 1's EPS came from a diluted or a basic label."""
    candidates = record.get("candidates") or []
    phrase = str(candidates[0].get("matched_phrase", "")).lower() if candidates else ""
    return "basic" if "basic" in phrase else "diluted" if "diluted" in phrase else "unspecified"


def make_view(company, fiscal_year, metric, record, indexed_name=None, missing_reason=None):
    """One compact, provenance-carrying view of an Agent 1 record."""
    view = {
        "company": company, "indexed_name": indexed_name or company,
        "fiscal_year": str(fiscal_year), "metric": metric,
        "metric_label": METRIC_LABELS.get(metric, metric),
        "status": "NOT_FOUND", "value": None, "unit": None, "currency": None,
        "basis": None, "source_file": None, "page_number": None, "evidence": None,
        "confidence": None, "calculation": None, "derived_from": None, "warnings": [],
        "reason": missing_reason, "normalized_from": None, "eps_type": None,
        "statement_type": (record.get("statement_type", "Consolidated") if record else "Consolidated"),
    }
    if record is None:
        view["reason"] = view["reason"] or "no record was returned"
        return view

    valid, why = analysis_agent.validate_metric_record(record)
    if not valid:
        view["reason"] = ("NOT_FOUND in the indexed document" if why == "not_found"
                          else f"unusable record ({why})")
        return view
    if str(record.get("fiscal_year")) != str(fiscal_year):
        view["reason"] = (f"fiscal year mismatch: the record is FY{record.get('fiscal_year')}, "
                          f"not FY{fiscal_year}")
        return view
    if str(record.get("company", "")).lower() != (indexed_name or company).lower():
        view["reason"] = (f"company mismatch: the record belongs to {record.get('company')!r}, "
                          f"not {indexed_name or company!r}")
        return view

    value, unit, converted = normalize_monetary_unit(record["value"], record.get("unit"))
    view.update({
        "status": normalize_status(record.get("status")), "value": value, "unit": unit,
        "currency": record.get("currency"),
        "basis": "DERIVED" if record.get("derived") else "REPORTED",
        "source_file": record.get("source_file"), "page_number": record.get("page_number"),
        "evidence": record.get("evidence"), "confidence": record.get("confidence"),
        "calculation": record.get("calculation"), "derived_from": record.get("derived_from"),
        "warnings": list(record.get("warnings") or []), "normalized_from": converted,
        "reason": None,
        "statement_type": record.get("statement_type", "Consolidated"),
    })
    if metric == "eps":
        view["eps_type"] = eps_type_of(record)
    return view


def normalized_record_copy(record):
    """A copy of an Agent 1 record with units normalized, for Agent 2's compare_metric()."""
    copy = dict(record)
    if isinstance(record.get("value"), (int, float)) and not isinstance(record.get("value"), bool):
        copy["value"], copy["unit"], _ = normalize_monetary_unit(record["value"], record.get("unit"))
    return copy


def signature_of(view):
    """(unit, currency) used to decide whether two values can be compared directly."""
    unit = view["unit"]
    if unit == "percent":
        return ("percent", None)  # a ratio has no currency
    return (unit, view["currency"])


def incompatibility_reason(signatures):
    """Plain-language reason why a set of (unit, currency) signatures cannot be compared."""
    currencies = {c for (_u, c) in signatures}
    if None in currencies and len(signatures) > 1:
        return "currency could not be determined for at least one company (not assumed)"
    if len(currencies) > 1:
        return "incompatible currencies (no conversion is applied)"
    return "incompatible units (the scale could not be normalized to the same unit)"


def format_amount(view):
    value, unit = view["value"], view["unit"]
    text = f"{value:,.2f}".rstrip("0").rstrip(".")
    return f"{text}%" if unit == "percent" else f"{text} {unit or ''}".strip()


# ---------------------------------------------------------------------------
# Comparison of one (fiscal year, metric) across companies
# ---------------------------------------------------------------------------

def pair_statement(first, second, label, year):
    """A neutral, factual sentence about two comparable values. No ranking language."""
    a, b = first["value"], second["value"]
    if a == b:
        relation, detail = "the same", ""
    else:
        relation = "a higher" if a > b else "a lower"
    if first["unit"] == "percent":
        points = analysis_agent.calculate_margin_change(a, b)
        detail = f"; difference {points:+.2f} percentage points"
    else:
        difference = analysis_agent.calculate_absolute_change(a, b)
        percent = analysis_agent.calculate_percentage_change(a, b)
        percent_text = f", {percent:+.2f}% relative to {second['company']}" if percent is not None else ""
        detail = f"; difference {difference:+,.2f} {first['unit'] or ''}".rstrip() + percent_text
    verb = "reported the same" if relation == "the same" else f"reported {relation}"
    return (f"{first['company']} {verb} {label} than {second['company']} in FY{year} "
            f"({first['company']}: {format_amount(first)}, {second['company']}: {format_amount(second)}){detail}.") \
        if relation != "the same" else (
            f"{first['company']} and {second['company']} reported the same {label} in FY{year} "
            f"({format_amount(first)}).")


def compare_metric_across_companies(year, metric, views):
    """Group by (unit, currency); compare only inside a group; report everything else explicitly."""
    label = METRIC_LABELS.get(metric, metric)
    found = [v for v in views if v["status"] == "FOUND"]
    excluded = [{"company": v["company"], "reason": v["reason"] or "NOT_FOUND"}
                for v in views if v["status"] != "FOUND"]

    groups = {}
    for view in found:
        groups.setdefault(signature_of(view), []).append(view)

    relations, not_comparable = [], []
    for members in groups.values():
        for i in range(len(members)):
            for j in range(i + 1, len(members)):
                relations.append({"first": members[i]["company"], "second": members[j]["company"],
                                  "statement": pair_statement(members[i], members[j], label, year)})

    incompatibility = None
    if len(groups) > 1:
        incompatibility = {"reason": incompatibility_reason(list(groups)),
                           "groups": {f"{u or 'unknown unit'} / {c or 'unknown currency'}": [v["company"] for v in vs]
                                      for (u, c), vs in groups.items()}}
        keys = list(groups)
        for a in range(len(keys)):
            for b in range(a + 1, len(keys)):
                for first in groups[keys[a]]:
                    for second in groups[keys[b]]:
                        not_comparable.append({
                            "first": first["company"], "second": second["company"],
                            "reason": incompatibility["reason"],
                            "detail": f"{first['company']}: {format_amount(first)}; "
                                      f"{second['company']}: {format_amount(second)}"})

    return {
        "fiscal_year": str(year), "metric": metric, "metric_label": label,
        "comparable": len(groups) == 1 and len(found) >= 2,
        "entries": views, "excluded": excluded, "relations": relations,
        "not_comparable": not_comparable, "incompatibility": incompatibility,
        "note": PER_SHARE_NOTE if metric == "eps" else None,
    }


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_comparison(companies, fiscal_years, metrics=None, *, extractor=None, catalog=None,
                     records=None, indexed_names=None, resolve_names=True, top_k=3):
    """
    Compare `metrics` for `companies` over `fiscal_years`.

    extractor : callable(company, fiscal_year, metric_keys=..., top_k=...) -> {metric: record}
                (default: Agent 1's extract_metrics)
    records   : optional pre-fetched {(company, fiscal_year): {metric: record}} so a
                caller that already ran Agent 1 (the Research Agent) is not charged twice
    indexed_names : optional {(company, fiscal_year): name stored in the index}, used
                with `records` so the company-mismatch guard compares against the
                stored name (for example "Amazon.com Inc") rather than the typed one
    catalog   : optional index catalog; loaded from the existing collection if absent
    """
    companies = list(dict.fromkeys(c.strip() for c in (companies or []) if isinstance(c, str) and c.strip()))
    years = sorted(dict.fromkeys(str(y).strip() for y in (fiscal_years or [])))
    result = {"status": "success", "companies": companies, "fiscal_years": years, "metrics": [],
              "rows": [], "comparisons": [], "year_over_year": [], "sources": [],
              "limitations": [], "errors": [], "notes": []}

    if not companies or not years:
        result["status"] = "invalid_input"
        result["errors"].append({"stage": "input", "company": None, "fiscal_year": None,
                                 "error_type": "invalid_input",
                                 "message": "at least one company and one fiscal year are required"})
        return result

    metrics = list(dict.fromkeys(metrics or DEFAULT_METRICS))
    for metric in [m for m in metrics if m not in SUPPORTED_METRICS]:
        result["limitations"].append({"type": "unsupported_metric", "company": None, "fiscal_year": None,
                                      "metric": metric, "message": f"'{metric}' is not a metric Agent 1 extracts"})
    metrics = [m for m in metrics if m in SUPPORTED_METRICS]
    result["metrics"] = metrics

    records = records or {}
    indexed_names = indexed_names or {}
    needs_extraction = any((c, y) not in records for c in companies for y in years)
    if needs_extraction and extractor is None:
        import extraction_agent
        extractor = extraction_agent.extract_metrics
    if needs_extraction and resolve_names and catalog is None:
        try:
            catalog = load_index_catalog()
        except (Exception, SystemExit) as problem:
            catalog = None
            result["limitations"].append({
                "type": "index_catalog_unavailable", "company": None, "fiscal_year": None, "metric": None,
                "message": f"index company names could not be read ({type(problem).__name__}); "
                           f"names are used exactly as typed"})

    views = {}
    raw = {}
    failures = 0
    for company in companies:
        for year in years:
            stored, record_set = indexed_names.get((company, year), company), records.get((company, year))
            if record_set is None:
                if catalog is not None:
                    stored, reason = pick_stored_name(company, year, catalog)
                    if stored is None:
                        message = ("no indexed document for this company" if reason == "company_not_indexed"
                                   else f"no indexed document for FY{year}")
                        result["limitations"].append({"type": reason, "company": company, "fiscal_year": year,
                                                      "metric": None, "message": f"{company}: {message}"})
                        for metric in metrics:
                            views[(company, year, metric)] = make_view(company, year, metric, None,
                                                                       missing_reason=message)
                        continue
                try:
                    record_set = extractor(stored, year, metric_keys=metrics, top_k=top_k)
                except (Exception, SystemExit) as problem:
                    failures += 1
                    result["errors"].append({
                        "stage": "agent_1", "company": company, "fiscal_year": year,
                        "error_type": "system_exit" if isinstance(problem, SystemExit) else "exception",
                        "message": f"Agent 1 failed for {company} FY{year}: {type(problem).__name__}: {problem}"})
                    for metric in metrics:
                        views[(company, year, metric)] = make_view(company, year, metric, None,
                                                                   missing_reason="Agent 1 failed for this company and year")
                    continue
            raw[(company, year)] = record_set
            for metric in metrics:
                record = record_set.get(metric) if isinstance(record_set, dict) else None
                views[(company, year, metric)] = make_view(company, year, metric, record, indexed_name=stored)

    for (company, year, metric), view in views.items():
        if view["status"] != "FOUND":
            result["limitations"].append({"type": "missing_data", "company": company, "fiscal_year": year,
                                          "metric": metric, "message": f"{company} FY{year} {view['metric_label']}: "
                                                                       f"NOT_FOUND ({view['reason']})"})

    for year in years:
        for company in companies:
            result["rows"].append({"company": company, "fiscal_year": year,
                                   "metrics": {m: views[(company, year, m)] for m in metrics}})
        for metric in metrics:
            comparison = compare_metric_across_companies(
                year, metric, [views[(c, year, metric)] for c in companies])
            result["comparisons"].append(comparison)
            if comparison["incompatibility"]:
                result["limitations"].append({
                    "type": "incompatible_values", "company": None, "fiscal_year": year, "metric": metric,
                    "message": f"FY{year} {comparison['metric_label']}: {comparison['incompatibility']['reason']}"})

    if len(years) > 1:
        for company in companies:
            for metric in metrics:
                for previous, current in zip(years, years[1:]):
                    prev_raw = (raw.get((company, previous)) or {}).get(metric)
                    curr_raw = (raw.get((company, current)) or {}).get(metric)
                    change = analysis_agent.compare_metric(
                        metric,
                        normalized_record_copy(prev_raw) if prev_raw else None,
                        normalized_record_copy(curr_raw) if curr_raw else None)
                    change.update({"company": company, "previous_year": previous, "current_year": current,
                                   "previous_source": _source_of(views[(company, previous, metric)]),
                                   "current_source": _source_of(views[(company, current, metric)])})
                    result["year_over_year"].append(change)

    seen = {}
    for view in views.values():
        if view["status"] == "FOUND":
            key = (view["source_file"], view["page_number"])
            seen.setdefault(key, {"company": view["company"], "fiscal_year": view["fiscal_year"],
                                  "source_file": view["source_file"], "page_number": view["page_number"],
                                  "metrics": []})["metrics"].append(view["metric"])
            for part in view["derived_from"] or []:
                pkey = (part.get("source_file"), part.get("page_number"))
                seen.setdefault(pkey, {"company": view["company"], "fiscal_year": view["fiscal_year"],
                                       "source_file": pkey[0], "page_number": pkey[1],
                                       "metrics": []})["metrics"].append(part.get("metric"))
    result["sources"] = list(seen.values())

    if len(companies) > 1:
        result["notes"].append(FISCAL_YEAR_NOTE)
    if failures == len(companies) * len(years):
        result["status"] = "error"
    elif result["errors"]:
        result["status"] = "partial"
    return result


def _source_of(view):
    return {"source_file": view["source_file"], "page_number": view["page_number"]}


# ---------------------------------------------------------------------------
# Readable output
# ---------------------------------------------------------------------------

def format_comparison_table(result):
    lines = ["=" * 78, "COMPARISON", f"Companies: {', '.join(result['companies'])}   "
             f"Fiscal years: {', '.join('FY' + y for y in result['fiscal_years'])}", "=" * 78]
    for year in result["fiscal_years"]:
        lines += ["", f"FY{year}"]
        header = ["Company"] + [METRIC_LABELS.get(m, m) for m in result["metrics"]]
        rows = [header]
        for row in (r for r in result["rows"] if r["fiscal_year"] == year):
            cells = [row["company"]]
            for metric in result["metrics"]:
                view = row["metrics"][metric]
                text = format_amount(view) if view["status"] == "FOUND" else "NOT_FOUND"
                if view["status"] == "FOUND" and view["basis"] == "DERIVED":
                    text += " (derived)"
                cells.append(text)
            rows.append(cells)
        widths = [max(len(r[i]) for r in rows) for i in range(len(header))]
        for r in rows:
            lines.append("  " + " | ".join(c.ljust(widths[i]) for i, c in enumerate(r)))

    for comparison in result["comparisons"]:
        if comparison["relations"] or comparison["incompatibility"]:
            lines += ["", f"{comparison['metric_label']} FY{comparison['fiscal_year']}"]
            lines += [f"  - {r['statement']}" for r in comparison["relations"]]
            if comparison["incompatibility"]:
                lines.append(f"  - NOT COMPARED ACROSS GROUPS: {comparison['incompatibility']['reason']}")
                lines += [f"      {k}: {', '.join(v)}" for k, v in comparison["incompatibility"]["groups"].items()]
            if comparison["note"]:
                lines.append(f"  note: {comparison['note']}")

    if result["year_over_year"]:
        lines += ["", "YEAR OVER YEAR (calculated by Agent 2)"]
        for change in result["year_over_year"]:
            if change["status"] == "calculated":
                moved = (f"{change['margin_change_points']:+} percentage points" if change["is_margin"]
                         else f"{change['percentage_change']:+}%")
                lines.append(f"  {change['company']} {change['metric']} FY{change['previous_year']} -> "
                             f"FY{change['current_year']}: {moved} ({change['direction']})")
            else:
                lines.append(f"  {change['company']} {change['metric']}: not calculated ({change['reason']})")

    for title, items in (("ERRORS", result["errors"]), ("LIMITATIONS", result["limitations"])):
        if items:
            lines += ["", title]
            lines += [f"  - {i['message']}" for i in items]
    lines += ["", "SOURCES"] + [f"  - {s['company']} FY{s['fiscal_year']}: {s['source_file']} p.{s['page_number']}"
                                 for s in result["sources"]]
    lines += [""] + [f"Note: {n}" for n in result["notes"]]
    return "\n".join(lines)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Comparison Agent")
    parser.add_argument("companies", nargs="*", default=["Apple", "Microsoft", "Amazon"])
    parser.add_argument("--years", nargs="+", default=["2024"])
    parser.add_argument("--metrics", nargs="+", default=DEFAULT_METRICS)
    arguments = parser.parse_args()
    outcome = build_comparison(arguments.companies, arguments.years, arguments.metrics)
    print(format_comparison_table(outcome))
    sys.exit(0 if outcome["status"] in ("success", "partial") else 1)