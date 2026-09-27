"""Step 7 — QA gate. Every Claude-generated sentence and every auto-computed sensitivity
table is a draft until a human sanity-checks it (per the reference doc's "Honest scope" /
Step 7 note). This module never blocks silently and never auto-fixes anything — it only
collects flags into one place so a human reviewer has a single checklist before publish.
"""
from __future__ import annotations
from typing import Any

from .narrative import flag_engine_anomalies

# Sanity bounds for a handful of headline outputs. Not exhaustive — extend as new
# failure modes turn up (this is exactly the kind of judgment call Step 7 exists for).
_RATIO_BOUNDS = {
    "gross_margin": (-1.0, 1.0),
    "ebitda_margin": (-1.0, 1.0),
    "ebit_margin": (-1.0, 1.0),
    "net_profit_margin": (-1.0, 1.0),
    "roe": (-3.0, 3.0),
    "roic": (-3.0, 3.0),
    "altman_z_score": (-50.0, 50.0),
}


def _check_ratio_bounds(ratios_records: list[dict]) -> list[dict]:
    flags = []
    for row in ratios_records:
        year = row.get("year")
        for col, (lo, hi) in _RATIO_BOUNDS.items():
            val = row.get(col)
            if isinstance(val, (int, float)) and (val < lo or val > hi):
                flags.append({
                    "table": "ratios", "column": col, "year": year, "value": val,
                    "issue": f"{col}={val} falls outside sanity bounds [{lo}, {hi}]",
                    "severity": "high",
                })
    return flags


def _check_dcf_sanity(dcf_summary: dict | None) -> list[dict]:
    flags = []
    if not dcf_summary:
        return flags
    wacc = dcf_summary.get("wacc")
    g = dcf_summary.get("terminal_growth")
    if wacc is not None and g is not None and wacc <= g:
        flags.append({
            "table": "dcf", "column": "wacc/terminal_growth", "year": None,
            "value": f"wacc={wacc}, terminal_growth={g}",
            "issue": "WACC <= terminal growth rate — Gordon growth terminal value is undefined/negative",
            "severity": "high",
        })
    eps = dcf_summary.get("equity_value_per_share")
    if isinstance(eps, (int, float)):
        if eps < 0:
            flags.append({
                "table": "dcf", "column": "equity_value_per_share", "year": None, "value": eps,
                "issue": "DCF implies negative equity value per share",
                "severity": "high",
            })
        elif eps < 1:
            flags.append({
                "table": "dcf", "column": "equity_value_per_share", "year": None, "value": eps,
                "issue": f"equity_value_per_share={eps:.6g} is implausibly small for a listed "
                         "large-cap — likely a units mismatch (e.g. equity value in crore vs. "
                         "shares_outstanding as a raw count) rather than a genuine valuation",
                "severity": "high",
            })
    return flags


def qa_check(results: dict, narratives: dict[str, dict]) -> dict:
    """Runs before anything is rendered/sent (Step 7). `results` = the same Bucket-A
    payload fed to narrative.generate_narrative(); `narratives` = {module_name: parsed
    narrative dict} for every module that was run.

    Returns {"ok": bool, "flags": [...]} — a single deduplicated review checklist. "ok"
    is only a convenience signal (True iff no "high" severity flags); it is NOT a
    green light to auto-publish. A human should read every flag regardless, per the
    doc's Step 7 rule that nothing skips human sanity-check before it reaches a client.
    """
    flags: list[dict] = []

    ratios_records = results.get("ratios") or []
    flags.extend(flag_engine_anomalies(ratios_records))     # doc's own -100%-margin-style sweep
    flags.extend(_check_ratio_bounds(ratios_records))
    flags.extend(_check_dcf_sanity(results.get("dcf_summary")))

    for module, parsed in (narratives or {}).items():
        for msg in parsed.get("data_quality_flags", []) or []:
            flags.append({"table": f"narrative:{module}", "column": None, "year": None,
                           "value": None, "issue": msg, "severity": "medium"})
        for f in parsed.get("flags", []) or []:   # anomaly_flags module's own structured flags
            f = dict(f)
            f.setdefault("severity", "medium")
            f["table"] = f"narrative:{module}:" + str(f.get("table", ""))
            flags.append(f)

    # de-dupe identical (table, column, year, issue) tuples across the different sources above
    seen, deduped = set(), []
    for f in flags:
        key = (f.get("table"), f.get("column"), f.get("year"), f.get("issue"))
        if key not in seen:
            seen.add(key)
            deduped.append(f)

    ok = not any(f.get("severity") == "high" for f in deduped)
    return {"ok": ok, "flags": deduped}


def print_qa_report(qa_result: dict) -> None:
    flags = qa_result["flags"]
    if not flags:
        print("QA: no flags raised. Still — human review required before publish.")
        return
    print(f"QA: {len(flags)} flag(s) raised (ok={qa_result['ok']}):")
    for f in flags:
        loc = f"{f.get('table')}" + (f".{f['column']}" if f.get("column") else "") + \
              (f" ({f['year']})" if f.get("year") else "")
        print(f"  [{f.get('severity', '?').upper()}] {loc}: {f.get('issue')}")
