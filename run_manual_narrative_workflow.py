"""
Manual (no-API-key) narrative workflow for a company already processed by main.py
(Bucket A / Step 1+2). Wraps the build_web_prompt() / paste_narrative() functions
that already exist in financial_engine/narrative.py -- no changes needed there.

Usage:
    1. python run_manual_narrative_workflow.py prompt COMPANY MODULE
       -> writes data/processed/COMPANY/narrative_<module>_PROMPT.txt
       -> open that file, select-all, paste into a Claude web chat, send it.

    2. Copy Claude's reply (the JSON block Claude answers with) into
       data/processed/COMPANY/narrative_<module>_REPLY.txt (just paste the raw
       reply -- extra chatter around the JSON is fine, paste_narrative() extracts
       the {...} block for you).

    3. python run_manual_narrative_workflow.py save COMPANY MODULE
       -> validates the reply (schema + numeric-fidelity checks) and, if it
          passes, writes data/processed/COMPANY/narrative_<module>.json --
          the file Step 5 (report assembly) reads.
"""
import json
import sys
from pathlib import Path

import pandas as pd

from financial_engine.narrative import build_web_prompt, paste_narrative, NarrativeError, NARRATIVE_SCHEMAS

COMPANY = "RELIANCE"


def load_results(company: str) -> dict:
    """Assembles the same `results` dict Step 3 would build from Bucket A's CSVs.

    NOTE: this repo's main.py (Step 1+2) currently only produces financials,
    market_data, ratios and the monte_carlo tables for RELIANCE -- it never calls
    financial_engine/dcf.py or financial_engine/wacc.py (those need a peer-company
    input file and DCF assumptions that aren't checked into the repo), so dcf.csv
    and peer_comps.csv don't exist yet. This loader includes them automatically if
    you add them later; for now it proceeds with what's actually on disk.
    """
    d = Path("data/processed") / company
    results = {}
    file_map = {
        "financials": d / "financials.csv",
        "ratios": d / "ratios.csv",
        "market_data": d / "market_data.csv",
        "monte_carlo_var_summary": d / "monte_carlo_var_summary.csv",
        "dcf": d / "dcf.csv",                 # not yet produced -- see note above
        "peer_comps": d / "peer_comps.csv",   # not yet produced -- see note above
    }
    missing = []
    for key, path in file_map.items():
        if path.exists():
            results[key] = json.loads(pd.read_csv(path).to_json(orient="records"))
        else:
            missing.append(path.name)
    if missing:
        print(f"[warning] not found on disk, skipping: {missing}")
    return results


FEW_SHOT_TEMPLATE = {
    "company_profile": {
        "title": "TCS -- Company Profile",
        "body": (
            "Tata Consultancy Services (TCS) is India's largest IT services exporter, "
            "generating the bulk of its revenue from long-term outsourcing contracts "
            "with global enterprise clients. Over FY21-FY24 the company grew sales at a "
            "low-double-digit CAGR while holding EBITDA margins in the high-20s, "
            "reflecting a stable, cash-generative services model rather than a capital-"
            "intensive one. Balance sheet leverage is minimal and the company remains "
            "net cash."
        ),
        "recent_updates": [
            "Won a multi-year cloud transformation deal with a major European bank",
            "Announced a buyback alongside the Q4 results",
        ],
        "data_quality_flags": [],
    },
}


def cmd_prompt(company: str, module: str):
    results = load_results(company)
    few_shot = FEW_SHOT_TEMPLATE.get(module, {"note": "fill in a real few-shot example for this module"})
    prompt_text = build_web_prompt(module, company, results, few_shot)
    out = Path("data/processed") / company / f"narrative_{module}_PROMPT.txt"
    out.write_text(prompt_text)
    print(f"Wrote {out} ({len(prompt_text)} chars). Paste its full contents into a Claude web chat.")
    return results


def cmd_save(company: str, module: str):
    results = load_results(company)
    reply_path = Path("data/processed") / company / f"narrative_{module}_REPLY.txt"
    if not reply_path.exists():
        print(f"Paste Claude's reply into {reply_path} first, then re-run this command.")
        sys.exit(1)
    raw_text = reply_path.read_text()
    out_path = Path("data/processed") / company / f"narrative_{module}.json"
    try:
        parsed = paste_narrative(module, results, raw_text, out_path=str(out_path))
    except NarrativeError as e:
        print(f"[REJECTED] {e}")
        sys.exit(1)
    print(f"[OK] validated + saved to {out_path}")
    print(json.dumps(parsed, indent=2))


if __name__ == "__main__":
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
    action, company = sys.argv[1], sys.argv[2]
    module = sys.argv[3] if len(sys.argv) > 3 else "company_profile"
    if module not in NARRATIVE_SCHEMAS:
        print(f"Unknown module '{module}'. Choose from {list(NARRATIVE_SCHEMAS)}")
        sys.exit(1)
    if action == "prompt":
        cmd_prompt(company, module)
    elif action == "save":
        cmd_save(company, module)
    else:
        print(__doc__)