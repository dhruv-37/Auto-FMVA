"""Orchestrator — builds all 9 tables for one company and saves them to data/processed/<company>/."""
from pathlib import Path
from financial_engine.ingest import build_company_dataset
from financial_engine.ratios import compute_ratios
from financial_engine.wacc import build_peer_comps, compute_wacc, compute_beta_drifting
from financial_engine.monte_carlo import run_monte_carlo_var

COMPANY = "RELIANCE"
SCREENER_XLSX = "data/raw/RELIANCE.xlsx"
TICKER = "RELIANCE.NS"

if __name__ == "__main__":
    out_dir = Path("data/processed") / COMPANY

    # Table 2 + Table 5
    financials, price_history = build_company_dataset(COMPANY, SCREENER_XLSX, TICKER)

    # Table 3 (pass market_data if you have it for Altman's market_cap term)
    ratios_df = compute_ratios(financials)
    ratios_df.to_csv(out_dir / "ratios.csv", index=False)

    # Table 5 -> Beta Drifting
    beta = compute_beta_drifting(price_history)
    print("Adjusted beta:", beta["adjusted_beta"])

    # Table 9
    trials, mc_summary = run_monte_carlo_var(price_history)
    trials.to_csv(out_dir / "monte_carlo_trials.csv", index=False)
    mc_summary["var_table"].to_csv(out_dir / "monte_carlo_var_summary.csv", index=False)

    print(f"Done — all tables written to {out_dir}/")
