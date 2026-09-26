"""Orchestrator — builds all 9 tables for one company and saves them to data/processed/<company>/."""
from pathlib import Path
from financial_engine.ingest import build_company_dataset, load_screener_export
from financial_engine.ratios import compute_ratios
from financial_engine.market_data import build_company_profile, build_market_data, fill_52w_high_low
from financial_engine.wacc import build_peer_comps, compute_wacc, compute_beta_drifting
from financial_engine.monte_carlo import run_monte_carlo_var

COMPANY = "RELIANCE"
SCREENER_XLSX = "data/raw/RELIANCE.xlsx"
TICKER = "RELIANCE.NS"

if __name__ == "__main__":
    out_dir = Path("data/processed") / COMPANY

    # Table 2 + Table 5
    financials, price_history = build_company_dataset(COMPANY, SCREENER_XLSX, TICKER)

    # Table 1 + Table 4
    sheets = load_screener_export(SCREENER_XLSX)
    profile = build_company_profile(sheets, ticker_nse=TICKER, ticker_bse="500325")
    market = build_market_data(sheets, financials)
    market = fill_52w_high_low(market, price_history)
    market.to_csv(out_dir / "market_data.csv", index=False)

    # Table 3 (now with market_data for Altman Z)
    ratios_df = compute_ratios(financials, market=market)
    ratios_df.to_csv(out_dir / "ratios.csv", index=False)

    # Table 5 -> Beta Drifting
    beta = compute_beta_drifting(price_history)
    print("Adjusted beta:", beta["adjusted_beta"])

    # Table 9
    trials, mc_summary = run_monte_carlo_var(price_history)
    trials.to_csv(out_dir / "monte_carlo_trials.csv", index=False)
    mc_summary["var_table"].to_csv(out_dir / "monte_carlo_var_summary.csv", index=False)

    print(f"Done — all tables written to {out_dir}/")
