"""Step 6 — orchestration (full pipeline):

    1. ingest(company_ticker)              -> raw financials + price history
    2. normalize(raw)                      -> standard schema
    3. engine.compute_all(standard_schema) -> results JSON      (Bucket A, no LLM)
    4. llm.generate_narratives(results)   -> narrative JSON    (Bucket B)
    5. render_excel(results, narrative)    -> .xlsx
    6. qa_check(results, narrative)        -> flags for human review before send
    7. human reviews the QA sheet before the workbook goes to a client

Bucket A (steps 1-3) never touches a model provider. Step 4 is the only place an LLM enters,
through the JSON-in/JSON-out contract in financial_engine/narrative.py and the free/local
provider adapter in financial_engine/llm_narrative.py.
"""
import argparse
import json
from pathlib import Path

import pandas as pd

from financial_engine.ingest import build_company_dataset, load_screener_export
from financial_engine.market_data import build_company_profile, build_market_data, fill_52w_high_low
from financial_engine.ratios import compute_ratios
from financial_engine.statements import compute_common_size, compute_comparative
from financial_engine.forecasting import run_regression_forecast
from financial_engine.wacc import build_peer_comps, compute_comps, compute_wacc, compute_beta_drifting
from financial_engine.dcf import run_dcf, dcf_sensitivity, build_valuation_summary
from financial_engine.monte_carlo import run_monte_carlo_var
from financial_engine.narrative import NarrativeError, NARRATIVE_SCHEMAS
from financial_engine.llm_narrative import generate_narrative_free, PROVIDER_PRESETS
from financial_engine.qa import qa_check, print_qa_report
from financial_engine.report import render_excel

DEFAULT_ASSUMPTIONS = dict(
    tax_rate=0.25,
    revenue_growth_fwd=[0.09, 0.085, 0.08, 0.075, 0.07],   # 5Y fade
    investment_rate_fwd=[0.35, 0.35, 0.33, 0.30, 0.28],
    terminal_growth=0.05,
    target_debt_equity=0.30,
    risk_free_rate=0.069,     # ~10Y India G-Sec
    equity_risk_premium=0.06,
    pre_tax_cost_of_debt=0.075,
)

FEW_SHOT = {
    "company_profile": {
        "title": "TCS -- Company Profile",
        "body": ("Tata Consultancy Services (TCS) is India's largest IT services exporter, "
                 "generating the bulk of its revenue from long-term outsourcing contracts "
                 "with global enterprise clients. Over FY21-FY24 the company grew sales at a "
                 "low-double-digit CAGR while holding EBITDA margins in the high-20s. Balance "
                 "sheet leverage is minimal and the company remains net cash."),
        "recent_updates": ["Won a multi-year cloud transformation deal with a major European bank",
                            "Announced a buyback alongside the Q4 results"],
        "data_quality_flags": [],
    },
    "interpretive_commentary": {
        "title": "TCS -- Interpretive Commentary",
        "body": ("ROIC has stayed comfortably above WACC for all five years shown, consistent "
                 "with a capital-light, high-margin services model rather than balance-sheet "
                 "leverage driving returns. The Altman Z-score has stayed in the Safe zone "
                 "throughout, reflecting minimal leverage and high asset turnover."),
        "data_quality_flags": [],
    },
    "anomaly_flags": {"flags": [], "data_quality_flags": []},
    "football_field": {
        "title": "TCS -- Valuation Football Field",
        "body": ("Comps and DCF ranges overlap around the current trading price, with the "
                 "DCF Base case sitting close to the 52-week midpoint -- the market is pricing "
                 "in roughly the base-case growth/margin trajectory, not the bull case."),
        "data_quality_flags": [],
    },
}


def step_1_2_ingest_normalize(company: str, screener_xlsx: str, ticker: str, peers_csv: str | None):
    from financial_engine.ingest import normalize_financials
    cached_prices = Path("data/processed") / company / "price_history.csv"
    if cached_prices.exists():
        # yfinance needs live network access; reuse the already-fetched price history
        # (e.g. from a prior main.py run) instead of re-hitting the API every time.
        sheets = load_screener_export(screener_xlsx)
        financials = normalize_financials(sheets)
        price_history = pd.read_csv(cached_prices, parse_dates=["date"])
    else:
        financials, price_history = build_company_dataset(company, screener_xlsx, ticker)
    sheets = load_screener_export(screener_xlsx)
    profile = build_company_profile(sheets, ticker_nse=ticker)
    market = build_market_data(sheets, financials)
    market = fill_52w_high_low(market, price_history)
    peer_comps = build_peer_comps(pd.read_csv(peers_csv).to_dict("records")) if peers_csv else None
    return dict(profile=profile, financials=financials, price_history=price_history,
                market=market, peer_comps=peer_comps)


def step_3_compute_all(data: dict, assumptions: dict) -> dict:
    """Bucket A only -- no Claude anywhere in this function."""
    fin, market, price_history, peer_comps = data["financials"], data["market"], data["price_history"], data["peer_comps"]

    ratios_df = compute_ratios(fin, market=market)
    common_size_df = compute_common_size(fin)
    comparative_df = compute_comparative(fin)
    forecast_df, forecast_summary = run_regression_forecast(fin, forecast_years=[2027, 2028, 2029])
    beta = compute_beta_drifting(price_history)
    trials_df, mc_summary = run_monte_carlo_var(price_history)

    dcf_summary = dcf_df = comps_df = wacc_out = None
    if peer_comps is not None and not peer_comps.empty:
        last = fin.sort_values("year").iloc[-1]
        wacc_out = compute_wacc(
            peer_comps, target_debt_equity=assumptions["target_debt_equity"],
            tax_rate=assumptions["tax_rate"], risk_free_rate=assumptions["risk_free_rate"],
            equity_risk_premium=assumptions["equity_risk_premium"],
            pre_tax_cost_of_debt=assumptions["pre_tax_cost_of_debt"],
            current_debt=float(last["borrowings"]), current_equity=float(last["equity_share_capital"] + last["reserves"]),
        )
        last_ebit = float(last["ebitda"] - last["depreciation"])
        dcf_df, dcf_summary = run_dcf(
            last_actual_ebit=last_ebit, tax_rate=assumptions["tax_rate"],
            revenue_growth_fwd=assumptions["revenue_growth_fwd"],
            investment_rate_fwd=assumptions["investment_rate_fwd"],
            forecast_years=[int(last["year"]) + i for i in range(1, 6)],
            wacc=wacc_out["wacc"], terminal_growth=assumptions["terminal_growth"],
            cash=float(last["cash_bank"]), debt=float(last["borrowings"]),
            # financials.csv (cash/debt/ebit) are in Rs crore; shares_outstanding is a raw
            # share count. Convert shares to crore-of-shares so equity_value_per_share comes
            # out in Rs/share, not a ~1e-4 unit-mismatch artifact.
            shares_outstanding=float(last["shares_outstanding"]) / 1e7, base_year=int(last["year"]),
        )
        comps_df = compute_comps(
            peer_comps, target_revenue=float(last["sales"]), target_ebitda=float(last["ebitda"]),
            target_eps=float(last["eps"]), target_net_debt=float(last["borrowings"] - last["cash_bank"]),
            target_shares_outstanding=float(last["shares_outstanding"]) / 1e7,  # crore, to match EV units
        )

    results = dict(
        profile=data["profile"],
        financials=json.loads(fin.to_json(orient="records")),
        ratios=json.loads(ratios_df.to_json(orient="records")),
        market_data=json.loads(market.to_json(orient="records")),
        common_size=json.loads(common_size_df.to_json(orient="records")),
        comparative=json.loads(comparative_df.to_json(orient="records")),
        forecast=dict(summary=forecast_summary, table=json.loads(forecast_df.to_json(orient="records"))),
        beta_drifting=beta,
        monte_carlo_var_summary=json.loads(mc_summary["var_table"].to_json(orient="records")),
        peer_comps=json.loads(peer_comps.to_json(orient="records")) if peer_comps is not None else None,
        wacc=wacc_out,
        dcf=json.loads(dcf_df.to_json(orient="records")) if dcf_df is not None else None,
        dcf_summary=dcf_summary,
        comps_valuation=json.loads(comps_df.to_json(orient="records")) if comps_df is not None else None,
    )
    tables = {
        "financials": fin, "ratios": ratios_df, "market_data": market,
        "common_size": common_size_df, "comparative": comparative_df,
        "monte_carlo_var_summary": mc_summary["var_table"],
    }
    if peer_comps is not None:
        tables["peer_comps"] = peer_comps
    if dcf_df is not None:
        tables["dcf"] = dcf_df
    if comps_df is not None:
        tables["comps_valuation"] = comps_df
    return results, tables


def step_4_narratives(company: str, results: dict, out_dir: Path, provider: str = "groq",
                       model: str | None = None, api_key: str | None = None) -> dict:
    """Generate all narrative modules through the automated provider path."""
    narratives = {}
    for module in NARRATIVE_SCHEMAS:
        few_shot = FEW_SHOT.get(module, {})
        try:
            parsed = generate_narrative_free(module, company, results, few_shot,
                                              provider=provider, model=model, api_key=api_key)
            narratives[module] = parsed
            (out_dir / f"narrative_{module}.json").write_text(json.dumps(parsed, indent=2))
            print(f"[OK] {module}: generated via {provider}")
        except NarrativeError as e:
            print(f"[REJECTED] {module} via {provider}: {e}")
    return narratives


def run(company: str, screener_xlsx: str, ticker: str, peers_csv: str | None,
        provider: str = "groq", model: str | None = None, api_key: str | None = None,
        assumptions: dict | None = None) -> None:
    assumptions = {**DEFAULT_ASSUMPTIONS, **(assumptions or {})}
    out_dir = Path("data/processed") / company
    out_dir.mkdir(parents=True, exist_ok=True)

    print("Step 1-2: ingest + normalize ...")
    data = step_1_2_ingest_normalize(company, screener_xlsx, ticker, peers_csv)

    print("Step 3: engine.compute_all (Bucket A, no Claude) ...")
    results, tables = step_3_compute_all(data, assumptions)
    (out_dir / "results.json").write_text(json.dumps(results, indent=2, default=str))

    print(f"Step 4: generate narratives (Bucket B, provider={provider}) ...")
    narratives = step_4_narratives(company, results, out_dir, provider=provider, model=model, api_key=api_key)

    print("Step 6: render_excel ...")
    print("Step 7: qa_check ...")
    qa_result = qa_check(results, narratives)
    print_qa_report(qa_result)

    xlsx_path = render_excel(str(out_dir / f"{company}_report.xlsx"), tables, narratives, qa_result)
    print(f"Workbook written: {xlsx_path}")
    print("Step 7 (human): open the QA Review sheet before this goes anywhere near a client.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--company", default="RELIANCE")
    p.add_argument("--screener-xlsx", default="data/raw/RELIANCE.xlsx")
    p.add_argument("--ticker", default="RELIANCE.NS")
    p.add_argument("--peers-csv", default="data/raw/RELIANCE_peers.csv")
    p.add_argument("--provider", choices=[*PROVIDER_PRESETS], default="groq",
                    help="OpenAI-compatible free/local provider for Step 4 (Groq, Gemini, OpenRouter, or Ollama).")
    p.add_argument("--model", default=None, help="override the provider's default model")
    p.add_argument("--api-key", default=None,
                    help="API key for --provider groq/gemini/openrouter. Overrides the "
                         "{PROVIDER}_API_KEY env var if both are set. Not needed for ollama.")
    args = p.parse_args()
    run(args.company, args.screener_xlsx, args.ticker, args.peers_csv, args.provider, args.model, args.api_key)
