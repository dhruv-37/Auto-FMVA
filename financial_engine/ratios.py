"""Table 3 (ratios) — pure math, computed from Table 2 (financials). Bucket A, never Claude."""
import pandas as pd
import numpy as np
from .schema import RATIOS_COLS


def compute_ratios(fin: pd.DataFrame, market: pd.DataFrame = None) -> pd.DataFrame:
    """`fin` = financials table. `market` = optional market_data table (for Altman's market_cap term)."""
    r = pd.DataFrame({"year": fin["year"]})

    r["gross_margin"] = fin["gross_profit"] / fin["sales"]
    r["ebitda_margin"] = fin["ebitda"] / fin["sales"]
    ebit = fin["ebitda"] - fin["depreciation"]
    r["ebit_margin"] = ebit / fin["sales"]
    r["ebt_margin"] = fin["ebt"] / fin["sales"]
    r["net_profit_margin"] = fin["net_profit"] / fin["sales"]
    r["sales_expense_pct"] = fin["selling_admin_exp"] / fin["sales"]
    r["depreciation_pct_sales"] = fin["depreciation"] / fin["sales"]
    r["operating_income_pct_sales"] = ebit / fin["sales"]

    avg_assets = (fin["total_assets"] + fin["total_assets"].shift(1)) / 2
    avg_equity = ((fin["equity_share_capital"] + fin["reserves"]) +
                  (fin["equity_share_capital"] + fin["reserves"]).shift(1)) / 2
    r["roce"] = ebit / (fin["total_assets"] - fin["total_current_assets"] +
                         fin.get("cfo_working_capital_change", 0).fillna(0) * 0)  # capital employed proxy
    r["roe"] = fin["net_profit"] / avg_equity
    r["retained_earnings_pct"] = fin["retained_earnings_pct"]
    r["self_sustained_growth_rate"] = r["roe"] * fin["retained_earnings_pct"]

    r["interest_coverage_ratio"] = ebit / fin["interest"]
    r["debtors_turnover_ratio"] = fin["sales"] / fin["receivables"]
    # No dedicated "payables"/"creditors" row in the Screener export schema — use other_liabilities
    # as a payables proxy when a real payables figure isn't available.
    payables_proxy = fin["other_liabilities"] if "other_liabilities" in fin.columns else np.nan
    r["creditors_turnover_ratio"] = fin["cogs"] / payables_proxy
    r["inventory_turnover_ratio"] = fin["cogs"] / fin["inventory"]
    r["fixed_assets_turnover_ratio"] = fin["sales"] / fin["fixed_assets_net_block"]
    r["capital_turnover_ratio"] = fin["sales"] / fin["total_assets"]

    r["debtor_days"] = 365 / r["debtors_turnover_ratio"]
    r["payable_days"] = 365 / r["creditors_turnover_ratio"]
    r["inventory_days"] = 365 / r["inventory_turnover_ratio"]
    r["cash_conversion_cycle_days"] = r["debtor_days"] + r["inventory_days"] - r["payable_days"]

    r["cfo_to_sales"] = fin["cash_from_operating_activities"] / fin["sales"]
    r["cfo_to_total_assets"] = fin["cash_from_operating_activities"] / fin["total_assets"]
    r["cfo_to_total_debt"] = fin["cash_from_operating_activities"] / fin["borrowings"]

    # Invested Capital / ROIC / Reinvestment / Intrinsic Growth
    nwc = (fin["receivables"] + fin["inventory"]) - fin.get("other_liabilities", 0)
    r["net_working_capital"] = nwc
    r["invested_capital"] = nwc + (fin["total_assets"] - fin["total_current_assets"])
    r["roic"] = ebit * (1 - fin["effective_tax_rate"].fillna(0.25)) / r["invested_capital"]
    r["net_capex"] = fin["fixed_assets_net_block"].diff() + fin["depreciation"]
    r["change_in_working_capital"] = nwc.diff()
    r["reinvestment"] = r["net_capex"] + r["change_in_working_capital"]
    ebit_1_tax = ebit * (1 - fin["effective_tax_rate"].fillna(0.25))
    r["reinvestment_rate"] = r["reinvestment"] / ebit_1_tax
    r["intrinsic_growth_rate"] = r["reinvestment_rate"] * r["roic"]

    # DuPont (reuses margin/turnover already computed)
    r["net_profit_margin_dupont"] = r["net_profit_margin"]
    r["asset_turnover_dupont"] = fin["sales"] / avg_assets
    r["equity_multiplier_dupont"] = avg_assets / avg_equity

    # Altman Z (needs market_cap from market_data table — pass it in)
    if market is not None:
        mkt = market.set_index("year")["market_cap"].reindex(fin["year"]).values
    else:
        mkt = np.nan
    long_term_liab = fin["borrowings"]
    r["altman_wc_to_ta"] = (fin["total_current_assets"] - fin.get("other_liabilities", 0)) / fin["total_assets"]
    r["altman_re_to_ta"] = (fin["net_profit"] * fin["retained_earnings_pct"]).cumsum() / fin["total_assets"]
    r["altman_ebit_to_ta"] = ebit / fin["total_assets"]
    r["altman_mcap_to_ltl"] = mkt / long_term_liab
    r["altman_sales_to_ta"] = fin["sales"] / fin["total_assets"]
    r["altman_z_score"] = (1.2 * r["altman_wc_to_ta"] + 1.4 * r["altman_re_to_ta"] +
                            3.3 * r["altman_ebit_to_ta"] + 0.6 * r["altman_mcap_to_ltl"] +
                            1.0 * r["altman_sales_to_ta"])
    r["altman_zone"] = pd.cut(r["altman_z_score"], bins=[-np.inf, 1.8, 3.0, np.inf],
                               labels=["Distressed", "Grey Zone", "Safe"])

    for col in RATIOS_COLS:
        if col not in r.columns:
            r[col] = pd.NA
    return r[RATIOS_COLS]