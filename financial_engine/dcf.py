"""Table 7 (dcf) + Table 8 (valuation_summary). Bucket A."""
import pandas as pd
import numpy as np
from .schema import DCF_COLS, VALUATION_SUMMARY_COLS


def run_dcf(last_actual_ebit: float, tax_rate: float, revenue_growth_fwd: list[float],
            investment_rate_fwd: list[float], forecast_years: list[int], wacc: float,
            terminal_growth: float, cash: float, debt: float, shares_outstanding: float,
            base_year: int) -> tuple[pd.DataFrame, dict]:
    """forecast_years e.g. [2026..2030]; revenue_growth_fwd/investment_rate_fwd same length."""
    n = len(forecast_years)
    ebit = [last_actual_ebit]
    for g in revenue_growth_fwd:
        ebit.append(ebit[-1] * (1 + g))
    ebit = ebit[1:]  # forecast years only

    ebt_1_tax = [e * (1 - tax_rate) for e in ebit]
    fcff = [ebt_1_tax[i] * (1 - investment_rate_fwd[i]) for i in range(n)]
    mid_year = [0.5 + i for i in range(n)]
    discount_factor = [1 / (1 + wacc) ** t for t in mid_year]
    pv_fcff = [fcff[i] * discount_factor[i] for i in range(n)]

    dcf_df = pd.DataFrame({
        "year": forecast_years, "is_actual": False, "ebit": ebit, "tax_rate": tax_rate,
        "ebt_1_minus_tax": ebt_1_tax, "investment_rate": investment_rate_fwd, "fcff": fcff,
        "mid_year_convention_period": mid_year, "discounting_factor": discount_factor,
        "pv_fcff": pv_fcff,
    })[DCF_COLS]

    terminal_fcff_next = fcff[-1] * (1 + terminal_growth)
    terminal_value = terminal_fcff_next / (wacc - terminal_growth)
    pv_terminal_value = terminal_value * discount_factor[-1]

    value_of_operating_assets = sum(pv_fcff) + pv_terminal_value
    equity_value = value_of_operating_assets + cash - debt
    equity_value_per_share = equity_value / shares_outstanding

    summary = dict(
        terminal_value=terminal_value, pv_terminal_value=pv_terminal_value,
        value_of_operating_assets=value_of_operating_assets, cash=cash, debt=debt,
        equity_value=equity_value, shares_outstanding=shares_outstanding,
        equity_value_per_share=equity_value_per_share, wacc=wacc, terminal_growth=terminal_growth,
    )
    return dcf_df, summary


def dcf_sensitivity(base_fcff_next: float, cash: float, debt: float, shares_outstanding: float,
                     wacc_range: list[float], growth_range: list[float]) -> pd.DataFrame:
    """2D grid: rows=WACC, cols=terminal growth -> equity value per share."""
    grid = pd.DataFrame(index=wacc_range, columns=growth_range, dtype=float)
    for w in wacc_range:
        for g in growth_range:
            tv = base_fcff_next * (1 + g) / (w - g)
            equity_val = tv + cash - debt
            grid.loc[w, g] = equity_val / shares_outstanding
    grid.index.name, grid.columns.name = "wacc", "terminal_growth"
    return grid


def build_valuation_summary(comps_low: float, comps_high: float,
                             dcf_bear: float, dcf_base: float, dcf_bull: float,
                             price_52w_low: float, price_52w_high: float) -> pd.DataFrame:
    """Table 8 — feeds the Football Field chart directly."""
    rows = [
        ("Comps", comps_low, comps_high, comps_low, comps_high),
        ("DCF Bear", dcf_bear * 0.94, dcf_bear * 1.05, dcf_bear * 0.94, dcf_bear * 1.05),
        ("DCF Base", dcf_base * 0.98, dcf_base * 1.02, dcf_base * 0.98, dcf_base * 1.02),
        ("DCF Bull", dcf_bull * 0.98, dcf_bull * 1.02, dcf_bull * 0.98, dcf_bull * 1.02),
        ("52W H/L", price_52w_low, price_52w_high, price_52w_low, price_52w_high),
    ]
    return pd.DataFrame(rows, columns=VALUATION_SUMMARY_COLS)
