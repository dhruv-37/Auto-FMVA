"""Column contracts for all 9 tables. Every other module imports from here."""

FINANCIALS_COLS = [
    "year", "sales", "sales_growth", "cogs", "change_in_inventory",
    "power_fuel_exp", "other_mfr_exp", "employee_cost", "selling_admin_exp",
    "other_expenses", "other_income", "gross_profit", "gross_margin",
    "ebitda", "ebitda_margin", "interest", "interest_pct_sales",
    "depreciation", "depreciation_pct_sales", "ebt", "ebt_pct_sales",
    "tax", "effective_tax_rate", "net_profit", "net_margin",
    "shares_outstanding", "eps", "eps_growth", "dps", "dps_growth",
    "retained_earnings_pct", "dividend_payout_ratio",
    "equity_share_capital", "reserves", "borrowings", "other_liabilities",
    "total_equity_liabilities", "fixed_assets_net_block", "capital_wip",
    "investments_noncurrent", "other_noncurrent_assets",
    "total_noncurrent_assets", "receivables", "inventory", "cash_bank",
    "total_current_assets", "total_assets", "balance_check",
    "cfo_profit_from_ops", "cfo_receivables_change", "cfo_inventory_change",
    "cfo_payables_change", "cfo_working_capital_change", "cfo_direct_taxes",
    "cash_from_operating_activities", "cfi_fixed_assets_purchased",
    "cfi_fixed_assets_sold", "cfi_investments_purchased",
    "cfi_investments_sold", "cfi_interest_received", "cfi_dividends_received",
    "cfi_other", "cash_from_investing_activities", "cff_proceeds_shares",
    "cff_proceeds_borrowings", "cff_repayment_borrowings", "cff_interest_paid",
    "cff_dividends_paid", "cff_financial_liabilities",
    "cff_share_application_money", "cff_other",
    "cash_from_financing_activities", "net_cash_flow",
]

RATIOS_COLS = [
    "year", "gross_margin", "ebitda_margin", "ebit_margin", "ebt_margin",
    "net_profit_margin", "sales_expense_pct", "depreciation_pct_sales",
    "operating_income_pct_sales", "roce", "roe", "retained_earnings_pct",
    "self_sustained_growth_rate", "interest_coverage_ratio",
    "debtors_turnover_ratio", "creditors_turnover_ratio",
    "inventory_turnover_ratio", "fixed_assets_turnover_ratio",
    "capital_turnover_ratio", "debtor_days", "payable_days",
    "inventory_days", "cash_conversion_cycle_days", "cfo_to_sales",
    "cfo_to_total_assets", "cfo_to_total_debt", "roic", "invested_capital",
    "net_working_capital", "net_capex", "change_in_working_capital",
    "reinvestment", "reinvestment_rate", "intrinsic_growth_rate",
    "net_profit_margin_dupont", "asset_turnover_dupont",
    "equity_multiplier_dupont", "altman_wc_to_ta", "altman_re_to_ta",
    "altman_ebit_to_ta", "altman_mcap_to_ltl", "altman_sales_to_ta",
    "altman_z_score", "altman_zone",
]

MARKET_DATA_COLS = [
    "year", "share_price", "market_cap", "shares_outstanding",
    "cash_equivalents", "total_debt", "minority_interest",
    "enterprise_value", "pe_ratio", "ev_ebitda", "ev_sales",
    "price_to_book", "price_52w_high", "price_52w_low",
]

PRICE_HISTORY_COLS = [
    "date", "close_price", "volume", "index_close_price",
    "stock_return_pct", "index_return_pct",
]

PEER_COMPS_COLS = [
    "peer_ticker", "peer_name", "country", "total_debt", "total_equity",
    "tax_rate", "debt_to_equity", "debt_to_capital", "levered_beta",
    "unlevered_beta", "share_price", "shares_outstanding", "equity_value",
    "net_debt", "enterprise_value", "revenue", "ebitda", "net_income",
    "ev_revenue", "ev_ebitda", "pe_ratio",
]

DCF_COLS = [
    "year", "is_actual", "ebit", "tax_rate", "ebt_1_minus_tax",
    "investment_rate", "fcff", "mid_year_convention_period",
    "discounting_factor", "pv_fcff",
]

VALUATION_SUMMARY_COLS = ["method_label", "low_value", "high_value", "open_low", "open_high"]

MONTE_CARLO_COLS = ["trial_id", "simulated_return"]

REQUIRED_FINANCIALS = ["year", "sales", "ebitda", "net_profit", "total_assets", "total_equity_liabilities"]


def validate_financials(df):
    """Raise if required fields missing or null. Call right after normalize_financials()."""
    missing = [c for c in REQUIRED_FINANCIALS if c not in df.columns]
    if missing:
        raise ValueError(f"Schema violation in `financials` — missing columns: {missing}")
    nulls = df[REQUIRED_FINANCIALS].isnull().any()
    bad = nulls[nulls].index.tolist()
    if bad:
        print(f"Warning: null values in required columns: {bad}")
