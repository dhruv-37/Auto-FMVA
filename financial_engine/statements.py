"""Table: Common Size + Table: Comparative Statements — pure math off Table 2 (financials).
Bucket A, never Claude."""
import pandas as pd
import numpy as np

# Income-statement lines -> expressed as % of sales.
PNL_COLS = [
    "sales", "cogs", "change_in_inventory", "power_fuel_exp", "other_mfr_exp",
    "employee_cost", "selling_admin_exp", "other_expenses", "other_income",
    "gross_profit", "ebitda", "depreciation", "interest", "ebt", "tax", "net_profit",
]

# Balance-sheet lines -> expressed as % of total assets.
BS_COLS = [
    "equity_share_capital", "reserves", "borrowings", "other_liabilities",
    "total_equity_liabilities", "fixed_assets_net_block", "capital_wip",
    "investments_noncurrent", "other_noncurrent_assets", "total_noncurrent_assets",
    "receivables", "inventory", "cash_bank", "total_current_assets", "total_assets",
]


def compute_common_size(fin: pd.DataFrame) -> pd.DataFrame:
    """`fin` = financials table (Table 2). Returns year + one `<col>_pct_sales` column per P&L
    line and one `<col>_pct_assets` column per balance-sheet line."""
    out = pd.DataFrame({"year": fin["year"]})
    for col in PNL_COLS:
        if col in fin.columns:
            out[f"{col}_pct_sales"] = fin[col] / fin["sales"]
    for col in BS_COLS:
        if col in fin.columns:
            out[f"{col}_pct_assets"] = fin[col] / fin["total_assets"]
    return out


def compute_comparative(fin: pd.DataFrame) -> pd.DataFrame:
    """`fin` = financials table (Table 2), sorted by year ascending. Returns year + one
    `<col>_yoy_abs` and `<col>_yoy_pct` per numeric column — YoY absolute and % change.

    Columns are coerced with pd.to_numeric rather than filtered by dtype: normalize_financials
    can hand back numeric-valued columns typed as `object` (a side effect of the pivot/merge
    chain in ingest.py), which pandas' own is_numeric_dtype would silently skip."""
    fin = fin.sort_values("year").reset_index(drop=True)
    numeric = fin.drop(columns=["year"]).apply(pd.to_numeric, errors="coerce")
    numeric_cols = [c for c in numeric.columns if numeric[c].notna().any()]

    out = pd.DataFrame({"year": fin["year"]})
    for col in numeric_cols:
        prev = numeric[col].shift(1)
        out[f"{col}_yoy_abs"] = numeric[col] - prev
        # Standard % change = (curr - prev) / |prev|. Divide-by-zero base -> NaN, not inf, since
        # a fabricated infinite/huge % is more misleading than an honest "not computable".
        pct = (numeric[col] - prev) / prev.abs()
        pct = pct.where(prev.notna() & (prev != 0))
        out[f"{col}_yoy_pct"] = pct
    return out