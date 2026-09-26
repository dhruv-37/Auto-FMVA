"""Table 2 (financials) + Table 5 (price_history). Screener .xlsx -> normalized DataFrames."""
import pandas as pd
import yfinance as yf
from pathlib import Path
from .schema import FINANCIALS_COLS, validate_financials

# Screener's row labels vary by company/sector/export type (full vs "lite" export) and by
# case ("Net Profit" vs "Net profit"). Matching below is case-insensitive; extend this map
# with new label variants as you hit them, lowercase or not, it works either way.
ROW_MAP = {
    "Sales": "sales", "Net Sales": "sales",
    "Raw Material Cost": "cogs", "Expenses": "cogs",
    "Change in Inventory": "change_in_inventory",
    "Power and Fuel": "power_fuel_exp",
    "Other Mfr. Exp": "other_mfr_exp",
    "Employee Cost": "employee_cost",
    "Selling and admin": "selling_admin_exp",
    "Other Expenses": "other_expenses",
    "Other Income": "other_income",
    "Operating Profit": "ebitda",          # "lite" Screener export gives EBITDA directly
    "OPM": "ebitda_margin",
    "Depreciation": "depreciation",
    "Interest": "interest",
    "Profit before tax": "ebt",
    "Tax": "tax",
    "Net Profit": "net_profit", "Net profit": "net_profit",   # case variant that caused the KeyError
    "No. of Equity Shares Oustanding": "shares_outstanding",
    "No. of Equity Shares": "shares_outstanding",     # Data Sheet's label (no "Outstanding")
    "Earnings Per Share": "eps", "EPS": "eps",
    "Dividend Per Share": "dps",
    "Dividend Amount": "dividend_amount",              # Data Sheet gives total ₹, not per-share
    "Dividend Payout": "dividend_payout_ratio",
    "Equity Share Capital": "equity_share_capital",
    "Reserves": "reserves",
    "Borrowings": "borrowings",
    "Other Liabilities": "other_liabilities",
    "Total Equity and Liabilities": "total_equity_liabilities",   # remapped from bare "Total" below
    "Net Block": "fixed_assets_net_block", "Fixed Assets Net Block": "fixed_assets_net_block",
    "Capital Work in Progress": "capital_wip", "Capital Work In Progress": "capital_wip",
    "Investments": "investments_noncurrent",
    "Other Assets": "other_noncurrent_assets",
    "Total Non-Current Assets": "total_noncurrent_assets",
    "Total Assets": "total_assets",          # remapped from second bare "Total" below
    "Recievables": "receivables", "Receivables": "receivables", "Debtors": "receivables",
    "Inventory": "inventory",
    "Cash & Bank": "cash_bank",
    "Total Current Assets": "total_current_assets",
    "Working Capital": "net_working_capital_screener",   # informational only, not in schema
    "Cash from Operating Activity": "cash_from_operating_activities",
    "Cash from Investing Activity": "cash_from_investing_activities",
    "Cash from Financing Activity": "cash_from_financing_activities",
    "Net Cash Flow": "net_cash_flow",
}


DATA_SHEET_SECTIONS = ["PROFIT & LOSS", "QUARTERS", "BALANCE SHEET", "CASH FLOW:", "PRICE:", "DERIVED:"]


def load_screener_export(filepath: str) -> dict:
    """Reads the "Data Sheet" tab — the ONE sheet whose values survive when a Screener export is
    saved/re-saved outside Excel. The pretty "Profit & Loss" / "Balance Sheet" / "Cash Flow" tabs
    are formula-driven off Data Sheet; if the file was saved by a tool that doesn't recalculate
    formulas (LibreOffice headless, some scripted exports), those tabs come through completely
    blank — which is what produced the KeyError. Data Sheet is the source of truth either way."""
    xls = pd.ExcelFile(filepath)
    sheets = {"Data Sheet": pd.read_excel(xls, sheet_name="Data Sheet", header=None)}
    for name in ["Profit & Loss", "Balance Sheet", "Cash Flow"]:
        if name in xls.sheet_names:
            sheets[name] = pd.read_excel(xls, sheet_name=name)
    return sheets


def _extract_data_sheet_section(raw: pd.DataFrame, section_label: str) -> pd.DataFrame:
    """Slice out one section block (e.g. "BALANCE SHEET") from the raw Data Sheet: the label row
    itself, the "Report Date" row (years) right after it, and every line-item row up to the next
    section header. Returns a tidy DataFrame with 'line_item' + one column per year."""
    col0 = raw[0].astype(str).str.strip().str.upper()
    starts = raw.index[col0 == section_label.upper()]
    if len(starts) == 0:
        return pd.DataFrame()
    start = starts[0]
    later_headers = raw.index[(raw.index > start) & col0.isin([s.upper() for s in DATA_SHEET_SECTIONS])]
    end = later_headers.min() if len(later_headers) else len(raw)

    block = raw.loc[start + 1:end - 1].reset_index(drop=True)
    if block.empty or str(block.iloc[0, 0]).strip().lower() != "report date":
        return pd.DataFrame()

    years = block.iloc[0, 1:].tolist()
    year_labels = [pd.to_datetime(y).year if pd.notna(y) else None for y in years]

    body = block.iloc[1:].reset_index(drop=True)
    body.columns = ["line_item"] + year_labels
    body["line_item"] = body["line_item"].astype(str).str.strip()
    body = body[body["line_item"].notna() & (body["line_item"] != "") & (body["line_item"] != "nan")]

    # Balance Sheet reuses the bare label "Total" for both Total Equity+Liabilities (1st) and
    # Total Assets (2nd) — position, not text, disambiguates them.
    if section_label.upper() == "BALANCE SHEET":
        total_rows = body.index[body["line_item"].str.lower() == "total"].tolist()
        if len(total_rows) >= 1:
            body.loc[total_rows[0], "line_item"] = "Total Equity and Liabilities"
        if len(total_rows) >= 2:
            body.loc[total_rows[1], "line_item"] = "Total Assets"
    return body


def _melt_section(body: pd.DataFrame) -> pd.DataFrame:
    """line_item + year columns -> tidy (year, field, value), matched against ROW_MAP case-insensitively."""
    if body.empty:
        return pd.DataFrame(columns=["year"])
    row_map_lower = {k.lower(): v for k, v in ROW_MAP.items()}
    body = body.copy()
    body["field"] = body["line_item"].str.lower().map(row_map_lower)
    body = body[body["field"].notna()]
    year_cols = [c for c in body.columns if c not in ("line_item", "field")]
    long_df = body.melt(id_vars="field", value_vars=year_cols, var_name="year", value_name="value")
    return long_df.pivot_table(index="year", columns="field", values="value", aggfunc="first").reset_index()


def normalize_financials(sheets: dict) -> pd.DataFrame:
    """Merge P&L + Balance Sheet + Cash Flow (parsed from Data Sheet) into one financials-schema DataFrame."""
    raw = sheets["Data Sheet"]
    parts = [_melt_section(_extract_data_sheet_section(raw, sec))
             for sec in ("PROFIT & LOSS", "BALANCE SHEET", "CASH FLOW:")]
    parts = [p for p in parts if not p.empty]
    if not parts:
        raise ValueError("Could not find PROFIT & LOSS / BALANCE SHEET / CASH FLOW sections in Data Sheet — "
                          "check the export isn't a blank template (no 'Report Date' row with real years).")

    out = parts[0]
    for p in parts[1:]:
        out = out.merge(p, on="year", how="outer")

    out["year"] = out["year"].astype(float)
    out = out.sort_values("year").reset_index(drop=True)

    # Derived fields not directly in ROW_MAP. Guard every derivation with .get() / column checks —
    # different Screener export tiers ("full" vs "lite") don't provide the same set of raw rows.
    out["sales_growth"] = out["sales"].pct_change()
    out["gross_profit"] = out["sales"] - out.get("cogs", pd.Series(0, index=out.index))
    out["gross_margin"] = out["gross_profit"] / out["sales"]

    if "ebitda" not in out.columns or out["ebitda"].isna().all():
        # No direct "Operating Profit" row — derive from expense line items instead
        expense_cols = [c for c in ("power_fuel_exp", "other_mfr_exp", "employee_cost",
                                     "selling_admin_exp", "other_expenses") if c in out.columns]
        out["ebitda"] = out["gross_profit"] - out[expense_cols].sum(axis=1) + out.get(
            "other_income", pd.Series(0, index=out.index))
    out["ebitda_margin"] = out["ebitda"] / out["sales"]

    out["effective_tax_rate"] = out["tax"] / out["ebt"]
    out["net_margin"] = out["net_profit"] / out["sales"]

    # EPS/DPS aren't always present as their own rows — derive from shares_outstanding /
    # dividend_amount when Screener's export gives totals instead of per-share figures.
    if ("eps" not in out.columns or out["eps"].isna().all()) and "shares_outstanding" in out.columns:
        # net_profit is in Rs crore, shares_outstanding is a raw share count — convert crore to
        # rupees (x 1e7) before dividing, or EPS comes out ~1e-5 too small. Same units bug applies
        # to any other "figure in crore" / "figure in absolute units" combination — check both sides.
        out["eps"] = (out["net_profit"] * 1e7) / out["shares_outstanding"]
    if "eps" in out.columns:
        out["eps_growth"] = out["eps"].pct_change()

    if ("dps" not in out.columns or out["dps"].isna().all()) and \
       "dividend_amount" in out.columns and "shares_outstanding" in out.columns:
        out["dps"] = out["dividend_amount"] / out["shares_outstanding"]
    if "dps" in out.columns:
        out["dps_growth"] = out["dps"].pct_change()

    if ("dividend_payout_ratio" not in out.columns or out["dividend_payout_ratio"].isna().all()) and \
       "dps" in out.columns and "eps" in out.columns:
        out["dividend_payout_ratio"] = out["dps"] / out["eps"]
    if "dividend_payout_ratio" in out.columns:
        out["retained_earnings_pct"] = 1 - out["dividend_payout_ratio"]

    if "total_current_assets" not in out.columns or out["total_current_assets"].isna().all():
        cur_asset_cols = [c for c in ("receivables", "inventory", "cash_bank") if c in out.columns]
        out["total_current_assets"] = out[cur_asset_cols].sum(axis=1) if cur_asset_cols else pd.NA

    out["balance_check"] = (out["total_assets"].round(0) == out["total_equity_liabilities"].round(0))

    # Ensure every schema column exists (fills missing with NaN)
    for col in FINANCIALS_COLS:
        if col not in out.columns:
            out[col] = pd.NA
    out = out[FINANCIALS_COLS]

    validate_financials(out)
    return out


def load_price_history(ticker_symbol: str, index_symbol: str = "^NSEI", period: str = "5y") -> pd.DataFrame:
    """Table 5 — stock + index prices, daily returns."""
    stock = yf.Ticker(ticker_symbol).history(period=period)[["Close", "Volume"]]
    stock = stock.rename(columns={"Close": "close_price", "Volume": "volume"})
    index = yf.Ticker(index_symbol).history(period=period)[["Close"]]
    index = index.rename(columns={"Close": "index_close_price"})

    merged = stock.join(index, how="inner")
    merged["stock_return_pct"] = merged["close_price"].pct_change()
    merged["index_return_pct"] = merged["index_close_price"].pct_change()
    merged = merged.reset_index().rename(columns={"Date": "date"})
    return merged[["date", "close_price", "volume", "index_close_price",
                    "stock_return_pct", "index_return_pct"]]


def build_company_dataset(company_name: str, screener_xlsx: str, ticker_symbol: str,
                           index_symbol: str = "^NSEI", out_dir: str = "data/processed"):
    sheets = load_screener_export(screener_xlsx)
    financials = normalize_financials(sheets)
    prices = load_price_history(ticker_symbol, index_symbol)

    company_dir = Path(out_dir) / company_name
    company_dir.mkdir(parents=True, exist_ok=True)
    financials.to_csv(company_dir / "financials.csv", index=False)
    prices.to_csv(company_dir / "price_history.csv", index=False)
    return financials, prices


if __name__ == "__main__":
    financials, prices = build_company_dataset(
        company_name="RELIANCE",
        screener_xlsx="data/raw/RELIANCE.xlsx",
        ticker_symbol="RELIANCE.NS",
    )
    print(financials.tail())
    print(prices.tail())
