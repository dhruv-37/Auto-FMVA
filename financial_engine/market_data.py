"""Table 1 (company_profile) + Table 4 (market_data). Built from Data Sheet's META and PRICE:
sections plus the financials table (Table 2) for debt/cash/EBITDA/sales."""
import pandas as pd
from .schema import MARKET_DATA_COLS
from .ingest import _extract_data_sheet_section, DATA_SHEET_SECTIONS


def _meta_value(raw: pd.DataFrame, label: str):
    row = raw[raw[0].astype(str).str.strip().str.lower() == label.lower()]
    return row.iloc[0, 1] if not row.empty else None


def build_company_profile(sheets: dict, ticker_nse: str = None, ticker_bse: str = None) -> dict:
    """Table 1 — single dict/1-row record, not year-indexed."""
    raw = sheets["Data Sheet"]
    return dict(
        ticker_nse=ticker_nse, ticker_bse=ticker_bse,
        company_name=_meta_value(raw, "COMPANY NAME"),
        current_price=_meta_value(raw, "Current Price"),
        market_cap=_meta_value(raw, "Market Capitalization"),
        face_value=_meta_value(raw, "Face Value"),
    )


def _extract_price_section(raw: pd.DataFrame) -> pd.Series:
    """PRICE: section in Data Sheet is a single row (year-end close), unlike the 3-statement
    sections — it has no separate 'Report Date' sub-row, it reuses the BALANCE SHEET one."""
    col0 = raw[0].astype(str).str.strip().str.upper()
    price_idx = col0[col0 == "PRICE:"].index
    if len(price_idx) == 0:
        return pd.Series(dtype=float)
    row = raw.loc[price_idx[0]]
    # Years come from the nearest preceding "Report Date" row (Balance Sheet's, by Data Sheet layout)
    report_date_rows = raw.index[col0 == "REPORT DATE"]
    years_row = raw.loc[[i for i in report_date_rows if i < price_idx[0]][-1]]
    years = [pd.to_datetime(y).year for y in years_row[1:] if pd.notna(y)]
    prices = row[1:1 + len(years)].astype(float)
    return pd.Series(prices.values, index=years)


def build_market_data(sheets: dict, financials: pd.DataFrame) -> pd.DataFrame:
    """Table 4 — one row per year. Needs `financials` (Table 2) for ebitda/sales/debt/cash."""
    raw = sheets["Data Sheet"]
    share_price_by_year = _extract_price_section(raw)

    m = pd.DataFrame({"year": financials["year"]})
    m["share_price"] = m["year"].map(share_price_by_year)
    m["shares_outstanding"] = financials["shares_outstanding"]
    m["market_cap"] = m["share_price"] * m["shares_outstanding"] / 1e7   # shares raw count -> crore basis
    m["cash_equivalents"] = financials["cash_bank"]
    m["total_debt"] = financials["borrowings"]
    m["minority_interest"] = pd.NA
    m["enterprise_value"] = m["market_cap"] + m["total_debt"] - m["cash_equivalents"] + m["minority_interest"].fillna(0)

    eps = financials["eps"]
    m["pe_ratio"] = m["share_price"] / eps
    m["ev_ebitda"] = m["enterprise_value"] / financials["ebitda"]
    m["ev_sales"] = m["enterprise_value"] / financials["sales"]

    book_value_per_share = (financials["equity_share_capital"] + financials["reserves"]) * 1e7 / financials["shares_outstanding"]
    m["price_to_book"] = m["share_price"] / book_value_per_share

    m["price_52w_high"] = pd.NA   # needs daily price_history (Table 5) — fill in from that table
    m["price_52w_low"] = pd.NA

    for col in MARKET_DATA_COLS:
        if col not in m.columns:
            m[col] = pd.NA
    return m[MARKET_DATA_COLS]


def fill_52w_high_low(market_data: pd.DataFrame, price_history: pd.DataFrame) -> pd.DataFrame:
    """Populate price_52w_high/low for the latest year using Table 5 (price_history)."""
    latest_year = market_data["year"].max()
    last_365 = price_history.tail(252)  # ~1 trading year
    market_data = market_data.copy()
    market_data.loc[market_data["year"] == latest_year, "price_52w_high"] = last_365["close_price"].max()
    market_data.loc[market_data["year"] == latest_year, "price_52w_low"] = last_365["close_price"].min()
    return market_data
