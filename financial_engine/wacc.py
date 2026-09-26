"""Table 6 (peer_comps) + WACC calculation. Bucket A."""
import pandas as pd
from .schema import PEER_COMPS_COLS


def build_peer_comps(raw_peers: list[dict]) -> pd.DataFrame:
    """raw_peers: list of dicts with peer_ticker, peer_name, country, total_debt, total_equity,
    tax_rate, levered_beta, share_price, shares_outstanding, net_debt, revenue, ebitda, net_income."""
    df = pd.DataFrame(raw_peers)
    df["debt_to_equity"] = df["total_debt"] / df["total_equity"]
    df["debt_to_capital"] = df["total_debt"] / (df["total_debt"] + df["total_equity"])
    df["unlevered_beta"] = df["levered_beta"] / (1 + (1 - df["tax_rate"]) * df["debt_to_equity"])
    df["equity_value"] = df["share_price"] * df["shares_outstanding"]
    df["enterprise_value"] = df["equity_value"] + df["net_debt"]
    df["ev_revenue"] = df["enterprise_value"] / df["revenue"]
    df["ev_ebitda"] = df["enterprise_value"] / df["ebitda"]
    df["pe_ratio"] = df["equity_value"] / df["net_income"]
    for col in PEER_COMPS_COLS:
        if col not in df.columns:
            df[col] = pd.NA
    return df[PEER_COMPS_COLS]


def compute_wacc(peer_comps: pd.DataFrame, target_debt_equity: float, tax_rate: float,
                  risk_free_rate: float, equity_risk_premium: float,
                  pre_tax_cost_of_debt: float, current_debt: float, current_equity: float) -> dict:
    """Returns a dict of single-value WACC outputs (store as JSON or 1-row DataFrame)."""
    median_unlevered_beta = peer_comps["unlevered_beta"].median()
    levered_beta = median_unlevered_beta * (1 + (1 - tax_rate) * target_debt_equity)

    cost_of_equity = risk_free_rate + levered_beta * equity_risk_premium
    post_tax_cost_of_debt = pre_tax_cost_of_debt * (1 - tax_rate)

    total_capital = current_debt + current_equity
    current_debt_weight = current_debt / total_capital
    current_equity_weight = current_equity / total_capital
    target_debt_weight = target_debt_equity / (1 + target_debt_equity)
    target_equity_weight = 1 - target_debt_weight

    wacc = (post_tax_cost_of_debt * target_debt_weight) + (cost_of_equity * target_equity_weight)

    return dict(
        comps_median_unlevered_beta=median_unlevered_beta,
        levered_beta=levered_beta,
        cost_of_equity=cost_of_equity,
        pre_tax_cost_of_debt=pre_tax_cost_of_debt,
        post_tax_cost_of_debt=post_tax_cost_of_debt,
        current_debt_weight=current_debt_weight,
        current_equity_weight=current_equity_weight,
        target_debt_weight=target_debt_weight,
        target_equity_weight=target_equity_weight,
        wacc=wacc,
    )


def compute_beta_drifting(price_history: pd.DataFrame, raw_beta_weight: float = 0.75,
                           market_beta: float = 1.0) -> dict:
    """Table 5 -> regression beta + adjusted (Bloomberg-style) beta. Bucket A."""
    import numpy as np
    x = price_history["index_return_pct"].dropna()
    y = price_history["stock_return_pct"].dropna()
    n = min(len(x), len(y))
    x, y = x.iloc[-n:], y.iloc[-n:]
    raw_beta = np.cov(y, x)[0, 1] / np.var(x)
    adjusted_beta = raw_beta * raw_beta_weight + market_beta * (1 - raw_beta_weight)
    return dict(raw_beta=raw_beta, raw_beta_weight=raw_beta_weight,
                market_beta=market_beta, market_beta_weight=1 - raw_beta_weight,
                adjusted_beta=adjusted_beta)
