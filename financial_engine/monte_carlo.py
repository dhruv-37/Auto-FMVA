"""Table 9 (monte_carlo) — VaR simulation from Table 5 (price_history). Bucket A."""
import numpy as np
import pandas as pd


def run_monte_carlo_var(price_history: pd.DataFrame, n_trials: int = 10000,
                         percentiles: list[float] = (5.0, 3.0, 0.5, 8.0),
                         current_price: float = None, seed: int = 42) -> tuple[pd.DataFrame, dict]:
    returns = price_history["stock_return_pct"].dropna().values
    mean, std = returns.mean(), returns.std()

    rng = np.random.default_rng(seed)
    simulated_returns = rng.normal(loc=mean, scale=std, size=n_trials)

    trials_df = pd.DataFrame({
        "trial_id": np.arange(1, n_trials + 1),
        "simulated_return": simulated_returns,
    })

    cmp = current_price if current_price is not None else price_history["close_price"].iloc[-1]
    var_rows = []
    for p in percentiles:
        threshold_return = np.percentile(simulated_returns, p)
        implied_price = cmp * (1 + threshold_return)
        var_inr = implied_price - cmp
        var_rows.append(dict(percentile=p, confidence=100 - p, var_pct=threshold_return * 100,
                              var_stock_price=implied_price, var_inr=var_inr))

    summary = dict(
        hist_mean=mean, hist_std=std, hist_min=returns.min(), hist_max=returns.max(), cmp=cmp,
        sim_mean=simulated_returns.mean(), sim_std=simulated_returns.std(),
        sim_min=simulated_returns.min(), sim_max=simulated_returns.max(),
        var_table=pd.DataFrame(var_rows),
    )
    return trials_df, summary
