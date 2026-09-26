"""Table: Forecasting — linear regression on revenue (numpy). Bucket A, never Claude."""
import numpy as np
import pandas as pd


def run_regression_forecast(fin: pd.DataFrame, forecast_years: list[int],
                             y_col: str = "sales") -> tuple[pd.DataFrame, dict]:
    """`fin` = financials table (Table 2). Fits year -> y_col via OLS (numpy.polyfit, degree 1)
    on actual (non-null) history, then projects `forecast_years`.
    Returns (forecast_df with year/predicted/is_actual, summary dict with slope/intercept/r_squared)."""
    hist = fin[["year", y_col]].dropna()
    if len(hist) < 2:
        raise ValueError(f"Need at least 2 non-null ({'year'}, {y_col}) rows to fit a regression, got {len(hist)}")

    x = hist["year"].to_numpy(dtype=float)
    y = hist[y_col].to_numpy(dtype=float)

    slope, intercept = np.polyfit(x, y, deg=1)
    fitted = slope * x + intercept
    ss_res = np.sum((y - fitted) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r_squared = 1 - ss_res / ss_tot if ss_tot != 0 else float("nan")

    forecast_x = np.array(forecast_years, dtype=float)
    forecast_y = slope * forecast_x + intercept

    forecast_df = pd.concat([
        pd.DataFrame({"year": x, y_col: y, "is_actual": True}),
        pd.DataFrame({"year": forecast_x, y_col: forecast_y, "is_actual": False}),
    ], ignore_index=True).sort_values("year").reset_index(drop=True)

    summary = dict(
        y_col=y_col, slope=slope, intercept=intercept, r_squared=r_squared,
        n_obs=len(hist), forecast_years=list(forecast_years),
        forecast_values=forecast_y.tolist(),
    )
    return forecast_df, summary