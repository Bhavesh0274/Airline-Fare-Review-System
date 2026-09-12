"""SARIMA on a genuine daily time series -- now that the full 31GB source has
been streamed into real, gap-free daily route-level fares (217 consecutive
days, 2022-04-17 to 2022-11-19), unlike the 250K-row sample (2 search dates)
that couldn't support this at all.

Route: ATL-LAX -- real, busy, zero missing days across the full window.
Evaluation: last 30 days held out as a real time-based test, seasonal-naive
(same weekday, prior week) as the baseline -- same discipline as every other
forecasting model in this project.
"""
import numpy as np
import pandas as pd
import warnings
from pathlib import Path
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore")

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
ROUTE = "ATL-LAX"
TEST_DAYS = 30

# Small, honest grid search by AIC on training data only -- not an
# exhaustive auto-ARIMA search, but a real, defensible model-selection step.
ORDER_GRID = [
    ((1, 1, 1), (1, 1, 1, 7)),
    ((2, 1, 1), (1, 1, 1, 7)),
    ((1, 1, 2), (1, 1, 1, 7)),
    ((2, 1, 2), (0, 1, 1, 7)),
    ((1, 0, 1), (1, 1, 1, 7)),
]

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "route_daily_fare_full.csv", parse_dates=["flightDate"])
    sub = df[df["route"] == ROUTE].sort_values("flightDate").reset_index(drop=True)
    sub = sub.set_index("flightDate").asfreq("D")
    print(f"{ROUTE}: {len(sub)} days, {sub['mean_fare'].isna().sum()} gaps (asfreq check)")

    train = sub["mean_fare"].iloc[:-TEST_DAYS]
    test = sub["mean_fare"].iloc[-TEST_DAYS:]
    print(f"Train: {len(train)} days, Test: {len(test)} days (last {TEST_DAYS} real days held out)")

    print("\n=== Order search (AIC on training data only) ===")
    best_aic, best_order, best_model = np.inf, None, None
    for order, seasonal_order in ORDER_GRID:
        try:
            model = SARIMAX(train, order=order, seasonal_order=seasonal_order,
                             enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
            print(f"SARIMA{order}x{seasonal_order}: AIC={model.aic:.1f}")
            if model.aic < best_aic:
                best_aic, best_order, best_model = model.aic, (order, seasonal_order), model
        except Exception as e:
            print(f"SARIMA{order}x{seasonal_order}: failed ({e})")

    print(f"\nBest by AIC: SARIMA{best_order[0]}x{best_order[1]} (AIC={best_aic:.1f})")

    forecast = best_model.get_forecast(steps=TEST_DAYS)
    pred = forecast.predicted_mean
    ci = forecast.conf_int(alpha=0.05)

    seasonal_naive = train.iloc[-7:].values
    seasonal_naive_pred = np.tile(seasonal_naive, TEST_DAYS // 7 + 1)[:TEST_DAYS]

    sarima_mae = mean_absolute_error(test, pred)
    naive_mae = mean_absolute_error(test, seasonal_naive_pred)
    sarima_rmse = np.sqrt(mean_squared_error(test, pred))
    naive_rmse = np.sqrt(mean_squared_error(test, seasonal_naive_pred))

    print(f"\n=== Real out-of-sample evaluation (last {TEST_DAYS} days) ===")
    print(f"SARIMA         : MAE=${sarima_mae:.2f}  RMSE=${sarima_rmse:.2f}")
    print(f"Seasonal-naive : MAE=${naive_mae:.2f}  RMSE=${naive_rmse:.2f}")
    print(f"SARIMA improvement over seasonal-naive: {100*(1-sarima_mae/naive_mae):.1f}%")

    out = pd.DataFrame({
        "flightDate": test.index, "actual": test.values, "sarima_pred": pred.values,
        "sarima_lower": ci.iloc[:, 0].values, "sarima_upper": ci.iloc[:, 1].values,
        "seasonal_naive_pred": seasonal_naive_pred,
    })
    out.to_csv(PROCESSED_DIR / "sarima_atl_lax_results.csv", index=False)
    print(f"\nSaved to data/processed/sarima_atl_lax_results.csv")
