"""Direct comparison: the original 250K cross-sectional sample (2 search
dates, no real booking-curve history possible) vs. the full-data longitudinal
extract (1,085 real flights, ~39 search dates each on average).

Same model family (Random Forest -- consistent with the rest of the project,
no boosting), same evaluation discipline (genuine time-based holdout), two
feature sets on the SAME longitudinal data to isolate what's actually driving
any improvement:
  (A) cross-sectional only -- route, weekday, days-to-departure, cabin,
      airline, distance -- the only features the 250K sample could ever
      support, since it never had within-flight history.
  (B) cross-sectional + real per-flight lag features -- fare/seats this
      SAME flight showed at its previous 1/3 searches, and a rolling mean --
      only possible because this data has genuine longitudinal depth.

If (B) beats (A) by more than noise, that's the real value of the full
dataset -- not just "more rows," but genuine booking-curve history no
cross-sectional sample could ever provide.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
CATEGORICAL = ["route", "segmentsCabinCode", "segmentsAirlineName"]
CROSS_SECTIONAL_FEATURES = ["route_code", "day_of_week", "days_to_departure", "isNonStop",
                             "isBasicEconomy", "isRefundable", "segmentsCabinCode_code",
                             "segmentsAirlineName_code", "totalTravelDistance"]
LAG_FEATURES = ["lag1_fare", "lag3_fare", "roll_mean_fare", "lag1_seats", "lag3_seats", "roll_mean_seats"]


def load_and_engineer():
    df = pd.read_csv(PROCESSED_DIR / "longitudinal_extract.csv", parse_dates=["searchDate", "flightDate"])
    df["route"] = df["startingAirport"] + "-" + df["destinationAirport"]
    df["day_of_week"] = df["flightDate"].dt.dayofweek
    df["days_to_departure"] = (df["flightDate"] - df["searchDate"]).dt.days
    df["flight_key"] = df["route"] + "|" + df["flightDate"].astype(str)
    for c in CATEGORICAL:
        df[f"{c}_code"] = df[c].astype("category").cat.codes
    for c in ["isNonStop", "isBasicEconomy", "isRefundable"]:
        df[c] = df[c].astype(int)

    # One row per (flight, searchDate) -- if a flight has multiple cabin/fare
    # options per search, take the cheapest (what a shopper would actually see
    # first), so lag features track one coherent price series per flight.
    df = df.sort_values(["flight_key", "searchDate", "totalFare"])
    df = df.drop_duplicates(["flight_key", "searchDate"], keep="first")
    df = df.sort_values(["flight_key", "searchDate"]).reset_index(drop=True)

    g = df.groupby("flight_key")
    df["lag1_fare"] = g["totalFare"].shift(1)
    df["lag3_fare"] = g["totalFare"].shift(3)
    df["roll_mean_fare"] = g["totalFare"].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)
    df["lag1_seats"] = g["seatsRemaining"].shift(1)
    df["lag3_seats"] = g["seatsRemaining"].shift(3)
    df["roll_mean_seats"] = g["seatsRemaining"].shift(1).rolling(5, min_periods=1).mean().reset_index(level=0, drop=True)

    return df.dropna(subset=LAG_FEATURES + CROSS_SECTIONAL_FEATURES)


def time_split(df, cutoff="2022-08-15"):
    train = df[df["searchDate"] < cutoff]
    test = df[df["searchDate"] >= cutoff]
    return train, test


def evaluate(train, test, features, target, label):
    model = RandomForestRegressor(n_estimators=200, max_depth=10, random_state=42, n_jobs=-1)
    model.fit(train[features], train[target])
    pred = model.predict(test[features])
    mae = mean_absolute_error(test[target], pred)
    naive_mae = mean_absolute_error(test[target], np.full(len(test), train[target].mean()))
    print(f"{label:45s} MAE={mae:7.2f}  (flat-mean baseline={naive_mae:7.2f}, "
          f"{100*(1-mae/naive_mae):5.1f}% better)")
    return mae


if __name__ == "__main__":
    df = load_and_engineer()
    print(f"Rows after lag-feature engineering (first few searches per flight dropped): {len(df):,}")
    print(f"Unique flights retained: {df['flight_key'].nunique():,}")

    train, test = time_split(df)
    print(f"Train: {len(train):,} rows (before 2022-08-15) | Test: {len(test):,} rows (on/after 2022-08-15)")

    print(f"\n=== Fare prediction ===")
    mae_a_fare = evaluate(train, test, CROSS_SECTIONAL_FEATURES, "totalFare",
                           "(A) Cross-sectional only (what 250K sample could ever support)")
    mae_b_fare = evaluate(train, test, CROSS_SECTIONAL_FEATURES + LAG_FEATURES, "totalFare",
                           "(B) + real per-flight booking-curve lags")

    print(f"\n=== Seats-remaining prediction ===")
    mae_a_seats = evaluate(train, test, CROSS_SECTIONAL_FEATURES, "seatsRemaining",
                            "(A) Cross-sectional only (what 250K sample could ever support)")
    mae_b_seats = evaluate(train, test, CROSS_SECTIONAL_FEATURES + LAG_FEATURES, "seatsRemaining",
                            "(B) + real per-flight booking-curve lags")

    print(f"\n=== The actual comparison ===")
    print(f"Fare:  lag features improve MAE by {100*(1-mae_b_fare/mae_a_fare):.1f}% over cross-sectional-only")
    print(f"Seats: lag features improve MAE by {100*(1-mae_b_seats/mae_a_seats):.1f}% over cross-sectional-only")

    df.to_csv(PROCESSED_DIR / "longitudinal_features.csv", index=False)
