"""Phase 2: EDA -- does the data actually show real revenue-management behavior?

Two angles:
1. Cross-sectional: across many different flights, does fare rise and
   seats-remaining fall as departure approaches? (the expected curve)
2. Longitudinal (real, not assumed): for the 76,517 flights observed on BOTH
   search dates, did fare/scarcity actually move in the expected direction
   day-over-day? This is a genuine before/after check, not a cross-sectional proxy.
"""
import pandas as pd
import numpy as np
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"


def cross_sectional_curve(df: pd.DataFrame):
    curve = df.groupby("days_to_departure").agg(
        mean_fare=("totalFare", "mean"),
        median_fare=("totalFare", "median"),
        mean_seats_remaining=("seatsRemaining", "mean"),
        n_flights=("legId", "count"),
    ).reset_index().sort_values("days_to_departure")
    print("=== Cross-sectional: fare & seats-remaining by days-to-departure ===")
    print(curve.to_string(index=False))
    return curve


def longitudinal_check(df: pd.DataFrame):
    counts = df["legId"].value_counts()
    paired_ids = counts[counts > 1].index
    paired = df[df["legId"].isin(paired_ids)].sort_values(["legId", "searchDate"])

    first = paired.groupby("legId").first()
    last = paired.groupby("legId").last()

    fare_change = last["totalFare"].values - first["totalFare"].values
    seats_change = last["seatsRemaining"].values - first["seatsRemaining"].values

    print(f"\n=== Longitudinal: {len(first)} real flights observed 1 day apart ===")
    print(f"Fare change (later search - earlier search): "
          f"mean={fare_change.mean():+.2f}, median={np.median(fare_change):+.2f}, "
          f"% that rose={100*(fare_change > 0).mean():.1f}%, "
          f"% unchanged={100*(fare_change == 0).mean():.1f}%, "
          f"% that fell={100*(fare_change < 0).mean():.1f}%")
    print(f"Seats-remaining change: "
          f"mean={seats_change.mean():+.2f}, median={np.median(seats_change):+.2f}, "
          f"% that fell (normal depletion)={100*(seats_change < 0).mean():.1f}%, "
          f"% unchanged={100*(seats_change == 0).mean():.1f}%, "
          f"% that ROSE (the anomaly flagged earlier)={100*(seats_change > 0).mean():.1f}%")

    return pd.DataFrame({
        "legId": first.index, "fare_change": fare_change, "seats_change": seats_change,
        "days_to_departure_at_first_search": first["days_to_departure"].values,
    })


def competitor_dispersion(df: pd.DataFrame):
    grp = df.groupby(["route", "flightDate"])["totalFare"]
    disp = grp.agg(["count", "min", "max", "mean", "std"]).reset_index()
    disp = disp[disp["count"] >= 5]  # routes/dates with enough real competing options
    disp["spread_pct"] = (disp["max"] - disp["min"]) / disp["mean"] * 100
    print(f"\n=== Competitor fare dispersion (route+date with >=5 real fare options) ===")
    print(f"{len(disp)} route-date combinations qualify")
    print(disp[["route", "flightDate", "count", "min", "max", "spread_pct"]]
          .sort_values("spread_pct", ascending=False).head(8).to_string(index=False))
    print(f"\nMedian spread across all qualifying route-dates: {disp['spread_pct'].median():.1f}%")


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "flights_clean.csv", parse_dates=["searchDate", "flightDate"])
    curve = cross_sectional_curve(df)
    curve.to_csv(PROCESSED_DIR / "expected_curve.csv", index=False)

    longit = longitudinal_check(df)
    longit.to_csv(PROCESSED_DIR / "longitudinal_pairs.csv", index=False)

    competitor_dispersion(df)
