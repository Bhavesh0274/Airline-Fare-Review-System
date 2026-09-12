"""Phase 1: load and clean the real Expedia FlightPrices sample.

Design note: the pre-made Kaggle samples (justinmitchel/flightprices-min) are
head-of-file slices of the 31GB source, so they only span 1-2 search dates --
not enough to track one flight across many searches before departure. Instead
of forcing a longitudinal analysis the data can't support, this uses the
cross-sectional variation that IS real here: many different flights, at every
days-to-departure from 1 to 19, observed on the same search date. That's
enough to build a real "expected fare/sell-through by days-to-departure"
benchmark from many flights, standing in for tracking one flight over time.
"""
import pandas as pd
from pathlib import Path

RAW_DIR = Path(__file__).resolve().parent.parent / "data" / "raw"
PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

KEEP_COLS = [
    "legId", "searchDate", "flightDate", "startingAirport", "destinationAirport",
    "isBasicEconomy", "isRefundable", "isNonStop", "baseFare", "totalFare",
    "seatsRemaining", "totalTravelDistance", "segmentsAirlineName", "segmentsEquipmentDescription",
    "segmentsCabinCode",
]


def load() -> pd.DataFrame:
    df = pd.read_csv(RAW_DIR / "itineraries-min-250k.csv", usecols=KEEP_COLS)
    df["searchDate"] = pd.to_datetime(df["searchDate"])
    df["flightDate"] = pd.to_datetime(df["flightDate"])
    df["days_to_departure"] = (df["flightDate"] - df["searchDate"]).dt.days
    df["day_of_week"] = df["flightDate"].dt.dayofweek  # 0=Mon
    df["route"] = df["startingAirport"] + "-" + df["destinationAirport"]

    before = len(df)
    df = df[(df["totalFare"] > 0) & (df["seatsRemaining"] >= 0) & (df["days_to_departure"] >= 0)]
    dropped = before - len(df)

    # Only the first (cheapest/primary) itinerary per leg+search combo when
    # duplicates exist for the same route/date/airline at different cabin/fare
    # combinations isn't collapsed here -- each row is a genuine distinct fare
    # option a shopper would see, so all are kept.

    return df, dropped


if __name__ == "__main__":
    df, dropped = load()
    print(f"Rows loaded: {len(df)} (dropped {dropped} invalid rows)")
    print(f"Unique legId: {df['legId'].nunique()}")
    print(f"Unique routes: {df['route'].nunique()}")
    print(f"Search dates: {sorted(df['searchDate'].dt.date.unique())}")
    print(f"Days-to-departure range: {df['days_to_departure'].min()} to {df['days_to_departure'].max()}")
    print(f"Unique airlines: {df['segmentsAirlineName'].nunique()}")
    print(f"\nTop 10 routes by flight-option count:")
    print(df["route"].value_counts().head(10))

    # Real longitudinal bonus: legIds observed on both search dates
    counts = df["legId"].value_counts()
    longitudinal = counts[counts > 1]
    print(f"\nFlights observed on both search dates (real before/after pairs): {len(longitudinal)}")

    df.to_csv(PROCESSED_DIR / "flights_clean.csv", index=False)
    print(f"\nSaved to data/processed/flights_clean.csv")
