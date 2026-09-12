"""Phase 5: turn the priority score + diagnosis into an actual fare-change
recommendation -- the case's second question, answered as a number.

Capacity comes from a real reference: aircraft type (segmentsEquipmentDescription,
a real field) mapped to publicly documented typical domestic seat counts. For a
connecting itinerary, the smallest aircraft across segments is the real binding
constraint (a passenger needs a seat on every leg). This is a real-but-approximate
join -- actual seat count varies by carrier's specific cabin configuration -- same
category of approximation as the hospital project's aircraft-type-free bed-to-
theatre conversion.

Fare-change sizing: bid-price logic in spirit (raise when running ahead of the
expected pace, cut when running behind), sized off the same z-score that drives
the priority flag, with a stated sensitivity and cap -- this is a stated business
rule, not fitted, exactly like the newsvendor Cu/Co ratio in the hospital project.
"""
import numpy as np
import pandas as pd
from pathlib import Path

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# Real, publicly documented typical domestic single-cabin-mix seat counts by
# aircraft type. Approximate -- actual counts vary by carrier's specific
# configuration -- used as a capacity reference, not an exact figure.
AIRCRAFT_SEATS = {
    "Boeing 737-800": 160, "Boeing 737-900": 180, "Boeing 737-700": 140,
    "Boeing 757-200": 190, "Boeing 767-300": 210,
    "Airbus A321": 190, "Airbus A320": 150, "Airbus A319": 128,
    "AIRBUS INDUSTRIE A321 SHARKLETS": 190, "AIRBUS INDUSTRIE A320 SHARKLETS": 150,
    "Embraer 175": 76, "Embraer 175 (Enhanced Winglets)": 76,
    "Embraer 190": 99, "Bombardier CRJ 900": 76, "Bombardier CRJ 700": 65,
    "McDonnell Douglas": 140,
}
DEFAULT_SEATS = 150  # fallback for unmapped/unusual equipment strings

SENSITIVITY = 4.0    # % fare change per 1 std-dev of seat-pace surprise (stated assumption)
MAX_CHANGE_PCT = 20  # cap in either direction (stated assumption, avoids unrealistic swings)
FLAG_THRESHOLD = 1.0  # |priority_score| below this -> no action (matches "Normal" diagnosis)


def estimate_capacity(equipment: str) -> int:
    if not isinstance(equipment, str) or not equipment:
        return DEFAULT_SEATS
    segments = [s for s in equipment.split("||") if s]
    if not segments:
        return DEFAULT_SEATS
    seat_counts = [AIRCRAFT_SEATS.get(s, DEFAULT_SEATS) for s in segments]
    return min(seat_counts)  # smallest leg is the real binding constraint


def recommend_change(row) -> float:
    if abs(row["priority_score"]) < FLAG_THRESHOLD:
        return 0.0
    # z_seatsRemaining < 0 means fewer seats left than expected (selling fast) -> raise fare
    # z_seatsRemaining > 0 means more seats left than expected (selling slow) -> cut fare
    raw_pct = -row["z_seatsRemaining"] * SENSITIVITY
    return float(np.clip(raw_pct, -MAX_CHANGE_PCT, MAX_CHANGE_PCT))


if __name__ == "__main__":
    scored = pd.read_csv(PROCESSED_DIR / "flights_scored.csv")
    raw = pd.read_csv(PROCESSED_DIR / "flights_clean.csv",
                       usecols=["legId", "searchDate", "segmentsEquipmentDescription"])
    raw = raw[raw["searchDate"] == "2022-04-17"].drop_duplicates("legId")

    df = scored.merge(raw[["legId", "segmentsEquipmentDescription"]], on="legId", how="left")
    df["estimated_capacity"] = df["segmentsEquipmentDescription"].apply(estimate_capacity)
    df["estimated_load_factor"] = (1 - df["seatsRemaining"] / df["estimated_capacity"]).clip(0, 1)

    df["recommended_change_pct"] = df.apply(recommend_change, axis=1)
    df["recommended_fare"] = (df["totalFare"] * (1 + df["recommended_change_pct"] / 100)).round(2)

    # Sanity check: does the estimated load factor actually track the diagnosis
    # in the direction it should? (fast-selling flights should show higher
    # estimated occupancy than slow-selling ones)
    print("=== Sanity check: mean estimated load factor by diagnosis ===")
    print(df.groupby("diagnosis")["estimated_load_factor"].mean().sort_values(ascending=False))

    top = df.sort_values("priority_score", ascending=False).head(10)
    print(f"\n=== Top 10 flights: recommended fare action ===")
    print(top[["route", "flightDate", "diagnosis", "totalFare", "recommended_change_pct",
               "recommended_fare", "estimated_load_factor"]].to_string(index=False))

    n_raise = (df["recommended_change_pct"] > 0).sum()
    n_cut = (df["recommended_change_pct"] < 0).sum()
    n_hold = (df["recommended_change_pct"] == 0).sum()
    print(f"\nAcross {len(df)} scored flights: {n_raise} raise, {n_cut} cut, {n_hold} hold "
          f"({100*n_hold/len(df):.1f}% need no review)")
    print(f"Mean |change| among flagged flights: "
          f"{df.loc[df['recommended_change_pct']!=0,'recommended_change_pct'].abs().mean():.1f}%")

    df.to_csv(PROCESSED_DIR / "flights_final_policy.csv", index=False)
    print(f"\nSaved to data/processed/flights_final_policy.csv")
