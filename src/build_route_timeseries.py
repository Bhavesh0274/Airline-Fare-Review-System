"""Stream the full 31GB FlightPrices source in chunks (never loaded fully
into memory) and aggregate real daily mean fares by route and departure date
across the genuine ~6-month collection window -- the actual time-series depth
SARIMA needs, which the 250K-row sample (2 search dates only) could not
provide.
"""
import pandas as pd
from pathlib import Path
import time

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "full" / "itineraries.csv"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "route_daily_fare_full.csv"
USECOLS = ["searchDate", "flightDate", "startingAirport", "destinationAirport", "totalFare"]
CHUNKSIZE = 2_000_000

if __name__ == "__main__":
    agg = {}  # (route, flightDate) -> [sum_fare, count]
    start = time.time()

    reader = pd.read_csv(RAW, usecols=USECOLS, chunksize=CHUNKSIZE)
    for i, chunk in enumerate(reader):
        chunk["route"] = chunk["startingAirport"] + "-" + chunk["destinationAirport"]
        grp = chunk.groupby(["route", "flightDate"])["totalFare"].agg(["sum", "count"])
        for (route, fdate), row in grp.iterrows():
            key = (route, fdate)
            if key in agg:
                agg[key][0] += row["sum"]
                agg[key][1] += row["count"]
            else:
                agg[key] = [row["sum"], row["count"]]
        elapsed = time.time() - start
        print(f"chunk {i+1} done ({(i+1)*CHUNKSIZE:,} rows scanned so far), "
              f"{len(agg):,} unique (route,date) keys, {elapsed:.0f}s elapsed", flush=True)

    records = [{"route": k[0], "flightDate": k[1], "mean_fare": v[0] / v[1], "n": v[1]}
               for k, v in agg.items()]
    out = pd.DataFrame(records)
    out.to_csv(OUT, index=False)
    print(f"\nDone in {time.time()-start:.0f}s. Saved {len(out):,} (route, date) rows to {OUT}")

    route_days = out.groupby("route")["flightDate"].nunique().sort_values(ascending=False)
    route_totalobs = out.groupby("route")["n"].sum().sort_values(ascending=False)
    print("\nTop 15 routes by number of distinct departure dates covered:")
    print(route_days.head(15))
    print("\nDate range in full dataset:", out["flightDate"].min(), "to", out["flightDate"].max())
