"""Second streaming pass over the full 31GB source: this time extracting
every raw row (not aggregating) for a handful of well-covered routes, to get
genuine longitudinal booking curves -- the same future flight (route +
flightDate) tracked across dozens of real search dates as departure
approaches. The 250K-row sample only ever had 2 search dates; this is the
actual advantage of the full dataset worth comparing against.
"""
import pandas as pd
from pathlib import Path
import time

RAW = Path(__file__).resolve().parent.parent / "data" / "raw" / "full" / "itineraries.csv"
OUT = Path(__file__).resolve().parent.parent / "data" / "processed" / "longitudinal_extract.csv"
USECOLS = ["legId", "searchDate", "flightDate", "startingAirport", "destinationAirport",
           "isBasicEconomy", "isRefundable", "isNonStop", "totalFare", "seatsRemaining",
           "totalTravelDistance", "segmentsAirlineName", "segmentsCabinCode"]
CHUNKSIZE = 2_000_000

# 5 routes already confirmed to have full 217-day coverage with zero gaps in
# the daily aggregation built earlier.
TARGET_ROUTES = {"ATL-LAX", "ATL-JFK", "ATL-ORD", "ATL-BOS", "ATL-SFO"}

if __name__ == "__main__":
    start = time.time()
    first_write = True
    total_kept = 0

    reader = pd.read_csv(RAW, usecols=USECOLS, chunksize=CHUNKSIZE)
    for i, chunk in enumerate(reader):
        chunk["route"] = chunk["startingAirport"] + "-" + chunk["destinationAirport"]
        matched = chunk[chunk["route"].isin(TARGET_ROUTES)]
        if len(matched):
            matched.to_csv(OUT, mode="w" if first_write else "a", header=first_write, index=False)
            first_write = False
            total_kept += len(matched)
        elapsed = time.time() - start
        print(f"chunk {i+1} done ({(i+1)*CHUNKSIZE:,} rows scanned), "
              f"{total_kept:,} matching rows kept so far, {elapsed:.0f}s elapsed", flush=True)

    print(f"\nDone in {time.time()-start:.0f}s. Extracted {total_kept:,} rows to {OUT}")
