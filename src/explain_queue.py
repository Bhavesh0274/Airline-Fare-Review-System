"""Explain specifically the flights shown in the dashboard's priority queue,
rather than hoping a random SHAP sample happens to overlap with them --
PermutationExplainer is slow enough (~2.4s/flight/output) that explaining
exactly the 12 displayed flights is both fast and precisely targeted.
"""
import numpy as np
import pandas as pd
from tensorflow import keras
from pathlib import Path

from priority_model import CATEGORICAL, NUMERIC, prep, make_inputs
from explainability import explain, top_drivers, MODEL_PATH

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "flights_clean.csv", parse_dates=["searchDate", "flightDate"])
    df = prep(df)
    train = df[df["searchDate"] == "2022-04-16"].reset_index(drop=True)
    test = df[df["searchDate"] == "2022-04-17"].reset_index(drop=True)

    X_train, num_scaler, distance_fill = make_inputs(train, fit=True)
    X_test, _, _ = make_inputs(test, scaler=num_scaler, distance_fill=distance_fill)

    def to_flat(inputs):
        cat_cols = [inputs[f"{c}_input"] for c in CATEGORICAL]
        return np.concatenate(cat_cols + [inputs["numeric_input"]], axis=1).astype("float32")

    train_flat = to_flat(X_train)
    test_flat = to_flat(X_test)
    background_flat = train_flat[np.random.RandomState(42).choice(len(train_flat), 30, replace=False)]

    model = keras.models.load_model(MODEL_PATH)

    policy = pd.read_csv(PROCESSED_DIR / "flights_final_policy.csv")
    top12 = policy.sort_values("priority_score", ascending=False).head(12)

    test_legid_to_pos = {lid: i for i, lid in enumerate(test["legId"])}
    positions = [test_legid_to_pos[lid] for lid in top12["legId"]]
    sample_flat = test_flat[positions]

    print(f"Explaining {len(positions)} flights (the actual displayed queue)...")
    fare_shap = explain(model, 0, background_flat, sample_flat)
    seats_shap = explain(model, 1, background_flat, sample_flat)

    top12 = top12.copy()
    top12["top_drivers_totalFare"] = [top_drivers(r) for r in fare_shap]
    top12["top_drivers_seatsRemaining"] = [top_drivers(r) for r in seats_shap]

    print(top12[["route", "flightDate", "diagnosis", "top_drivers_totalFare"]].to_string(index=False))
    top12.to_csv(PROCESSED_DIR / "queue_explained.csv", index=False)
    print(f"\nSaved to data/processed/queue_explained.csv")
