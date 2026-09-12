"""Phase 6: real SHAP explanations for the neural network's predictions.

TreeExplainer doesn't apply to a Keras model -- this uses shap.GradientExplainer,
built for differentiable models, on two single-output views of the trained
network (one per head) so each explanation is with respect to one target.
"""
import numpy as np
import pandas as pd
import shap
import tensorflow as tf
from tensorflow import keras
from pathlib import Path

from priority_model import CATEGORICAL, NUMERIC, prep, make_inputs

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"
MODEL_PATH = Path(__file__).resolve().parent / "pricing_model.keras"

FEATURE_LABELS = {
    "route": "this route's typical pattern", "segmentsCabinCode": "cabin/itinerary complexity",
    "segmentsAirlineName": "airline", "day_of_week": "day-of-week pattern",
    "days_to_departure": "how far out this is from departure", "isNonStop": "nonstop vs. connecting",
    "isBasicEconomy": "basic economy fare type", "isRefundable": "refundable fare type",
    "totalTravelDistance": "trip distance",
}
ALL_FEATURE_NAMES = CATEGORICAL + NUMERIC  # order matches how SHAP values are concatenated below


def flat_to_model_inputs(flat: np.ndarray) -> dict:
    """Split a (n, 9) flat feature array -- columns in ALL_FEATURE_NAMES order
    -- back into the model's named multi-input dict."""
    n_cat = len(CATEGORICAL)
    inputs = {f"{c}_input": flat[:, i:i+1].astype("int32") for i, c in enumerate(CATEGORICAL)}
    inputs["numeric_input"] = flat[:, n_cat:].astype("float32")
    return inputs


def explain(model: keras.Model, output_index: int, background_flat: np.ndarray,
            sample_flat: np.ndarray) -> np.ndarray:
    # GradientExplainer failed here: gradients don't flow meaningfully to an
    # integer embedding-index input the way they do to a continuous one (a
    # real architectural mismatch -- confirmed by the "zero-dimensional
    # arrays cannot be concatenated" error, not a fixable off-by-one). A
    # model-agnostic explainer sidesteps it entirely: treat the whole model
    # as a black-box function of one flat feature vector, and let SHAP handle
    # the categorical/numeric split without needing gradients at all.
    def predict_fn(flat: np.ndarray) -> np.ndarray:
        inputs = flat_to_model_inputs(flat)
        preds = model.predict(inputs, verbose=0)
        return preds[output_index].flatten()

    explainer = shap.PermutationExplainer(predict_fn, background_flat, seed=42)
    result = explainer(sample_flat)
    return result.values


def top_drivers(vals_row: np.ndarray, top_n: int = 2) -> str:
    order = np.argsort(-np.abs(vals_row))[:top_n]
    parts = []
    for idx in order:
        fname = ALL_FEATURE_NAMES[idx]
        label = FEATURE_LABELS.get(fname, fname)
        sign = "+" if vals_row[idx] >= 0 else "-"
        parts.append(f"{label} ({sign}{abs(vals_row[idx]):.2f})")
    return "; ".join(parts)


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "flights_clean.csv", parse_dates=["searchDate", "flightDate"])
    df = prep(df)
    train = df[df["searchDate"] == "2022-04-16"].reset_index(drop=True)
    test = df[df["searchDate"] == "2022-04-17"].reset_index(drop=True)

    X_train, num_scaler, distance_fill = make_inputs(train, fit=True)
    X_test, _, _ = make_inputs(test, scaler=num_scaler, distance_fill=distance_fill)

    model = keras.models.load_model(MODEL_PATH)

    # Flat (n, 9) arrays -- categorical codes then scaled numeric, matching
    # ALL_FEATURE_NAMES order -- since the permutation explainer treats the
    # model as a black box over one feature vector, not the multi-input dict.
    def to_flat(inputs: dict) -> np.ndarray:
        cat_cols = [inputs[f"{c}_input"] for c in CATEGORICAL]
        return np.concatenate(cat_cols + [inputs["numeric_input"]], axis=1).astype("float32")

    train_flat = to_flat(X_train)
    test_flat = to_flat(X_test)

    background_flat = train_flat[np.random.RandomState(42).choice(len(train_flat), 30, replace=False)]

    # SHAP over the full 162K test set is expensive for a model-agnostic
    # explainer (each explained row needs multiple full forward passes -- in
    # practice ~3.5s/row here, so 1000 rows x 2 outputs was a ~2 hour run,
    # cut down after timing the first batch); a small representative sample
    # is the same trade-off made for the tree-based version, just forced
    # much smaller for a different underlying reason this time.
    SAMPLE_N = 60
    idx = np.random.RandomState(1).choice(len(test_flat), SAMPLE_N, replace=False)
    sample_flat = test_flat[idx]

    fare_shap = explain(model, 0, background_flat, sample_flat)
    seats_shap = explain(model, 1, background_flat, sample_flat)

    out = test.iloc[idx][["legId", "route", "flightDate", "diagnosis"] if "diagnosis" in test.columns
                          else ["legId", "route", "flightDate"]].copy()
    out["top_drivers_totalFare"] = [top_drivers(r) for r in fare_shap]
    out["top_drivers_seatsRemaining"] = [top_drivers(r) for r in seats_shap]

    print(f"=== Sample explanations ({SAMPLE_N} flights) ===")
    print(out.head(5).to_string(index=False))
    print(f"\nUnique fare-driver explanations: {out['top_drivers_totalFare'].nunique()}")
    print(f"Unique seats-driver explanations: {out['top_drivers_seatsRemaining'].nunique()}")

    out.to_csv(PROCESSED_DIR / "flights_explained.csv", index=False)
    print(f"\nSaved to data/processed/flights_explained.csv")
