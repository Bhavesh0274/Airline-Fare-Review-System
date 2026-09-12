"""Phase 3-4: a properly-controlled expected curve, and the priority score
built directly from it -- via a multi-task neural network with entity
embeddings for the categorical features, not gradient-boosted trees.

Fix from Phase 2: pooling by days_to_departure alone was noisy because it
conflated different routes and weekdays. This model predicts fare and
seats-remaining as a function of route, weekday, days-to-departure, cabin,
nonstop, basic-economy, airline, and distance -- so "expected" is specific to
each flight's actual context. The residual (actual minus expected) is the
anomaly signal.

Architecture: route, cabin, and airline each get a learned embedding -- a
real distributed representation, not just one-hot encoding -- concatenated
with the numeric features into a shared trunk, then split into two heads
(fare, seats-remaining) trained jointly. The point of sharing the trunk is
that each route's embedding has to be useful for predicting BOTH tasks, not
fit to just one -- a route that's genuinely different (e.g. a leisure route
with different weekday patterns) should end up nearby in embedding space,
learned from data rather than hand-engineered.

Evaluation: same discipline as before -- train on the 2022-04-16 search,
score the 2022-04-17 search, a real out-of-sample day, not a random shuffle.
"""
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

CATEGORICAL = ["route", "segmentsCabinCode", "segmentsAirlineName"]
NUMERIC = ["day_of_week", "days_to_departure", "isNonStop", "isBasicEconomy",
           "isRefundable", "totalTravelDistance"]
EMBED_DIMS = {"route": 16, "segmentsCabinCode": 6, "segmentsAirlineName": 10}

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATEGORICAL:
        df[f"{c}_code"] = df[c].astype("category").cat.codes
    for c in ["isNonStop", "isBasicEconomy", "isRefundable"]:
        df[c] = df[c].astype(int)
    return df


def build_model(vocab_sizes: dict, n_numeric: int) -> keras.Model:
    cat_inputs, cat_embeds = [], []
    for c in CATEGORICAL:
        inp = layers.Input(shape=(1,), name=f"{c}_input")
        emb = layers.Embedding(input_dim=vocab_sizes[c] + 1, output_dim=EMBED_DIMS[c],
                                name=f"{c}_embedding")(inp)
        cat_inputs.append(inp)
        cat_embeds.append(layers.Flatten()(emb))

    num_input = layers.Input(shape=(n_numeric,), name="numeric_input")

    # BatchNorm right after the concat: embeddings and pre-scaled numeric
    # features land on very different initial scales, and without
    # renormalizing the combined vector the first Dense/ReLU layer can
    # saturate every unit into the dead (negative, zero-gradient) region on
    # the very first batch -- which is exactly what happened without this
    # (loss frozen at the "always predict the mean" value from epoch 1,
    # prediction std ~1e-9 regardless of input). BatchNorm after every Dense
    # keeps pre-activations centered as training progresses, not just at init.
    x = layers.Concatenate()(cat_embeds + [num_input])
    x = layers.BatchNormalization()(x)
    x = layers.Dense(128, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(64, kernel_initializer="he_normal")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    trunk = layers.Dense(32, activation="relu", kernel_initializer="he_normal", name="shared_trunk")(x)

    fare_out = layers.Dense(1, name="fare_output")(layers.Dense(16, activation="relu")(trunk))
    seats_out = layers.Dense(1, name="seats_output")(layers.Dense(16, activation="relu")(trunk))

    model = keras.Model(inputs=cat_inputs + [num_input], outputs=[fare_out, seats_out])
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.001),
                  loss={"fare_output": "mse", "seats_output": "mse"})
    return model


def make_inputs(df: pd.DataFrame, scaler: StandardScaler = None, distance_fill: float = None, fit: bool = False):
    # totalTravelDistance is missing on ~5% of rows (real gap in the scrape,
    # not a coding bug -- confirmed by checking isna().sum() directly).
    # StandardScaler doesn't raise on NaN here, it just silently carries NaN
    # through the transform, which then poisons every downstream computation
    # once concatenated with the embeddings -- the actual root cause of the
    # model collapsing to a constant output regardless of architecture fixes.
    numeric_df = df[NUMERIC].copy()
    if fit:
        distance_fill = numeric_df["totalTravelDistance"].median()
    numeric_df["totalTravelDistance"] = numeric_df["totalTravelDistance"].fillna(distance_fill)

    numeric_raw = numeric_df.values.astype("float32")
    if fit:
        scaler = StandardScaler().fit(numeric_raw)
    numeric_scaled = scaler.transform(numeric_raw).astype("float32")
    inputs = {f"{c}_input": df[f"{c}_code"].values.reshape(-1, 1).astype("int32") for c in CATEGORICAL}
    inputs["numeric_input"] = numeric_scaled
    return inputs, scaler, distance_fill


def diagnose(row):
    if row["z_seatsRemaining"] < -1.0 and row["z_totalFare"] < -0.5:
        return "Underpriced -- selling fast at a below-peer fare"
    if row["z_seatsRemaining"] < -1.0:
        return "Selling fast despite normal pricing -- possible demand surge"
    if row["z_seatsRemaining"] > 1.0 and row["z_totalFare"] > 0.5:
        return "Overpriced -- selling slow at an above-peer fare"
    if row["z_seatsRemaining"] > 1.0:
        return "Selling slow despite normal pricing -- possible weak demand/competitor pressure"
    return "Normal -- no action needed"


if __name__ == "__main__":
    df = pd.read_csv(PROCESSED_DIR / "flights_clean.csv", parse_dates=["searchDate", "flightDate"])
    df = prep(df)

    train = df[df["searchDate"] == "2022-04-16"].reset_index(drop=True)
    test = df[df["searchDate"] == "2022-04-17"].reset_index(drop=True)
    vocab_sizes = {c: int(df[f"{c}_code"].max()) + 1 for c in CATEGORICAL}

    X_train, num_scaler, distance_fill = make_inputs(train, fit=True)
    X_test, _, _ = make_inputs(test, scaler=num_scaler, distance_fill=distance_fill)

    fare_scaler = StandardScaler().fit(train[["totalFare"]])
    seats_scaler = StandardScaler().fit(train[["seatsRemaining"]])
    y_train = {
        "fare_output": fare_scaler.transform(train[["totalFare"]]).astype("float32"),
        "seats_output": seats_scaler.transform(train[["seatsRemaining"]]).astype("float32"),
    }

    model = build_model(vocab_sizes, n_numeric=len(NUMERIC))
    model.summary()
    model.fit(X_train, y_train, validation_split=0.1, epochs=40, batch_size=512, verbose=2,
              callbacks=[keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True)])

    pred_fare_scaled, pred_seats_scaled = model.predict(X_test, verbose=0)
    print("Prediction spread check -- fare std:", pred_fare_scaled.std(), "seats std:", pred_seats_scaled.std())
    pred_fare = fare_scaler.inverse_transform(pred_fare_scaled).flatten()
    pred_seats = seats_scaler.inverse_transform(pred_seats_scaled).flatten()

    mae_fare = mean_absolute_error(test["totalFare"], pred_fare)
    naive_fare = mean_absolute_error(test["totalFare"], np.full(len(test), train["totalFare"].mean()))
    mae_seats = mean_absolute_error(test["seatsRemaining"], pred_seats)
    naive_seats = mean_absolute_error(test["seatsRemaining"], np.full(len(test), train["seatsRemaining"].mean()))

    print(f"totalFare: model MAE={mae_fare:.2f} vs. flat-mean baseline MAE={naive_fare:.2f} "
          f"({100*(1-mae_fare/naive_fare):.1f}% better)")
    print(f"seatsRemaining: model MAE={mae_seats:.2f} vs. flat-mean baseline MAE={naive_seats:.2f} "
          f"({100*(1-mae_seats/naive_seats):.1f}% better)")

    scored = test[["legId", "route", "flightDate", "days_to_departure", "totalFare", "seatsRemaining"]].copy()
    scored["expected_totalFare"] = pred_fare
    scored["expected_seatsRemaining"] = pred_seats
    scored["residual_totalFare"] = scored["totalFare"] - scored["expected_totalFare"]
    scored["residual_seatsRemaining"] = scored["seatsRemaining"] - scored["expected_seatsRemaining"]
    scored["z_totalFare"] = scored["residual_totalFare"] / scored["residual_totalFare"].std()
    scored["z_seatsRemaining"] = scored["residual_seatsRemaining"] / scored["residual_seatsRemaining"].std()
    scored["priority_score"] = scored["z_seatsRemaining"].abs()
    scored["diagnosis"] = scored.apply(diagnose, axis=1)
    scored = scored.sort_values("priority_score", ascending=False)

    print(f"\n=== Top 10 flights to review first (highest priority score) ===")
    print(scored.head(10)[["route", "flightDate", "days_to_departure", "totalFare",
                            "expected_totalFare", "seatsRemaining", "expected_seatsRemaining",
                            "diagnosis"]].to_string(index=False))

    print(f"\nDiagnosis breakdown across all {len(scored)} scored flights:")
    print(scored["diagnosis"].value_counts())

    scored.to_csv(PROCESSED_DIR / "flights_scored.csv", index=False)
    model.save(PROCESSED_DIR.parent.parent / "src" / "pricing_model.keras")
    print(f"\nSaved to data/processed/flights_scored.csv and src/pricing_model.keras")
