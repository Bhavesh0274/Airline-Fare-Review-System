"""Phase 9: a genuine classification model -- deliberately not a boosting
family algorithm (Random Forest instead), and deliberately not circular.

The diagnosis label was originally derived by thresholding the neural
network's regression residuals (z_totalFare, z_seatsRemaining). Feeding a
classifier those same two z-scores as features would let it trivially learn
the exact threshold rule -- not a real supervised-learning problem, just a
rule wrapped in a model.

Instead, this predicts the diagnosis label from a flight's RAW observable
state -- route, cabin, airline, days-to-departure, current fare, current
seats-remaining -- withholding the residual/z-score entirely. That's a
genuinely harder, non-circular task: can a classifier learn to flag a likely
mispriced flight directly from its raw characteristics, without first
running the regression? Real accuracy below 100% is expected and reported
honestly, not treated as a failure.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

RAW_FEATURES = ["route", "day_of_week", "days_to_departure", "isNonStop", "isBasicEconomy",
                "isRefundable", "segmentsCabinCode", "segmentsAirlineName", "totalTravelDistance",
                "totalFare", "seatsRemaining"]
CATEGORICAL = ["route", "segmentsCabinCode", "segmentsAirlineName"]
SEED = 42


def prep(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    for c in CATEGORICAL:
        df[c] = df[c].astype("category").cat.codes
    for c in ["isNonStop", "isBasicEconomy", "isRefundable"]:
        df[c] = df[c].astype(int)
    return df


if __name__ == "__main__":
    policy = pd.read_csv(PROCESSED_DIR / "flights_final_policy.csv")
    raw = pd.read_csv(PROCESSED_DIR / "flights_clean.csv",
                       usecols=["legId", "route", "day_of_week", "days_to_departure", "isNonStop",
                                "isBasicEconomy", "isRefundable", "segmentsCabinCode",
                                "segmentsAirlineName", "totalTravelDistance"]).drop_duplicates("legId")

    df = policy[["legId", "totalFare", "seatsRemaining", "diagnosis"]].merge(raw, on="legId", how="left")
    df = prep(df)

    print("Class distribution (what the classifier has to work with):")
    print(df["diagnosis"].value_counts())
    majority_baseline = df["diagnosis"].value_counts(normalize=True).max()
    print(f"\nMajority-class baseline accuracy (always predict 'Normal'): {majority_baseline:.1%}")

    # No natural temporal split here (all 162,343 rows are the same test
    # search-date snapshot) -- a stratified random split is the right choice
    # for this sub-task, unlike the time-based split used for the regression.
    X_train, X_test, y_train, y_test = train_test_split(
        df[RAW_FEATURES], df["diagnosis"], test_size=0.3, random_state=SEED, stratify=df["diagnosis"])

    clf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight="balanced",
                                  random_state=SEED, n_jobs=-1)
    clf.fit(X_train, y_train)
    pred = clf.predict(X_test)

    acc = accuracy_score(y_test, pred)
    macro_f1 = f1_score(y_test, pred, average="macro")

    print(f"\n=== Random Forest diagnosis classifier (raw features only, no residual/z-score input) ===")
    print(f"Accuracy: {acc:.1%} (vs. {majority_baseline:.1%} majority-class baseline)")
    print(f"Macro F1: {macro_f1:.3f} (unweighted across classes -- the honest number given class imbalance)")
    print(f"\nPer-class report:")
    print(classification_report(y_test, pred, zero_division=0))

    print("Confusion matrix (rows=actual, cols=predicted):")
    labels = sorted(y_test.unique())
    cm = confusion_matrix(y_test, pred, labels=labels)
    cm_df = pd.DataFrame(cm, index=[l[:30] for l in labels], columns=[l[:20] for l in labels])
    print(cm_df.to_string())

    importances = pd.Series(clf.feature_importances_, index=RAW_FEATURES).sort_values(ascending=False)
    print(f"\nFeature importances:")
    print(importances)

    out = X_test.copy()
    out["actual_diagnosis"] = y_test.values
    out["predicted_diagnosis"] = pred
    out["legId"] = df.loc[X_test.index, "legId"].values
    out.to_csv(PROCESSED_DIR / "diagnosis_classifier_results.csv", index=False)

    import json
    summary = {
        "accuracy": round(float(acc), 4), "majority_baseline": round(float(majority_baseline), 4),
        "macro_f1": round(float(macro_f1), 4), "n_train": len(X_train), "n_test": len(X_test),
        "feature_importances": importances.round(4).to_dict(),
    }
    with open(PROCESSED_DIR / "diagnosis_classifier_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to data/processed/diagnosis_classifier_results.csv and _summary.json")
