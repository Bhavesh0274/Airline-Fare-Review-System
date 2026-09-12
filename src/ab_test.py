"""Phase 8: A/B test the fare policy itself -- not just build a rule for
"what fare to charge," but validate whether *changing* the fare actually
beats doing nothing.

Honest framing: no live pricing experiment ran on real customers here (this
is a historical scrape, not an experiment), so there's no real "did revenue
go up" outcome to read off. What IS real: the randomization, the hypothesis
test, the power analysis, and the flights/fares themselves. What's stated:
a demand-response (elasticity) model standing in for the unobserved outcome,
using a value from the middle of published airline demand-elasticity
estimates (commonly cited in the 1.0-1.5 range) rather than an invented
number. This mirrors the same real/stated split used everywhere else in
this project -- the methodology is real and reusable against actual booking
outcomes the moment a real experiment produces them; only the response curve
underneath it is a placeholder.
"""
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import stats
from statsmodels.stats.power import TTestIndPower

PROCESSED_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

ELASTICITY = 1.3  # stated assumption, middle of commonly cited airline demand-elasticity literature (~1.0-1.5)
SEED = 42
ALPHA = 0.05


def simulate_sell_probability(baseline_p: np.ndarray, fare_ratio: np.ndarray) -> np.ndarray:
    """Iso-elastic demand response: P(sell) scales as (new_fare/old_fare)^-elasticity.
    A stated model, not fitted -- there's no real observed outcome to fit it to."""
    return np.clip(baseline_p * np.power(fare_ratio, -ELASTICITY), 0, 1)


if __name__ == "__main__":
    policy = pd.read_csv(PROCESSED_DIR / "flights_final_policy.csv")
    # flights_clean.csv has one row per (legId, searchDate) -- 76,517 legIds
    # appear on both real search dates (the Phase 2 longitudinal pairs), so
    # merging on legId alone fans out every one of those into 2 rows. Dedup
    # first: isBasicEconomy is a property of the fare product, identical
    # across a legId's two observations, so this loses no information.
    raw = pd.read_csv(PROCESSED_DIR / "flights_clean.csv", usecols=["legId", "isBasicEconomy"]).drop_duplicates("legId")
    policy = policy.merge(raw, on="legId", how="left")
    assert len(policy) == policy["legId"].nunique(), "merge fan-out still present"

    # Only flagged flights are eligible -- there's no proposed treatment for
    # a flight the policy already says needs no action.
    eligible = policy[policy["recommended_change_pct"] != 0].reset_index(drop=True)
    print(f"Eligible population (flagged flights): {len(eligible)} of {len(policy)} total")

    rng = np.random.RandomState(SEED)
    eligible["arm"] = rng.choice(["treatment", "control"], size=len(eligible), p=[0.5, 0.5])

    eligible["new_fare"] = np.where(eligible["arm"] == "treatment",
                                     eligible["recommended_fare"], eligible["totalFare"])
    eligible["fare_ratio"] = eligible["new_fare"] / eligible["totalFare"]
    eligible["sim_sell_prob"] = simulate_sell_probability(eligible["estimated_load_factor"].values,
                                                            eligible["fare_ratio"].values)
    eligible["sim_expected_revenue"] = eligible["new_fare"] * eligible["sim_sell_prob"]

    treat = eligible[eligible["arm"] == "treatment"]["sim_expected_revenue"]
    ctrl = eligible[eligible["arm"] == "control"]["sim_expected_revenue"]

    print(f"\nTreatment n={len(treat)}, mean sim. revenue=${treat.mean():.2f}")
    print(f"Control n={len(ctrl)}, mean sim. revenue=${ctrl.mean():.2f}")

    t_stat, p_value = stats.ttest_ind(treat, ctrl, equal_var=False)
    pooled_std = np.sqrt((treat.var() + ctrl.var()) / 2)
    cohens_d = (treat.mean() - ctrl.mean()) / pooled_std
    diff = treat.mean() - ctrl.mean()
    se_diff = np.sqrt(treat.var()/len(treat) + ctrl.var()/len(ctrl))
    ci_low, ci_high = diff - 1.96*se_diff, diff + 1.96*se_diff

    print(f"\n=== Welch's t-test ===")
    print(f"Mean lift: ${diff:.2f}/flight ({100*diff/ctrl.mean():.1f}%), 95% CI [${ci_low:.2f}, ${ci_high:.2f}]")
    print(f"t={t_stat:.2f}, p={p_value:.2e}, Cohen's d={cohens_d:.3f}")
    print(f"Result: {'statistically significant' if p_value < ALPHA else 'not significant'} at alpha={ALPHA}")

    power_analysis = TTestIndPower()
    achieved_power = power_analysis.power(effect_size=cohens_d, nobs1=len(treat), ratio=len(ctrl)/len(treat), alpha=ALPHA)
    mde = power_analysis.solve_power(nobs1=len(treat), ratio=len(ctrl)/len(treat), alpha=ALPHA, power=0.8)

    print(f"\n=== Power analysis ===")
    print(f"Achieved power at observed effect size: {achieved_power:.4f}")
    print(f"Minimum detectable effect size (Cohen's d) at 80% power, this sample size: {mde:.4f}")

    print(f"\n=== Guardrail: is the treatment disproportionately raising basic-economy fares? ===")
    guardrail = eligible[eligible["arm"] == "treatment"].groupby("isBasicEconomy")["recommended_change_pct"].agg(
        ["mean", "count"])
    print(guardrail)

    eligible.to_csv(PROCESSED_DIR / "ab_test_results.csv", index=False)
    summary = {
        "n_treatment": int(len(treat)), "n_control": int(len(ctrl)),
        "mean_revenue_treatment": round(float(treat.mean()), 2), "mean_revenue_control": round(float(ctrl.mean()), 2),
        "lift_pct": round(float(100*diff/ctrl.mean()), 2), "ci_low": round(float(ci_low), 2), "ci_high": round(float(ci_high), 2),
        "p_value": float(p_value), "cohens_d": round(float(cohens_d), 4),
        "achieved_power": round(float(achieved_power), 4), "mde_cohens_d": round(float(mde), 4),
        "elasticity_assumption": ELASTICITY,
    }
    import json
    with open(PROCESSED_DIR / "ab_test_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved to data/processed/ab_test_results.csv and ab_test_summary.json")
