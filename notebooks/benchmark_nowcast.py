# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-nowcast: Benchmark vs Chain-Ladder
# MAGIC
# MAGIC Compares ML-EM nowcasting against the chain-ladder method for estimating
# MAGIC completion factors on synthetic UK motor claims data with known covariate
# MAGIC effects on reporting delay.
# MAGIC
# MAGIC **What this benchmark measures:**
# MAGIC - MAE on known completion factors (we know the DGP)
# MAGIC - Development pattern recovery accuracy
# MAGIC - Segment-level variation: does the model detect that different risk
# MAGIC   profiles report at different speeds?
# MAGIC
# MAGIC **The chain-ladder problem:** It aggregates all claims into a single
# MAGIC triangle and applies uniform development factors. When reporting delay
# MAGIC varies by risk characteristic — which it always does in practice — the
# MAGIC aggregate LDFs are wrong for every segment.

# COMMAND ----------

# MAGIC %pip install insurance-nowcast matplotlib

# COMMAND ----------

# MAGIC %restart_python

# COMMAND ----------

import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings("ignore")

from insurance_nowcast import ReportingDelayModel, NowcastSimulator

print(f"insurance-nowcast version: {__import__('insurance_nowcast').__version__}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Simulate data with known delay structure
# MAGIC
# MAGIC We generate 24 months of UK motor BI claims with strong covariate effects:
# MAGIC - **age_group**: older drivers (age_group=2) report 0.3 units faster on
# MAGIC   the geometric scale — i.e., their rho is higher, delay is shorter
# MAGIC - **risk_score**: high risk scores increase frequency but not delay
# MAGIC - **channel**: neutral effect
# MAGIC
# MAGIC The evaluation date is end of period 23 (all 24 periods observed but
# MAGIC the most recent have incomplete development).

# COMMAND ----------

sim = NowcastSimulator(
    n_occurrence_periods=24,
    max_delay_periods=12,
    base_frequency=0.06,   # 6% motor BI frequency — UK-realistic
    delay_shape="geometric",
    feature_effects={"risk_score": 0.6, "age_group": -0.15},
    delay_feature_effects={"age_group": 0.30},  # older -> faster reporting
    random_state=42,
)

df = sim.generate(n_policies=1500, eval_period=23)

print(f"Dataset: {len(df):,} claim records")
print(f"Occurrence periods: {df['occurrence_period'].nunique()}")
print(f"Reported claims:    {df['report_period'].notna().sum():,}")
print(f"IBNR (unreported):  {df['report_period'].isna().sum():,}")
print(f"\nColumns: {list(df.columns)}")
print(f"\nFirst 5 rows:")
print(df.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. True completion factors from the DGP
# MAGIC
# MAGIC These are what we're trying to estimate. The most recent periods have
# MAGIC the highest completion factors (least developed, most IBNR).

# COMMAND ----------

true_cf = sim.get_true_completion_factors(eval_period=23)
print("True (DGP) completion factors by occurrence period:")
print(true_cf.to_string(index=False))
print(f"\nRange: {true_cf['true_completion_factor'].min():.3f} – {true_cf['true_completion_factor'].max():.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Chain-ladder baseline
# MAGIC
# MAGIC We build a simple paid development triangle and apply volume-weighted
# MAGIC age-to-age factors. This is the standard actuarial IBNR method.
# MAGIC
# MAGIC Note what chain-ladder cannot do: it applies the same development
# MAGIC pattern to all risk segments, ignoring the age_group delay effect.

# COMMAND ----------

def build_chain_ladder_cfs(df, occurrence_col, report_col, eval_period, max_delay):
    """
    Build a paid development triangle and derive chain-ladder completion factors.

    Returns a DataFrame with columns: occurrence_period, cl_completion_factor.
    """
    # Aggregate: count reported claims by (occurrence_period, delay)
    reported = df[df[report_col].notna()].copy()
    reported["delay"] = (reported[report_col] - reported[occurrence_col]).astype(int)
    reported = reported[reported["delay"] >= 0]

    # Build cumulative development triangle
    triangle = pd.crosstab(
        reported[occurrence_col],
        reported["delay"].clip(upper=max_delay - 1),
    )

    # Fill missing delays with 0
    for d in range(max_delay):
        if d not in triangle.columns:
            triangle[d] = 0
    triangle = triangle.sort_index(axis=1)

    # Cumulative triangle
    cum_triangle = triangle.cumsum(axis=1)

    # Age-to-age factors (volume-weighted)
    ldfs = []
    for d in range(max_delay - 1):
        c0 = d
        c1 = d + 1
        if c0 not in cum_triangle.columns or c1 not in cum_triangle.columns:
            ldfs.append(1.0)
            continue
        # Use periods where both columns are observed (i.e., tau_i > d+1)
        periods = cum_triangle.index
        tau_vec = [min(max_delay, eval_period - p + 1) for p in periods]
        mask = [tau > c1 for tau in tau_vec]
        denom = cum_triangle.loc[[p for p, m in zip(periods, mask) if m], c0].sum()
        numer = cum_triangle.loc[[p for p, m in zip(periods, mask) if m], c1].sum()
        ldf = float(numer / denom) if denom > 0 else 1.0
        ldfs.append(ldf)

    # Completion factors: multiply tail factors from truncation depth to max_delay
    records = []
    for period in cum_triangle.index:
        tau = min(max_delay, eval_period - period + 1)
        tau = max(tau, 0)

        # Development fraction = cumulative at tau / ultimate
        # Ultimate = project forward using age-to-age factors
        obs_at_tau = float(cum_triangle.loc[period, min(tau, max_delay - 1)])

        # Project to ultimate by chaining LDFs from tau onwards
        projected_ultimate = obs_at_tau
        for d in range(tau, max_delay - 1):
            if d < len(ldfs):
                projected_ultimate *= ldfs[d]

        total_at_max = float(cum_triangle.loc[period, max_delay - 1]) if max_delay - 1 in cum_triangle.columns else obs_at_tau
        # Use observed at max as denominator if period is fully developed
        if tau >= max_delay:
            cf = 1.0
        elif projected_ultimate > 0:
            cf = float(projected_ultimate / obs_at_tau) if obs_at_tau > 0 else np.nan
        else:
            cf = np.nan

        records.append({
            "occurrence_period": period,
            "cl_completion_factor": cf,
            "obs_at_tau": obs_at_tau,
        })

    return pd.DataFrame(records)


cl_cf = build_chain_ladder_cfs(
    df,
    occurrence_col="occurrence_period",
    report_col="report_period",
    eval_period=23,
    max_delay=12,
)

print("Chain-ladder completion factors:")
print(cl_cf[["occurrence_period", "cl_completion_factor"]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Fit ML-EM models

# COMMAND ----------

# GLM baseline (fast, interpretable)
glm_model = ReportingDelayModel(
    occurrence_model="glm",
    delay_model="glm",
    max_delay_periods=12,
    em_patience=8,
    max_em_iterations=30,
    n_bootstrap=0,
    verbose=False,
    random_state=42,
)

glm_model.fit(
    df,
    occurrence_col="occurrence_period",
    report_col="report_period",
    exposure_col="exposure",
    feature_cols=["age_group", "risk_score", "channel"],
    eval_date=23,
)
print(f"GLM-EM: {glm_model.n_iter_} iterations, converged={glm_model.is_converged_}")

# COMMAND ----------

# XGBoost (the recommended approach)
xgb_model = ReportingDelayModel(
    occurrence_model="xgboost",
    delay_model="xgboost",
    max_delay_periods=12,
    em_patience=8,
    max_em_iterations=30,
    n_bootstrap=0,
    verbose=False,
    random_state=42,
    occurrence_model_kwargs={"n_estimators_per_iter": 30, "max_depth": 3},
    delay_model_kwargs={"n_estimators_per_iter": 30, "max_depth": 3},
)

xgb_model.fit(
    df,
    occurrence_col="occurrence_period",
    report_col="report_period",
    exposure_col="exposure",
    feature_cols=["age_group", "risk_score", "channel"],
    eval_date=23,
)
print(f"XGBoost-EM: {xgb_model.n_iter_} iterations, converged={xgb_model.is_converged_}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Head-to-head: MAE on known completion factors

# COMMAND ----------

glm_cf = glm_model.predict_completion_factors(include_ci=False)
xgb_cf = xgb_model.predict_completion_factors(include_ci=False)

# Merge everything
comparison = (
    true_cf
    .merge(
        cl_cf[["occurrence_period", "cl_completion_factor"]],
        on="occurrence_period", how="left"
    )
    .merge(
        glm_cf[["occurrence_period", "completion_factor"]].rename(
            columns={"completion_factor": "glm_cf"}
        ),
        on="occurrence_period", how="left"
    )
    .merge(
        xgb_cf[["occurrence_period", "completion_factor"]].rename(
            columns={"completion_factor": "xgb_cf"}
        ),
        on="occurrence_period", how="left"
    )
)

# Clip extreme values to avoid one outlier dominating the MAE
true_vals = comparison["true_completion_factor"].values
cl_vals   = np.clip(comparison["cl_completion_factor"].fillna(99).values, 0, 20)
glm_vals  = np.clip(comparison["glm_cf"].fillna(99).values, 0, 20)
xgb_vals  = np.clip(comparison["xgb_cf"].fillna(99).values, 0, 20)

# All periods
mae_cl_all  = float(np.mean(np.abs(cl_vals  - true_vals)))
mae_glm_all = float(np.mean(np.abs(glm_vals - true_vals)))
mae_xgb_all = float(np.mean(np.abs(xgb_vals - true_vals)))

# Last 8 periods only (most affected by IBNR)
last8 = slice(-8, None)
mae_cl_l8  = float(np.mean(np.abs(cl_vals[last8]  - true_vals[last8])))
mae_glm_l8 = float(np.mean(np.abs(glm_vals[last8] - true_vals[last8])))
mae_xgb_l8 = float(np.mean(np.abs(xgb_vals[last8] - true_vals[last8])))

print("Mean absolute error — completion factor estimates:")
print(f"\n{'Method':<20} {'MAE (all)':>12} {'MAE (last 8)':>14}")
print("-" * 48)
print(f"{'Chain-ladder':<20} {mae_cl_all:>12.3f} {mae_cl_l8:>14.3f}")
print(f"{'GLM-EM':<20} {mae_glm_all:>12.3f} {mae_glm_l8:>14.3f}")
print(f"{'XGBoost-EM':<20} {mae_xgb_all:>12.3f} {mae_xgb_l8:>14.3f}")
print()
print(f"XGBoost-EM improvement vs chain-ladder (last 8):")
print(f"  {(1 - mae_xgb_l8/mae_cl_l8)*100:.1f}% lower MAE")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Completion factor comparison table (last 8 periods)
# MAGIC
# MAGIC The story is in the last 8 periods where IBNR is largest.

# COMMAND ----------

comp_display = comparison.copy()
comp_display["cl_error_pct"]  = (cl_vals  - true_vals) / np.maximum(true_vals, 0.01) * 100
comp_display["glm_error_pct"] = (glm_vals - true_vals) / np.maximum(true_vals, 0.01) * 100
comp_display["xgb_error_pct"] = (xgb_vals - true_vals) / np.maximum(true_vals, 0.01) * 100

print("Completion factor comparison — last 8 occurrence periods:")
print(f"\n{'Period':>8} {'True CF':>9} {'CL CF':>9} {'GLM CF':>9} {'XGB CF':>9} {'CL err%':>9} {'XGB err%':>9}")
print("-" * 68)
for _, row in comp_display.tail(8).iterrows():
    print(
        f"{int(row['occurrence_period']):>8} "
        f"{row['true_completion_factor']:>9.3f} "
        f"{cl_vals[int(row.name)]:>9.3f} "
        f"{glm_vals[int(row.name)]:>9.3f} "
        f"{xgb_vals[int(row.name)]:>9.3f} "
        f"{row['cl_error_pct']:>8.1f}% "
        f"{row['xgb_error_pct']:>8.1f}%"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Development pattern recovery
# MAGIC
# MAGIC Chain-ladder produces a single aggregate development pattern.
# MAGIC ML-EM produces segment-specific patterns. Here we compare:
# MAGIC - The aggregate pattern recovered by each method vs the DGP average
# MAGIC - The segment variation: do older vs younger drivers have different patterns?

# COMMAND ----------

# Average delay distribution from DGP
mock_feats_0 = pd.DataFrame({"age_group": [0.0], "risk_score": [0.5], "channel": [0.0]})
mock_feats_2 = pd.DataFrame({"age_group": [2.0], "risk_score": [0.5], "channel": [0.0]})

p_young = xgb_model.predict_development_pattern(mock_feats_0)[0]
p_older = xgb_model.predict_development_pattern(mock_feats_2)[0]

# Compute average delay from DGP
def true_delay_prob(age_group_val, sim_obj):
    """Get true average delay probability for a given age_group."""
    mock = {"age_group": np.array([age_group_val]), "risk_score": np.array([0.5]), "channel": np.array([0.0])}
    return sim_obj._compute_delay_probs(mock, 0)

true_young = true_delay_prob(0.0, sim)
true_older = true_delay_prob(2.0, sim)

print("Reporting delay distribution: young drivers (age_group=0)")
print(f"\n{'Delay':>6} {'True':>8} {'XGB-EM':>8} {'Diff':>8}")
print("-" * 34)
for d in range(min(8, len(true_young))):
    diff = p_young[d] - true_young[d]
    print(f"{d:>6} {true_young[d]:>8.4f} {p_young[d]:>8.4f} {diff:>+8.4f}")

print(f"\nReporting delay distribution: older drivers (age_group=2)")
print(f"\n{'Delay':>6} {'True':>8} {'XGB-EM':>8} {'Diff':>8}")
print("-" * 34)
for d in range(min(8, len(true_older))):
    diff = p_older[d] - true_older[d]
    print(f"{d:>6} {true_older[d]:>8.4f} {p_older[d]:>8.4f} {diff:>+8.4f}")

print(f"\nMedian delay (delay where CDF >= 0.5):")
print(f"  True young:  period {np.searchsorted(np.cumsum(true_young), 0.5)}")
print(f"  True older:  period {np.searchsorted(np.cumsum(true_older), 0.5)}")
print(f"  XGB young:   period {np.searchsorted(np.cumsum(p_young), 0.5)}")
print(f"  XGB older:   period {np.searchsorted(np.cumsum(p_older), 0.5)}")
print(f"\n(Chain-ladder uses a single pattern for all segments — no differentiation)")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Segment-level completion factors
# MAGIC
# MAGIC This is where chain-ladder cannot compete at all: segment-level CFs.
# MAGIC Chain-ladder applies the same LDF to every risk group.

# COMMAND ----------

cf_by_age = xgb_model.predict_completion_factors(
    df=df,
    by=["age_group"],
    include_ci=False,
)

print("Completion factors by occurrence period × age_group (last 6 periods):")
try:
    pivot = cf_by_age.pivot_table(
        values="completion_factor",
        index="occurrence_period",
        columns="age_group",
    ).tail(6).round(3)
    print(pivot.to_string())
    print()
    # Spread: how much do CFs differ across age groups in most recent period?
    last_period = cf_by_age["occurrence_period"].max()
    last_cfs = cf_by_age[cf_by_age["occurrence_period"] == last_period]["completion_factor"]
    spread = last_cfs.max() - last_cfs.min()
    print(f"CF spread across age groups in most recent period: {spread:.3f}")
    print("(Chain-ladder assigns a single CF = this spread is ignored)")
except Exception as e:
    print(f"Pivot failed: {e}")
    print(cf_by_age.tail(12).to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. IBNR estimates

# COMMAND ----------

ibnr = xgb_model.predict_ibnr()
total_observed = ibnr["observed_claims"].sum()
total_ibnr     = ibnr["ibnr_count"].sum()
total_ultimate = ibnr["estimated_ultimate_claims"].sum()

print(f"Total observed claims:           {total_observed:.0f}")
print(f"Total IBNR (estimated):          {total_ibnr:.0f}")
print(f"Total estimated ultimate claims: {total_ultimate:.0f}")
print(f"IBNR as % of ultimate:           {total_ibnr/total_ultimate:.1%}")
print()
print("IBNR by period (last 8):")
print(ibnr.tail(8)[["occurrence_period", "observed_claims", "ibnr_count", "ibnr_fraction"]].to_string(index=False))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Summary

# COMMAND ----------

print("=" * 65)
print("BENCHMARK SUMMARY: ML-EM Nowcasting vs Chain-Ladder")
print("=" * 65)
print()
print("MAE on completion factors (last 8 periods, most affected by IBNR):")
print(f"  Chain-ladder: {mae_cl_l8:.3f}")
print(f"  GLM-EM:       {mae_glm_l8:.3f}")
print(f"  XGBoost-EM:   {mae_xgb_l8:.3f}")
print(f"  XGBoost-EM improvement vs CL: {(1 - mae_xgb_l8/mae_cl_l8)*100:.1f}%")
print()
print("Development pattern differentiation:")
print(f"  Chain-ladder: single aggregate pattern for all segments")
print(f"  XGBoost-EM:   segment-specific patterns — detects age_group delay effect")
print(f"  Median delay: young={np.searchsorted(np.cumsum(p_young), 0.5)}, "
      f"older={np.searchsorted(np.cumsum(p_older), 0.5)} (XGB estimate)")
print()
print("IBNR loading:")
print(f"  Total IBNR estimated: {total_ibnr:.0f} claims ({total_ibnr/total_ultimate:.1%} of ultimate)")
print()
print("Limitations:")
print("  - On small portfolios (<500 claims), XGBoost may overfit; prefer GLM")
print("  - Chain-ladder is unbiased when all segments have the same delay — ML-EM")
print("    pays off only when delay is covariate-dependent")
print("  - EM convergence requires reasonable initialisation; very recent periods")
print("    (tau < 3) have high uncertainty regardless of method")

assert mae_xgb_l8 < mae_cl_l8, "XGBoost-EM should outperform chain-ladder on last 8 periods"
print("\nAssertion passed: XGBoost-EM beats chain-ladder on last-8-period MAE.")
