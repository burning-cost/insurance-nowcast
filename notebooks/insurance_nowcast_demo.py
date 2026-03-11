# Databricks notebook source
# MAGIC %md
# MAGIC # insurance-nowcast: ML-Enhanced EM Nowcasting Demo
# MAGIC
# MAGIC This notebook demonstrates the full workflow for using `insurance-nowcast`
# MAGIC to estimate covariate-conditioned completion factors and IBNR counts from
# MAGIC individual claims data.
# MAGIC
# MAGIC **What we cover:**
# MAGIC 1. Generate synthetic UK motor BI claims data with known delay structure
# MAGIC 2. Fit the ML-EM model (GLM baseline and XGBoost)
# MAGIC 3. Compare estimated completion factors to the known true values
# MAGIC 4. Produce segment-level completion factors for pricing GLM use
# MAGIC 5. Visualise convergence and development patterns
# MAGIC
# MAGIC **Audience:** Pricing actuaries evaluating whether to use this library
# MAGIC in a frequency GLM review.

# COMMAND ----------
# MAGIC %pip install insurance-nowcast matplotlib

# COMMAND ----------
# MAGIC %restart_python

# COMMAND ----------
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from insurance_nowcast import ReportingDelayModel, NowcastSimulator, ReportingDelayDiagnostic

print(f"insurance-nowcast version: {__import__('insurance_nowcast').__version__}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 1. Simulate UK-style motor BI claims data
# MAGIC
# MAGIC We generate 24 months of occurrence data with:
# MAGIC - 1,000 policies per month
# MAGIC - Base frequency 6% (typical motor BI)
# MAGIC - Geometric delay distribution (median ~4 months, tail to 12 months)
# MAGIC - Feature effects: older drivers claim less (age_group effect), higher risk
# MAGIC   scores claim more
# MAGIC - Evaluation date at end of period 23 (month 24)

# COMMAND ----------
sim = NowcastSimulator(
    n_occurrence_periods=24,
    max_delay_periods=12,
    base_frequency=0.06,
    delay_shape="geometric",
    feature_effects={"risk_score": 0.5, "age_group": -0.2},
    delay_feature_effects={"age_group": 0.3},  # older drivers report faster
    random_state=42,
)

df = sim.generate(n_policies=1000, eval_period=23)
print(f"Dataset: {len(df):,} claim records")
print(f"Occurrence periods: {df['occurrence_period'].nunique()}")
print(f"Reported claims: {df['report_period'].notna().sum():,}")
print(f"IBNR (unreported) claims: {df['report_period'].isna().sum():,}")
print(f"\nFirst few rows:")
display(df.head(10))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 2. True completion factors from the DGP
# MAGIC
# MAGIC Since we generated the data, we know the true completion factors.
# MAGIC We'll use these to evaluate how well the model recovers them.

# COMMAND ----------
true_cf = sim.get_true_completion_factors(eval_period=23)
print("True completion factors (DGP):")
display(true_cf)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 3. Fit GLM baseline model

# COMMAND ----------
glm_model = ReportingDelayModel(
    occurrence_model="glm",
    delay_model="glm",
    max_delay_periods=12,
    em_patience=8,
    max_em_iterations=30,
    n_bootstrap=0,  # Skip bootstrap for speed
    verbose=True,
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

print(f"\nGLM model: {glm_model.n_iter_} iterations, "
      f"converged={glm_model.is_converged_}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 4. Fit XGBoost model (the recommended approach)

# COMMAND ----------
xgb_model = ReportingDelayModel(
    occurrence_model="xgboost",
    delay_model="xgboost",
    max_delay_periods=12,
    em_patience=8,
    max_em_iterations=30,
    n_bootstrap=0,
    verbose=True,
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

print(f"\nXGBoost model: {xgb_model.n_iter_} iterations, "
      f"converged={xgb_model.is_converged_}")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 5. Compare estimated vs true completion factors

# COMMAND ----------
glm_cf = glm_model.predict_completion_factors(include_ci=False)
xgb_cf = xgb_model.predict_completion_factors(include_ci=False)

comparison = true_cf.merge(
    glm_cf[["occurrence_period", "completion_factor"]].rename(
        columns={"completion_factor": "glm_cf"}
    ),
    on="occurrence_period"
).merge(
    xgb_cf[["occurrence_period", "completion_factor"]].rename(
        columns={"completion_factor": "xgb_cf"}
    ),
    on="occurrence_period"
)

# Compute errors
comparison["glm_error_pct"] = (
    (comparison["glm_cf"] - comparison["true_completion_factor"]) /
    comparison["true_completion_factor"] * 100
)
comparison["xgb_error_pct"] = (
    (comparison["xgb_cf"] - comparison["true_completion_factor"]) /
    comparison["true_completion_factor"] * 100
)

print("Completion factor comparison — last 8 periods (most affected by IBNR):")
display(comparison.tail(8)[
    ["occurrence_period", "true_completion_factor", "glm_cf", "xgb_cf",
     "glm_error_pct", "xgb_error_pct"]
].round(3))

print(f"\nMean absolute error (last 8 periods):")
print(f"  GLM:     {comparison.tail(8)['glm_error_pct'].abs().mean():.1f}%")
print(f"  XGBoost: {comparison.tail(8)['xgb_error_pct'].abs().mean():.1f}%")

# COMMAND ----------
# MAGIC %md
# MAGIC ## 6. Plot: estimated vs true completion factors

# COMMAND ----------
fig, ax = plt.subplots(figsize=(12, 5))

ax.plot(comparison["occurrence_period"], comparison["true_completion_factor"],
        "k-o", label="True (DGP)", linewidth=2, markersize=6)
ax.plot(comparison["occurrence_period"], comparison["glm_cf"],
        "b--s", label="GLM-EM", linewidth=1.5, markersize=5, alpha=0.8)
ax.plot(comparison["occurrence_period"], comparison["xgb_cf"],
        "r-^", label="XGBoost-EM", linewidth=1.5, markersize=5, alpha=0.8)

ax.set_xlabel("Occurrence Period")
ax.set_ylabel("Completion Factor")
ax.set_title("Estimated vs True Completion Factors — Motor BI Example")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
display(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 7. Convergence plots

# COMMAND ----------
fig, axes = plt.subplots(1, 2, figsize=(14, 4))

diag = ReportingDelayDiagnostic()
diag.plot_convergence(glm_model, ax=axes[0])
axes[0].set_title("GLM-EM Convergence")

diag.plot_convergence(xgb_model, ax=axes[1])
axes[1].set_title("XGBoost-EM Convergence")

plt.tight_layout()
display(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 8. Development pattern plot
# MAGIC
# MAGIC Shows how reporting accumulates across delay periods for each occurrence
# MAGIC period. Recent periods (right side) have few observed data points and
# MAGIC rely more heavily on model predictions.

# COMMAND ----------
fig = diag.plot_development_pattern(
    xgb_model,
    figsize=(12, 5),
)
display(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 9. IBNR by occurrence period
# MAGIC
# MAGIC Total IBNR loading by period. Recent periods dominate — as expected,
# MAGIC since they have fewer months of observation.

# COMMAND ----------
ibnr = xgb_model.predict_ibnr()
print(f"Total estimated IBNR: {ibnr['ibnr_count'].sum():.1f} claims")
print(f"Total observed claims: {ibnr['observed_claims'].sum():.1f}")
print(f"Total estimated ultimate: {ibnr['estimated_ultimate_claims'].sum():.1f}")

display(ibnr.tail(8))

# COMMAND ----------
# MAGIC %md
# MAGIC ## 10. Segment-level completion factors
# MAGIC
# MAGIC The pricing actuary's primary output: completion factors broken down by
# MAGIC risk segment. Use these to adjust observed claim counts in your GLM
# MAGIC training data, rather than applying a single aggregate factor.

# COMMAND ----------
cf_by_channel = xgb_model.predict_completion_factors(
    df=df,
    by=["channel"],
    include_ci=False,
)
print("Completion factors by occurrence period × channel:")
display(
    cf_by_channel.pivot_table(
        values="completion_factor",
        index="occurrence_period",
        columns="channel",
    ).tail(8).round(3)
)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 11. Delay distribution by risk profile
# MAGIC
# MAGIC Compare delay distributions for different risk profiles to understand
# MAGIC how reporting speed varies across segments.

# COMMAND ----------
# Compare: young (age_group=0) vs older (age_group=2), low vs high risk
profiles = pd.DataFrame({
    "age_group": [0.0, 0.0, 2.0, 2.0],
    "risk_score": [0.2, 0.8, 0.2, 0.8],
    "channel": [0.0, 0.0, 0.0, 0.0],
})
labels = [
    "Young, low risk",
    "Young, high risk",
    "Older, low risk",
    "Older, high risk",
]

fig = diag.plot_delay_distribution(xgb_model, profiles, labels=labels)
display(fig)

# COMMAND ----------
# MAGIC %md
# MAGIC ## 12. How to use completion factors in a pricing GLM
# MAGIC
# MAGIC The workflow is:
# MAGIC 1. Fit the nowcast model on individual claims data
# MAGIC 2. Get completion factors by occurrence period (and segment if needed)
# MAGIC 3. Merge completion factors onto your GLM training data
# MAGIC 4. Multiply observed claims by CF to get 'as-if ultimate' claims
# MAGIC 5. Fit GLM on adjusted claim counts

# COMMAND ----------
# Example: adjust observed claims for use in a pricing GLM
cf = xgb_model.predict_completion_factors(include_ci=False)

# Merge onto your training data
df_glm = df.merge(
    cf[["occurrence_period", "completion_factor"]],
    on="occurrence_period",
    how="left",
)

# Adjusted claim count = observed * completion_factor
# (in practice, you'd do this at portfolio/period level)
print("Completion factor lookup for GLM adjustment:")
display(cf[["occurrence_period", "completion_factor", "ibnr_count"]].round(3))

# COMMAND ----------
# MAGIC %md
# MAGIC ## Summary
# MAGIC
# MAGIC This notebook demonstrated:
# MAGIC - `NowcastSimulator`: generates synthetic claims data with known delay structure
# MAGIC - `ReportingDelayModel`: fits ML-EM nowcasting model
# MAGIC   - GLM baseline: fast, interpretable, good for small portfolios
# MAGIC   - XGBoost: better for large portfolios with non-linear effects
# MAGIC - `predict_completion_factors()`: period-level and segment-level CFs
# MAGIC - `predict_ibnr()`: IBNR loading by period
# MAGIC - `ReportingDelayDiagnostic`: convergence and development pattern plots
# MAGIC
# MAGIC Next steps for production use:
# MAGIC - Replace synthetic data with your actual claims data
# MAGIC - Set `max_delay_periods` appropriate to your line of business
# MAGIC - Enable `n_bootstrap=100` for confidence intervals on CFs
# MAGIC - Compare segment-level CFs against the reserving team's triangle LDFs
