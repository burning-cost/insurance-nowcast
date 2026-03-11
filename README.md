# insurance-nowcast

**ML-Enhanced EM Nowcasting for insurance claims reporting delays.**

Pricing actuaries routinely face a problem that has no good Python solution: the most recent 6–24 months of experience data is partially developed — claims have occurred but not yet been reported. Applying aggregate completion factors from a reserving triangle ignores that reporting delay varies by risk characteristics. A young driver making a motor BI claim has a different reporting delay than a fleet driver making a motor PD claim.

This library implements the Wilsens/Antonio/Claeskens (arXiv:2512.07335) ML-EM algorithm, adapted for insurance pricing, to produce **covariate-conditioned completion factors and IBNR counts by risk segment**.

## The problem in concrete terms

You're fitting a frequency GLM on 3 years of motor BI data. Your training data extract is as of 31 December 2024. Policies from Q4 2024 have been exposed for 1–3 months — but motor BI claims have a median reporting delay of 4 months. This means roughly 50–60% of claims from Q4 2024 are still unreported. If you feed raw claim counts into your GLM, Q4 2024 will appear to be a low-frequency quarter, and your model will be biased.

Standard practice is to apply aggregate development factors from the reserving team's triangle. This is better than nothing, but:

- The factors come from aggregate data and don't condition on risk mix
- If your recent business has a different risk profile than historical average, the aggregate factor is wrong
- You can't quantify the uncertainty in the completion factor

This library solves all three problems.

## Install

```bash
pip install insurance-nowcast
```

For diagnostic plots:

```bash
pip install "insurance-nowcast[plots]"
```

## Quick start

```python
from insurance_nowcast import ReportingDelayModel, NowcastSimulator

# Generate synthetic data to test
sim = NowcastSimulator(
    n_occurrence_periods=24,
    max_delay_periods=12,
    base_frequency=0.08,
    delay_shape="geometric",
)
df = sim.generate(n_policies=2000, eval_period=23)

# Fit the model
model = ReportingDelayModel(
    occurrence_model="xgboost",
    delay_model="xgboost",
    max_delay_periods=12,
    verbose=True,
)
model.fit(
    df,
    occurrence_col="occurrence_period",
    report_col="report_period",
    exposure_col="exposure",
    feature_cols=["age_group", "risk_score", "channel"],
    eval_date=23,
)

# Get completion factors by occurrence period
cf = model.predict_completion_factors()
print(cf[["occurrence_period", "completion_factor", "ibnr_count"]])

# Get IBNR counts
ibnr = model.predict_ibnr()
print(f"Total IBNR: {ibnr['ibnr_count'].sum():.1f} claims")

# Segment-level completion factors (for GLM adjustment)
cf_by_channel = model.predict_completion_factors(df=df, by=["channel"])
```

## Input data format

The model expects individual claims data with one row per claim event:

| Column | Type | Description |
|--------|------|-------------|
| `occurrence_period` | int | Period when the claim occurred (e.g., month as integer) |
| `report_period` | float/int, nullable | Period when claim was reported. Null = IBNR |
| `exposure` | float | Policy exposure for this claim (policy-years at risk) |
| feature columns | float/int | Risk covariates — must be numeric |

This mirrors individual claims data that pricing teams already maintain. No triangle aggregation required.

## Algorithm

The model implements the EM algorithm from Wilsens, Antonio & Claeskens (arXiv:2512.07335):

**Joint Poisson-Multinomial model:**
- Occurrence: N_i ~ Poisson(λ(xᵢ) × exposure_i)
- Delay: N_{i,j} | N_i ~ Multinomial(p_j(xᵢ))

**E-step:** For censored periods (j ≥ τᵢ), impute:
`N̂_{i,j}^{(k)} = λ̂^{(k-1)}(xᵢ) × p̂_j^{(k-1)}(xᵢ)`

**M-step:** Fit XGBoost (or GLM) on imputed complete data for:
- Occurrence: Poisson regression with exposure offset
- Delay: Multinomial softmax regression

**XGBoost additive construction:** New trees are added to the previous model at each EM iteration rather than refitting from scratch. This is the key contribution of the Wilsens paper — it provides de facto monotone likelihood improvement structurally similar to classical EM's guarantee.

**Insurance adaptation:** The original paper has no exposure offset. This library adds `log(exposure)` as an offset in the Poisson occurrence model via XGBoost's `base_margin` parameter. This is essential for pricing use — without it, the occurrence model conflates claim frequency rate with exposure volume.

## Model parameters

```python
ReportingDelayModel(
    occurrence_model="xgboost",    # "glm" or "xgboost"
    delay_model="xgboost",         # "glm" or "xgboost"
    max_delay_periods=24,          # Set to 95th-99th percentile of observed delays
    exposure_offset=True,          # Always True for pricing use
    em_patience=10,                # Stop if LL doesn't improve for 10 iterations
    max_em_iterations=50,          # Hard upper limit
    convergence_tol=1e-4,          # Minimum LL improvement to reset patience
    n_bootstrap=100,               # Bootstrap replications for CIs; 0 to skip
    bootstrap_confidence=0.90,     # 90% CI by default
)
```

### Choosing max_delay_periods

This is the most important parameter. Set it too small and you'll understate IBNR. Typical values by UK line:

| Line | Suggested max_delay_periods |
|------|-----------------------------|
| Motor property damage | 6 months |
| Motor bodily injury | 18–24 months |
| Employers' liability | 36–48 months |
| Public liability | 24–36 months |
| Professional indemnity | 36–60 months |

The model will warn if >10% of observed delays are at or beyond the boundary.

### When to use GLM vs XGBoost

Use `occurrence_model="glm", delay_model="glm"` when:
- Portfolio is small (<5,000 claims)
- Interpretability is important
- You want a baseline to compare against

Use `occurrence_model="xgboost", delay_model="xgboost"` when:
- Portfolio is large (>10,000 claims)
- You expect non-linear effects on delay speed (e.g., claim type × territory)
- Per the Wilsens paper experiments, XGBoost outperforms GLM on non-linear data

## Diagnostics

```python
from insurance_nowcast import ReportingDelayDiagnostic

diag = ReportingDelayDiagnostic()
diag.plot_convergence(model)             # EM log-likelihood by iteration
diag.plot_development_pattern(model)    # Cumulative delay curves by period
diag.plot_ibnr_by_period(model)         # Observed vs IBNR bar chart
diag.plot_delay_distribution(model, X)  # Delay PMF by risk profile
```

## What this is not

This is a **pricing tool**, not a **reserving tool**. The outputs are:
- Completion factors for adjusting claim counts in a pricing GLM training dataset
- IBNR counts for understanding development loading by segment

The numbers should be comparable to the reserving team's LDFs. If they diverge materially, that's worth investigating — but don't present these as financial reserves.

The model handles IBNR (unreported claims) only, not RBNS (reported but not settled). For pricing frequency models, this is sufficient — we need ultimate claim *counts*, not ultimate *paid amounts*.

## References

- Wilsens, Antonio, Claeskens (2024): [arXiv:2512.07335](https://arxiv.org/abs/2512.07335) — the ML-EM framework this implements
- Verbelen, Antonio, Claeskens, Crevecoeur (2022): [Statistical Science 37(3)](https://projecteuclid.org/journals/statistical-science/volume-37/issue-3/Modeling-the-Occurrence-of-Events-Subject-to-a-Reporting-Delay/10.1214/21-STS831.short) — the foundational GLM-EM paper
- Hiabu, Hofman, Pittarello (2023): [arXiv:2312.14549](https://arxiv.org/abs/2312.14549) — parallel survival analysis approach (R package: ReSurv)

## Development

```bash
git clone https://github.com/burning-cost/insurance-nowcast
cd insurance-nowcast
uv sync --all-extras
uv run pytest tests/ -v
```
