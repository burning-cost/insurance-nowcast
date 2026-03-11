"""
insurance-nowcast: ML-Enhanced EM Nowcasting for Insurance Claims Reporting Delays
===================================================================================

Implements the Wilsens/Antonio/Claeskens (arXiv:2512.07335) ML-EM nowcasting
algorithm, adapted for insurance pricing with an exposure offset.

The core problem: when fitting a frequency GLM on individual claims data, the most
recent 6-24 months of policy experience is partially developed — claims have
occurred but not yet been reported. Applying average completion factors from a
aggregate triangle ignores that reporting delay varies by risk characteristics.
This library estimates covariate-conditioned completion factors.

Quick start
-----------
>>> from insurance_nowcast import ReportingDelayModel, NowcastSimulator
>>> sim = NowcastSimulator(n_occurrence_periods=24, max_delay_periods=12)
>>> df = sim.generate(n_policies=2000, eval_period=23)
>>> model = ReportingDelayModel(max_delay_periods=12, verbose=True)
>>> model.fit(
...     df,
...     occurrence_col='occurrence_period',
...     report_col='report_period',
...     exposure_col='exposure',
...     feature_cols=['age_group', 'risk_score', 'channel'],
...     eval_date=23,
... )
>>> cf = model.predict_completion_factors()
>>> ibnr = model.predict_ibnr()

References
----------
Wilsens, Antonio, Claeskens (2024): arXiv:2512.07335
Verbelen, Antonio, Claeskens, Crevecoeur (2022): Statistical Science 37(3)
"""
from ._model import ReportingDelayModel
from ._simulator import NowcastSimulator
from ._diagnostics import ReportingDelayDiagnostic

__version__ = "0.1.0"
__all__ = [
    "ReportingDelayModel",
    "NowcastSimulator",
    "ReportingDelayDiagnostic",
    "__version__",
]
