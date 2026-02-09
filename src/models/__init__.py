"""Forecasting models module."""

from .base import (
    BaseForecaster,
    BaseDeepLearningForecaster,
    PredictionResult,
    MultiPeriodPredictionResult,
)

from .traditional import (
    # Naïve baselines
    HistoricalQuantile,
    # Parametric models
    NormalAssumption,
    # Resampling-based methods
    BootstrappedNewsvendor,
    # OR methods
    SampleAverageApproximation,
    TwoStageStochastic,
    # Conformal prediction methods
    ConformalPrediction,
    QuantileRegression,
    # Ensemble methods
    EnsembleBatchPI,
    # Robust optimization
    DistributionallyRobustOptimization,
    # SPO with RF base
    SPORandomForest,
    # Utilities
    ExpectedValue,
    Seer,
)

from .deep_learning import (
    LSTMQuantileRegression,
    LSTMQuantileLossOnly,
    TransformerQuantileRegression,
    DeepEnsemble,
    MCDropoutLSTM,
    TemporalFusionTransformer,
    SPOEndToEnd,
    QuantileLoss,
)

from .multi_period import (
    MultiPeriodForecaster,
    MultiPeriodEnsembleForecaster,
    create_multi_period_forecaster,
)

__all__ = [
    # Base classes
    "BaseForecaster",
    "BaseDeepLearningForecaster",
    "PredictionResult",
    "MultiPeriodPredictionResult",

    # Traditional models (Simple → Advanced)
    "HistoricalQuantile",       # Naïve baseline
    "NormalAssumption",         # Parametric
    "BootstrappedNewsvendor",   # Resampling
    "SampleAverageApproximation",  # Standard OR
    "TwoStageStochastic",       # Scenario optimization
    "ConformalPrediction",      # Distribution-free
    "QuantileRegression",       # Direct quantile
    "EnsembleBatchPI",          # EnbPI+CQR
    "DistributionallyRobustOptimization",  # Wasserstein DRO
    "SPORandomForest",              # SPO with RF base
    "ExpectedValue",
    "Seer",

    # Deep learning models
    "LSTMQuantileRegression",     # LSTM with conformal calibration
    "LSTMQuantileLossOnly",       # LSTM without calibration
    "TransformerQuantileRegression",
    "DeepEnsemble",
    "MCDropoutLSTM",
    "TemporalFusionTransformer",
    "SPOEndToEnd",
    "QuantileLoss",

    # Multi-period forecasting
    "MultiPeriodForecaster",
    "MultiPeriodEnsembleForecaster",
    "create_multi_period_forecaster",
]
