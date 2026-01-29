"""Forecasting models module."""

from .base import BaseForecaster, BaseDeepLearningForecaster, PredictionResult

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

__all__ = [
    # Base classes
    "BaseForecaster",
    "BaseDeepLearningForecaster",
    "PredictionResult",

    # Traditional models (Simple → Advanced)
    "HistoricalQuantile",       # Naïve baseline
    "NormalAssumption",         # Parametric
    "BootstrappedNewsvendor",   # Resampling
    "SampleAverageApproximation",  # Standard OR
    "TwoStageStochastic",       # Scenario optimization
    "ConformalPrediction",      # Distribution-free
    "QuantileRegression",       # Direct quantile
    "EnsembleBatchPI",          # EnbPI+CQR
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
]
