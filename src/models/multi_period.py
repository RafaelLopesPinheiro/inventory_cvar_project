"""
Multi-period forecasting wrappers for direct multi-step-ahead prediction.

This module provides wrapper classes that enable any single-horizon forecaster
to produce multi-period predictions using the "direct" strategy, where separate
models are trained for each forecast horizon.

References:
-----------
- Taieb et al. (2012) "A review and comparison of strategies for multi-step
  ahead time series forecasting"
- Hyndman & Athanasopoulos (2021) "Forecasting: Principles and Practice" Ch. 13
"""

import numpy as np
from typing import Dict, List, Type, Optional, Any
from dataclasses import dataclass
import logging
import copy

from .base import BaseForecaster, PredictionResult, MultiPeriodPredictionResult

logger = logging.getLogger(__name__)


class MultiPeriodForecaster:
    """
    Wrapper to enable multi-period forecasting using the direct strategy.

    The direct strategy trains a separate model for each forecast horizon.
    This is the most common approach for multi-step-ahead forecasting and
    generally provides the best accuracy across horizons.

    Attributes
    ----------
    base_model_class : Type[BaseForecaster]
        The base forecaster class to use for each horizon.
    horizons : List[int]
        List of forecast horizons (days ahead).
    models : Dict[int, BaseForecaster]
        Dictionary mapping each horizon to its trained model.
    model_kwargs : Dict[str, Any]
        Keyword arguments to pass to the base model constructor.

    Examples
    --------
    >>> from src.models import ConformalPrediction
    >>> mp_forecaster = MultiPeriodForecaster(
    ...     base_model_class=ConformalPrediction,
    ...     horizons=[1, 7, 14, 21, 28],
    ...     alpha=0.05
    ... )
    >>> mp_forecaster.fit(X_train, y_horizons_train, X_cal, y_horizons_cal)
    >>> mp_result = mp_forecaster.predict(X_test)
    >>> mp_result.get_horizon(7)  # Get 7-day ahead predictions
    """

    def __init__(
        self,
        base_model_class: Type[BaseForecaster],
        horizons: List[int],
        **model_kwargs
    ):
        """
        Initialize the multi-period forecaster.

        Parameters
        ----------
        base_model_class : Type[BaseForecaster]
            The base forecaster class to use.
        horizons : List[int]
            List of forecast horizons (days ahead) to predict.
        **model_kwargs
            Keyword arguments passed to base model constructor.
        """
        self.base_model_class = base_model_class
        self.horizons = sorted(horizons)
        self.model_kwargs = model_kwargs
        self.models: Dict[int, BaseForecaster] = {}
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Dict[int, np.ndarray],
        X_cal: np.ndarray,
        y_cal: Dict[int, np.ndarray]
    ) -> "MultiPeriodForecaster":
        """
        Fit separate models for each forecast horizon.

        Parameters
        ----------
        X_train : np.ndarray
            Training features of shape (n_train_samples, n_features).
        y_train : Dict[int, np.ndarray]
            Training targets for each horizon.
        X_cal : np.ndarray
            Calibration features of shape (n_cal_samples, n_features).
        y_cal : Dict[int, np.ndarray]
            Calibration targets for each horizon.

        Returns
        -------
        self
            The fitted multi-period forecaster.
        """
        logger.info(f"Fitting MultiPeriodForecaster with horizons: {self.horizons}")

        for horizon in self.horizons:
            logger.info(f"  Training model for horizon h={horizon}")

            # Create and fit model for this horizon
            model = self.base_model_class(**self.model_kwargs)
            model.fit(X_train, y_train[horizon], X_cal, y_cal[horizon])
            self.models[horizon] = model

        self._is_fitted = True
        logger.info(f"Fitted {len(self.models)} horizon models")
        return self

    def predict(self, X: np.ndarray) -> MultiPeriodPredictionResult:
        """
        Generate predictions for all horizons.

        Parameters
        ----------
        X : np.ndarray
            Features for prediction of shape (n_samples, n_features).

        Returns
        -------
        MultiPeriodPredictionResult
            Container with predictions for all horizons.

        Raises
        ------
        RuntimeError
            If the forecaster has not been fitted.
        """
        if not self._is_fitted:
            raise RuntimeError("MultiPeriodForecaster is not fitted. Call fit() first.")

        predictions = {}

        for horizon in self.horizons:
            pred = self.models[horizon].predict(X)
            predictions[horizon] = pred

        return MultiPeriodPredictionResult(
            horizons=self.horizons,
            predictions=predictions
        )

    def get_model(self, horizon: int) -> BaseForecaster:
        """
        Get the trained model for a specific horizon.

        Parameters
        ----------
        horizon : int
            The forecast horizon.

        Returns
        -------
        BaseForecaster
            The trained model for this horizon.
        """
        if horizon not in self.models:
            raise KeyError(f"No model for horizon {horizon}. Available: {self.horizons}")
        return self.models[horizon]

    @property
    def is_fitted(self) -> bool:
        """Check if the forecaster has been fitted."""
        return self._is_fitted


class MultiPeriodEnsembleForecaster:
    """
    Multi-period forecaster using ensemble of models for robustness.

    Trains multiple models (possibly of different types) for each horizon
    and combines their predictions.

    This provides additional robustness by:
    1. Reducing variance through ensemble averaging
    2. Combining different model strengths
    3. Producing more calibrated prediction intervals
    """

    def __init__(
        self,
        model_configs: List[Dict[str, Any]],
        horizons: List[int],
        ensemble_method: str = "mean"
    ):
        """
        Initialize the multi-period ensemble forecaster.

        Parameters
        ----------
        model_configs : List[Dict[str, Any]]
            List of model configurations. Each dict should have:
            - 'class': The model class
            - 'kwargs': Keyword arguments for the model
            - 'weight': Optional weight for ensemble (default 1.0)
        horizons : List[int]
            List of forecast horizons.
        ensemble_method : str
            How to combine predictions: "mean", "median", "weighted".
        """
        self.model_configs = model_configs
        self.horizons = sorted(horizons)
        self.ensemble_method = ensemble_method
        self.models: Dict[int, List[BaseForecaster]] = {}
        self.weights = [cfg.get('weight', 1.0) for cfg in model_configs]
        self._is_fitted = False

    def fit(
        self,
        X_train: np.ndarray,
        y_train: Dict[int, np.ndarray],
        X_cal: np.ndarray,
        y_cal: Dict[int, np.ndarray]
    ) -> "MultiPeriodEnsembleForecaster":
        """
        Fit ensemble of models for each horizon.

        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : Dict[int, np.ndarray]
            Training targets for each horizon.
        X_cal : np.ndarray
            Calibration features.
        y_cal : Dict[int, np.ndarray]
            Calibration targets for each horizon.

        Returns
        -------
        self
        """
        logger.info(f"Fitting MultiPeriodEnsembleForecaster with {len(self.model_configs)} models")

        for horizon in self.horizons:
            logger.info(f"  Training ensemble for horizon h={horizon}")
            horizon_models = []

            for config in self.model_configs:
                model_class = config['class']
                model_kwargs = config.get('kwargs', {})

                model = model_class(**model_kwargs)
                model.fit(X_train, y_train[horizon], X_cal, y_cal[horizon])
                horizon_models.append(model)

            self.models[horizon] = horizon_models

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> MultiPeriodPredictionResult:
        """
        Generate ensemble predictions for all horizons.

        Parameters
        ----------
        X : np.ndarray
            Features for prediction.

        Returns
        -------
        MultiPeriodPredictionResult
            Container with ensemble predictions for all horizons.
        """
        if not self._is_fitted:
            raise RuntimeError("Forecaster not fitted. Call fit() first.")

        predictions = {}
        weights = np.array(self.weights)
        weights = weights / weights.sum()

        for horizon in self.horizons:
            all_points = []
            all_lowers = []
            all_uppers = []

            for model in self.models[horizon]:
                pred = model.predict(X)
                all_points.append(pred.point)
                if pred.has_intervals:
                    all_lowers.append(pred.lower)
                    all_uppers.append(pred.upper)

            all_points = np.array(all_points)

            if self.ensemble_method == "mean":
                point = np.average(all_points, axis=0, weights=weights)
            elif self.ensemble_method == "median":
                point = np.median(all_points, axis=0)
            elif self.ensemble_method == "weighted":
                point = np.average(all_points, axis=0, weights=weights)
            else:
                raise ValueError(f"Unknown ensemble method: {self.ensemble_method}")

            # Combine intervals (use widest interval for coverage)
            if all_lowers:
                all_lowers = np.array(all_lowers)
                all_uppers = np.array(all_uppers)
                lower = np.min(all_lowers, axis=0)
                upper = np.max(all_uppers, axis=0)
            else:
                lower, upper = None, None

            predictions[horizon] = PredictionResult(
                point=point, lower=lower, upper=upper
            )

        return MultiPeriodPredictionResult(
            horizons=self.horizons,
            predictions=predictions
        )

    @property
    def is_fitted(self) -> bool:
        return self._is_fitted


def create_multi_period_forecaster(
    base_model_name: str,
    horizons: List[int],
    **model_kwargs
) -> MultiPeriodForecaster:
    """
    Factory function to create a multi-period forecaster from model name.

    Parameters
    ----------
    base_model_name : str
        Name of the base model class.
    horizons : List[int]
        List of forecast horizons.
    **model_kwargs
        Arguments passed to the model.

    Returns
    -------
    MultiPeriodForecaster
        Configured multi-period forecaster.

    Examples
    --------
    >>> mp = create_multi_period_forecaster(
    ...     'ConformalPrediction',
    ...     horizons=[1, 7, 14, 21, 28],
    ...     alpha=0.05
    ... )
    """
    from . import (
        HistoricalQuantile,
        NormalAssumption,
        BootstrappedNewsvendor,
        SampleAverageApproximation,
        TwoStageStochastic,
        ConformalPrediction,
        QuantileRegression,
        EnsembleBatchPI,
        DistributionallyRobustOptimization,
        Seer,
    )

    model_map = {
        'HistoricalQuantile': HistoricalQuantile,
        'NormalAssumption': NormalAssumption,
        'BootstrappedNewsvendor': BootstrappedNewsvendor,
        'SampleAverageApproximation': SampleAverageApproximation,
        'SAA': SampleAverageApproximation,
        'TwoStageStochastic': TwoStageStochastic,
        'ConformalPrediction': ConformalPrediction,
        'QuantileRegression': QuantileRegression,
        'EnsembleBatchPI': EnsembleBatchPI,
        'EnbPI_CQR_CVaR': EnsembleBatchPI,
        'DistributionallyRobustOptimization': DistributionallyRobustOptimization,
        'DRO': DistributionallyRobustOptimization,
        'Seer': Seer,
    }

    if base_model_name not in model_map:
        raise ValueError(f"Unknown model: {base_model_name}. Available: {list(model_map.keys())}")

    return MultiPeriodForecaster(
        base_model_class=model_map[base_model_name],
        horizons=horizons,
        **model_kwargs
    )
