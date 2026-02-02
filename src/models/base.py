"""
Base classes for forecasting models.

This module defines the abstract interface that all forecasting models must implement.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple, Optional, List, Dict, Any
import numpy as np


@dataclass
class PredictionResult:
    """Container for model predictions."""
    point: np.ndarray  # Point predictions
    lower: Optional[np.ndarray] = None  # Lower bound of prediction interval
    upper: Optional[np.ndarray] = None  # Upper bound of prediction interval

    @property
    def has_intervals(self) -> bool:
        """Check if prediction intervals are available."""
        return self.lower is not None and self.upper is not None

    @property
    def interval_width(self) -> Optional[np.ndarray]:
        """Compute prediction interval width."""
        if self.has_intervals:
            return self.upper - self.lower
        return None

    def __len__(self) -> int:
        return len(self.point)


@dataclass
class MultiPeriodPredictionResult:
    """
    Container for multi-period (multi-horizon) forecasting predictions.

    This structure supports predicting multiple future horizons simultaneously,
    which enables more robust scientific evaluation and joint optimization.

    Attributes
    ----------
    horizons : List[int]
        List of forecast horizons (days ahead) that were predicted.
        Example: [1, 7, 14, 21, 28] for 1-day, 1-week, 2-week, 3-week, 4-week ahead.
    predictions : Dict[int, PredictionResult]
        Dictionary mapping each horizon to its PredictionResult.
        Keys are horizon values (e.g., 1, 7, 14, 21, 28).

    Examples
    --------
    >>> mp_result = MultiPeriodPredictionResult(
    ...     horizons=[1, 7, 14],
    ...     predictions={
    ...         1: PredictionResult(point=..., lower=..., upper=...),
    ...         7: PredictionResult(point=..., lower=..., upper=...),
    ...         14: PredictionResult(point=..., lower=..., upper=...),
    ...     }
    ... )
    >>> mp_result.get_horizon(7)  # Get 7-day ahead predictions
    PredictionResult(...)
    >>> mp_result.aggregate_point('mean')  # Mean across all horizons
    array([...])

    References
    ----------
    - Taieb et al. (2012) "A review and comparison of strategies for multi-step ahead
      time series forecasting based on NN5 competition"
    - Hyndman & Athanasopoulos (2021) "Forecasting: Principles and Practice" Ch. 13
    """
    horizons: List[int]  # List of forecast horizons (days ahead)
    predictions: Dict[int, PredictionResult]  # horizon -> PredictionResult

    def get_horizon(self, horizon: int) -> PredictionResult:
        """
        Get predictions for a specific horizon.

        Parameters
        ----------
        horizon : int
            The forecast horizon to retrieve.

        Returns
        -------
        PredictionResult
            Predictions for the specified horizon.

        Raises
        ------
        KeyError
            If the horizon is not in the predictions.
        """
        if horizon not in self.predictions:
            raise KeyError(f"Horizon {horizon} not found. Available: {self.horizons}")
        return self.predictions[horizon]

    @property
    def n_horizons(self) -> int:
        """Number of horizons in the multi-period prediction."""
        return len(self.horizons)

    @property
    def n_samples(self) -> int:
        """Number of samples (time points) predicted."""
        if self.horizons:
            return len(self.predictions[self.horizons[0]])
        return 0

    def aggregate_point(self, method: str = "mean") -> np.ndarray:
        """
        Aggregate point predictions across all horizons.

        Parameters
        ----------
        method : str
            Aggregation method: "mean", "median", "max", "min", "sum".

        Returns
        -------
        np.ndarray
            Aggregated point predictions with shape (n_samples,).
        """
        points = np.array([self.predictions[h].point for h in self.horizons])
        # Shape: (n_horizons, n_samples)

        if method == "mean":
            return np.mean(points, axis=0)
        elif method == "median":
            return np.median(points, axis=0)
        elif method == "max":
            return np.max(points, axis=0)
        elif method == "min":
            return np.min(points, axis=0)
        elif method == "sum":
            return np.sum(points, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def aggregate_intervals(
        self, method: str = "union"
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Aggregate prediction intervals across all horizons.

        Parameters
        ----------
        method : str
            Aggregation method:
            - "union": Take widest interval (min lower, max upper)
            - "intersection": Take narrowest interval (max lower, min upper)
            - "mean": Average the bounds

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            Aggregated (lower, upper) bounds with shape (n_samples,).
        """
        lowers = []
        uppers = []

        for h in self.horizons:
            pred = self.predictions[h]
            if pred.has_intervals:
                lowers.append(pred.lower)
                uppers.append(pred.upper)

        if not lowers:
            return None, None

        lowers = np.array(lowers)  # (n_horizons, n_samples)
        uppers = np.array(uppers)

        if method == "union":
            return np.min(lowers, axis=0), np.max(uppers, axis=0)
        elif method == "intersection":
            return np.max(lowers, axis=0), np.min(uppers, axis=0)
        elif method == "mean":
            return np.mean(lowers, axis=0), np.mean(uppers, axis=0)
        else:
            raise ValueError(f"Unknown aggregation method: {method}")

    def to_single_period(self, aggregation: str = "mean") -> PredictionResult:
        """
        Convert multi-period predictions to a single PredictionResult.

        This is useful for backward compatibility with single-period methods.

        Parameters
        ----------
        aggregation : str
            Method for aggregating predictions across horizons.

        Returns
        -------
        PredictionResult
            Single aggregated prediction result.
        """
        point = self.aggregate_point(aggregation)
        lower, upper = self.aggregate_intervals("union")
        return PredictionResult(point=point, lower=lower, upper=upper)

    def get_point_matrix(self) -> np.ndarray:
        """
        Get point predictions as a matrix.

        Returns
        -------
        np.ndarray
            Matrix of shape (n_samples, n_horizons) with point predictions.
        """
        return np.array([self.predictions[h].point for h in self.horizons]).T

    def get_interval_matrices(
        self,
    ) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Get prediction intervals as matrices.

        Returns
        -------
        Tuple[Optional[np.ndarray], Optional[np.ndarray]]
            Lower and upper bound matrices of shape (n_samples, n_horizons).
        """
        if not all(self.predictions[h].has_intervals for h in self.horizons):
            return None, None

        lower = np.array([self.predictions[h].lower for h in self.horizons]).T
        upper = np.array([self.predictions[h].upper for h in self.horizons]).T
        return lower, upper

    def __len__(self) -> int:
        return self.n_samples


class BaseForecaster(ABC):
    """
    Abstract base class for all forecasting models.
    
    All forecasting models must implement:
    - fit(): Train the model
    - predict(): Generate predictions
    
    Optional methods:
    - get_params(): Return model parameters
    - set_params(): Set model parameters
    """
    
    def __init__(self, alpha: float = 0.05, random_state: int = 42):
        """
        Initialize the forecaster.
        
        Parameters
        ----------
        alpha : float
            Significance level for prediction intervals (1-alpha coverage).
        random_state : int
            Random seed for reproducibility.
        """
        self.alpha = alpha
        self.random_state = random_state
        self._is_fitted = False
    
    @abstractmethod
    def fit(
        self, 
        X_train: np.ndarray, 
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "BaseForecaster":
        """
        Train the model on training data and calibrate on calibration data.
        
        Parameters
        ----------
        X_train : np.ndarray
            Training features.
        y_train : np.ndarray
            Training targets.
        X_cal : np.ndarray
            Calibration features.
        y_cal : np.ndarray
            Calibration targets.
            
        Returns
        -------
        self
            The fitted model.
        """
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Generate predictions for new data.
        
        Parameters
        ----------
        X : np.ndarray
            Features for prediction.
            
        Returns
        -------
        PredictionResult
            Predictions with optional intervals.
        """
        pass
    
    def fit_predict(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray,
        X_test: np.ndarray
    ) -> PredictionResult:
        """
        Fit the model and generate predictions.
        
        Parameters
        ----------
        X_train, y_train : Training data
        X_cal, y_cal : Calibration data
        X_test : Test features
        
        Returns
        -------
        PredictionResult
            Predictions on test data.
        """
        self.fit(X_train, y_train, X_cal, y_cal)
        return self.predict(X_test)
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        return {
            "alpha": self.alpha,
            "random_state": self.random_state
        }
    
    def set_params(self, **params) -> "BaseForecaster":
        """Set model parameters."""
        for key, value in params.items():
            if hasattr(self, key):
                setattr(self, key, value)
        return self
    
    @property
    def is_fitted(self) -> bool:
        """Check if the model has been fitted."""
        return self._is_fitted
    
    def _check_is_fitted(self):
        """Raise error if model is not fitted."""
        if not self._is_fitted:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted. "
                "Call fit() before predict()."
            )


class BaseDeepLearningForecaster(BaseForecaster):
    """
    Abstract base class for deep learning forecasting models.
    
    Extends BaseForecaster with deep learning specific attributes:
    - sequence_length: Input sequence length
    - device: Computation device (cpu/cuda)
    - training_losses: Training loss history
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        sequence_length: int = 28,
        random_state: int = 42,
        device: str = "cpu"
    ):
        super().__init__(alpha=alpha, random_state=random_state)
        self.sequence_length = sequence_length
        self.device = device
        self.training_losses: List[float] = []
        
        # Normalization parameters
        self.scaler_mean: Optional[np.ndarray] = None
        self.scaler_std: Optional[np.ndarray] = None
        
        # Calibration adjustment
        self.calibration_adjustment: float = 0.0
    
    def _normalize(self, X: np.ndarray, fit: bool = False) -> np.ndarray:
        """
        Normalize features using standardization.
        
        Parameters
        ----------
        X : np.ndarray
            Features to normalize.
        fit : bool
            If True, compute normalization parameters from X.
            
        Returns
        -------
        np.ndarray
            Normalized features.
        """
        if fit:
            self.scaler_mean = X.mean(axis=(0, 1), keepdims=True)
            self.scaler_std = X.std(axis=(0, 1), keepdims=True) + 1e-8
        return (X - self.scaler_mean) / self.scaler_std
    
    def _calibrate(self, X_cal: np.ndarray, y_cal: np.ndarray):
        """
        Apply conformal calibration to ensure coverage guarantees.
        
        Parameters
        ----------
        X_cal : np.ndarray
            Calibration features.
        y_cal : np.ndarray
            Calibration targets.
        """
        # Get uncalibrated predictions
        predictions = self._predict_raw(X_cal)
        
        if predictions.lower is None or predictions.upper is None:
            return
        
        # Compute violations
        lower_violations = np.maximum(0, predictions.lower - y_cal)
        upper_violations = np.maximum(0, y_cal - predictions.upper)
        max_violations = np.maximum(lower_violations, upper_violations)
        
        # Compute conformal quantile
        n = len(y_cal)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.calibration_adjustment = np.quantile(
            max_violations, 
            min(quantile_level, 1.0)
        )
    
    @abstractmethod
    def _predict_raw(self, X: np.ndarray) -> PredictionResult:
        """
        Generate raw (uncalibrated) predictions.
        
        Parameters
        ----------
        X : np.ndarray
            Features for prediction.
            
        Returns
        -------
        PredictionResult
            Uncalibrated predictions.
        """
        pass
    
    def get_params(self) -> Dict[str, Any]:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            "sequence_length": self.sequence_length,
            "device": self.device
        })
        return params
