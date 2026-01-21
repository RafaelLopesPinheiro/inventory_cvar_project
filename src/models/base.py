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
