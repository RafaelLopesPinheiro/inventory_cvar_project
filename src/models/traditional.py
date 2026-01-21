"""
Traditional (non-deep learning) forecasting models.

Models implemented:
- ConformalPrediction: Distribution-free prediction intervals with coverage guarantees
- NormalAssumption: Parametric Gaussian assumption
- QuantileRegression: Direct quantile estimation with conformal calibration
- SampleAverageApproximation: Standard OR benchmark
"""

import numpy as np
from scipy import stats
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from typing import Optional
import logging

from .base import BaseForecaster, PredictionResult

logger = logging.getLogger(__name__)


class ConformalPrediction(BaseForecaster):
    """
    Conformal Prediction with Random Forest.
    
    Uses split conformal prediction to create distribution-free prediction
    intervals with finite-sample coverage guarantees.
    
    The method:
    1. Trains a point predictor (Random Forest) on training data
    2. Computes nonconformity scores on calibration data
    3. Uses the (1-alpha) quantile of scores for prediction intervals
    
    References
    ----------
    - Vovk et al. (2005) "Algorithmic Learning in a Random World"
    - Romano et al. (2019) "Conformalized Quantile Regression"
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        super().__init__(alpha=alpha, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.q_conformal: Optional[float] = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "ConformalPrediction":
        """
        Train on training data and calibrate on calibration data.
        """
        logger.info("Training Conformal Prediction model...")
        
        # Train point predictor
        self.model.fit(X_train, y_train)
        
        # Compute nonconformity scores on calibration data
        cal_predictions = self.model.predict(X_cal)
        nonconformity_scores = np.abs(y_cal - cal_predictions)
        
        # Calculate conformal quantile at (1-alpha) coverage
        n = len(y_cal)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.q_conformal = np.quantile(nonconformity_scores, min(quantile_level, 1.0))
        
        logger.info(f"Conformal quantile: {self.q_conformal:.4f}")
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Generate predictions with conformal intervals.
        """
        self._check_is_fitted()
        
        point_pred = self.model.predict(X)
        lower = point_pred - self.q_conformal
        upper = point_pred + self.q_conformal
        
        return PredictionResult(point=point_pred, lower=lower, upper=upper)
    
    def get_params(self):
        params = super().get_params()
        params.update({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "q_conformal": self.q_conformal
        })
        return params


class NormalAssumption(BaseForecaster):
    """
    Normal Assumption model with empirical variance estimation.
    
    Assumes residuals follow a Gaussian distribution and constructs
    prediction intervals using the estimated standard deviation.
    
    The method:
    1. Trains a point predictor (Random Forest)
    2. Estimates residual standard deviation on calibration data
    3. Uses Normal quantiles for prediction intervals
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        super().__init__(alpha=alpha, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        self.sigma: Optional[float] = None
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "NormalAssumption":
        """
        Train and estimate residual standard deviation.
        """
        logger.info("Training Normal Assumption model...")
        
        # Train point predictor
        self.model.fit(X_train, y_train)
        
        # Estimate sigma from calibration residuals
        cal_predictions = self.model.predict(X_cal)
        residuals = y_cal - cal_predictions
        self.sigma = np.std(residuals)
        
        logger.info(f"Estimated sigma: {self.sigma:.4f}")
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Generate predictions with Normal-based intervals.
        """
        self._check_is_fitted()
        
        point_pred = self.model.predict(X)
        z_alpha = stats.norm.ppf(1 - self.alpha / 2)
        lower = point_pred - z_alpha * self.sigma
        upper = point_pred + z_alpha * self.sigma
        
        return PredictionResult(point=point_pred, lower=lower, upper=upper)
    
    def get_params(self):
        params = super().get_params()
        params.update({
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "sigma": self.sigma
        })
        return params


class QuantileRegression(BaseForecaster):
    """
    Quantile Regression with conformal calibration.
    
    Trains separate Gradient Boosting models for each quantile and applies
    conformal calibration to ensure proper coverage.
    
    The method:
    1. Trains quantile models for lower, median, and upper quantiles
    2. Applies conformal calibration to expand intervals for coverage
    """
    
    def __init__(
        self,
        alpha: float = 0.05,
        n_estimators: int = 100,
        max_depth: int = 8,
        random_state: int = 42
    ):
        super().__init__(alpha=alpha, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        self.q_low = alpha / 2
        self.q_high = 1 - alpha / 2
        
        self.model_low = GradientBoostingRegressor(
            loss='quantile', alpha=self.q_low,
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state
        )
        self.model_median = GradientBoostingRegressor(
            loss='quantile', alpha=0.5,
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state
        )
        self.model_high = GradientBoostingRegressor(
            loss='quantile', alpha=self.q_high,
            n_estimators=n_estimators, max_depth=max_depth,
            random_state=random_state
        )
        
        self.calibration_adjustment: float = 0.0
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "QuantileRegression":
        """
        Train quantile models and apply conformal calibration.
        """
        logger.info(f"Training Quantile Regression ({self.q_low:.3f}, 0.5, {self.q_high:.3f})...")
        
        # Train quantile models
        self.model_low.fit(X_train, y_train)
        self.model_median.fit(X_train, y_train)
        self.model_high.fit(X_train, y_train)
        
        # Conformal calibration
        logger.info("Applying conformal calibration...")
        cal_lower = self.model_low.predict(X_cal)
        cal_upper = self.model_high.predict(X_cal)
        
        # Compute violations
        lower_violations = np.maximum(0, cal_lower - y_cal)
        upper_violations = np.maximum(0, y_cal - cal_upper)
        max_violations = np.maximum(lower_violations, upper_violations)
        
        # Conformal quantile
        n = len(y_cal)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.calibration_adjustment = np.quantile(max_violations, min(quantile_level, 1.0))
        
        logger.info(f"Calibration adjustment: {self.calibration_adjustment:.4f}")
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Generate predictions with calibrated quantile intervals.
        """
        self._check_is_fitted()
        
        point_pred = self.model_median.predict(X)
        lower = self.model_low.predict(X) - self.calibration_adjustment
        upper = self.model_high.predict(X) + self.calibration_adjustment
        
        return PredictionResult(point=point_pred, lower=lower, upper=upper)


class SampleAverageApproximation(BaseForecaster):
    """
    Sample Average Approximation (SAA) for the Newsvendor Problem.
    
    SAA is the standard benchmark in operations research. It directly
    minimizes the empirical newsvendor loss over historical data.
    
    For feature-based SAA:
    1. Train a point predictor on training data
    2. Compute residuals on calibration data
    3. For each test point, generate demand scenarios by adding historical residuals
    4. Find optimal order quantity as the critical quantile
    
    References
    ----------
    - Levi et al. (2015) "The Data-Driven Newsvendor Problem"
    - Ban & Rudin (2019) "The Big Data Newsvendor"
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42,
        stockout_cost: float = 50.0,
        holding_cost: float = 2.0
    ):
        # SAA doesn't use alpha for prediction intervals
        super().__init__(alpha=0.05, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.stockout_cost = stockout_cost
        self.holding_cost = holding_cost
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
        
        self.residuals: Optional[np.ndarray] = None
        
    @property
    def critical_ratio(self) -> float:
        """Critical ratio: cu / (cu + co)"""
        return self.stockout_cost / (self.stockout_cost + self.holding_cost)
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "SampleAverageApproximation":
        """
        Train model and store calibration residuals.
        """
        logger.info("Training SAA model...")
        
        # Train point predictor
        self.model.fit(X_train, y_train)
        
        # Store calibration residuals for demand scenarios
        cal_predictions = self.model.predict(X_cal)
        self.residuals = y_cal - cal_predictions
        
        logger.info(f"Stored {len(self.residuals)} residuals")
        logger.info(f"Critical ratio: {self.critical_ratio:.4f}")
        
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Generate point predictions (no intervals for SAA).
        """
        self._check_is_fitted()
        point_pred = self.model.predict(X)
        return PredictionResult(point=point_pred, lower=None, upper=None)
    
    def compute_order_quantities(self, X: np.ndarray) -> np.ndarray:
        """
        Compute SAA-optimal order quantities.
        
        For each prediction, generate demand scenarios and find the
        critical quantile.
        
        Parameters
        ----------
        X : np.ndarray
            Features for prediction.
            
        Returns
        -------
        np.ndarray
            Optimal order quantities.
        """
        self._check_is_fitted()
        
        point_predictions = self.model.predict(X)
        order_quantities = []
        
        for point_pred in point_predictions:
            # Generate demand scenarios
            demand_scenarios = point_pred + self.residuals
            
            # SAA solution: critical quantile
            q_optimal = np.quantile(demand_scenarios, self.critical_ratio)
            order_quantities.append(max(0, q_optimal))
        
        return np.array(order_quantities)


class ExpectedValue(BaseForecaster):
    """
    Simple expected value (risk-neutral) model.
    
    Uses point predictions directly as order quantities.
    No uncertainty quantification.
    """
    
    def __init__(
        self,
        n_estimators: int = 100,
        max_depth: int = 10,
        random_state: int = 42
    ):
        super().__init__(alpha=0.05, random_state=random_state)
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        
        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )
    
    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "ExpectedValue":
        """Train the point predictor."""
        logger.info("Training Expected Value model...")
        self.model.fit(X_train, y_train)
        self._is_fitted = True
        return self
    
    def predict(self, X: np.ndarray) -> PredictionResult:
        """Generate point predictions (no intervals)."""
        self._check_is_fitted()
        point_pred = self.model.predict(X)
        return PredictionResult(point=point_pred, lower=None, upper=None)
