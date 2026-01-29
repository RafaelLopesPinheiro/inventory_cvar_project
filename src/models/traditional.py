"""
Traditional (non-deep learning) forecasting models.

Models implemented:
- HistoricalQuantile: Naïve empirical quantile baseline
- NormalAssumption: Parametric Gaussian assumption
- BootstrappedNewsvendor: Bootstrap resampling for uncertainty quantification
- SampleAverageApproximation: Standard OR benchmark
- TwoStageStochastic: Scenario-based stochastic optimization
- ConformalPrediction: Distribution-free prediction intervals with coverage guarantees
- QuantileRegression: Direct quantile estimation with conformal calibration
- EnsembleBatchPI: EnbPI + CQR for adaptive prediction intervals
- Seer: Oracle with perfect foresight (upper bound)

References:
- Vovk et al. (2005) "Algorithmic Learning in a Random World"
- Romano et al. (2019) "Conformalized Quantile Regression"
- Xu & Xie (2021) "Conformal prediction interval for dynamic time-series"
- Birge & Louveaux (2011) "Introduction to Stochastic Programming"
- Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
"""

import numpy as np
from scipy import stats
from scipy.optimize import minimize, linprog
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from typing import Optional, List, Tuple, Set
import logging

from .base import BaseForecaster, PredictionResult

logger = logging.getLogger(__name__)


# =============================================================================
# HISTORICAL QUANTILE (NAÏVE BASELINE)
# =============================================================================

class HistoricalQuantile(BaseForecaster):
    """
    Historical Quantile (Naïve) Baseline.

    This is the simplest possible uncertainty quantification method that uses
    empirical quantiles directly from historical data without any modeling.

    The method:
    1. Computes point prediction as the historical mean (or median)
    2. Uses empirical quantiles of historical demand for prediction intervals
    3. Does NOT use features - purely data-driven baseline

    This represents the "do nothing sophisticated" approach and serves as a
    lower bound for what any reasonable method should beat.

    Mathematical formulation:
    - Point prediction: μ̂ = mean(D_train)
    - Lower bound: q̂_α/2 = Quantile(D_train ∪ D_cal, α/2)
    - Upper bound: q̂_(1-α/2) = Quantile(D_train ∪ D_cal, 1-α/2)

    References
    ----------
    - Hyndman & Athanasopoulos (2021) "Forecasting: Principles and Practice"
    - Makridakis et al. (2020) "The M5 Accuracy Competition"
    """

    def __init__(
        self,
        alpha: float = 0.05,
        use_median: bool = False,
        random_state: int = 42
    ):
        """
        Initialize the Historical Quantile model.

        Parameters
        ----------
        alpha : float
            Significance level for prediction intervals (1-alpha coverage).
        use_median : bool
            If True, use median as point prediction; otherwise use mean.
        random_state : int
            Random seed for reproducibility (not used, for API consistency).
        """
        super().__init__(alpha=alpha, random_state=random_state)
        self.use_median = use_median

        # Computed quantities
        self.point_estimate: Optional[float] = None
        self.lower_quantile: Optional[float] = None
        self.upper_quantile: Optional[float] = None
        self.historical_demand: Optional[np.ndarray] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "HistoricalQuantile":
        """
        Compute historical statistics from training and calibration data.

        Note: X_train and X_cal are NOT used - this is a purely data-driven
        baseline that only uses historical demand values.
        """
        logger.info("Training Historical Quantile (Naïve) baseline...")

        # Combine all historical demand
        self.historical_demand = np.concatenate([y_train, y_cal])

        # Compute point estimate
        if self.use_median:
            self.point_estimate = np.median(self.historical_demand)
            logger.info(f"Point estimate (median): {self.point_estimate:.2f}")
        else:
            self.point_estimate = np.mean(self.historical_demand)
            logger.info(f"Point estimate (mean): {self.point_estimate:.2f}")

        # Compute empirical quantiles for prediction intervals
        self.lower_quantile = np.quantile(self.historical_demand, self.alpha / 2)
        self.upper_quantile = np.quantile(self.historical_demand, 1 - self.alpha / 2)

        logger.info(f"Prediction interval: [{self.lower_quantile:.2f}, {self.upper_quantile:.2f}]")
        logger.info(f"Historical demand stats: min={self.historical_demand.min():.2f}, "
                   f"max={self.historical_demand.max():.2f}, std={self.historical_demand.std():.2f}")

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Generate predictions using historical statistics.

        Note: X is NOT used - all predictions are the same constant values.
        """
        self._check_is_fitted()

        n_samples = len(X)

        # Constant predictions (same for all time points)
        point_pred = np.full(n_samples, self.point_estimate)
        lower = np.full(n_samples, self.lower_quantile)
        upper = np.full(n_samples, self.upper_quantile)

        return PredictionResult(point=point_pred, lower=lower, upper=upper)

    def get_params(self) -> dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            "use_median": self.use_median,
            "point_estimate": self.point_estimate,
            "lower_quantile": self.lower_quantile,
            "upper_quantile": self.upper_quantile
        })
        return params


# =============================================================================
# BOOTSTRAPPED NEWSVENDOR (RESAMPLING-BASED)
# =============================================================================

class BootstrappedNewsvendor(BaseForecaster):
    """
    Bootstrapped Newsvendor with Resampling-Based Uncertainty Quantification.

    This method uses bootstrap resampling to estimate the uncertainty in
    demand forecasts and compute prediction intervals without parametric
    assumptions.

    The method:
    1. Train a point predictor (Random Forest) on training data
    2. Bootstrap resample residuals from calibration data
    3. Generate multiple demand scenarios by adding resampled residuals
    4. Compute prediction intervals from the bootstrap distribution
    5. Order quantity is the (1-α) quantile of scenarios (newsvendor solution)

    Key advantages:
    - Non-parametric: No distributional assumptions
    - Captures heteroscedasticity through residual resampling
    - Directly interpretable uncertainty estimates

    Mathematical formulation:
    For each prediction x_i:
    1. Point prediction: ŷ_i = f(x_i)
    2. Residuals: ε_j = y_j - f(x_j), j ∈ calibration set
    3. Bootstrap scenarios: D_i^(b) = ŷ_i + ε*_b, where ε*_b ~ Bootstrap(ε)
    4. Prediction interval: [Q_{α/2}(D_i), Q_{1-α/2}(D_i)]

    References
    ----------
    - Efron & Tibshirani (1993) "An Introduction to the Bootstrap"
    - Levi et al. (2015) "The Data-Driven Newsvendor Problem"
    - Ban & Rudin (2019) "The Big Data Newsvendor"
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_bootstrap: int = 1000,
        n_estimators: int = 100,
        max_depth: int = 10,
        block_bootstrap: bool = False,
        block_size: int = 7,
        random_state: int = 42
    ):
        """
        Initialize the Bootstrapped Newsvendor model.

        Parameters
        ----------
        alpha : float
            Significance level for prediction intervals (1-alpha coverage).
        n_bootstrap : int
            Number of bootstrap samples to generate.
        n_estimators : int
            Number of trees in the Random Forest.
        max_depth : int
            Maximum depth of trees.
        block_bootstrap : bool
            If True, use block bootstrap to preserve temporal structure.
        block_size : int
            Size of blocks for block bootstrap.
        random_state : int
            Random seed for reproducibility.
        """
        super().__init__(alpha=alpha, random_state=random_state)
        self.n_bootstrap = n_bootstrap
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.block_bootstrap = block_bootstrap
        self.block_size = block_size

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

        # Stored quantities
        self.residuals: Optional[np.ndarray] = None
        self.bootstrap_indices: Optional[np.ndarray] = None

    def _generate_block_bootstrap_indices(
        self,
        n_samples: int,
        n_residuals: int,
        rng: np.random.RandomState
    ) -> np.ndarray:
        """
        Generate block bootstrap indices to preserve temporal dependence.

        Parameters
        ----------
        n_samples : int
            Number of bootstrap samples to generate.
        n_residuals : int
            Number of residuals available.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        np.ndarray
            Bootstrap indices of shape (n_samples,).
        """
        indices = []
        n_blocks = (n_samples + self.block_size - 1) // self.block_size

        for _ in range(n_blocks):
            # Randomly select block starting position
            start_idx = rng.randint(0, n_residuals - self.block_size + 1)
            block_indices = np.arange(start_idx, start_idx + self.block_size)
            indices.extend(block_indices)

        return np.array(indices[:n_samples])

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "BootstrappedNewsvendor":
        """
        Train model and compute residuals for bootstrap resampling.
        """
        logger.info("Training Bootstrapped Newsvendor...")
        logger.info(f"Bootstrap samples: {self.n_bootstrap}, "
                   f"Block bootstrap: {self.block_bootstrap}")

        # Train point predictor
        self.model.fit(X_train, y_train)

        # Compute residuals on calibration data
        cal_predictions = self.model.predict(X_cal)
        self.residuals = y_cal - cal_predictions

        logger.info(f"Residual statistics: mean={np.mean(self.residuals):.2f}, "
                   f"std={np.std(self.residuals):.2f}")

        # Pre-generate bootstrap indices for efficiency
        rng = np.random.RandomState(self.random_state)
        if self.block_bootstrap and len(self.residuals) >= self.block_size:
            self.bootstrap_indices = self._generate_block_bootstrap_indices(
                self.n_bootstrap, len(self.residuals), rng
            )
        else:
            self.bootstrap_indices = rng.choice(
                len(self.residuals),
                size=self.n_bootstrap,
                replace=True
            )

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Generate predictions with bootstrap-based prediction intervals.
        """
        self._check_is_fitted()

        # Point predictions
        point_pred = self.model.predict(X)

        # Generate bootstrap scenarios for each prediction
        n_samples = len(X)
        lower = np.zeros(n_samples)
        upper = np.zeros(n_samples)

        # Resample residuals
        bootstrap_residuals = self.residuals[self.bootstrap_indices]

        for i in range(n_samples):
            # Generate demand scenarios
            scenarios = point_pred[i] + bootstrap_residuals

            # Compute prediction interval quantiles
            lower[i] = np.quantile(scenarios, self.alpha / 2)
            upper[i] = np.quantile(scenarios, 1 - self.alpha / 2)

        return PredictionResult(point=point_pred, lower=lower, upper=upper)

    def compute_order_quantities(
        self,
        X: np.ndarray,
        critical_ratio: float
    ) -> np.ndarray:
        """
        Compute newsvendor-optimal order quantities.

        Uses the bootstrap distribution to find the critical quantile.

        Parameters
        ----------
        X : np.ndarray
            Features for prediction.
        critical_ratio : float
            Newsvendor critical ratio: cu / (cu + co).

        Returns
        -------
        np.ndarray
            Optimal order quantities.
        """
        self._check_is_fitted()

        point_pred = self.model.predict(X)
        bootstrap_residuals = self.residuals[self.bootstrap_indices]

        order_quantities = []
        for i in range(len(X)):
            scenarios = point_pred[i] + bootstrap_residuals
            q_optimal = np.quantile(scenarios, critical_ratio)
            order_quantities.append(max(0, q_optimal))

        return np.array(order_quantities)

    def get_params(self) -> dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            "n_bootstrap": self.n_bootstrap,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "block_bootstrap": self.block_bootstrap,
            "block_size": self.block_size
        })
        return params


# =============================================================================
# TWO-STAGE STOCHASTIC PROGRAMMING
# =============================================================================

class TwoStageStochastic(BaseForecaster):
    """
    Two-Stage Stochastic Programming for Inventory Optimization.

    This method formulates the newsvendor problem as a two-stage stochastic
    program and solves it using scenario-based optimization.

    Stage 1 (Here-and-now): Decide order quantity q before demand is known
    Stage 2 (Wait-and-see): Realize demand D, incur holding/stockout costs

    Mathematical formulation:
    min_q  c_o * q + E[Q(q, D)]

    where Q(q, D) is the recourse function:
    Q(q, D) = min_{y^+, y^-}  c_h * y^+ + c_u * y^-
              s.t.  y^+ - y^- = q - D
                    y^+, y^- >= 0

    This simplifies to:
    Q(q, D) = c_h * max(0, q - D) + c_u * max(0, D - q)

    Using SAA (Sample Average Approximation):
    min_q  c_o * q + (1/N) * Σ_i [c_h * max(0, q - d_i) + c_u * max(0, d_i - q)]

    Key differences from simple SAA:
    - Explicit two-stage formulation
    - Can incorporate additional constraints
    - Extensible to multi-stage problems
    - Provides bounds on optimal value

    References
    ----------
    - Birge & Louveaux (2011) "Introduction to Stochastic Programming"
    - Shapiro, Dentcheva, Ruszczynski (2009) "Lectures on Stochastic Programming"
    - Kleywegt et al. (2002) "The Sample Average Approximation Method"
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_scenarios: int = 500,
        n_estimators: int = 100,
        max_depth: int = 10,
        ordering_cost: float = 10.0,
        holding_cost: float = 2.0,
        stockout_cost: float = 50.0,
        use_cvar: bool = False,
        cvar_beta: float = 0.90,
        random_state: int = 42
    ):
        """
        Initialize the Two-Stage Stochastic model.

        Parameters
        ----------
        alpha : float
            Significance level for prediction intervals.
        n_scenarios : int
            Number of demand scenarios for SAA.
        n_estimators : int
            Number of trees in the Random Forest.
        max_depth : int
            Maximum depth of trees.
        ordering_cost : float
            Cost per unit ordered.
        holding_cost : float
            Cost per unit of overage.
        stockout_cost : float
            Cost per unit of underage.
        use_cvar : bool
            If True, optimize CVaR instead of expected cost.
        cvar_beta : float
            CVaR level (only used if use_cvar=True).
        random_state : int
            Random seed for reproducibility.
        """
        super().__init__(alpha=alpha, random_state=random_state)
        self.n_scenarios = n_scenarios
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.ordering_cost = ordering_cost
        self.holding_cost = holding_cost
        self.stockout_cost = stockout_cost
        self.use_cvar = use_cvar
        self.cvar_beta = cvar_beta

        self.model = RandomForestRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=random_state,
            n_jobs=-1
        )

        # Stored quantities
        self.residuals: Optional[np.ndarray] = None
        self.sigma: Optional[float] = None

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "TwoStageStochastic":
        """
        Train model and compute residual distribution for scenario generation.
        """
        logger.info("Training Two-Stage Stochastic Programming model...")
        logger.info(f"Scenarios: {self.n_scenarios}, CVaR: {self.use_cvar}")

        # Train point predictor
        self.model.fit(X_train, y_train)

        # Compute residuals for scenario generation
        cal_predictions = self.model.predict(X_cal)
        self.residuals = y_cal - cal_predictions
        self.sigma = np.std(self.residuals)

        logger.info(f"Residual std: {self.sigma:.2f}")

        self._is_fitted = True
        return self

    def _solve_two_stage(
        self,
        point_pred: float,
        scenarios: np.ndarray
    ) -> Tuple[float, float, float]:
        """
        Solve the two-stage stochastic program for a single time period.

        Parameters
        ----------
        point_pred : float
            Point prediction (not directly used, but available).
        scenarios : np.ndarray
            Demand scenarios.

        Returns
        -------
        Tuple[float, float, float]
            Optimal order quantity, lower bound, upper bound.
        """
        n = len(scenarios)

        def objective(q):
            """Expected cost objective."""
            overage = np.maximum(0, q - scenarios)
            underage = np.maximum(0, scenarios - q)
            stage1_cost = self.ordering_cost * q
            stage2_cost = np.mean(
                self.holding_cost * overage + self.stockout_cost * underage
            )
            return stage1_cost + stage2_cost

        def cvar_objective(params):
            """CVaR objective using Rockafellar-Uryasev formulation."""
            q, tau = params
            overage = np.maximum(0, q - scenarios)
            underage = np.maximum(0, scenarios - q)
            costs = (
                self.ordering_cost * q +
                self.holding_cost * overage +
                self.stockout_cost * underage
            )
            cvar_term = tau + (1 / (n * (1 - self.cvar_beta))) * np.sum(
                np.maximum(0, costs - tau)
            )
            return cvar_term

        if self.use_cvar:
            # CVaR optimization
            q0 = np.mean(scenarios)
            tau0 = np.median(scenarios) * self.ordering_cost
            result = minimize(
                cvar_objective,
                [q0, tau0],
                method='L-BFGS-B',
                bounds=[(0, None), (None, None)]
            )
            q_optimal = max(0, result.x[0])
        else:
            # Expected cost optimization
            # Analytical solution: critical quantile
            critical_ratio = self.stockout_cost / (self.stockout_cost + self.holding_cost)
            q_optimal = np.quantile(scenarios, critical_ratio)
            q_optimal = max(0, q_optimal)

        # Prediction intervals from scenarios
        lower = np.quantile(scenarios, self.alpha / 2)
        upper = np.quantile(scenarios, 1 - self.alpha / 2)

        return q_optimal, lower, upper

    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Generate predictions with scenario-based prediction intervals.
        """
        self._check_is_fitted()

        rng = np.random.RandomState(self.random_state)
        point_pred = self.model.predict(X)

        n_samples = len(X)
        lower = np.zeros(n_samples)
        upper = np.zeros(n_samples)

        for i in range(n_samples):
            # Generate demand scenarios
            residual_samples = rng.choice(self.residuals, size=self.n_scenarios, replace=True)
            scenarios = point_pred[i] + residual_samples

            # Solve two-stage problem (just for bounds here)
            _, lower[i], upper[i] = self._solve_two_stage(point_pred[i], scenarios)

        return PredictionResult(point=point_pred, lower=lower, upper=upper)

    def compute_order_quantities(self, X: np.ndarray) -> np.ndarray:
        """
        Compute optimal order quantities using two-stage stochastic optimization.

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

        rng = np.random.RandomState(self.random_state)
        point_pred = self.model.predict(X)

        order_quantities = []
        for i in range(len(X)):
            # Generate demand scenarios
            residual_samples = rng.choice(self.residuals, size=self.n_scenarios, replace=True)
            scenarios = point_pred[i] + residual_samples

            # Solve two-stage problem
            q_optimal, _, _ = self._solve_two_stage(point_pred[i], scenarios)
            order_quantities.append(q_optimal)

        return np.array(order_quantities)

    def get_params(self) -> dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            "n_scenarios": self.n_scenarios,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "ordering_cost": self.ordering_cost,
            "holding_cost": self.holding_cost,
            "stockout_cost": self.stockout_cost,
            "use_cvar": self.use_cvar,
            "cvar_beta": self.cvar_beta,
            "sigma": self.sigma
        })
        return params


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


class EnsembleBatchPI(BaseForecaster):
    """
    Ensemble Batch Prediction Intervals with Conformalized Quantile Regression (EnbPI+CQR).

    This method combines three powerful techniques for uncertainty quantification:

    1. **Ensemble of Bootstrap Aggregated Learners (EnbPI)**:
       - Xu & Xie (2021) "Conformal prediction interval for dynamic time-series"
       - Uses multiple models trained on bootstrap samples
       - Computes leave-one-out style residuals to avoid overfitting

    2. **Conformalized Quantile Regression (CQR)**:
       - Romano et al. (2019) "Conformalized Quantile Regression"
       - Trains quantile models for adaptive interval width
       - Intervals adapt to local uncertainty (heteroscedasticity)

    3. **Conformal Calibration**:
       - Provides finite-sample coverage guarantees
       - Distribution-free: no parametric assumptions required

    The method:
    1. Train B bootstrap ensemble members (Random Forest base learners)
    2. For each calibration point, compute leave-one-out residuals using
       ensemble members that didn't include that point in training
    3. Train quantile regressors for adaptive lower/upper bounds
    4. Apply conformal calibration using the CQR conformity score:
       E_i = max(q_lo(X_i) - Y_i, Y_i - q_hi(X_i))
    5. Adjust intervals by the conformal quantile of E for coverage guarantee

    Key advantages over standard conformal prediction:
    - Adaptive intervals (narrower where uncertainty is low)
    - Better suited for time series (LOO residuals handle temporal structure)
    - Robust ensemble aggregation reduces variance

    References
    ----------
    - Xu, C., & Xie, Y. (2021). "Conformal prediction interval for dynamic
      time-series." ICML 2021.
    - Romano, Y., Patterson, E., & Candès, E. (2019). "Conformalized quantile
      regression." NeurIPS 2019.
    - Barber, R. F., et al. (2019). "Predictive inference with the jackknife+."
      Annals of Statistics.
    """

    def __init__(
        self,
        alpha: float = 0.05,
        n_ensemble: int = 10,
        n_estimators: int = 100,
        max_depth: int = 10,
        bootstrap_fraction: float = 0.8,
        use_quantile_regression: bool = True,
        random_state: int = 42
    ):
        """
        Initialize the EnbPI+CQR model.

        Parameters
        ----------
        alpha : float
            Significance level for prediction intervals (1-alpha coverage).
            Default is 0.05 for 95% coverage.
        n_ensemble : int
            Number of bootstrap ensemble members. More members = more robust
            but slower. Recommended: 10-50.
        n_estimators : int
            Number of trees per Random Forest ensemble member.
        max_depth : int
            Maximum depth of trees in the Random Forest.
        bootstrap_fraction : float
            Fraction of training data to use for each bootstrap sample.
            Default 0.8 ensures diversity while maintaining good training.
        use_quantile_regression : bool
            If True, use CQR-style adaptive intervals via quantile regression.
            If False, use symmetric intervals (standard EnbPI).
        random_state : int
            Random seed for reproducibility.
        """
        super().__init__(alpha=alpha, random_state=random_state)
        self.n_ensemble = n_ensemble
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.bootstrap_fraction = bootstrap_fraction
        self.use_quantile_regression = use_quantile_regression

        # Ensemble members (Random Forests)
        self.ensemble_models: List[RandomForestRegressor] = []

        # Bootstrap indices for each ensemble member (for LOO computation)
        self.bootstrap_indices: List[np.ndarray] = []

        # Quantile regressors for CQR (adaptive intervals)
        self.q_low = alpha / 2
        self.q_high = 1 - alpha / 2
        self.quantile_model_low: Optional[GradientBoostingRegressor] = None
        self.quantile_model_high: Optional[GradientBoostingRegressor] = None

        # Conformal calibration parameters
        self.conformal_adjustment: float = 0.0

        # Store calibration residuals for analysis
        self.loo_residuals: Optional[np.ndarray] = None

    def _create_bootstrap_samples(
        self,
        X: np.ndarray,
        y: np.ndarray,
        n_samples: int
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Create bootstrap samples with their indices.

        Parameters
        ----------
        X : np.ndarray
            Training features.
        y : np.ndarray
            Training targets.
        n_samples : int
            Number of bootstrap samples to create.

        Returns
        -------
        List[Tuple[np.ndarray, np.ndarray, np.ndarray]]
            List of (X_boot, y_boot, indices) tuples.
        """
        np.random.seed(self.random_state)
        n = len(y)
        sample_size = int(n * self.bootstrap_fraction)

        bootstrap_data = []
        for i in range(n_samples):
            # Sample with replacement
            indices = np.random.choice(n, size=sample_size, replace=True)
            X_boot = X[indices]
            y_boot = y[indices]
            bootstrap_data.append((X_boot, y_boot, indices))

        return bootstrap_data

    def _compute_loo_ensemble_predictions(
        self,
        X: np.ndarray,
        exclude_indices: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute leave-one-out style ensemble predictions.

        For each calibration point, average predictions only from ensemble
        members that did NOT include that point in their bootstrap sample.
        This prevents overfitting and provides valid residuals for conformal
        calibration.

        Parameters
        ----------
        X : np.ndarray
            Features to predict.
        exclude_indices : np.ndarray, optional
            Original indices in training data. If None, use all ensemble members.

        Returns
        -------
        Tuple[np.ndarray, np.ndarray]
            Mean predictions and standard deviations from the ensemble.
        """
        n_points = len(X)
        all_predictions = np.zeros((len(self.ensemble_models), n_points))

        # Get predictions from all ensemble members
        for i, model in enumerate(self.ensemble_models):
            all_predictions[i] = model.predict(X)

        if exclude_indices is None:
            # Use all ensemble members
            mean_pred = np.mean(all_predictions, axis=0)
            std_pred = np.std(all_predictions, axis=0)
            return mean_pred, std_pred

        # LOO-style: exclude models that included this point in training
        mean_pred = np.zeros(n_points)
        std_pred = np.zeros(n_points)

        for j in range(n_points):
            # Find ensemble members that did NOT include this point
            point_idx = exclude_indices[j]
            valid_members = []
            for i, boot_indices in enumerate(self.bootstrap_indices):
                if point_idx not in boot_indices:
                    valid_members.append(i)

            if len(valid_members) == 0:
                # Fallback: use all members if all included this point
                valid_members = list(range(len(self.ensemble_models)))

            # Average predictions from valid members only
            valid_preds = all_predictions[valid_members, j]
            mean_pred[j] = np.mean(valid_preds)
            std_pred[j] = np.std(valid_preds) if len(valid_preds) > 1 else 0.0

        return mean_pred, std_pred

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "EnsembleBatchPI":
        """
        Train the EnbPI+CQR model.

        Training proceeds in three phases:
        1. Bootstrap ensemble training
        2. Quantile regression training (for CQR)
        3. Conformal calibration

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
        """
        logger.info(f"Training EnbPI+CQR model with {self.n_ensemble} ensemble members...")

        # Phase 1: Train bootstrap ensemble
        logger.info("Phase 1: Training bootstrap ensemble...")
        bootstrap_data = self._create_bootstrap_samples(
            X_train, y_train, self.n_ensemble
        )

        self.ensemble_models = []
        self.bootstrap_indices = []

        for i, (X_boot, y_boot, indices) in enumerate(bootstrap_data):
            model = RandomForestRegressor(
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state + i,  # Different seed for diversity
                n_jobs=-1
            )
            model.fit(X_boot, y_boot)
            self.ensemble_models.append(model)
            self.bootstrap_indices.append(set(indices))  # Store as set for fast lookup

        logger.info(f"  Trained {len(self.ensemble_models)} ensemble members")

        # Compute ensemble predictions on calibration data
        # Use standard ensemble (no LOO) for quantile regression training
        cal_mean, cal_std = self._compute_loo_ensemble_predictions(X_cal)

        # Phase 2: Train quantile regressors for adaptive intervals (CQR)
        if self.use_quantile_regression:
            logger.info("Phase 2: Training quantile regressors for CQR...")

            # Combine original features with ensemble predictions
            X_cal_augmented = np.column_stack([X_cal, cal_mean, cal_std])
            X_train_mean, X_train_std = self._compute_loo_ensemble_predictions(X_train)
            X_train_augmented = np.column_stack([X_train, X_train_mean, X_train_std])

            # Train lower quantile model
            self.quantile_model_low = GradientBoostingRegressor(
                loss='quantile',
                alpha=self.q_low,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            self.quantile_model_low.fit(X_train_augmented, y_train)

            # Train upper quantile model
            self.quantile_model_high = GradientBoostingRegressor(
                loss='quantile',
                alpha=self.q_high,
                n_estimators=self.n_estimators,
                max_depth=self.max_depth,
                random_state=self.random_state
            )
            self.quantile_model_high.fit(X_train_augmented, y_train)

            logger.info(f"  Trained quantile models for q={self.q_low:.3f} and q={self.q_high:.3f}")

        # Phase 3: Conformal calibration
        logger.info("Phase 3: Computing conformal calibration...")

        if self.use_quantile_regression:
            # CQR conformity score: E_i = max(q_lo(X_i) - Y_i, Y_i - q_hi(X_i))
            X_cal_augmented = np.column_stack([X_cal, cal_mean, cal_std])
            cal_lower = self.quantile_model_low.predict(X_cal_augmented)
            cal_upper = self.quantile_model_high.predict(X_cal_augmented)

            # Compute CQR conformity scores
            lower_violations = cal_lower - y_cal  # Positive if lower > actual
            upper_violations = y_cal - cal_upper  # Positive if actual > upper
            conformity_scores = np.maximum(lower_violations, upper_violations)
        else:
            # Standard EnbPI: symmetric residuals
            conformity_scores = np.abs(y_cal - cal_mean)

        # Store residuals for analysis
        self.loo_residuals = conformity_scores

        # Conformal quantile with finite-sample correction
        n = len(y_cal)
        quantile_level = np.ceil((n + 1) * (1 - self.alpha)) / n
        self.conformal_adjustment = np.quantile(
            conformity_scores,
            min(quantile_level, 1.0)
        )

        logger.info(f"  Conformal adjustment: {self.conformal_adjustment:.4f}")
        logger.info(f"  Coverage target: {(1 - self.alpha) * 100:.1f}%")

        # Compute empirical coverage on calibration set (sanity check)
        if self.use_quantile_regression:
            final_lower = cal_lower - self.conformal_adjustment
            final_upper = cal_upper + self.conformal_adjustment
        else:
            final_lower = cal_mean - self.conformal_adjustment
            final_upper = cal_mean + self.conformal_adjustment

        cal_coverage = np.mean((y_cal >= final_lower) & (y_cal <= final_upper))
        logger.info(f"  Calibration set coverage: {cal_coverage * 100:.1f}%")

        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        Generate predictions with calibrated prediction intervals.

        Parameters
        ----------
        X : np.ndarray
            Features for prediction.

        Returns
        -------
        PredictionResult
            Point predictions with calibrated lower/upper bounds.
        """
        self._check_is_fitted()

        # Get ensemble predictions
        point_pred, pred_std = self._compute_loo_ensemble_predictions(X)

        if self.use_quantile_regression:
            # CQR: Use quantile models for adaptive intervals
            X_augmented = np.column_stack([X, point_pred, pred_std])
            lower = self.quantile_model_low.predict(X_augmented) - self.conformal_adjustment
            upper = self.quantile_model_high.predict(X_augmented) + self.conformal_adjustment
        else:
            # Standard EnbPI: Symmetric intervals
            lower = point_pred - self.conformal_adjustment
            upper = point_pred + self.conformal_adjustment

        return PredictionResult(point=point_pred, lower=lower, upper=upper)

    def get_params(self) -> dict:
        """Get model parameters."""
        params = super().get_params()
        params.update({
            "n_ensemble": self.n_ensemble,
            "n_estimators": self.n_estimators,
            "max_depth": self.max_depth,
            "bootstrap_fraction": self.bootstrap_fraction,
            "use_quantile_regression": self.use_quantile_regression,
            "conformal_adjustment": self.conformal_adjustment,
        })
        return params


class Seer(BaseForecaster):
    """
    Seer (Perfect Foresight) Oracle Baseline.

    This model represents the theoretical upper bound of performance by
    assuming perfect knowledge of future demand. It provides a benchmark
    to understand:
    1. Maximum achievable performance (theoretical limit)
    2. Value of better forecasting (gap between current models and Seer)
    3. Whether we're approaching the performance ceiling

    The Seer "predicts" by using actual demand values, then computes
    optimal order quantities. This gives the minimum possible cost
    achievable with perfect information.

    Note: This is NOT a practical model - it's a theoretical benchmark
    to measure how much room for improvement exists.

    Usage in experiments:
    - Shows the "ceiling" performance
    - Helps interpret if 5% improvement is significant or trivial
    - Validates that the problem has meaningful uncertainty
    """

    def __init__(self, alpha: float = 0.05, random_state: int = 42):
        super().__init__(alpha=alpha, random_state=random_state)
        self.y_actual = None  # Will store actual demand during prediction

    def fit(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        X_cal: np.ndarray,
        y_cal: np.ndarray
    ) -> "Seer":
        """
        'Fit' the Seer model (no actual training needed).

        The Seer doesn't learn from historical data - it magically
        knows future demand. This method exists for API compatibility.
        """
        logger.info("'Training' Seer (Perfect Foresight) Oracle...")
        logger.info("⚠️  Note: Seer has perfect knowledge of future demand")
        logger.info("⚠️  This is a theoretical upper bound, not a real model")
        self._is_fitted = True
        return self

    def predict(self, X: np.ndarray) -> PredictionResult:
        """
        'Predict' using actual demand values.

        This method requires the actual demand to be set via
        predict_with_actuals() before calling this method.

        Returns perfect predictions with zero-width intervals
        (since there's no uncertainty).
        """
        self._check_is_fitted()

        if self.y_actual is None:
            raise ValueError(
                "Seer requires actual demand values. "
                "Use predict_with_actuals(X, y_actual) instead."
            )

        # Perfect predictions = actual demand
        point_pred = self.y_actual.copy()

        # Zero-width intervals (perfect certainty)
        lower = point_pred.copy()
        upper = point_pred.copy()

        return PredictionResult(point=point_pred, lower=lower, upper=upper)

    def predict_with_actuals(
        self,
        X: np.ndarray,
        y_actual: np.ndarray
    ) -> PredictionResult:
        """
        'Predict' using actual demand values.

        Parameters
        ----------
        X : np.ndarray
            Features (not used, only for API compatibility).
        y_actual : np.ndarray
            Actual demand values (the "predictions").

        Returns
        -------
        PredictionResult
            Perfect predictions with zero-width intervals.
        """
        self._check_is_fitted()

        # Store actual demand
        self.y_actual = y_actual

        # Perfect predictions = actual demand
        point_pred = y_actual.copy()

        # Zero-width intervals (perfect certainty)
        lower = point_pred.copy()
        upper = point_pred.copy()

        return PredictionResult(point=point_pred, lower=lower, upper=upper)

    def compute_order_quantities(
        self,
        y_actual: np.ndarray,
        ordering_cost: float = 10.0,
        holding_cost: float = 2.0,
        stockout_cost: float = 50.0
    ) -> np.ndarray:
        """
        Compute optimal order quantities with perfect foresight.

        With perfect knowledge of demand, the optimal order quantity
        is exactly equal to the demand (to minimize holding and stockout
        costs while meeting all demand).

        Parameters
        ----------
        y_actual : np.ndarray
            Actual demand values.
        ordering_cost : float
            Cost per unit ordered (used for computing total cost).
        holding_cost : float
            Cost per unit of excess inventory.
        stockout_cost : float
            Cost per unit of shortage.

        Returns
        -------
        np.ndarray
            Optimal order quantities (= actual demand).
        """
        # With perfect foresight, order exactly the demand
        # This minimizes holding (0) and stockout (0) costs
        return y_actual.copy()
