"""
Configuration settings for the Inventory CVaR Optimization project.

This module centralizes all hyperparameters, cost settings, and model configurations.
"""

from dataclasses import dataclass, field
from typing import List, Optional
import torch


# =============================================================================
# COST PARAMETERS
# =============================================================================

@dataclass
class CostConfig:
    """Newsvendor cost parameters."""
    ordering_cost: float = 10.0
    holding_cost: float = 2.0
    stockout_cost: float = 50.0
    
    @property
    def critical_ratio(self) -> float:
        """Critical ratio for newsvendor problem: cu / (cu + co)"""
        return self.stockout_cost / (self.stockout_cost + self.holding_cost)


# =============================================================================
# DATA CONFIGURATION
# =============================================================================

@dataclass
class DataConfig:
    """Data loading and preprocessing settings."""
    filepath: str = "train.csv"
    store_ids: List[int] = field(default_factory=lambda: [1])
    item_ids: List[int] = field(default_factory=lambda: [1])

    # Temporal splits
    train_years: List[int] = field(default_factory=lambda: [2013, 2014])
    cal_years: List[int] = field(default_factory=lambda: [2015, 2016])
    test_years: List[int] = field(default_factory=lambda: [2017])

    # Feature engineering
    lag_features: List[int] = field(default_factory=lambda: [1, 7, 28])
    rolling_windows: List[int] = field(default_factory=lambda: [7, 28])

    # Sequence length for DL models
    sequence_length: int = 28

    # Prediction horizon (days ahead to predict)
    prediction_horizon: int = 30


# =============================================================================
# MODEL CONFIGURATIONS
# =============================================================================

@dataclass
class ConformalConfig:
    """Conformal Prediction settings."""
    alpha: float = 0.05  # 1 - alpha = coverage level
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42


@dataclass
class NormalConfig:
    """Normal Assumption model settings."""
    alpha: float = 0.05
    n_estimators: int = 100
    max_depth: int = 10
    random_state: int = 42


@dataclass
class QuantileRegressionConfig:
    """Quantile Regression settings."""
    alpha: float = 0.05
    n_estimators: int = 100
    max_depth: int = 8
    random_state: int = 42


@dataclass
class LSTMConfig:
    """LSTM Quantile Regression settings."""
    alpha: float = 0.05
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    quantiles: List[float] = field(default_factory=lambda: [0.025, 0.5, 0.975])


@dataclass
class TransformerConfig:
    """Transformer Quantile Regression settings."""
    alpha: float = 0.05
    d_model: int = 64
    nhead: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32


@dataclass
class DeepEnsembleConfig:
    """Deep Ensemble settings."""
    alpha: float = 0.05
    n_ensemble: int = 5
    hidden_size: int = 64
    learning_rate: float = 0.001
    epochs: int = 50
    batch_size: int = 32


@dataclass
class MCDropoutConfig:
    """MC Dropout LSTM settings."""
    alpha: float = 0.05
    hidden_size: int = 64
    num_layers: int = 2
    dropout: float = 0.2
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32
    n_mc_samples: int = 100


@dataclass
class TFTConfig:
    """Temporal Fusion Transformer settings."""
    alpha: float = 0.05
    hidden_size: int = 64
    num_heads: int = 4
    num_layers: int = 2
    dropout: float = 0.1
    learning_rate: float = 0.001
    epochs: int = 100
    batch_size: int = 32


# =============================================================================
# ROLLING WINDOW CONFIGURATION
# =============================================================================

@dataclass
class RollingWindowConfig:
    """Rolling window split settings."""
    enabled: bool = False  # Set to True to use rolling windows
    initial_train_days: int = 730  # 2 years
    calibration_days: int = 365     # 1 year
    test_window_days: int = 30      # 1 month prediction
    step_days: int = 30             # Roll forward by 1 month


# =============================================================================
# OPTIMIZATION CONFIGURATION
# =============================================================================

@dataclass
class CVaRConfig:
    """CVaR optimization settings."""
    beta: float = 0.90  # CVaR level (tail probability)
    n_samples: int = 1000  # Number of demand samples
    random_seed: int = 42


# =============================================================================
# EXPERIMENT CONFIGURATION
# =============================================================================

@dataclass
class ExperimentConfig:
    """Main experiment configuration."""
    # Sub-configurations
    cost: CostConfig = field(default_factory=CostConfig)
    data: DataConfig = field(default_factory=DataConfig)
    cvar: CVaRConfig = field(default_factory=CVaRConfig)
    rolling_window: RollingWindowConfig = field(default_factory=RollingWindowConfig)

    # Model configs
    conformal: ConformalConfig = field(default_factory=ConformalConfig)
    normal: NormalConfig = field(default_factory=NormalConfig)
    quantile_reg: QuantileRegressionConfig = field(default_factory=QuantileRegressionConfig)
    lstm: LSTMConfig = field(default_factory=LSTMConfig)
    transformer: TransformerConfig = field(default_factory=TransformerConfig)
    deep_ensemble: DeepEnsembleConfig = field(default_factory=DeepEnsembleConfig)
    mc_dropout: MCDropoutConfig = field(default_factory=MCDropoutConfig)
    tft: TFTConfig = field(default_factory=TFTConfig)

    # Experiment settings
    random_seed: int = 42
    device: str = field(default_factory=lambda: "cuda" if torch.cuda.is_available() else "cpu")

    # Output settings
    results_dir: str = "results"
    save_models: bool = True
    verbose: bool = True


# =============================================================================
# DEFAULT CONFIGURATION INSTANCE
# =============================================================================

def get_default_config() -> ExperimentConfig:
    """Get default experiment configuration."""
    return ExperimentConfig()


def get_multi_store_config() -> ExperimentConfig:
    """Get configuration for multi-store experiment."""
    config = ExperimentConfig()
    config.data.store_ids = list(range(1, 11))  # Stores 1-10
    config.data.item_ids = list(range(1, 51))   # Items 1-50
    return config
