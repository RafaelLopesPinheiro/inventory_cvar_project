#!/usr/bin/env python
"""
Comprehensive Multi-SKU Multi-Period Expanding Window Experiment

This script implements a rigorous comparison of 13 forecasting methods for
inventory optimization using expanding window cross-validation across
MULTIPLE store-item (SKU) combinations with MULTI-PERIOD evaluation.

MULTI-PERIOD FORECASTING APPROACH:
==================================
Instead of predicting only a single horizon (e.g., 30 days ahead), this
experiment evaluates models across multiple forecast horizons simultaneously:
- Day 1: Immediate next-day demand
- Day 7: Week-ahead demand
- Day 14: Two-week-ahead demand
- Day 21: Three-week-ahead demand
- Day 28: Month-ahead demand

This provides:
1. More robust evaluation across different planning horizons
2. Better understanding of model performance degradation over time
3. Joint optimization considering multiple future periods
4. Scientific rigor through multi-horizon cross-validation

Model Hierarchy (Simple -> Advanced -> Your Method -> DRO -> Oracle):
=====================================================================
1. Historical Quantile      - Naive empirical quantile baseline
2. Normal Assumption        - Parametric Gaussian assumption
3. Bootstrapped Newsvendor  - Resampling-based uncertainty quantification
4. SAA                      - Standard Operations Research benchmark
5. Two-Stage Stochastic     - Scenario-based optimization
6. Conformal Prediction     - Distribution-free intervals
7. Quantile Regression      - Direct quantile estimation with CQR
8. LSTM Quantile Loss       - Deep learning WITHOUT calibration
9. LSTM+Conformal           - Deep learning WITH conformal calibration
10. SPO                     - Decision-focused deep learning
11. EnbPI+CQR+CVaR          - Your contribution (ensemble + CQR + CVaR)
12. DRO                     - Distributionally Robust Optimization (Wasserstein)
13. Seer                    - Oracle upper bound (perfect foresight)

Key Experimental Design:
========================
- Multi-SKU: Runs across multiple store-item combinations
- Multi-Period: Evaluates across multiple forecast horizons (1, 7, 14, 21, 28 days)
- Expanding Window: Training set grows over time (not sliding)
- Calibration Set: Fixed size for conformal calibration
- Direct Strategy: Separate model trained for each horizon
- Joint Optimization: Order quantities optimized across all horizons
- Metrics: Per-horizon and aggregated Mean Cost, CVaR-90, CVaR-95, Coverage

References:
===========
- Taieb et al. (2012) "A review and comparison of strategies for multi-step ahead forecasting"
- Hyndman & Athanasopoulos (2021) "Forecasting: Principles and Practice" Ch. 13
- Vovk et al. (2005) "Algorithmic Learning in a Random World"
- Romano et al. (2019) "Conformalized Quantile Regression"
- Xu & Xie (2021) "Conformal prediction interval for dynamic time-series"

Usage:
    # Single SKU with multi-period evaluation (default)
    python run_comprehensive_expanding_window.py

    # Multiple SKUs
    python run_comprehensive_expanding_window.py --stores 1,2,3 --items 1,2,3,4,5

    # Custom horizons
    python run_comprehensive_expanding_window.py --horizons 1,7,14,28

    # Skip deep learning models for faster execution
    python run_comprehensive_expanding_window.py --stores 1,2,3 --items 1,2,3 --no-dl
"""

import argparse
import logging
import os
import sys
import time
import warnings
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

# Determine optimal number of workers for parallel processing
_NUM_WORKERS = min(multiprocessing.cpu_count(), 8)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    load_raw_data,
    filter_store_item,
    create_all_features,
    create_rolling_window_splits,
    prepare_rolling_sequence_data,
    RollingWindowSplit,
    # Multi-period data structures
    MultiPeriodDataSplit,
    MultiPeriodRollingWindowSplit,
    create_multi_period_rolling_window_splits,
)
from src.models import (
    # Traditional models (Simple -> Advanced)
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
    # Deep learning models
    LSTMQuantileLossOnly,
    LSTMQuantileRegression,
    SPOEndToEnd,
    PredictionResult,
    # Multi-period forecasting
    MultiPeriodForecaster,
    MultiPeriodPredictionResult,
    create_multi_period_forecaster,
)
from src.optimization import (
    compute_order_quantities_cvar,
    CostParameters,
    # Multi-period optimization
    compute_order_quantities_multi_period_cvar,
    compute_multi_period_metrics,
    MultiPeriodCostMetrics,
)
from src.evaluation import (
    compute_all_metrics,
    MethodResults,
    paired_t_test,
)
from configs import get_default_config, ExperimentConfig

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL DEFINITIONS WITH CATEGORIES
# =============================================================================

MODEL_CATEGORIES = {
    "1_Naive": ["HistoricalQuantile"],
    "2_Parametric": ["NormalAssumption"],
    "3_Resampling": ["BootstrappedNewsvendor"],
    "4_OR_Standard": ["SAA"],
    "5_Stochastic": ["TwoStageStochastic"],
    "6_DistributionFree": ["ConformalPrediction"],
    "7_DirectQuantile": ["QuantileRegression"],
    "8_DL_Uncalibrated": ["LSTMQuantileLoss"],
    "9_DL_Calibrated": ["LSTMConformal"],
    "10_DecisionFocused": ["SPO"],
    "11_YourContribution": ["EnbPI_CQR_CVaR"],
    "12_RobustOptimization": ["DRO"],
    "13_Oracle": ["Seer"],
}


def get_model_display_name(method_name: str) -> str:
    """Get a clean display name for a method."""
    display_names = {
        "HistoricalQuantile": "1. Historical Quantile",
        "NormalAssumption": "2. Normal Assumption",
        "BootstrappedNewsvendor": "3. Bootstrap Newsvendor",
        "SAA": "4. SAA",
        "TwoStageStochastic": "5. Two-Stage Stochastic",
        "ConformalPrediction": "6. Conformal Prediction",
        "QuantileRegression": "7. Quantile Regression",
        "LSTMQuantileLoss": "8. LSTM (Uncalibrated)",
        "LSTMConformal": "9. LSTM+Conformal",
        "SPO": "10. SPO",
        "EnbPI_CQR_CVaR": "11. EnbPI+CQR+CVaR",
        "DRO": "12. DRO (Wasserstein)",
        "Seer": "13. Seer (Oracle)",
    }
    return display_names.get(method_name, method_name)


# =============================================================================
# MULTI-SKU DATA LOADING
# =============================================================================

def load_expanding_window_data_multi_sku(
    filepath: str,
    store_ids: List[int],
    item_ids: List[int],
    lag_periods: List[int] = [1, 7, 28],
    rolling_windows: List[int] = [7, 28],
    initial_train_days: int = 730,
    calibration_days: int = 365,
    test_window_days: int = 30,
    step_days: int = 30,
    min_records: int = 365 * 3
) -> Dict[Tuple[int, int], List[RollingWindowSplit]]:
    """
    Load expanding window data for multiple store-item combinations.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    store_ids : List[int]
        List of store IDs.
    item_ids : List[int]
        List of item IDs.
    lag_periods : List[int]
        Lag periods for features.
    rolling_windows : List[int]
        Rolling window sizes.
    initial_train_days : int
        Initial training period in days.
    calibration_days : int
        Calibration period in days.
    test_window_days : int
        Test window size in days.
    step_days : int
        Step size for rolling windows.
    min_records : int
        Minimum records required for a store-item combo.

    Returns
    -------
    Dict[Tuple[int, int], List[RollingWindowSplit]]
        Dictionary mapping (store_id, item_id) to list of rolling window splits.
    """
    logger.info(f"Loading expanding window data for {len(store_ids)} stores x {len(item_ids)} items")

    # Load raw data once
    df_raw = load_raw_data(filepath)

    results = {}
    skipped = []

    total = len(store_ids) * len(item_ids)
    with tqdm(total=total, desc="Loading SKU data") as pbar:
        for store_id in store_ids:
            for item_id in item_ids:
                try:
                    # Filter for this store-item
                    df = filter_store_item(df_raw, store_id, item_id)

                    if len(df) < min_records:
                        skipped.append((store_id, item_id, f"insufficient data ({len(df)} < {min_records})"))
                        pbar.update(1)
                        continue

                    # Create features
                    df, feature_cols = create_all_features(
                        df,
                        lag_periods=lag_periods,
                        rolling_windows=rolling_windows
                    )

                    # Create expanding window splits
                    splits = create_rolling_window_splits(
                        df,
                        feature_cols,
                        initial_train_days=initial_train_days,
                        calibration_days=calibration_days,
                        test_window_days=test_window_days,
                        step_days=step_days
                    )

                    if len(splits) > 0:
                        results[(store_id, item_id)] = splits
                    else:
                        skipped.append((store_id, item_id, "no valid windows"))

                except Exception as e:
                    skipped.append((store_id, item_id, str(e)))

                pbar.update(1)

    logger.info(f"Successfully loaded {len(results)} store-item combinations")
    if skipped:
        logger.warning(f"Skipped {len(skipped)} combinations:")
        for store_id, item_id, reason in skipped[:5]:
            logger.warning(f"  Store {store_id}, Item {item_id}: {reason}")
        if len(skipped) > 5:
            logger.warning(f"  ... and {len(skipped) - 5} more")

    return results


def load_multi_period_expanding_window_data_multi_sku(
    filepath: str,
    store_ids: List[int],
    item_ids: List[int],
    horizons: List[int] = [1, 7, 14, 21, 28],
    lag_periods: List[int] = [1, 7, 28],
    rolling_windows: List[int] = [7, 28],
    initial_train_days: int = 730,
    calibration_days: int = 365,
    test_window_days: int = 28,
    step_days: int = 28,
    min_records: int = 365 * 3
) -> Dict[Tuple[int, int], List[MultiPeriodRollingWindowSplit]]:
    """
    Load multi-period expanding window data for multiple store-item combinations.

    Parameters
    ----------
    filepath : str
        Path to the CSV file.
    store_ids : List[int]
        List of store IDs.
    item_ids : List[int]
        List of item IDs.
    horizons : List[int]
        List of forecast horizons (days ahead).
    lag_periods : List[int]
        Lag periods for features.
    rolling_windows : List[int]
        Rolling window sizes.
    initial_train_days : int
        Initial training period in days.
    calibration_days : int
        Calibration period in days.
    test_window_days : int
        Test window size in days (should be >= max horizon).
    step_days : int
        Step size for rolling windows.
    min_records : int
        Minimum records required for a store-item combo.

    Returns
    -------
    Dict[Tuple[int, int], List[MultiPeriodRollingWindowSplit]]
        Dictionary mapping (store_id, item_id) to list of multi-period rolling window splits.
    """
    logger.info(f"Loading multi-period expanding window data for {len(store_ids)} stores x {len(item_ids)} items")
    logger.info(f"  Horizons: {horizons}")

    # Load raw data once
    df_raw = load_raw_data(filepath)

    results = {}
    skipped = []

    total = len(store_ids) * len(item_ids)
    with tqdm(total=total, desc="Loading multi-period SKU data") as pbar:
        for store_id in store_ids:
            for item_id in item_ids:
                try:
                    # Filter for this store-item
                    df = filter_store_item(df_raw, store_id, item_id)

                    if len(df) < min_records:
                        skipped.append((store_id, item_id, f"insufficient data ({len(df)} < {min_records})"))
                        pbar.update(1)
                        continue

                    # Create features
                    df, feature_cols = create_all_features(
                        df,
                        lag_periods=lag_periods,
                        rolling_windows=rolling_windows
                    )

                    # Create multi-period expanding window splits
                    splits = create_multi_period_rolling_window_splits(
                        df,
                        feature_cols,
                        horizons=horizons,
                        initial_train_days=initial_train_days,
                        calibration_days=calibration_days,
                        test_window_days=test_window_days,
                        step_days=step_days
                    )

                    if len(splits) > 0:
                        results[(store_id, item_id)] = splits
                    else:
                        skipped.append((store_id, item_id, "no valid windows"))

                except Exception as e:
                    skipped.append((store_id, item_id, str(e)))

                pbar.update(1)

    logger.info(f"Successfully loaded {len(results)} store-item combinations")
    if skipped:
        logger.warning(f"Skipped {len(skipped)} combinations:")
        for store_id, item_id, reason in skipped[:5]:
            logger.warning(f"  Store {store_id}, Item {item_id}: {reason}")
        if len(skipped) > 5:
            logger.warning(f"  ... and {len(skipped) - 5} more")

    return results


# =============================================================================
# MULTI-PERIOD EXPERIMENT RUNNER
# =============================================================================

def run_multi_period_single_window(
    mp_window_split: MultiPeriodRollingWindowSplit,
    config: ExperimentConfig,
    run_dl_models: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run all models on a single multi-period expanding window.

    Uses the direct forecasting strategy: trains separate models for each horizon.

    Parameters
    ----------
    mp_window_split : MultiPeriodRollingWindowSplit
        Multi-period window split data.
    config : ExperimentConfig
        Experiment configuration.
    run_dl_models : bool
        If True, run deep learning models (slower but complete).

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Summary dataframe and detailed results dictionary.
    """
    results = {}
    timings = {}
    costs = config.cost
    horizons = mp_window_split.horizons

    # Data for traditional methods
    X_train = mp_window_split.train.X
    y_train = mp_window_split.train.y_horizons
    X_cal = mp_window_split.calibration.X
    y_cal = mp_window_split.calibration.y_horizons
    X_test = mp_window_split.test.X
    y_test = mp_window_split.test.y_horizons

    # =========================================================================
    # TRADITIONAL MODELS WITH MULTI-PERIOD FORECASTING
    # =========================================================================

    # List of traditional models to run with multi-period wrapper
    traditional_models = [
        ("HistoricalQuantile", HistoricalQuantile, {"alpha": config.conformal.alpha, "random_state": config.random_seed}),
        ("NormalAssumption", NormalAssumption, {"alpha": config.normal.alpha, "n_estimators": config.normal.n_estimators, "max_depth": config.normal.max_depth, "random_state": config.random_seed}),
        ("BootstrappedNewsvendor", BootstrappedNewsvendor, {"alpha": config.conformal.alpha, "n_bootstrap": 1000, "n_estimators": config.conformal.n_estimators, "max_depth": config.conformal.max_depth, "random_state": config.random_seed}),
        ("SAA", SampleAverageApproximation, {"n_estimators": 100, "max_depth": 10, "stockout_cost": costs.stockout_cost, "holding_cost": costs.holding_cost, "random_state": config.random_seed}),
        ("TwoStageStochastic", TwoStageStochastic, {"alpha": config.conformal.alpha, "n_scenarios": 500, "n_estimators": config.conformal.n_estimators, "max_depth": config.conformal.max_depth, "ordering_cost": costs.ordering_cost, "holding_cost": costs.holding_cost, "stockout_cost": costs.stockout_cost, "use_cvar": True, "cvar_beta": config.cvar.beta, "random_state": config.random_seed}),
        ("ConformalPrediction", ConformalPrediction, {"alpha": config.conformal.alpha, "n_estimators": config.conformal.n_estimators, "max_depth": config.conformal.max_depth, "random_state": config.random_seed}),
        ("QuantileRegression", QuantileRegression, {"alpha": config.quantile_reg.alpha, "n_estimators": config.quantile_reg.n_estimators, "max_depth": config.quantile_reg.max_depth, "random_state": config.random_seed}),
        ("EnbPI_CQR_CVaR", EnsembleBatchPI, {"alpha": config.ensemble_batch_pi.alpha, "n_ensemble": config.ensemble_batch_pi.n_ensemble, "n_estimators": config.ensemble_batch_pi.n_estimators, "max_depth": config.ensemble_batch_pi.max_depth, "bootstrap_fraction": config.ensemble_batch_pi.bootstrap_fraction, "use_quantile_regression": config.ensemble_batch_pi.use_quantile_regression, "random_state": config.random_seed}),
        ("DRO", DistributionallyRobustOptimization, {"alpha": config.conformal.alpha, "epsilon": 0.1, "n_estimators": config.conformal.n_estimators, "max_depth": config.conformal.max_depth, "n_scenarios": 500, "ordering_cost": costs.ordering_cost, "holding_cost": costs.holding_cost, "stockout_cost": costs.stockout_cost, "cvar_beta": config.cvar.beta, "adaptive_epsilon": True, "random_state": config.random_seed}),
    ]

    for model_name, model_class, model_kwargs in traditional_models:
        try:
            start_time = time.time()

            # Create multi-period forecaster
            mp_forecaster = MultiPeriodForecaster(
                base_model_class=model_class,
                horizons=horizons,
                **model_kwargs
            )

            # Fit on all horizons
            mp_forecaster.fit(X_train, y_train, X_cal, y_cal)

            # Predict for all horizons
            mp_pred = mp_forecaster.predict(X_test)

            # Extract predictions as dicts for optimization
            point_pred = {h: mp_pred.get_horizon(h).point for h in horizons}
            lower_pred = {h: mp_pred.get_horizon(h).lower if mp_pred.get_horizon(h).has_intervals else mp_pred.get_horizon(h).point * 0.8 for h in horizons}
            upper_pred = {h: mp_pred.get_horizon(h).upper if mp_pred.get_horizon(h).has_intervals else mp_pred.get_horizon(h).point * 1.2 for h in horizons}

            # Multi-period CVaR optimization
            order_quantities = compute_order_quantities_multi_period_cvar(
                point_pred, lower_pred, upper_pred,
                horizons=horizons,
                beta=config.cvar.beta,
                n_samples=config.cvar.n_samples,
                ordering_cost=costs.ordering_cost,
                holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost,
                aggregation=config.multi_period.aggregation,
                joint_optimization=config.multi_period.joint_optimization,
                random_seed=config.cvar.random_seed,
                verbose=False
            )

            # Compute multi-period metrics
            mp_metrics = compute_multi_period_metrics(
                order_quantities, y_test, horizons,
                ordering_cost=costs.ordering_cost,
                holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost,
                aggregation=config.multi_period.aggregation
            )

            timings[model_name] = time.time() - start_time
            results[model_name] = {
                'predictions': mp_pred,
                'order_quantities': order_quantities,
                'metrics': mp_metrics,
            }

        except Exception as e:
            logger.debug(f"{model_name} failed: {e}")

    # =========================================================================
    # SEER (ORACLE - UPPER BOUND) - Multi-period
    # =========================================================================
    try:
        start_time = time.time()

        # Seer has perfect knowledge, so we compute optimal orders directly
        seer_orders = {}
        seer_predictions = {}

        for h in horizons:
            seer_model = Seer(alpha=0.05, random_state=config.random_seed)
            seer_model.fit(X_train, y_train[h], X_cal, y_cal[h])
            seer_pred = seer_model.predict_with_actuals(X_test, y_test[h])
            seer_orders[h] = seer_model.compute_order_quantities(
                y_test[h],
                ordering_cost=costs.ordering_cost,
                holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost
            )
            seer_predictions[h] = seer_pred

        seer_metrics = compute_multi_period_metrics(
            seer_orders, y_test, horizons,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost,
            aggregation=config.multi_period.aggregation
        )

        timings["Seer"] = time.time() - start_time
        results["Seer"] = {
            'predictions': seer_predictions,
            'order_quantities': seer_orders,
            'metrics': seer_metrics,
        }

    except Exception as e:
        logger.debug(f"Seer failed: {e}")

    # =========================================================================
    # CREATE SUMMARY DATAFRAME
    # =========================================================================
    summary_data = []
    for method_name, result_data in results.items():
        metrics = result_data['metrics']

        # Aggregated metrics
        row = {
            'Method': method_name,
            'DisplayName': get_model_display_name(method_name),
            'Mean_Cost': metrics.aggregated_mean_cost,
            'CVaR_90': metrics.aggregated_cvar_90,
            'CVaR_95': metrics.aggregated_cvar_95,
            'Service_Level': metrics.aggregated_service_level,
            'Time_Seconds': timings.get(method_name, np.nan),
        }

        # Per-horizon metrics
        for h in horizons:
            row[f'Mean_Cost_h{h}'] = metrics.horizon_mean_costs.get(h, np.nan)
            row[f'CVaR_90_h{h}'] = metrics.horizon_cvar_90.get(h, np.nan)
            row[f'Service_Level_h{h}'] = metrics.horizon_service_levels.get(h, np.nan)

        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df['window_idx'] = mp_window_split.window_idx
    summary_df['test_start'] = mp_window_split.test_start_date
    summary_df['test_end'] = mp_window_split.test_end_date
    summary_df['horizons'] = str(horizons)

    return summary_df, results


def run_multi_period_single_sku(
    sku_splits: List[MultiPeriodRollingWindowSplit],
    store_id: int,
    item_id: int,
    config: ExperimentConfig,
    run_dl_models: bool = False,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run multi-period expanding window experiment for a single SKU.

    Parameters
    ----------
    sku_splits : List[MultiPeriodRollingWindowSplit]
        List of multi-period expanding window splits for this SKU.
    store_id : int
        Store ID.
    item_id : int
        Item ID.
    config : ExperimentConfig
        Experiment configuration.
    run_dl_models : bool
        Whether to run deep learning models.
    verbose : bool
        Whether to print detailed logs.

    Returns
    -------
    pd.DataFrame
        Results for all windows for this SKU.
    """
    all_window_results = []

    for mp_window_split in sku_splits:
        if verbose:
            logger.info(f"  Window {mp_window_split.window_idx}: {mp_window_split.test_start_date.date()} to {mp_window_split.test_end_date.date()}")

        summary_df, _ = run_multi_period_single_window(mp_window_split, config, run_dl_models)
        summary_df['store_id'] = store_id
        summary_df['item_id'] = item_id
        all_window_results.append(summary_df)

    return pd.concat(all_window_results, ignore_index=True)


# =============================================================================
# SINGLE-PERIOD EXPERIMENT RUNNER (BACKWARD COMPATIBLE)
# =============================================================================

def run_single_window(
    window_split: RollingWindowSplit,
    config: ExperimentConfig,
    run_dl_models: bool = True
) -> pd.DataFrame:
    """
    Run all 12 models on a single expanding window.

    Parameters
    ----------
    window_split : RollingWindowSplit
        Window split data.
    config : ExperimentConfig
        Experiment configuration.
    run_dl_models : bool
        If True, run deep learning models (slower but complete).

    Returns
    -------
    pd.DataFrame
        Results summary for this window.
    """
    results = {}
    timings = {}  # Store execution time for each model (in seconds)
    costs = config.cost

    # Data for traditional methods
    X_train, y_train = window_split.train.X, window_split.train.y
    X_cal, y_cal = window_split.calibration.X, window_split.calibration.y
    X_test, y_test = window_split.test.X, window_split.test.y

    # Prepare sequence data for DL models
    if run_dl_models:
        seq_data = prepare_rolling_sequence_data(
            window_split,
            seq_length=config.data.sequence_length,
            prediction_horizon=config.data.prediction_horizon
        )
        X_train_seq, y_train_seq = seq_data.X_train, seq_data.y_train
        X_cal_seq, y_cal_seq = seq_data.X_cal, seq_data.y_cal
        X_test_seq, y_test_seq = seq_data.X_test, seq_data.y_test
        n_test_seq = len(y_test_seq)
    else:
        n_test_seq = len(y_test)

    # =========================================================================
    # 1. HISTORICAL QUANTILE (NAIVE BASELINE)
    # =========================================================================
    try:
        start_time = time.time()
        hq_model = HistoricalQuantile(alpha=config.conformal.alpha, random_state=config.random_seed)
        hq_model.fit(X_train, y_train, X_cal, y_cal)
        hq_pred = hq_model.predict(X_test)
        hq_orders = compute_order_quantities_cvar(
            hq_pred.point, hq_pred.lower, hq_pred.upper,
            beta=config.cvar.beta, n_samples=config.cvar.n_samples,
            ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
            verbose=False
        )
        timings["HistoricalQuantile"] = time.time() - start_time
        results["HistoricalQuantile"] = compute_all_metrics(
            "HistoricalQuantile", y_test, hq_pred.point, hq_orders,
            hq_pred.lower, hq_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.debug(f"Historical Quantile failed: {e}")

    # =========================================================================
    # 2. NORMAL ASSUMPTION (PARAMETRIC)
    # =========================================================================
    try:
        start_time = time.time()
        normal_model = NormalAssumption(
            alpha=config.normal.alpha,
            n_estimators=config.normal.n_estimators,
            max_depth=config.normal.max_depth,
            random_state=config.random_seed
        )
        normal_model.fit(X_train, y_train, X_cal, y_cal)
        normal_pred = normal_model.predict(X_test)
        normal_orders = compute_order_quantities_cvar(
            normal_pred.point, normal_pred.lower, normal_pred.upper,
            beta=config.cvar.beta, n_samples=config.cvar.n_samples,
            ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
            verbose=False
        )
        timings["NormalAssumption"] = time.time() - start_time
        results["NormalAssumption"] = compute_all_metrics(
            "NormalAssumption", y_test, normal_pred.point, normal_orders,
            normal_pred.lower, normal_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.debug(f"Normal Assumption failed: {e}")

    # =========================================================================
    # 3. BOOTSTRAPPED NEWSVENDOR (RESAMPLING)
    # =========================================================================
    try:
        start_time = time.time()
        boot_model = BootstrappedNewsvendor(
            alpha=config.conformal.alpha,
            n_bootstrap=1000,
            n_estimators=config.conformal.n_estimators,
            max_depth=config.conformal.max_depth,
            random_state=config.random_seed
        )
        boot_model.fit(X_train, y_train, X_cal, y_cal)
        boot_pred = boot_model.predict(X_test)
        boot_orders = compute_order_quantities_cvar(
            boot_pred.point, boot_pred.lower, boot_pred.upper,
            beta=config.cvar.beta, n_samples=config.cvar.n_samples,
            ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
            verbose=False
        )
        timings["BootstrappedNewsvendor"] = time.time() - start_time
        results["BootstrappedNewsvendor"] = compute_all_metrics(
            "BootstrappedNewsvendor", y_test, boot_pred.point, boot_orders,
            boot_pred.lower, boot_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.debug(f"Bootstrapped Newsvendor failed: {e}")

    # =========================================================================
    # 4. SAA (STANDARD OR BENCHMARK)
    # =========================================================================
    try:
        start_time = time.time()
        saa_model = SampleAverageApproximation(
            n_estimators=100,
            max_depth=10,
            stockout_cost=costs.stockout_cost,
            holding_cost=costs.holding_cost,
            random_state=config.random_seed
        )
        saa_model.fit(X_train, y_train, X_cal, y_cal)
        saa_pred = saa_model.predict(X_test)
        saa_orders = saa_model.compute_order_quantities(X_test)
        timings["SAA"] = time.time() - start_time
        results["SAA"] = compute_all_metrics(
            "SAA", y_test, saa_pred.point, saa_orders,
            None, None,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.debug(f"SAA failed: {e}")

    # =========================================================================
    # 5. TWO-STAGE STOCHASTIC (SCENARIO OPTIMIZATION)
    # =========================================================================
    try:
        start_time = time.time()
        tss_model = TwoStageStochastic(
            alpha=config.conformal.alpha,
            n_scenarios=500,
            n_estimators=config.conformal.n_estimators,
            max_depth=config.conformal.max_depth,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost,
            use_cvar=True,
            cvar_beta=config.cvar.beta,
            random_state=config.random_seed
        )
        tss_model.fit(X_train, y_train, X_cal, y_cal)
        tss_pred = tss_model.predict(X_test)
        tss_orders = tss_model.compute_order_quantities(X_test)
        timings["TwoStageStochastic"] = time.time() - start_time
        results["TwoStageStochastic"] = compute_all_metrics(
            "TwoStageStochastic", y_test, tss_pred.point, tss_orders,
            tss_pred.lower, tss_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.debug(f"Two-Stage Stochastic failed: {e}")

    # =========================================================================
    # 6. CONFORMAL PREDICTION (DISTRIBUTION-FREE)
    # =========================================================================
    try:
        start_time = time.time()
        cp_model = ConformalPrediction(
            alpha=config.conformal.alpha,
            n_estimators=config.conformal.n_estimators,
            max_depth=config.conformal.max_depth,
            random_state=config.random_seed
        )
        cp_model.fit(X_train, y_train, X_cal, y_cal)
        cp_pred = cp_model.predict(X_test)
        cp_orders = compute_order_quantities_cvar(
            cp_pred.point, cp_pred.lower, cp_pred.upper,
            beta=config.cvar.beta, n_samples=config.cvar.n_samples,
            ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
            verbose=False
        )
        timings["ConformalPrediction"] = time.time() - start_time
        results["ConformalPrediction"] = compute_all_metrics(
            "ConformalPrediction", y_test, cp_pred.point, cp_orders,
            cp_pred.lower, cp_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.debug(f"Conformal Prediction failed: {e}")

    # =========================================================================
    # 7. QUANTILE REGRESSION (DIRECT QUANTILE + CQR)
    # =========================================================================
    try:
        start_time = time.time()
        qr_model = QuantileRegression(
            alpha=config.quantile_reg.alpha,
            n_estimators=config.quantile_reg.n_estimators,
            max_depth=config.quantile_reg.max_depth,
            random_state=config.random_seed
        )
        qr_model.fit(X_train, y_train, X_cal, y_cal)
        qr_pred = qr_model.predict(X_test)
        qr_orders = compute_order_quantities_cvar(
            qr_pred.point, qr_pred.lower, qr_pred.upper,
            beta=config.cvar.beta, n_samples=config.cvar.n_samples,
            ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
            verbose=False
        )
        timings["QuantileRegression"] = time.time() - start_time
        results["QuantileRegression"] = compute_all_metrics(
            "QuantileRegression", y_test, qr_pred.point, qr_orders,
            qr_pred.lower, qr_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.debug(f"Quantile Regression failed: {e}")

    # =========================================================================
    # 8-10. DEEP LEARNING MODELS (Optional)
    # =========================================================================
    if run_dl_models:
        # 8. LSTM QUANTILE LOSS (WITHOUT CALIBRATION)
        try:
            start_time = time.time()
            lstm_uncal_model = LSTMQuantileLossOnly(
                alpha=config.lstm.alpha,
                sequence_length=config.data.sequence_length,
                hidden_size=config.lstm.hidden_size,
                num_layers=config.lstm.num_layers,
                dropout=config.lstm.dropout,
                learning_rate=config.lstm.learning_rate,
                epochs=config.lstm.epochs,
                batch_size=config.lstm.batch_size,
                random_state=config.random_seed,
                device=config.device
            )
            lstm_uncal_model.fit(X_train_seq, y_train_seq, X_cal_seq, y_cal_seq)
            lstm_uncal_pred = lstm_uncal_model.predict(X_test_seq)
            lstm_uncal_orders = compute_order_quantities_cvar(
                lstm_uncal_pred.point, lstm_uncal_pred.lower, lstm_uncal_pred.upper,
                beta=config.cvar.beta, n_samples=config.cvar.n_samples,
                ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
                verbose=False
            )
            timings["LSTMQuantileLoss"] = time.time() - start_time
            results["LSTMQuantileLoss"] = compute_all_metrics(
                "LSTMQuantileLoss", y_test_seq, lstm_uncal_pred.point, lstm_uncal_orders,
                lstm_uncal_pred.lower, lstm_uncal_pred.upper,
                costs.ordering_cost, costs.holding_cost, costs.stockout_cost
            )
        except Exception as e:
            logger.debug(f"LSTM Quantile Loss failed: {e}")

        # 9. LSTM + CONFORMAL (WITH CALIBRATION)
        try:
            start_time = time.time()
            lstm_cal_model = LSTMQuantileRegression(
                alpha=config.lstm.alpha,
                sequence_length=config.data.sequence_length,
                hidden_size=config.lstm.hidden_size,
                num_layers=config.lstm.num_layers,
                dropout=config.lstm.dropout,
                learning_rate=config.lstm.learning_rate,
                epochs=config.lstm.epochs,
                batch_size=config.lstm.batch_size,
                random_state=config.random_seed,
                device=config.device
            )
            lstm_cal_model.fit(X_train_seq, y_train_seq, X_cal_seq, y_cal_seq)
            lstm_cal_pred = lstm_cal_model.predict(X_test_seq)
            lstm_cal_orders = compute_order_quantities_cvar(
                lstm_cal_pred.point, lstm_cal_pred.lower, lstm_cal_pred.upper,
                beta=config.cvar.beta, n_samples=config.cvar.n_samples,
                ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
                verbose=False
            )
            timings["LSTMConformal"] = time.time() - start_time
            results["LSTMConformal"] = compute_all_metrics(
                "LSTMConformal", y_test_seq, lstm_cal_pred.point, lstm_cal_orders,
                lstm_cal_pred.lower, lstm_cal_pred.upper,
                costs.ordering_cost, costs.holding_cost, costs.stockout_cost
            )
        except Exception as e:
            logger.debug(f"LSTM+Conformal failed: {e}")

        # 10. SPO (DECISION-FOCUSED LEARNING)
        try:
            start_time = time.time()
            spo_model = SPOEndToEnd(
                alpha=config.lstm.alpha,
                sequence_length=config.data.sequence_length,
                hidden_size=config.lstm.hidden_size,
                num_layers=config.lstm.num_layers,
                dropout=config.lstm.dropout,
                learning_rate=config.lstm.learning_rate,
                epochs=config.lstm.epochs,
                batch_size=config.lstm.batch_size,
                ordering_cost=costs.ordering_cost,
                holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost,
                beta=config.cvar.beta,
                random_state=config.random_seed,
                device=config.device
            )
            spo_model.fit(X_train_seq, y_train_seq, X_cal_seq, y_cal_seq)
            spo_pred = spo_model.predict(X_test_seq)
            spo_orders = compute_order_quantities_cvar(
                spo_pred.point, spo_pred.lower, spo_pred.upper,
                beta=config.cvar.beta, n_samples=config.cvar.n_samples,
                ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
                verbose=False
            )
            timings["SPO"] = time.time() - start_time
            results["SPO"] = compute_all_metrics(
                "SPO", y_test_seq, spo_pred.point, spo_orders,
                spo_pred.lower, spo_pred.upper,
                costs.ordering_cost, costs.holding_cost, costs.stockout_cost
            )
        except Exception as e:
            logger.debug(f"SPO failed: {e}")

    # =========================================================================
    # 11. EnbPI + CQR + CVaR (YOUR CONTRIBUTION)
    # =========================================================================
    try:
        start_time = time.time()
        enbpi_model = EnsembleBatchPI(
            alpha=config.ensemble_batch_pi.alpha,
            n_ensemble=config.ensemble_batch_pi.n_ensemble,
            n_estimators=config.ensemble_batch_pi.n_estimators,
            max_depth=config.ensemble_batch_pi.max_depth,
            bootstrap_fraction=config.ensemble_batch_pi.bootstrap_fraction,
            use_quantile_regression=config.ensemble_batch_pi.use_quantile_regression,
            random_state=config.random_seed
        )
        enbpi_model.fit(X_train, y_train, X_cal, y_cal)
        enbpi_pred = enbpi_model.predict(X_test)
        enbpi_orders = compute_order_quantities_cvar(
            enbpi_pred.point, enbpi_pred.lower, enbpi_pred.upper,
            beta=config.cvar.beta, n_samples=config.cvar.n_samples,
            ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
            verbose=False
        )
        timings["EnbPI_CQR_CVaR"] = time.time() - start_time
        results["EnbPI_CQR_CVaR"] = compute_all_metrics(
            "EnbPI_CQR_CVaR", y_test, enbpi_pred.point, enbpi_orders,
            enbpi_pred.lower, enbpi_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.debug(f"EnbPI+CQR+CVaR failed: {e}")

    # =========================================================================
    # 12. DRO (DISTRIBUTIONALLY ROBUST OPTIMIZATION)
    # =========================================================================
    try:
        start_time = time.time()
        dro_model = DistributionallyRobustOptimization(
            alpha=config.conformal.alpha,
            epsilon=0.1,  # Wasserstein ball radius
            n_estimators=config.conformal.n_estimators,
            max_depth=config.conformal.max_depth,
            n_scenarios=500,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost,
            cvar_beta=config.cvar.beta,
            adaptive_epsilon=True,
            random_state=config.random_seed
        )
        dro_model.fit(X_train, y_train, X_cal, y_cal)
        dro_pred = dro_model.predict(X_test)
        dro_orders = dro_model.compute_order_quantities(
            X_test, dro_pred.point, dro_pred.lower, dro_pred.upper
        )
        timings["DRO"] = time.time() - start_time
        results["DRO"] = compute_all_metrics(
            "DRO", y_test, dro_pred.point, dro_orders,
            dro_pred.lower, dro_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.debug(f"DRO failed: {e}")

    # =========================================================================
    # 13. SEER (ORACLE - UPPER BOUND)
    # =========================================================================
    try:
        start_time = time.time()
        seer_model = Seer(alpha=0.05, random_state=config.random_seed)
        seer_model.fit(X_train, y_train, X_cal, y_cal)
        seer_pred = seer_model.predict_with_actuals(X_test, y_test)
        seer_orders = seer_model.compute_order_quantities(
            y_test,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost
        )
        timings["Seer"] = time.time() - start_time
        results["Seer"] = compute_all_metrics(
            "Seer", y_test, seer_pred.point, seer_orders,
            seer_pred.lower, seer_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.debug(f"Seer failed: {e}")

    # =========================================================================
    # ALIGN RESULTS TO COMMON LENGTH
    # =========================================================================
    if run_dl_models:
        # Align traditional methods to sequence-aligned test length
        for name in list(results.keys()):
            if name not in ["LSTMQuantileLoss", "LSTMConformal", "SPO"]:
                result = results[name]
                results[name] = MethodResults(
                    method_name=name,
                    forecast_metrics=result.forecast_metrics,
                    inventory_metrics=result.inventory_metrics,
                    costs=result.costs[-n_test_seq:] if len(result.costs) >= n_test_seq else result.costs,
                    order_quantities=result.order_quantities[-n_test_seq:] if len(result.order_quantities) >= n_test_seq else result.order_quantities,
                    point_predictions=result.point_predictions[-n_test_seq:] if len(result.point_predictions) >= n_test_seq else result.point_predictions,
                    lower_bounds=result.lower_bounds[-n_test_seq:] if result.lower_bounds is not None and len(result.lower_bounds) >= n_test_seq else result.lower_bounds,
                    upper_bounds=result.upper_bounds[-n_test_seq:] if result.upper_bounds is not None and len(result.upper_bounds) >= n_test_seq else result.upper_bounds
                )

    # =========================================================================
    # CREATE SUMMARY DATAFRAME
    # =========================================================================
    summary_data = []
    for method_name, result in results.items():
        row = {
            'Method': method_name,
            'DisplayName': get_model_display_name(method_name),
            'Mean_Cost': result.inventory_metrics.mean_cost,
            'CVaR_90': result.inventory_metrics.cvar_90,
            'CVaR_95': result.inventory_metrics.cvar_95,
            'Service_Level': result.inventory_metrics.service_level,
            'Coverage': result.forecast_metrics.coverage if result.forecast_metrics.coverage is not None else np.nan,
            'Interval_Width': result.forecast_metrics.avg_interval_width if result.forecast_metrics.avg_interval_width is not None else np.nan,
            'MAE': result.forecast_metrics.mae,
            'RMSE': result.forecast_metrics.rmse,
            'MAPE': result.forecast_metrics.mape,
            'Time_Seconds': timings.get(method_name, np.nan),
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df['window_idx'] = window_split.window_idx
    summary_df['test_start'] = window_split.test_start_date
    summary_df['test_end'] = window_split.test_end_date

    return summary_df, results


def run_single_sku(
    sku_splits: List[RollingWindowSplit],
    store_id: int,
    item_id: int,
    config: ExperimentConfig,
    run_dl_models: bool = True,
    verbose: bool = False
) -> pd.DataFrame:
    """
    Run expanding window experiment for a single SKU.

    Parameters
    ----------
    sku_splits : List[RollingWindowSplit]
        List of expanding window splits for this SKU.
    store_id : int
        Store ID.
    item_id : int
        Item ID.
    config : ExperimentConfig
        Experiment configuration.
    run_dl_models : bool
        Whether to run deep learning models.
    verbose : bool
        Whether to print detailed logs.

    Returns
    -------
    pd.DataFrame
        Results for all windows for this SKU.
    """
    all_window_results = []

    for window_split in sku_splits:
        if verbose:
            logger.info(f"  Window {window_split.window_idx}: {window_split.test_start_date.date()} to {window_split.test_end_date.date()}")

        summary_df, _ = run_single_window(window_split, config, run_dl_models)
        summary_df['store_id'] = store_id
        summary_df['item_id'] = item_id
        all_window_results.append(summary_df)

    return pd.concat(all_window_results, ignore_index=True)


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_comprehensive_visualizations(
    combined_df: pd.DataFrame,
    aggregated: pd.DataFrame,
    output_dir: str,
    multi_sku: bool = False
):
    """
    Create comprehensive visualizations of the experiment results.

    Parameters
    ----------
    combined_df : pd.DataFrame
        All window results combined.
    aggregated : pd.DataFrame
        Aggregated results across windows.
    output_dir : str
        Directory to save plots.
    multi_sku : bool
        Whether this is a multi-SKU experiment.
    """
    logger.info("\nCreating visualizations...")

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Define model order (Simple -> Advanced -> Your Method -> DRO -> Oracle)
    model_order = [
        'HistoricalQuantile', 'NormalAssumption', 'BootstrappedNewsvendor',
        'SAA', 'TwoStageStochastic', 'ConformalPrediction', 'QuantileRegression',
        'LSTMQuantileLoss', 'LSTMConformal', 'SPO', 'EnbPI_CQR_CVaR', 'DRO', 'Seer'
    ]

    # Filter to existing methods
    existing_methods = [m for m in model_order if m in combined_df['Method'].unique()]

    # 1. CVaR-90 Comparison (Main Metric)
    fig, ax = plt.subplots(figsize=(14, 6))
    df_plot = combined_df[combined_df['Method'].isin(existing_methods)].copy()
    df_plot['Method'] = pd.Categorical(df_plot['Method'], categories=existing_methods, ordered=True)

    sns.boxplot(data=df_plot, x='Method', y='CVaR_90', ax=ax)
    title = 'CVaR-90 Comparison Across All Windows'
    if multi_sku:
        title += f' and {combined_df["store_id"].nunique()} Stores x {combined_df["item_id"].nunique()} Items'
    ax.set_title(f'{title}\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('CVaR-90 ($)', fontsize=12)
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cvar90_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 2. Mean Cost vs Coverage Trade-off
    fig, ax = plt.subplots(figsize=(12, 8))
    agg_flat = aggregated.reset_index()

    # Get mean values
    mean_cost = agg_flat.set_index('Method')[('Mean_Cost', 'mean')]
    coverage = agg_flat.set_index('Method')[('Coverage', 'mean')]

    for method in existing_methods:
        if method in mean_cost.index and method in coverage.index:
            mc = mean_cost[method]
            cov = coverage[method]
            if not np.isnan(cov):
                marker = '*' if method == 'EnbPI_CQR_CVaR' else 'o'
                color = 'red' if method == 'EnbPI_CQR_CVaR' else 'green' if method == 'Seer' else 'blue'
                size = 200 if method in ['EnbPI_CQR_CVaR', 'Seer'] else 100
                ax.scatter(cov * 100, mc, s=size, c=color, alpha=0.7, label=get_model_display_name(method))
                ax.annotate(method.replace('_', '\n'), (cov * 100, mc), fontsize=8, ha='center', va='bottom')

    ax.axhline(y=mean_cost.get('Seer', 0), color='green', linestyle='--', alpha=0.5, label='Oracle Lower Bound')
    ax.axvline(x=95, color='gray', linestyle='--', alpha=0.5, label='95% Target Coverage')
    ax.set_xlabel('Coverage (%)', fontsize=12)
    ax.set_ylabel('Mean Cost ($)', fontsize=12)
    ax.set_title('Mean Cost vs Coverage Trade-off', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_coverage_tradeoff.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 3. Method Ranking Heatmap
    fig, ax = plt.subplots(figsize=(10, 8))

    # Create ranking matrix
    metrics = ['Mean_Cost', 'CVaR_90', 'CVaR_95', 'MAE', 'RMSE']
    rank_data = []

    for method in existing_methods:
        if method in agg_flat['Method'].values:
            row = {'Method': method}
            for metric in metrics:
                try:
                    val = agg_flat[agg_flat['Method'] == method][(metric, 'mean')].values[0]
                    row[metric] = val
                except (KeyError, IndexError):
                    row[metric] = np.nan
            rank_data.append(row)

    rank_df = pd.DataFrame(rank_data).set_index('Method')

    # Compute ranks (lower is better for all these metrics)
    rank_matrix = rank_df.rank()

    sns.heatmap(rank_matrix, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax)
    ax.set_title('Method Rankings Across Metrics\n(1 = Best)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Metric', fontsize=12)
    ax.set_ylabel('Method', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_rankings.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 4. Performance Progression Across Windows
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    for idx, metric in enumerate(['Mean_Cost', 'CVaR_90', 'Coverage', 'Service_Level']):
        ax = axes[idx // 2, idx % 2]
        for method in ['EnbPI_CQR_CVaR', 'DRO', 'ConformalPrediction', 'NormalAssumption', 'SAA']:
            if method in combined_df['Method'].values:
                # Aggregate across SKUs for each window
                method_data = combined_df[combined_df['Method'] == method].groupby('window_idx')[metric].mean().reset_index()
                if metric in method_data.columns:
                    ax.plot(method_data['window_idx'], method_data[metric],
                           marker='o', label=get_model_display_name(method), linewidth=2, markersize=6)

        ax.set_xlabel('Window Index', fontsize=10)
        ax.set_ylabel(metric.replace('_', ' '), fontsize=10)
        ax.set_title(f'{metric.replace("_", " ")} Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=8, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Performance Progression Across Expanding Windows', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_progression.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 5. Summary Bar Chart
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))

    metrics_to_plot = ['Mean_Cost', 'CVaR_90', 'Coverage']
    titles = ['Mean Cost (Lower=Better)', 'CVaR-90 (Lower=Better)', 'Coverage (Higher=Better, Target=95%)']

    for idx, (metric, title) in enumerate(zip(metrics_to_plot, titles)):
        ax = axes[idx]
        values = []
        errors = []
        colors = []

        for method in existing_methods:
            if method in agg_flat['Method'].values:
                try:
                    mean_val = agg_flat[agg_flat['Method'] == method][(metric, 'mean')].values[0]
                    std_val = agg_flat[agg_flat['Method'] == method][(metric, 'std')].values[0]
                    values.append(mean_val * 100 if metric == 'Coverage' else mean_val)
                    errors.append(std_val * 100 if metric == 'Coverage' else std_val)
                    colors.append('red' if method == 'EnbPI_CQR_CVaR' else 'green' if method == 'Seer' else 'steelblue')
                except (KeyError, IndexError):
                    values.append(np.nan)
                    errors.append(0)
                    colors.append('gray')

        x_pos = range(len(existing_methods))
        bars = ax.bar(x_pos, values, yerr=errors, color=colors, alpha=0.8, capsize=3)
        ax.set_xticks(x_pos)
        ax.set_xticklabels([m.replace('_', '\n') for m in existing_methods], rotation=45, ha='right', fontsize=8)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_ylabel(metric.replace('_', ' '))

        if metric == 'Coverage':
            ax.axhline(y=95, color='gray', linestyle='--', alpha=0.7, label='95% Target')

    plt.suptitle('Summary of Key Metrics Across All Methods', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'summary_bar_chart.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # 6. Multi-SKU specific visualizations
    if multi_sku and 'store_id' in combined_df.columns:
        # Heatmap of CVaR-90 by Store-Item for best method
        fig, ax = plt.subplots(figsize=(12, 8))

        # Get EnbPI results or fall back to best performing method
        best_method = 'EnbPI_CQR_CVaR' if 'EnbPI_CQR_CVaR' in existing_methods else existing_methods[0]
        method_df = combined_df[combined_df['Method'] == best_method]

        pivot_data = method_df.groupby(['store_id', 'item_id'])['CVaR_90'].mean().unstack()

        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax)
        ax.set_title(f'CVaR-90 by Store-Item ({best_method})', fontsize=14, fontweight='bold')
        ax.set_xlabel('Item ID', fontsize=12)
        ax.set_ylabel('Store ID', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cvar90_by_sku.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # Performance variance across SKUs
        fig, ax = plt.subplots(figsize=(14, 6))

        sku_variance = combined_df.groupby(['Method', 'store_id', 'item_id'])['CVaR_90'].mean().reset_index()
        sku_variance = sku_variance[sku_variance['Method'].isin(existing_methods)]
        sku_variance['Method'] = pd.Categorical(sku_variance['Method'], categories=existing_methods, ordered=True)

        sns.boxplot(data=sku_variance, x='Method', y='CVaR_90', ax=ax)
        ax.set_title('CVaR-90 Variance Across SKUs\n(Each box = distribution over store-item combos)', fontsize=14, fontweight='bold')
        ax.set_xlabel('Method', fontsize=12)
        ax.set_ylabel('CVaR-90 ($)', fontsize=12)
        ax.tick_params(axis='x', rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cvar90_variance_by_sku.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # 7. Timing Comparison
    if 'Time_Seconds' in combined_df.columns:
        fig, ax = plt.subplots(figsize=(14, 6))

        # Get mean timing for each method
        timing_df = combined_df.groupby('Method')['Time_Seconds'].mean().reset_index()
        timing_df = timing_df[timing_df['Method'].isin(existing_methods)]
        timing_df['Method'] = pd.Categorical(timing_df['Method'], categories=existing_methods, ordered=True)
        timing_df = timing_df.sort_values('Method')

        colors = ['red' if m == 'EnbPI_CQR_CVaR' else 'purple' if m == 'DRO' else
                  'green' if m == 'Seer' else 'orange' if 'LSTM' in m or m == 'SPO' else 'steelblue'
                  for m in timing_df['Method']]

        bars = ax.bar(range(len(timing_df)), timing_df['Time_Seconds'], color=colors, alpha=0.8)
        ax.set_xticks(range(len(timing_df)))
        ax.set_xticklabels([m.replace('_', '\n') for m in timing_df['Method']], rotation=45, ha='right', fontsize=9)
        ax.set_ylabel('Time (seconds)', fontsize=12)
        ax.set_title('Average Execution Time per Window (Training + Prediction + Optimization)',
                    fontsize=14, fontweight='bold')

        # Add value labels on bars
        for bar, val in zip(bars, timing_df['Time_Seconds']):
            if not np.isnan(val):
                ax.annotate(f'{val:.1f}s', xy=(bar.get_x() + bar.get_width()/2, bar.get_height()),
                           ha='center', va='bottom', fontsize=8)

        ax.set_yscale('log')  # Log scale for better visibility of small times
        ax.set_ylabel('Time (seconds, log scale)', fontsize=12)
        ax.grid(True, alpha=0.3, axis='y')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'timing_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

        # 8. Timing vs CVaR-90 Trade-off
        fig, ax = plt.subplots(figsize=(12, 8))

        # Merge timing with CVaR data
        cvar_timing = combined_df.groupby('Method').agg({
            'Time_Seconds': 'mean',
            'CVaR_90': 'mean'
        }).reset_index()
        cvar_timing = cvar_timing[cvar_timing['Method'].isin(existing_methods)]

        for _, row in cvar_timing.iterrows():
            method = row['Method']
            if np.isnan(row['Time_Seconds']) or np.isnan(row['CVaR_90']):
                continue
            marker = '*' if method == 'EnbPI_CQR_CVaR' else 's' if method == 'DRO' else 'o'
            color = ('red' if method == 'EnbPI_CQR_CVaR' else 'purple' if method == 'DRO' else
                     'green' if method == 'Seer' else 'orange' if 'LSTM' in method or method == 'SPO' else 'blue')
            size = 200 if method in ['EnbPI_CQR_CVaR', 'DRO', 'Seer'] else 100
            ax.scatter(row['Time_Seconds'], row['CVaR_90'], s=size, c=color, alpha=0.7, marker=marker)
            ax.annotate(method.replace('_', '\n'), (row['Time_Seconds'], row['CVaR_90']),
                       fontsize=8, ha='left', va='bottom')

        ax.set_xlabel('Execution Time (seconds)', fontsize=12)
        ax.set_ylabel('CVaR-90 ($)', fontsize=12)
        ax.set_title('Execution Time vs CVaR-90 Trade-off\n(Bottom-left is better)', fontsize=14, fontweight='bold')
        ax.set_xscale('log')
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'timing_vs_cvar_tradeoff.png'), dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


def create_summary_report(
    combined_df: pd.DataFrame,
    aggregated: pd.DataFrame,
    output_dir: str,
    store_ids: List[int],
    item_ids: List[int]
) -> str:
    """
    Create a text summary report of the experiment results.

    Parameters
    ----------
    combined_df : pd.DataFrame
        All window results combined.
    aggregated : pd.DataFrame
        Aggregated results across windows.
    output_dir : str
        Directory to save the report.
    store_ids : List[int]
        List of store IDs used.
    item_ids : List[int]
        List of item IDs used.

    Returns
    -------
    str
        The report content.
    """
    multi_sku = len(store_ids) > 1 or len(item_ids) > 1

    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE EXPANDING WINDOW EXPERIMENT REPORT")
    if multi_sku:
        report.append("Multi-SKU Analysis")
    report.append("Simple -> Advanced -> Your Method")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Stores: {store_ids}")
    report.append(f"Items: {item_ids}")
    report.append(f"Total SKU Combinations: {len(store_ids) * len(item_ids)}")
    report.append(f"Number of Windows: {combined_df['window_idx'].nunique()}")
    report.append(f"Number of Methods: {combined_df['Method'].nunique()}")
    report.append(f"Total Observations: {len(combined_df)}")

    # Best performers
    report.append("\n" + "-" * 80)
    report.append("KEY FINDINGS")
    report.append("-" * 80)

    agg_flat = aggregated.reset_index()

    # CVaR-90 ranking (excluding Seer)
    cvar90_mean = agg_flat.set_index('Method')[('CVaR_90', 'mean')]
    cvar90_no_oracle = cvar90_mean.drop('Seer', errors='ignore')
    best_cvar = cvar90_no_oracle.idxmin()
    report.append(f"\n[BEST CVaR-90] {get_model_display_name(best_cvar)}: ${cvar90_no_oracle[best_cvar]:.2f}")

    # Mean Cost ranking
    mean_cost = agg_flat.set_index('Method')[('Mean_Cost', 'mean')]
    mean_cost_no_oracle = mean_cost.drop('Seer', errors='ignore')
    best_cost = mean_cost_no_oracle.idxmin()
    report.append(f"[BEST Cost]    {get_model_display_name(best_cost)}: ${mean_cost_no_oracle[best_cost]:.2f}")

    # Coverage ranking
    coverage = agg_flat.set_index('Method')[('Coverage', 'mean')]
    best_coverage = coverage.idxmax()
    report.append(f"[BEST Coverage] {get_model_display_name(best_coverage)}: {coverage[best_coverage]*100:.1f}%")

    # Oracle performance
    if 'Seer' in mean_cost.index:
        report.append(f"\n[ORACLE (Seer)] Mean Cost: ${mean_cost['Seer']:.2f} (theoretical lower bound)")

    # Your contribution performance
    if 'EnbPI_CQR_CVaR' in cvar90_mean.index:
        report.append(f"\n[YOUR METHOD: EnbPI+CQR+CVaR]")
        report.append(f"  - CVaR-90: ${cvar90_mean['EnbPI_CQR_CVaR']:.2f}")
        report.append(f"  - Mean Cost: ${mean_cost['EnbPI_CQR_CVaR']:.2f}")
        report.append(f"  - Coverage: {coverage['EnbPI_CQR_CVaR']*100:.1f}%")

        # Compare to baselines
        improvement_vs_saa = (cvar90_mean['SAA'] - cvar90_mean['EnbPI_CQR_CVaR']) / cvar90_mean['SAA'] * 100 if 'SAA' in cvar90_mean.index else 0
        improvement_vs_normal = (cvar90_mean['NormalAssumption'] - cvar90_mean['EnbPI_CQR_CVaR']) / cvar90_mean['NormalAssumption'] * 100 if 'NormalAssumption' in cvar90_mean.index else 0

        report.append(f"  - CVaR-90 improvement vs SAA: {improvement_vs_saa:+.1f}%")
        report.append(f"  - CVaR-90 improvement vs Normal: {improvement_vs_normal:+.1f}%")

    # Multi-SKU specific statistics
    if multi_sku and 'store_id' in combined_df.columns:
        report.append("\n" + "-" * 80)
        report.append("MULTI-SKU ANALYSIS")
        report.append("-" * 80)

        # Per-SKU best methods
        sku_best = combined_df.groupby(['store_id', 'item_id', 'Method'])['CVaR_90'].mean().reset_index()
        sku_best = sku_best[sku_best['Method'] != 'Seer']  # Exclude oracle
        sku_winners = sku_best.loc[sku_best.groupby(['store_id', 'item_id'])['CVaR_90'].idxmin()]

        method_wins = sku_winners['Method'].value_counts()
        report.append(f"\nMethod wins across SKUs (by CVaR-90):")
        for method, wins in method_wins.items():
            pct = wins / len(sku_winners) * 100
            report.append(f"  {get_model_display_name(method)}: {wins} wins ({pct:.1f}%)")

        # Variance analysis
        report.append(f"\nCVaR-90 standard deviation across SKUs:")
        for method in ['EnbPI_CQR_CVaR', 'DRO', 'ConformalPrediction', 'NormalAssumption', 'SAA']:
            if method in combined_df['Method'].values:
                method_df = combined_df[combined_df['Method'] == method]
                sku_means = method_df.groupby(['store_id', 'item_id'])['CVaR_90'].mean()
                report.append(f"  {get_model_display_name(method)}: {sku_means.std():.2f}")

    # Full results table
    report.append("\n" + "-" * 80)
    report.append("AGGREGATED RESULTS (Mean +/- Std across all windows and SKUs)")
    report.append("-" * 80)
    report.append("\n" + aggregated.to_string())

    # Timing analysis
    if 'Time_Seconds' in combined_df.columns:
        report.append("\n" + "-" * 80)
        report.append("EXECUTION TIME ANALYSIS")
        report.append("-" * 80)

        timing_stats = combined_df.groupby('Method')['Time_Seconds'].agg(['mean', 'std', 'min', 'max'])
        timing_stats = timing_stats.sort_values('mean')

        report.append("\nAverage Execution Time per Window (sorted by speed):")
        report.append(f"{'Method':<25} {'Mean (s)':<12} {'Std (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
        report.append("-" * 73)

        for method, row in timing_stats.iterrows():
            if not np.isnan(row['mean']):
                report.append(f"{method:<25} {row['mean']:>10.2f}s {row['std']:>10.2f}s {row['min']:>10.2f}s {row['max']:>10.2f}s")

        # Speedup comparisons
        report.append("\nSpeedup Comparison (relative to EnbPI+CQR+CVaR):")
        if 'EnbPI_CQR_CVaR' in timing_stats.index:
            enbpi_time = timing_stats.loc['EnbPI_CQR_CVaR', 'mean']
            for method in timing_stats.index:
                if method != 'EnbPI_CQR_CVaR' and not np.isnan(timing_stats.loc[method, 'mean']):
                    method_time = timing_stats.loc[method, 'mean']
                    if method_time > 0:
                        speedup = enbpi_time / method_time
                        if speedup >= 1:
                            report.append(f"  {method}: {speedup:.1f}x slower than EnbPI")
                        else:
                            report.append(f"  {method}: {1/speedup:.1f}x faster than EnbPI")

        # DRO vs EnbPI timing
        if 'DRO' in timing_stats.index and 'EnbPI_CQR_CVaR' in timing_stats.index:
            dro_time = timing_stats.loc['DRO', 'mean']
            enbpi_time = timing_stats.loc['EnbPI_CQR_CVaR', 'mean']
            report.append(f"\n[DRO vs EnbPI]")
            report.append(f"  DRO execution time: {dro_time:.2f}s")
            report.append(f"  EnbPI execution time: {enbpi_time:.2f}s")
            if dro_time > enbpi_time:
                report.append(f"  DRO is {dro_time/enbpi_time:.1f}x slower than EnbPI")
            else:
                report.append(f"  DRO is {enbpi_time/dro_time:.1f}x faster than EnbPI")

    # Save report
    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, "experiment_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    logger.info(f"Report saved to {report_path}")
    return report_text


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_id_range(id_string: str) -> List[int]:
    """
    Parse a string of IDs that can be comma-separated or a range.

    Examples:
        "1,2,3" -> [1, 2, 3]
        "1-5" -> [1, 2, 3, 4, 5]
        "1,3,5-8" -> [1, 3, 5, 6, 7, 8]
    """
    ids = []
    for part in id_string.split(','):
        part = part.strip()
        if '-' in part:
            start, end = part.split('-')
            ids.extend(range(int(start), int(end) + 1))
        else:
            ids.append(int(part))
    return sorted(list(set(ids)))


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(
    config: ExperimentConfig,
    store_ids: List[int],
    item_ids: List[int],
    run_dl_models: bool = True,
    max_windows: Optional[int] = None
):
    """
    Main experiment runner for multi-SKU expanding window experiment.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    store_ids : List[int]
        List of store IDs to process.
    item_ids : List[int]
        List of item IDs to process.
    run_dl_models : bool
        If True, run deep learning models.
    max_windows : Optional[int]
        Maximum number of windows per SKU (for testing).
    """
    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

    multi_sku = len(store_ids) > 1 or len(item_ids) > 1

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE EXPANDING WINDOW CVaR OPTIMIZATION EXPERIMENT")
    if multi_sku:
        logger.info("MULTI-SKU MODE")
    logger.info("Simple -> Advanced -> Your Method (EnbPI+CQR+CVaR)")
    logger.info("=" * 80)
    logger.info(f"Device: {config.device}")
    logger.info(f"Stores: {store_ids}")
    logger.info(f"Items: {item_ids}")
    logger.info(f"Total SKUs: {len(store_ids) * len(item_ids)}")
    logger.info(f"Run DL Models: {run_dl_models}")

    # Create output directory
    os.makedirs(config.results_dir, exist_ok=True)

    # Load expanding window data for all SKUs
    logger.info("\nLoading expanding window data...")
    logger.info(f"  Initial train: {config.rolling_window.initial_train_days} days")
    logger.info(f"  Calibration: {config.rolling_window.calibration_days} days")
    logger.info(f"  Test window: {config.rolling_window.test_window_days} days")
    logger.info(f"  Step: {config.rolling_window.step_days} days")

    all_sku_splits = load_expanding_window_data_multi_sku(
        filepath=config.data.filepath,
        store_ids=store_ids,
        item_ids=item_ids,
        lag_periods=config.data.lag_features,
        rolling_windows=config.data.rolling_windows,
        initial_train_days=config.rolling_window.initial_train_days,
        calibration_days=config.rolling_window.calibration_days,
        test_window_days=config.rolling_window.test_window_days,
        step_days=config.rolling_window.step_days
    )

    if not all_sku_splits:
        logger.error("No valid store-item combinations found! Check your data.")
        return

    # Limit windows if specified (for testing)
    if max_windows is not None:
        for key in all_sku_splits:
            all_sku_splits[key] = all_sku_splits[key][:max_windows]

    total_windows = sum(len(splits) for splits in all_sku_splits.values())
    logger.info(f"\nTotal SKUs loaded: {len(all_sku_splits)}")
    logger.info(f"Total windows to process: {total_windows}")

    # Determine parallel execution strategy
    # For DL models, use sequential to avoid GPU contention
    # For traditional models only, use parallel SKU processing
    use_parallel_skus = not run_dl_models and len(all_sku_splits) > 1
    n_parallel_skus = min(_NUM_WORKERS, len(all_sku_splits)) if use_parallel_skus else 1

    logger.info(f"Parallel SKU processing: {use_parallel_skus} ({n_parallel_skus} workers)")

    # Run experiment on all SKUs
    all_results = []

    if use_parallel_skus:
        # Parallel SKU processing for traditional models only
        def process_sku(args):
            (store_id, item_id), sku_splits = args
            return run_single_sku(
                sku_splits, store_id, item_id, config, run_dl_models, verbose=False
            )

        sku_items = list(all_sku_splits.items())

        with ThreadPoolExecutor(max_workers=n_parallel_skus) as executor:
            futures = {executor.submit(process_sku, item): item for item in sku_items}

            with tqdm(total=len(futures), desc="Processing SKUs (parallel)") as pbar:
                for future in as_completed(futures):
                    try:
                        sku_results = future.result()
                        all_results.append(sku_results)
                    except Exception as e:
                        logger.error(f"Error processing SKU: {e}")
                    pbar.update(1)
    else:
        # Sequential processing (for DL models or single SKU)
        with tqdm(total=len(all_sku_splits), desc="Processing SKUs") as pbar:
            for (store_id, item_id), sku_splits in all_sku_splits.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing Store {store_id}, Item {item_id} ({len(sku_splits)} windows)")
                logger.info(f"{'='*60}")

                sku_results = run_single_sku(
                    sku_splits, store_id, item_id, config, run_dl_models, verbose=False
                )
                all_results.append(sku_results)

                pbar.update(1)

    # Combine all results
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATED RESULTS ACROSS ALL WINDOWS AND SKUs")
    logger.info("=" * 80)

    combined_df = pd.concat(all_results, ignore_index=True)

    # Compute mean and std across all windows and SKUs
    aggregated = combined_df.groupby('Method').agg({
        'Mean_Cost': ['mean', 'std'],
        'CVaR_90': ['mean', 'std'],
        'CVaR_95': ['mean', 'std'],
        'Service_Level': ['mean', 'std'],
        'Coverage': ['mean', 'std'],
        'Interval_Width': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'MAPE': ['mean', 'std']
    }).round(2)

    print("\n", aggregated.to_string())

    # Save results
    agg_path = os.path.join(config.results_dir, "comprehensive_aggregated.csv")
    aggregated.to_csv(agg_path)
    logger.info(f"\n[OK] Saved aggregated results: {agg_path}")

    all_path = os.path.join(config.results_dir, "comprehensive_all_windows.csv")
    combined_df.to_csv(all_path, index=False)
    logger.info(f"[OK] Saved all window results: {all_path}")

    # Save per-SKU aggregated results
    if multi_sku:
        sku_agg = combined_df.groupby(['store_id', 'item_id', 'Method']).agg({
            'Mean_Cost': ['mean', 'std'],
            'CVaR_90': ['mean', 'std'],
            'CVaR_95': ['mean', 'std'],
            'Coverage': ['mean', 'std'],
            'Service_Level': ['mean', 'std'],
        }).round(2)
        sku_agg_path = os.path.join(config.results_dir, "comprehensive_by_sku.csv")
        sku_agg.to_csv(sku_agg_path)
        logger.info(f"[OK] Saved per-SKU results: {sku_agg_path}")

    # Create visualizations
    create_comprehensive_visualizations(combined_df, aggregated, config.results_dir, multi_sku)

    # Create summary report
    report = create_summary_report(combined_df, aggregated, config.results_dir, store_ids, item_ids)
    print("\n" + report)

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)


def main_multi_period(
    config: ExperimentConfig,
    store_ids: List[int],
    item_ids: List[int],
    horizons: List[int] = [1, 7, 14, 21, 28],
    run_dl_models: bool = False,
    max_windows: Optional[int] = None
):
    """
    Main experiment runner for multi-period multi-SKU expanding window experiment.

    This is the recommended scientific approach that evaluates models across
    multiple forecast horizons for more robust conclusions.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    store_ids : List[int]
        List of store IDs to process.
    item_ids : List[int]
        List of item IDs to process.
    horizons : List[int]
        List of forecast horizons (days ahead).
    run_dl_models : bool
        If True, run deep learning models.
    max_windows : Optional[int]
        Maximum number of windows per SKU (for testing).
    """
    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

    multi_sku = len(store_ids) > 1 or len(item_ids) > 1

    logger.info("=" * 80)
    logger.info("MULTI-PERIOD COMPREHENSIVE EXPANDING WINDOW CVaR OPTIMIZATION EXPERIMENT")
    if multi_sku:
        logger.info("MULTI-SKU MODE")
    logger.info("Direct Strategy: Separate models trained for each horizon")
    logger.info("Joint Optimization: Orders optimized across all horizons")
    logger.info("=" * 80)
    logger.info(f"Horizons: {horizons}")
    logger.info(f"Device: {config.device}")
    logger.info(f"Stores: {store_ids}")
    logger.info(f"Items: {item_ids}")
    logger.info(f"Total SKUs: {len(store_ids) * len(item_ids)}")
    logger.info(f"Run DL Models: {run_dl_models}")
    logger.info(f"Multi-period aggregation: {config.multi_period.aggregation}")
    logger.info(f"Joint optimization: {config.multi_period.joint_optimization}")

    # Create output directory
    os.makedirs(config.results_dir, exist_ok=True)

    # Load multi-period expanding window data for all SKUs
    logger.info("\nLoading multi-period expanding window data...")
    logger.info(f"  Horizons: {horizons}")
    logger.info(f"  Initial train: {config.rolling_window.initial_train_days} days")
    logger.info(f"  Calibration: {config.rolling_window.calibration_days} days")
    logger.info(f"  Test window: {max(horizons)} days (aligned with max horizon)")
    logger.info(f"  Step: {max(horizons)} days")

    all_sku_splits = load_multi_period_expanding_window_data_multi_sku(
        filepath=config.data.filepath,
        store_ids=store_ids,
        item_ids=item_ids,
        horizons=horizons,
        lag_periods=config.data.lag_features,
        rolling_windows=config.data.rolling_windows,
        initial_train_days=config.rolling_window.initial_train_days,
        calibration_days=config.rolling_window.calibration_days,
        test_window_days=max(horizons),
        step_days=max(horizons)
    )

    if not all_sku_splits:
        logger.error("No valid store-item combinations found! Check your data.")
        return

    # Limit windows if specified (for testing)
    if max_windows is not None:
        for key in all_sku_splits:
            all_sku_splits[key] = all_sku_splits[key][:max_windows]

    total_windows = sum(len(splits) for splits in all_sku_splits.values())
    logger.info(f"\nTotal SKUs loaded: {len(all_sku_splits)}")
    logger.info(f"Total windows to process: {total_windows}")

    # Determine parallel execution strategy
    use_parallel_skus = not run_dl_models and len(all_sku_splits) > 1
    n_parallel_skus = min(_NUM_WORKERS, len(all_sku_splits)) if use_parallel_skus else 1

    logger.info(f"Parallel SKU processing: {use_parallel_skus} ({n_parallel_skus} workers)")

    # Run multi-period experiment on all SKUs
    all_results = []

    if use_parallel_skus:
        # Parallel SKU processing for traditional models only
        def process_sku_mp(args):
            (store_id, item_id), sku_splits = args
            return run_multi_period_single_sku(
                sku_splits, store_id, item_id, config, run_dl_models, verbose=False
            )

        sku_items = list(all_sku_splits.items())

        with ThreadPoolExecutor(max_workers=n_parallel_skus) as executor:
            futures = {executor.submit(process_sku_mp, item): item for item in sku_items}

            with tqdm(total=len(futures), desc="Processing SKUs (multi-period, parallel)") as pbar:
                for future in as_completed(futures):
                    try:
                        sku_results = future.result()
                        all_results.append(sku_results)
                    except Exception as e:
                        logger.error(f"Error processing SKU: {e}")
                    pbar.update(1)
    else:
        # Sequential processing
        with tqdm(total=len(all_sku_splits), desc="Processing SKUs (multi-period)") as pbar:
            for (store_id, item_id), sku_splits in all_sku_splits.items():
                logger.info(f"\n{'='*60}")
                logger.info(f"Processing Store {store_id}, Item {item_id} ({len(sku_splits)} windows)")
                logger.info(f"{'='*60}")

                sku_results = run_multi_period_single_sku(
                    sku_splits, store_id, item_id, config, run_dl_models, verbose=False
                )
                all_results.append(sku_results)

                pbar.update(1)

    # Combine all results
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATED MULTI-PERIOD RESULTS ACROSS ALL WINDOWS AND SKUs")
    logger.info("=" * 80)

    combined_df = pd.concat(all_results, ignore_index=True)

    # Compute aggregated metrics
    agg_columns = ['Mean_Cost', 'CVaR_90', 'CVaR_95', 'Service_Level']
    # Add per-horizon columns
    for h in horizons:
        agg_columns.extend([f'Mean_Cost_h{h}', f'CVaR_90_h{h}', f'Service_Level_h{h}'])

    # Only aggregate columns that exist
    existing_cols = [col for col in agg_columns if col in combined_df.columns]
    agg_dict = {col: ['mean', 'std'] for col in existing_cols}

    aggregated = combined_df.groupby('Method').agg(agg_dict).round(2)

    print("\n", aggregated.to_string())

    # Save results
    agg_path = os.path.join(config.results_dir, "multi_period_aggregated.csv")
    aggregated.to_csv(agg_path)
    logger.info(f"\n[OK] Saved multi-period aggregated results: {agg_path}")

    all_path = os.path.join(config.results_dir, "multi_period_all_windows.csv")
    combined_df.to_csv(all_path, index=False)
    logger.info(f"[OK] Saved multi-period all window results: {all_path}")

    # Save per-SKU aggregated results if multi-sku
    if multi_sku:
        sku_agg = combined_df.groupby(['store_id', 'item_id', 'Method'])[
            ['Mean_Cost', 'CVaR_90', 'CVaR_95', 'Service_Level']
        ].agg(['mean', 'std']).round(2)
        sku_agg_path = os.path.join(config.results_dir, "multi_period_by_sku.csv")
        sku_agg.to_csv(sku_agg_path)
        logger.info(f"[OK] Saved per-SKU multi-period results: {sku_agg_path}")

    # Per-horizon analysis
    logger.info("\n" + "-" * 80)
    logger.info("PER-HORIZON ANALYSIS")
    logger.info("-" * 80)

    for h in horizons:
        logger.info(f"\n[Horizon h={h} days ahead]")
        cost_col = f'Mean_Cost_h{h}'
        if cost_col in combined_df.columns:
            h_agg = combined_df.groupby('Method')[cost_col].agg(['mean', 'std'])
            h_agg = h_agg.sort_values('mean')
            for method, row in h_agg.iterrows():
                if method != 'Seer':
                    logger.info(f"  {get_model_display_name(method)}: ${row['mean']:.2f} +/- ${row['std']:.2f}")

    # Create multi-period summary report
    report_lines = []
    report_lines.append("=" * 80)
    report_lines.append("MULTI-PERIOD COMPREHENSIVE EXPERIMENT REPORT")
    report_lines.append("=" * 80)
    report_lines.append(f"\nForecast Horizons: {horizons}")
    report_lines.append(f"Aggregation Method: {config.multi_period.aggregation}")
    report_lines.append(f"Joint Optimization: {config.multi_period.joint_optimization}")
    report_lines.append(f"\nTotal SKUs: {len(all_sku_splits)}")
    report_lines.append(f"Total Windows: {total_windows}")

    report_lines.append("\n" + "-" * 80)
    report_lines.append("AGGREGATED RESULTS (Mean Cost - lower is better)")
    report_lines.append("-" * 80)

    if ('Mean_Cost', 'mean') in aggregated.columns:
        cost_ranking = aggregated[('Mean_Cost', 'mean')].sort_values()
        for i, (method, cost) in enumerate(cost_ranking.items()):
            marker = " [ORACLE]" if method == "Seer" else ""
            marker = " [YOUR METHOD]" if method == "EnbPI_CQR_CVaR" else marker
            report_lines.append(f"{i+1}. {get_model_display_name(method)}: ${cost:.2f}{marker}")

    report_text = "\n".join(report_lines)
    report_path = os.path.join(config.results_dir, "multi_period_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.info(f"\n[OK] Saved multi-period report: {report_path}")

    print("\n" + report_text)

    logger.info("\n" + "=" * 80)
    logger.info("MULTI-PERIOD EXPERIMENT COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive Multi-Period Multi-SKU Expanding Window CVaR Optimization Experiment"
    )
    parser.add_argument(
        "--output", type=str, default="results/multi_period_expanding",
        help="Output directory for results"
    )
    parser.add_argument(
        "--epochs", type=int, default=50,
        help="Training epochs for DL models"
    )
    parser.add_argument(
        "--device", type=str, default=None,
        help="Device (cpu/cuda)"
    )
    parser.add_argument(
        "--no-dl", action="store_true",
        help="Skip deep learning models (faster, recommended for multi-period)"
    )
    parser.add_argument(
        "--windows", type=int, default=None,
        help="Limit number of windows per SKU (for testing)"
    )
    parser.add_argument(
        "--stores", type=str, default="1",
        help="Store IDs (comma-separated or range, e.g., '1,2,3' or '1-5')"
    )
    parser.add_argument(
        "--items", type=str, default="1",
        help="Item IDs (comma-separated or range, e.g., '1,2,3' or '1-10')"
    )
    parser.add_argument(
        "--data", type=str, default="train.csv",
        help="Path to data file"
    )
    parser.add_argument(
        "--horizons", type=str, default="1,7,14,21,28",
        help="Forecast horizons in days (comma-separated, e.g., '1,7,14,21,28')"
    )
    parser.add_argument(
        "--single-period", action="store_true",
        help="Use single-period mode (legacy, not recommended)"
    )
    parser.add_argument(
        "--aggregation", type=str, default="mean",
        choices=["mean", "sum", "worst_case"],
        help="How to aggregate metrics across horizons"
    )
    parser.add_argument(
        "--no-joint-opt", action="store_true",
        help="Disable joint optimization (optimize separately per horizon)"
    )

    args = parser.parse_args()

    # Parse store and item IDs
    store_ids = parse_id_range(args.stores)
    item_ids = parse_id_range(args.items)

    # Parse horizons
    horizons = [int(h.strip()) for h in args.horizons.split(',')]

    # Create config
    config = get_default_config()
    config.results_dir = args.output
    config.rolling_window.enabled = True
    config.data.filepath = args.data

    # Multi-period settings
    config.multi_period.enabled = not args.single_period
    config.multi_period.forecast_horizons = horizons
    config.multi_period.aggregation = args.aggregation
    config.multi_period.joint_optimization = not args.no_joint_opt

    # Set epochs for DL models
    if args.epochs:
        config.lstm.epochs = args.epochs

    if args.device:
        config.device = args.device

    if args.single_period:
        logger.info("Running in SINGLE-PERIOD mode (legacy)")
        main(
            config,
            store_ids=store_ids,
            item_ids=item_ids,
            run_dl_models=not args.no_dl,
            max_windows=args.windows
        )
    else:
        logger.info("Running in MULTI-PERIOD mode (recommended)")
        main_multi_period(
            config,
            store_ids=store_ids,
            item_ids=item_ids,
            horizons=horizons,
            run_dl_models=not args.no_dl,
            max_windows=args.windows
        )
