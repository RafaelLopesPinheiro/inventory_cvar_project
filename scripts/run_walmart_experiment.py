#!/usr/bin/env python
"""
Walmart Retail Dataset – Multi-SKU Expanding Window Inventory Experiment
========================================================================

Applies the same 9-method comparison framework as run_comprehensive_expanding_window.py
to the Kaggle "Walmart Recruiting – Store Sales Forecasting" dataset
(https://www.kaggle.com/c/walmart-recruiting-store-sales-forecasting).

DATASET (walmart/ directory):
------------------------------
  train.csv    : Store, Dept, Date, Weekly_Sales, IsHoliday
  features.csv : Store, Date, Temperature, Fuel_Price, MarkDown1-5, CPI, Unemployment, IsHoliday
  stores.csv   : Store, Type, Size (metadata only, not used for modelling)

DATA CHARACTERISTICS:
----------------------
  - Granularity : Weekly (every Friday, Feb 2010 – Oct 2012 ≈ 143 weeks)
  - SKU key     : (Store, Dept) → mapped to (store, item) for pipeline compatibility
  - Target      : Weekly_Sales (dollar value of department weekly sales)
  - Exogenous   : IsHoliday; optional Temperature, CPI, Fuel_Price, MarkDown1-5, Unemployment

KEY DIFFERENCES FROM DAILY EXPERIMENT:
----------------------------------------
  - Lag features     : [1, 4, 13, 52] rows  (1 wk, 1 mo, 1 quarter, 1 yr)
  - Rolling windows  : [4, 13]  rows        (1 mo, 1 quarter)
  - Window sizes     : train=52, cal=26, test=4, step=4  (rows = weeks)
  - LSTM seq length  : 13 weeks  (1 quarter)
  - Warmup period    : 13 rows   (rows with partial lags are filled via bfill/ffill)
  - Default capacity : 500 000   (dollar-scale; override with --capacity)

MODEL HIERARCHY (9 Methods):
=============================
  0. (s,S) Policy           – Simple rule-based reorder benchmark
  1. SAA                    – Sample Average Approximation
  2. Conformal + CVaR       – Conformal PI + CVaR optimisation
  3. Wasserstein DRO        – Distributionally Robust Optimisation
  4. EnbPI+CQR+CVaR (SL95) – Ensemble PI + CQR + CVaR with SL≥95% constraint
  5. SPO (RF, CVaR)         – Smart Predict-then-Optimise
  6. LSTM+Conformal+CVaR    – LSTM quantile regression + conformal calibration + CVaR
  7. Seer                   – Oracle upper bound (perfect foresight)
  8. CQR+SPO (Hybrid)       – EnbPI+CQR intervals + residual-based CVaR scenarios

Usage:
    # Single store-dept (fast smoke-test)
    python run_walmart_experiment.py --stores 1 --depts 1

    # Default: stores 1-3, depts 1-5
    python run_walmart_experiment.py

    # Custom selection with exogenous features
    python run_walmart_experiment.py --stores 1,2,3 --depts 1,2,3 --features walmart/features.csv

    # Disable LSTM for speed
    python run_walmart_experiment.py --no-lstm --stores 1-5 --depts 1-10

    # Limit windows for a quick test
    python run_walmart_experiment.py --windows 3
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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import multiprocessing

_NUM_WORKERS = min(multiprocessing.cpu_count(), 8)

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    filter_store_item,
    create_all_features,
    create_rolling_window_splits,
    RollingWindowSplit,
)
from src.models import (
    SampleAverageApproximation,
    ConformalPrediction,
    EnsembleBatchPI,
    DistributionallyRobustOptimization,
    SPORandomForest,
    Seer,
    LSTMQuantileRegression,
    PredictionResult,
)
from src.optimization import (
    compute_order_quantities_cvar,
    compute_inventory_aware_orders_cvar,
    compute_inventory_aware_orders_dro,
    CostParameters,
    simulate_inventory_with_carryover,
    simulate_sS_policy_with_carryover,
    InventorySimulationResult,
)
from src.evaluation import (
    compute_all_metrics,
    MethodResults,
)
from configs import get_default_config, ExperimentConfig

warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# MODEL DEFINITIONS (identical to run_comprehensive_expanding_window.py)
# =============================================================================

MODEL_CATEGORIES = {
    "0_SimplePolicy": ["sS_Policy"],
    "1_OR_Standard": ["SAA"],
    "2_DistributionFree": ["Conformal_CVaR"],
    "3_RobustOptimization": ["Wasserstein_DRO"],
    "4_YourContribution": ["EnbPI_CQR_CVaR"],
    "5_EndToEnd": ["SPO_EndToEnd"],
    "6_DeepLearning": ["LSTM_Conformal_CVaR"],
    "7_Oracle": ["Seer"],
    "8_HybridCQR_SPO": ["CQR_SPO"],
}

MODEL_ORDER = [
    'sS_Policy', 'SAA', 'Conformal_CVaR', 'Wasserstein_DRO',
    'EnbPI_CQR_CVaR', 'SPO_EndToEnd', 'LSTM_Conformal_CVaR', 'Seer',
    'CQR_SPO',
]

MODEL_DISPLAY_NAMES = {
    "sS_Policy": "0. (s,S) Policy",
    "SAA": "1. SAA",
    "Conformal_CVaR": "2. Conformal + CVaR",
    "Wasserstein_DRO": "3. Wasserstein DRO",
    "EnbPI_CQR_CVaR": "4. EnbPI+CQR+CVaR (SL95)",
    "SPO_EndToEnd": "5. SPO (RF, CVaR)",
    "LSTM_Conformal_CVaR": "6. LSTM+Conformal+CVaR",
    "Seer": "7. Seer (Oracle)",
    "CQR_SPO": "8. CQR+SPO (Hybrid)",
}

MODEL_COLORS = {
    "sS_Policy": "#8c564b",
    "SAA": "#1f77b4",
    "Conformal_CVaR": "#ff7f0e",
    "Wasserstein_DRO": "#9467bd",
    "EnbPI_CQR_CVaR": "#d62728",
    "SPO_EndToEnd": "#e377c2",
    "LSTM_Conformal_CVaR": "#17becf",
    "Seer": "#2ca02c",
    "CQR_SPO": "#bcbd22",
}


def get_model_display_name(method_name: str) -> str:
    return MODEL_DISPLAY_NAMES.get(method_name, method_name)


# Module-level flag toggled by --no-lstm
_ENABLE_LSTM = True


# =============================================================================
# SEQUENCE CREATION HELPER FOR LSTM
# =============================================================================

def _make_sequences(
    X: np.ndarray,
    seq_len: int,
    X_context: Optional[np.ndarray] = None,
) -> np.ndarray:
    """Convert 2-D feature array → 3-D sequence tensor for LSTM input."""
    n, f = X.shape

    if X_context is not None:
        ctx_len = min(seq_len - 1, len(X_context))
        context = X_context[-ctx_len:]
        pad_len = seq_len - 1 - ctx_len
        if pad_len > 0:
            pad = np.repeat(X[:1], pad_len, axis=0)
            X_full = np.vstack([pad, context, X])
        else:
            X_full = np.vstack([context, X])
        return np.stack([X_full[i: i + seq_len] for i in range(n)])
    else:
        if n < seq_len:
            pad = np.repeat(X[:1], seq_len - n, axis=0)
            X_full = np.vstack([pad, X])
        else:
            X_full = X
        n_seq = len(X_full) - seq_len + 1
        return np.stack([X_full[i: i + seq_len] for i in range(n_seq)])


# =============================================================================
# WALMART-SPECIFIC DATA LOADING
# =============================================================================

# Exogenous feature columns available in features.csv
_EXOG_COLS = [
    'Temperature', 'Fuel_Price',
    'MarkDown1', 'MarkDown2', 'MarkDown3', 'MarkDown4', 'MarkDown5',
    'CPI', 'Unemployment',
]

# Lag and rolling-window settings tuned for weekly granularity
_WEEKLY_LAGS = [1, 4, 13, 52]       # 1 wk, 1 mo, 1 quarter, 1 yr
_WEEKLY_ROLLING = [4, 13]            # 1 mo, 1 quarter
_WEEKLY_WARMUP = 13                  # drop first 13 rows (partial lag period)

# Expanding-window sizes (in rows = weeks)
_TRAIN_ROWS = 52        # 1 year
_CAL_ROWS = 26          # 6 months
_TEST_ROWS = 4          # 1 month (test horizon)
_STEP_ROWS = 4          # roll forward 1 month per window
_MIN_RECORDS = 100      # minimum raw rows to include a Store-Dept pair


def load_walmart_data(
    train_path: str,
    features_path: Optional[str] = None,
) -> pd.DataFrame:
    """
    Load and normalise the Walmart Store Sales dataset.

    Reads train.csv (Store, Dept, Date, Weekly_Sales, IsHoliday) and
    renames columns to (store, item, date, sales) for pipeline compatibility.
    Optionally joins exogenous features from features.csv on (Store, Date).

    Parameters
    ----------
    train_path : str
        Path to walmart/train.csv.
    features_path : str or None
        Path to walmart/features.csv. When provided, Temperature, Fuel_Price,
        MarkDown1-5, CPI, and Unemployment are added as extra columns.

    Returns
    -------
    pd.DataFrame
        Long-format dataframe with columns:
          date (datetime), store (int), item (int), sales (float),
          is_holiday (int), [optional exog columns]
    """
    logger.info(f"Loading Walmart train data from {train_path}")
    df = pd.read_csv(train_path)

    # Standardise column names to match pipeline expectations
    df = df.rename(columns={
        'Store': 'store',
        'Dept': 'item',
        'Date': 'date',
        'Weekly_Sales': 'sales',
        'IsHoliday': 'is_holiday',
    })
    df['date'] = pd.to_datetime(df['date'])
    df['is_holiday'] = df['is_holiday'].astype(int)

    # Optionally merge store-level exogenous features
    if features_path is not None and os.path.isfile(features_path):
        logger.info(f"Merging exogenous features from {features_path}")
        feat = pd.read_csv(features_path)
        feat = feat.rename(columns={'Store': 'store', 'Date': 'date'})
        feat['date'] = pd.to_datetime(feat['date'])

        # Keep only the numeric feature columns; drop the duplicate IsHoliday
        keep_cols = ['store', 'date'] + _EXOG_COLS
        feat = feat[[c for c in keep_cols if c in feat.columns]]

        # Fill MarkDown NaN with 0 (missing = no promotion active)
        for col in [c for c in _EXOG_COLS if c.startswith('MarkDown')]:
            if col in feat.columns:
                feat[col] = feat[col].fillna(0.0)

        df = df.merge(feat, on=['store', 'date'], how='left')
        # Fill any remaining NaN from the merge
        for col in _EXOG_COLS:
            if col in df.columns:
                df[col] = df[col].fillna(method='bfill').fillna(method='ffill').fillna(0.0)

    logger.info(f"Walmart dataset loaded: {len(df):,} rows, "
                f"{df['store'].nunique()} stores, {df['item'].nunique()} departments")
    return df


def load_expanding_window_data_walmart(
    train_path: str,
    store_ids: List[int],
    dept_ids: List[int],
    features_path: Optional[str] = None,
    lag_periods: List[int] = _WEEKLY_LAGS,
    rolling_windows: List[int] = _WEEKLY_ROLLING,
    warmup_period: int = _WEEKLY_WARMUP,
    initial_train_rows: int = _TRAIN_ROWS,
    calibration_rows: int = _CAL_ROWS,
    test_window_rows: int = _TEST_ROWS,
    step_rows: int = _STEP_ROWS,
    min_records: int = _MIN_RECORDS,
) -> Dict[Tuple[int, int], List[RollingWindowSplit]]:
    """
    Load Walmart expanding-window data for multiple (Store, Dept) pairs.

    Returns
    -------
    Dict[Tuple[int, int], List[RollingWindowSplit]]
        Maps (store_id, dept_id) → list of rolling window splits.
    """
    logger.info(
        f"Loading Walmart expanding-window data for "
        f"{len(store_ids)} stores × {len(dept_ids)} depts"
    )

    df_raw = load_walmart_data(train_path, features_path)

    # Detect exogenous feature columns present after loading
    exog_in_data = [c for c in _EXOG_COLS if c in df_raw.columns]

    results = {}
    skipped = []

    total = len(store_ids) * len(dept_ids)
    with tqdm(total=total, desc="Loading Walmart SKU data") as pbar:
        for store_id in store_ids:
            for dept_id in dept_ids:
                try:
                    df = filter_store_item(df_raw, store_id, dept_id)

                    if len(df) < min_records:
                        skipped.append((store_id, dept_id,
                                        f"insufficient data ({len(df)} < {min_records})"))
                        pbar.update(1)
                        continue

                    # Store is_holiday + exog columns before feature engineering
                    # so we can re-attach them to feature_cols
                    extra_cols_pre = ['is_holiday'] + exog_in_data
                    extra_cols_present = [c for c in extra_cols_pre if c in df.columns]

                    # Standard feature engineering (time, lags, rolling stats)
                    df, feature_cols = create_all_features(
                        df,
                        lag_periods=lag_periods,
                        rolling_windows=rolling_windows,
                        warmup_period=warmup_period,
                    )

                    # Add Walmart-specific extra columns to the feature set
                    for col in extra_cols_present:
                        if col in df.columns and col not in feature_cols:
                            feature_cols.append(col)

                    splits = create_rolling_window_splits(
                        df, feature_cols,
                        initial_train_days=initial_train_rows,
                        calibration_days=calibration_rows,
                        test_window_days=test_window_rows,
                        step_days=step_rows,
                    )

                    if len(splits) > 0:
                        results[(store_id, dept_id)] = splits
                    else:
                        skipped.append((store_id, dept_id, "no valid windows"))

                except Exception as e:
                    skipped.append((store_id, dept_id, str(e)))
                pbar.update(1)

    logger.info(f"Successfully loaded {len(results)} store-dept combinations")
    if skipped:
        logger.warning(f"Skipped {len(skipped)} combinations:")
        for sid, did, reason in skipped[:5]:
            logger.warning(f"  Store {sid}, Dept {did}: {reason}")
        if len(skipped) > 5:
            logger.warning(f"  ... and {len(skipped) - 5} more")

    return results


# =============================================================================
# EXPERIMENT RUNNER (identical logic to run_comprehensive_expanding_window.py)
# =============================================================================

def _compute_forecast_metrics(
    y_test: np.ndarray,
    pred: PredictionResult,
) -> dict:
    """Compute forecast quality metrics from predictions."""
    point = pred.point
    mae = float(np.mean(np.abs(y_test - point)))
    rmse = float(np.sqrt(np.mean((y_test - point) ** 2)))

    mask = y_test != 0
    if mask.sum() > 0:
        mape = float(np.mean(np.abs((y_test[mask] - point[mask]) / y_test[mask])) * 100)
    else:
        mape = np.nan

    coverage = np.nan
    avg_interval_width = np.nan
    if pred.lower is not None and pred.upper is not None:
        coverage = float(np.mean((y_test >= pred.lower) & (y_test <= pred.upper)))
        avg_interval_width = float(np.mean(pred.upper - pred.lower))

    return {
        'Coverage': coverage,
        'Avg_Interval_Width': avg_interval_width,
        'MAE': mae,
        'RMSE': rmse,
        'MAPE': mape,
    }


def run_single_window(
    window_split: RollingWindowSplit,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, Dict[str, InventorySimulationResult]]:
    """
    Run all 9 models on a single expanding window with carryover and capacity.

    Identical to run_comprehensive_expanding_window.py's run_single_window;
    reproduced here so the Walmart script is fully self-contained.
    """
    results = {}
    sim_results = {}
    timings = {}
    costs = config.cost

    X_train, y_train = window_split.train.X, window_split.train.y
    X_cal, y_cal = window_split.calibration.X, window_split.calibration.y
    X_test, y_test = window_split.test.X, window_split.test.y

    # =========================================================================
    # 0. (s, S) POLICY
    # =========================================================================
    try:
        start_time = time.time()
        historical_demand = np.concatenate([y_train, y_cal])
        critical_ratio = costs.stockout_cost / (costs.stockout_cost + costs.holding_cost)
        s_param = float(np.quantile(historical_demand, critical_ratio))
        S_quantile = min(0.99, critical_ratio + (1.0 - critical_ratio) * 0.5)
        S_param = float(np.quantile(historical_demand, S_quantile))
        S_param = max(S_param, s_param * 1.05 + 1.0)

        sS_sim = simulate_sS_policy_with_carryover(
            y_test, s=s_param, S=S_param,
            initial_inventory=costs.initial_inventory,
            carryover_rate=costs.carryover_rate,
            capacity=costs.capacity,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost,
        )
        sS_pred = PredictionResult(point=np.full(len(y_test), s_param), lower=None, upper=None)
        timings["sS_Policy"] = time.time() - start_time
        sim_results["sS_Policy"] = sS_sim
        results["sS_Policy"] = {
            'pred': sS_pred, 'target_orders': sS_sim.actual_orders,
            'sim': sS_sim, 'time': timings["sS_Policy"],
        }
    except Exception as e:
        logger.debug(f"(s,S) Policy failed: {e}")

    # =========================================================================
    # 1. SAA
    # =========================================================================
    try:
        start_time = time.time()
        saa_model = SampleAverageApproximation(
            n_estimators=100, max_depth=10,
            stockout_cost=costs.stockout_cost,
            holding_cost=costs.holding_cost,
            random_state=config.random_seed,
        )
        saa_model.fit(X_train, y_train, X_cal, y_cal)
        saa_pred = saa_model.predict(X_test)
        saa_orders = saa_model.compute_order_quantities(X_test)
        saa_sim = simulate_inventory_with_carryover(
            saa_orders, y_test,
            initial_inventory=costs.initial_inventory,
            carryover_rate=costs.carryover_rate,
            capacity=costs.capacity,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost,
            inventory_aware=True,
        )
        timings["SAA"] = time.time() - start_time
        sim_results["SAA"] = saa_sim
        results["SAA"] = {
            'pred': saa_pred, 'target_orders': saa_orders,
            'sim': saa_sim, 'time': timings["SAA"],
        }
    except Exception as e:
        logger.debug(f"SAA failed: {e}")

    # =========================================================================
    # 2. CONFORMAL + CVaR
    # =========================================================================
    try:
        start_time = time.time()
        cp_model = ConformalPrediction(
            alpha=config.conformal.alpha,
            n_estimators=config.conformal.n_estimators,
            max_depth=config.conformal.max_depth,
            random_state=config.random_seed,
        )
        cp_model.fit(X_train, y_train, X_cal, y_cal)
        cp_pred = cp_model.predict(X_test)
        cp_sim = compute_inventory_aware_orders_cvar(
            cp_pred.point, cp_pred.lower, cp_pred.upper,
            actual_demands=y_test,
            initial_inventory=costs.initial_inventory,
            carryover_rate=costs.carryover_rate,
            capacity=costs.capacity,
            beta=config.cvar.beta, n_samples=config.cvar.n_samples,
            ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
            verbose=False,
        )
        timings["Conformal_CVaR"] = time.time() - start_time
        sim_results["Conformal_CVaR"] = cp_sim
        results["Conformal_CVaR"] = {
            'pred': cp_pred, 'target_orders': cp_sim.actual_orders,
            'sim': cp_sim, 'time': timings["Conformal_CVaR"],
        }
    except Exception as e:
        logger.debug(f"Conformal+CVaR failed: {e}")

    # =========================================================================
    # 3. WASSERSTEIN DRO
    # =========================================================================
    try:
        start_time = time.time()
        dro_model = DistributionallyRobustOptimization(
            alpha=config.conformal.alpha,
            epsilon=0.1,
            n_estimators=config.conformal.n_estimators,
            max_depth=config.conformal.max_depth,
            n_scenarios=500,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost,
            cvar_beta=config.cvar.beta,
            adaptive_epsilon=True,
            random_state=config.random_seed,
        )
        dro_model.fit(X_train, y_train, X_cal, y_cal)
        dro_pred = dro_model.predict(X_test)
        dro_sim = compute_inventory_aware_orders_dro(
            dro_pred.point, dro_pred.lower, dro_pred.upper,
            actual_demands=y_test,
            epsilon=0.1,
            initial_inventory=costs.initial_inventory,
            carryover_rate=costs.carryover_rate,
            capacity=costs.capacity,
            beta=config.cvar.beta, n_samples=500,
            ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
            verbose=False,
        )
        timings["Wasserstein_DRO"] = time.time() - start_time
        sim_results["Wasserstein_DRO"] = dro_sim
        results["Wasserstein_DRO"] = {
            'pred': dro_pred, 'target_orders': dro_sim.actual_orders,
            'sim': dro_sim, 'time': timings["Wasserstein_DRO"],
        }
    except Exception as e:
        logger.debug(f"Wasserstein DRO failed: {e}")

    # =========================================================================
    # 4. EnbPI + CQR + CVaR  (SL ≥ 95%)
    # =========================================================================
    enbpi_model = None
    enbpi_pred = None
    try:
        start_time = time.time()
        enbpi_model = EnsembleBatchPI(
            alpha=config.ensemble_batch_pi.alpha,
            n_ensemble=config.ensemble_batch_pi.n_ensemble,
            n_estimators=config.ensemble_batch_pi.n_estimators,
            max_depth=config.ensemble_batch_pi.max_depth,
            bootstrap_fraction=config.ensemble_batch_pi.bootstrap_fraction,
            use_quantile_regression=config.ensemble_batch_pi.use_quantile_regression,
            random_state=config.random_seed,
        )
        enbpi_model.fit(X_train, y_train, X_cal, y_cal)
        enbpi_pred = enbpi_model.predict(X_test)
        enbpi_sim = compute_inventory_aware_orders_cvar(
            enbpi_pred.point, enbpi_pred.lower, enbpi_pred.upper,
            actual_demands=y_test,
            initial_inventory=costs.initial_inventory,
            carryover_rate=costs.carryover_rate,
            capacity=costs.capacity,
            beta=config.cvar.beta, n_samples=config.cvar.n_samples,
            ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
            verbose=False,
            sl_target=0.95,
        )
        timings["EnbPI_CQR_CVaR"] = time.time() - start_time
        sim_results["EnbPI_CQR_CVaR"] = enbpi_sim
        results["EnbPI_CQR_CVaR"] = {
            'pred': enbpi_pred, 'target_orders': enbpi_sim.actual_orders,
            'sim': enbpi_sim, 'time': timings["EnbPI_CQR_CVaR"],
        }
    except Exception as e:
        logger.debug(f"EnbPI+CQR+CVaR failed: {e}")

    # =========================================================================
    # 5. SPO  (RF-based, decision-focused)
    # =========================================================================
    try:
        start_time = time.time()
        spo_model = SPORandomForest(
            alpha=config.conformal.alpha,
            n_estimators=100,
            max_depth=10,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost,
            cvar_beta=config.cvar.beta,
            n_scenarios=config.cvar.n_samples,
            random_state=config.random_seed,
        )
        spo_model.fit(X_train, y_train, X_cal, y_cal)
        spo_pred = spo_model.predict(X_test)
        spo_orders = spo_model.compute_order_quantities(X_test)
        spo_sim = simulate_inventory_with_carryover(
            spo_orders, y_test,
            initial_inventory=costs.initial_inventory,
            carryover_rate=costs.carryover_rate,
            capacity=costs.capacity,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost,
            inventory_aware=True,
        )
        timings["SPO_EndToEnd"] = time.time() - start_time
        sim_results["SPO_EndToEnd"] = spo_sim
        results["SPO_EndToEnd"] = {
            'pred': spo_pred, 'target_orders': spo_orders,
            'sim': spo_sim, 'time': timings["SPO_EndToEnd"],
        }
    except Exception as e:
        logger.debug(f"SPO (RF) failed: {e}")

    # =========================================================================
    # 8. CQR + SPO HYBRID  (Task 2)
    # =========================================================================
    try:
        if 'EnbPI_CQR_CVaR' in results and enbpi_model is not None:
            start_time = time.time()
            enbpi_cal_pred = enbpi_model.predict(X_cal)
            cqr_spo_residuals = y_cal - enbpi_cal_pred.point
            cqr_spo_sim = compute_inventory_aware_orders_cvar(
                enbpi_pred.point, enbpi_pred.lower, enbpi_pred.upper,
                actual_demands=y_test,
                initial_inventory=costs.initial_inventory,
                carryover_rate=costs.carryover_rate,
                capacity=costs.capacity,
                beta=config.cvar.beta, n_samples=config.cvar.n_samples,
                ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
                verbose=False,
                demand_residuals=cqr_spo_residuals,
                sl_target=None,
            )
            timings["CQR_SPO"] = time.time() - start_time
            sim_results["CQR_SPO"] = cqr_spo_sim
            results["CQR_SPO"] = {
                'pred': enbpi_pred,
                'target_orders': cqr_spo_sim.actual_orders,
                'sim': cqr_spo_sim, 'time': timings["CQR_SPO"],
            }
    except Exception as e:
        logger.debug(f"CQR+SPO hybrid failed: {e}")

    # =========================================================================
    # 6. LSTM + CONFORMAL + CVaR
    # =========================================================================
    if _ENABLE_LSTM:
        try:
            start_time = time.time()
            seq_len = config.data.sequence_length

            X_train_3d = _make_sequences(X_train, seq_len)
            n_train_seq = len(X_train_3d)
            y_train_seq = y_train[-n_train_seq:]

            X_cal_3d = _make_sequences(X_cal, seq_len, X_context=X_train)
            y_cal_seq = y_cal

            X_test_3d = _make_sequences(X_test, seq_len, X_context=X_cal)

            lstm_model = LSTMQuantileRegression(
                alpha=config.conformal.alpha,
                sequence_length=seq_len,
                hidden_size=config.lstm.hidden_size,
                num_layers=config.lstm.num_layers,
                dropout=config.lstm.dropout,
                learning_rate=config.lstm.learning_rate,
                epochs=config.lstm.epochs,
                batch_size=config.lstm.batch_size,
                random_state=config.random_seed,
                device=config.device,
            )
            lstm_model.fit(
                X_train_3d, y_train_seq,
                X_cal_3d, y_cal_seq,
                early_stopping_patience=20,
                min_epochs=20,
            )
            lstm_pred = lstm_model.predict(X_test_3d)
            lstm_sim = compute_inventory_aware_orders_cvar(
                lstm_pred.point, lstm_pred.lower, lstm_pred.upper,
                actual_demands=y_test,
                initial_inventory=costs.initial_inventory,
                carryover_rate=costs.carryover_rate,
                capacity=costs.capacity,
                beta=config.cvar.beta, n_samples=config.cvar.n_samples,
                ordering_cost=costs.ordering_cost,
                holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost,
                random_seed=config.cvar.random_seed,
                verbose=False,
            )
            timings["LSTM_Conformal_CVaR"] = time.time() - start_time
            sim_results["LSTM_Conformal_CVaR"] = lstm_sim
            results["LSTM_Conformal_CVaR"] = {
                'pred': lstm_pred, 'target_orders': lstm_sim.actual_orders,
                'sim': lstm_sim, 'time': timings["LSTM_Conformal_CVaR"],
            }
        except Exception as e:
            logger.debug(f"LSTM+Conformal+CVaR failed: {e}")

    # =========================================================================
    # 7. SEER (ORACLE)
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
            stockout_cost=costs.stockout_cost,
        )
        seer_sim = simulate_inventory_with_carryover(
            seer_orders, y_test,
            initial_inventory=costs.initial_inventory,
            carryover_rate=costs.carryover_rate,
            capacity=costs.capacity,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost,
            inventory_aware=True,
        )
        timings["Seer"] = time.time() - start_time
        sim_results["Seer"] = seer_sim
        results["Seer"] = {
            'pred': seer_pred, 'target_orders': seer_orders,
            'sim': seer_sim, 'time': timings["Seer"],
        }
    except Exception as e:
        logger.debug(f"Seer failed: {e}")

    # =========================================================================
    # BUILD SUMMARY DATAFRAME
    # =========================================================================
    summary_data = []
    for method_name, result_data in results.items():
        sim = result_data['sim']
        pred = result_data['pred']
        n_pred = len(pred.point)
        y_eval = y_test[-n_pred:] if n_pred < len(y_test) else y_test[:n_pred]
        forecast = _compute_forecast_metrics(y_eval, pred)
        row = {
            'Method': method_name,
            'DisplayName': get_model_display_name(method_name),
            'Coverage': forecast['Coverage'],
            'Avg_Interval_Width': forecast['Avg_Interval_Width'],
            'MAE': forecast['MAE'],
            'RMSE': forecast['RMSE'],
            'MAPE': forecast['MAPE'],
            'Mean_Cost': sim.mean_cost,
            'CVaR_90': sim.cvar_90,
            'CVaR_95': sim.cvar_95,
            'Service_Level': sim.service_level,
            'Avg_Carryover': sim.avg_carryover,
            'Avg_Capacity_Util': sim.avg_capacity_utilization,
            'Total_Ordering_Cost': float(np.sum(sim.ordering_costs)),
            'Total_Holding_Cost': float(np.sum(sim.holding_costs)),
            'Total_Stockout_Cost': float(np.sum(sim.stockout_costs)),
            'Time_Seconds': result_data['time'],
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df['window_idx'] = window_split.window_idx
    summary_df['test_start'] = window_split.test_start_date
    summary_df['test_end'] = window_split.test_end_date

    return summary_df, sim_results


def run_single_sku(
    sku_splits: List[RollingWindowSplit],
    store_id: int,
    item_id: int,
    config: ExperimentConfig,
    verbose: bool = False,
) -> Tuple[pd.DataFrame, Dict]:
    """Run expanding window experiment for a single (Store, Dept) pair."""
    all_window_results = []
    last_sim_results = None

    for window_split in sku_splits:
        if verbose:
            logger.info(f"  Window {window_split.window_idx}: "
                        f"{window_split.test_start_date.date()} to "
                        f"{window_split.test_end_date.date()}")
        summary_df, sim_results = run_single_window(window_split, config)
        summary_df['store_id'] = store_id
        summary_df['item_id'] = item_id
        all_window_results.append(summary_df)
        last_sim_results = sim_results

    combined = pd.concat(all_window_results, ignore_index=True)
    return combined, last_sim_results


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comprehensive_visualizations(
    combined_df: pd.DataFrame,
    aggregated: pd.DataFrame,
    last_sim_results: Dict[str, InventorySimulationResult],
    output_dir: str,
    config: ExperimentConfig,
    multi_sku: bool = False,
):
    """Create comprehensive visualisations (identical structure to main script)."""
    logger.info("\nCreating visualizations...")
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    existing_methods = [m for m in MODEL_ORDER if m in combined_df['Method'].unique()]

    # --- Coverage & Interval Width ---
    if 'Coverage' in combined_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for ax, (col, ylabel, title) in zip(
            axes,
            [('Coverage', 'Coverage (%)', 'Prediction Interval Coverage\n(Target = 95%)'),
             ('Avg_Interval_Width', 'Average Interval Width', 'Prediction Interval Width\n(Narrower = Better at Same Coverage)')]
        ):
            vals, labels, colors = [], [], []
            for m in existing_methods:
                if m in combined_df['Method'].values:
                    v = combined_df[combined_df['Method'] == m][col].mean()
                    if not np.isnan(v):
                        vals.append(v * 100 if col == 'Coverage' else v)
                        labels.append(get_model_display_name(m))
                        colors.append(MODEL_COLORS.get(m, 'steelblue'))
            if vals:
                bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.8,
                              edgecolor='black', linewidth=0.5)
                if col == 'Coverage':
                    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.7,
                               linewidth=2, label='95% Target')
                    ax.set_ylim(0, 105)
                    ax.legend(fontsize=9)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                for bar, val in zip(bars, vals):
                    ax.annotate(f'{val:.1f}{"%" if col=="Coverage" else ""}',
                                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'forecast_coverage_width.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # --- RMSE & MAE ---
    if 'RMSE' in combined_df.columns:
        fig, axes = plt.subplots(1, 2, figsize=(16, 6))
        for idx, (metric, ylabel, title) in enumerate([
            ('RMSE', 'RMSE', 'Root Mean Squared Error\n(Lower is Better)'),
            ('MAE', 'MAE', 'Mean Absolute Error\n(Lower is Better)'),
        ]):
            ax = axes[idx]
            vals, labels, colors = [], [], []
            for m in existing_methods:
                if m in combined_df['Method'].values:
                    v = combined_df[combined_df['Method'] == m][metric].mean()
                    if not np.isnan(v):
                        vals.append(v)
                        labels.append(get_model_display_name(m))
                        colors.append(MODEL_COLORS.get(m, 'steelblue'))
            if vals:
                bars = ax.bar(range(len(labels)), vals, color=colors, alpha=0.8,
                              edgecolor='black', linewidth=0.5)
                ax.set_xticks(range(len(labels)))
                ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=9)
                ax.set_ylabel(ylabel, fontsize=12)
                ax.set_title(title, fontsize=12, fontweight='bold')
                ax.grid(True, alpha=0.3, axis='y')
                for bar, val in zip(bars, vals):
                    ax.annotate(f'{val:.2f}',
                                xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                                ha='center', va='bottom', fontsize=8, fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'forecast_rmse_mae.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # --- CVaR-90 Boxplot ---
    fig, ax = plt.subplots(figsize=(12, 6))
    colors = [MODEL_COLORS.get(m, 'steelblue') for m in existing_methods]
    bp = ax.boxplot(
        [combined_df[combined_df['Method'] == m]['CVaR_90'].dropna().values
         for m in existing_methods],
        labels=[get_model_display_name(m) for m in existing_methods],
        patch_artist=True, widths=0.6,
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax.set_title('CVaR-90 Comparison Across All Windows\n(Lower is Better)',
                 fontsize=14, fontweight='bold')
    ax.set_ylabel('CVaR-90 ($)', fontsize=12)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cvar90_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Mean Cost Bar Chart ---
    fig, ax = plt.subplots(figsize=(12, 6))
    agg_flat = aggregated.reset_index()
    means, stds, bar_colors, labels = [], [], [], []
    for m in existing_methods:
        if m in agg_flat['Method'].values:
            try:
                mv = agg_flat[agg_flat['Method'] == m][('Mean_Cost', 'mean')].values[0]
                sv = agg_flat[agg_flat['Method'] == m][('Mean_Cost', 'std')].values[0]
            except (KeyError, IndexError):
                mv = combined_df[combined_df['Method'] == m]['Mean_Cost'].mean()
                sv = combined_df[combined_df['Method'] == m]['Mean_Cost'].std()
            means.append(mv)
            stds.append(sv)
            bar_colors.append(MODEL_COLORS.get(m, 'steelblue'))
            labels.append(get_model_display_name(m))
    bars = ax.bar(range(len(labels)), means, yerr=stds, color=bar_colors,
                  alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_xticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Mean Cost ($)', fontsize=12)
    ax.set_title('Mean Cost Comparison (with Carryover & Capacity)\n(Lower is Better)',
                 fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, means):
        ax.annotate(f'${val:.1f}',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_cost_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Cost Breakdown (stacked bar) ---
    fig, ax = plt.subplots(figsize=(12, 6))
    ord_costs, hold_costs, stock_costs, mlabels = [], [], [], []
    for m in existing_methods:
        if m in combined_df['Method'].values:
            mdf = combined_df[combined_df['Method'] == m]
            ord_costs.append(mdf['Total_Ordering_Cost'].mean())
            hold_costs.append(mdf['Total_Holding_Cost'].mean())
            stock_costs.append(mdf['Total_Stockout_Cost'].mean())
            mlabels.append(get_model_display_name(m))
    x_pos = range(len(mlabels))
    ax.bar(x_pos, ord_costs, 0.6, label='Ordering', color='#3498db', alpha=0.85)
    ax.bar(x_pos, hold_costs, 0.6, bottom=ord_costs, label='Holding', color='#f39c12', alpha=0.85)
    bottoms = [o + h for o, h in zip(ord_costs, hold_costs)]
    ax.bar(x_pos, stock_costs, 0.6, bottom=bottoms, label='Stockout', color='#e74c3c', alpha=0.85)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(mlabels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Cost ($)', fontsize=12)
    ax.set_title('Cost Breakdown by Component', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Service Level ---
    fig, ax = plt.subplots(figsize=(12, 6))
    sl_vals, sl_labels, sl_colors = [], [], []
    for m in existing_methods:
        if m in combined_df['Method'].values:
            sl_vals.append(combined_df[combined_df['Method'] == m]['Service_Level'].mean() * 100)
            sl_labels.append(get_model_display_name(m))
            sl_colors.append(MODEL_COLORS.get(m, 'steelblue'))
    bars = ax.bar(range(len(sl_labels)), sl_vals, color=sl_colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='95% Target')
    ax.set_xticks(range(len(sl_labels)))
    ax.set_xticklabels(sl_labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Service Level (%)', fontsize=12)
    ax.set_title('Service Level Comparison\n(Higher is Better, Target = 95%)',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)
    for bar, val in zip(bars, sl_vals):
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'service_level_comparison.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- Capacity Utilisation ---
    fig, ax = plt.subplots(figsize=(12, 6))
    cu_vals, cu_labels, cu_colors = [], [], []
    for m in existing_methods:
        if m in combined_df['Method'].values:
            cu_vals.append(combined_df[combined_df['Method'] == m]['Avg_Capacity_Util'].mean() * 100)
            cu_labels.append(get_model_display_name(m))
            cu_colors.append(MODEL_COLORS.get(m, 'steelblue'))
    bars = ax.bar(range(len(cu_labels)), cu_vals, color=cu_colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Full Capacity')
    ax.set_xticks(range(len(cu_labels)))
    ax.set_xticklabels(cu_labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Capacity Utilization (%)', fontsize=12)
    ax.set_title(f'Average Capacity Utilization (Capacity = {config.cost.capacity:,.0f})\n'
                 f'(Carryover Rate = {config.cost.carryover_rate})',
                 fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    for bar, val in zip(bars, cu_vals):
        ax.annotate(f'{val:.1f}%',
                    xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'capacity_utilization.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- Inventory Dynamics (last window) ---
    if last_sim_results:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)
        first_method = list(last_sim_results.keys())[0]
        n_periods = len(last_sim_results[first_method].demands)
        periods = range(n_periods)

        ax1 = axes[0]
        ax1.plot(periods, last_sim_results[first_method].demands, 'k-',
                 linewidth=2, label='Actual Demand', alpha=0.8, zorder=10)
        for m in existing_methods:
            if m in last_sim_results:
                ax1.plot(periods, last_sim_results[m].actual_orders, '--',
                         linewidth=1.5, color=MODEL_COLORS.get(m, 'gray'),
                         label=f'{get_model_display_name(m)} Orders', alpha=0.7)
        ax1.set_ylabel('Sales ($)', fontsize=12)
        ax1.set_title('Demand vs Order Quantities (Last Window)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        ax2 = axes[1]
        ax2.axhline(y=config.cost.capacity, color='red', linestyle='--',
                    linewidth=2, alpha=0.7, label=f'Capacity ({config.cost.capacity:,.0f})')
        for m in existing_methods:
            if m in last_sim_results:
                ax2.plot(periods, last_sim_results[m].inventory_levels, '-',
                         linewidth=1.5, color=MODEL_COLORS.get(m, 'gray'),
                         label=get_model_display_name(m), alpha=0.8)
        ax2.set_xlabel('Week', fontsize=12)
        ax2.set_ylabel('Inventory Level ($)', fontsize=12)
        ax2.set_title('Inventory Levels Over Time (with Carryover & Capacity)',
                      fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inventory_dynamics.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # --- Cumulative Cost (last window) ---
    if last_sim_results:
        fig, ax = plt.subplots(figsize=(14, 6))
        for m in existing_methods:
            if m in last_sim_results:
                cumulative = np.cumsum(last_sim_results[m].costs)
                ax.plot(range(len(cumulative)), cumulative, '-', linewidth=2,
                        color=MODEL_COLORS.get(m, 'gray'),
                        label=f'{get_model_display_name(m)} (Total: ${cumulative[-1]:.0f})',
                        alpha=0.8)
        ax.set_xlabel('Week', fontsize=12)
        ax.set_ylabel('Cumulative Cost ($)', fontsize=12)
        ax.set_title('Cumulative Cost Over Time (Last Window)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cumulative_cost.png'),
                    dpi=150, bbox_inches='tight')
        plt.close()

    # --- Performance Progression ---
    fig, axes = plt.subplots(2, 3, figsize=(22, 10))
    metrics_list = [
        ('RMSE', 'RMSE (Forecast)'),
        ('Coverage', 'Coverage (Forecast)'),
        ('Avg_Interval_Width', 'Interval Width (Forecast)'),
        ('Mean_Cost', 'Mean Cost ($)'),
        ('CVaR_90', 'CVaR-90 ($)'),
        ('Service_Level', 'Service Level'),
    ]
    for idx, (metric, ylabel) in enumerate(metrics_list):
        ax = axes[idx // 3, idx % 3]
        for m in existing_methods:
            if m in combined_df['Method'].values:
                mdata = (combined_df[combined_df['Method'] == m]
                         .groupby('window_idx')[metric].mean()
                         .reset_index())
                if metric in mdata.columns:
                    ax.plot(mdata['window_idx'], mdata[metric], '-o',
                            label=get_model_display_name(m),
                            color=MODEL_COLORS.get(m, 'gray'),
                            linewidth=2, markersize=5, alpha=0.8)
        ax.set_xlabel('Window Index', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'{ylabel} Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)
    plt.suptitle('Performance Progression Across Expanding Windows (Walmart)',
                 fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_progression.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    # --- Rankings Heatmap ---
    fig, ax = plt.subplots(figsize=(10, 6))
    metrics = ['RMSE', 'Mean_Cost', 'CVaR_90', 'CVaR_95']
    rank_data = []
    for m in existing_methods:
        if m in combined_df['Method'].values:
            row = {'Method': get_model_display_name(m)}
            for met in metrics:
                row[met] = combined_df[combined_df['Method'] == m][met].mean()
            rank_data.append(row)
    if rank_data:
        rank_df = pd.DataFrame(rank_data).set_index('Method')
        sns.heatmap(rank_df.rank(), annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax,
                    linewidths=0.5, linecolor='white')
        ax.set_title('Method Rankings Across Metrics\n(1 = Best)',
                     fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_rankings.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # --- Summary Dashboard ---
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)
    panels = [
        (gs[0, 0], 'Mean_Cost', 'Mean Cost ($)', 'Mean Cost'),
        (gs[0, 1], 'CVaR_90', 'CVaR-90 ($)', 'CVaR-90 (Tail Risk)'),
        (gs[0, 2], 'Service_Level', 'Service Level (%)', 'Service Level'),
        (gs[1, 0], 'Avg_Capacity_Util', 'Cap. Util (%)', 'Capacity Utilization'),
        (gs[1, 1], 'Avg_Carryover', 'Avg Carryover ($)', 'Average Carryover'),
    ]
    mc_labels_short = [get_model_display_name(m).split('. ')[-1] for m in existing_methods
                       if m in combined_df['Method'].values]
    mc_colors_d = [MODEL_COLORS.get(m, 'steelblue') for m in existing_methods
                   if m in combined_df['Method'].values]

    for spec, col, xlabel, title in panels:
        ax = fig.add_subplot(spec)
        vals = [combined_df[combined_df['Method'] == m][col].mean() *
                (100 if col in ('Service_Level', 'Avg_Capacity_Util') else 1)
                for m in existing_methods if m in combined_df['Method'].values]
        ax.barh(range(len(mc_labels_short)), vals, color=mc_colors_d, alpha=0.8)
        ax.set_yticks(range(len(mc_labels_short)))
        ax.set_yticklabels(mc_labels_short, fontsize=9)
        ax.set_xlabel(xlabel)
        ax.set_title(title, fontweight='bold')
        ax.invert_yaxis()
        if col == 'Service_Level':
            ax.axvline(x=95, color='gray', linestyle='--', alpha=0.7)

    plt.suptitle('Walmart Experiment – Summary Dashboard', fontsize=16, fontweight='bold')
    plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'),
                dpi=150, bbox_inches='tight')
    plt.close()

    logger.info(f"All visualizations saved to {output_dir}/")


# =============================================================================
# STATISTICAL TESTS (identical to run_comprehensive_expanding_window.py)
# =============================================================================

def compute_statistical_tests(
    combined_df: pd.DataFrame,
    output_dir: str,
    reference_method: str = "EnbPI_CQR_CVaR",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Paired t-test and Wilcoxon signed-rank test vs reference method.
    Applies Bonferroni correction; reports Cohen's d effect size.
    """
    existing_methods = [m for m in MODEL_ORDER if m in combined_df['Method'].unique()]
    comparators = [m for m in existing_methods if m != reference_method]

    if reference_method not in combined_df['Method'].values:
        logger.warning(f"Reference method '{reference_method}' not in results; skipping stats.")
        return pd.DataFrame()

    ref_data = combined_df[combined_df['Method'] == reference_method].sort_values(
        ['store_id', 'item_id', 'window_idx'])
    ref_costs = ref_data['Mean_Cost'].values

    rows = []
    n_tests = len(comparators)
    bonferroni_alpha = alpha / max(n_tests, 1)

    for method in comparators:
        mdf = combined_df[combined_df['Method'] == method].sort_values(
            ['store_id', 'item_id', 'window_idx'])
        m_costs = mdf['Mean_Cost'].values

        min_len = min(len(ref_costs), len(m_costs))
        if min_len < 3:
            continue

        rc = ref_costs[:min_len]
        mc = m_costs[:min_len]
        diff = rc - mc  # positive = reference is worse (higher cost)

        try:
            t_stat, t_pval = stats.ttest_rel(rc, mc)
        except Exception:
            t_stat, t_pval = np.nan, np.nan

        try:
            w_stat, w_pval = stats.wilcoxon(diff)
        except Exception:
            w_stat, w_pval = np.nan, np.nan

        pooled_std = np.std(np.concatenate([rc, mc]))
        cohen_d = np.mean(diff) / pooled_std if pooled_std > 0 else 0.0

        rows.append({
            'Method': method,
            'DisplayName': get_model_display_name(method),
            'Mean_Diff': float(np.mean(diff)),
            'Std_Diff': float(np.std(diff)),
            'T_Statistic': float(t_stat),
            'T_PValue': float(t_pval),
            'W_Statistic': float(w_stat) if not np.isnan(w_stat) else np.nan,
            'W_PValue': float(w_pval) if not np.isnan(w_pval) else np.nan,
            'Cohen_D': float(cohen_d),
            'Bonferroni_Alpha': bonferroni_alpha,
            'T_Significant': t_pval < bonferroni_alpha if not np.isnan(t_pval) else False,
            'W_Significant': w_pval < bonferroni_alpha if not np.isnan(w_pval) else False,
        })

    if not rows:
        return pd.DataFrame()

    stat_df = pd.DataFrame(rows)
    stat_path = os.path.join(output_dir, 'statistical_tests.csv')
    stat_df.to_csv(stat_path, index=False)

    logger.info("\n" + "=" * 80)
    logger.info(f"STATISTICAL TESTS (reference: {reference_method})")
    logger.info(f"Bonferroni-corrected α = {bonferroni_alpha:.4f}  (original α = {alpha}, "
                f"n_tests = {n_tests})")
    logger.info("=" * 80)
    for _, row in stat_df.iterrows():
        sig_t = '***' if row['T_Significant'] else ''
        sig_w = '***' if row['W_Significant'] else ''
        logger.info(
            f"{row['DisplayName']:<35}: "
            f"ΔCost={row['Mean_Diff']:+.2f}  "
            f"t-p={row['T_PValue']:.4f}{sig_t}  "
            f"W-p={row['W_PValue']:.4f}{sig_w}  "
            f"d={row['Cohen_D']:.3f}"
        )

    return stat_df


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def create_summary_report(
    combined_df: pd.DataFrame,
    aggregated: pd.DataFrame,
    output_dir: str,
    store_ids: List[int],
    dept_ids: List[int],
    config: ExperimentConfig,
) -> str:
    """Generate text summary report for the Walmart experiment."""
    report = []
    report.append("=" * 80)
    report.append("WALMART STORE SALES – INVENTORY OPTIMISATION EXPERIMENT REPORT")
    report.append("=" * 80)
    report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Dataset  : Kaggle Walmart Recruiting – Store Sales Forecasting (weekly)")
    report.append(f"Stores   : {store_ids}")
    report.append(f"Depts    : {dept_ids}")
    report.append(f"SKUs     : {len(store_ids) * len(dept_ids)}")
    report.append(f"Windows  : {combined_df['window_idx'].nunique()} per SKU")
    report.append("")
    report.append("WINDOW PARAMETERS (weekly rows):")
    report.append(f"  Training period     : {_TRAIN_ROWS} weeks (1 year)")
    report.append(f"  Calibration period  : {_CAL_ROWS} weeks (6 months)")
    report.append(f"  Test horizon        : {_TEST_ROWS} weeks per window")
    report.append(f"  Step size           : {_STEP_ROWS} weeks")
    report.append("")
    report.append("INVENTORY PARAMETERS:")
    report.append(f"  Carryover Rate : {config.cost.carryover_rate}")
    report.append(f"  Capacity       : {config.cost.capacity:,.0f} units (dollar-scale)")
    report.append(f"  Ordering Cost  : ${config.cost.ordering_cost}/unit")
    report.append(f"  Holding Cost   : ${config.cost.holding_cost}/unit")
    report.append(f"  Stockout Cost  : ${config.cost.stockout_cost}/unit")
    report.append(f"  Critical Ratio : {config.cost.critical_ratio:.4f}")

    existing_methods = [m for m in MODEL_ORDER if m in combined_df['Method'].unique()]
    mean_costs = {m: combined_df[combined_df['Method'] == m]['Mean_Cost'].mean()
                  for m in existing_methods if m != 'Seer'}

    if mean_costs:
        best = min(mean_costs, key=mean_costs.get)
        report.append(f"\n[BEST Mean Cost] {get_model_display_name(best)}: ${mean_costs[best]:.2f}")

    cvar90s = {m: combined_df[combined_df['Method'] == m]['CVaR_90'].mean()
               for m in existing_methods if m != 'Seer'}
    if cvar90s:
        best_c = min(cvar90s, key=cvar90s.get)
        report.append(f"[BEST CVaR-90]   {get_model_display_name(best_c)}: ${cvar90s[best_c]:.2f}")

    sls = {m: combined_df[combined_df['Method'] == m]['Service_Level'].mean()
           for m in existing_methods}
    if sls:
        best_sl = max(sls, key=sls.get)
        report.append(f"[BEST Service]   {get_model_display_name(best_sl)}: "
                      f"{sls[best_sl]*100:.1f}%")

    # Value of optimisation vs (s,S)
    if 'sS_Policy' in combined_df['Method'].values:
        sS_cost = combined_df[combined_df['Method'] == 'sS_Policy']['Mean_Cost'].mean()
        sS_cvar = combined_df[combined_df['Method'] == 'sS_Policy']['CVaR_90'].mean()
        sS_sl = combined_df[combined_df['Method'] == 'sS_Policy']['Service_Level'].mean()

        report.append("\n" + "-" * 80)
        report.append("VALUE OF OPTIMISATION: gain vs. (s,S) simple policy")
        report.append("-" * 80)
        hdr = (f"{'Method':<25} {'Cost Saving ($)':>16} {'Cost Saving (%)':>16} "
               f"{'CVaR-90 Red (%)':>16} {'SL Delta (pp)':>14}")
        report.append(f"\n{hdr}")
        report.append("-" * len(hdr))
        for m in existing_methods:
            if m in ('sS_Policy', 'Seer'):
                continue
            mdf = combined_df[combined_df['Method'] == m]
            mc = mdf['Mean_Cost'].mean()
            mv = mdf['CVaR_90'].mean()
            ms = mdf['Service_Level'].mean()
            cs_abs = sS_cost - mc
            cs_pct = cs_abs / sS_cost * 100 if sS_cost else 0
            cv_pct = (sS_cvar - mv) / sS_cvar * 100 if sS_cvar else 0
            sl_pp = (ms - sS_sl) * 100
            report.append(
                f"{get_model_display_name(m):<25} "
                f"{cs_abs:>+15.2f}  {cs_pct:>+15.1f}% "
                f"{cv_pct:>+15.1f}% {sl_pp:>+13.1f}pp"
            )

    # Decision quality table
    report.append("\n" + "-" * 80)
    report.append("DECISION QUALITY (Mean across all windows and SKUs)")
    report.append("-" * 80)
    hdr = f"{'Method':<25} {'Mean Cost':>10} {'CVaR-90':>10} {'CVaR-95':>10} {'SL (%)':>8} {'Cap Util':>10} {'Time(s)':>8}"
    report.append(f"\n{hdr}")
    report.append("-" * len(hdr))
    for m in existing_methods:
        mdf = combined_df[combined_df['Method'] == m]
        report.append(
            f"{get_model_display_name(m):<25} "
            f"${mdf['Mean_Cost'].mean():>8.2f} "
            f"${mdf['CVaR_90'].mean():>8.2f} "
            f"${mdf['CVaR_95'].mean():>8.2f} "
            f"{mdf['Service_Level'].mean()*100:>7.1f} "
            f"{mdf['Avg_Capacity_Util'].mean()*100:>9.1f}% "
            f"{mdf['Time_Seconds'].mean():>7.2f}"
        )

    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, "experiment_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)
    logger.info(f"Report saved to {report_path}")
    return report_text


# =============================================================================
# HELPERS
# =============================================================================

def parse_id_range(s: str) -> List[int]:
    """Parse '1,2,3' or '1-5' into a sorted list of ints."""
    ids = []
    for part in s.split(','):
        part = part.strip()
        if '-' in part:
            lo, hi = part.split('-')
            ids.extend(range(int(lo), int(hi) + 1))
        else:
            ids.append(int(part))
    return sorted(list(set(ids)))


# =============================================================================
# MAIN
# =============================================================================

def main(
    config: ExperimentConfig,
    store_ids: List[int],
    dept_ids: List[int],
    train_path: str,
    features_path: Optional[str],
    max_windows: Optional[int] = None,
):
    """Main Walmart experiment runner."""
    np.random.seed(config.random_seed)
    multi_sku = len(store_ids) > 1 or len(dept_ids) > 1

    logger.info("=" * 80)
    logger.info("WALMART EXPERIMENT – INVENTORY OPTIMISATION WITH CARRYOVER & CAPACITY")
    logger.info("9-Method Comparison on Weekly Store-Department Sales Data")
    logger.info("=" * 80)
    logger.info(f"Stores : {store_ids}")
    logger.info(f"Depts  : {dept_ids}")
    logger.info(f"SKUs   : {len(store_ids) * len(dept_ids)}")
    logger.info(f"Carryover Rate: {config.cost.carryover_rate}  |  "
                f"Capacity: {config.cost.capacity:,.0f}  |  "
                f"LSTM: {'ON' if _ENABLE_LSTM else 'OFF'}")

    os.makedirs(config.results_dir, exist_ok=True)

    logger.info("\nLoading Walmart expanding-window data (weekly rows)...")
    all_sku_splits = load_expanding_window_data_walmart(
        train_path=train_path,
        store_ids=store_ids,
        dept_ids=dept_ids,
        features_path=features_path,
    )

    if not all_sku_splits:
        logger.error("No valid store-dept combinations found. "
                     "Check paths and --stores/--depts arguments.")
        return

    if max_windows is not None:
        for key in all_sku_splits:
            all_sku_splits[key] = all_sku_splits[key][:max_windows]

    total_windows = sum(len(v) for v in all_sku_splits.values())
    logger.info(f"Loaded {len(all_sku_splits)} SKUs, {total_windows} total windows")

    all_results = []
    last_sim_results = None

    with tqdm(total=len(all_sku_splits), desc="Processing SKUs") as pbar:
        for (store_id, dept_id), sku_splits in all_sku_splits.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Store {store_id}, Dept {dept_id}  ({len(sku_splits)} windows)")
            logger.info(f"{'='*60}")
            sku_results, sku_sim_results = run_single_sku(
                sku_splits, store_id, dept_id, config, verbose=False
            )
            all_results.append(sku_results)
            last_sim_results = sku_sim_results
            pbar.update(1)

    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATED RESULTS ACROSS ALL WINDOWS AND SKUs")
    logger.info("=" * 80)

    combined_df = pd.concat(all_results, ignore_index=True)

    agg_metrics = ['Coverage', 'Avg_Interval_Width', 'MAE', 'RMSE', 'MAPE',
                   'Mean_Cost', 'CVaR_90', 'CVaR_95', 'Service_Level',
                   'Avg_Carryover', 'Avg_Capacity_Util',
                   'Total_Ordering_Cost', 'Total_Holding_Cost', 'Total_Stockout_Cost',
                   'Time_Seconds']
    existing_agg = [c for c in agg_metrics if c in combined_df.columns]
    aggregated = combined_df.groupby('Method').agg(
        {col: ['mean', 'std'] for col in existing_agg}
    ).round(3)

    print("\n", aggregated.to_string())

    agg_path = os.path.join(config.results_dir, "aggregated_results.csv")
    aggregated.to_csv(agg_path)
    logger.info(f"[OK] Saved aggregated results: {agg_path}")

    all_path = os.path.join(config.results_dir, "all_windows_results.csv")
    combined_df.to_csv(all_path, index=False)
    logger.info(f"[OK] Saved all window results: {all_path}")

    if multi_sku:
        sku_agg = combined_df.groupby(['store_id', 'item_id', 'Method'])[
            ['Mean_Cost', 'CVaR_90', 'CVaR_95', 'Service_Level',
             'Avg_Carryover', 'Avg_Capacity_Util']
        ].agg(['mean', 'std']).round(3)
        sku_agg.to_csv(os.path.join(config.results_dir, "results_by_sku.csv"))
        logger.info(f"[OK] Saved per-SKU results.")

    create_comprehensive_visualizations(
        combined_df, aggregated, last_sim_results,
        config.results_dir, config, multi_sku,
    )

    compute_statistical_tests(
        combined_df, config.results_dir,
        reference_method="EnbPI_CQR_CVaR",
        alpha=0.05,
    )

    report = create_summary_report(
        combined_df, aggregated, config.results_dir,
        store_ids, dept_ids, config,
    )
    print("\n" + report)

    logger.info("\n" + "=" * 80)
    logger.info("WALMART EXPERIMENT COMPLETE")
    logger.info("=" * 80)


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Walmart Retail Dataset – Multi-SKU Inventory Optimisation Experiment\n"
            "Applies the same 9-method framework as run_comprehensive_expanding_window.py\n"
            "to weekly Walmart store-department sales data."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # --- Data paths ---
    parser.add_argument(
        "--train", type=str,
        default=os.path.join("walmart", "train.csv"),
        help="Path to walmart/train.csv (default: walmart/train.csv)",
    )
    parser.add_argument(
        "--features", type=str, default=None,
        help=(
            "Path to walmart/features.csv. When provided, Temperature, Fuel_Price, "
            "MarkDown1-5, CPI, and Unemployment are added as exogenous features."
        ),
    )

    # --- SKU selection ---
    parser.add_argument(
        "--stores", type=str, default="1,2,3",
        help="Store IDs (comma-separated or range, e.g. '1,2,3' or '1-5')",
    )
    parser.add_argument(
        "--depts", type=str, default="1,2,3,4,5",
        help="Department IDs (comma-separated or range, e.g. '1-10')",
    )

    # --- Output ---
    parser.add_argument(
        "--output", type=str, default=os.path.join("results", "walmart_experiment"),
        help="Output directory for results and plots",
    )
    parser.add_argument(
        "--windows", type=int, default=None,
        help="Limit number of windows per SKU (useful for quick smoke-tests)",
    )

    # --- Inventory dynamics ---
    parser.add_argument("--carryover", type=float, default=0.95,
                        help="Carryover rate: fraction of leftover inventory carried forward (0-1)")
    parser.add_argument("--capacity", type=float, default=500_000.0,
                        help="Warehouse capacity in $ of weekly sales (default 500000)")
    parser.add_argument("--initial-inventory", type=float, default=0.0,
                        help="Initial inventory level")

    # --- Cost parameters ---
    parser.add_argument("--ordering-cost", type=float, default=10.0,
                        help="Ordering cost per unit ($)")
    parser.add_argument("--holding-cost", type=float, default=2.0,
                        help="Holding cost per unit ($)")
    parser.add_argument("--stockout-cost", type=float, default=50.0,
                        help="Stockout cost per unit ($)")

    # --- Model flags ---
    parser.add_argument("--no-lstm", action="store_true", default=False,
                        help="Disable LSTM+Conformal+CVaR (speeds up experiment significantly)")

    args = parser.parse_args()

    # Resolve data paths relative to the project root (parent of scripts/)
    project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    train_path = args.train if os.path.isabs(args.train) else os.path.join(project_root, args.train)
    features_path = None
    if args.features:
        features_path = (args.features if os.path.isabs(args.features)
                         else os.path.join(project_root, args.features))

    store_ids = parse_id_range(args.stores)
    dept_ids = parse_id_range(args.depts)

    _ENABLE_LSTM = not args.no_lstm

    config = get_default_config()
    config.results_dir = args.output
    config.rolling_window.enabled = True

    # Weekly-appropriate sequence length for LSTM (1 quarter = 13 weeks)
    config.data.sequence_length = 13

    # Inventory dynamics
    config.cost.carryover_rate = args.carryover
    config.cost.capacity = args.capacity
    config.cost.initial_inventory = args.initial_inventory
    config.cost.ordering_cost = args.ordering_cost
    config.cost.holding_cost = args.holding_cost
    config.cost.stockout_cost = args.stockout_cost

    logger.info("Walmart Inventory Optimisation Experiment")
    logger.info(f"  Train data  : {train_path}")
    logger.info(f"  Features    : {features_path or 'not used'}")
    logger.info(f"  Stores      : {store_ids}")
    logger.info(f"  Departments : {dept_ids}")
    logger.info(f"  LSTM        : {'enabled' if _ENABLE_LSTM else 'disabled'}")

    main(
        config,
        store_ids=store_ids,
        dept_ids=dept_ids,
        train_path=train_path,
        features_path=features_path,
        max_windows=args.windows,
    )
