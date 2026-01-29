#!/usr/bin/env python
"""
Comprehensive Expanding Window Experiment: Simple → Advanced → Your Method

This script implements a rigorous comparison of 12 forecasting methods for
inventory optimization using expanding window cross-validation.

Model Hierarchy (Simple → Advanced → Your Method):
==================================================
1. Historical Quantile      - Naïve empirical quantile baseline
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
12. Seer                    - Oracle upper bound (perfect foresight)

Key Experimental Design:
========================
- Expanding Window: Training set grows over time (not sliding)
- Calibration Set: Fixed size for conformal calibration
- Test Window: Monthly prediction horizon
- Metrics: Mean Cost, CVaR-90, CVaR-95, Coverage, Service Level

References:
===========
- Vovk et al. (2005) "Algorithmic Learning in a Random World"
- Romano et al. (2019) "Conformalized Quantile Regression"
- Xu & Xie (2021) "Conformal prediction interval for dynamic time-series"
- Elmachtoub & Grigas (2017) "Smart 'Predict, then Optimize'"
- Birge & Louveaux (2011) "Introduction to Stochastic Programming"
- Efron & Tibshirani (1993) "An Introduction to the Bootstrap"

Usage:
    python run_comprehensive_expanding_window.py [--epochs EPOCHS] [--output OUTPUT_DIR]
"""

import argparse
import logging
import os
import sys
import warnings
from datetime import datetime
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
import seaborn as sns

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    load_and_prepare_rolling_data,
    prepare_rolling_sequence_data,
    RollingWindowSplit,
)
from src.models import (
    # Traditional models (Simple → Advanced)
    HistoricalQuantile,
    NormalAssumption,
    BootstrappedNewsvendor,
    SampleAverageApproximation,
    TwoStageStochastic,
    ConformalPrediction,
    QuantileRegression,
    EnsembleBatchPI,
    Seer,
    # Deep learning models
    LSTMQuantileLossOnly,
    LSTMQuantileRegression,
    SPOEndToEnd,
    PredictionResult,
)
from src.optimization import (
    compute_order_quantities_cvar,
    CostParameters,
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
    "12_Oracle": ["Seer"],
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
        "EnbPI_CQR_CVaR": "11. EnbPI+CQR+CVaR ★",
        "Seer": "12. Seer (Oracle)",
    }
    return display_names.get(method_name, method_name)


# =============================================================================
# MAIN EXPERIMENT RUNNER
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
    logger.info(f"\n{'='*80}")
    logger.info(f"WINDOW {window_split.window_idx}: {window_split.test_start_date.date()} to {window_split.test_end_date.date()}")
    logger.info(f"{'='*80}")

    results = {}
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
    # 1. HISTORICAL QUANTILE (NAÏVE BASELINE)
    # =========================================================================
    logger.info("\n[1/12] Running Historical Quantile (Naïve)...")
    try:
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
        results["HistoricalQuantile"] = compute_all_metrics(
            "HistoricalQuantile", y_test, hq_pred.point, hq_orders,
            hq_pred.lower, hq_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.error(f"Historical Quantile failed: {e}")

    # =========================================================================
    # 2. NORMAL ASSUMPTION (PARAMETRIC)
    # =========================================================================
    logger.info("\n[2/12] Running Normal Assumption (Parametric)...")
    try:
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
        results["NormalAssumption"] = compute_all_metrics(
            "NormalAssumption", y_test, normal_pred.point, normal_orders,
            normal_pred.lower, normal_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.error(f"Normal Assumption failed: {e}")

    # =========================================================================
    # 3. BOOTSTRAPPED NEWSVENDOR (RESAMPLING)
    # =========================================================================
    logger.info("\n[3/12] Running Bootstrapped Newsvendor (Resampling)...")
    try:
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
        results["BootstrappedNewsvendor"] = compute_all_metrics(
            "BootstrappedNewsvendor", y_test, boot_pred.point, boot_orders,
            boot_pred.lower, boot_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.error(f"Bootstrapped Newsvendor failed: {e}")

    # =========================================================================
    # 4. SAA (STANDARD OR BENCHMARK)
    # =========================================================================
    logger.info("\n[4/12] Running SAA (Standard OR)...")
    try:
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
        results["SAA"] = compute_all_metrics(
            "SAA", y_test, saa_pred.point, saa_orders,
            None, None,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.error(f"SAA failed: {e}")

    # =========================================================================
    # 5. TWO-STAGE STOCHASTIC (SCENARIO OPTIMIZATION)
    # =========================================================================
    logger.info("\n[5/12] Running Two-Stage Stochastic (Scenario Optimization)...")
    try:
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
        results["TwoStageStochastic"] = compute_all_metrics(
            "TwoStageStochastic", y_test, tss_pred.point, tss_orders,
            tss_pred.lower, tss_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.error(f"Two-Stage Stochastic failed: {e}")

    # =========================================================================
    # 6. CONFORMAL PREDICTION (DISTRIBUTION-FREE)
    # =========================================================================
    logger.info("\n[6/12] Running Conformal Prediction (Distribution-Free)...")
    try:
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
        results["ConformalPrediction"] = compute_all_metrics(
            "ConformalPrediction", y_test, cp_pred.point, cp_orders,
            cp_pred.lower, cp_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.error(f"Conformal Prediction failed: {e}")

    # =========================================================================
    # 7. QUANTILE REGRESSION (DIRECT QUANTILE + CQR)
    # =========================================================================
    logger.info("\n[7/12] Running Quantile Regression (Direct Quantile)...")
    try:
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
        results["QuantileRegression"] = compute_all_metrics(
            "QuantileRegression", y_test, qr_pred.point, qr_orders,
            qr_pred.lower, qr_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.error(f"Quantile Regression failed: {e}")

    # =========================================================================
    # 8-10. DEEP LEARNING MODELS (Optional)
    # =========================================================================
    if run_dl_models:
        # 8. LSTM QUANTILE LOSS (WITHOUT CALIBRATION)
        logger.info("\n[8/12] Running LSTM Quantile Loss (Uncalibrated)...")
        try:
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
            results["LSTMQuantileLoss"] = compute_all_metrics(
                "LSTMQuantileLoss", y_test_seq, lstm_uncal_pred.point, lstm_uncal_orders,
                lstm_uncal_pred.lower, lstm_uncal_pred.upper,
                costs.ordering_cost, costs.holding_cost, costs.stockout_cost
            )
        except Exception as e:
            logger.error(f"LSTM Quantile Loss failed: {e}")

        # 9. LSTM + CONFORMAL (WITH CALIBRATION)
        logger.info("\n[9/12] Running LSTM+Conformal (Calibrated)...")
        try:
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
            results["LSTMConformal"] = compute_all_metrics(
                "LSTMConformal", y_test_seq, lstm_cal_pred.point, lstm_cal_orders,
                lstm_cal_pred.lower, lstm_cal_pred.upper,
                costs.ordering_cost, costs.holding_cost, costs.stockout_cost
            )
        except Exception as e:
            logger.error(f"LSTM+Conformal failed: {e}")

        # 10. SPO (DECISION-FOCUSED LEARNING)
        logger.info("\n[10/12] Running SPO (Decision-Focused)...")
        try:
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
            results["SPO"] = compute_all_metrics(
                "SPO", y_test_seq, spo_pred.point, spo_orders,
                spo_pred.lower, spo_pred.upper,
                costs.ordering_cost, costs.holding_cost, costs.stockout_cost
            )
        except Exception as e:
            logger.error(f"SPO failed: {e}")

    # =========================================================================
    # 11. EnbPI + CQR + CVaR (YOUR CONTRIBUTION)
    # =========================================================================
    logger.info("\n[11/12] Running EnbPI+CQR+CVaR (Your Contribution)...")
    try:
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
        results["EnbPI_CQR_CVaR"] = compute_all_metrics(
            "EnbPI_CQR_CVaR", y_test, enbpi_pred.point, enbpi_orders,
            enbpi_pred.lower, enbpi_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.error(f"EnbPI+CQR+CVaR failed: {e}")

    # =========================================================================
    # 12. SEER (ORACLE - UPPER BOUND)
    # =========================================================================
    logger.info("\n[12/12] Running Seer (Oracle Upper Bound)...")
    try:
        seer_model = Seer(alpha=0.05, random_state=config.random_seed)
        seer_model.fit(X_train, y_train, X_cal, y_cal)
        seer_pred = seer_model.predict_with_actuals(X_test, y_test)
        seer_orders = seer_model.compute_order_quantities(
            y_test,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost
        )
        results["Seer"] = compute_all_metrics(
            "Seer", y_test, seer_pred.point, seer_orders,
            seer_pred.lower, seer_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
    except Exception as e:
        logger.error(f"Seer failed: {e}")

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
        }
        summary_data.append(row)

    summary_df = pd.DataFrame(summary_data)
    summary_df['window_idx'] = window_split.window_idx
    summary_df['test_start'] = window_split.test_start_date
    summary_df['test_end'] = window_split.test_end_date

    return summary_df, results


# =============================================================================
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_comprehensive_visualizations(
    combined_df: pd.DataFrame,
    aggregated: pd.DataFrame,
    output_dir: str
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
    """
    logger.info("\nCreating visualizations...")

    # Set style
    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    # Define model order (Simple → Advanced → Your Method → Oracle)
    model_order = [
        'HistoricalQuantile', 'NormalAssumption', 'BootstrappedNewsvendor',
        'SAA', 'TwoStageStochastic', 'ConformalPrediction', 'QuantileRegression',
        'LSTMQuantileLoss', 'LSTMConformal', 'SPO', 'EnbPI_CQR_CVaR', 'Seer'
    ]

    # Filter to existing methods
    existing_methods = [m for m in model_order if m in combined_df['Method'].unique()]

    # 1. CVaR-90 Comparison (Main Metric)
    fig, ax = plt.subplots(figsize=(14, 6))
    df_plot = combined_df[combined_df['Method'].isin(existing_methods)].copy()
    df_plot['Method'] = pd.Categorical(df_plot['Method'], categories=existing_methods, ordered=True)

    sns.boxplot(data=df_plot, x='Method', y='CVaR_90', ax=ax)
    ax.set_title('CVaR-90 Comparison Across All Windows\n(Lower is Better)', fontsize=14, fontweight='bold')
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
                marker = '★' if method == 'EnbPI_CQR_CVaR' else 'o'
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
        for method in ['EnbPI_CQR_CVaR', 'ConformalPrediction', 'NormalAssumption', 'SAA']:
            if method in combined_df['Method'].values:
                method_data = combined_df[combined_df['Method'] == method].sort_values('window_idx')
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

    logger.info(f"Visualizations saved to {output_dir}")


def create_summary_report(
    combined_df: pd.DataFrame,
    aggregated: pd.DataFrame,
    output_dir: str
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

    Returns
    -------
    str
        The report content.
    """
    report = []
    report.append("=" * 80)
    report.append("COMPREHENSIVE EXPANDING WINDOW EXPERIMENT REPORT")
    report.append("Simple → Advanced → Your Method")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"Number of Windows: {combined_df['window_idx'].nunique()}")
    report.append(f"Number of Methods: {combined_df['Method'].nunique()}")

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

    # Full results table
    report.append("\n" + "-" * 80)
    report.append("AGGREGATED RESULTS (Mean ± Std across windows)")
    report.append("-" * 80)
    report.append("\n" + aggregated.to_string())

    # Save report
    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, "experiment_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    logger.info(f"Report saved to {report_path}")
    return report_text


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main(config: ExperimentConfig, run_dl_models: bool = True):
    """
    Main experiment runner.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    run_dl_models : bool
        If True, run deep learning models.
    """
    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

    logger.info("=" * 80)
    logger.info("COMPREHENSIVE EXPANDING WINDOW CVaR OPTIMIZATION EXPERIMENT")
    logger.info("Simple → Advanced → Your Method (EnbPI+CQR+CVaR)")
    logger.info("=" * 80)
    logger.info(f"Device: {config.device}")
    logger.info(f"Store: {config.data.store_ids[0]}, Item: {config.data.item_ids[0]}")
    logger.info(f"Run DL Models: {run_dl_models}")

    # Create output directory
    os.makedirs(config.results_dir, exist_ok=True)

    # Load expanding window data
    logger.info("\nLoading expanding window data...")
    logger.info(f"  Initial train: {config.rolling_window.initial_train_days} days")
    logger.info(f"  Calibration: {config.rolling_window.calibration_days} days")
    logger.info(f"  Test window: {config.rolling_window.test_window_days} days")
    logger.info(f"  Step: {config.rolling_window.step_days} days")

    rolling_splits = load_and_prepare_rolling_data(
        filepath=config.data.filepath,
        store_id=config.data.store_ids[0],
        item_id=config.data.item_ids[0],
        lag_periods=config.data.lag_features,
        rolling_windows=config.data.rolling_windows,
        initial_train_days=config.rolling_window.initial_train_days,
        calibration_days=config.rolling_window.calibration_days,
        test_window_days=config.rolling_window.test_window_days,
        step_days=config.rolling_window.step_days
    )

    logger.info(f"Created {len(rolling_splits)} expanding windows\n")

    # Run experiment on each window
    all_window_results = []
    all_method_results = {}

    for window_split in rolling_splits:
        summary_df, method_results = run_single_window(window_split, config, run_dl_models)
        all_window_results.append(summary_df)

        # Store results for statistical tests
        for method_name, result in method_results.items():
            if method_name not in all_method_results:
                all_method_results[method_name] = []
            all_method_results[method_name].append(result)

        logger.info(f"\nWindow {window_split.window_idx} Results (Top 5 by CVaR-90):")
        top5 = summary_df.nsmallest(5, 'CVaR_90')[['DisplayName', 'Mean_Cost', 'CVaR_90', 'Coverage']]
        print(top5.to_string(index=False))

    # Aggregate results
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATED RESULTS ACROSS ALL WINDOWS")
    logger.info("=" * 80)

    combined_df = pd.concat(all_window_results, ignore_index=True)

    # Compute mean and std
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

    # Create visualizations
    create_comprehensive_visualizations(combined_df, aggregated, config.results_dir)

    # Create summary report
    report = create_summary_report(combined_df, aggregated, config.results_dir)
    print("\n" + report)

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Comprehensive Expanding Window CVaR Optimization Experiment"
    )
    parser.add_argument(
        "--output", type=str, default="results/comprehensive_expanding",
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
        help="Skip deep learning models (faster)"
    )
    parser.add_argument(
        "--windows", type=int, default=None,
        help="Limit number of windows (for testing)"
    )

    args = parser.parse_args()

    # Create config
    config = get_default_config()
    config.results_dir = args.output
    config.rolling_window.enabled = True

    # Set epochs for DL models
    if args.epochs:
        config.lstm.epochs = args.epochs

    if args.device:
        config.device = args.device

    main(config, run_dl_models=not args.no_dl)
