#!/usr/bin/env python
"""
Expanding Window Experiment Runner with SPO/End-to-End Baseline

This script runs experiments using expanding window cross-validation,
comparing all models including the critical SPO/End-to-End competitor.

Key differences from rolling window:
- Training set expands over time (not sliding)
- Includes SPO baseline for decision-focused learning comparison

Usage:
    python run_expanding_window_experiment.py [--epochs EPOCHS] [--output OUTPUT_DIR]
"""

import argparse
import logging
import os
import sys
from typing import Dict, List
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    load_and_prepare_rolling_data,
    prepare_rolling_sequence_data,
    RollingWindowSplit,
    TemporalSplits,
)
from src.models import (
    ConformalPrediction,
    NormalAssumption,
    QuantileRegression,
    SampleAverageApproximation,
    ExpectedValue,
    Seer,
    LSTMQuantileRegression,
    TransformerQuantileRegression,
    DeepEnsemble,
    MCDropoutLSTM,
    TemporalFusionTransformer,
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
)
from configs import get_default_config, ExperimentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_window_experiment(
    window_split: RollingWindowSplit,
    config: ExperimentConfig,
    is_expanding: bool = True
) -> pd.DataFrame:
    """
    Run experiment on a single window.

    Parameters
    ----------
    window_split : RollingWindowSplit
        Window split data.
    config : ExperimentConfig
        Experiment configuration.
    is_expanding : bool
        If True, uses expanding window (default).

    Returns
    -------
    pd.DataFrame
        Results summary for this window.
    """
    logger.info(f"\n{'='*70}")
    logger.info(f"Window {window_split.window_idx}: {window_split.test_start_date.date()} to {window_split.test_end_date.date()}")
    logger.info(f"{'='*70}")

    # Prepare sequence data for DL models
    seq_data = prepare_rolling_sequence_data(
        window_split,
        seq_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon
    )

    results = {}
    costs = config.cost

    # Data for traditional methods
    X_train, y_train = window_split.train.X, window_split.train.y
    X_cal, y_cal = window_split.calibration.X, window_split.calibration.y
    X_test, y_test = window_split.test.X, window_split.test.y

    # Data for DL methods
    X_train_seq, y_train_seq = seq_data.X_train, seq_data.y_train
    X_cal_seq, y_cal_seq = seq_data.X_cal, seq_data.y_cal
    X_test_seq, y_test_seq = seq_data.X_test, seq_data.y_test

    # =========================================================================
    # TRADITIONAL METHODS
    # =========================================================================

    # Conformal Prediction
    logger.info("Running Conformal Prediction...")
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
    results["Conformal_CVaR"] = compute_all_metrics(
        "Conformal_CVaR", y_test, cp_pred.point, cp_orders,
        cp_pred.lower, cp_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )

    # Normal Assumption
    logger.info("Running Normal Assumption...")
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
    results["Normal_CVaR"] = compute_all_metrics(
        "Normal_CVaR", y_test, normal_pred.point, normal_orders,
        normal_pred.lower, normal_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )

    # Quantile Regression
    logger.info("Running Quantile Regression...")
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
    results["QuantileReg_CVaR"] = compute_all_metrics(
        "QuantileReg_CVaR", y_test, qr_pred.point, qr_orders,
        qr_pred.lower, qr_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )

    # SAA
    logger.info("Running SAA...")
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

    # =========================================================================
    # SEER (PERFECT FORESIGHT ORACLE - THEORETICAL UPPER BOUND)
    # =========================================================================

    logger.info("Running Seer (Perfect Foresight Oracle)...")
    seer_model = Seer(alpha=0.05, random_state=config.random_seed)
    seer_model.fit(X_train, y_train, X_cal, y_cal)

    # Seer uses actual demand as "predictions"
    seer_pred = seer_model.predict_with_actuals(X_test, y_test)

    # With perfect foresight, order exactly the demand (optimal)
    seer_orders = seer_model.compute_order_quantities(
        y_test,
        ordering_cost=costs.ordering_cost,
        holding_cost=costs.holding_cost,
        stockout_cost=costs.stockout_cost
    )

    results["Seer_Oracle"] = compute_all_metrics(
        "Seer_Oracle", y_test, seer_pred.point, seer_orders,
        seer_pred.lower, seer_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )

    # =========================================================================
    # DEEP LEARNING METHODS
    # =========================================================================

    # Align test data length for comparison
    n_test_seq = len(y_test_seq)

    # LSTM
    logger.info("Running LSTM Quantile Regression...")
    lstm_model = LSTMQuantileRegression(
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
    lstm_model.fit(X_train_seq, y_train_seq, X_cal_seq, y_cal_seq)
    lstm_pred = lstm_model.predict(X_test_seq)
    lstm_orders = compute_order_quantities_cvar(
        lstm_pred.point, lstm_pred.lower, lstm_pred.upper,
        beta=config.cvar.beta, n_samples=config.cvar.n_samples,
        ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
        stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
        verbose=False
    )
    results["LSTM_QR"] = compute_all_metrics(
        "LSTM_QR", y_test_seq, lstm_pred.point, lstm_orders,
        lstm_pred.lower, lstm_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )

    # Transformer
    logger.info("Running Transformer Quantile Regression...")
    trans_model = TransformerQuantileRegression(
        alpha=config.transformer.alpha,
        sequence_length=config.data.sequence_length,
        d_model=config.transformer.d_model,
        nhead=config.transformer.nhead,
        num_layers=config.transformer.num_layers,
        dropout=config.transformer.dropout,
        learning_rate=config.transformer.learning_rate,
        epochs=config.transformer.epochs,
        batch_size=config.transformer.batch_size,
        random_state=config.random_seed,
        device=config.device
    )
    trans_model.fit(X_train_seq, y_train_seq, X_cal_seq, y_cal_seq)
    trans_pred = trans_model.predict(X_test_seq)
    trans_orders = compute_order_quantities_cvar(
        trans_pred.point, trans_pred.lower, trans_pred.upper,
        beta=config.cvar.beta, n_samples=config.cvar.n_samples,
        ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
        stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
        verbose=False
    )
    results["Transformer_QR"] = compute_all_metrics(
        "Transformer_QR", y_test_seq, trans_pred.point, trans_orders,
        trans_pred.lower, trans_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )

    # TFT
    logger.info("Running Temporal Fusion Transformer...")
    tft_model = TemporalFusionTransformer(
        alpha=config.tft.alpha,
        sequence_length=config.data.sequence_length,
        hidden_size=config.tft.hidden_size,
        num_heads=config.tft.num_heads,
        num_layers=config.tft.num_layers,
        dropout=config.tft.dropout,
        learning_rate=config.tft.learning_rate,
        epochs=config.tft.epochs,
        batch_size=config.tft.batch_size,
        random_state=config.random_seed,
        device=config.device
    )
    tft_model.fit(X_train_seq, y_train_seq, X_cal_seq, y_cal_seq)
    tft_pred = tft_model.predict(X_test_seq)
    tft_orders = compute_order_quantities_cvar(
        tft_pred.point, tft_pred.lower, tft_pred.upper,
        beta=config.cvar.beta, n_samples=config.cvar.n_samples,
        ordering_cost=costs.ordering_cost, holding_cost=costs.holding_cost,
        stockout_cost=costs.stockout_cost, random_seed=config.cvar.random_seed,
        verbose=False
    )
    results["TFT"] = compute_all_metrics(
        "TFT", y_test_seq, tft_pred.point, tft_orders,
        tft_pred.lower, tft_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )

    # =========================================================================
    # SPO/END-TO-END BASELINE (CRITICAL COMPETITOR)
    # =========================================================================

    logger.info("Running SPO/End-to-End Baseline...")
    spo_model = SPOEndToEnd(
        alpha=config.lstm.alpha,  # Use LSTM config as base
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
    results["SPO_EndToEnd"] = compute_all_metrics(
        "SPO_EndToEnd", y_test_seq, spo_pred.point, spo_orders,
        spo_pred.lower, spo_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )

    # =========================================================================
    # COMPILE RESULTS
    # =========================================================================

    # Align traditional methods to sequence-aligned test data
    for name in ["Conformal_CVaR", "Normal_CVaR", "QuantileReg_CVaR", "SAA", "Seer_Oracle"]:
        result = results[name]
        results[name] = MethodResults(
            method_name=name,
            forecast_metrics=result.forecast_metrics,
            inventory_metrics=result.inventory_metrics,
            costs=result.costs[-n_test_seq:],
            order_quantities=result.order_quantities[-n_test_seq:],
            point_predictions=result.point_predictions[-n_test_seq:],
            lower_bounds=result.lower_bounds[-n_test_seq:] if result.lower_bounds is not None else None,
            upper_bounds=result.upper_bounds[-n_test_seq:] if result.upper_bounds is not None else None
        )

    # Create summary dataframe
    summary_data = []
    for method_name, result in results.items():
        row = {
            'Method': method_name,
            'Mean_Cost': result.inventory_metrics.mean_cost,
            'CVaR-90': result.inventory_metrics.cvar_90,
            'CVaR-95': result.inventory_metrics.cvar_95,
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

    return summary_df


def main(config: ExperimentConfig):
    """
    Main expanding window experiment runner.

    Parameters
    ----------
    config : ExperimentConfig
        Experiment configuration.
    """
    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)

    logger.info("=" * 70)
    logger.info("EXPANDING WINDOW CVaR OPTIMIZATION EXPERIMENT")
    logger.info("WITH SPO/END-TO-END BASELINE")
    logger.info("=" * 70)
    logger.info(f"Device: {config.device}")
    logger.info(f"Store: {config.data.store_ids[0]}, Item: {config.data.item_ids[0]}")

    # Create output directory
    os.makedirs(config.results_dir, exist_ok=True)

    # Load rolling window data (we'll use it as expanding by keeping initial date fixed)
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

    for window_split in rolling_splits:
        summary_df = run_window_experiment(window_split, config, is_expanding=True)
        all_window_results.append(summary_df)

        logger.info(f"\nWindow {window_split.window_idx} Results:")
        print(summary_df[['Method', 'Mean_Cost', 'CVaR-90', 'Coverage', 'MAE']].to_string(index=False))

    # Aggregate results across all windows
    logger.info("\n" + "=" * 70)
    logger.info("AGGREGATED RESULTS ACROSS ALL WINDOWS")
    logger.info("=" * 70)

    combined_df = pd.concat(all_window_results, ignore_index=True)

    # Compute mean and std across windows for each method
    aggregated = combined_df.groupby('Method').agg({
        'Mean_Cost': ['mean', 'std'],
        'CVaR-90': ['mean', 'std'],
        'CVaR-95': ['mean', 'std'],
        'Service_Level': ['mean', 'std'],
        'Coverage': ['mean', 'std'],
        'Interval_Width': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'MAPE': ['mean', 'std']
    }).round(2)

    print("\n", aggregated.to_string())

    # Save results
    agg_path = os.path.join(config.results_dir, "expanding_window_aggregated.csv")
    aggregated.to_csv(agg_path)
    logger.info(f"\n[OK] Saved aggregated results: {agg_path}")

    all_path = os.path.join(config.results_dir, "expanding_window_all.csv")
    combined_df.to_csv(all_path, index=False)
    logger.info(f"[OK] Saved all window results: {all_path}")

    # Print key findings
    logger.info("\n" + "=" * 70)
    logger.info("KEY FINDINGS")
    logger.info("=" * 70)

    mean_cvar90 = aggregated[('CVaR-90', 'mean')]
    best_method = mean_cvar90.idxmin()
    logger.info(f"[BEST] Lowest average CVaR-90: {best_method} (${mean_cvar90[best_method]:.2f})")

    mean_cost = aggregated[('Mean_Cost', 'mean')]
    best_cost_method = mean_cost.idxmin()
    logger.info(f"[BEST] Lowest average cost: {best_cost_method} (${mean_cost[best_cost_method]:.2f})")

    # SPO comparison
    if 'SPO_EndToEnd' in mean_cvar90.index:
        spo_cvar = mean_cvar90['SPO_EndToEnd']
        logger.info(f"\n[SPO] SPO/End-to-End CVaR-90: ${spo_cvar:.2f}")
        logger.info(f"[SPO] SPO vs Best: {((spo_cvar / mean_cvar90[best_method] - 1) * 100):+.2f}%")

    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Expanding Window CVaR Optimization Experiment with SPO")
    parser.add_argument("--output", type=str, default="results/expanding_window_spo", help="Output directory")
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs for DL models")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")

    args = parser.parse_args()

    # Create config
    config = get_default_config()
    config.results_dir = args.output

    # Enable rolling window mode (we use the same data loading)
    config.rolling_window.enabled = True

    # Set epochs for DL models
    if args.epochs:
        config.lstm.epochs = args.epochs
        config.transformer.epochs = args.epochs
        config.tft.epochs = args.epochs

    if args.device:
        config.device = args.device

    main(config)
