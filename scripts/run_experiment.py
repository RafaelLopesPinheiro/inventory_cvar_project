#!/usr/bin/env python
"""
Main Experiment Runner for Inventory CVaR Optimization

This script runs the comprehensive comparison study between:
- Traditional methods (Conformal, Normal, Quantile Regression, SAA)
- Deep learning methods (LSTM, Transformer, DeepEnsemble, MC Dropout)

Usage:
    python run_experiment.py [--config CONFIG_PATH] [--output OUTPUT_DIR]
"""

import argparse
import logging
import os
import sys
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import pandas as pd
import torch

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    load_and_prepare_data,
    load_and_prepare_rolling_data,
    prepare_sequence_data,
    prepare_rolling_sequence_data,
    TemporalSplits,
    SequenceData,
    RollingWindowSplit,
)
from src.models import (
    ConformalPrediction,
    NormalAssumption,
    QuantileRegression,
    SampleAverageApproximation,
    ExpectedValue,
    LSTMQuantileRegression,
    TransformerQuantileRegression,
    DeepEnsemble,
    MCDropoutLSTM,
    TemporalFusionTransformer,
    PredictionResult,
)
from src.optimization import (
    compute_order_quantities_cvar,
    CostParameters,
)
from src.evaluation import (
    compute_all_metrics,
    compare_methods,
    create_results_summary,
    MethodResults,
)
from src.visualization import create_comprehensive_visualization
from configs import get_default_config, ExperimentConfig


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_traditional_methods(
    splits: TemporalSplits,
    config: ExperimentConfig
) -> Dict[str, MethodResults]:
    """
    Run all traditional (non-DL) methods.
    
    Parameters
    ----------
    splits : TemporalSplits
        Train/calibration/test data splits.
    config : ExperimentConfig
        Experiment configuration.
        
    Returns
    -------
    Dict[str, MethodResults]
        Results for each method.
    """
    results = {}
    costs = config.cost
    
    X_train, y_train = splits.train.X, splits.train.y
    X_cal, y_cal = splits.calibration.X, splits.calibration.y
    X_test, y_test = splits.test.X, splits.test.y
    
    # =========================================================================
    # METHOD 1: Conformal Prediction + CVaR
    # =========================================================================
    logger.info("=" * 70)
    logger.info("METHOD 1: CONFORMAL PREDICTION + CVaR")
    logger.info("=" * 70)
    
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
        beta=config.cvar.beta,
        n_samples=config.cvar.n_samples,
        ordering_cost=costs.ordering_cost,
        holding_cost=costs.holding_cost,
        stockout_cost=costs.stockout_cost,
        random_seed=config.cvar.random_seed
    )
    
    results["Conformal_CVaR"] = compute_all_metrics(
        "Conformal_CVaR", y_test, cp_pred.point, cp_orders,
        cp_pred.lower, cp_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )
    
    # =========================================================================
    # METHOD 2: Normal Assumption + CVaR
    # =========================================================================
    logger.info("=" * 70)
    logger.info("METHOD 2: NORMAL ASSUMPTION + CVaR")
    logger.info("=" * 70)
    
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
        beta=config.cvar.beta,
        n_samples=config.cvar.n_samples,
        ordering_cost=costs.ordering_cost,
        holding_cost=costs.holding_cost,
        stockout_cost=costs.stockout_cost,
        random_seed=config.cvar.random_seed
    )
    
    results["Normal_CVaR"] = compute_all_metrics(
        "Normal_CVaR", y_test, normal_pred.point, normal_orders,
        normal_pred.lower, normal_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )
    
    # =========================================================================
    # METHOD 3: Quantile Regression + CVaR
    # =========================================================================
    logger.info("=" * 70)
    logger.info("METHOD 3: QUANTILE REGRESSION + CVaR")
    logger.info("=" * 70)
    
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
        beta=config.cvar.beta,
        n_samples=config.cvar.n_samples,
        ordering_cost=costs.ordering_cost,
        holding_cost=costs.holding_cost,
        stockout_cost=costs.stockout_cost,
        random_seed=config.cvar.random_seed
    )
    
    results["QuantileReg_CVaR"] = compute_all_metrics(
        "QuantileReg_CVaR", y_test, qr_pred.point, qr_orders,
        qr_pred.lower, qr_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )
    
    # =========================================================================
    # METHOD 4: Sample Average Approximation (SAA)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("METHOD 4: SAMPLE AVERAGE APPROXIMATION (SAA)")
    logger.info("=" * 70)
    
    saa_model = SampleAverageApproximation(
        n_estimators=100,
        max_depth=10,
        random_state=config.random_seed,
        stockout_cost=costs.stockout_cost,
        holding_cost=costs.holding_cost
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
    # METHOD 5: Expected Value (Risk-Neutral)
    # =========================================================================
    logger.info("=" * 70)
    logger.info("METHOD 5: EXPECTED VALUE (RISK-NEUTRAL)")
    logger.info("=" * 70)
    
    ev_model = ExpectedValue(
        n_estimators=100,
        max_depth=10,
        random_state=config.random_seed
    )
    ev_model.fit(X_train, y_train, X_cal, y_cal)
    ev_pred = ev_model.predict(X_test)
    ev_orders = ev_pred.point  # Order = predicted demand
    
    results["ExpectedValue"] = compute_all_metrics(
        "ExpectedValue", y_test, ev_pred.point, ev_orders,
        None, None,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )
    
    return results


def run_deep_learning_methods(
    seq_data: SequenceData,
    y_test: np.ndarray,
    config: ExperimentConfig
) -> tuple[Dict[str, MethodResults], Dict[str, List[float]]]:
    """
    Run all deep learning methods.
    
    Parameters
    ----------
    seq_data : SequenceData
        Sequence data for DL models.
    y_test : np.ndarray
        Test targets (aligned with sequences).
    config : ExperimentConfig
        Experiment configuration.
        
    Returns
    -------
    Tuple[Dict[str, MethodResults], Dict[str, List[float]]]
        Results and training losses for each method.
    """
    results = {}
    training_losses = {}
    costs = config.cost
    
    X_train, y_train = seq_data.X_train, seq_data.y_train
    X_cal, y_cal = seq_data.X_cal, seq_data.y_cal
    X_test = seq_data.X_test
    
    # =========================================================================
    # METHOD: LSTM Quantile Regression
    # =========================================================================
    logger.info("=" * 70)
    logger.info("METHOD: LSTM QUANTILE REGRESSION")
    logger.info("=" * 70)
    
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
    lstm_model.fit(X_train, y_train, X_cal, y_cal)
    lstm_pred = lstm_model.predict(X_test)
    
    lstm_orders = compute_order_quantities_cvar(
        lstm_pred.point, lstm_pred.lower, lstm_pred.upper,
        beta=config.cvar.beta,
        n_samples=config.cvar.n_samples,
        ordering_cost=costs.ordering_cost,
        holding_cost=costs.holding_cost,
        stockout_cost=costs.stockout_cost,
        random_seed=config.cvar.random_seed
    )
    
    results["LSTM_QR"] = compute_all_metrics(
        "LSTM_QR", y_test, lstm_pred.point, lstm_orders,
        lstm_pred.lower, lstm_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )
    training_losses["LSTM_QR"] = lstm_model.training_losses
    
    # =========================================================================
    # METHOD: Transformer Quantile Regression
    # =========================================================================
    logger.info("=" * 70)
    logger.info("METHOD: TRANSFORMER QUANTILE REGRESSION")
    logger.info("=" * 70)
    
    transformer_model = TransformerQuantileRegression(
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
    transformer_model.fit(X_train, y_train, X_cal, y_cal)
    trans_pred = transformer_model.predict(X_test)
    
    trans_orders = compute_order_quantities_cvar(
        trans_pred.point, trans_pred.lower, trans_pred.upper,
        beta=config.cvar.beta,
        n_samples=config.cvar.n_samples,
        ordering_cost=costs.ordering_cost,
        holding_cost=costs.holding_cost,
        stockout_cost=costs.stockout_cost,
        random_seed=config.cvar.random_seed
    )
    
    results["Transformer_QR"] = compute_all_metrics(
        "Transformer_QR", y_test, trans_pred.point, trans_orders,
        trans_pred.lower, trans_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )
    training_losses["Transformer_QR"] = transformer_model.training_losses
    
    # =========================================================================
    # METHOD: Deep Ensemble
    # =========================================================================
    logger.info("=" * 70)
    logger.info("METHOD: DEEP ENSEMBLE")
    logger.info("=" * 70)
    
    ensemble_model = DeepEnsemble(
        alpha=config.deep_ensemble.alpha,
        sequence_length=config.data.sequence_length,
        n_ensemble=config.deep_ensemble.n_ensemble,
        hidden_size=config.deep_ensemble.hidden_size,
        learning_rate=config.deep_ensemble.learning_rate,
        epochs=config.deep_ensemble.epochs,
        batch_size=config.deep_ensemble.batch_size,
        random_state=config.random_seed,
        device=config.device
    )
    ensemble_model.fit(X_train, y_train, X_cal, y_cal)
    ensemble_pred = ensemble_model.predict(X_test)
    
    ensemble_orders = compute_order_quantities_cvar(
        ensemble_pred.point, ensemble_pred.lower, ensemble_pred.upper,
        beta=config.cvar.beta,
        n_samples=config.cvar.n_samples,
        ordering_cost=costs.ordering_cost,
        holding_cost=costs.holding_cost,
        stockout_cost=costs.stockout_cost,
        random_seed=config.cvar.random_seed
    )
    
    results["DeepEnsemble"] = compute_all_metrics(
        "DeepEnsemble", y_test, ensemble_pred.point, ensemble_orders,
        ensemble_pred.lower, ensemble_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )
    training_losses["DeepEnsemble"] = []
    
    # =========================================================================
    # METHOD: MC Dropout LSTM
    # =========================================================================
    logger.info("=" * 70)
    logger.info("METHOD: MC DROPOUT LSTM")
    logger.info("=" * 70)
    
    mc_model = MCDropoutLSTM(
        alpha=config.mc_dropout.alpha,
        sequence_length=config.data.sequence_length,
        hidden_size=config.mc_dropout.hidden_size,
        num_layers=config.mc_dropout.num_layers,
        dropout=config.mc_dropout.dropout,
        learning_rate=config.mc_dropout.learning_rate,
        epochs=config.mc_dropout.epochs,
        batch_size=config.mc_dropout.batch_size,
        n_mc_samples=config.mc_dropout.n_mc_samples,
        random_state=config.random_seed,
        device=config.device
    )
    mc_model.fit(X_train, y_train, X_cal, y_cal)
    mc_pred = mc_model.predict(X_test)
    
    mc_orders = compute_order_quantities_cvar(
        mc_pred.point, mc_pred.lower, mc_pred.upper,
        beta=config.cvar.beta,
        n_samples=config.cvar.n_samples,
        ordering_cost=costs.ordering_cost,
        holding_cost=costs.holding_cost,
        stockout_cost=costs.stockout_cost,
        random_seed=config.cvar.random_seed
    )
    
    results["MCDropout_LSTM"] = compute_all_metrics(
        "MCDropout_LSTM", y_test, mc_pred.point, mc_orders,
        mc_pred.lower, mc_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )
    training_losses["MCDropout_LSTM"] = []

    # =========================================================================
    # METHOD: Temporal Fusion Transformer
    # =========================================================================
    logger.info("=" * 70)
    logger.info("METHOD: TEMPORAL FUSION TRANSFORMER (TFT)")
    logger.info("=" * 70)

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
    tft_model.fit(X_train, y_train, X_cal, y_cal)
    tft_pred = tft_model.predict(X_test)

    tft_orders = compute_order_quantities_cvar(
        tft_pred.point, tft_pred.lower, tft_pred.upper,
        beta=config.cvar.beta,
        n_samples=config.cvar.n_samples,
        ordering_cost=costs.ordering_cost,
        holding_cost=costs.holding_cost,
        stockout_cost=costs.stockout_cost,
        random_seed=config.cvar.random_seed
    )

    results["TFT"] = compute_all_metrics(
        "TFT", y_test, tft_pred.point, tft_orders,
        tft_pred.lower, tft_pred.upper,
        costs.ordering_cost, costs.holding_cost, costs.stockout_cost
    )
    training_losses["TFT"] = tft_model.training_losses

    return results, training_losses


def print_key_findings(results_dict: Dict[str, MethodResults]):
    """Print key findings from the experiment."""
    logger.info("=" * 70)
    logger.info("KEY FINDINGS")
    logger.info("=" * 70)
    
    # Best coverage
    coverage_methods = [
        (name, r.forecast_metrics.coverage)
        for name, r in results_dict.items()
        if r.forecast_metrics.coverage is not None
    ]
    if coverage_methods:
        best_coverage = min(coverage_methods, key=lambda x: abs(x[1] - 0.95))
        logger.info(f"[BEST] Coverage (closest to 95%): {best_coverage[0]} ({best_coverage[1]:.1%})")
    
    # Lowest CVaR-90
    best_cvar = min(results_dict.items(), key=lambda x: x[1].inventory_metrics.cvar_90)
    logger.info(f"[BEST] Lowest CVaR-90: {best_cvar[0]} (${best_cvar[1].inventory_metrics.cvar_90:.2f})")
    
    # Lowest mean cost
    best_mean = min(results_dict.items(), key=lambda x: x[1].inventory_metrics.mean_cost)
    logger.info(f"[BEST] Lowest Mean Cost: {best_mean[0]} (${best_mean[1].inventory_metrics.mean_cost:.2f})")
    
    # DL vs Traditional comparison
    dl_methods = ["LSTM_QR", "Transformer_QR", "DeepEnsemble", "MCDropout_LSTM", "TFT"]
    trad_methods = ["Conformal_CVaR", "Normal_CVaR", "QuantileReg_CVaR", "SAA"]
    
    dl_cvar90 = np.mean([
        results_dict[m].inventory_metrics.cvar_90 
        for m in dl_methods if m in results_dict
    ])
    trad_cvar90 = np.mean([
        results_dict[m].inventory_metrics.cvar_90 
        for m in trad_methods if m in results_dict
    ])
    
    logger.info(f"\nAverage Traditional CVaR-90: ${trad_cvar90:.2f}")
    logger.info(f"Average DL CVaR-90: ${dl_cvar90:.2f}")
    
    if dl_cvar90 < trad_cvar90:
        improvement = (trad_cvar90 - dl_cvar90) / trad_cvar90 * 100
        logger.info(f"[RESULT] DL methods reduce tail risk by {improvement:.1f}%")
    else:
        increase = (dl_cvar90 - trad_cvar90) / trad_cvar90 * 100
        logger.info(f"[RESULT] Traditional methods have {increase:.1f}% lower tail risk")


def main(config: Optional[ExperimentConfig] = None):
    """
    Main experiment runner.
    
    Parameters
    ----------
    config : ExperimentConfig, optional
        Experiment configuration. Uses default if not provided.
    """
    if config is None:
        config = get_default_config()
    
    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(config.random_seed)
    
    logger.info("=" * 70)
    logger.info("INVENTORY CVaR OPTIMIZATION EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Device: {config.device}")
    logger.info(f"Cost Parameters: Ordering=${config.cost.ordering_cost}, "
                f"Holding=${config.cost.holding_cost}, "
                f"Stockout=${config.cost.stockout_cost}")
    logger.info("=" * 70)
    
    # Create output directory
    os.makedirs(config.results_dir, exist_ok=True)
    
    # Load and prepare data
    logger.info("Loading and preparing data...")
    splits = load_and_prepare_data(
        filepath=config.data.filepath,
        store_id=config.data.store_ids[0],
        item_id=config.data.item_ids[0],
        lag_periods=config.data.lag_features,
        rolling_windows=config.data.rolling_windows,
        train_years=config.data.train_years,
        cal_years=config.data.cal_years,
        test_years=config.data.test_years
    )
    
    # Prepare sequence data for DL models
    seq_data = prepare_sequence_data(
        splits,
        seq_length=config.data.sequence_length,
        prediction_horizon=config.data.prediction_horizon
    )
    
    # Run traditional methods
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING TRADITIONAL METHODS")
    logger.info("=" * 70)
    traditional_results = run_traditional_methods(splits, config)
    
    # Run deep learning methods
    logger.info("\n" + "=" * 70)
    logger.info("RUNNING DEEP LEARNING METHODS")
    logger.info("=" * 70)
    dl_results, training_losses = run_deep_learning_methods(
        seq_data, seq_data.y_test, config
    )
    
    # Combine results
    # Note: Need to align test data for proper comparison
    # Traditional methods use full test set, DL methods use sequence-aligned test set
    all_results = {}
    
    # For traditional methods, align to sequence-based test data
    for name, result in traditional_results.items():
        n_test = len(seq_data.y_test)
        aligned_result = MethodResults(
            method_name=name,
            forecast_metrics=result.forecast_metrics,
            inventory_metrics=result.inventory_metrics,
            costs=result.costs[-n_test:],
            order_quantities=result.order_quantities[-n_test:],
            point_predictions=result.point_predictions[-n_test:],
            lower_bounds=result.lower_bounds[-n_test:] if result.lower_bounds is not None else None,
            upper_bounds=result.upper_bounds[-n_test:] if result.upper_bounds is not None else None
        )
        all_results[name] = aligned_result
    
    all_results.update(dl_results)
    
    # Create results summary
    logger.info("\n" + "=" * 70)
    logger.info("RESULTS SUMMARY")
    logger.info("=" * 70)
    
    summary_df = create_results_summary(all_results)
    print("\n", summary_df.to_string(index=False))
    
    # Save results
    summary_path = os.path.join(config.results_dir, "results_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    logger.info(f"\n[OK] Saved: {summary_path}")
    
    # Statistical testing
    logger.info("\n" + "=" * 70)
    logger.info("STATISTICAL TESTING")
    logger.info("=" * 70)
    
    test_results = compare_methods(all_results, baseline_name="Conformal_CVaR")
    for result in test_results:
        logger.info(str(result))
    
    # Visualization
    logger.info("\n" + "=" * 70)
    logger.info("GENERATING VISUALIZATIONS")
    logger.info("=" * 70)
    
    create_comprehensive_visualization(
        seq_data.y_test,
        all_results,
        training_losses,
        output_dir=config.results_dir
    )
    
    # Key findings
    print_key_findings(all_results)
    
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)
    
    return all_results, summary_df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Inventory CVaR Optimization Experiment")
    parser.add_argument("--output", type=str, default="results", help="Output directory")
    parser.add_argument("--epochs", type=int, default=100, help="Training epochs for DL models")
    parser.add_argument("--device", type=str, default=None, help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    # Create config with command line overrides
    config = get_default_config()
    config.results_dir = args.output
    
    if args.epochs:
        config.lstm.epochs = args.epochs
        config.transformer.epochs = args.epochs
        config.mc_dropout.epochs = args.epochs
        config.tft.epochs = args.epochs

    if args.device:
        config.device = args.device

    main(config)
