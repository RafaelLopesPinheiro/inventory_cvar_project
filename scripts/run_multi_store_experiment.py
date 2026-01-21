#!/usr/bin/env python
"""
Multi-Store Experiment Runner

Runs experiments across multiple store-item combinations for robust evaluation.
This is important for publication to demonstrate generalizability.

Usage:
    python run_multi_store_experiment.py --stores 1,2,3 --items 1,2,3,4,5
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    load_raw_data,
    filter_store_item,
    create_all_features,
    create_temporal_splits,
    prepare_sequence_data,
)
from src.models import (
    ConformalPrediction,
    NormalAssumption,
    SampleAverageApproximation,
    LSTMQuantileRegression,
    MCDropoutLSTM,
)
from src.optimization import compute_order_quantities_cvar
from src.evaluation import compute_all_metrics, MethodResults
from configs import get_default_config, ExperimentConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def run_single_store_item(
    df_raw: pd.DataFrame,
    store_id: int,
    item_id: int,
    config: ExperimentConfig
) -> Dict[str, MethodResults]:
    """
    Run experiment for a single store-item combination.
    
    Returns None if insufficient data.
    """
    try:
        # Filter data
        df = filter_store_item(df_raw, store_id, item_id)
        if len(df) < 365 * 3:  # Need at least 3 years of data
            return None
        
        # Create features
        df, feature_cols = create_all_features(df)
        
        # Create splits
        splits = create_temporal_splits(df, feature_cols)
        
        # Prepare sequence data
        seq_data = prepare_sequence_data(splits, seq_length=config.data.sequence_length)
        
        results = {}
        costs = config.cost
        
        X_train, y_train = splits.train.X, splits.train.y
        X_cal, y_cal = splits.calibration.X, splits.calibration.y
        X_test, y_test = splits.test.X, splits.test.y
        
        # =====================================================================
        # Conformal Prediction + CVaR
        # =====================================================================
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
            random_seed=config.cvar.random_seed,
            verbose=False
        )
        
        results["Conformal_CVaR"] = compute_all_metrics(
            "Conformal_CVaR", y_test, cp_pred.point, cp_orders,
            cp_pred.lower, cp_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
        
        # =====================================================================
        # Normal Assumption + CVaR
        # =====================================================================
        normal_model = NormalAssumption(
            alpha=config.normal.alpha,
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
            random_seed=config.cvar.random_seed,
            verbose=False
        )
        
        results["Normal_CVaR"] = compute_all_metrics(
            "Normal_CVaR", y_test, normal_pred.point, normal_orders,
            normal_pred.lower, normal_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
        
        # =====================================================================
        # SAA
        # =====================================================================
        saa_model = SampleAverageApproximation(
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
        
        # =====================================================================
        # MC Dropout LSTM (representative DL method)
        # =====================================================================
        # Use reduced epochs for multi-store to save time
        mc_model = MCDropoutLSTM(
            alpha=config.mc_dropout.alpha,
            sequence_length=config.data.sequence_length,
            hidden_size=32,  # Reduced for speed
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001,
            epochs=50,  # Reduced for speed
            batch_size=32,
            n_mc_samples=50,  # Reduced for speed
            random_state=config.random_seed,
            device=config.device
        )
        mc_model.fit(
            seq_data.X_train, seq_data.y_train,
            seq_data.X_cal, seq_data.y_cal
        )
        mc_pred = mc_model.predict(seq_data.X_test)
        
        mc_orders = compute_order_quantities_cvar(
            mc_pred.point, mc_pred.lower, mc_pred.upper,
            beta=config.cvar.beta,
            n_samples=config.cvar.n_samples,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost,
            random_seed=config.cvar.random_seed,
            verbose=False
        )
        
        # Align test data
        y_test_aligned = seq_data.y_test
        results["MCDropout_LSTM"] = compute_all_metrics(
            "MCDropout_LSTM", y_test_aligned, mc_pred.point, mc_orders,
            mc_pred.lower, mc_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
        
        return results
        
    except Exception as e:
        logger.warning(f"Error processing store {store_id}, item {item_id}: {e}")
        return None


def run_multi_store_experiment(
    filepath: str,
    store_ids: List[int],
    item_ids: List[int],
    config: ExperimentConfig,
    output_dir: str = "results/multi_store"
) -> pd.DataFrame:
    """
    Run experiments across multiple store-item combinations.
    
    Parameters
    ----------
    filepath : str
        Path to data file.
    store_ids : List[int]
        List of store IDs.
    item_ids : List[int]
        List of item IDs.
    config : ExperimentConfig
        Experiment configuration.
    output_dir : str
        Output directory.
        
    Returns
    -------
    pd.DataFrame
        Aggregated results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("MULTI-STORE EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Stores: {store_ids}")
    logger.info(f"Items: {item_ids}")
    logger.info(f"Total combinations: {len(store_ids) * len(item_ids)}")
    
    # Load raw data once
    df_raw = load_raw_data(filepath)
    
    # Collect all results
    all_results = []
    
    total = len(store_ids) * len(item_ids)
    with tqdm(total=total, desc="Processing") as pbar:
        for store_id in store_ids:
            for item_id in item_ids:
                results = run_single_store_item(df_raw, store_id, item_id, config)
                
                if results is not None:
                    for method_name, method_results in results.items():
                        row = {
                            "store_id": store_id,
                            "item_id": item_id,
                            "method": method_name,
                            "coverage": method_results.forecast_metrics.coverage,
                            "avg_interval_width": method_results.forecast_metrics.avg_interval_width,
                            "mean_cost": method_results.inventory_metrics.mean_cost,
                            "cvar_90": method_results.inventory_metrics.cvar_90,
                            "cvar_95": method_results.inventory_metrics.cvar_95,
                            "service_level": method_results.inventory_metrics.service_level,
                            "total_cost": method_results.inventory_metrics.total_cost,
                        }
                        all_results.append(row)
                
                pbar.update(1)
    
    # Create DataFrame
    results_df = pd.DataFrame(all_results)
    
    # Save detailed results
    detailed_path = os.path.join(output_dir, "detailed_results.csv")
    results_df.to_csv(detailed_path, index=False)
    logger.info(f"Saved detailed results: {detailed_path}")
    
    # Create summary statistics
    summary = results_df.groupby("method").agg({
        "coverage": ["mean", "std"],
        "mean_cost": ["mean", "std"],
        "cvar_90": ["mean", "std"],
        "cvar_95": ["mean", "std"],
        "service_level": ["mean", "std"],
    }).round(4)
    
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary.to_csv(summary_path)
    logger.info(f"Saved summary statistics: {summary_path}")
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY STATISTICS (Mean ± Std)")
    logger.info("=" * 70)
    print(summary.to_string())
    
    # Method comparison
    logger.info("\n" + "=" * 70)
    logger.info("METHOD RANKINGS")
    logger.info("=" * 70)
    
    mean_cvar90 = results_df.groupby("method")["cvar_90"].mean().sort_values()
    logger.info("\nBy CVaR-90 (lower is better):")
    for i, (method, value) in enumerate(mean_cvar90.items(), 1):
        logger.info(f"  {i}. {method}: ${value:.2f}")
    
    mean_coverage = results_df.groupby("method")["coverage"].mean()
    coverage_gap = abs(mean_coverage - 0.95).sort_values()
    logger.info("\nBy Coverage (closest to 95% is better):")
    for i, (method, gap) in enumerate(coverage_gap.items(), 1):
        actual = mean_coverage[method]
        if not np.isnan(actual):
            logger.info(f"  {i}. {method}: {actual:.1%} (gap: {gap:.1%})")
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Multi-Store Experiment")
    parser.add_argument("--data", type=str, default="train.csv", help="Data file path")
    parser.add_argument("--stores", type=str, default="1,2,3", help="Comma-separated store IDs")
    parser.add_argument("--items", type=str, default="1,2,3,4,5", help="Comma-separated item IDs")
    parser.add_argument("--output", type=str, default="results/multi_store", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    
    args = parser.parse_args()
    
    store_ids = [int(s) for s in args.stores.split(",")]
    item_ids = [int(i) for i in args.items.split(",")]
    
    config = get_default_config()
    config.device = args.device
    
    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    run_multi_store_experiment(
        filepath=args.data,
        store_ids=store_ids,
        item_ids=item_ids,
        config=config,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()
