#!/usr/bin/env python
"""
Comprehensive Multi-SKU Expanding Window Experiment with Inventory Dynamics

This script compares 6 forecasting/optimization methods for inventory management
using expanding window cross-validation, enriched with carryover and capacity
constraints that model realistic warehouse operations.

INVENTORY DYNAMICS:
===================
Unlike the standard single-period newsvendor, this experiment models:
- Carryover: Leftover inventory carries to the next period (with optional decay)
- Capacity: Warehouse has a maximum storage limit
- Sequential decisions: Each period's order depends on current inventory state

MODEL HIERARCHY (6 Methods):
=============================
1. SAA                 - Sample Average Approximation (OR benchmark)
2. Conformal + CVaR    - Conformal Prediction intervals + CVaR optimization
3. Wasserstein DRO     - Distributionally Robust Optimization
4. EnbPI + CQR + CVaR  - Ensemble Batch PI + Conformalized Quantile Regression + CVaR
5. SPO (End-to-End)    - Smart Predict-then-Optimize (decision-focused learning)
6. Seer               - Oracle upper bound (perfect foresight)

STATISTICAL VALIDITY:
=====================
- Paired t-tests and Wilcoxon signed-rank tests across windows
- Bonferroni correction for multiple comparisons
- Effect size (Cohen's d) for practical significance

KEY EXPERIMENTAL DESIGN:
========================
- Multi-SKU: Runs across multiple store-item combinations
- Expanding Window: Training set grows over time (not sliding)
- Carryover: Leftover inventory carries between periods (configurable decay)
- Capacity: Maximum warehouse storage constraint
- Metrics: Mean Cost, CVaR-90, CVaR-95, Service Level, Capacity Utilization
- Visualizations: Comprehensive graphs for analysis

Usage:
    python run_comprehensive_expanding_window.py
    python run_comprehensive_expanding_window.py --stores 1,2,3 --items 1,2,3
    python run_comprehensive_expanding_window.py --carryover 0.9 --capacity 150
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
)
from src.models import (
    SampleAverageApproximation,
    ConformalPrediction,
    EnsembleBatchPI,
    DistributionallyRobustOptimization,
    SPOEndToEnd,
    Seer,
    PredictionResult,
)
from src.optimization import (
    compute_order_quantities_cvar,
    CostParameters,
    simulate_inventory_with_carryover,
    InventorySimulationResult,
)
from src.evaluation import (
    compute_all_metrics,
    MethodResults,
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
# MODEL DEFINITIONS (5 METHODS ONLY)
# =============================================================================

MODEL_CATEGORIES = {
    "1_OR_Standard": ["SAA"],
    "2_DistributionFree": ["Conformal_CVaR"],
    "3_RobustOptimization": ["Wasserstein_DRO"],
    "4_YourContribution": ["EnbPI_CQR_CVaR"],
    "5_EndToEnd": ["SPO_EndToEnd"],
    "6_Oracle": ["Seer"],
}

MODEL_ORDER = ['SAA', 'Conformal_CVaR', 'Wasserstein_DRO', 'EnbPI_CQR_CVaR', 'SPO_EndToEnd', 'Seer']

MODEL_DISPLAY_NAMES = {
    "SAA": "1. SAA",
    "Conformal_CVaR": "2. Conformal + CVaR",
    "Wasserstein_DRO": "3. Wasserstein DRO",
    "EnbPI_CQR_CVaR": "4. EnbPI+CQR+CVaR",
    "SPO_EndToEnd": "5. SPO (End-to-End)",
    "Seer": "6. Seer (Oracle)",
}

MODEL_COLORS = {
    "SAA": "#1f77b4",
    "Conformal_CVaR": "#ff7f0e",
    "Wasserstein_DRO": "#9467bd",
    "EnbPI_CQR_CVaR": "#d62728",
    "SPO_EndToEnd": "#e377c2",
    "Seer": "#2ca02c",
}


def get_model_display_name(method_name: str) -> str:
    """Get a clean display name for a method."""
    return MODEL_DISPLAY_NAMES.get(method_name, method_name)


# =============================================================================
# DATA LOADING
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

    Returns
    -------
    Dict[Tuple[int, int], List[RollingWindowSplit]]
        Dictionary mapping (store_id, item_id) to list of rolling window splits.
    """
    logger.info(f"Loading expanding window data for {len(store_ids)} stores x {len(item_ids)} items")

    df_raw = load_raw_data(filepath)
    results = {}
    skipped = []

    total = len(store_ids) * len(item_ids)
    with tqdm(total=total, desc="Loading SKU data") as pbar:
        for store_id in store_ids:
            for item_id in item_ids:
                try:
                    df = filter_store_item(df_raw, store_id, item_id)
                    if len(df) < min_records:
                        skipped.append((store_id, item_id, f"insufficient data ({len(df)} < {min_records})"))
                        pbar.update(1)
                        continue

                    df, feature_cols = create_all_features(
                        df, lag_periods=lag_periods, rolling_windows=rolling_windows
                    )

                    splits = create_rolling_window_splits(
                        df, feature_cols,
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
# EXPERIMENT RUNNER WITH INVENTORY DYNAMICS
# =============================================================================

def run_single_window(
    window_split: RollingWindowSplit,
    config: ExperimentConfig,
) -> Tuple[pd.DataFrame, Dict[str, InventorySimulationResult]]:
    """
    Run all 6 models on a single expanding window with carryover and capacity.

    Parameters
    ----------
    window_split : RollingWindowSplit
        Window split data.
    config : ExperimentConfig
        Experiment configuration.

    Returns
    -------
    Tuple[pd.DataFrame, Dict[str, InventorySimulationResult]]
        Summary dataframe and detailed simulation results per method.
    """
    results = {}
    sim_results = {}
    timings = {}
    costs = config.cost

    X_train, y_train = window_split.train.X, window_split.train.y
    X_cal, y_cal = window_split.calibration.X, window_split.calibration.y
    X_test, y_test = window_split.test.X, window_split.test.y

    # =========================================================================
    # 1. SAA (SAMPLE AVERAGE APPROXIMATION)
    # =========================================================================
    try:
        start_time = time.time()
        saa_model = SampleAverageApproximation(
            n_estimators=100, max_depth=10,
            stockout_cost=costs.stockout_cost,
            holding_cost=costs.holding_cost,
            random_state=config.random_seed
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
            stockout_cost=costs.stockout_cost
        )
        timings["SAA"] = time.time() - start_time
        sim_results["SAA"] = saa_sim
        results["SAA"] = {
            'pred': saa_pred, 'target_orders': saa_orders,
            'sim': saa_sim, 'time': timings["SAA"]
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

        cp_sim = simulate_inventory_with_carryover(
            cp_orders, y_test,
            initial_inventory=costs.initial_inventory,
            carryover_rate=costs.carryover_rate,
            capacity=costs.capacity,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost
        )
        timings["Conformal_CVaR"] = time.time() - start_time
        sim_results["Conformal_CVaR"] = cp_sim
        results["Conformal_CVaR"] = {
            'pred': cp_pred, 'target_orders': cp_orders,
            'sim': cp_sim, 'time': timings["Conformal_CVaR"]
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
            random_state=config.random_seed
        )
        dro_model.fit(X_train, y_train, X_cal, y_cal)
        dro_pred = dro_model.predict(X_test)
        dro_orders = dro_model.compute_order_quantities(
            X_test, dro_pred.point, dro_pred.lower, dro_pred.upper
        )

        dro_sim = simulate_inventory_with_carryover(
            dro_orders, y_test,
            initial_inventory=costs.initial_inventory,
            carryover_rate=costs.carryover_rate,
            capacity=costs.capacity,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost
        )
        timings["Wasserstein_DRO"] = time.time() - start_time
        sim_results["Wasserstein_DRO"] = dro_sim
        results["Wasserstein_DRO"] = {
            'pred': dro_pred, 'target_orders': dro_orders,
            'sim': dro_sim, 'time': timings["Wasserstein_DRO"]
        }
    except Exception as e:
        logger.debug(f"Wasserstein DRO failed: {e}")

    # =========================================================================
    # 4. EnbPI + CQR + CVaR (YOUR CONTRIBUTION)
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

        enbpi_sim = simulate_inventory_with_carryover(
            enbpi_orders, y_test,
            initial_inventory=costs.initial_inventory,
            carryover_rate=costs.carryover_rate,
            capacity=costs.capacity,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost
        )
        timings["EnbPI_CQR_CVaR"] = time.time() - start_time
        sim_results["EnbPI_CQR_CVaR"] = enbpi_sim
        results["EnbPI_CQR_CVaR"] = {
            'pred': enbpi_pred, 'target_orders': enbpi_orders,
            'sim': enbpi_sim, 'time': timings["EnbPI_CQR_CVaR"]
        }
    except Exception as e:
        logger.debug(f"EnbPI+CQR+CVaR failed: {e}")

    # =========================================================================
    # 5. SPO/END-TO-END (DECISION-FOCUSED LEARNING)
    # =========================================================================
    try:
        start_time = time.time()

        # Prepare sequence data for SPO (LSTM-based model)
        seq_data = prepare_rolling_sequence_data(
            window_split,
            seq_length=config.data.sequence_length,
            prediction_horizon=config.data.prediction_horizon
        )
        X_train_seq, y_train_seq = seq_data.X_train, seq_data.y_train
        X_cal_seq, y_cal_seq = seq_data.X_cal, seq_data.y_cal
        X_test_seq, y_test_seq = seq_data.X_test, seq_data.y_test

        spo_model = SPOEndToEnd(
            alpha=config.conformal.alpha,
            sequence_length=config.data.sequence_length,
            hidden_size=64,
            num_layers=2,
            dropout=0.2,
            learning_rate=0.001,
            epochs=getattr(config, '_spo_epochs', 50),
            batch_size=32,
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

        # Align to same test length as traditional methods
        n_test_trad = len(y_test)
        n_test_seq = len(y_test_seq)
        # Use the sequence-aligned test set for simulation
        spo_demand = y_test[-n_test_seq:] if n_test_seq < n_test_trad else y_test_seq

        spo_sim = simulate_inventory_with_carryover(
            spo_orders, spo_demand,
            initial_inventory=costs.initial_inventory,
            carryover_rate=costs.carryover_rate,
            capacity=costs.capacity,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost
        )
        timings["SPO_EndToEnd"] = time.time() - start_time
        sim_results["SPO_EndToEnd"] = spo_sim
        results["SPO_EndToEnd"] = {
            'pred': spo_pred, 'target_orders': spo_orders,
            'sim': spo_sim, 'time': timings["SPO_EndToEnd"]
        }
    except Exception as e:
        logger.debug(f"SPO/End-to-End failed: {e}")

    # =========================================================================
    # 6. SEER (ORACLE - UPPER BOUND)
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

        seer_sim = simulate_inventory_with_carryover(
            seer_orders, y_test,
            initial_inventory=costs.initial_inventory,
            carryover_rate=costs.carryover_rate,
            capacity=costs.capacity,
            ordering_cost=costs.ordering_cost,
            holding_cost=costs.holding_cost,
            stockout_cost=costs.stockout_cost
        )
        timings["Seer"] = time.time() - start_time
        sim_results["Seer"] = seer_sim
        results["Seer"] = {
            'pred': seer_pred, 'target_orders': seer_orders,
            'sim': seer_sim, 'time': timings["Seer"]
        }
    except Exception as e:
        logger.debug(f"Seer failed: {e}")

    # =========================================================================
    # CREATE SUMMARY DATAFRAME
    # =========================================================================
    summary_data = []
    for method_name, result_data in results.items():
        sim = result_data['sim']
        row = {
            'Method': method_name,
            'DisplayName': get_model_display_name(method_name),
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
    verbose: bool = False
) -> Tuple[pd.DataFrame, Dict]:
    """
    Run expanding window experiment for a single SKU.

    Returns
    -------
    Tuple[pd.DataFrame, Dict]
        Results for all windows and last window's simulation results.
    """
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
# VISUALIZATION FUNCTIONS
# =============================================================================

def create_comprehensive_visualizations(
    combined_df: pd.DataFrame,
    aggregated: pd.DataFrame,
    last_sim_results: Dict[str, InventorySimulationResult],
    output_dir: str,
    config: ExperimentConfig,
    multi_sku: bool = False
):
    """
    Create comprehensive visualizations including inventory dynamics graphs.
    """
    logger.info("\nCreating visualizations...")

    plt.style.use('seaborn-v0_8-whitegrid')
    sns.set_palette("husl")

    existing_methods = [m for m in MODEL_ORDER if m in combined_df['Method'].unique()]

    # =========================================================================
    # 1. CVaR-90 Comparison (Boxplot)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    df_plot = combined_df[combined_df['Method'].isin(existing_methods)].copy()
    df_plot['Method'] = pd.Categorical(df_plot['Method'], categories=existing_methods, ordered=True)

    colors = [MODEL_COLORS.get(m, 'steelblue') for m in existing_methods]
    bp = ax.boxplot(
        [df_plot[df_plot['Method'] == m]['CVaR_90'].dropna().values for m in existing_methods],
        labels=[get_model_display_name(m) for m in existing_methods],
        patch_artist=True, widths=0.6
    )
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    ax.set_title('CVaR-90 Comparison Across All Windows\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.set_ylabel('CVaR-90 ($)', fontsize=12)
    ax.tick_params(axis='x', rotation=20)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cvar90_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 2. Mean Cost Comparison (Bar Chart with Error Bars)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))
    agg_flat = aggregated.reset_index()

    means = []
    stds = []
    bar_colors = []
    labels = []
    for method in existing_methods:
        if method in agg_flat['Method'].values:
            try:
                mean_val = agg_flat[agg_flat['Method'] == method][('Mean_Cost', 'mean')].values[0]
                std_val = agg_flat[agg_flat['Method'] == method][('Mean_Cost', 'std')].values[0]
            except (KeyError, IndexError):
                mean_val = combined_df[combined_df['Method'] == method]['Mean_Cost'].mean()
                std_val = combined_df[combined_df['Method'] == method]['Mean_Cost'].std()
            means.append(mean_val)
            stds.append(std_val)
            bar_colors.append(MODEL_COLORS.get(method, 'steelblue'))
            labels.append(get_model_display_name(method))

    x_pos = range(len(labels))
    bars = ax.bar(x_pos, means, yerr=stds, color=bar_colors, alpha=0.8, capsize=5, edgecolor='black', linewidth=0.5)
    ax.set_xticks(x_pos)
    ax.set_xticklabels(labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Mean Cost ($)', fontsize=12)
    ax.set_title('Mean Cost Comparison (with Carryover & Capacity)\n(Lower is Better)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, means):
        ax.annotate(f'${val:.1f}', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'mean_cost_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 3. Cost Breakdown by Component (Stacked Bar Chart)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    ordering_costs = []
    holding_costs_list = []
    stockout_costs = []
    method_labels = []

    for method in existing_methods:
        if method in combined_df['Method'].values:
            m_df = combined_df[combined_df['Method'] == method]
            ordering_costs.append(m_df['Total_Ordering_Cost'].mean())
            holding_costs_list.append(m_df['Total_Holding_Cost'].mean())
            stockout_costs.append(m_df['Total_Stockout_Cost'].mean())
            method_labels.append(get_model_display_name(method))

    x_pos = range(len(method_labels))
    w = 0.6

    p1 = ax.bar(x_pos, ordering_costs, w, label='Ordering Cost', color='#3498db', alpha=0.85)
    p2 = ax.bar(x_pos, holding_costs_list, w, bottom=ordering_costs, label='Holding Cost', color='#f39c12', alpha=0.85)
    bottoms = [o + h for o, h in zip(ordering_costs, holding_costs_list)]
    p3 = ax.bar(x_pos, stockout_costs, w, bottom=bottoms, label='Stockout Cost', color='#e74c3c', alpha=0.85)

    ax.set_xticks(x_pos)
    ax.set_xticklabels(method_labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Cost ($)', fontsize=12)
    ax.set_title('Cost Breakdown by Component\n(Ordering + Holding + Stockout)', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cost_breakdown.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 4. Service Level Comparison (Bar Chart)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    service_levels = []
    sl_labels = []
    sl_colors = []
    for method in existing_methods:
        if method in combined_df['Method'].values:
            sl = combined_df[combined_df['Method'] == method]['Service_Level'].mean()
            service_levels.append(sl * 100)
            sl_labels.append(get_model_display_name(method))
            sl_colors.append(MODEL_COLORS.get(method, 'steelblue'))

    x_pos = range(len(sl_labels))
    bars = ax.bar(x_pos, service_levels, color=sl_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=95, color='gray', linestyle='--', alpha=0.7, linewidth=2, label='95% Target')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(sl_labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Service Level (%)', fontsize=12)
    ax.set_title('Service Level Comparison\n(Higher is Better, Target = 95%)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_ylim(0, 105)

    for bar, val in zip(bars, service_levels):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'service_level_comparison.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 5. Capacity Utilization Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 6))

    cap_utils = []
    cap_labels = []
    cap_colors = []
    for method in existing_methods:
        if method in combined_df['Method'].values:
            cu = combined_df[combined_df['Method'] == method]['Avg_Capacity_Util'].mean()
            cap_utils.append(cu * 100)
            cap_labels.append(get_model_display_name(method))
            cap_colors.append(MODEL_COLORS.get(method, 'steelblue'))

    x_pos = range(len(cap_labels))
    bars = ax.bar(x_pos, cap_utils, color=cap_colors, alpha=0.8, edgecolor='black', linewidth=0.5)
    ax.axhline(y=100, color='red', linestyle='--', alpha=0.5, linewidth=2, label='Full Capacity')
    ax.set_xticks(x_pos)
    ax.set_xticklabels(cap_labels, rotation=20, ha='right', fontsize=10)
    ax.set_ylabel('Capacity Utilization (%)', fontsize=12)
    ax.set_title(f'Average Capacity Utilization (Capacity = {config.cost.capacity} units)\n'
                 f'(Carryover Rate = {config.cost.carryover_rate})', fontsize=14, fontweight='bold')
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3, axis='y')

    for bar, val in zip(bars, cap_utils):
        ax.annotate(f'{val:.1f}%', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                    ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'capacity_utilization.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 6. Inventory Dynamics Over Time (Last Window)
    # =========================================================================
    if last_sim_results:
        fig, axes = plt.subplots(2, 1, figsize=(16, 10), sharex=True)

        # 6a. Demand vs Orders
        ax1 = axes[0]
        first_method = list(last_sim_results.keys())[0]
        n_days = len(last_sim_results[first_method].demands)
        days = range(n_days)

        ax1.plot(days, last_sim_results[first_method].demands, 'k-', linewidth=2,
                 label='Actual Demand', alpha=0.8, zorder=10)

        for method in existing_methods:
            if method in last_sim_results:
                sim = last_sim_results[method]
                ax1.plot(days, sim.actual_orders, '--', linewidth=1.5,
                         color=MODEL_COLORS.get(method, 'gray'),
                         label=f'{get_model_display_name(method)} Orders', alpha=0.7)

        ax1.set_ylabel('Units', fontsize=12)
        ax1.set_title('Demand vs Order Quantities (Last Window)', fontsize=14, fontweight='bold')
        ax1.legend(loc='upper right', fontsize=8, ncol=2)
        ax1.grid(True, alpha=0.3)

        # 6b. Inventory Levels with Capacity Line
        ax2 = axes[1]
        ax2.axhline(y=config.cost.capacity, color='red', linestyle='--',
                     linewidth=2, alpha=0.7, label=f'Capacity ({config.cost.capacity})')

        for method in existing_methods:
            if method in last_sim_results:
                sim = last_sim_results[method]
                ax2.plot(days, sim.inventory_levels, '-', linewidth=1.5,
                         color=MODEL_COLORS.get(method, 'gray'),
                         label=f'{get_model_display_name(method)}', alpha=0.8)

        ax2.fill_between(days, 0, [last_sim_results[first_method].demands[d] for d in days],
                         alpha=0.1, color='gray', label='Demand Zone')
        ax2.set_xlabel('Day', fontsize=12)
        ax2.set_ylabel('Inventory Level (units)', fontsize=12)
        ax2.set_title('Inventory Levels Over Time (with Carryover & Capacity)', fontsize=14, fontweight='bold')
        ax2.legend(loc='upper right', fontsize=8, ncol=2)
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'inventory_dynamics.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # 7. Carryover Inventory Over Time (Last Window)
    # =========================================================================
    if last_sim_results:
        fig, ax = plt.subplots(figsize=(14, 6))

        for method in existing_methods:
            if method in last_sim_results:
                sim = last_sim_results[method]
                ax.plot(range(len(sim.carryover_inventory)), sim.carryover_inventory,
                        '-o', linewidth=1.5, markersize=3,
                        color=MODEL_COLORS.get(method, 'gray'),
                        label=get_model_display_name(method), alpha=0.8)

        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Carryover Inventory (units)', fontsize=12)
        ax.set_title(f'Carryover Inventory Over Time (Rate = {config.cost.carryover_rate})\n'
                     f'(Leftover from previous period)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'carryover_inventory.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # 8. Cumulative Cost Over Time (Last Window)
    # =========================================================================
    if last_sim_results:
        fig, ax = plt.subplots(figsize=(14, 6))

        for method in existing_methods:
            if method in last_sim_results:
                sim = last_sim_results[method]
                cumulative = np.cumsum(sim.costs)
                ax.plot(range(len(cumulative)), cumulative, '-', linewidth=2,
                        color=MODEL_COLORS.get(method, 'gray'),
                        label=f'{get_model_display_name(method)} (Total: ${cumulative[-1]:.0f})', alpha=0.8)

        ax.set_xlabel('Day', fontsize=12)
        ax.set_ylabel('Cumulative Cost ($)', fontsize=12)
        ax.set_title('Cumulative Cost Over Time (Last Window)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper left', fontsize=9)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cumulative_cost.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # 9. Daily Cost Distribution (Violin Plot)
    # =========================================================================
    if last_sim_results:
        fig, ax = plt.subplots(figsize=(12, 6))

        cost_data = []
        method_names = []
        for method in existing_methods:
            if method in last_sim_results:
                sim = last_sim_results[method]
                cost_data.append(sim.costs)
                method_names.append(get_model_display_name(method))

        if cost_data:
            parts = ax.violinplot(cost_data, showmeans=True, showmedians=True)
            for i, pc in enumerate(parts['bodies']):
                method = existing_methods[i] if i < len(existing_methods) else None
                pc.set_facecolor(MODEL_COLORS.get(method, 'steelblue'))
                pc.set_alpha(0.7)

            ax.set_xticks(range(1, len(method_names) + 1))
            ax.set_xticklabels(method_names, rotation=20, ha='right', fontsize=10)
            ax.set_ylabel('Daily Cost ($)', fontsize=12)
            ax.set_title('Daily Cost Distribution (Last Window)\n(Violin Plot)', fontsize=14, fontweight='bold')
            ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cost_distribution.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # 10. Performance Progression Across Windows
    # =========================================================================
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))

    metrics_list = [
        ('Mean_Cost', 'Mean Cost ($)'),
        ('CVaR_90', 'CVaR-90 ($)'),
        ('Service_Level', 'Service Level'),
        ('Avg_Capacity_Util', 'Avg Capacity Utilization'),
    ]

    for idx, (metric, ylabel) in enumerate(metrics_list):
        ax = axes[idx // 2, idx % 2]
        for method in existing_methods:
            if method in combined_df['Method'].values:
                method_data = combined_df[combined_df['Method'] == method].groupby('window_idx')[metric].mean().reset_index()
                if metric in method_data.columns:
                    ax.plot(method_data['window_idx'], method_data[metric],
                            '-o', label=get_model_display_name(method),
                            color=MODEL_COLORS.get(method, 'gray'),
                            linewidth=2, markersize=5, alpha=0.8)

        ax.set_xlabel('Window Index', fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.set_title(f'{ylabel} Over Time', fontsize=12, fontweight='bold')
        ax.legend(fontsize=7, loc='best')
        ax.grid(True, alpha=0.3)

    plt.suptitle('Performance Progression Across Expanding Windows', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'performance_progression.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 11. Method Rankings Heatmap
    # =========================================================================
    fig, ax = plt.subplots(figsize=(10, 6))

    metrics = ['Mean_Cost', 'CVaR_90', 'CVaR_95']
    rank_data = []

    for method in existing_methods:
        if method in combined_df['Method'].values:
            row = {'Method': get_model_display_name(method)}
            for metric in metrics:
                row[metric] = combined_df[combined_df['Method'] == method][metric].mean()
            rank_data.append(row)

    if rank_data:
        rank_df = pd.DataFrame(rank_data).set_index('Method')
        rank_matrix = rank_df.rank()
        sns.heatmap(rank_matrix, annot=True, fmt='.0f', cmap='RdYlGn_r', ax=ax,
                    linewidths=0.5, linecolor='white')
        ax.set_title('Method Rankings Across Metrics\n(1 = Best, Lower Cost is Better)', fontsize=14, fontweight='bold')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'method_rankings.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 12. Timing Comparison
    # =========================================================================
    if 'Time_Seconds' in combined_df.columns:
        fig, ax = plt.subplots(figsize=(12, 6))

        timing_data = []
        timing_labels = []
        timing_colors = []
        for method in existing_methods:
            if method in combined_df['Method'].values:
                t = combined_df[combined_df['Method'] == method]['Time_Seconds'].mean()
                if not np.isnan(t):
                    timing_data.append(t)
                    timing_labels.append(get_model_display_name(method))
                    timing_colors.append(MODEL_COLORS.get(method, 'steelblue'))

        if timing_data:
            x_pos = range(len(timing_labels))
            bars = ax.bar(x_pos, timing_data, color=timing_colors, alpha=0.8,
                          edgecolor='black', linewidth=0.5)
            ax.set_xticks(x_pos)
            ax.set_xticklabels(timing_labels, rotation=20, ha='right', fontsize=10)
            ax.set_ylabel('Time (seconds)', fontsize=12)
            ax.set_title('Average Execution Time per Window\n(Training + Prediction + Simulation)',
                         fontsize=14, fontweight='bold')
            ax.set_yscale('log')
            ax.set_ylabel('Time (seconds, log scale)', fontsize=12)
            ax.grid(True, alpha=0.3, axis='y')

            for bar, val in zip(bars, timing_data):
                ax.annotate(f'{val:.2f}s', xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
                            ha='center', va='bottom', fontsize=9)

        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'timing_comparison.png'), dpi=150, bbox_inches='tight')
        plt.close()

    # =========================================================================
    # 13. Summary Dashboard (Combined Key Metrics)
    # =========================================================================
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3)

    # Panel 1: Mean Cost
    ax1 = fig.add_subplot(gs[0, 0])
    mc_vals = []
    mc_labels = []
    mc_colors = []
    for method in existing_methods:
        if method in combined_df['Method'].values:
            mc_vals.append(combined_df[combined_df['Method'] == method]['Mean_Cost'].mean())
            mc_labels.append(get_model_display_name(method).split('. ')[-1])
            mc_colors.append(MODEL_COLORS.get(method, 'steelblue'))
    ax1.barh(range(len(mc_labels)), mc_vals, color=mc_colors, alpha=0.8)
    ax1.set_yticks(range(len(mc_labels)))
    ax1.set_yticklabels(mc_labels, fontsize=9)
    ax1.set_xlabel('Mean Cost ($)')
    ax1.set_title('Mean Cost', fontweight='bold')
    ax1.invert_yaxis()

    # Panel 2: CVaR-90
    ax2 = fig.add_subplot(gs[0, 1])
    cv_vals = []
    for method in existing_methods:
        if method in combined_df['Method'].values:
            cv_vals.append(combined_df[combined_df['Method'] == method]['CVaR_90'].mean())
    ax2.barh(range(len(mc_labels)), cv_vals, color=mc_colors, alpha=0.8)
    ax2.set_yticks(range(len(mc_labels)))
    ax2.set_yticklabels(mc_labels, fontsize=9)
    ax2.set_xlabel('CVaR-90 ($)')
    ax2.set_title('CVaR-90 (Tail Risk)', fontweight='bold')
    ax2.invert_yaxis()

    # Panel 3: Service Level
    ax3 = fig.add_subplot(gs[0, 2])
    sl_vals = []
    for method in existing_methods:
        if method in combined_df['Method'].values:
            sl_vals.append(combined_df[combined_df['Method'] == method]['Service_Level'].mean() * 100)
    ax3.barh(range(len(mc_labels)), sl_vals, color=mc_colors, alpha=0.8)
    ax3.axvline(x=95, color='gray', linestyle='--', alpha=0.7, label='95% Target')
    ax3.set_yticks(range(len(mc_labels)))
    ax3.set_yticklabels(mc_labels, fontsize=9)
    ax3.set_xlabel('Service Level (%)')
    ax3.set_title('Service Level', fontweight='bold')
    ax3.invert_yaxis()
    ax3.legend(fontsize=8)

    # Panel 4: Capacity Utilization
    ax4 = fig.add_subplot(gs[1, 0])
    cu_vals = []
    for method in existing_methods:
        if method in combined_df['Method'].values:
            cu_vals.append(combined_df[combined_df['Method'] == method]['Avg_Capacity_Util'].mean() * 100)
    ax4.barh(range(len(mc_labels)), cu_vals, color=mc_colors, alpha=0.8)
    ax4.set_yticks(range(len(mc_labels)))
    ax4.set_yticklabels(mc_labels, fontsize=9)
    ax4.set_xlabel('Capacity Utilization (%)')
    ax4.set_title('Capacity Utilization', fontweight='bold')
    ax4.invert_yaxis()

    # Panel 5: Avg Carryover
    ax5 = fig.add_subplot(gs[1, 1])
    co_vals = []
    for method in existing_methods:
        if method in combined_df['Method'].values:
            co_vals.append(combined_df[combined_df['Method'] == method]['Avg_Carryover'].mean())
    ax5.barh(range(len(mc_labels)), co_vals, color=mc_colors, alpha=0.8)
    ax5.set_yticks(range(len(mc_labels)))
    ax5.set_yticklabels(mc_labels, fontsize=9)
    ax5.set_xlabel('Avg Carryover (units)')
    ax5.set_title('Average Carryover Inventory', fontweight='bold')
    ax5.invert_yaxis()

    # Panel 6: Timing
    ax6 = fig.add_subplot(gs[1, 2])
    t_vals = []
    for method in existing_methods:
        if method in combined_df['Method'].values:
            t_vals.append(combined_df[combined_df['Method'] == method]['Time_Seconds'].mean())
    ax6.barh(range(len(mc_labels)), t_vals, color=mc_colors, alpha=0.8)
    ax6.set_yticks(range(len(mc_labels)))
    ax6.set_yticklabels(mc_labels, fontsize=9)
    ax6.set_xlabel('Time (seconds)')
    ax6.set_title('Execution Time', fontweight='bold')
    ax6.invert_yaxis()

    fig.suptitle(f'Inventory Optimization Summary Dashboard\n'
                 f'Carryover Rate={config.cost.carryover_rate}, '
                 f'Capacity={config.cost.capacity}, '
                 f'Costs: order={config.cost.ordering_cost}, '
                 f'hold={config.cost.holding_cost}, '
                 f'stockout={config.cost.stockout_cost}',
                 fontsize=14, fontweight='bold', y=1.02)
    plt.savefig(os.path.join(output_dir, 'summary_dashboard.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # 14. Multi-SKU specific visualizations
    # =========================================================================
    if multi_sku and 'store_id' in combined_df.columns:
        # Heatmap of CVaR-90 by Store-Item
        fig, ax = plt.subplots(figsize=(12, 8))
        best_method = 'EnbPI_CQR_CVaR' if 'EnbPI_CQR_CVaR' in existing_methods else existing_methods[0]
        method_df = combined_df[combined_df['Method'] == best_method]
        pivot_data = method_df.groupby(['store_id', 'item_id'])['CVaR_90'].mean().unstack()

        sns.heatmap(pivot_data, annot=True, fmt='.1f', cmap='RdYlGn_r', ax=ax)
        ax.set_title(f'CVaR-90 by Store-Item ({get_model_display_name(best_method)})',
                     fontsize=14, fontweight='bold')
        ax.set_xlabel('Item ID', fontsize=12)
        ax.set_ylabel('Store ID', fontsize=12)
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, 'cvar90_by_sku.png'), dpi=150, bbox_inches='tight')
        plt.close()

    logger.info(f"Visualizations saved to {output_dir}")


# =============================================================================
# SUMMARY REPORT
# =============================================================================

def create_summary_report(
    combined_df: pd.DataFrame,
    aggregated: pd.DataFrame,
    output_dir: str,
    store_ids: List[int],
    item_ids: List[int],
    config: ExperimentConfig
) -> str:
    """Create a text summary report of the experiment results."""
    multi_sku = len(store_ids) > 1 or len(item_ids) > 1

    report = []
    report.append("=" * 80)
    report.append("INVENTORY OPTIMIZATION WITH CARRYOVER & CAPACITY CONSTRAINTS")
    report.append("Expanding Window Experiment Report")
    report.append("=" * 80)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    report.append(f"\nPROBLEM SETUP:")
    report.append(f"  Ordering Cost:     ${config.cost.ordering_cost:.2f}/unit")
    report.append(f"  Holding Cost:      ${config.cost.holding_cost:.2f}/unit")
    report.append(f"  Stockout Cost:     ${config.cost.stockout_cost:.2f}/unit")
    report.append(f"  Carryover Rate:    {config.cost.carryover_rate:.2f} ({config.cost.carryover_rate*100:.0f}% of leftover carries over)")
    report.append(f"  Capacity:          {config.cost.capacity:.0f} units")
    report.append(f"  Initial Inventory: {config.cost.initial_inventory:.0f} units")
    report.append(f"  Critical Ratio:    {config.cost.critical_ratio:.3f}")
    report.append(f"\nEXPERIMENT SETTINGS:")
    report.append(f"  Stores: {store_ids}")
    report.append(f"  Items: {item_ids}")
    report.append(f"  Total SKU Combinations: {len(store_ids) * len(item_ids)}")
    report.append(f"  Number of Windows: {combined_df['window_idx'].nunique()}")
    report.append(f"  Number of Methods: {combined_df['Method'].nunique()}")
    report.append(f"  CVaR Level (beta): {config.cvar.beta}")

    # Key findings
    report.append("\n" + "-" * 80)
    report.append("KEY FINDINGS")
    report.append("-" * 80)

    existing_methods = [m for m in MODEL_ORDER if m in combined_df['Method'].unique()]

    # Rankings by Mean Cost (excluding Seer)
    mean_costs = {}
    for method in existing_methods:
        if method != 'Seer':
            mean_costs[method] = combined_df[combined_df['Method'] == method]['Mean_Cost'].mean()

    if mean_costs:
        best_cost_method = min(mean_costs, key=mean_costs.get)
        report.append(f"\n[BEST Mean Cost] {get_model_display_name(best_cost_method)}: "
                      f"${mean_costs[best_cost_method]:.2f}")

    # Rankings by CVaR-90
    cvar90s = {}
    for method in existing_methods:
        if method != 'Seer':
            cvar90s[method] = combined_df[combined_df['Method'] == method]['CVaR_90'].mean()

    if cvar90s:
        best_cvar_method = min(cvar90s, key=cvar90s.get)
        report.append(f"[BEST CVaR-90]   {get_model_display_name(best_cvar_method)}: "
                      f"${cvar90s[best_cvar_method]:.2f}")

    # Service Level
    sls = {}
    for method in existing_methods:
        sls[method] = combined_df[combined_df['Method'] == method]['Service_Level'].mean()
    best_sl_method = max(sls, key=sls.get)
    report.append(f"[BEST Service]   {get_model_display_name(best_sl_method)}: "
                  f"{sls[best_sl_method]*100:.1f}%")

    # Oracle performance
    if 'Seer' in combined_df['Method'].values:
        seer_cost = combined_df[combined_df['Method'] == 'Seer']['Mean_Cost'].mean()
        report.append(f"\n[ORACLE (Seer)]  Mean Cost: ${seer_cost:.2f} (theoretical lower bound)")

    # EnbPI+CQR+CVaR performance
    if 'EnbPI_CQR_CVaR' in combined_df['Method'].values:
        report.append(f"\n[YOUR METHOD: EnbPI+CQR+CVaR]")
        enbpi_df = combined_df[combined_df['Method'] == 'EnbPI_CQR_CVaR']
        report.append(f"  Mean Cost:           ${enbpi_df['Mean_Cost'].mean():.2f}")
        report.append(f"  CVaR-90:             ${enbpi_df['CVaR_90'].mean():.2f}")
        report.append(f"  Service Level:       {enbpi_df['Service_Level'].mean()*100:.1f}%")
        report.append(f"  Capacity Util:       {enbpi_df['Avg_Capacity_Util'].mean()*100:.1f}%")
        report.append(f"  Avg Carryover:       {enbpi_df['Avg_Carryover'].mean():.1f} units")

        # Improvement comparisons
        if 'SAA' in mean_costs and 'EnbPI_CQR_CVaR' in mean_costs:
            imp = (mean_costs['SAA'] - mean_costs['EnbPI_CQR_CVaR']) / mean_costs['SAA'] * 100
            report.append(f"  Improvement vs SAA:  {imp:+.1f}%")
        if 'Wasserstein_DRO' in mean_costs and 'EnbPI_CQR_CVaR' in mean_costs:
            imp = (mean_costs['Wasserstein_DRO'] - mean_costs['EnbPI_CQR_CVaR']) / mean_costs['Wasserstein_DRO'] * 100
            report.append(f"  Improvement vs DRO:  {imp:+.1f}%")
        if 'SPO_EndToEnd' in mean_costs and 'EnbPI_CQR_CVaR' in mean_costs:
            imp = (mean_costs['SPO_EndToEnd'] - mean_costs['EnbPI_CQR_CVaR']) / mean_costs['SPO_EndToEnd'] * 100
            report.append(f"  Improvement vs SPO:  {imp:+.1f}%")

    # SPO/End-to-End performance
    if 'SPO_EndToEnd' in combined_df['Method'].values:
        report.append(f"\n[SPO/END-TO-END BASELINE]")
        spo_df = combined_df[combined_df['Method'] == 'SPO_EndToEnd']
        report.append(f"  Mean Cost:           ${spo_df['Mean_Cost'].mean():.2f}")
        report.append(f"  CVaR-90:             ${spo_df['CVaR_90'].mean():.2f}")
        report.append(f"  Service Level:       {spo_df['Service_Level'].mean()*100:.1f}%")
        report.append(f"  Capacity Util:       {spo_df['Avg_Capacity_Util'].mean()*100:.1f}%")
        report.append(f"  Avg Carryover:       {spo_df['Avg_Carryover'].mean():.1f} units")

    # Detailed results table
    report.append("\n" + "-" * 80)
    report.append("DETAILED RESULTS (Mean across all windows and SKUs)")
    report.append("-" * 80)

    header = f"{'Method':<25} {'Mean Cost':>10} {'CVaR-90':>10} {'CVaR-95':>10} {'SL (%)':>8} {'Cap Util':>10} {'Carryover':>10} {'Time(s)':>8}"
    report.append(f"\n{header}")
    report.append("-" * len(header))

    for method in existing_methods:
        m_df = combined_df[combined_df['Method'] == method]
        report.append(
            f"{get_model_display_name(method):<25} "
            f"${m_df['Mean_Cost'].mean():>8.2f} "
            f"${m_df['CVaR_90'].mean():>8.2f} "
            f"${m_df['CVaR_95'].mean():>8.2f} "
            f"{m_df['Service_Level'].mean()*100:>7.1f} "
            f"{m_df['Avg_Capacity_Util'].mean()*100:>9.1f}% "
            f"{m_df['Avg_Carryover'].mean():>9.1f} "
            f"{m_df['Time_Seconds'].mean():>7.2f}"
        )

    # Carryover & Capacity Impact Analysis
    report.append("\n" + "-" * 80)
    report.append("CARRYOVER & CAPACITY IMPACT ANALYSIS")
    report.append("-" * 80)

    for method in existing_methods:
        m_df = combined_df[combined_df['Method'] == method]
        avg_carry = m_df['Avg_Carryover'].mean()
        avg_cap = m_df['Avg_Capacity_Util'].mean() * 100
        avg_hold = m_df['Total_Holding_Cost'].mean()
        avg_stock = m_df['Total_Stockout_Cost'].mean()
        report.append(f"\n  {get_model_display_name(method)}:")
        report.append(f"    Avg Carryover:    {avg_carry:.1f} units")
        report.append(f"    Capacity Util:    {avg_cap:.1f}%")
        report.append(f"    Total Holding:    ${avg_hold:.2f}")
        report.append(f"    Total Stockout:   ${avg_stock:.2f}")
        report.append(f"    Hold/Stock Ratio: {avg_hold/(avg_stock+1e-6):.2f}")

    # Save report
    report_text = "\n".join(report)
    report_path = os.path.join(output_dir, "experiment_report.txt")
    with open(report_path, 'w') as f:
        f.write(report_text)

    logger.info(f"Report saved to {report_path}")
    return report_text


# =============================================================================
# STATISTICAL VALIDITY TESTS
# =============================================================================

def compute_statistical_tests(
    combined_df: pd.DataFrame,
    output_dir: str,
    reference_method: str = "EnbPI_CQR_CVaR",
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Test statistical validity of results across expanding windows.

    Performs paired comparisons between the reference method and all others
    using paired t-tests and Wilcoxon signed-rank tests on per-window metrics.
    Applies Bonferroni correction for multiple comparisons and reports
    Cohen's d effect sizes.

    Parameters
    ----------
    combined_df : pd.DataFrame
        All window results with columns Method, Mean_Cost, CVaR_90, etc.
    output_dir : str
        Directory to save statistical test results.
    reference_method : str
        Method to compare against (default: EnbPI_CQR_CVaR).
    alpha : float
        Significance level before correction (default: 0.05).

    Returns
    -------
    pd.DataFrame
        Statistical test results for all method-metric pairs.
    """
    logger.info("\n" + "=" * 80)
    logger.info("STATISTICAL VALIDITY TESTS")
    logger.info("=" * 80)

    existing_methods = [m for m in MODEL_ORDER if m in combined_df['Method'].unique()]

    if reference_method not in existing_methods:
        logger.warning(f"Reference method '{reference_method}' not found. "
                       f"Using first non-oracle method.")
        reference_method = [m for m in existing_methods if m != 'Seer'][0]

    test_metrics = ['Mean_Cost', 'CVaR_90', 'CVaR_95', 'Service_Level']
    comparison_methods = [m for m in existing_methods if m != reference_method]

    # Number of comparisons for Bonferroni correction
    n_comparisons = len(comparison_methods) * len(test_metrics)
    alpha_corrected = alpha / n_comparisons if n_comparisons > 0 else alpha

    logger.info(f"Reference method: {get_model_display_name(reference_method)}")
    logger.info(f"Comparisons: {len(comparison_methods)} methods x {len(test_metrics)} metrics = {n_comparisons}")
    logger.info(f"Bonferroni-corrected alpha: {alpha_corrected:.6f}")

    # Group by window (and SKU if multi-SKU) to get per-window observations
    group_cols = ['window_idx']
    if 'store_id' in combined_df.columns and 'item_id' in combined_df.columns:
        group_cols = ['store_id', 'item_id', 'window_idx']

    stat_results = []

    for metric in test_metrics:
        if metric not in combined_df.columns:
            continue

        # Get per-window values for reference method
        ref_df = combined_df[combined_df['Method'] == reference_method].set_index(group_cols)[metric]

        for comp_method in comparison_methods:
            comp_df = combined_df[combined_df['Method'] == comp_method].set_index(group_cols)[metric]

            # Align on common windows
            common_idx = ref_df.index.intersection(comp_df.index)
            if len(common_idx) < 3:
                logger.warning(f"  Skipping {comp_method} vs {reference_method} for {metric}: "
                               f"only {len(common_idx)} paired observations")
                continue

            ref_vals = ref_df.loc[common_idx].values.astype(float)
            comp_vals = comp_df.loc[common_idx].values.astype(float)
            diff = ref_vals - comp_vals
            n_obs = len(diff)

            # Paired t-test
            t_stat, t_pval = stats.ttest_rel(ref_vals, comp_vals)

            # Wilcoxon signed-rank test (non-parametric alternative)
            try:
                # Wilcoxon requires non-zero differences
                nonzero_diff = diff[diff != 0]
                if len(nonzero_diff) >= 5:
                    w_stat, w_pval = stats.wilcoxon(nonzero_diff)
                else:
                    w_stat, w_pval = np.nan, np.nan
            except ValueError:
                w_stat, w_pval = np.nan, np.nan

            # Cohen's d effect size
            pooled_std = np.sqrt((np.var(ref_vals, ddof=1) + np.var(comp_vals, ddof=1)) / 2)
            cohens_d = (np.mean(ref_vals) - np.mean(comp_vals)) / pooled_std if pooled_std > 0 else 0.0

            # Effect size interpretation
            abs_d = abs(cohens_d)
            if abs_d < 0.2:
                effect_interp = "negligible"
            elif abs_d < 0.5:
                effect_interp = "small"
            elif abs_d < 0.8:
                effect_interp = "medium"
            else:
                effect_interp = "large"

            # For cost metrics, negative diff means reference is better (lower cost)
            # For service level, positive diff means reference is better (higher SL)
            is_cost_metric = metric in ['Mean_Cost', 'CVaR_90', 'CVaR_95']
            ref_better = np.mean(diff) < 0 if is_cost_metric else np.mean(diff) > 0

            stat_results.append({
                'Metric': metric,
                'Reference': get_model_display_name(reference_method),
                'Comparison': get_model_display_name(comp_method),
                'Ref_Mean': np.mean(ref_vals),
                'Comp_Mean': np.mean(comp_vals),
                'Mean_Diff': np.mean(diff),
                'Std_Diff': np.std(diff, ddof=1),
                'N_Observations': n_obs,
                'T_Statistic': t_stat,
                'T_PValue': t_pval,
                'T_Significant': t_pval < alpha_corrected,
                'Wilcoxon_Stat': w_stat,
                'Wilcoxon_PValue': w_pval,
                'Wilcoxon_Significant': w_pval < alpha_corrected if not np.isnan(w_pval) else False,
                'Cohens_d': cohens_d,
                'Effect_Size': effect_interp,
                'Ref_Better': ref_better,
                'Alpha_Corrected': alpha_corrected,
            })

    stat_df = pd.DataFrame(stat_results)

    if stat_df.empty:
        logger.warning("No statistical tests could be performed (insufficient data).")
        return stat_df

    # Print results
    logger.info(f"\nPaired comparisons vs {get_model_display_name(reference_method)}:")
    logger.info(f"{'Metric':<15} {'vs Method':<25} {'Diff':>8} {'t-stat':>8} {'p-val':>10} {'Sig?':>5} {'Cohen d':>8} {'Effect':>10}")
    logger.info("-" * 100)

    for _, row in stat_df.iterrows():
        sig_marker = "*" if row['T_Significant'] else ""
        logger.info(
            f"{row['Metric']:<15} "
            f"{row['Comparison']:<25} "
            f"{row['Mean_Diff']:>+8.2f} "
            f"{row['T_Statistic']:>8.3f} "
            f"{row['T_PValue']:>10.6f} "
            f"{'YES' if row['T_Significant'] else 'no':>5}{sig_marker} "
            f"{row['Cohens_d']:>+8.3f} "
            f"{row['Effect_Size']:>10}"
        )

    # Summary of significance
    n_sig_t = stat_df['T_Significant'].sum()
    n_sig_w = stat_df['Wilcoxon_Significant'].sum()
    n_ref_better = stat_df['Ref_Better'].sum()
    logger.info(f"\nSummary:")
    logger.info(f"  Significant (paired t-test, Bonferroni): {n_sig_t}/{len(stat_df)}")
    logger.info(f"  Significant (Wilcoxon, Bonferroni):      {n_sig_w}/{len(stat_df)}")
    logger.info(f"  Reference method better in:              {n_ref_better}/{len(stat_df)} comparisons")

    # Save
    stat_path = os.path.join(output_dir, "statistical_tests.csv")
    stat_df.to_csv(stat_path, index=False)
    logger.info(f"\n[OK] Saved statistical tests: {stat_path}")

    # Create statistical test visualization
    _plot_statistical_results(stat_df, output_dir, reference_method)

    return stat_df


def _plot_statistical_results(
    stat_df: pd.DataFrame,
    output_dir: str,
    reference_method: str,
):
    """Create visualizations for statistical test results."""
    if stat_df.empty:
        return

    # =========================================================================
    # P-value heatmap
    # =========================================================================
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Paired t-test p-values
    metrics = stat_df['Metric'].unique()
    methods = stat_df['Comparison'].unique()

    pval_matrix = pd.DataFrame(index=methods, columns=metrics, dtype=float)
    effect_matrix = pd.DataFrame(index=methods, columns=metrics, dtype=float)

    for _, row in stat_df.iterrows():
        pval_matrix.loc[row['Comparison'], row['Metric']] = row['T_PValue']
        effect_matrix.loc[row['Comparison'], row['Metric']] = row['Cohens_d']

    # Log-transform p-values for visualization
    pval_log = -np.log10(pval_matrix.astype(float).clip(lower=1e-20))

    ax = axes[0]
    pval_annot = pval_matrix.apply(lambda col: col.map(lambda x: f'{x:.4f}' if pd.notna(x) else ''))
    sns.heatmap(pval_log, annot=pval_annot,
                fmt='', cmap='RdYlGn', ax=ax, linewidths=0.5, linecolor='white',
                vmin=0, vmax=max(5, pval_log.max().max()))
    ax.set_title(f'Paired t-test p-values vs {get_model_display_name(reference_method)}\n'
                 f'(color: -log10(p), annotation: raw p-value)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('')

    # Effect sizes
    ax = axes[1]
    sns.heatmap(effect_matrix.astype(float), annot=True, fmt='.3f',
                cmap='RdBu_r', center=0, ax=ax, linewidths=0.5, linecolor='white')
    ax.set_title(f"Cohen's d Effect Size vs {get_model_display_name(reference_method)}\n"
                 f'(negative = reference better for cost metrics)',
                 fontsize=11, fontweight='bold')
    ax.set_ylabel('')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_tests.png'), dpi=150, bbox_inches='tight')
    plt.close()

    # =========================================================================
    # Forest plot of mean differences with confidence intervals
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, max(6, len(stat_df) * 0.4)))

    y_positions = range(len(stat_df))
    labels = []
    for _, row in stat_df.iterrows():
        labels.append(f"{row['Metric']} | {row['Comparison']}")

    means = stat_df['Mean_Diff'].values
    stds = stat_df['Std_Diff'].values
    n_obs = stat_df['N_Observations'].values
    ci_95 = 1.96 * stds / np.sqrt(n_obs)
    significants = stat_df['T_Significant'].values

    colors = ['#d62728' if sig else '#1f77b4' for sig in significants]

    ax.errorbar(means, y_positions, xerr=ci_95, fmt='o', color='black',
                ecolor='gray', elinewidth=1.5, capsize=3, markersize=0)
    for i, (m, y, c) in enumerate(zip(means, y_positions, colors)):
        ax.scatter(m, y, color=c, s=80, zorder=5, edgecolors='black', linewidth=0.5)

    ax.axvline(x=0, color='gray', linestyle='--', alpha=0.7, linewidth=1.5)
    ax.set_yticks(list(y_positions))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel('Mean Difference (Reference - Comparison)', fontsize=11)
    ax.set_title(f'Forest Plot: Paired Differences vs {get_model_display_name(reference_method)}\n'
                 f'(Red = significant after Bonferroni correction, 95% CI shown)',
                 fontsize=12, fontweight='bold')
    ax.grid(True, alpha=0.3, axis='x')
    ax.invert_yaxis()

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'statistical_forest_plot.png'), dpi=150, bbox_inches='tight')
    plt.close()


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def parse_id_range(id_string: str) -> List[int]:
    """Parse a string of IDs (comma-separated or range)."""
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
    max_windows: Optional[int] = None
):
    """
    Main experiment runner with carryover and capacity constraints.
    """
    np.random.seed(config.random_seed)

    multi_sku = len(store_ids) > 1 or len(item_ids) > 1

    logger.info("=" * 80)
    logger.info("INVENTORY OPTIMIZATION WITH CARRYOVER & CAPACITY CONSTRAINTS")
    logger.info("6-Model Comparison: SAA, Conformal+CVaR, Wasserstein DRO, EnbPI+CQR+CVaR, SPO, Seer")
    logger.info("=" * 80)
    logger.info(f"Stores: {store_ids}")
    logger.info(f"Items: {item_ids}")
    logger.info(f"Total SKUs: {len(store_ids) * len(item_ids)}")
    logger.info(f"\nINVENTORY DYNAMICS:")
    logger.info(f"  Carryover Rate: {config.cost.carryover_rate}")
    logger.info(f"  Capacity: {config.cost.capacity} units")
    logger.info(f"  Initial Inventory: {config.cost.initial_inventory} units")
    logger.info(f"\nCOST PARAMETERS:")
    logger.info(f"  Ordering: ${config.cost.ordering_cost}/unit")
    logger.info(f"  Holding: ${config.cost.holding_cost}/unit")
    logger.info(f"  Stockout: ${config.cost.stockout_cost}/unit")

    os.makedirs(config.results_dir, exist_ok=True)

    # Load data
    logger.info("\nLoading expanding window data...")
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

    if max_windows is not None:
        for key in all_sku_splits:
            all_sku_splits[key] = all_sku_splits[key][:max_windows]

    total_windows = sum(len(splits) for splits in all_sku_splits.values())
    logger.info(f"\nTotal SKUs loaded: {len(all_sku_splits)}")
    logger.info(f"Total windows to process: {total_windows}")

    # Run experiments
    all_results = []
    last_sim_results = None

    with tqdm(total=len(all_sku_splits), desc="Processing SKUs") as pbar:
        for (store_id, item_id), sku_splits in all_sku_splits.items():
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing Store {store_id}, Item {item_id} ({len(sku_splits)} windows)")
            logger.info(f"{'='*60}")

            sku_results, sku_sim_results = run_single_sku(
                sku_splits, store_id, item_id, config, verbose=False
            )
            all_results.append(sku_results)
            last_sim_results = sku_sim_results
            pbar.update(1)

    # Combine all results
    logger.info("\n" + "=" * 80)
    logger.info("AGGREGATED RESULTS ACROSS ALL WINDOWS AND SKUs")
    logger.info("=" * 80)

    combined_df = pd.concat(all_results, ignore_index=True)

    # Compute aggregated statistics
    agg_metrics = ['Mean_Cost', 'CVaR_90', 'CVaR_95', 'Service_Level',
                   'Avg_Carryover', 'Avg_Capacity_Util',
                   'Total_Ordering_Cost', 'Total_Holding_Cost', 'Total_Stockout_Cost',
                   'Time_Seconds']
    existing_agg = [c for c in agg_metrics if c in combined_df.columns]
    agg_dict = {col: ['mean', 'std'] for col in existing_agg}
    aggregated = combined_df.groupby('Method').agg(agg_dict).round(3)

    print("\n", aggregated.to_string())

    # Save results
    agg_path = os.path.join(config.results_dir, "aggregated_results.csv")
    aggregated.to_csv(agg_path)
    logger.info(f"\n[OK] Saved aggregated results: {agg_path}")

    all_path = os.path.join(config.results_dir, "all_windows_results.csv")
    combined_df.to_csv(all_path, index=False)
    logger.info(f"[OK] Saved all window results: {all_path}")

    if multi_sku:
        sku_agg = combined_df.groupby(['store_id', 'item_id', 'Method'])[
            ['Mean_Cost', 'CVaR_90', 'CVaR_95', 'Service_Level', 'Avg_Carryover', 'Avg_Capacity_Util']
        ].agg(['mean', 'std']).round(3)
        sku_agg_path = os.path.join(config.results_dir, "results_by_sku.csv")
        sku_agg.to_csv(sku_agg_path)
        logger.info(f"[OK] Saved per-SKU results: {sku_agg_path}")

    # Create visualizations
    create_comprehensive_visualizations(
        combined_df, aggregated, last_sim_results,
        config.results_dir, config, multi_sku
    )

    # Statistical validity tests
    stat_df = compute_statistical_tests(
        combined_df, config.results_dir,
        reference_method="EnbPI_CQR_CVaR",
        alpha=0.05,
    )

    # Create summary report
    report = create_summary_report(
        combined_df, aggregated, config.results_dir,
        store_ids, item_ids, config
    )
    print("\n" + report)

    logger.info("\n" + "=" * 80)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Inventory Optimization with Carryover & Capacity Constraints"
    )
    parser.add_argument(
        "--output", type=str, default="results/expanding_window_carryover",
        help="Output directory for results"
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
        "--spo-epochs", type=int, default=50,
        help="Training epochs for SPO model"
    )
    # Inventory dynamics arguments
    parser.add_argument(
        "--carryover", type=float, default=0.95,
        help="Carryover rate: fraction of leftover inventory that carries to next period (0-1)"
    )
    parser.add_argument(
        "--capacity", type=float, default=200.0,
        help="Warehouse capacity in units"
    )
    parser.add_argument(
        "--initial-inventory", type=float, default=0.0,
        help="Initial inventory level"
    )
    # Cost arguments
    parser.add_argument(
        "--ordering-cost", type=float, default=10.0,
        help="Ordering cost per unit"
    )
    parser.add_argument(
        "--holding-cost", type=float, default=2.0,
        help="Holding cost per unit of overage"
    )
    parser.add_argument(
        "--stockout-cost", type=float, default=50.0,
        help="Stockout cost per unit of underage"
    )

    args = parser.parse_args()

    store_ids = parse_id_range(args.stores)
    item_ids = parse_id_range(args.items)

    config = get_default_config()
    config.results_dir = args.output
    config.rolling_window.enabled = True
    config.data.filepath = args.data

    # Set SPO epochs
    config._spo_epochs = args.spo_epochs

    # Set inventory dynamics
    config.cost.carryover_rate = args.carryover
    config.cost.capacity = args.capacity
    config.cost.initial_inventory = args.initial_inventory
    config.cost.ordering_cost = args.ordering_cost
    config.cost.holding_cost = args.holding_cost
    config.cost.stockout_cost = args.stockout_cost

    logger.info("Running Inventory Optimization with Carryover & Capacity Constraints")
    logger.info(f"  Carryover Rate: {args.carryover}")
    logger.info(f"  Capacity: {args.capacity} units")
    logger.info(f"  Initial Inventory: {args.initial_inventory} units")

    main(
        config,
        store_ids=store_ids,
        item_ids=item_ids,
        max_windows=args.windows
    )
