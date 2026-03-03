#!/usr/bin/env python
"""
Sensitivity Analysis: Effect of Beta, Alpha, and Cost Ratio on Method Rankings

Sweeps over:
  - beta  (CVaR level):        {0.70, 0.80, 0.90, 0.95}
  - alpha (conformal level):   {0.05, 0.10, 0.20}
  - cost_ratio (c_u / c_o):   {2, 5, 10, 20}  → stockout_cost = cost_ratio * ordering_cost

Methods compared (RF-based to isolate uncertainty methodology):
  - SAA             (baseline)
  - Conformal+CVaR  (distribution-free)
  - EnbPI+CQR+CVaR  (proposed, with CQR-based SL constraint)
  - CQR+SPO         (proposed hybrid)

Outputs (saved to --output dir):
  - sensitivity_results.csv       Raw results for all parameter combos
  - heatmap_mean_cost_<method>.png   Mean cost heatmap (beta x alpha)
  - heatmap_cvar90_<method>.png      CVaR-90 heatmap (beta x alpha)
  - heatmap_service_level.png        Service level heatmap per method
  - sensitivity_report.txt           Human-readable summary

Usage:
    python scripts/run_sensitivity_analysis.py
    python scripts/run_sensitivity_analysis.py --stores 1 --items 1,2 --windows 3
    python scripts/run_sensitivity_analysis.py --output results/sensitivity
"""

import argparse
import logging
import os
import sys
import time
import warnings
from itertools import product
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    load_raw_data,
    filter_store_item,
    create_all_features,
    create_rolling_window_splits,
)
from src.models import (
    SampleAverageApproximation,
    ConformalPrediction,
    EnsembleBatchPI,
    SPORandomForest,
    PredictionResult,
)
from src.optimization import (
    compute_inventory_aware_orders_cvar,
    simulate_inventory_with_carryover,
    InventorySimulationResult,
)

warnings.filterwarnings('ignore')
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# =============================================================================
# PARAMETER GRIDS
# =============================================================================

BETA_VALUES  = [0.70, 0.80, 0.90, 0.95]   # CVaR tail level
ALPHA_VALUES = [0.05, 0.10, 0.20]          # Conformal miscoverage rate
COST_RATIOS  = [2, 5, 10, 20]             # c_u / c_o  (c_o fixed at 10)

ORDERING_COST = 10.0
HOLDING_COST  = 2.0   # fixed

METHOD_KEYS = ['SAA', 'Conformal_CVaR', 'EnbPI_CQR_CVaR', 'CQR_SPO']
METHOD_LABELS = {
    'SAA':             'SAA',
    'Conformal_CVaR':  'Conformal+CVaR',
    'EnbPI_CQR_CVaR':  'EnbPI+CQR+CVaR',
    'CQR_SPO':         'CQR+SPO',
}


# =============================================================================
# SINGLE WINDOW RUNNER (SENSITIVITY VERSION)
# =============================================================================

def run_window_sensitivity(
    split,
    beta: float,
    alpha: float,
    cost_ratio: float,
    random_seed: int = 42,
    n_samples: int = 500,
    carryover_rate: float = 0.95,
    capacity: float = 200.0,
    initial_inventory: float = 0.0,
) -> Dict[str, dict]:
    """
    Run all sensitivity methods on one expanding window split.

    Returns dict of {method_key: {mean_cost, cvar_90, service_level}}.
    """
    stockout_cost = cost_ratio * ORDERING_COST

    X_train, y_train = split.X_train, split.y_train
    X_cal,   y_cal   = split.X_cal,   split.y_cal
    X_test,  y_test  = split.X_test,  split.y_test

    results: Dict[str, dict] = {}

    # -------------------------------------------------------------------------
    # 1. SAA
    # -------------------------------------------------------------------------
    try:
        saa = SampleAverageApproximation(
            alpha=alpha,
            n_estimators=100,
            max_depth=10,
            ordering_cost=ORDERING_COST,
            holding_cost=HOLDING_COST,
            stockout_cost=stockout_cost,
            cvar_beta=beta,
            n_scenarios=n_samples,
            random_state=random_seed,
        )
        saa.fit(X_train, y_train, X_cal, y_cal)
        saa_pred  = saa.predict(X_test)
        saa_orders = saa.compute_order_quantities(X_test)

        saa_sim = simulate_inventory_with_carryover(
            saa_orders, y_test,
            initial_inventory=initial_inventory,
            carryover_rate=carryover_rate,
            capacity=capacity,
            ordering_cost=ORDERING_COST,
            holding_cost=HOLDING_COST,
            stockout_cost=stockout_cost,
            inventory_aware=True,
        )
        results['SAA'] = _sim_to_dict(saa_sim)
    except Exception as e:
        logger.debug(f"SAA failed: {e}")

    # -------------------------------------------------------------------------
    # 2. Conformal + CVaR
    # -------------------------------------------------------------------------
    try:
        conf = ConformalPrediction(
            alpha=alpha,
            n_estimators=100,
            max_depth=10,
            random_state=random_seed,
        )
        conf.fit(X_train, y_train, X_cal, y_cal)
        conf_pred = conf.predict(X_test)

        conf_sim = compute_inventory_aware_orders_cvar(
            conf_pred.point, conf_pred.lower, conf_pred.upper,
            actual_demands=y_test,
            initial_inventory=initial_inventory,
            carryover_rate=carryover_rate,
            capacity=capacity,
            beta=beta, n_samples=n_samples,
            ordering_cost=ORDERING_COST,
            holding_cost=HOLDING_COST,
            stockout_cost=stockout_cost,
            random_seed=random_seed,
            verbose=False,
        )
        results['Conformal_CVaR'] = _sim_to_dict(conf_sim)
    except Exception as e:
        logger.debug(f"Conformal+CVaR failed: {e}")

    # -------------------------------------------------------------------------
    # 3. EnbPI + CQR + CVaR  (with CQR-based SL constraint = upper[t])
    # -------------------------------------------------------------------------
    try:
        enbpi = EnsembleBatchPI(
            alpha=alpha,
            n_ensemble=10,
            n_estimators=100,
            max_depth=10,
            bootstrap_fraction=0.8,
            use_quantile_regression=True,
            random_state=random_seed,
        )
        enbpi.fit(X_train, y_train, X_cal, y_cal)
        enbpi_pred = enbpi.predict(X_test)

        # sl_target=0.95 triggers the CQR-upper-bound SL constraint (after fix)
        enbpi_sim = compute_inventory_aware_orders_cvar(
            enbpi_pred.point, enbpi_pred.lower, enbpi_pred.upper,
            actual_demands=y_test,
            initial_inventory=initial_inventory,
            carryover_rate=carryover_rate,
            capacity=capacity,
            beta=beta, n_samples=n_samples,
            ordering_cost=ORDERING_COST,
            holding_cost=HOLDING_COST,
            stockout_cost=stockout_cost,
            random_seed=random_seed,
            verbose=False,
            sl_target=0.95,
        )
        results['EnbPI_CQR_CVaR'] = _sim_to_dict(enbpi_sim)
    except Exception as e:
        logger.debug(f"EnbPI+CQR+CVaR failed: {e}")

    # -------------------------------------------------------------------------
    # 4. CQR + SPO (Hybrid)
    # -------------------------------------------------------------------------
    try:
        # Re-use fitted enbpi if available; otherwise refit
        if 'EnbPI_CQR_CVaR' in results:
            enbpi_cal_pred = enbpi.predict(X_cal)
        else:
            enbpi = EnsembleBatchPI(
                alpha=alpha, n_ensemble=10, n_estimators=100,
                max_depth=10, bootstrap_fraction=0.8,
                use_quantile_regression=True, random_state=random_seed,
            )
            enbpi.fit(X_train, y_train, X_cal, y_cal)
            enbpi_pred = enbpi.predict(X_test)
            enbpi_cal_pred = enbpi.predict(X_cal)

        residuals = y_cal - enbpi_cal_pred.point

        cqr_spo_sim = compute_inventory_aware_orders_cvar(
            enbpi_pred.point, enbpi_pred.lower, enbpi_pred.upper,
            actual_demands=y_test,
            initial_inventory=initial_inventory,
            carryover_rate=carryover_rate,
            capacity=capacity,
            beta=beta, n_samples=n_samples,
            ordering_cost=ORDERING_COST,
            holding_cost=HOLDING_COST,
            stockout_cost=stockout_cost,
            random_seed=random_seed,
            verbose=False,
            demand_residuals=residuals,
            sl_target=None,
        )
        results['CQR_SPO'] = _sim_to_dict(cqr_spo_sim)
    except Exception as e:
        logger.debug(f"CQR+SPO failed: {e}")

    return results


def _sim_to_dict(sim: InventorySimulationResult) -> dict:
    return {
        'mean_cost':     sim.mean_cost,
        'cvar_90':       sim.cvar_90,
        'service_level': sim.service_level * 100,
        'avg_carryover': sim.avg_carryover,
    }


# =============================================================================
# DATA LOADING
# =============================================================================

def load_sku_windows(
    filepath: str,
    store_ids: List[int],
    item_ids: List[int],
    initial_train_days: int = 730,
    calibration_days:   int = 365,
    test_window_days:   int = 30,
    max_windows:        Optional[int] = None,
):
    """Load expanding window splits for all requested SKUs."""
    df_raw = load_raw_data(filepath)
    sku_windows = {}

    for store_id in store_ids:
        for item_id in item_ids:
            try:
                df = filter_store_item(df_raw, store_id, item_id)
                if len(df) < 365 * 3:
                    continue
                df, feature_cols = create_all_features(
                    df, lag_periods=[1, 7, 28], rolling_windows=[7, 28]
                )
                splits = create_rolling_window_splits(
                    df, feature_cols,
                    initial_train_days=initial_train_days,
                    calibration_days=calibration_days,
                    test_window_days=test_window_days,
                    step_days=test_window_days,
                )
                if splits:
                    if max_windows:
                        splits = splits[:max_windows]
                    sku_windows[(store_id, item_id)] = splits
                    logger.info(f"Loaded SKU ({store_id},{item_id}): {len(splits)} windows")
            except Exception as e:
                logger.warning(f"Skipped SKU ({store_id},{item_id}): {e}")

    return sku_windows


# =============================================================================
# MAIN EXPERIMENT LOOP
# =============================================================================

def run_sensitivity(
    sku_windows: dict,
    beta_values:   List[float],
    alpha_values:  List[float],
    cost_ratios:   List[float],
    n_samples:     int = 500,
    random_seed:   int = 42,
    carryover_rate: float = 0.95,
    capacity:      float = 200.0,
) -> pd.DataFrame:
    """
    Sweep beta × alpha × cost_ratio, run methods, collect results.
    Returns a tidy DataFrame with one row per (SKU, window, beta, alpha, cost_ratio, method).
    """
    records = []
    param_combos = list(product(beta_values, alpha_values, cost_ratios))
    total_skus = len(sku_windows)
    total_windows = sum(len(v) for v in sku_windows.values())
    total_runs = len(param_combos) * total_windows

    logger.info(
        f"Sensitivity sweep: {len(param_combos)} param combos × "
        f"{total_windows} windows × {len(METHOD_KEYS)} methods = "
        f"{total_runs * len(METHOD_KEYS)} LP solves"
    )

    with tqdm(total=total_runs, desc="Sensitivity sweep") as pbar:
        for (store_id, item_id), splits in sku_windows.items():
            for win_idx, split in enumerate(splits):
                for beta, alpha, cost_ratio in param_combos:
                    try:
                        window_results = run_window_sensitivity(
                            split, beta, alpha, cost_ratio,
                            random_seed=random_seed,
                            n_samples=n_samples,
                            carryover_rate=carryover_rate,
                            capacity=capacity,
                        )
                        for method_key, metrics in window_results.items():
                            records.append({
                                'store_id':   store_id,
                                'item_id':    item_id,
                                'window':     win_idx,
                                'beta':       beta,
                                'alpha':      alpha,
                                'cost_ratio': cost_ratio,
                                'method':     method_key,
                                **metrics,
                            })
                    except Exception as e:
                        logger.debug(f"Window ({store_id},{item_id},{win_idx}) b={beta} a={alpha} cr={cost_ratio}: {e}")
                    pbar.update(1)

    return pd.DataFrame(records)


# =============================================================================
# VISUALIZATIONS
# =============================================================================

def plot_heatmap_beta_alpha(
    df: pd.DataFrame,
    metric: str,
    metric_label: str,
    output_dir: str,
    cost_ratio: float = 5.0,
):
    """
    For each method, plot a heatmap of metric vs (beta, alpha)
    at a fixed cost_ratio.
    """
    df_cr = df[df['cost_ratio'] == cost_ratio]
    agg = (
        df_cr.groupby(['method', 'beta', 'alpha'])[metric]
        .mean()
        .reset_index()
    )

    methods = METHOD_KEYS
    n_methods = len(methods)
    fig, axes = plt.subplots(1, n_methods, figsize=(5 * n_methods, 4), sharey=True)
    fig.suptitle(
        f"{metric_label} by (Beta, Alpha)  |  cost_ratio={cost_ratio}",
        fontsize=14, fontweight='bold'
    )

    for ax, method in zip(axes, methods):
        pivot = (
            agg[agg['method'] == method]
            .pivot(index='alpha', columns='beta', values=metric)
        )
        if pivot.empty:
            ax.set_visible(False)
            continue
        sns.heatmap(
            pivot, ax=ax, annot=True, fmt='.1f',
            cmap='RdYlGn_r' if 'cost' in metric or 'cvar' in metric else 'RdYlGn',
            cbar=True, linewidths=0.5,
        )
        ax.set_title(METHOD_LABELS.get(method, method), fontsize=11)
        ax.set_xlabel('Beta (CVaR level)')
        ax.set_ylabel('Alpha (conformal)' if ax == axes[0] else '')

    plt.tight_layout()
    fname = os.path.join(output_dir, f'heatmap_{metric}_cr{int(cost_ratio)}.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {fname}")


def plot_cost_ratio_effect(df: pd.DataFrame, output_dir: str):
    """
    Line plot: mean cost vs cost_ratio for each method,
    at fixed beta=0.90 and alpha=0.05.
    """
    df_fixed = df[(df['beta'] == 0.90) & (df['alpha'] == 0.05)]
    agg = (
        df_fixed.groupby(['method', 'cost_ratio'])['mean_cost']
        .mean()
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(8, 5))
    for method in METHOD_KEYS:
        sub = agg[agg['method'] == method]
        if sub.empty:
            continue
        ax.plot(sub['cost_ratio'], sub['mean_cost'],
                marker='o', label=METHOD_LABELS.get(method, method), linewidth=2)

    ax.set_xlabel('Cost Ratio  (c_u / c_o)', fontsize=12)
    ax.set_ylabel('Mean Cost per Period ($)', fontsize=12)
    ax.set_title('Effect of Cost Ratio on Mean Cost  (beta=0.90, alpha=0.05)', fontsize=13)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fname = os.path.join(output_dir, 'cost_ratio_effect.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {fname}")


def plot_method_ranking_stability(df: pd.DataFrame, output_dir: str):
    """
    For each (beta, alpha) combo, determine method ranking by mean_cost
    and show how rankings shift with cost_ratio.
    """
    fig, axes = plt.subplots(
        len(BETA_VALUES), len(ALPHA_VALUES),
        figsize=(5 * len(ALPHA_VALUES), 4 * len(BETA_VALUES)),
        sharex=True,
    )
    fig.suptitle('Method Rankings (lower rank = lower cost) by (Beta, Alpha)',
                 fontsize=14, fontweight='bold')

    for bi, beta in enumerate(BETA_VALUES):
        for ai, alpha in enumerate(ALPHA_VALUES):
            ax = axes[bi][ai]
            df_sub = df[(df['beta'] == beta) & (df['alpha'] == alpha)]
            rank_data = (
                df_sub.groupby(['cost_ratio', 'method'])['mean_cost']
                .mean()
                .reset_index()
            )
            for method in METHOD_KEYS:
                method_data = rank_data[rank_data['method'] == method]
                if method_data.empty:
                    continue
                # Compute rank within each cost_ratio
                ranked = (
                    rank_data.groupby('cost_ratio')['mean_cost']
                    .rank()
                    .loc[rank_data['method'] == method]
                )
                cost_ratios_sorted = method_data['cost_ratio'].values
                ax.plot(
                    cost_ratios_sorted,
                    ranked.values,
                    marker='o',
                    label=METHOD_LABELS.get(method, method),
                    linewidth=2,
                )
            ax.set_title(f'β={beta}, α={alpha}', fontsize=9)
            ax.set_xlabel('Cost Ratio')
            ax.set_ylabel('Rank (1=best)')
            ax.set_yticks([1, 2, 3, 4])
            ax.invert_yaxis()
            ax.grid(True, alpha=0.3)
            if bi == 0 and ai == 0:
                ax.legend(fontsize=7)

    plt.tight_layout()
    fname = os.path.join(output_dir, 'method_ranking_stability.png')
    plt.savefig(fname, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {fname}")


# =============================================================================
# REPORT
# =============================================================================

def write_report(df: pd.DataFrame, output_dir: str):
    report_path = os.path.join(output_dir, 'sensitivity_report.txt')
    lines = []
    lines.append("=" * 70)
    lines.append("SENSITIVITY ANALYSIS REPORT")
    lines.append("=" * 70)
    lines.append(f"Total records: {len(df)}")
    lines.append(f"SKUs: {df[['store_id','item_id']].drop_duplicates().shape[0]}")
    lines.append(f"Windows: {df['window'].nunique()}")
    lines.append(f"Parameter combos: beta={BETA_VALUES}, alpha={ALPHA_VALUES}, cost_ratio={COST_RATIOS}")
    lines.append("")

    # Best method by metric for each cost_ratio
    lines.append("-" * 70)
    lines.append("BEST METHOD BY COST RATIO (beta=0.90, alpha=0.05)")
    lines.append("-" * 70)
    df_ref = df[(df['beta'] == 0.90) & (df['alpha'] == 0.05)]
    for cr in COST_RATIOS:
        agg = (
            df_ref[df_ref['cost_ratio'] == cr]
            .groupby('method')[['mean_cost', 'cvar_90', 'service_level']]
            .mean()
        )
        if agg.empty:
            continue
        best_cost   = agg['mean_cost'].idxmin()
        best_cvar   = agg['cvar_90'].idxmin()
        best_sl     = agg['service_level'].idxmax()
        lines.append(f"\n  cost_ratio={cr}  (c_u={cr*ORDERING_COST:.0f}, c_o={ORDERING_COST:.0f})")
        lines.append(f"    Best mean cost:     {METHOD_LABELS.get(best_cost, best_cost)}  ({agg.loc[best_cost,'mean_cost']:.2f})")
        lines.append(f"    Best CVaR-90:       {METHOD_LABELS.get(best_cvar, best_cvar)}  ({agg.loc[best_cvar,'cvar_90']:.2f})")
        lines.append(f"    Best service level: {METHOD_LABELS.get(best_sl,   best_sl)}  ({agg.loc[best_sl,'service_level']:.1f}%)")
        lines.append(f"    Full ranking (mean cost):")
        for m, row in agg.sort_values('mean_cost').iterrows():
            lines.append(f"      {METHOD_LABELS.get(m,m):25s}  cost={row['mean_cost']:.2f}  CVaR90={row['cvar_90']:.2f}  SL={row['service_level']:.1f}%")

    # Effect of beta
    lines.append("\n" + "-" * 70)
    lines.append("EFFECT OF BETA ON METHOD RANKINGS  (alpha=0.05, cost_ratio=5)")
    lines.append("-" * 70)
    df_b = df[(df['alpha'] == 0.05) & (df['cost_ratio'] == 5)]
    for beta in BETA_VALUES:
        agg = (
            df_b[df_b['beta'] == beta]
            .groupby('method')['cvar_90']
            .mean()
            .sort_values()
        )
        if agg.empty:
            continue
        lines.append(f"\n  beta={beta}  (CVaR-90 ranking):")
        for m, val in agg.items():
            lines.append(f"    {METHOD_LABELS.get(m,m):25s}  CVaR-90={val:.2f}")

    with open(report_path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Saved {report_path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Sensitivity analysis for inventory CVaR methods')
    p.add_argument('--output', default='results/sensitivity',
                   help='Output directory (default: results/sensitivity)')
    p.add_argument('--data', default='train.csv', help='Path to train.csv')
    p.add_argument('--stores', default='1', help='Comma-separated store IDs (default: 1)')
    p.add_argument('--items',  default='1', help='Comma-separated item IDs (default: 1)')
    p.add_argument('--windows', type=int, default=None,
                   help='Max windows per SKU (default: all)')
    p.add_argument('--n-samples', type=int, default=500,
                   help='CVaR scenario samples per period (default: 500)')
    p.add_argument('--carryover', type=float, default=0.95, help='Carryover rate')
    p.add_argument('--capacity',  type=float, default=200.0, help='Warehouse capacity')
    p.add_argument('--seed', type=int, default=42, help='Random seed')
    return p.parse_args()


def main():
    args = parse_args()

    os.makedirs(args.output, exist_ok=True)

    store_ids = [int(s) for s in args.stores.split(',')]
    item_ids  = [int(s) for s in args.items.split(',')]

    logger.info(f"Stores={store_ids}, Items={item_ids}")
    logger.info(f"Beta grid: {BETA_VALUES}")
    logger.info(f"Alpha grid: {ALPHA_VALUES}")
    logger.info(f"Cost ratio grid: {COST_RATIOS}")

    # Load data
    sku_windows = load_sku_windows(
        args.data, store_ids, item_ids,
        max_windows=args.windows,
    )
    if not sku_windows:
        logger.error("No SKUs loaded. Check --stores, --items, and --data arguments.")
        sys.exit(1)

    # Run sweep
    df = run_sensitivity(
        sku_windows,
        beta_values=BETA_VALUES,
        alpha_values=ALPHA_VALUES,
        cost_ratios=COST_RATIOS,
        n_samples=args.n_samples,
        random_seed=args.seed,
        carryover_rate=args.carryover,
        capacity=args.capacity,
    )

    # Save raw results
    csv_path = os.path.join(args.output, 'sensitivity_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved raw results to {csv_path}")

    if df.empty:
        logger.error("No results collected. Check logs for errors.")
        sys.exit(1)

    # Plots
    for metric, label in [('mean_cost', 'Mean Cost ($)'), ('cvar_90', 'CVaR-90 ($)'),
                           ('service_level', 'Service Level (%)')]:
        for cr in COST_RATIOS:
            plot_heatmap_beta_alpha(df, metric, label, args.output, cost_ratio=cr)

    plot_cost_ratio_effect(df, args.output)
    plot_method_ranking_stability(df, args.output)
    write_report(df, args.output)

    logger.info(f"Sensitivity analysis complete. Results in {args.output}/")


if __name__ == '__main__':
    main()
