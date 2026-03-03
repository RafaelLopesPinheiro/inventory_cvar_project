#!/usr/bin/env python
"""
Lead-Time Sensitivity Experiment

Tests how replenishment lead time (L = 1, 3, 7 days) affects method performance.

When L > 1 the decision-maker must commit to an order L periods early:
  - Order placed at t, received at t + L
  - Demand forecast must cover the cumulative window [t+1, t+L]
  - All demand uncertainty compounds over L periods

Scientific hypothesis: CQR-based methods (wider, more conservative intervals)
become MORE valuable at longer lead times, while SAA degrades faster because its
empirical scenarios underestimate cumulative uncertainty.

Methods compared:
  - (s,S) Policy     (rule-based, no forecasting)
  - SAA              (OR baseline)
  - Conformal+CVaR   (distribution-free, single-period intervals)
  - EnbPI+CQR+CVaR   (conformal guarantee, CQR-based SL constraint)
  - CQR+SPO          (hybrid, residual-based scenarios)

Output (--output dir):
  - lead_time_results.csv
  - lead_time_cost_comparison.png
  - lead_time_cvar_comparison.png
  - lead_time_service_level.png
  - lead_time_report.txt

Usage:
    python scripts/run_lead_time_experiment.py
    python scripts/run_lead_time_experiment.py --lead-times 1,3,7 --stores 1 --items 1,2
"""

import argparse
import logging
import os
import sys
import warnings
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import (
    load_raw_data, filter_store_item, create_all_features, create_rolling_window_splits,
)
from src.models import (
    SampleAverageApproximation, ConformalPrediction, EnsembleBatchPI,
    SPORandomForest, PredictionResult,
)
from src.optimization import (
    compute_inventory_aware_orders_cvar,
    simulate_inventory_with_carryover,
    simulate_inventory_with_lead_time,
    simulate_sS_policy_with_carryover,
    InventorySimulationResult,
)

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ORDERING_COST = 10.0
HOLDING_COST  = 2.0
STOCKOUT_COST = 50.0

METHOD_LABELS = {
    'sS_Policy':      '(s,S) Policy',
    'SAA':            'SAA',
    'Conformal_CVaR': 'Conformal+CVaR',
    'EnbPI_CQR_CVaR': 'EnbPI+CQR+CVaR',
    'CQR_SPO':        'CQR+SPO',
}


# =============================================================================
# CUMULATIVE INTERVAL SCALING FOR LEAD TIME
# =============================================================================

def scale_intervals_for_lead_time(
    point: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    lead_time: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Scale single-period prediction intervals to cover cumulative L-period demand.

    Approximation: cumulative demand D_L = sum_{k=1}^{L} d_{t+k}.
    Under independence:
      E[D_L] = L * E[d_{t+1}]
      Var[D_L] = L * Var[d_{t+1}]
      Std[D_L] = sqrt(L) * Std[d_{t+1}]

    The interval half-width is proportional to std, so we scale it by sqrt(L).
    The point prediction (mean) scales linearly by L.

    Parameters
    ----------
    point, lower, upper : np.ndarray
        Single-period predictions.
    lead_time : int
        Number of periods to accumulate (L >= 1).

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray]
        Scaled (point_L, lower_L, upper_L) representing L-period cumulative demand.
    """
    if lead_time <= 1:
        return point.copy(), lower.copy(), upper.copy()

    half_width = (upper - lower) / 2.0
    center     = (upper + lower) / 2.0

    # Scale center (mean) linearly, half-width by sqrt(L)
    scaled_center     = center * lead_time
    scaled_half_width = half_width * np.sqrt(lead_time)

    scaled_point = point * lead_time
    scaled_lower = np.maximum(0.0, scaled_center - scaled_half_width)
    scaled_upper = scaled_center + scaled_half_width

    return scaled_point, scaled_lower, scaled_upper


# =============================================================================
# SINGLE WINDOW WITH LEAD-TIME SIMULATION
# =============================================================================

def run_window_lead_time(
    split,
    lead_time: int,
    beta: float = 0.90,
    alpha: float = 0.05,
    n_samples: int = 500,
    carryover_rate: float = 0.95,
    capacity: float = 200.0,
    initial_inventory: float = 0.0,
    random_seed: int = 42,
) -> Dict[str, dict]:
    """Run all methods on one window at a given lead time."""
    X_train, y_train = split.X_train, split.y_train
    X_cal,   y_cal   = split.X_cal,   split.y_cal
    X_test,  y_test  = split.X_test,  split.y_test

    # Cumulative demand over lead_time periods (for simulation target)
    if lead_time > 1:
        # Cumulative demand: sum each consecutive chunk of lead_time days
        # (if test is shorter than lead_time, we use what's available)
        y_test_cumul = np.array([
            np.sum(y_test[max(0, i - lead_time + 1):i + 1])
            for i in range(len(y_test))
        ])
    else:
        y_test_cumul = y_test.copy()

    results: Dict[str, dict] = {}

    sim_kwargs = dict(
        initial_inventory=initial_inventory,
        carryover_rate=carryover_rate,
        capacity=capacity,
        ordering_cost=ORDERING_COST,
        holding_cost=HOLDING_COST,
        stockout_cost=STOCKOUT_COST,
    )

    # -------------------------------------------------------------------------
    # 0. (s,S) Policy
    # -------------------------------------------------------------------------
    try:
        ss_sim = simulate_sS_policy_with_carryover(
            y_train, y_cal, y_test,
            critical_ratio=STOCKOUT_COST / (STOCKOUT_COST + ORDERING_COST),
            **sim_kwargs,
        )
        results['sS_Policy'] = _sim_to_dict(ss_sim)
    except Exception as e:
        logger.debug(f"(s,S) failed: {e}")

    def _apply_lead_time_sim(orders: np.ndarray, y: np.ndarray) -> InventorySimulationResult:
        """Apply lead-time simulation (wraps single-period sim for L=1)."""
        if lead_time <= 1:
            return simulate_inventory_with_carryover(
                orders, y, inventory_aware=True, **sim_kwargs
            )
        return simulate_inventory_with_lead_time(
            orders, y, lead_time=lead_time, **sim_kwargs
        )

    # -------------------------------------------------------------------------
    # 1. SAA
    # -------------------------------------------------------------------------
    try:
        saa = SampleAverageApproximation(
            alpha=alpha, n_estimators=100, max_depth=10,
            ordering_cost=ORDERING_COST, holding_cost=HOLDING_COST,
            stockout_cost=STOCKOUT_COST, cvar_beta=beta,
            n_scenarios=n_samples, random_state=random_seed,
        )
        saa.fit(X_train, y_train, X_cal, y_cal)
        saa_orders = saa.compute_order_quantities(X_test)
        results['SAA'] = _sim_to_dict(_apply_lead_time_sim(saa_orders, y_test))
    except Exception as e:
        logger.debug(f"SAA failed: {e}")

    # -------------------------------------------------------------------------
    # 2. Conformal + CVaR (with lead-time scaled intervals)
    # -------------------------------------------------------------------------
    try:
        conf = ConformalPrediction(alpha=alpha, n_estimators=100, max_depth=10,
                                   random_state=random_seed)
        conf.fit(X_train, y_train, X_cal, y_cal)
        conf_pred = conf.predict(X_test)

        pt_L, lo_L, hi_L = scale_intervals_for_lead_time(
            conf_pred.point, conf_pred.lower, conf_pred.upper, lead_time
        )
        conf_sim = compute_inventory_aware_orders_cvar(
            pt_L, lo_L, hi_L, actual_demands=y_test,
            beta=beta, n_samples=n_samples, random_seed=random_seed,
            verbose=False, **sim_kwargs,
        )
        # Re-simulate with lead time pipeline
        if lead_time > 1:
            conf_sim = simulate_inventory_with_lead_time(
                conf_sim.actual_orders, y_test, lead_time=lead_time, **sim_kwargs
            )
        results['Conformal_CVaR'] = _sim_to_dict(conf_sim)
    except Exception as e:
        logger.debug(f"Conformal+CVaR failed: {e}")

    # -------------------------------------------------------------------------
    # 3. EnbPI + CQR + CVaR (CQR-based SL constraint, scaled intervals)
    # -------------------------------------------------------------------------
    try:
        enbpi = EnsembleBatchPI(
            alpha=alpha, n_ensemble=10, n_estimators=100, max_depth=10,
            bootstrap_fraction=0.8, use_quantile_regression=True,
            random_state=random_seed,
        )
        enbpi.fit(X_train, y_train, X_cal, y_cal)
        enbpi_pred = enbpi.predict(X_test)

        pt_L, lo_L, hi_L = scale_intervals_for_lead_time(
            enbpi_pred.point, enbpi_pred.lower, enbpi_pred.upper, lead_time
        )
        enbpi_sim = compute_inventory_aware_orders_cvar(
            pt_L, lo_L, hi_L, actual_demands=y_test,
            beta=beta, n_samples=n_samples, random_seed=random_seed,
            verbose=False, sl_target=0.95, **sim_kwargs,
        )
        if lead_time > 1:
            enbpi_sim = simulate_inventory_with_lead_time(
                enbpi_sim.actual_orders, y_test, lead_time=lead_time, **sim_kwargs
            )
        results['EnbPI_CQR_CVaR'] = _sim_to_dict(enbpi_sim)
    except Exception as e:
        logger.debug(f"EnbPI+CQR+CVaR failed: {e}")

    # -------------------------------------------------------------------------
    # 4. CQR + SPO Hybrid (scaled intervals, residual scenarios)
    # -------------------------------------------------------------------------
    try:
        if 'EnbPI_CQR_CVaR' in results:
            enbpi_cal_pred = enbpi.predict(X_cal)
        else:
            enbpi = EnsembleBatchPI(
                alpha=alpha, n_ensemble=10, n_estimators=100, max_depth=10,
                bootstrap_fraction=0.8, use_quantile_regression=True,
                random_state=random_seed,
            )
            enbpi.fit(X_train, y_train, X_cal, y_cal)
            enbpi_pred = enbpi.predict(X_test)
            enbpi_cal_pred = enbpi.predict(X_cal)

        residuals = y_cal - enbpi_cal_pred.point

        pt_L, lo_L, hi_L = scale_intervals_for_lead_time(
            enbpi_pred.point, enbpi_pred.lower, enbpi_pred.upper, lead_time
        )
        cqr_spo_sim = compute_inventory_aware_orders_cvar(
            pt_L, lo_L, hi_L, actual_demands=y_test,
            beta=beta, n_samples=n_samples, random_seed=random_seed,
            verbose=False, demand_residuals=residuals, **sim_kwargs,
        )
        if lead_time > 1:
            cqr_spo_sim = simulate_inventory_with_lead_time(
                cqr_spo_sim.actual_orders, y_test, lead_time=lead_time, **sim_kwargs
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

def load_sku_windows(filepath, store_ids, item_ids, max_windows=None):
    df_raw = load_raw_data(filepath)
    sku_windows = {}
    for store_id in store_ids:
        for item_id in item_ids:
            try:
                df = filter_store_item(df_raw, store_id, item_id)
                if len(df) < 365 * 3:
                    continue
                df, feature_cols = create_all_features(df, lag_periods=[1, 7, 28],
                                                       rolling_windows=[7, 28])
                splits = create_rolling_window_splits(
                    df, feature_cols,
                    initial_train_days=730, calibration_days=365,
                    test_window_days=30, step_days=30,
                )
                if splits:
                    sku_windows[(store_id, item_id)] = splits[:max_windows] if max_windows else splits
            except Exception as e:
                logger.warning(f"Skipped ({store_id},{item_id}): {e}")
    return sku_windows


# =============================================================================
# MAIN LOOP + VISUALIZATIONS
# =============================================================================

def run_lead_time_sweep(sku_windows, lead_times, **kwargs):
    records = []
    total = sum(len(v) for v in sku_windows.values()) * len(lead_times)
    with tqdm(total=total, desc="Lead-time sweep") as pbar:
        for (store_id, item_id), splits in sku_windows.items():
            for win_idx, split in enumerate(splits):
                for L in lead_times:
                    try:
                        window_results = run_window_lead_time(split, lead_time=L, **kwargs)
                        for method, metrics in window_results.items():
                            records.append({
                                'store_id': store_id, 'item_id': item_id,
                                'window': win_idx, 'lead_time': L,
                                'method': method, **metrics,
                            })
                    except Exception as e:
                        logger.debug(f"Lead-time L={L} win={win_idx}: {e}")
                    pbar.update(1)
    return pd.DataFrame(records)


def plot_lead_time_metric(df, metric, ylabel, output_dir, filename):
    agg = df.groupby(['method', 'lead_time'])[metric].mean().reset_index()
    fig, ax = plt.subplots(figsize=(8, 5))
    for method, label in METHOD_LABELS.items():
        sub = agg[agg['method'] == method]
        if sub.empty:
            continue
        ax.plot(sub['lead_time'], sub[metric], marker='o', label=label, linewidth=2)
    ax.set_xlabel('Lead Time  (days)', fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(f'{ylabel} vs Replenishment Lead Time', fontsize=13)
    ax.set_xticks(sorted(df['lead_time'].unique()))
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    path = os.path.join(output_dir, filename)
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved {path}")


def write_lead_time_report(df, output_dir):
    path = os.path.join(output_dir, 'lead_time_report.txt')
    lines = ['=' * 65, 'LEAD-TIME SENSITIVITY REPORT', '=' * 65, '']
    for L in sorted(df['lead_time'].unique()):
        agg = (
            df[df['lead_time'] == L]
            .groupby('method')[['mean_cost', 'cvar_90', 'service_level']]
            .mean()
            .sort_values('mean_cost')
        )
        lines.append(f'Lead time L = {L} day(s)')
        lines.append('-' * 45)
        for m, row in agg.iterrows():
            label = METHOD_LABELS.get(m, m)
            lines.append(
                f"  {label:25s}  cost={row['mean_cost']:.2f}  "
                f"CVaR90={row['cvar_90']:.2f}  SL={row['service_level']:.1f}%"
            )
        lines.append('')

    # Degradation analysis: cost increase vs L=1
    lines.append('=' * 65)
    lines.append('COST DEGRADATION vs L=1 (relative increase)')
    lines.append('=' * 65)
    base = df[df['lead_time'] == df['lead_time'].min()].groupby('method')['mean_cost'].mean()
    for L in sorted(df['lead_time'].unique()):
        if L == df['lead_time'].min():
            continue
        curr = df[df['lead_time'] == L].groupby('method')['mean_cost'].mean()
        lines.append(f'\nL = {L}:')
        for m in METHOD_LABELS:
            if m in base.index and m in curr.index:
                delta = (curr[m] - base[m]) / base[m] * 100
                lines.append(f"  {METHOD_LABELS[m]:25s}  +{delta:.1f}%")

    with open(path, 'w') as f:
        f.write('\n'.join(lines))
    logger.info(f"Saved {path}")


# =============================================================================
# CLI
# =============================================================================

def parse_args():
    p = argparse.ArgumentParser(description='Lead-time sensitivity experiment')
    p.add_argument('--output', default='results/lead_time', help='Output directory')
    p.add_argument('--data', default='train.csv')
    p.add_argument('--stores', default='1')
    p.add_argument('--items', default='1')
    p.add_argument('--lead-times', default='1,3,7',
                   help='Comma-separated lead times to test (default: 1,3,7)')
    p.add_argument('--windows', type=int, default=None)
    p.add_argument('--beta', type=float, default=0.90)
    p.add_argument('--alpha', type=float, default=0.05)
    p.add_argument('--n-samples', type=int, default=500)
    p.add_argument('--carryover', type=float, default=0.95)
    p.add_argument('--capacity', type=float, default=200.0)
    p.add_argument('--seed', type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    os.makedirs(args.output, exist_ok=True)
    store_ids  = [int(s) for s in args.stores.split(',')]
    item_ids   = [int(s) for s in args.items.split(',')]
    lead_times = [int(l) for l in args.lead_times.split(',')]

    logger.info(f"Lead times: {lead_times}")
    logger.info(f"Stores={store_ids}, Items={item_ids}")

    sku_windows = load_sku_windows(args.data, store_ids, item_ids, args.windows)
    if not sku_windows:
        logger.error("No SKUs loaded.")
        sys.exit(1)

    df = run_lead_time_sweep(
        sku_windows, lead_times,
        beta=args.beta, alpha=args.alpha,
        n_samples=args.n_samples,
        carryover_rate=args.carryover,
        capacity=args.capacity,
        random_seed=args.seed,
    )

    csv_path = os.path.join(args.output, 'lead_time_results.csv')
    df.to_csv(csv_path, index=False)
    logger.info(f"Saved {csv_path}")

    if df.empty:
        logger.error("No results collected.")
        sys.exit(1)

    plot_lead_time_metric(df, 'mean_cost',     'Mean Cost per Period ($)',
                          args.output, 'lead_time_cost_comparison.png')
    plot_lead_time_metric(df, 'cvar_90',       'CVaR-90 ($)',
                          args.output, 'lead_time_cvar_comparison.png')
    plot_lead_time_metric(df, 'service_level', 'Service Level (%)',
                          args.output, 'lead_time_service_level.png')
    write_lead_time_report(df, args.output)

    logger.info(f"Lead-time experiment complete. Results in {args.output}/")


if __name__ == '__main__':
    main()
