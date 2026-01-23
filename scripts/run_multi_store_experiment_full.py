#!/usr/bin/env python
"""
Enhanced Multi-Store Experiment Runner

Runs experiments across multiple store-item combinations with ALL models
and comprehensive visualizations.

Usage:
    python run_multi_store_experiment_full.py --stores 1,2,3 --items 1,2,3,4,5
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
import matplotlib.pyplot as plt
import seaborn as sns

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
    QuantileRegression,
    SampleAverageApproximation,
    ExpectedValue,
    LSTMQuantileRegression,
    TransformerQuantileRegression,
    DeepEnsemble,
    MCDropoutLSTM,
    TemporalFusionTransformer,
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

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


def run_single_store_item_full(
    df_raw: pd.DataFrame,
    store_id: int,
    item_id: int,
    config: ExperimentConfig,
    use_deep_learning: bool = True,
    dl_epochs: int = 50
) -> Dict[str, MethodResults]:
    """
    Run ALL models for a single store-item combination.
    
    Parameters
    ----------
    df_raw : pd.DataFrame
        Raw data.
    store_id : int
        Store ID.
    item_id : int
        Item ID.
    config : ExperimentConfig
        Configuration.
    use_deep_learning : bool
        Whether to run DL models (slower).
    dl_epochs : int
        Epochs for DL models.
    
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
        
        # Prepare sequence data for DL models
        seq_data = prepare_sequence_data(
            splits,
            seq_length=config.data.sequence_length,
            prediction_horizon=config.data.prediction_horizon
        )
        
        results = {}
        costs = config.cost
        
        X_train, y_train = splits.train.X, splits.train.y
        X_cal, y_cal = splits.calibration.X, splits.calibration.y
        X_test, y_test = splits.test.X, splits.test.y
        
        # =====================================================================
        # TRADITIONAL METHODS
        # =====================================================================
        
        # 1. Conformal Prediction + CVaR
        logger.info(f"  Running Conformal Prediction...")
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
        
        # 2. Normal Assumption + CVaR
        logger.info(f"  Running Normal Assumption...")
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
            random_seed=config.cvar.random_seed,
            verbose=False
        )
        
        results["Normal_CVaR"] = compute_all_metrics(
            "Normal_CVaR", y_test, normal_pred.point, normal_orders,
            normal_pred.lower, normal_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
        
        # 3. Quantile Regression + CVaR
        logger.info(f"  Running Quantile Regression...")
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
            random_seed=config.cvar.random_seed,
            verbose=False
        )
        
        results["QuantileReg_CVaR"] = compute_all_metrics(
            "QuantileReg_CVaR", y_test, qr_pred.point, qr_orders,
            qr_pred.lower, qr_pred.upper,
            costs.ordering_cost, costs.holding_cost, costs.stockout_cost
        )
        
        # 4. SAA (Sample Average Approximation)
        logger.info(f"  Running SAA...")
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
        
        # 5. Expected Value (Risk-Neutral Baseline)
        logger.info(f"  Running Expected Value...")
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
        
        # =====================================================================
        # DEEP LEARNING METHODS (if enabled)
        # =====================================================================
        
        if use_deep_learning:
            # Align test data
            y_test_aligned = seq_data.y_test
            
            # 6. LSTM Quantile Regression
            logger.info(f"  Running LSTM Quantile Regression...")
            lstm_model = LSTMQuantileRegression(
                alpha=config.lstm.alpha,
                sequence_length=config.data.sequence_length,
                hidden_size=32,  # Reduced for speed
                num_layers=2,
                dropout=0.2,
                learning_rate=0.001,
                epochs=dl_epochs,
                batch_size=32,
                random_state=config.random_seed,
                device=config.device
            )
            lstm_model.fit(
                seq_data.X_train, seq_data.y_train,
                seq_data.X_cal, seq_data.y_cal
            )
            lstm_pred = lstm_model.predict(seq_data.X_test)
            
            lstm_orders = compute_order_quantities_cvar(
                lstm_pred.point, lstm_pred.lower, lstm_pred.upper,
                beta=config.cvar.beta,
                n_samples=config.cvar.n_samples,
                ordering_cost=costs.ordering_cost,
                holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost,
                random_seed=config.cvar.random_seed,
                verbose=False
            )
            
            results["LSTM_QR"] = compute_all_metrics(
                "LSTM_QR", y_test_aligned, lstm_pred.point, lstm_orders,
                lstm_pred.lower, lstm_pred.upper,
                costs.ordering_cost, costs.holding_cost, costs.stockout_cost
            )
            
            # 7. Transformer Quantile Regression
            logger.info(f"  Running Transformer Quantile Regression...")
            trans_model = TransformerQuantileRegression(
                alpha=config.transformer.alpha,
                sequence_length=config.data.sequence_length,
                d_model=32,  # Reduced for speed
                nhead=4,
                num_layers=2,
                dropout=0.1,
                learning_rate=0.001,
                epochs=dl_epochs,
                batch_size=32,
                random_state=config.random_seed,
                device=config.device
            )
            trans_model.fit(
                seq_data.X_train, seq_data.y_train,
                seq_data.X_cal, seq_data.y_cal
            )
            trans_pred = trans_model.predict(seq_data.X_test)
            
            trans_orders = compute_order_quantities_cvar(
                trans_pred.point, trans_pred.lower, trans_pred.upper,
                beta=config.cvar.beta,
                n_samples=config.cvar.n_samples,
                ordering_cost=costs.ordering_cost,
                holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost,
                random_seed=config.cvar.random_seed,
                verbose=False
            )
            
            results["Transformer_QR"] = compute_all_metrics(
                "Transformer_QR", y_test_aligned, trans_pred.point, trans_orders,
                trans_pred.lower, trans_pred.upper,
                costs.ordering_cost, costs.holding_cost, costs.stockout_cost
            )
            
            # 8. Deep Ensemble
            logger.info(f"  Running Deep Ensemble...")
            ensemble_model = DeepEnsemble(
                alpha=config.deep_ensemble.alpha,
                sequence_length=config.data.sequence_length,
                n_ensemble=3,  # Reduced for speed
                hidden_size=32,
                learning_rate=0.001,
                epochs=dl_epochs,
                batch_size=32,
                random_state=config.random_seed,
                device=config.device
            )
            ensemble_model.fit(
                seq_data.X_train, seq_data.y_train,
                seq_data.X_cal, seq_data.y_cal
            )
            ensemble_pred = ensemble_model.predict(seq_data.X_test)
            
            ensemble_orders = compute_order_quantities_cvar(
                ensemble_pred.point, ensemble_pred.lower, ensemble_pred.upper,
                beta=config.cvar.beta,
                n_samples=config.cvar.n_samples,
                ordering_cost=costs.ordering_cost,
                holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost,
                random_seed=config.cvar.random_seed,
                verbose=False
            )
            
            results["DeepEnsemble"] = compute_all_metrics(
                "DeepEnsemble", y_test_aligned, ensemble_pred.point, ensemble_orders,
                ensemble_pred.lower, ensemble_pred.upper,
                costs.ordering_cost, costs.holding_cost, costs.stockout_cost
            )
            
            # 9. MC Dropout LSTM
            logger.info(f"  Running MC Dropout LSTM...")
            mc_model = MCDropoutLSTM(
                alpha=config.mc_dropout.alpha,
                sequence_length=config.data.sequence_length,
                hidden_size=32,
                num_layers=2,
                dropout=0.2,
                learning_rate=0.001,
                epochs=dl_epochs,
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
            
            results["MCDropout_LSTM"] = compute_all_metrics(
                "MCDropout_LSTM", y_test_aligned, mc_pred.point, mc_orders,
                mc_pred.lower, mc_pred.upper,
                costs.ordering_cost, costs.holding_cost, costs.stockout_cost
            )
            
            # 10. Temporal Fusion Transformer
            logger.info(f"  Running TFT...")
            tft_model = TemporalFusionTransformer(
                alpha=config.tft.alpha,
                sequence_length=config.data.sequence_length,
                hidden_size=32,
                num_heads=4,
                num_layers=2,
                dropout=0.1,
                learning_rate=0.001,
                epochs=dl_epochs,
                batch_size=32,
                random_state=config.random_seed,
                device=config.device
            )
            tft_model.fit(
                seq_data.X_train, seq_data.y_train,
                seq_data.X_cal, seq_data.y_cal
            )
            tft_pred = tft_model.predict(seq_data.X_test)
            
            tft_orders = compute_order_quantities_cvar(
                tft_pred.point, tft_pred.lower, tft_pred.upper,
                beta=config.cvar.beta,
                n_samples=config.cvar.n_samples,
                ordering_cost=costs.ordering_cost,
                holding_cost=costs.holding_cost,
                stockout_cost=costs.stockout_cost,
                random_seed=config.cvar.random_seed,
                verbose=False
            )
            
            results["TFT"] = compute_all_metrics(
                "TFT", y_test_aligned, tft_pred.point, tft_orders,
                tft_pred.lower, tft_pred.upper,
                costs.ordering_cost, costs.holding_cost, costs.stockout_cost
            )
        
        return results
        
    except Exception as e:
        logger.warning(f"Error processing store {store_id}, item {item_id}: {e}")
        return None


def create_multi_store_visualizations(
    results_df: pd.DataFrame,
    output_dir: str
):
    """
    Create comprehensive visualizations for multi-store results.
    
    Parameters
    ----------
    results_df : pd.DataFrame
        Detailed results dataframe.
    output_dir : str
        Output directory for plots.
    """
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    methods = results_df['method'].unique()
    n_methods = len(methods)
    colors = plt.cm.tab10(np.linspace(0, 1, n_methods))
    method_colors = dict(zip(methods, colors))
    
    # =========================================================================
    # 1. Mean Cost Comparison (Box Plot)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    # Order methods by median cost
    method_order = results_df.groupby('method')['mean_cost'].median().sort_values().index.tolist()
    
    bp_data = [results_df[results_df['method'] == m]['mean_cost'].dropna().values for m in method_order]
    bp = ax.boxplot(bp_data, labels=[m.replace('_', '\n') for m in method_order], patch_artist=True)
    
    for patch, method in zip(bp['boxes'], method_order):
        patch.set_facecolor(method_colors[method])
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Mean Cost ($)', fontsize=12)
    ax.set_title('Mean Cost Distribution by Method (All SKUs)', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "mean_cost_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: mean_cost_boxplot.png")
    
    # =========================================================================
    # 2. CVaR-90 Comparison (Box Plot)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 8))
    
    method_order = results_df.groupby('method')['cvar_90'].median().sort_values().index.tolist()
    
    bp_data = [results_df[results_df['method'] == m]['cvar_90'].dropna().values for m in method_order]
    bp = ax.boxplot(bp_data, labels=[m.replace('_', '\n') for m in method_order], patch_artist=True)
    
    for patch, method in zip(bp['boxes'], method_order):
        patch.set_facecolor(method_colors[method])
        patch.set_alpha(0.7)
    
    ax.set_ylabel('CVaR-90 ($)', fontsize=12)
    ax.set_title('CVaR-90 (Tail Risk) Distribution by Method', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='y')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "cvar90_boxplot.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: cvar90_boxplot.png")
    
    # =========================================================================
    # 3. Coverage Comparison (Bar Plot with Error Bars)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    coverage_stats = results_df.groupby('method')['coverage'].agg(['mean', 'std']).reset_index()
    coverage_stats = coverage_stats.sort_values('mean', ascending=False)
    
    x = np.arange(len(coverage_stats))
    bars = ax.bar(x, coverage_stats['mean'] * 100, 
                   yerr=coverage_stats['std'] * 100, 
                   capsize=5,
                   color=[method_colors.get(m, 'gray') for m in coverage_stats['method']],
                   alpha=0.7, edgecolor='black')
    
    ax.axhline(95, color='red', linestyle='--', linewidth=2, label='95% Target')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in coverage_stats['method']], rotation=45, ha='right')
    ax.set_ylabel('Coverage (%)', fontsize=12)
    ax.set_title('Prediction Interval Coverage by Method', fontsize=14, fontweight='bold')
    ax.set_ylim([80, 100])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "coverage_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: coverage_comparison.png")
    
    # =========================================================================
    # 4. Service Level Comparison
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 7))
    
    service_stats = results_df.groupby('method')['service_level'].agg(['mean', 'std']).reset_index()
    service_stats = service_stats.sort_values('mean', ascending=False)
    
    x = np.arange(len(service_stats))
    bars = ax.bar(x, service_stats['mean'] * 100,
                   yerr=service_stats['std'] * 100,
                   capsize=5,
                   color=[method_colors.get(m, 'gray') for m in service_stats['method']],
                   alpha=0.7, edgecolor='black')
    
    ax.axhline(95, color='red', linestyle='--', linewidth=2, label='95% Target')
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in service_stats['method']], rotation=45, ha='right')
    ax.set_ylabel('Service Level (%)', fontsize=12)
    ax.set_title('Service Level by Method', fontsize=14, fontweight='bold')
    ax.set_ylim([70, 100])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "service_level_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: service_level_comparison.png")
    
    # =========================================================================
    # 5. Risk-Return Tradeoff (Scatter)
    # =========================================================================
    fig, ax = plt.subplots(figsize=(12, 8))
    
    agg = results_df.groupby('method').agg({
        'mean_cost': ['mean', 'std'],
        'cvar_90': ['mean', 'std']
    }).reset_index()
    agg.columns = ['method', 'mean_cost_avg', 'mean_cost_std', 'cvar_90_avg', 'cvar_90_std']
    
    for _, row in agg.iterrows():
        ax.scatter(row['mean_cost_avg'], row['cvar_90_avg'], 
                  s=200, color=method_colors.get(row['method'], 'gray'),
                  alpha=0.7, edgecolor='black', linewidth=2)
        ax.annotate(row['method'].replace('_', '\n'), 
                   (row['mean_cost_avg'], row['cvar_90_avg']),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=9, fontweight='bold')
    
    ax.set_xlabel('Mean Cost ($) - Lower is Better', fontsize=12)
    ax.set_ylabel('CVaR-90 ($) - Lower is Better', fontsize=12)
    ax.set_title('Risk-Return Tradeoff: Mean Cost vs Tail Risk', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "risk_return_tradeoff.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: risk_return_tradeoff.png")
    
    # =========================================================================
    # 6. Traditional vs Deep Learning Comparison
    # =========================================================================
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    traditional = ['Conformal_CVaR', 'Normal_CVaR', 'QuantileReg_CVaR', 'SAA', 'ExpectedValue']
    deep_learning = ['LSTM_QR', 'Transformer_QR', 'DeepEnsemble', 'MCDropout_LSTM', 'TFT']
    
    results_df['category'] = results_df['method'].apply(
        lambda x: 'Traditional' if x in traditional else 'Deep Learning'
    )
    
    metrics = [('mean_cost', 'Mean Cost ($)'), ('cvar_90', 'CVaR-90 ($)'), ('service_level', 'Service Level')]
    
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
        cat_data = results_df.groupby('category')[metric].agg(['mean', 'std']).reset_index()
        
        x = np.arange(len(cat_data))
        colors_cat = ['steelblue', 'coral']
        bars = ax.bar(x, cat_data['mean'], yerr=cat_data['std'], capsize=8,
                       color=colors_cat, alpha=0.7, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(cat_data['category'], fontsize=11)
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(alpha=0.3, axis='y')
        
        # Add value labels
        for bar, val in zip(bars, cat_data['mean']):
            height = bar.get_height()
            if metric == 'service_level':
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{val:.1%}', ha='center', va='bottom', fontsize=10)
            else:
                ax.text(bar.get_x() + bar.get_width()/2., height + 5,
                       f'${val:.0f}', ha='center', va='bottom', fontsize=10)
    
    plt.suptitle('Traditional Methods vs Deep Learning', fontsize=14, fontweight='bold', y=1.02)
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "traditional_vs_dl.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: traditional_vs_dl.png")
    
    # =========================================================================
    # 7. Heatmap: Method Performance Across SKUs
    # =========================================================================
    fig, ax = plt.subplots(figsize=(14, 10))
    
    # Create pivot table for CVaR-90
    pivot = results_df.pivot_table(
        values='cvar_90', 
        index=['store_id', 'item_id'], 
        columns='method',
        aggfunc='mean'
    )
    
    # Normalize by row (each SKU)
    pivot_norm = pivot.sub(pivot.min(axis=1), axis=0).div(
        pivot.max(axis=1) - pivot.min(axis=1), axis=0
    )
    
    sns.heatmap(pivot_norm, cmap='RdYlGn_r', annot=False, ax=ax,
                cbar_kws={'label': 'Normalized CVaR-90 (0=best, 1=worst)'})
    ax.set_title('Method Performance Across SKUs (CVaR-90)', fontsize=14, fontweight='bold')
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('Store-Item', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, "performance_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: performance_heatmap.png")
    
    # =========================================================================
    # 8. Summary Table Image
    # =========================================================================
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')
    
    summary = results_df.groupby('method').agg({
        'mean_cost': ['mean', 'std'],
        'cvar_90': ['mean', 'std'],
        'cvar_95': ['mean', 'std'],
        'service_level': ['mean', 'std'],
        'coverage': ['mean', 'std'],
    }).round(2)
    
    table_data = []
    for method in summary.index:
        row = [
            method,
            f"${summary.loc[method, ('mean_cost', 'mean')]:.0f} ± {summary.loc[method, ('mean_cost', 'std')]:.0f}",
            f"${summary.loc[method, ('cvar_90', 'mean')]:.0f} ± {summary.loc[method, ('cvar_90', 'std')]:.0f}",
            f"${summary.loc[method, ('cvar_95', 'mean')]:.0f} ± {summary.loc[method, ('cvar_95', 'std')]:.0f}",
            f"{summary.loc[method, ('service_level', 'mean')]*100:.1f}%",
            f"{summary.loc[method, ('coverage', 'mean')]*100:.1f}%" if not np.isnan(summary.loc[method, ('coverage', 'mean')]) else "N/A",
        ]
        table_data.append(row)
    
    columns = ['Method', 'Mean Cost', 'CVaR-90', 'CVaR-95', 'Service Level', 'Coverage']
    
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])
    
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
    
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(table_data) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.title('Multi-Store Experiment Summary\n(Mean ± Std across all SKUs)',
             fontsize=14, fontweight='bold', pad=20)
    
    plt.savefig(os.path.join(viz_dir, "summary_table.png"), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Saved: summary_table.png")
    
    logger.info(f"\n✅ All visualizations saved to: {viz_dir}/")


def run_multi_store_experiment_full(
    filepath: str,
    store_ids: List[int],
    item_ids: List[int],
    config: ExperimentConfig,
    output_dir: str = "results/multi_store_full",
    use_deep_learning: bool = True,
    dl_epochs: int = 50
) -> pd.DataFrame:
    """
    Run full experiments across multiple store-item combinations.
    
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
    use_deep_learning : bool
        Whether to run DL models.
    dl_epochs : int
        Epochs for DL models.
        
    Returns
    -------
    pd.DataFrame
        Aggregated results.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("MULTI-STORE FULL EXPERIMENT")
    logger.info("=" * 70)
    logger.info(f"Stores: {store_ids}")
    logger.info(f"Items: {item_ids}")
    logger.info(f"Total combinations: {len(store_ids) * len(item_ids)}")
    logger.info(f"Deep Learning: {'Enabled' if use_deep_learning else 'Disabled'}")
    if use_deep_learning:
        logger.info(f"DL Epochs: {dl_epochs}")
    logger.info(f"Device: {config.device}")
    
    # Load raw data once
    df_raw = load_raw_data(filepath)
    
    # Collect all results
    all_results = []
    
    total = len(store_ids) * len(item_ids)
    with tqdm(total=total, desc="Processing SKUs") as pbar:
        for store_id in store_ids:
            for item_id in item_ids:
                logger.info(f"\nProcessing Store {store_id}, Item {item_id}")
                
                results = run_single_store_item_full(
                    df_raw, store_id, item_id, config,
                    use_deep_learning=use_deep_learning,
                    dl_epochs=dl_epochs
                )
                
                if results is not None:
                    for method_name, method_results in results.items():
                        row = {
                            "store_id": store_id,
                            "item_id": item_id,
                            "method": method_name,
                            "coverage": method_results.forecast_metrics.coverage,
                            "avg_interval_width": method_results.forecast_metrics.avg_interval_width,
                            "mae": method_results.forecast_metrics.mae,
                            "rmse": method_results.forecast_metrics.rmse,
                            "mape": method_results.forecast_metrics.mape,
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
    logger.info(f"\n✓ Saved detailed results: {detailed_path}")
    
    # Create summary statistics
    summary = results_df.groupby("method").agg({
        "coverage": ["mean", "std"],
        "mean_cost": ["mean", "std"],
        "cvar_90": ["mean", "std"],
        "cvar_95": ["mean", "std"],
        "service_level": ["mean", "std"],
        "mae": ["mean", "std"],
    }).round(4)
    
    summary_path = os.path.join(output_dir, "summary_statistics.csv")
    summary.to_csv(summary_path)
    logger.info(f"✓ Saved summary statistics: {summary_path}")
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("SUMMARY STATISTICS (Mean ± Std)")
    logger.info("=" * 70)
    print(summary.to_string())
    
    # Method rankings
    logger.info("\n" + "=" * 70)
    logger.info("METHOD RANKINGS")
    logger.info("=" * 70)
    
    mean_cvar90 = results_df.groupby("method")["cvar_90"].mean().sort_values()
    logger.info("\nBy CVaR-90 (lower is better):")
    for i, (method, value) in enumerate(mean_cvar90.items(), 1):
        logger.info(f"  {i}. {method}: ${value:.2f}")
    
    mean_cost = results_df.groupby("method")["mean_cost"].mean().sort_values()
    logger.info("\nBy Mean Cost (lower is better):")
    for i, (method, value) in enumerate(mean_cost.items(), 1):
        logger.info(f"  {i}. {method}: ${value:.2f}")
    
    mean_coverage = results_df.groupby("method")["coverage"].mean()
    coverage_gap = abs(mean_coverage - 0.95).sort_values()
    logger.info("\nBy Coverage (closest to 95% is better):")
    for i, (method, gap) in enumerate(coverage_gap.items(), 1):
        actual = mean_coverage[method]
        if not np.isnan(actual):
            logger.info(f"  {i}. {method}: {actual:.1%} (gap: {gap:.1%})")
    
    # Create visualizations
    logger.info("\n" + "=" * 70)
    logger.info("CREATING VISUALIZATIONS")
    logger.info("=" * 70)
    create_multi_store_visualizations(results_df, output_dir)
    
    return results_df


def main():
    parser = argparse.ArgumentParser(description="Run Full Multi-Store Experiment")
    parser.add_argument("--data", type=str, default="train.csv", help="Data file path")
    parser.add_argument("--stores", type=str, default="1,2,3", help="Comma-separated store IDs")
    parser.add_argument("--items", type=str, default="1,2,3,4,5", help="Comma-separated item IDs")
    parser.add_argument("--output", type=str, default="results/multi_store_full", help="Output directory")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/cuda)")
    parser.add_argument("--no-dl", action="store_true", help="Skip deep learning models")
    parser.add_argument("--dl-epochs", type=int, default=50, help="Epochs for DL models")
    
    args = parser.parse_args()
    
    store_ids = [int(s) for s in args.stores.split(",")]
    item_ids = [int(i) for i in args.items.split(",")]
    
    config = get_default_config()
    config.device = args.device
    
    # Set random seeds
    np.random.seed(config.random_seed)
    torch.manual_seed(config.random_seed)
    
    run_multi_store_experiment_full(
        filepath=args.data,
        store_ids=store_ids,
        item_ids=item_ids,
        config=config,
        output_dir=args.output,
        use_deep_learning=not args.no_dl,
        dl_epochs=args.dl_epochs
    )
    
    logger.info("\n" + "=" * 70)
    logger.info("EXPERIMENT COMPLETE")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()