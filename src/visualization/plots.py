"""
Visualization module for forecasting and inventory results.

This module provides:
- Prediction interval plots
- Cost comparison charts
- Coverage analysis
- Training curves for DL models
"""

import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Tuple
import logging

from ..evaluation.metrics import MethodResults

logger = logging.getLogger(__name__)

# Set style
plt.style.use('seaborn-v0_8-whitegrid')


def plot_prediction_intervals(
    y_true: np.ndarray,
    results_dict: Dict[str, MethodResults],
    title: str = "Prediction Intervals",
    figsize: Tuple[int, int] = (14, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot prediction intervals for multiple methods.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual demand values.
    results_dict : Dict[str, MethodResults]
        Results for each method.
    title : str
        Plot title.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save the figure.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    x = np.arange(len(y_true))
    ax.plot(x, y_true, 'k-', linewidth=2, label='Actual Demand', alpha=0.8)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for i, (method_name, results) in enumerate(results_dict.items()):
        if results.lower_bounds is not None and results.upper_bounds is not None:
            ax.fill_between(
                x, results.lower_bounds, results.upper_bounds,
                alpha=0.2, color=colors[i], label=f"{method_name} interval"
            )
    
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Demand', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig


def plot_coverage_comparison(
    results_dict: Dict[str, MethodResults],
    target_coverage: float = 0.95,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot coverage comparison across methods.
    
    Parameters
    ----------
    results_dict : Dict[str, MethodResults]
        Results for each method.
    target_coverage : float
        Target coverage level.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save the figure.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = []
    coverages = []
    
    for method_name, results in results_dict.items():
        if results.forecast_metrics.coverage is not None:
            methods.append(method_name.replace('_', '\n'))
            coverages.append(results.forecast_metrics.coverage)
    
    colors = ['green' if abs(c - target_coverage) < 0.02 else 'orange' if c > target_coverage else 'red' 
              for c in coverages]
    
    bars = ax.bar(methods, coverages, color=colors, alpha=0.7, edgecolor='black')
    ax.axhline(target_coverage, color='red', linestyle='--', linewidth=2, label=f'{target_coverage:.0%} Target')
    
    ax.set_ylabel('Coverage Rate', fontsize=12)
    ax.set_title('Coverage Comparison', fontsize=14)
    ax.set_ylim([0.75, 1.0])
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.005,
                f'{height:.1%}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig


def plot_cvar_comparison(
    results_dict: Dict[str, MethodResults],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot CVaR comparison across methods.
    
    Parameters
    ----------
    results_dict : Dict[str, MethodResults]
        Results for each method.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save the figure.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = list(results_dict.keys())
    cvar_90 = [results_dict[m].inventory_metrics.cvar_90 for m in methods]
    cvar_95 = [results_dict[m].inventory_metrics.cvar_95 for m in methods]
    
    # Sort by CVaR-90
    sorted_idx = np.argsort(cvar_90)
    methods = [methods[i] for i in sorted_idx]
    cvar_90 = [cvar_90[i] for i in sorted_idx]
    cvar_95 = [cvar_95[i] for i in sorted_idx]
    
    x = np.arange(len(methods))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, cvar_90, width, label='CVaR-90', alpha=0.7, color='steelblue')
    bars2 = ax.bar(x + width/2, cvar_95, width, label='CVaR-95', alpha=0.7, color='coral')
    
    ax.set_xlabel('Method', fontsize=12)
    ax.set_ylabel('CVaR ($)', fontsize=12)
    ax.set_title('CVaR Comparison (Lower is Better)', fontsize=14)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace('_', '\n') for m in methods], fontsize=9)
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig


def plot_cost_distributions(
    results_dict: Dict[str, MethodResults],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cost distributions as boxplots.
    
    Parameters
    ----------
    results_dict : Dict[str, MethodResults]
        Results for each method.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save the figure.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    methods = list(results_dict.keys())
    cost_data = [results_dict[m].costs for m in methods]
    
    bp = ax.boxplot(cost_data, labels=[m.replace('_', '\n') for m in methods],
                    patch_artist=True)
    
    colors = plt.cm.Set3(np.linspace(0, 1, len(methods)))
    for patch, color in zip(bp['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Daily Cost ($)', fontsize=12)
    ax.set_title('Cost Distributions', fontsize=14)
    ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig


def plot_cumulative_costs(
    results_dict: Dict[str, MethodResults],
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot cumulative costs over time.
    
    Parameters
    ----------
    results_dict : Dict[str, MethodResults]
        Results for each method.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save the figure.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.tab10(np.linspace(0, 1, len(results_dict)))
    
    for i, (method_name, results) in enumerate(results_dict.items()):
        cumulative = np.cumsum(results.costs)
        ax.plot(cumulative, linewidth=2, color=colors[i],
                label=f"{method_name} (${cumulative[-1]:,.0f})")
    
    ax.set_xlabel('Day', fontsize=12)
    ax.set_ylabel('Cumulative Cost ($)', fontsize=12)
    ax.set_title('Cumulative Costs Over Time', fontsize=14)
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig


def plot_training_curves(
    training_losses: Dict[str, List[float]],
    figsize: Tuple[int, int] = (12, 8),
    save_path: Optional[str] = None
) -> plt.Figure:
    """
    Plot training loss curves for DL models.
    
    Parameters
    ----------
    training_losses : Dict[str, List[float]]
        Training losses for each model.
    figsize : Tuple[int, int]
        Figure size.
    save_path : str, optional
        Path to save the figure.
        
    Returns
    -------
    plt.Figure
        The figure object.
    """
    n_models = len(training_losses)
    n_cols = 2
    n_rows = (n_models + 1) // 2
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
    axes = axes.flatten() if n_models > 1 else [axes]
    
    for idx, (name, losses) in enumerate(training_losses.items()):
        if idx < len(axes) and losses:
            axes[idx].plot(losses, 'b-', linewidth=1)
            axes[idx].set_xlabel('Epoch')
            axes[idx].set_ylabel('Loss')
            axes[idx].set_title(f'{name}')
            axes[idx].grid(alpha=0.3)
    
    # Hide unused subplots
    for idx in range(len(training_losses), len(axes)):
        axes[idx].set_visible(False)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Saved: {save_path}")
    
    return fig


def create_comprehensive_visualization(
    y_true: np.ndarray,
    results_dict: Dict[str, MethodResults],
    training_losses: Optional[Dict[str, List[float]]] = None,
    output_dir: str = "results",
    prefix: str = ""
) -> List[plt.Figure]:
    """
    Create all visualization plots.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual demand values.
    results_dict : Dict[str, MethodResults]
        Results for each method.
    training_losses : Dict[str, List[float]], optional
        Training losses for DL models.
    output_dir : str
        Directory to save plots.
    prefix : str
        Prefix for file names.
        
    Returns
    -------
    List[plt.Figure]
        List of figure objects.
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    figures = []
    
    # 1. Prediction intervals
    fig1 = plot_prediction_intervals(
        y_true, results_dict,
        save_path=os.path.join(output_dir, f"{prefix}prediction_intervals.png")
    )
    figures.append(fig1)
    
    # 2. Coverage comparison
    fig2 = plot_coverage_comparison(
        results_dict,
        save_path=os.path.join(output_dir, f"{prefix}coverage_comparison.png")
    )
    figures.append(fig2)
    
    # 3. CVaR comparison
    fig3 = plot_cvar_comparison(
        results_dict,
        save_path=os.path.join(output_dir, f"{prefix}cvar_comparison.png")
    )
    figures.append(fig3)
    
    # 4. Cost distributions
    fig4 = plot_cost_distributions(
        results_dict,
        save_path=os.path.join(output_dir, f"{prefix}cost_distributions.png")
    )
    figures.append(fig4)
    
    # 5. Cumulative costs
    fig5 = plot_cumulative_costs(
        results_dict,
        save_path=os.path.join(output_dir, f"{prefix}cumulative_costs.png")
    )
    figures.append(fig5)
    
    # 6. Training curves (if available)
    if training_losses:
        fig6 = plot_training_curves(
            training_losses,
            save_path=os.path.join(output_dir, f"{prefix}training_curves.png")
        )
        figures.append(fig6)
    
    logger.info(f"Created {len(figures)} visualization plots in {output_dir}")
    
    return figures
