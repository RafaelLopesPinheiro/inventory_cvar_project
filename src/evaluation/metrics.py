"""
Evaluation metrics for forecasting and inventory optimization.

This module provides:
- Forecasting metrics (coverage, interval width, etc.)
- Inventory metrics (cost, service level, CVaR)
- Statistical tests for method comparison
"""

import numpy as np
import pandas as pd
from scipy import stats
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
import logging

logger = logging.getLogger(__name__)


@dataclass
class ForecastMetrics:
    """Container for forecasting evaluation metrics."""
    coverage: Optional[float] = None
    avg_interval_width: Optional[float] = None
    mae: Optional[float] = None
    rmse: Optional[float] = None
    mape: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "coverage": self.coverage,
            "avg_interval_width": self.avg_interval_width,
            "mae": self.mae,
            "rmse": self.rmse,
            "mape": self.mape
        }


@dataclass
class InventoryMetrics:
    """Container for inventory optimization metrics."""
    mean_cost: float = 0.0
    total_cost: float = 0.0
    cvar_90: float = 0.0
    cvar_95: float = 0.0
    service_level: float = 0.0
    avg_order_quantity: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "mean_cost": self.mean_cost,
            "total_cost": self.total_cost,
            "cvar_90": self.cvar_90,
            "cvar_95": self.cvar_95,
            "service_level": self.service_level,
            "avg_order_quantity": self.avg_order_quantity
        }


@dataclass
class MethodResults:
    """Complete results for a single method."""
    method_name: str
    forecast_metrics: ForecastMetrics
    inventory_metrics: InventoryMetrics
    costs: np.ndarray = field(default_factory=lambda: np.array([]))
    order_quantities: np.ndarray = field(default_factory=lambda: np.array([]))
    point_predictions: np.ndarray = field(default_factory=lambda: np.array([]))
    lower_bounds: Optional[np.ndarray] = None
    upper_bounds: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        result = {"method": self.method_name}
        result.update(self.forecast_metrics.to_dict())
        result.update(self.inventory_metrics.to_dict())
        return result


def compute_forecast_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None
) -> ForecastMetrics:
    """
    Compute forecasting evaluation metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual values.
    y_pred : np.ndarray
        Point predictions.
    lower : np.ndarray, optional
        Lower bounds of prediction intervals.
    upper : np.ndarray, optional
        Upper bounds of prediction intervals.
        
    Returns
    -------
    ForecastMetrics
        Computed metrics.
    """
    # Point prediction metrics
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred) ** 2))
    
    # MAPE (handle zeros)
    mask = y_true != 0
    if mask.sum() > 0:
        mape = np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100
    else:
        mape = np.nan
    
    # Interval metrics
    coverage = None
    avg_width = None
    
    if lower is not None and upper is not None:
        coverage = np.mean((y_true >= lower) & (y_true <= upper))
        avg_width = np.mean(upper - lower)
    
    return ForecastMetrics(
        coverage=coverage,
        avg_interval_width=avg_width,
        mae=mae,
        rmse=rmse,
        mape=mape
    )


def compute_inventory_metrics(
    y_true: np.ndarray,
    order_quantities: np.ndarray,
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0
) -> Tuple[InventoryMetrics, np.ndarray]:
    """
    Compute inventory optimization metrics.
    
    Parameters
    ----------
    y_true : np.ndarray
        Actual demand.
    order_quantities : np.ndarray
        Order quantities.
    ordering_cost, holding_cost, stockout_cost : float
        Cost parameters.
        
    Returns
    -------
    Tuple[InventoryMetrics, np.ndarray]
        Metrics and daily costs.
    """
    # Compute costs
    overage = np.maximum(0, order_quantities - y_true)
    underage = np.maximum(0, y_true - order_quantities)
    costs = ordering_cost * order_quantities + holding_cost * overage + stockout_cost * underage
    
    # Basic metrics
    mean_cost = np.mean(costs)
    total_cost = np.sum(costs)
    
    # CVaR metrics
    sorted_costs = np.sort(costs)
    n = len(costs)
    cvar_90_idx = int(np.ceil(0.90 * n))
    cvar_95_idx = int(np.ceil(0.95 * n))
    cvar_90 = np.mean(sorted_costs[cvar_90_idx:])
    cvar_95 = np.mean(sorted_costs[cvar_95_idx:])
    
    # Service level
    service_level = np.mean(order_quantities >= y_true)
    
    # Average order quantity
    avg_order = np.mean(order_quantities)
    
    metrics = InventoryMetrics(
        mean_cost=mean_cost,
        total_cost=total_cost,
        cvar_90=cvar_90,
        cvar_95=cvar_95,
        service_level=service_level,
        avg_order_quantity=avg_order
    )
    
    return metrics, costs


def compute_all_metrics(
    method_name: str,
    y_true: np.ndarray,
    y_pred: np.ndarray,
    order_quantities: np.ndarray,
    lower: Optional[np.ndarray] = None,
    upper: Optional[np.ndarray] = None,
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0
) -> MethodResults:
    """
    Compute all metrics for a method.
    
    Parameters
    ----------
    method_name : str
        Name of the method.
    y_true : np.ndarray
        Actual demand.
    y_pred : np.ndarray
        Point predictions.
    order_quantities : np.ndarray
        Order quantities.
    lower, upper : np.ndarray, optional
        Prediction interval bounds.
    ordering_cost, holding_cost, stockout_cost : float
        Cost parameters.
        
    Returns
    -------
    MethodResults
        Complete results for the method.
    """
    forecast_metrics = compute_forecast_metrics(y_true, y_pred, lower, upper)
    inventory_metrics, costs = compute_inventory_metrics(
        y_true, order_quantities,
        ordering_cost, holding_cost, stockout_cost
    )
    
    return MethodResults(
        method_name=method_name,
        forecast_metrics=forecast_metrics,
        inventory_metrics=inventory_metrics,
        costs=costs,
        order_quantities=order_quantities,
        point_predictions=y_pred,
        lower_bounds=lower,
        upper_bounds=upper
    )


# =============================================================================
# STATISTICAL TESTS
# =============================================================================

@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    method1: str
    method2: str
    mean_diff: float
    t_statistic: float
    p_value: float
    significant: bool
    better_method: Optional[str] = None
    
    def __str__(self) -> str:
        sig_str = "***" if self.p_value < 0.001 else "**" if self.p_value < 0.01 else "*" if self.p_value < 0.05 else "ns"
        return (
            f"{self.method1} vs {self.method2}: "
            f"diff={self.mean_diff:+.2f}, t={self.t_statistic:.3f}, "
            f"p={self.p_value:.4f} {sig_str}"
        )


def paired_t_test(
    costs1: np.ndarray,
    costs2: np.ndarray,
    method1_name: str,
    method2_name: str,
    alpha: float = 0.05
) -> StatisticalTestResult:
    """
    Perform paired t-test comparing two methods.
    
    Parameters
    ----------
    costs1, costs2 : np.ndarray
        Daily costs for each method.
    method1_name, method2_name : str
        Names of the methods.
    alpha : float
        Significance level.
        
    Returns
    -------
    StatisticalTestResult
        Test results.
    """
    t_stat, p_value = stats.ttest_rel(costs1, costs2)
    mean_diff = np.mean(costs1) - np.mean(costs2)
    significant = p_value < alpha
    
    better_method = None
    if significant:
        better_method = method2_name if mean_diff > 0 else method1_name
    
    return StatisticalTestResult(
        method1=method1_name,
        method2=method2_name,
        mean_diff=mean_diff,
        t_statistic=t_stat,
        p_value=p_value,
        significant=significant,
        better_method=better_method
    )


def compare_methods(
    results_dict: Dict[str, MethodResults],
    baseline_name: str = "Conformal_CVaR",
    alpha: float = 0.05
) -> List[StatisticalTestResult]:
    """
    Compare all methods against a baseline using paired t-tests.
    
    Parameters
    ----------
    results_dict : Dict[str, MethodResults]
        Results for all methods.
    baseline_name : str
        Name of the baseline method.
    alpha : float
        Significance level.
        
    Returns
    -------
    List[StatisticalTestResult]
        Test results for all comparisons.
    """
    if baseline_name not in results_dict:
        raise ValueError(f"Baseline {baseline_name} not found in results")
    
    baseline_costs = results_dict[baseline_name].costs
    test_results = []
    
    for method_name, method_results in results_dict.items():
        if method_name == baseline_name:
            continue
        
        result = paired_t_test(
            method_results.costs,
            baseline_costs,
            method_name,
            baseline_name,
            alpha
        )
        test_results.append(result)
    
    return test_results


def create_results_summary(
    results_dict: Dict[str, MethodResults]
) -> pd.DataFrame:
    """
    Create a summary DataFrame of all results.
    
    Parameters
    ----------
    results_dict : Dict[str, MethodResults]
        Results for all methods.
        
    Returns
    -------
    pd.DataFrame
        Summary table.
    """
    rows = []
    for method_name, results in results_dict.items():
        row = results.to_dict()
        rows.append(row)
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    column_order = [
        "method", "coverage", "avg_interval_width",
        "mean_cost", "cvar_90", "cvar_95",
        "service_level", "total_cost"
    ]
    existing_cols = [c for c in column_order if c in df.columns]
    other_cols = [c for c in df.columns if c not in column_order]
    df = df[existing_cols + other_cols]
    
    return df.sort_values("cvar_90")
