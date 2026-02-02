"""Optimization module for inventory decisions."""

from .cvar import (
    CostParameters,
    newsvendor_loss,
    optimize_cvar_single,
    compute_order_quantities_cvar,
    compute_order_quantities_newsvendor,
    compute_expected_cost,
    # DRO optimization
    optimize_wasserstein_dro_single,
    compute_order_quantities_dro,
    # Multi-period optimization
    multi_period_newsvendor_loss,
    optimize_multi_period_cvar_single,
    compute_order_quantities_multi_period_cvar,
    MultiPeriodCostMetrics,
    compute_multi_period_metrics,
)

__all__ = [
    "CostParameters",
    "newsvendor_loss",
    "optimize_cvar_single",
    "compute_order_quantities_cvar",
    "compute_order_quantities_newsvendor",
    "compute_expected_cost",
    # DRO optimization
    "optimize_wasserstein_dro_single",
    "compute_order_quantities_dro",
    # Multi-period optimization
    "multi_period_newsvendor_loss",
    "optimize_multi_period_cvar_single",
    "compute_order_quantities_multi_period_cvar",
    "MultiPeriodCostMetrics",
    "compute_multi_period_metrics",
]
