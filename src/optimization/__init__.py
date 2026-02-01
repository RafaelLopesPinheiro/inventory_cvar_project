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
]
