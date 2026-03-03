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
    # Inventory-aware sequential optimization
    compute_inventory_aware_orders_cvar,
    compute_inventory_aware_orders_dro,
    # Multi-period optimization
    multi_period_newsvendor_loss,
    optimize_multi_period_cvar_single,
    compute_order_quantities_multi_period_cvar,
    MultiPeriodCostMetrics,
    compute_multi_period_metrics,
    # Inventory simulation with carryover and capacity
    InventorySimulationResult,
    simulate_inventory_with_carryover,
    simulate_sS_policy_with_carryover,
    # Lead-time simulation
    simulate_inventory_with_lead_time,
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
    # Inventory-aware sequential optimization
    "compute_inventory_aware_orders_cvar",
    "compute_inventory_aware_orders_dro",
    # Multi-period optimization
    "multi_period_newsvendor_loss",
    "optimize_multi_period_cvar_single",
    "compute_order_quantities_multi_period_cvar",
    "MultiPeriodCostMetrics",
    "compute_multi_period_metrics",
    # Inventory simulation with carryover and capacity
    "InventorySimulationResult",
    "simulate_inventory_with_carryover",
    # (s, S) policy simulation
    "simulate_sS_policy_with_carryover",
    # Lead-time simulation
    "simulate_inventory_with_lead_time",
]
