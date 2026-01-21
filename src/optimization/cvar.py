"""
CVaR (Conditional Value-at-Risk) optimization for inventory decisions.

This module implements:
- Newsvendor loss function
- CVaR optimization via Rockafellar-Uryasev formulation
- Order quantity computation

References:
- Rockafellar & Uryasev (2000) "Optimization of conditional value-at-risk"
- Rockafellar & Uryasev (2002) "Conditional value-at-risk for general loss distributions"
"""

import numpy as np
from scipy.optimize import minimize
from typing import Tuple, Optional
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class CostParameters:
    """Newsvendor cost parameters."""
    ordering_cost: float = 10.0
    holding_cost: float = 2.0
    stockout_cost: float = 50.0
    
    @property
    def critical_ratio(self) -> float:
        """Critical ratio: cu / (cu + co)"""
        return self.stockout_cost / (self.stockout_cost + self.holding_cost)


def newsvendor_loss(
    q: np.ndarray,
    d: np.ndarray,
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0
) -> np.ndarray:
    """
    Compute newsvendor loss for given order quantities and demands.
    
    Loss(q, d) = ordering_cost * q + holding_cost * max(0, q - d) 
                + stockout_cost * max(0, d - q)
    
    Parameters
    ----------
    q : np.ndarray
        Order quantities.
    d : np.ndarray
        Actual demand.
    ordering_cost : float
        Cost per unit ordered.
    holding_cost : float
        Cost per unit of overage (inventory left over).
    stockout_cost : float
        Cost per unit of underage (lost sales).
        
    Returns
    -------
    np.ndarray
        Loss values.
    """
    overage = np.maximum(0, q - d)
    underage = np.maximum(0, d - q)
    return ordering_cost * q + holding_cost * overage + stockout_cost * underage


def optimize_cvar_single(
    demand_samples: np.ndarray,
    beta: float = 0.90,
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0
) -> float:
    """
    Optimize order quantity using CVaR via Rockafellar-Uryasev formulation.
    
    The formulation is:
        min_{q, τ} τ + (1 / (N * (1 - β))) * Σ max(0, Loss(q, d_i) - τ)
    
    Parameters
    ----------
    demand_samples : np.ndarray
        Samples from the demand distribution.
    beta : float
        CVaR level (tail probability).
    ordering_cost, holding_cost, stockout_cost : float
        Cost parameters.
        
    Returns
    -------
    float
        Optimal order quantity.
    """
    n_samples = len(demand_samples)
    
    def cvar_objective(x: np.ndarray) -> float:
        q, tau = x
        losses = newsvendor_loss(
            q, demand_samples,
            ordering_cost, holding_cost, stockout_cost
        )
        cvar_term = tau + (1 / (n_samples * (1 - beta))) * np.sum(
            np.maximum(0, losses - tau)
        )
        return cvar_term
    
    # Initial guess
    q0 = np.mean(demand_samples)
    tau0 = np.median(newsvendor_loss(q0, demand_samples, ordering_cost, holding_cost, stockout_cost))
    
    # Bounds: q >= 0, tau unrestricted
    bounds = [(0, None), (None, None)]
    
    result = minimize(
        cvar_objective,
        [q0, tau0],
        method='L-BFGS-B',
        bounds=bounds
    )
    
    return max(0, result.x[0])


def compute_order_quantities_cvar(
    point_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    beta: float = 0.90,
    n_samples: int = 1000,
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0,
    random_seed: int = 42,
    verbose: bool = True
) -> np.ndarray:
    """
    Compute CVaR-optimal order quantities for all predictions.
    
    Samples demand scenarios uniformly from prediction intervals and
    optimizes CVaR for each day.
    
    Parameters
    ----------
    point_pred : np.ndarray
        Point predictions.
    lower : np.ndarray
        Lower bounds of prediction intervals.
    upper : np.ndarray
        Upper bounds of prediction intervals.
    beta : float
        CVaR level.
    n_samples : int
        Number of demand samples to generate.
    ordering_cost, holding_cost, stockout_cost : float
        Cost parameters.
    random_seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress.
        
    Returns
    -------
    np.ndarray
        Optimal order quantities.
    """
    if verbose:
        logger.info(f"Optimizing CVaR (beta={beta}) for {len(point_pred)} days...")
    
    rng = np.random.RandomState(random_seed)
    order_quantities = []
    
    for i in range(len(point_pred)):
        if verbose and (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(point_pred)} days...")
        
        # Sample demand scenarios from prediction interval
        demand_samples = rng.uniform(lower[i], upper[i], n_samples)
        
        # Optimize CVaR
        q_opt = optimize_cvar_single(
            demand_samples, beta,
            ordering_cost, holding_cost, stockout_cost
        )
        order_quantities.append(q_opt)
    
    return np.array(order_quantities)


def compute_order_quantities_newsvendor(
    point_pred: np.ndarray,
    sigma: np.ndarray,
    critical_ratio: float
) -> np.ndarray:
    """
    Compute order quantities using classical newsvendor formula.
    
    Assumes Normal demand: q* = μ + σ * Φ^(-1)(critical_ratio)
    
    Parameters
    ----------
    point_pred : np.ndarray
        Point predictions (mean).
    sigma : np.ndarray
        Standard deviation of demand.
    critical_ratio : float
        Critical ratio cu / (cu + co).
        
    Returns
    -------
    np.ndarray
        Order quantities.
    """
    from scipy.stats import norm
    z = norm.ppf(critical_ratio)
    return np.maximum(0, point_pred + sigma * z)


def compute_expected_cost(
    order_quantities: np.ndarray,
    demand_samples: np.ndarray,
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0
) -> Tuple[float, float, float]:
    """
    Compute expected cost and CVaR metrics.
    
    Parameters
    ----------
    order_quantities : np.ndarray
        Order quantities for each period.
    demand_samples : np.ndarray
        Actual demand for each period.
    ordering_cost, holding_cost, stockout_cost : float
        Cost parameters.
        
    Returns
    -------
    Tuple[float, float, float]
        Mean cost, CVaR-90, CVaR-95.
    """
    costs = newsvendor_loss(
        order_quantities, demand_samples,
        ordering_cost, holding_cost, stockout_cost
    )
    
    mean_cost = np.mean(costs)
    
    # CVaR computation
    sorted_costs = np.sort(costs)
    n = len(costs)
    cvar_90_idx = int(np.ceil(0.90 * n))
    cvar_95_idx = int(np.ceil(0.95 * n))
    
    cvar_90 = np.mean(sorted_costs[cvar_90_idx:])
    cvar_95 = np.mean(sorted_costs[cvar_95_idx:])
    
    return mean_cost, cvar_90, cvar_95
