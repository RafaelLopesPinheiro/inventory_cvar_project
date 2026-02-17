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
import pulp
from typing import Tuple, Optional
from dataclasses import dataclass
import logging
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
import multiprocessing
import os

logger = logging.getLogger(__name__)

# Determine optimal number of workers
_NUM_WORKERS = min(multiprocessing.cpu_count(), 8)


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
    Optimize order quantity using CVaR via Rockafellar-Uryasev LP formulation.

    Reformulates the CVaR minimization as a Linear Program (LP):

        min_{q, τ, h_i, u_i, z_i}  τ + (1 / (N * (1 - β))) * Σ z_i

        s.t.  h_i >= q - d_i          (overage linearization)
              u_i >= d_i - q          (underage linearization)
              z_i >= c_o*q + c_h*h_i + c_u*u_i - τ  (CVaR slack)
              h_i, u_i, z_i >= 0
              q >= 0

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
    n = len(demand_samples)
    c_o, c_h, c_u = ordering_cost, holding_cost, stockout_cost

    prob = pulp.LpProblem("CVaR_Newsvendor", pulp.LpMinimize)

    q = pulp.LpVariable("q", lowBound=0)
    tau = pulp.LpVariable("tau")

    # Linearization variables for each demand scenario
    h = [pulp.LpVariable(f"h_{i}", lowBound=0) for i in range(n)]  # overage
    u = [pulp.LpVariable(f"u_{i}", lowBound=0) for i in range(n)]  # underage
    z = [pulp.LpVariable(f"z_{i}", lowBound=0) for i in range(n)]  # CVaR slack

    # Objective: τ + (1 / (N * (1 - β))) * Σ z_i
    prob += tau + (1.0 / (n * (1.0 - beta))) * pulp.lpSum(z)

    # Constraints for each demand scenario
    for i in range(n):
        d_i = float(demand_samples[i])
        prob += h[i] >= q - d_i                                      # h_i >= q - d_i
        prob += u[i] >= d_i - q                                      # u_i >= d_i - q
        prob += z[i] >= c_o * q + c_h * h[i] + c_u * u[i] - tau    # z_i >= Loss_i - τ

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    q_val = pulp.value(q)
    return max(0.0, q_val if q_val is not None else 0.0)


def _optimize_cvar_single_worker(args):
    """
    Worker function for parallel CVaR optimization.

    This function is designed to be pickle-able for multiprocessing.
    """
    lower_i, upper_i, n_samples, seed, beta, ordering_cost, holding_cost, stockout_cost = args
    rng = np.random.RandomState(seed)
    demand_samples = rng.uniform(lower_i, upper_i, n_samples)
    return optimize_cvar_single(demand_samples, beta, ordering_cost, holding_cost, stockout_cost)


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
    verbose: bool = True,
    parallel: bool = True,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Compute CVaR-optimal order quantities for all predictions.

    Samples demand scenarios uniformly from prediction intervals and
    optimizes CVaR for each day. Uses parallel processing for speedup.

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
    parallel : bool
        Whether to use parallel processing. Default True for speedup.
    n_jobs : int
        Number of parallel jobs. -1 uses all available cores.

    Returns
    -------
    np.ndarray
        Optimal order quantities.
    """
    n_days = len(point_pred)

    if verbose:
        logger.info(f"Optimizing CVaR (beta={beta}) for {n_days} days...")

    # Determine number of workers
    if n_jobs == -1:
        n_workers = _NUM_WORKERS
    else:
        n_workers = min(n_jobs, _NUM_WORKERS)

    # For small number of days, sequential is faster (avoid overhead)
    if not parallel or n_days < 10:
        rng = np.random.RandomState(random_seed)
        order_quantities = []

        for i in range(n_days):
            demand_samples = rng.uniform(lower[i], upper[i], n_samples)
            q_opt = optimize_cvar_single(
                demand_samples, beta,
                ordering_cost, holding_cost, stockout_cost
            )
            order_quantities.append(q_opt)

        return np.array(order_quantities)

    # Parallel processing for larger problems
    # Prepare arguments for each day with unique seeds
    args_list = [
        (lower[i], upper[i], n_samples, random_seed + i, beta,
         ordering_cost, holding_cost, stockout_cost)
        for i in range(n_days)
    ]

    # Use ThreadPoolExecutor for parallel LP solves
    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        order_quantities = list(executor.map(_optimize_cvar_single_worker, args_list))

    if verbose:
        logger.info(f"Completed CVaR optimization for {n_days} days using {n_workers} workers")

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


def optimize_wasserstein_dro_single(
    demand_samples: np.ndarray,
    epsilon: float = 0.1,
    beta: float = 0.90,
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0
) -> float:
    """
    Optimize order quantity using Wasserstein Distributionally Robust Optimization.

    Solves the worst-case CVaR over all distributions within an epsilon-Wasserstein
    ball centered at the empirical distribution, via a Linear Program (LP).

    The formulation is:
        min_q max_{P: W(P, P̂) ≤ ε} CVaR_β(Loss(q, D))

    For the newsvendor problem with Wasserstein ambiguity, this is solved as:

        min_{q, λ, τ, r_u, r_h, wc_i, z_i}  λε + τ + (1/(N(1-β))) Σ z_i

        s.t.  r_u >= c_u - λ                               (adversarial gain, stockout side)
              r_h >= c_h - λ                               (adversarial gain, overage side)
              wc_i >= (c_o - c_u)*q + c_u*d̂_i + ε*r_u    (worst-case loss, stockout)
              wc_i >= (c_o + c_h)*q - c_h*d̂_i + ε*r_h    (worst-case loss, overage)
              z_i  >= wc_i - τ                             (CVaR slack)
              q, λ, r_u, r_h, wc_i, z_i >= 0

    Parameters
    ----------
    demand_samples : np.ndarray
        Samples from the empirical demand distribution.
    epsilon : float
        Wasserstein ball radius (controls robustness level).
        Larger epsilon = more robust but more conservative.
    beta : float
        CVaR level (tail probability).
    ordering_cost, holding_cost, stockout_cost : float
        Cost parameters.

    Returns
    -------
    float
        Optimal robust order quantity.

    References
    ----------
    - Mohajerin Esfahani & Kuhn (2018) "Data-driven distributionally robust
      optimization using the Wasserstein metric"
    - Blanchet & Murthy (2019) "Quantifying distributional model risk via
      optimal transport"
    - Gao & Kleywegt (2022) "Distributionally Robust Stochastic Optimization
      with Wasserstein Distance"
    """
    n = len(demand_samples)
    c_o, c_h, c_u = ordering_cost, holding_cost, stockout_cost
    lipschitz_constant = max(c_h, c_u)

    prob = pulp.LpProblem("DRO_Newsvendor", pulp.LpMinimize)

    q = pulp.LpVariable("q", lowBound=0)
    lam = pulp.LpVariable("lam", lowBound=0, upBound=lipschitz_constant)
    tau = pulp.LpVariable("tau")

    # r_u = max(0, c_u - λ): adversarial gain on stockout side
    # r_h = max(0, c_h - λ): adversarial gain on overage side
    r_u = pulp.LpVariable("r_u", lowBound=0)
    r_h = pulp.LpVariable("r_h", lowBound=0)

    # Per-scenario worst-case loss and CVaR slack
    wc = [pulp.LpVariable(f"wc_{i}", lowBound=0) for i in range(n)]
    z = [pulp.LpVariable(f"z_{i}", lowBound=0) for i in range(n)]

    # Objective: λε + τ + (1/(N*(1-β))) * Σ z_i
    prob += lam * epsilon + tau + (1.0 / (n * (1.0 - beta))) * pulp.lpSum(z)

    # Adversarial gain linearization
    prob += r_u >= c_u - lam   # r_u >= c_u - λ
    prob += r_h >= c_h - lam   # r_h >= c_h - λ

    for i in range(n):
        d_hat = float(demand_samples[i])
        # Worst-case loss from stockout side (adversary increases demand)
        prob += wc[i] >= (c_o - c_u) * q + c_u * d_hat + epsilon * r_u
        # Worst-case loss from overage side (adversary decreases demand)
        prob += wc[i] >= (c_o + c_h) * q - c_h * d_hat + epsilon * r_h
        # CVaR slack: z_i >= wc_i - τ
        prob += z[i] >= wc[i] - tau

    prob.solve(pulp.PULP_CBC_CMD(msg=0))

    q_val = pulp.value(q)
    return max(0.0, q_val if q_val is not None else 0.0)


def _optimize_dro_single_worker(args):
    """
    Worker function for parallel DRO optimization.

    This function is designed to be pickle-able for multiprocessing.
    """
    lower_i, upper_i, n_samples, seed, epsilon, beta, ordering_cost, holding_cost, stockout_cost = args
    rng = np.random.RandomState(seed)
    demand_samples = rng.uniform(lower_i, upper_i, n_samples)
    return optimize_wasserstein_dro_single(demand_samples, epsilon, beta, ordering_cost, holding_cost, stockout_cost)


def compute_order_quantities_dro(
    point_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    epsilon: float = 0.1,
    beta: float = 0.90,
    n_samples: int = 1000,
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0,
    random_seed: int = 42,
    verbose: bool = True,
    parallel: bool = True,
    n_jobs: int = -1
) -> np.ndarray:
    """
    Compute Wasserstein DRO-optimal order quantities for all predictions.

    Samples demand scenarios from prediction intervals, then optimizes
    using distributionally robust optimization with Wasserstein ambiguity.
    Uses parallel processing for speedup.

    Parameters
    ----------
    point_pred : np.ndarray
        Point predictions.
    lower : np.ndarray
        Lower bounds of prediction intervals.
    upper : np.ndarray
        Upper bounds of prediction intervals.
    epsilon : float
        Wasserstein ball radius for DRO.
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
    parallel : bool
        Whether to use parallel processing. Default True for speedup.
    n_jobs : int
        Number of parallel jobs. -1 uses all available cores.

    Returns
    -------
    np.ndarray
        Optimal robust order quantities.
    """
    n_days = len(point_pred)

    if verbose:
        logger.info(f"Optimizing Wasserstein DRO (epsilon={epsilon}, beta={beta}) for {n_days} days...")

    # Determine number of workers
    if n_jobs == -1:
        n_workers = _NUM_WORKERS
    else:
        n_workers = min(n_jobs, _NUM_WORKERS)

    # For small number of days, sequential is faster (avoid overhead)
    if not parallel or n_days < 10:
        rng = np.random.RandomState(random_seed)
        order_quantities = []

        for i in range(n_days):
            demand_samples = rng.uniform(lower[i], upper[i], n_samples)
            q_opt = optimize_wasserstein_dro_single(
                demand_samples, epsilon, beta,
                ordering_cost, holding_cost, stockout_cost
            )
            order_quantities.append(q_opt)

        return np.array(order_quantities)

    # Parallel processing for larger problems
    args_list = [
        (lower[i], upper[i], n_samples, random_seed + i, epsilon, beta,
         ordering_cost, holding_cost, stockout_cost)
        for i in range(n_days)
    ]

    with ThreadPoolExecutor(max_workers=n_workers) as executor:
        order_quantities = list(executor.map(_optimize_dro_single_worker, args_list))

    if verbose:
        logger.info(f"Completed DRO optimization for {n_days} days using {n_workers} workers")

    return np.array(order_quantities)


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


# =============================================================================
# MULTI-PERIOD CVaR OPTIMIZATION
# =============================================================================

from typing import Dict, List


def multi_period_newsvendor_loss(
    q: np.ndarray,
    d: Dict[int, np.ndarray],
    horizons: List[int],
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0,
    aggregation: str = "mean"
) -> np.ndarray:
    """
    Compute multi-period newsvendor loss aggregated across horizons.

    For each sample, computes the loss at each horizon and then aggregates
    using the specified method.

    Parameters
    ----------
    q : np.ndarray
        Order quantities of shape (n_samples,) or (n_samples, n_horizons).
    d : Dict[int, np.ndarray]
        Actual demand for each horizon. Keys are horizons, values are
        demand arrays of shape (n_scenarios,) for a single sample or
        (n_samples,) for multiple samples.
    horizons : List[int]
        List of forecast horizons.
    ordering_cost, holding_cost, stockout_cost : float
        Cost parameters.
    aggregation : str
        How to aggregate across horizons: "mean", "sum", "worst_case".

    Returns
    -------
    np.ndarray
        Aggregated loss values.
    """
    horizon_losses = []

    for h in horizons:
        if q.ndim == 1:
            # Same order quantity for all horizons
            q_h = q
        else:
            # Different order quantity per horizon
            h_idx = horizons.index(h)
            q_h = q[:, h_idx]

        d_h = d[h]
        loss_h = newsvendor_loss(q_h, d_h, ordering_cost, holding_cost, stockout_cost)
        horizon_losses.append(loss_h)

    horizon_losses = np.array(horizon_losses)  # (n_horizons, n_samples) or (n_horizons, n_scenarios)

    if aggregation == "mean":
        return np.mean(horizon_losses, axis=0)
    elif aggregation == "sum":
        return np.sum(horizon_losses, axis=0)
    elif aggregation == "worst_case":
        return np.max(horizon_losses, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def optimize_multi_period_cvar_single(
    demand_samples: Dict[int, np.ndarray],
    horizons: List[int],
    beta: float = 0.90,
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0,
    aggregation: str = "mean",
    joint_optimization: bool = True
) -> np.ndarray:
    """
    Optimize order quantities using multi-period CVaR.

    For a single forecast origin, optimizes order quantities to minimize
    the CVaR of aggregated loss across multiple forecast horizons.

    Parameters
    ----------
    demand_samples : Dict[int, np.ndarray]
        Dictionary mapping each horizon to its demand samples.
        Each array has shape (n_scenarios,).
    horizons : List[int]
        List of forecast horizons.
    beta : float
        CVaR level (tail probability).
    ordering_cost, holding_cost, stockout_cost : float
        Cost parameters.
    aggregation : str
        How to aggregate losses: "mean", "sum", "worst_case".
    joint_optimization : bool
        If True, optimizes a single order quantity for all horizons.
        If False, optimizes separately for each horizon.

    Returns
    -------
    np.ndarray
        Optimal order quantities. If joint_optimization=True, returns scalar.
        Otherwise returns array of shape (n_horizons,).
    """
    n_scenarios = len(demand_samples[horizons[0]])

    if joint_optimization:
        # Optimize single order quantity for all horizons via LP
        # Variables: q, τ, h_{i,h}, u_{i,h} (per scenario per horizon), z_i (per scenario)
        n_horizons = len(horizons)
        c_o, c_h, c_u = ordering_cost, holding_cost, stockout_cost

        prob = pulp.LpProblem("MultiPeriod_CVaR_Newsvendor", pulp.LpMinimize)

        q = pulp.LpVariable("q", lowBound=0)
        tau = pulp.LpVariable("tau")
        z = [pulp.LpVariable(f"z_{i}", lowBound=0) for i in range(n_scenarios)]

        # Per-scenario, per-horizon linearization variables
        h = {(i, h_idx): pulp.LpVariable(f"h_{i}_{h_idx}", lowBound=0)
             for i in range(n_scenarios) for h_idx in range(n_horizons)}
        u = {(i, h_idx): pulp.LpVariable(f"u_{i}_{h_idx}", lowBound=0)
             for i in range(n_scenarios) for h_idx in range(n_horizons)}

        # Objective: τ + (1/(N*(1-β))) * Σ z_i
        prob += tau + (1.0 / (n_scenarios * (1.0 - beta))) * pulp.lpSum(z)

        for i in range(n_scenarios):
            if aggregation == "worst_case":
                # Need per-horizon loss variables for worst-case
                w = [pulp.LpVariable(f"w_{i}_{h_idx}", lowBound=0)
                     for h_idx in range(n_horizons)]
                for h_idx, horizon in enumerate(horizons):
                    d_ih = float(demand_samples[horizon][i])
                    prob += h[i, h_idx] >= q - d_ih
                    prob += u[i, h_idx] >= d_ih - q
                    # w_{i,h} >= L(q, d_{i,h})
                    prob += w[h_idx] >= c_o * q + c_h * h[i, h_idx] + c_u * u[i, h_idx]
                # z_i >= max_h L_ih - τ  (i.e. z_i >= w_{i,h} - τ for all h)
                for h_idx in range(n_horizons):
                    prob += z[i] >= w[h_idx] - tau
            else:
                # mean or sum aggregation: build aggregated loss expression
                agg_loss_expr = pulp.lpSum(
                    c_o * q + c_h * h[i, h_idx] + c_u * u[i, h_idx]
                    for h_idx in range(n_horizons)
                )
                if aggregation == "mean":
                    agg_loss_expr = agg_loss_expr / n_horizons

                for h_idx, horizon in enumerate(horizons):
                    d_ih = float(demand_samples[horizon][i])
                    prob += h[i, h_idx] >= q - d_ih
                    prob += u[i, h_idx] >= d_ih - q

                prob += z[i] >= agg_loss_expr - tau

        prob.solve(pulp.PULP_CBC_CMD(msg=0))

        q_val = pulp.value(q)
        return max(0.0, q_val if q_val is not None else 0.0)

    else:
        # Optimize separately for each horizon
        optimal_quantities = []

        for h in horizons:
            q_opt = optimize_cvar_single(
                demand_samples[h], beta,
                ordering_cost, holding_cost, stockout_cost
            )
            optimal_quantities.append(q_opt)

        return np.array(optimal_quantities)


def _optimize_multi_period_joint_worker(args):
    """Worker for parallel multi-period joint optimization."""
    (lower_dict, upper_dict, horizons, n_samples, seed, beta,
     ordering_cost, holding_cost, stockout_cost, aggregation) = args
    rng = np.random.RandomState(seed)

    demand_scenarios = {}
    for h in horizons:
        demand_scenarios[h] = rng.uniform(lower_dict[h], upper_dict[h], n_samples)

    return optimize_multi_period_cvar_single(
        demand_scenarios, horizons, beta,
        ordering_cost, holding_cost, stockout_cost,
        aggregation, joint_optimization=True
    )


def _optimize_multi_period_separate_worker(args):
    """Worker for parallel multi-period separate optimization."""
    (lower_dict, upper_dict, horizons, n_samples, seed, beta,
     ordering_cost, holding_cost, stockout_cost) = args
    rng = np.random.RandomState(seed)

    results = {}
    for h in horizons:
        demand_samples = rng.uniform(lower_dict[h], upper_dict[h], n_samples)
        results[h] = optimize_cvar_single(
            demand_samples, beta, ordering_cost, holding_cost, stockout_cost
        )
    return results


def compute_order_quantities_multi_period_cvar(
    point_pred: Dict[int, np.ndarray],
    lower: Dict[int, np.ndarray],
    upper: Dict[int, np.ndarray],
    horizons: List[int],
    beta: float = 0.90,
    n_samples: int = 1000,
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0,
    aggregation: str = "mean",
    joint_optimization: bool = True,
    random_seed: int = 42,
    verbose: bool = True,
    parallel: bool = True,
    n_jobs: int = -1
) -> Dict[int, np.ndarray]:
    """
    Compute CVaR-optimal order quantities for multi-period predictions.

    For each forecast origin, samples demand scenarios from prediction
    intervals at each horizon, then optimizes considering all horizons jointly.
    Uses parallel processing for speedup.

    Parameters
    ----------
    point_pred : Dict[int, np.ndarray]
        Point predictions for each horizon.
    lower : Dict[int, np.ndarray]
        Lower bounds of prediction intervals for each horizon.
    upper : Dict[int, np.ndarray]
        Upper bounds of prediction intervals for each horizon.
    horizons : List[int]
        List of forecast horizons.
    beta : float
        CVaR level.
    n_samples : int
        Number of demand scenarios to generate per horizon.
    ordering_cost, holding_cost, stockout_cost : float
        Cost parameters.
    aggregation : str
        How to aggregate losses across horizons.
    joint_optimization : bool
        Whether to optimize jointly or separately per horizon.
    random_seed : int
        Random seed for reproducibility.
    verbose : bool
        Whether to print progress.
    parallel : bool
        Whether to use parallel processing. Default True.
    n_jobs : int
        Number of parallel jobs. -1 uses all available cores.

    Returns
    -------
    Dict[int, np.ndarray]
        Optimal order quantities for each horizon.
        If joint_optimization=True, all horizons get the same quantities.
    """
    n_days = len(point_pred[horizons[0]])

    if verbose:
        logger.info(f"Optimizing multi-period CVaR (beta={beta}) for {n_days} days...")
        logger.info(f"  Horizons: {horizons}")
        logger.info(f"  Aggregation: {aggregation}")
        logger.info(f"  Joint optimization: {joint_optimization}")

    # Determine number of workers
    if n_jobs == -1:
        n_workers = _NUM_WORKERS
    else:
        n_workers = min(n_jobs, _NUM_WORKERS)

    # For small number of days, sequential is faster
    if not parallel or n_days < 10:
        rng = np.random.RandomState(random_seed)

        if joint_optimization:
            joint_orders = []
            for i in range(n_days):
                demand_scenarios = {}
                for h in horizons:
                    demand_scenarios[h] = rng.uniform(lower[h][i], upper[h][i], n_samples)
                q_opt = optimize_multi_period_cvar_single(
                    demand_scenarios, horizons, beta,
                    ordering_cost, holding_cost, stockout_cost,
                    aggregation, joint_optimization=True
                )
                joint_orders.append(q_opt)
            joint_orders = np.array(joint_orders)
            return {h: joint_orders for h in horizons}
        else:
            horizon_orders = {h: [] for h in horizons}
            for i in range(n_days):
                for h in horizons:
                    demand_samples = rng.uniform(lower[h][i], upper[h][i], n_samples)
                    q_opt = optimize_cvar_single(
                        demand_samples, beta, ordering_cost, holding_cost, stockout_cost
                    )
                    horizon_orders[h].append(q_opt)
            return {h: np.array(horizon_orders[h]) for h in horizons}

    # Parallel processing
    if joint_optimization:
        args_list = [
            ({h: lower[h][i] for h in horizons},
             {h: upper[h][i] for h in horizons},
             horizons, n_samples, random_seed + i, beta,
             ordering_cost, holding_cost, stockout_cost, aggregation)
            for i in range(n_days)
        ]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            joint_orders = list(executor.map(_optimize_multi_period_joint_worker, args_list))

        if verbose:
            logger.info(f"Completed multi-period optimization for {n_days} days using {n_workers} workers")

        joint_orders = np.array(joint_orders)
        return {h: joint_orders for h in horizons}

    else:
        args_list = [
            ({h: lower[h][i] for h in horizons},
             {h: upper[h][i] for h in horizons},
             horizons, n_samples, random_seed + i, beta,
             ordering_cost, holding_cost, stockout_cost)
            for i in range(n_days)
        ]

        with ThreadPoolExecutor(max_workers=n_workers) as executor:
            results_list = list(executor.map(_optimize_multi_period_separate_worker, args_list))

        if verbose:
            logger.info(f"Completed multi-period optimization for {n_days} days using {n_workers} workers")

        # Reorganize results by horizon
        horizon_orders = {h: [] for h in horizons}
        for result in results_list:
            for h in horizons:
                horizon_orders[h].append(result[h])

        return {h: np.array(horizon_orders[h]) for h in horizons}


@dataclass
class InventorySimulationResult:
    """Results from inventory simulation with carryover and capacity constraints."""
    actual_orders: np.ndarray  # Adjusted order quantities (after capacity constraints)
    inventory_levels: np.ndarray  # Inventory level at start of each period (before demand)
    carryover_inventory: np.ndarray  # Inventory carried over from previous period
    costs: np.ndarray  # Total cost per period
    ordering_costs: np.ndarray  # Ordering cost component per period
    holding_costs: np.ndarray  # Holding cost component per period
    stockout_costs: np.ndarray  # Stockout cost component per period
    demands: np.ndarray  # Actual demands
    capacity_utilization: np.ndarray  # Fraction of capacity used each period

    @property
    def total_cost(self) -> float:
        return float(np.sum(self.costs))

    @property
    def mean_cost(self) -> float:
        return float(np.mean(self.costs))

    @property
    def cvar_90(self) -> float:
        sorted_costs = np.sort(self.costs)
        idx = int(np.ceil(0.90 * len(sorted_costs)))
        return float(np.mean(sorted_costs[idx:])) if idx < len(sorted_costs) else float(sorted_costs[-1])

    @property
    def cvar_95(self) -> float:
        sorted_costs = np.sort(self.costs)
        idx = int(np.ceil(0.95 * len(sorted_costs)))
        return float(np.mean(sorted_costs[idx:])) if idx < len(sorted_costs) else float(sorted_costs[-1])

    @property
    def service_level(self) -> float:
        return float(np.mean(self.inventory_levels >= self.demands))

    @property
    def avg_capacity_utilization(self) -> float:
        return float(np.mean(self.capacity_utilization))

    @property
    def avg_carryover(self) -> float:
        return float(np.mean(self.carryover_inventory))


def simulate_inventory_with_carryover(
    target_order_quantities: np.ndarray,
    actual_demands: np.ndarray,
    initial_inventory: float = 0.0,
    carryover_rate: float = 0.95,
    capacity: float = 200.0,
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0
) -> InventorySimulationResult:
    """
    Simulate inventory dynamics with carryover and capacity constraints.

    At each period t:
    1. Start with carryover inventory I_t from previous period
    2. Compute actual order: q_t = min(target_q_t, capacity - I_t)
    3. Available inventory: A_t = I_t + q_t
    4. Demand d_t arrives
    5. Overage: max(0, A_t - d_t), Underage: max(0, d_t - A_t)
    6. Cost: ordering_cost * q_t + holding_cost * overage + stockout_cost * underage
    7. Carryover to next period: I_{t+1} = carryover_rate * overage

    Parameters
    ----------
    target_order_quantities : np.ndarray
        Desired order quantities from the optimization (before constraints).
    actual_demands : np.ndarray
        Actual realized demand for each period.
    initial_inventory : float
        Starting inventory level.
    carryover_rate : float
        Fraction of leftover inventory that carries to next period (0 to 1).
        0 = no carryover (standard newsvendor), 1 = full carryover.
    capacity : float
        Maximum warehouse storage capacity.
    ordering_cost : float
        Cost per unit ordered.
    holding_cost : float
        Cost per unit of overage.
    stockout_cost : float
        Cost per unit of underage.

    Returns
    -------
    InventorySimulationResult
        Detailed simulation results including costs, inventory levels, etc.
    """
    n_days = len(target_order_quantities)
    inventory = initial_inventory

    actual_orders = np.zeros(n_days)
    inventory_levels = np.zeros(n_days)
    carryover_inv = np.zeros(n_days)
    costs = np.zeros(n_days)
    ord_costs = np.zeros(n_days)
    hold_costs = np.zeros(n_days)
    stock_costs = np.zeros(n_days)
    cap_util = np.zeros(n_days)

    for t in range(n_days):
        carryover_inv[t] = inventory

        # Constrain order by capacity (can't exceed warehouse limit)
        max_order = max(0, capacity - inventory)
        actual_order = max(0, min(target_order_quantities[t], max_order))

        # Available inventory after ordering
        available = inventory + actual_order
        inventory_levels[t] = available

        # Capacity utilization
        cap_util[t] = available / capacity if capacity > 0 else 0.0

        # Demand realization
        demand = actual_demands[t]

        # Cost components
        overage = max(0, available - demand)
        underage = max(0, demand - available)

        oc = ordering_cost * actual_order
        hc = holding_cost * overage
        sc = stockout_cost * underage

        actual_orders[t] = actual_order
        ord_costs[t] = oc
        hold_costs[t] = hc
        stock_costs[t] = sc
        costs[t] = oc + hc + sc

        # Carryover: leftover inventory carries to next period (possibly degraded)
        inventory = overage * carryover_rate

    return InventorySimulationResult(
        actual_orders=actual_orders,
        inventory_levels=inventory_levels,
        carryover_inventory=carryover_inv,
        costs=costs,
        ordering_costs=ord_costs,
        holding_costs=hold_costs,
        stockout_costs=stock_costs,
        demands=actual_demands.copy(),
        capacity_utilization=cap_util
    )


@dataclass
class MultiPeriodCostMetrics:
    """Container for multi-period cost metrics."""
    # Per-horizon metrics
    horizon_mean_costs: Dict[int, float]
    horizon_cvar_90: Dict[int, float]
    horizon_cvar_95: Dict[int, float]
    horizon_service_levels: Dict[int, float]

    # Aggregated metrics
    aggregated_mean_cost: float
    aggregated_cvar_90: float
    aggregated_cvar_95: float
    aggregated_service_level: float

    # Horizon list for reference
    horizons: List[int]


def compute_multi_period_metrics(
    order_quantities: Dict[int, np.ndarray],
    actual_demand: Dict[int, np.ndarray],
    horizons: List[int],
    ordering_cost: float = 10.0,
    holding_cost: float = 2.0,
    stockout_cost: float = 50.0,
    aggregation: str = "mean"
) -> MultiPeriodCostMetrics:
    """
    Compute cost metrics for multi-period forecasts.

    Calculates metrics both per-horizon and aggregated across all horizons.

    Parameters
    ----------
    order_quantities : Dict[int, np.ndarray]
        Order quantities for each horizon.
    actual_demand : Dict[int, np.ndarray]
        Actual demand for each horizon.
    horizons : List[int]
        List of forecast horizons.
    ordering_cost, holding_cost, stockout_cost : float
        Cost parameters.
    aggregation : str
        How to aggregate across horizons for overall metrics.

    Returns
    -------
    MultiPeriodCostMetrics
        Container with all computed metrics.
    """
    # Per-horizon metrics
    horizon_mean_costs = {}
    horizon_cvar_90 = {}
    horizon_cvar_95 = {}
    horizon_service_levels = {}

    all_costs = []

    for h in horizons:
        costs = newsvendor_loss(
            order_quantities[h], actual_demand[h],
            ordering_cost, holding_cost, stockout_cost
        )
        all_costs.append(costs)

        # Mean cost
        horizon_mean_costs[h] = np.mean(costs)

        # CVaR metrics
        sorted_costs = np.sort(costs)
        n = len(costs)
        cvar_90_idx = int(np.ceil(0.90 * n))
        cvar_95_idx = int(np.ceil(0.95 * n))

        horizon_cvar_90[h] = np.mean(sorted_costs[cvar_90_idx:])
        horizon_cvar_95[h] = np.mean(sorted_costs[cvar_95_idx:])

        # Service level (fill rate)
        service_level = np.mean(order_quantities[h] >= actual_demand[h])
        horizon_service_levels[h] = service_level

    # Aggregated metrics
    all_costs = np.array(all_costs)  # (n_horizons, n_samples)

    if aggregation == "mean":
        aggregated_costs = np.mean(all_costs, axis=0)
    elif aggregation == "sum":
        aggregated_costs = np.sum(all_costs, axis=0)
    elif aggregation == "worst_case":
        aggregated_costs = np.max(all_costs, axis=0)
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")

    aggregated_mean_cost = np.mean(aggregated_costs)

    sorted_agg_costs = np.sort(aggregated_costs)
    n = len(aggregated_costs)
    cvar_90_idx = int(np.ceil(0.90 * n))
    cvar_95_idx = int(np.ceil(0.95 * n))

    aggregated_cvar_90 = np.mean(sorted_agg_costs[cvar_90_idx:])
    aggregated_cvar_95 = np.mean(sorted_agg_costs[cvar_95_idx:])

    # Aggregated service level (average across horizons)
    aggregated_service_level = np.mean(list(horizon_service_levels.values()))

    return MultiPeriodCostMetrics(
        horizon_mean_costs=horizon_mean_costs,
        horizon_cvar_90=horizon_cvar_90,
        horizon_cvar_95=horizon_cvar_95,
        horizon_service_levels=horizon_service_levels,
        aggregated_mean_cost=aggregated_mean_cost,
        aggregated_cvar_90=aggregated_cvar_90,
        aggregated_cvar_95=aggregated_cvar_95,
        aggregated_service_level=aggregated_service_level,
        horizons=horizons
    )
