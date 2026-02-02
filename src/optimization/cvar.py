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

    # Use ThreadPoolExecutor (faster for I/O bound scipy.optimize)
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
    ball centered at the empirical distribution.

    The formulation is:
        min_q max_{P: W(P, P̂) ≤ ε} CVaR_β(Loss(q, D))

    For the newsvendor problem with Wasserstein ambiguity, this can be reformulated as:
        min_{q, λ≥0} λε + (1/N) Σ sup_{d} [ℓ_β(q, d) - λ|d - d̂_i|]

    where ℓ_β is the CVaR-transformed loss.

    For tractability, we use a conservative approximation that adds a robustness
    margin based on the Wasserstein radius and the Lipschitz constant of the loss.

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
    n_samples = len(demand_samples)

    # Lipschitz constant of the newsvendor loss with respect to demand
    # L(q, d) = c_o*q + c_h*max(0, q-d) + c_u*max(0, d-q)
    # |∂L/∂d| ≤ max(c_h, c_u) = c_u (since c_u > c_h typically)
    lipschitz_constant = max(holding_cost, stockout_cost)

    def dro_objective(x: np.ndarray) -> float:
        """
        Wasserstein DRO objective with CVaR.

        Uses the dual formulation:
        min_{q, λ, τ} λε + τ + (1/(N(1-β))) Σ max(0, sup_d[Loss(q,d) - λ|d-d̂_i|] - τ)
        """
        q, lam, tau = x

        if lam < 0:
            return 1e10  # Penalty for negative lambda

        # For each empirical sample, compute the worst-case contribution
        # The supremum over d of [Loss(q,d) - λ|d-d̂_i|] depends on the structure
        # of the loss function.

        # For newsvendor loss, the worst case can be computed analytically:
        # If λ ≥ c_u: worst case is at d = d̂_i (no adversarial shift)
        # If λ < c_u: adversary can shift demand to maximize loss

        worst_case_losses = np.zeros(n_samples)

        for i, d_hat in enumerate(demand_samples):
            if lam >= lipschitz_constant:
                # Adversary is too expensive, use empirical sample
                worst_case_losses[i] = newsvendor_loss(
                    q, np.array([d_hat]),
                    ordering_cost, holding_cost, stockout_cost
                )[0]
            else:
                # Adversary shifts demand to worst direction
                # For q > d̂: adversary decreases demand (more overage)
                # For q < d̂: adversary increases demand (more stockout)

                # Compute loss at empirical point
                base_loss = newsvendor_loss(
                    q, np.array([d_hat]),
                    ordering_cost, holding_cost, stockout_cost
                )[0]

                # Add worst-case margin based on effective adversarial power
                # The adversary can gain (c_u - λ) per unit of demand shift for stockouts
                # or (c_h - λ) per unit for overage (if λ < c_h)

                if q <= d_hat:
                    # Currently in stockout region, adversary increases demand
                    marginal_gain = stockout_cost - lam
                else:
                    # Currently in overage region, adversary decreases demand
                    marginal_gain = holding_cost - lam if lam < holding_cost else 0

                # Worst-case adds margin proportional to epsilon and marginal gain
                # (bounded by how much the adversary can shift within epsilon budget)
                worst_case_losses[i] = base_loss + max(0, marginal_gain) * epsilon

        # CVaR computation over worst-case losses
        cvar_term = tau + (1 / (n_samples * (1 - beta))) * np.sum(
            np.maximum(0, worst_case_losses - tau)
        )

        # Add Wasserstein penalty term
        return lam * epsilon + cvar_term

    # Initial guess
    q0 = np.mean(demand_samples)
    lam0 = lipschitz_constant  # Start at Lipschitz constant
    tau0 = np.median(newsvendor_loss(q0, demand_samples, ordering_cost, holding_cost, stockout_cost))

    # Bounds: q >= 0, lambda >= 0, tau unrestricted
    bounds = [(0, None), (0, None), (None, None)]

    result = minimize(
        dro_objective,
        [q0, lam0, tau0],
        method='L-BFGS-B',
        bounds=bounds
    )

    return max(0, result.x[0])


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
        # Optimize single order quantity for all horizons
        def cvar_objective(x: np.ndarray) -> float:
            q, tau = x
            q_arr = np.full(n_scenarios, q)

            # Create demand dict with repeated values for vectorization
            d_scenarios = {h: demand_samples[h] for h in horizons}

            losses = multi_period_newsvendor_loss(
                q_arr, d_scenarios, horizons,
                ordering_cost, holding_cost, stockout_cost,
                aggregation
            )

            cvar_term = tau + (1 / (n_scenarios * (1 - beta))) * np.sum(
                np.maximum(0, losses - tau)
            )
            return cvar_term

        # Initial guess based on mean demand across horizons
        mean_demands = [np.mean(demand_samples[h]) for h in horizons]
        q0 = np.mean(mean_demands)
        tau0 = 0.0

        bounds = [(0, None), (None, None)]

        result = minimize(
            cvar_objective,
            [q0, tau0],
            method='L-BFGS-B',
            bounds=bounds
        )

        return max(0, result.x[0])

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
