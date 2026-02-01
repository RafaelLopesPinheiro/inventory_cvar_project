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
    verbose: bool = True
) -> np.ndarray:
    """
    Compute Wasserstein DRO-optimal order quantities for all predictions.

    Samples demand scenarios from prediction intervals, then optimizes
    using distributionally robust optimization with Wasserstein ambiguity.

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

    Returns
    -------
    np.ndarray
        Optimal robust order quantities.
    """
    if verbose:
        logger.info(f"Optimizing Wasserstein DRO (epsilon={epsilon}, beta={beta}) for {len(point_pred)} days...")

    rng = np.random.RandomState(random_seed)
    order_quantities = []

    for i in range(len(point_pred)):
        if verbose and (i + 1) % 50 == 0:
            logger.info(f"Processed {i + 1}/{len(point_pred)} days...")

        # Sample demand scenarios from prediction interval
        demand_samples = rng.uniform(lower[i], upper[i], n_samples)

        # Optimize using Wasserstein DRO
        q_opt = optimize_wasserstein_dro_single(
            demand_samples, epsilon, beta,
            ordering_cost, holding_cost, stockout_cost
        )
        order_quantities.append(q_opt)

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
