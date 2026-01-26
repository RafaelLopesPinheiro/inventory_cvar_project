# Ensemble Batch Prediction Intervals + CQR + CVaR Optimization

## Overview

This document describes the **EnbPI+CQR+CVaR** method implemented in this project, which combines three powerful techniques for uncertainty-aware inventory optimization:

1. **Ensemble Batch Prediction Intervals (EnbPI)** - Xu & Xie (2021)
2. **Conformalized Quantile Regression (CQR)** - Romano et al. (2019)
3. **CVaR Optimization** - Rockafellar & Uryasev (2000)

## Scientific Background

### The Problem

In inventory management under demand uncertainty, we need to:
1. **Forecast demand** with reliable prediction intervals
2. **Quantify uncertainty** to enable risk-aware decisions
3. **Optimize order quantities** considering the cost of stockouts and overstocking

Traditional approaches often fail to provide valid coverage guarantees or adapt to local uncertainty patterns.

### Our Solution: EnbPI + CQR + CVaR

#### Component 1: Ensemble Batch Prediction Intervals (EnbPI)

**Reference**: Xu, C., & Xie, Y. (2021). "Conformal prediction interval for dynamic time-series." *ICML 2021*.

EnbPI addresses the challenge of conformal prediction in time series by:

1. **Bootstrap Ensemble**: Train `B` independent models on bootstrap samples
   - Each model sees ~80% of training data (with replacement)
   - Diversity from different random seeds and data subsets

2. **Leave-One-Out Residuals**: For calibration, compute residuals using only ensemble members that did NOT include that point in their bootstrap sample
   - Prevents overfitting to calibration data
   - Provides valid residuals for conformal calibration
   - Critical for time series where temporal structure matters

3. **Ensemble Aggregation**: Predictions combine all ensemble members
   - Mean for point predictions
   - Spread captures model uncertainty

```
Algorithm: EnbPI
Input: Training data (X_train, y_train), Calibration data (X_cal, y_cal)
For b = 1, ..., B:
    Sample bootstrap indices I_b from {1, ..., n} with replacement
    Train model f_b on (X_train[I_b], y_train[I_b])
For each calibration point i:
    Compute LOO prediction: avg of f_b(X_cal[i]) where i not in I_b
    Residual r_i = |y_cal[i] - LOO_prediction|
Conformal quantile q = (1-alpha) quantile of residuals
```

#### Component 2: Conformalized Quantile Regression (CQR)

**Reference**: Romano, Y., Patterson, E., & Candes, E. (2019). "Conformalized quantile regression." *NeurIPS 2019*.

CQR provides **adaptive prediction intervals** that vary with input features:

1. **Train Quantile Models**: Fit models for lower (alpha/2) and upper (1-alpha/2) quantiles
   - Uses Gradient Boosting with quantile loss
   - Intervals adapt to local variability (heteroscedasticity)

2. **CQR Conformity Score**:
   ```
   E_i = max(q_lo(X_i) - Y_i, Y_i - q_hi(X_i))
   ```
   - Measures how much we need to expand intervals
   - Handles both lower and upper bound violations

3. **Conformal Calibration**: Expand intervals by the (1-alpha) quantile of conformity scores
   - Provides finite-sample coverage guarantee
   - Distribution-free: no parametric assumptions

```
Algorithm: CQR
Input: Quantile predictions (q_lo, q_hi) on calibration set
For each calibration point i:
    E_i = max(q_lo(X_i) - Y_i, Y_i - q_hi(X_i))
Conformal adjustment = (1-alpha) quantile of {E_1, ..., E_n}
Final intervals:
    Lower = q_lo(X) - adjustment
    Upper = q_hi(X) + adjustment
```

#### Component 3: CVaR Optimization

**Reference**: Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of conditional value-at-risk." *Journal of Risk*.

CVaR (Conditional Value-at-Risk) optimizes for worst-case scenarios:

1. **Demand Scenarios**: Sample from prediction intervals
   ```
   D_samples ~ Uniform(lower, upper)
   ```

2. **Newsvendor Loss**: For order quantity q and demand d:
   ```
   L(q, d) = ordering_cost * q + holding_cost * max(0, q-d) + stockout_cost * max(0, d-q)
   ```

3. **CVaR Objective**: Minimize expected loss in worst (1-beta) fraction:
   ```
   CVaR_beta = min_{q, tau} { tau + (1/(1-beta)) * E[max(0, L(q,D) - tau)] }
   ```

## Implementation Details

### Model Architecture

```
EnsembleBatchPI
├── Bootstrap Ensemble (B=10 Random Forests)
│   ├── RF_1: 100 trees, max_depth=10, 80% bootstrap sample
│   ├── RF_2: 100 trees, max_depth=10, 80% bootstrap sample
│   └── ... (10 total)
├── Quantile Regressors (for CQR)
│   ├── Lower (alpha/2): Gradient Boosting with quantile loss
│   └── Upper (1-alpha/2): Gradient Boosting with quantile loss
└── Conformal Calibration
    └── Adjustment factor computed from CQR conformity scores
```

### Key Hyperparameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `alpha` | 0.05 | Significance level (95% coverage) |
| `n_ensemble` | 10 | Number of bootstrap ensemble members |
| `n_estimators` | 100 | Trees per Random Forest |
| `max_depth` | 10 | Maximum tree depth |
| `bootstrap_fraction` | 0.8 | Fraction of data per bootstrap sample |
| `use_quantile_regression` | True | Enable CQR for adaptive intervals |

### Configuration

```python
from configs.config import get_default_config

config = get_default_config()

# Customize EnbPI+CQR settings
config.ensemble_batch_pi.n_ensemble = 15  # More ensemble members
config.ensemble_batch_pi.bootstrap_fraction = 0.75  # Different bootstrap ratio
```

## Theoretical Guarantees

### Coverage Guarantee

For any distribution P and any ensemble model class:

```
P(Y_new in [Lower, Upper]) >= 1 - alpha
```

This holds in **finite samples** without any distributional assumptions (distribution-free).

### Adaptivity

Unlike standard conformal prediction with constant-width intervals, EnbPI+CQR provides:
- **Narrower intervals** where uncertainty is low
- **Wider intervals** where uncertainty is high
- Better decision-making with more informative uncertainty estimates

## Comparison with Other Methods

| Method | Adaptive Intervals | Coverage Guarantee | Ensemble | Time Series |
|--------|-------------------|-------------------|----------|-------------|
| Standard Conformal | No | Yes | No | Limited |
| Quantile Regression | Yes | No | No | Yes |
| EnbPI | No | Yes | Yes | Yes |
| **EnbPI+CQR** | **Yes** | **Yes** | **Yes** | **Yes** |

## Expected Performance

Based on the theoretical properties, EnbPI+CQR+CVaR should:

1. **Coverage**: Achieve >= 95% coverage (for alpha=0.05)
2. **Interval Width**: Narrower than non-adaptive methods on average
3. **CVaR Performance**: Competitive with best traditional methods
4. **Robustness**: Stable across different time windows due to ensemble

## Usage Example

```python
from src.models import EnsembleBatchPI
from src.optimization import compute_order_quantities_cvar

# Initialize model
model = EnsembleBatchPI(
    alpha=0.05,
    n_ensemble=10,
    n_estimators=100,
    use_quantile_regression=True,
    random_state=42
)

# Train
model.fit(X_train, y_train, X_cal, y_cal)

# Predict with intervals
predictions = model.predict(X_test)
print(f"Point: {predictions.point}")
print(f"Lower: {predictions.lower}")
print(f"Upper: {predictions.upper}")

# Compute CVaR-optimal order quantities
orders = compute_order_quantities_cvar(
    predictions.point,
    predictions.lower,
    predictions.upper,
    beta=0.90,  # Focus on worst 10% scenarios
    ordering_cost=10.0,
    holding_cost=2.0,
    stockout_cost=50.0
)
```

## Running Experiments

The EnbPI+CQR+CVaR method is automatically included in the expanding window experiment:

```bash
python scripts/run_expanding_window_experiment.py \
    --output results/expanding_window_with_enbpi \
    --epochs 50
```

Results will include comparisons against:
- Conformal Prediction
- Normal Assumption
- Quantile Regression
- SAA (Sample Average Approximation)
- Deep Learning methods (LSTM, Transformer, TFT)
- SPO (End-to-End learning)
- Seer (Perfect foresight oracle)

## References

1. Xu, C., & Xie, Y. (2021). "Conformal prediction interval for dynamic time-series." *International Conference on Machine Learning (ICML)*.

2. Romano, Y., Patterson, E., & Candes, E. (2019). "Conformalized quantile regression." *Advances in Neural Information Processing Systems (NeurIPS)*.

3. Rockafellar, R. T., & Uryasev, S. (2000). "Optimization of conditional value-at-risk." *Journal of Risk*, 2(3), 21-42.

4. Barber, R. F., Candes, E. J., Ramdas, A., & Tibshirani, R. J. (2019). "Predictive inference with the jackknife+." *Annals of Statistics*.

5. Lakshminarayanan, B., Pritzel, A., & Blundell, C. (2017). "Simple and scalable predictive uncertainty estimation using deep ensembles." *NeurIPS*.

## Authors

This implementation was added to the Inventory CVaR Optimization project to provide a state-of-the-art conformal prediction method specifically designed for time series with adaptive prediction intervals.
