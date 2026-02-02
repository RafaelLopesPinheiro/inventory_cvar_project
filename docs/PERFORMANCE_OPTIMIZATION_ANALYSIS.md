# Performance Optimization Analysis

## Project Overview

This document provides a comprehensive analysis of performance optimization opportunities for the Inventory CVaR Optimization project. The project implements probabilistic demand forecasting with CVaR-optimal inventory decisions, comparing traditional statistical methods with deep learning approaches.

**Key Objectives to Preserve:**
- Probabilistic demand forecasting with uncertainty quantification
- CVaR (Conditional Value-at-Risk) optimization for risk-averse inventory decisions
- Conformal calibration for coverage guarantees
- Fair comparison across traditional and deep learning methods

---

## Executive Summary

The main performance bottlenecks identified are:

| Category | Bottleneck | Estimated Impact | Optimization Difficulty |
|----------|-----------|------------------|------------------------|
| CVaR Optimization | Sequential scipy.minimize calls | **High** | Medium |
| Ensemble Training | Sequential Random Forest fitting | **High** | Low |
| Deep Learning | No mixed-precision training | **Medium** | Low |
| Experiment Pipeline | Sequential model execution | **High** | Medium |
| Data Processing | Redundant pandas operations | **Low** | Low |

---

## 1. CVaR Optimization Bottlenecks

### Current Implementation (`src/optimization/cvar.py`)

The most significant bottleneck is in `compute_order_quantities_cvar()` (lines 129-191):

```python
# Current: Sequential loop over each prediction
for i in range(len(point_pred)):
    demand_samples = rng.uniform(lower[i], upper[i], n_samples)
    q_opt = optimize_cvar_single(demand_samples, beta, ...)
    order_quantities.append(q_opt)
```

**Problems:**
1. Each day requires a separate scipy.optimize call
2. No vectorization or parallel processing
3. Each optimization starts from scratch (no warm-starting)

### Recommended Optimizations

#### 1.1 Vectorized CVaR Computation (High Impact)

For many scenarios, an analytical or semi-analytical solution exists:

```python
def compute_order_quantities_cvar_vectorized(
    point_pred: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    beta: float = 0.90,
    n_samples: int = 1000,
    ...
) -> np.ndarray:
    """Vectorized CVaR optimization using quantile-based approximation."""
    rng = np.random.RandomState(random_seed)

    # Generate all samples at once: shape (n_predictions, n_samples)
    n_pred = len(point_pred)
    uniform_samples = rng.random((n_pred, n_samples))
    demand_samples = lower[:, np.newaxis] + uniform_samples * (upper - lower)[:, np.newaxis]

    # For newsvendor with CVaR, optimal q is approximately the (beta + some_factor) quantile
    # This is a well-known result in OR literature
    critical_quantile = beta + (1 - beta) * (stockout_cost / (stockout_cost + holding_cost))

    order_quantities = np.quantile(demand_samples, critical_quantile, axis=1)
    return np.maximum(0, order_quantities)
```

**Expected speedup: 10-50x** for large datasets.

#### 1.2 Parallel Processing with joblib (Medium Impact)

```python
from joblib import Parallel, delayed

def compute_order_quantities_cvar_parallel(
    point_pred, lower, upper, beta=0.90, n_samples=1000, n_jobs=-1, ...
) -> np.ndarray:
    """Parallel CVaR optimization across multiple cores."""

    def optimize_single(i, seed):
        rng = np.random.RandomState(seed)
        demand_samples = rng.uniform(lower[i], upper[i], n_samples)
        return optimize_cvar_single(demand_samples, beta, ...)

    results = Parallel(n_jobs=n_jobs)(
        delayed(optimize_single)(i, random_seed + i)
        for i in range(len(point_pred))
    )
    return np.array(results)
```

**Expected speedup: 4-8x** on typical multi-core systems.

#### 1.3 Warm-Starting Optimization (Low-Medium Impact)

```python
def optimize_cvar_batch_warmstart(demand_samples_batch, beta, ...):
    """Optimize with warm-starting from previous solution."""
    results = []
    prev_solution = None

    for demand_samples in demand_samples_batch:
        if prev_solution is not None:
            q0 = prev_solution  # Use previous solution as starting point
        else:
            q0 = np.mean(demand_samples)

        result = minimize(..., x0=[q0, tau0], ...)
        prev_solution = result.x[0]
        results.append(max(0, result.x[0]))

    return np.array(results)
```

**Expected speedup: 1.5-3x** due to faster convergence.

---

## 2. Ensemble Model Training Bottlenecks

### Current Implementation (`src/models/traditional.py`)

`EnsembleBatchPI.fit()` (lines 1284-1416) trains ensemble members sequentially:

```python
# Current: Sequential training
for i, (X_boot, y_boot, indices) in enumerate(bootstrap_data):
    model = RandomForestRegressor(...)
    model.fit(X_boot, y_boot)
    self.ensemble_models.append(model)
```

### Recommended Optimizations

#### 2.1 Parallel Ensemble Training (High Impact)

```python
from joblib import Parallel, delayed

def _train_single_member(self, X_boot, y_boot, seed):
    model = RandomForestRegressor(
        n_estimators=self.n_estimators,
        max_depth=self.max_depth,
        random_state=seed,
        n_jobs=1  # Single-threaded to avoid over-subscription
    )
    model.fit(X_boot, y_boot)
    return model

def fit(self, X_train, y_train, X_cal, y_cal):
    bootstrap_data = self._create_bootstrap_samples(X_train, y_train, self.n_ensemble)

    # Parallel training
    self.ensemble_models = Parallel(n_jobs=-1)(
        delayed(self._train_single_member)(X_boot, y_boot, self.random_state + i)
        for i, (X_boot, y_boot, _) in enumerate(bootstrap_data)
    )
```

**Expected speedup: 3-5x** for ensemble training phase.

#### 2.2 Use HistGradientBoosting for Larger Datasets

For datasets with >10,000 samples, consider:

```python
from sklearn.ensemble import HistGradientBoostingRegressor

# Significantly faster than RandomForestRegressor for large datasets
model = HistGradientBoostingRegressor(
    max_iter=100,
    max_depth=10,
    random_state=seed
)
```

**Expected speedup: 2-10x** depending on dataset size.

---

## 3. Deep Learning Optimizations

### Current Implementation (`src/models/deep_learning.py`)

The deep learning models have several optimization opportunities:

### Recommended Optimizations

#### 3.1 Mixed-Precision Training (Medium Impact)

```python
import torch
from torch.cuda.amp import autocast, GradScaler

class LSTMQuantileRegression(BaseDeepLearningForecaster):
    def fit(self, X_train, y_train, X_cal, y_cal):
        scaler = GradScaler()

        for epoch in range(self.epochs):
            for batch_X, batch_y in dataloader:
                optimizer.zero_grad()

                with autocast():  # Mixed precision
                    outputs = self.model(batch_X)
                    loss = criterion(outputs, batch_y)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
```

**Expected speedup: 1.5-2x** on modern GPUs with Tensor Cores.

#### 3.2 DataLoader Optimization

```python
dataloader = DataLoader(
    dataset,
    batch_size=self.batch_size,
    shuffle=True,
    num_workers=4,          # Parallel data loading
    pin_memory=True,        # Faster GPU transfer
    persistent_workers=True # Keep workers alive between epochs
)
```

**Expected speedup: 1.2-1.5x** for training throughput.

#### 3.3 Compile Models with torch.compile (PyTorch 2.0+)

```python
def fit(self, ...):
    self.model = LSTMQuantileNet(...).to(self.device)

    # Compile for faster execution (PyTorch 2.0+)
    if hasattr(torch, 'compile'):
        self.model = torch.compile(self.model, mode='reduce-overhead')
```

**Expected speedup: 1.2-1.8x** for inference.

#### 3.4 Early Stopping to Reduce Training Time

```python
def fit(self, X_train, y_train, X_cal, y_cal):
    best_val_loss = float('inf')
    patience_counter = 0
    patience = 15

    for epoch in range(self.epochs):
        # Training...

        # Validation
        val_loss = self._validate(X_cal, y_cal)

        if val_loss < best_val_loss - 1e-4:
            best_val_loss = val_loss
            patience_counter = 0
            # Save best model
        else:
            patience_counter += 1
            if patience_counter >= patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
```

**Expected speedup: 1.5-3x** average, by avoiding unnecessary epochs.

---

## 4. Experiment Pipeline Optimizations

### Current Implementation (`scripts/run_expanding_window_experiment.py`)

Models are trained and evaluated sequentially:

```python
# Current: Sequential execution
cp_model.fit(X_train, y_train, X_cal, y_cal)
normal_model.fit(X_train, y_train, X_cal, y_cal)
qr_model.fit(X_train, y_train, X_cal, y_cal)
# ... more models ...
```

### Recommended Optimizations

#### 4.1 Parallel Model Training (High Impact)

```python
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def train_model(model_config):
    """Train a single model and return results."""
    model_class, model_kwargs, X_train, y_train, X_cal, y_cal, X_test, y_test = model_config
    model = model_class(**model_kwargs)
    model.fit(X_train, y_train, X_cal, y_cal)
    predictions = model.predict(X_test)
    return model, predictions

def run_window_experiment_parallel(window_split, config):
    # Prepare model configurations
    traditional_configs = [
        (ConformalPrediction, {...}, X_train, y_train, X_cal, y_cal, X_test, y_test),
        (NormalAssumption, {...}, X_train, y_train, X_cal, y_cal, X_test, y_test),
        # ... more models
    ]

    # Train traditional models in parallel (CPU-bound)
    with ProcessPoolExecutor(max_workers=4) as executor:
        traditional_results = list(executor.map(train_model, traditional_configs))

    # Train DL models sequentially (GPU-bound, typically)
    # Or use ThreadPoolExecutor if models fit on different GPU streams
```

**Expected speedup: 2-4x** for the traditional models phase.

#### 4.2 Shared Feature Cache

```python
class FeatureCache:
    """Cache computed features across model runs."""
    _cache = {}

    @classmethod
    def get_or_compute(cls, key, compute_fn):
        if key not in cls._cache:
            cls._cache[key] = compute_fn()
        return cls._cache[key]

# Usage in experiment
rf_predictions = FeatureCache.get_or_compute(
    f"rf_predictions_{window_idx}",
    lambda: base_rf_model.predict(X_test)
)
```

#### 4.3 Model Caching for Expanding Windows

In expanding window experiments, models can be fine-tuned rather than retrained from scratch:

```python
class IncrementalModel:
    """Wrapper for incremental model updates."""

    def __init__(self, base_model):
        self.base_model = base_model
        self.is_initial_fit = True

    def fit(self, X_train, y_train, X_cal, y_cal):
        if self.is_initial_fit:
            self.base_model.fit(X_train, y_train, X_cal, y_cal)
            self.is_initial_fit = False
        else:
            # Fine-tune with only new data
            self.base_model.partial_fit(X_new, y_new)
```

---

## 5. Data Processing Optimizations

### Current Implementation (`src/data/loader.py`)

### Recommended Optimizations

#### 5.1 Numba-Accelerated Rolling Features

```python
from numba import jit, prange
import numpy as np

@jit(nopython=True, parallel=True)
def rolling_mean_numba(arr, window):
    """Numba-accelerated rolling mean."""
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan

    for i in prange(window-1, n):
        result[i] = np.mean(arr[i-window+1:i+1])

    return result

@jit(nopython=True, parallel=True)
def rolling_std_numba(arr, window):
    """Numba-accelerated rolling std."""
    n = len(arr)
    result = np.empty(n)
    result[:window-1] = np.nan

    for i in prange(window-1, n):
        result[i] = np.std(arr[i-window+1:i+1])

    return result
```

**Expected speedup: 2-5x** for feature engineering on large datasets.

#### 5.2 Efficient Lag Feature Creation

```python
def create_lag_features_vectorized(df, target_col, lags):
    """Vectorized lag feature creation."""
    sales = df[target_col].values
    n = len(sales)

    # Pre-allocate array for all lag features
    max_lag = max(lags)
    lag_features = np.full((n, len(lags)), np.nan)

    for i, lag in enumerate(lags):
        lag_features[lag:, i] = sales[:-lag]

    # Add to dataframe at once
    for i, lag in enumerate(lags):
        df[f'{target_col}_lag_{lag}'] = lag_features[:, i]

    return df
```

---

## 6. Memory Optimization

### 6.1 Use Float32 Instead of Float64

```python
# In data preparation
X_train = X_train.astype(np.float32)
y_train = y_train.astype(np.float32)

# In PyTorch models
X_tensor = torch.FloatTensor(X_train)  # Already float32
```

**Expected memory reduction: 50%**, often with negligible accuracy impact.

### 6.2 Generator-Based Data Loading

```python
def data_generator(filepath, store_ids, item_ids, chunk_size=10000):
    """Memory-efficient data loading."""
    for chunk in pd.read_csv(filepath, chunksize=chunk_size):
        for store_id in store_ids:
            for item_id in item_ids:
                filtered = chunk[(chunk['store'] == store_id) & (chunk['item'] == item_id)]
                if len(filtered) > 0:
                    yield store_id, item_id, filtered
```

---

## 7. Profiling Recommendations

Before implementing optimizations, profile the code to identify actual bottlenecks:

### 7.1 cProfile for CPU Profiling

```bash
python -m cProfile -o profile_output.prof scripts/run_expanding_window_experiment.py
```

```python
import pstats
p = pstats.Stats('profile_output.prof')
p.sort_stats('cumulative').print_stats(30)
```

### 7.2 Memory Profiling

```bash
pip install memory_profiler
python -m memory_profiler scripts/run_expanding_window_experiment.py
```

### 7.3 Line Profiling for Specific Functions

```python
# Add @profile decorator to functions of interest
@profile
def compute_order_quantities_cvar(...):
    ...
```

```bash
kernprof -l -v script.py
```

---

## 8. Implementation Priority Matrix

| Optimization | Impact | Effort | Priority | Files Affected |
|-------------|--------|--------|----------|----------------|
| Vectorized CVaR | High | Medium | **1** | `cvar.py` |
| Parallel CVaR | High | Low | **2** | `cvar.py` |
| Parallel Ensemble | High | Low | **3** | `traditional.py` |
| Mixed Precision | Medium | Low | **4** | `deep_learning.py` |
| Early Stopping | Medium | Low | **5** | `deep_learning.py` |
| Parallel Experiment | High | Medium | **6** | `run_*.py` scripts |
| DataLoader Optimization | Low | Low | **7** | `deep_learning.py` |
| Numba Rolling Features | Low | Medium | **8** | `loader.py` |

---

## 9. Estimated Overall Improvement

With all optimizations implemented:

| Phase | Current Time (estimate) | Optimized Time | Speedup |
|-------|------------------------|----------------|---------|
| Data Loading | 5s | 2s | 2.5x |
| Traditional Models | 60s | 15s | 4x |
| Deep Learning | 300s | 150s | 2x |
| CVaR Optimization | 120s | 10s | 12x |
| **Total** | **~8 min** | **~3 min** | **~2.5x** |

*Note: Actual times depend on hardware and dataset size.*

---

## 10. Caveats and Considerations

### Preserving Correctness
- All optimizations must be validated against the original implementation
- Run unit tests after each optimization
- Compare numerical outputs (within floating-point tolerance)

### Reproducibility
- Parallel execution may affect random number generation order
- Use explicit seeds per parallel worker
- Document any changes to default behavior

### Trade-offs
- Some optimizations (e.g., early stopping) may slightly affect results
- Profile before optimizing - focus on actual bottlenecks
- Consider maintainability vs. performance gains

---

## Appendix: Quick Wins Checklist

- [ ] Add `n_jobs=-1` to all sklearn models (RandomForest, GradientBoosting)
- [ ] Enable `pin_memory=True` in PyTorch DataLoaders
- [ ] Use `torch.no_grad()` during inference (already done, verify)
- [ ] Convert data to float32 where appropriate
- [ ] Add early stopping to deep learning training loops
- [ ] Use `HistGradientBoostingRegressor` for quantile regression (faster)
- [ ] Pre-compute and cache prediction intervals for CVaR optimization

---

*Analysis prepared for the Inventory CVaR Optimization project.*
*Date: 2026-02-02*
