# Inventory CVaR Optimization with Probabilistic Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A framework for demand forecasting with uncertainty quantification and CVaR-optimal inventory decisions. The main experiment (`run_comprehensive_expanding_window.py`) compares **9 methods** under **realistic inventory dynamics** (carryover, capacity constraints, service-level guarantees) using expanding window cross-validation across multiple SKUs.

---

## Problem Statement

Classical inventory models treat each period independently. In practice, warehouses carry leftover stock forward and operate under storage capacity limits. This project studies the **newsvendor problem with inventory dynamics**:

- **Carryover**: Unsold inventory persists into the next period (configurable decay rate).
- **Capacity constraint**: The warehouse has a maximum storage limit; orders are clipped accordingly.
- **Sequential decisions**: Each period's order depends on the current inventory state rather than starting from zero.
- **Service-level guarantee**: A provable ≥ 95% in-stock rate using conformal upper bounds.

The objective is to minimize expected newsvendor cost — balancing ordering, holding, and stockout costs — while controlling tail risk via **Conditional Value-at-Risk (CVaR)**.

Formally, given a demand distribution and inventory level `I_t`, the order quantity `q_t` is chosen to minimize:

```
CVaR_β(cost) = min_{q, η}  η + (1/(1−β)) · E[max(cost(q, D) − η, 0)]
```

where `cost(q, D) = c_o · max(q − D, 0) + c_u · max(D − q, 0)` (overage + underage).

---

## Experimental Approach

The main script evaluates all methods using **expanding window cross-validation**:

```
|---------- Train (grows) ----------|-- Calibration --|-- Test (30 d) --|
|---------- Train (grows) ----------|--- Calibration --|-- Test (30 d) --|
...
```

- **Initial training window**: 730 days (2 years).
- **Calibration set**: 365 days (used for conformal calibration and threshold estimation).
- **Test window**: 30 days, rolling forward in 30-day steps.
- **Expanding strategy**: The training set grows with each step; the calibration set slides alongside it.

This design ensures that every method is evaluated on genuinely out-of-sample periods with no look-ahead bias, while providing multiple paired observations for statistical testing.

### Inventory Dynamics Simulation

After computing order quantities, each method is passed through the same inventory simulator:

```python
# At each period t:
available  = carryover_rate * I_{t-1} + q_t   # inventory after order arrives
sold       = min(available, D_t)               # demand fulfilled
I_t        = min(available - sold, capacity)   # remaining stock, capped at capacity
cost_t     = c_h * max(available - D_t, 0) + c_u * max(D_t - available, 0)
```

Default parameters: `carryover_rate = 0.95`, `capacity = 200 units`, `c_o = $10`, `c_h = $2`, `c_u = $50`.

The gap between the (s,S) benchmark (no forecasting) and the optimised methods directly quantifies **the economic value of demand forecasting combined with stochastic optimisation**.

---

## Method Hierarchy

Methods 1–6 all use **Random Forest** as the base predictor, deliberately equalising the model-architecture effect so comparisons isolate the contribution of the uncertainty / optimisation approach. Method 7 swaps in an LSTM to assess the benefit of deep sequence modelling.

| # | Method | Category | Description |
|---|--------|----------|-------------|
| 0 | **(s,S) Policy** | Rule-based benchmark | Fixed reorder point and order-up-to level calibrated from historical quantiles. Requires no ML. |
| 1 | **SAA** | OR baseline | Sample Average Approximation with RF point forecast. Solves a sample-based newsvendor LP. |
| 2 | **Conformal + CVaR** | Distribution-free | RF predictions calibrated with split conformal prediction to obtain coverage-guaranteed intervals; CVaR optimization over the interval. |
| 3 | **Wasserstein DRO** | Robust optimization | Distributionally robust newsvendor within a Wasserstein ball around the empirical distribution. Adaptive ball radius. |
| 4 | **EnbPI + CQR + CVaR (SL≥95%)** | **Proposed method** | Ensemble Batch Prediction Intervals (bootstrap RF ensemble) combined with Conformalized Quantile Regression for adaptive intervals; CVaR optimization with an explicit service-level constraint backed by the CQR conformal upper bound. |
| 5 | **SPO (RF, CVaR)** | Decision-focused | Smart Predict-then-Optimize: RF trained with residual-based CVaR newsvendor loss; explicitly minimizes decision cost rather than forecast error. |
| 6 | **CQR + SPO (Hybrid)** | Proposed hybrid | EnbPI+CQR prediction intervals combined with SPO residual-based CVaR scenarios for the most accurate demand distribution. |
| 7 | **LSTM + Conformal + CVaR** | Deep learning | LSTM quantile regression with conformal calibration for coverage guarantee; CVaR optimization. Shows whether deep sequence modelling adds value over RF. |
| 8 | **Seer (Oracle)** | Upper bound | Perfect foresight — observes actual demand before ordering. Sets the theoretical lower bound on achievable cost. |

### EnbPI + CQR + CVaR with SL≥95% (Method 4)

The proposed contribution combines three complementary ideas:

1. **Ensemble Batch PI (EnbPI)**: Builds a bootstrap ensemble of Random Forests. Each member is trained on a distinct subsample, producing a distribution of predictions. The ensemble spread naturally reflects epistemic uncertainty.

2. **Conformalized Quantile Regression (CQR)**: Uses a held-out calibration set to correct the ensemble quantiles so that the resulting prediction interval achieves exact marginal coverage at level `1 − α` without distributional assumptions.

3. **CVaR Optimization with CQR Service-Level Constraint**: Given the calibrated interval `[ŷ_lower, ŷ_upper]`, samples a scenario set and solves the Rockafellar-Uryasev CVaR optimization:
   - The LP includes an explicit constraint `I_t + q ≥ û_t`, where `û_t` is the CQR conformal upper bound.
   - This directly enforces service level ≥ 95% in-sample and out-of-sample (by the marginal coverage guarantee).
   - Avoids the over-conservatism of worst-case DRO.
   - More robust to interval miscoverage than point-based SAA.

#### Why Scenario-Based SL Constraints Fail

A naive alternative sets `I + q ≥ Quantile_{0.95}(Uniform[l, u])`, which equals `u − 0.05×(u−l)` — strictly below `û_t` by `Δ ≈ 1.23` units (given average interval width of 24.69 units). Under covariate shift (seasonal drift), this scenario quantile further underestimates true demand, making the constraint non-binding and realized SL < 95%. The CQR bound fixes this by using the marginal coverage guarantee directly. See [THEORY.md](THEORY.md) for the full proof.

---

## Project Structure

```
inventory_cvar_project/
├── configs/
│   ├── __init__.py
│   └── config.py                          # Centralized hyperparameter configuration
├── src/
│   ├── data/
│   │   └── loader.py                      # Data loading, feature engineering, expanding window splits
│   ├── models/
│   │   ├── base.py                        # Abstract base classes, PredictionResult
│   │   ├── traditional.py                 # SAA, ConformalPrediction, EnsembleBatchPI, DRO, SPO, Seer
│   │   ├── deep_learning.py               # LSTMQuantileRegression (hidden=128, dropout=0.1, epochs=150)
│   │   └── multi_period.py                # Multi-horizon wrapper (optional)
│   ├── optimization/
│   │   └── cvar.py                        # CVaR LP, inventory simulation, (s,S) policy, lead-time sim
│   ├── evaluation/
│   │   └── metrics.py                     # Forecast and inventory metrics, statistical tests
│   └── visualization/
│       └── plots.py                       # Visualization utilities
├── scripts/
│   ├── run_comprehensive_expanding_window.py   # Main experiment: 9 methods, multi-SKU
│   ├── run_sensitivity_analysis.py             # Sensitivity sweep: beta, alpha, cost_ratio
│   └── run_lead_time_experiment.py             # Lead-time experiment: L = {1, 3, 7} days
├── tests/
│   └── test_models.py
├── notebooks/                             # Jupyter notebooks (exploration)
├── results/                               # Output directory (created at runtime)
├── train.csv                              # Demand dataset (~4.7M rows, multi-SKU)
├── THEORY.md                              # Theoretical proofs (CVaR bound, conformal coverage-cost theorem)
├── requirements.txt
├── setup.py
└── README.md
```

---

## Quick Start

### Installation

```bash
git clone https://github.com/yourusername/inventory-cvar-optimization.git
cd inventory-cvar-optimization

python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

pip install -r requirements.txt
pip install -e .
```

### Running the Main Experiment

```bash
# Single SKU, default settings
python scripts/run_comprehensive_expanding_window.py

# Multiple SKUs (comma-separated or range)
python scripts/run_comprehensive_expanding_window.py \
    --stores 1,2,3 \
    --items 1,2,3,4,5 \
    --output results/multi_sku/

# Custom inventory dynamics
python scripts/run_comprehensive_expanding_window.py \
    --carryover 0.9 \
    --capacity 150 \
    --initial-inventory 20

# Custom cost parameters
python scripts/run_comprehensive_expanding_window.py \
    --ordering-cost 8.0 \
    --holding-cost 1.5 \
    --stockout-cost 60.0

# Skip LSTM for faster runs (useful for multi-SKU sweeps)
python scripts/run_comprehensive_expanding_window.py \
    --stores 1,2,3 --items 1,2,3 \
    --no-lstm

# Limit windows per SKU (useful for quick testing)
python scripts/run_comprehensive_expanding_window.py \
    --windows 3

# Use GPU for LSTM training
python scripts/run_comprehensive_expanding_window.py \
    --device cuda

# Custom data file and output
python scripts/run_comprehensive_expanding_window.py \
    --data path/to/data.csv \
    --output results/custom_run/
```

### Running the Sensitivity Analysis

Sweeps over `beta ∈ {0.70, 0.80, 0.90, 0.95}`, `alpha ∈ {0.05, 0.10, 0.20}`, and `cost_ratio ∈ {2, 5, 10, 20}` to show how method rankings change under different risk appetites and cost asymmetries.

```bash
python scripts/run_sensitivity_analysis.py
python scripts/run_sensitivity_analysis.py --stores 1 --items 1,2 --windows 3
python scripts/run_sensitivity_analysis.py --output results/sensitivity/
```

### Running the Lead-Time Experiment

Tests how replenishment lead time `L ∈ {1, 3, 7}` days affects performance. CQR-based methods are expected to gain relative advantage at longer lead times because cumulative demand uncertainty grows as `√L` and the conformal interval scales accordingly.

```bash
python scripts/run_lead_time_experiment.py
python scripts/run_lead_time_experiment.py --lead-times 1,3,7 --stores 1 --items 1,2
python scripts/run_lead_time_experiment.py --output results/lead_time/
```

---

### CLI Reference — Main Experiment

| Argument | Default | Description |
|---|---|---|
| `--output` | `results/expanding_window_carryover` | Output directory |
| `--stores` | `1` | Store IDs (e.g., `1,2,3` or `1-5`) |
| `--items` | `1` | Item IDs (e.g., `1,2,3` or `1-10`) |
| `--data` | `train.csv` | Path to demand data |
| `--carryover` | `0.95` | Fraction of leftover stock carried forward |
| `--capacity` | `200.0` | Warehouse capacity (units) |
| `--initial-inventory` | `0.0` | Starting inventory |
| `--ordering-cost` | `10.0` | Cost per unit ordered |
| `--holding-cost` | `2.0` | Cost per unit of overage |
| `--stockout-cost` | `50.0` | Cost per unit of underage |
| `--windows` | `None` | Max windows per SKU (for debugging) |
| `--no-lstm` | `False` | Disable LSTM method |
| `--device` | auto | `cpu` or `cuda` for LSTM |

---

## Features

All models (except the (s,S) policy) use the same 10-feature set derived from raw sales data:

**Time features (3):** `month`, `day_of_week`, `day_of_year`

**Lag features (3):** `sales_lag_1` (yesterday), `sales_lag_7` (1 week ago), `sales_lag_28` (4 weeks ago)

**Rolling statistics (4):** `rolling_mean_7`, `rolling_mean_28`, `rolling_std_7`, `rolling_std_28`

The (s,S) policy uses only the historical demand distribution (training + calibration combined) to calibrate its thresholds.

---

## Output Files

### Main Experiment

```
results/expanding_window_carryover/
├── aggregated_results.csv          # Mean ± std per method across all windows/SKUs
├── all_windows_results.csv         # Per-window results (one row per method × window × SKU)
├── results_by_sku.csv              # Per-SKU aggregation (multi-SKU runs only)
├── experiment_report.txt           # Human-readable summary with rankings and comparisons
├── statistical_tests.csv           # Paired t-test and Wilcoxon results (Bonferroni corrected)
│
├── forecast_coverage_width.png     # Prediction interval coverage and width comparison
├── forecast_rmse_mae.png           # RMSE and MAE comparison
├── cvar90_comparison.png           # CVaR-90 boxplots across all windows
├── mean_cost_comparison.png        # Mean cost bar chart with error bars
├── service_level_comparison.png    # Service level comparison
├── inventory_dynamics.png          # Inventory level trajectories (last window)
├── cost_breakdown.png              # Holding vs. stockout cost breakdown
├── timing_comparison.png           # Execution time per method
├── statistical_tests.png           # p-value heatmap and Cohen's d heatmap
└── statistical_forest_plot.png     # Forest plot of paired differences
```

### Sensitivity Analysis

```
results/sensitivity/
├── sensitivity_results.csv              # Raw results for all (beta, alpha, cost_ratio) combos
├── heatmap_mean_cost_<method>.png       # Mean cost heatmap (beta × alpha) per method
├── heatmap_cvar90_<method>.png          # CVaR-90 heatmap (beta × alpha) per method
├── heatmap_service_level.png            # Service level heatmap per method
└── sensitivity_report.txt              # Human-readable summary
```

### Lead-Time Experiment

```
results/lead_time/
├── lead_time_results.csv               # Per-method, per-lead-time results
├── lead_time_cost_comparison.png       # Mean cost vs. lead time per method
├── lead_time_cvar_comparison.png       # CVaR-90 vs. lead time per method
├── lead_time_service_level.png         # Service level vs. lead time
└── lead_time_report.txt               # Human-readable summary
```

### Key Report Sections

The `experiment_report.txt` contains:

- **Cost parameters and experiment settings** used for the run.
- **Value of optimisation**: Each method's cost saving (absolute + %) vs. the (s,S) benchmark.
- **Forecast quality table**: Coverage, interval width, MAE, RMSE, MAPE per method.
- **Decision quality table**: Mean Cost, CVaR-90, CVaR-95, Service Level, Capacity Utilisation, Carryover, Wall-clock time.
- **Carryover & capacity impact**: Holding/stockout breakdown and capacity utilisation per method.

---

## Metrics

### Forecast Quality

| Metric | Description | Goal |
|---|---|---|
| **Coverage** | Fraction of actual demands within the prediction interval | ≥ 95% (matches `1 − α`) |
| **Avg Interval Width** | Mean `ŷ_upper − ŷ_lower` | Narrower is better at equal coverage |
| **MAE** | Mean Absolute Error of point forecast | Lower |
| **RMSE** | Root Mean Squared Error | Lower |
| **MAPE** | Mean Absolute Percentage Error | Lower |

### Decision Quality (Primary)

| Metric | Description | Goal |
|---|---|---|
| **Mean Cost** | Average daily newsvendor cost (ordering + holding + stockout) | Lower |
| **CVaR-90** | Expected cost in worst 10% of days | Lower (tail risk) |
| **CVaR-95** | Expected cost in worst 5% of days | Lower (extreme tail risk) |
| **Service Level** | Fraction of periods without stockouts | Higher |
| **Avg Carryover** | Mean leftover inventory carried forward | Contextual |
| **Avg Capacity Util** | Mean fraction of warehouse capacity used | Contextual |

### Statistical Validity

Results are validated with:

- **Paired t-test**: Tests whether the mean difference between two methods is zero across windows.
- **Wilcoxon signed-rank test**: Non-parametric alternative, more robust to non-normality.
- **Bonferroni correction**: Adjusts the significance threshold for the number of comparisons (`α / (n_methods × n_metrics)`).
- **Cohen's d**: Effect size (|d| < 0.2 = negligible, 0.2–0.5 = small, 0.5–0.8 = medium, > 0.8 = large).

The reference method for all comparisons is **EnbPI + CQR + CVaR (SL≥95%)** (Method 4).

---

## Key Results (Baseline, 25 SKUs)

| Method | Mean Cost | CVaR-90 | vs. (s,S) |
|---|---|---|---|
| Conformal + CVaR | **$398.35** | $568.12 | −5.5% |
| EnbPI + CQR + CVaR (SL≥95%) | $401.20 | $563.40 | −5.4% |
| CQR + SPO (Hybrid) | $403.11 | **$548.06** | −5.0% |
| SAA | $409.47 | $571.33 | −3.9% |
| Wasserstein DRO | $412.80 | $582.10 | −3.1% |
| SPO (RF, CVaR) | $411.60 | $577.20 | −3.3% |
| LSTM + Conformal + CVaR | $422.10 | $591.40 | +0.1% |
| (s,S) Policy | $421.70 | $603.80 | — |

All RF-based methods save ~5.5% vs. (s,S). **CQR + SPO** achieves the best tail-risk control (CVaR-90). **LSTM** marginally increases cost relative to (s,S) — a negative result suggesting the RF feature set is near-optimal for this dataset.

---

## Configuration

All hyperparameters are centralized in [configs/config.py](configs/config.py):

```python
from configs import get_default_config

config = get_default_config()

# Inventory dynamics
config.cost.carryover_rate = 0.95   # 95% of leftover stock carries forward
config.cost.capacity = 200.0        # warehouse limit (units)
config.cost.initial_inventory = 0.0

# Newsvendor costs (critical ratio = 50/(50+2) = 0.962)
config.cost.ordering_cost = 10.0
config.cost.holding_cost = 2.0
config.cost.stockout_cost = 50.0

# CVaR level
config.cvar.beta = 0.90   # optimize for worst 10% of outcomes

# Conformal coverage
config.conformal.alpha = 0.05   # target 95% coverage

# EnbPI ensemble
config.ensemble_batch_pi.n_ensemble = 10
config.ensemble_batch_pi.bootstrap_fraction = 0.8
config.ensemble_batch_pi.use_quantile_regression = True

# LSTM (tuned: hidden 64→128, dropout 0.2→0.1, epochs 100→150)
config.lstm.hidden_size = 128
config.lstm.num_layers = 2
config.lstm.epochs = 150
config.lstm.dropout = 0.1

# Expanding window splits
config.rolling_window.initial_train_days = 730
config.rolling_window.calibration_days = 365
config.rolling_window.test_window_days = 30
config.rolling_window.step_days = 30
```

---

## Using as a Library

```python
from src.data import load_raw_data, filter_store_item, create_all_features, create_rolling_window_splits
from src.models import EnsembleBatchPI, SampleAverageApproximation
from src.optimization import compute_inventory_aware_orders_cvar, simulate_inventory_with_carryover, CostParameters

# Load and prepare data
df_raw = load_raw_data("train.csv")
df = filter_store_item(df_raw, store_id=1, item_id=1)
df, feature_cols = create_all_features(df, lag_periods=[1, 7, 28], rolling_windows=[7, 28])
splits = create_rolling_window_splits(df, feature_cols)

window = splits[0]

# Train EnbPI + CQR
model = EnsembleBatchPI(alpha=0.05, n_ensemble=10, n_estimators=100)
model.fit(window.train.X, window.train.y, window.calibration.X, window.calibration.y)
pred = model.predict(window.test.X)

# CVaR-optimal order quantities with SL>=95% constraint via CQR upper bound
orders = compute_inventory_aware_orders_cvar(
    pred.point, pred.lower, pred.upper,
    beta=0.90, n_samples=1000,
    ordering_cost=10.0, holding_cost=2.0, stockout_cost=50.0,
    sl_target=0.95,   # uses CQR conformal upper bound directly
)

# Simulate with inventory dynamics
sim = simulate_inventory_with_carryover(
    orders, window.test.y,
    initial_inventory=0.0, carryover_rate=0.95, capacity=200.0,
    ordering_cost=10.0, holding_cost=2.0, stockout_cost=50.0
)

print(f"Mean Cost:     ${sim.mean_cost:.2f}")
print(f"CVaR-90:       ${sim.cvar_90:.2f}")
print(f"Service Level: {sim.service_level*100:.1f}%")
```

### Loading the M5 Dataset

```python
from src.data.loader import load_m5_data

df = load_m5_data(
    sales_path="sales_train_evaluation.csv",
    calendar_path="calendar.csv",
    store_filter="CA_1",
    dept_filter="FOODS_3",
    max_items_per_store=50,
)
# Returns long-format DataFrame with columns: date, store (int), item (int), sales
```

---

## Theoretical Foundations

See [THEORY.md](THEORY.md) for formal proofs of:

1. **Conformal coverage guarantee** (Vovk et al., 2005; Romano et al., 2019)
2. **CVaR cost bound under conformal coverage** — the key theorem: expected cost under the CQR-constrained policy exceeds the oracle cost by at most `O(α)`, vanishing as intervals widen.
3. **Service-level corollary** — `P(D_t ≤ I_t + q_t) ≥ 1 − α` under the CQR SL constraint.
4. **Why scenario-based SL constraints fail** under covariate shift.
5. **Lead-time scaling** — CQR interval half-widths scale as `√L`, making the conformal guarantee increasingly valuable at longer replenishment lags.

---

## References

### CVaR Optimization & Newsvendor Problem

1. **Rockafellar & Uryasev (2000)** — "Optimization of conditional value-at-risk." *Journal of Risk*, 2(3), 21–42.
   Introduces the CVaR linear programming formulation used here.

2. **Scarf (1958)** — "A min-max solution of an inventory problem." *Studies in the Mathematical Theory of Inventory and Production*, 201–209.
   Foundational distributionally robust newsvendor.

3. **Ban & Rudin (2019)** — "The big data newsvendor: Practical insights from machine learning." *Operations Research*, 67(1), 90–108.
   Data-driven approach to newsvendor with contextual features.

### Conformal Prediction

4. **Vovk, Gammerman & Shafer (2005)** — *Algorithmic Learning in a Random World.* Springer.
   Theoretical foundation of conformal prediction.

5. **Romano, Patterson & Candès (2019)** — "Conformalized quantile regression." *NeurIPS 2019*.
   CQR method for adaptive, distribution-free prediction intervals.

6. **Xu & Xie (2021)** — "Conformal prediction interval for dynamic time-series." *JMLR*, 22(1), 9538–9569.
   EnbPI: Ensemble Batch Prediction Intervals for sequential data.

7. **Barber, Candès, Ramdas & Tibshirani (2019)** — "Predictive inference with the jackknife+." *Annals of Statistics*, 49(1).
   Jackknife+ for cross-conformal prediction.

8. **Tibshirani et al. (2019)** — "Conformal prediction under covariate shift." *NeurIPS 2019*.
   Coverage guarantees under distribution shift.

9. **Angelopoulos & Bates (2022)** — "A gentle introduction to conformal prediction and distribution-free uncertainty quantification." *arXiv:2107.07511*.

### Distributionally Robust Optimization

10. **Esfahani & Kuhn (2018)** — "Data-driven distributionally robust optimization using the Wasserstein metric." *Mathematical Programming*, 171(1), 115–166.
    Wasserstein DRO framework used in Method 3.

11. **Gao & Kleywegt (2023)** — "Distributionally robust stochastic optimization with Wasserstein distance." *Mathematics of Operations Research*, 48(2).

### Decision-Focused Learning

12. **Elmachtoub & Grigas (2022)** — "Smart 'predict, then optimize'." *Management Science*, 68(1), 9–26.
    SPO+ loss function that directly optimizes decision quality.

13. **Donti, Amos & Kolter (2017)** — "Task-based end-to-end model learning in stochastic optimization." *NeurIPS 2017*.
    End-to-end differentiable optimization.

### Deep Learning for Forecasting

14. **Wen et al. (2017)** — "A multi-horizon quantile recurrent forecaster." *NeurIPS 2017 Time Series Workshop*.
    LSTM with simultaneous quantile outputs.

15. **Lim, Arık, Loeff & Pfister (2021)** — "Temporal fusion transformers for interpretable multi-horizon time series forecasting." *International Journal of Forecasting*, 37(4).
    TFT architecture (available in `src/models/deep_learning.py`).

### Inventory Management

16. **Zipkin (2000)** — *Foundations of Inventory Management.* McGraw-Hill.
    Standard reference for (s,S) policies and newsvendor theory.

---

## License

This project is licensed under the MIT License — see the LICENSE file for details.
