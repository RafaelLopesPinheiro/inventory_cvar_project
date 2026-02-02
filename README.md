# Inventory CVaR Optimization with Probabilistic Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for demand forecasting with uncertainty quantification and CVaR-optimal inventory decisions. This project compares 13 forecasting methods hierarchically (Simple → Advanced → Your Method → DRO → Oracle) using **multi-period expanding window cross-validation** for the newsvendor problem.

## 📋 Overview

This project implements a rigorous comparison of **13 forecasting methods** for inventory optimization using expanding window cross-validation across multiple store-item (SKU) combinations with **multi-period evaluation**.

### Multi-Period Forecasting Approach

Instead of predicting only a single horizon (e.g., 30 days ahead), this experiment evaluates models across **multiple forecast horizons simultaneously**:
- **Day 1**: Immediate next-day demand
- **Day 7**: Week-ahead demand
- **Day 14**: Two-week-ahead demand
- **Day 21**: Three-week-ahead demand
- **Day 28**: Month-ahead demand

This provides:
1. More robust evaluation across different planning horizons
2. Better understanding of model performance degradation over time
3. Joint optimization considering multiple future periods
4. Scientific rigor through multi-horizon cross-validation

### Model Hierarchy (Simple → Advanced → Your Method → DRO → Oracle)

**Naive Baselines (1-3):**
1. **Historical Quantile** - Naive empirical quantile baseline (no features)
2. **Normal Assumption** - Parametric Gaussian assumption
3. **Bootstrapped Newsvendor** - Resampling-based uncertainty quantification

**Operations Research Benchmarks (4-5):**
4. **SAA** - Sample Average Approximation (standard OR baseline)
5. **Two-Stage Stochastic** - Scenario-based optimization

**Distribution-Free Methods (6-7):**
6. **Conformal Prediction** - Distribution-free intervals with coverage guarantees
7. **Quantile Regression** - Direct quantile estimation with Conformalized Quantile Regression (CQR)

**Deep Learning Methods (8-10):**
8. **LSTM Quantile Loss** - Deep learning WITHOUT calibration
9. **LSTM+Conformal** - Deep learning WITH conformal calibration
10. **SPO** - Decision-focused deep learning (Smart Predict-then-Optimize)

**Your Contribution (11):**
11. **EnbPI+CQR+CVaR** - Ensemble Batch PI + Conformalized Quantile Regression + CVaR optimization

**Advanced Benchmarks (12-13):**
12. **DRO** - Distributionally Robust Optimization (Wasserstein)
13. **Seer** - Oracle upper bound (perfect foresight)

All methods use **CVaR (Conditional Value-at-Risk)** optimization via the Rockafellar-Uryasev formulation for risk-aware inventory decisions.

## 🏗️ Project Structure

```
inventory_cvar_project/
├── configs/
│   ├── __init__.py
│   └── config.py                          # Centralized configuration
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py                      # Data loading, feature engineering, multi-period splits
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py                        # Abstract base classes
│   │   ├── traditional.py                 # 10 traditional forecasting models
│   │   ├── deep_learning.py               # LSTM, SPO models
│   │   └── multi_period.py                # Multi-period forecaster wrapper
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── cvar.py                        # CVaR optimization (single & multi-period)
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py                     # Evaluation metrics and statistical tests
│   └── visualization/
│       ├── __init__.py
│       └── plots.py                       # Visualization utilities
├── scripts/
│   └── run_comprehensive_expanding_window.py  # Main experiment runner (RECOMMENDED)
├── tests/                                 # Unit tests
├── notebooks/                             # Jupyter notebooks for exploration
├── results/                               # Output directory
├── requirements.txt
├── setup.py
└── README.md
```

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/inventory-cvar-optimization.git
cd inventory-cvar-optimization

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Install package in development mode
pip install -e .
```

### Running Experiments

The main experiment script is `run_comprehensive_expanding_window.py`, which evaluates all 13 methods using **multi-period expanding window cross-validation**.

```bash
# Single SKU with multi-period evaluation (default: horizons 1,7,14,21,28 days)
python scripts/run_comprehensive_expanding_window.py

# Multiple SKUs for robust evaluation
python scripts/run_comprehensive_expanding_window.py \
    --stores 1,2,3 \
    --items 1,2,3,4,5 \
    --output results/multi_sku/

# Custom horizons (e.g., only evaluate 1, 7, 14 days ahead)
python scripts/run_comprehensive_expanding_window.py \
    --horizons 1,7,14 \
    --output results/custom_horizons/

# Skip deep learning models for faster execution (recommended for multi-SKU)
python scripts/run_comprehensive_expanding_window.py \
    --stores 1,2,3 \
    --items 1,2,3 \
    --no-dl

# Use GPU for deep learning models
python scripts/run_comprehensive_expanding_window.py \
    --device cuda \
    --epochs 100

# Legacy single-period mode (not recommended)
python scripts/run_comprehensive_expanding_window.py \
    --single-period \
    --output results/single_period/
```

**What the script does:**

1. **Loads data** for specified store-item combinations
2. **Creates features** (time features, lags, rolling statistics)
3. **Generates expanding windows** - training set grows over time
4. **Trains all 13 methods** on each window for each horizon
5. **Computes CVaR-optimal order quantities** using joint optimization
6. **Evaluates performance** across multiple metrics
7. **Generates comprehensive visualizations** (8 plots)
8. **Creates detailed reports** with statistical comparisons

### Using as a Library

```python
from src.data import load_and_prepare_data, prepare_sequence_data
from src.models import ConformalPrediction, LSTMQuantileRegression
from src.optimization import compute_order_quantities_cvar
from src.evaluation import compute_all_metrics

# Load data
splits = load_and_prepare_data("train.csv", store_id=1, item_id=1)

# Train conformal prediction model
model = ConformalPrediction(alpha=0.05)
model.fit(splits.train.X, splits.train.y, 
          splits.calibration.X, splits.calibration.y)

# Generate predictions
predictions = model.predict(splits.test.X)

# Compute CVaR-optimal order quantities
orders = compute_order_quantities_cvar(
    predictions.point, predictions.lower, predictions.upper,
    beta=0.90, n_samples=1000
)

# Evaluate
results = compute_all_metrics(
    "Conformal_CVaR", splits.test.y, 
    predictions.point, orders,
    predictions.lower, predictions.upper
)
```

## 📊 Output & Results

When you run `run_comprehensive_expanding_window.py`, it generates:

### Generated Files

**Results Directory** (default: `results/multi_period_expanding/`):
```
results/multi_period_expanding/
├── combined_results.csv           # All window results combined
├── aggregated_results.csv         # Aggregated statistics per method
├── summary_report.txt             # Text summary with key findings
├── model_comparison.png           # Bar chart comparing methods
├── cvar_comparison.png            # CVaR-90 and CVaR-95 comparison
├── coverage_width_tradeoff.png    # Coverage vs interval width scatter
├── horizon_analysis.png           # Performance by forecast horizon
├── method_rankings.png            # Ranking across metrics
├── cost_distribution.png          # Box plots of cost distributions
├── timing_comparison.png          # Execution time comparison
└── timing_vs_cvar_tradeoff.png    # Speed vs performance trade-off
```

### Features Used (9 total)

All models (except HistoricalQuantile) use the same feature set:

**Time Features (3):**
- `month` - Month of year (1-12)
- `day_of_week` - Day of week (0-6)
- `day_of_year` - Day of year (1-365)

**Lag Features (3):**
- `sales_lag_1` - Yesterday's sales
- `sales_lag_7` - Sales 7 days ago
- `sales_lag_28` - Sales 28 days ago

**Rolling Statistics (4):**
- `rolling_mean_7` - 7-day rolling mean
- `rolling_mean_28` - 28-day rolling mean
- `rolling_std_7` - 7-day rolling standard deviation
- `rolling_std_28` - 28-day rolling standard deviation

**Note:** HistoricalQuantile is a naive baseline that intentionally uses NO features (only historical demand values), serving as the simplest possible baseline.

## 🔧 Configuration

All hyperparameters are centralized in `configs/config.py`:

```python
from configs import get_default_config

config = get_default_config()

# Modify cost parameters
config.cost.ordering_cost = 10.0
config.cost.holding_cost = 2.0
config.cost.stockout_cost = 50.0

# Modify CVaR settings
config.cvar.beta = 0.90  # 90% CVaR

# Modify deep learning settings
config.lstm.epochs = 100
config.lstm.hidden_size = 64
```

## 📈 Metrics

The experiment evaluates models across multiple metrics:

### Forecasting Metrics
- **Coverage**: Proportion of actual values within prediction intervals (target: 95%)
- **Average Interval Width**: Mean width of prediction intervals (narrower is better)

### Inventory Metrics (Primary)
- **Mean Cost**: Average daily newsvendor cost (lower is better)
- **CVaR-90**: Expected cost in the worst 10% of days (tail risk, lower is better)
- **CVaR-95**: Expected cost in the worst 5% of days (extreme tail risk, lower is better)
- **Service Level**: Proportion of days without stockouts (higher is better)

### Multi-Period Specific
- **Per-Horizon Metrics**: Separate evaluation for each forecast horizon (1, 7, 14, 21, 28 days)
- **Aggregated Metrics**: Combined performance across all horizons (mean/sum/worst-case)
- **Joint Optimization**: Order quantities optimized considering all horizons simultaneously

### Experimental Design Details

- **Expanding Window**: Training set grows over time (not sliding window)
- **Direct Strategy**: Separate model trained for each horizon
- **Calibration Set**: Fixed size (30 days default) for conformal calibration
- **Test Window**: Aligned with maximum horizon (28 days default)
- **Statistical Testing**: Paired t-tests for significance (p < 0.05)

## 📚 References

### Core Methodology
1. **Rockafellar & Uryasev (2000)** - "Optimization of conditional value-at-risk" - CVaR optimization
2. **Taieb et al. (2012)** - "A review and comparison of strategies for multi-step ahead forecasting" - Multi-period forecasting
3. **Hyndman & Athanasopoulos (2021)** - "Forecasting: Principles and Practice" - Time series foundations

### Conformal Prediction
4. **Vovk et al. (2005)** - "Algorithmic Learning in a Random World" - Conformal prediction theory
5. **Romano et al. (2019)** - "Conformalized Quantile Regression" - CQR method
6. **Xu & Xie (2021)** - "Conformal prediction interval for dynamic time-series" - EnbPI method
7. **Barber et al. (2019)** - "Predictive inference with the jackknife+" - Jackknife+ for time series

### Decision-Focused Learning
8. **Elmachtoub & Grigas (2017)** - "Smart 'Predict, then Optimize'" - SPO framework
9. **Donti et al. (2017)** - "Task-based End-to-end Model Learning" - End-to-end optimization
10. **Ban & Rudin (2019)** - "The Big Data Newsvendor" - Data-driven newsvendor

### Deep Learning for Forecasting
11. **Wen et al. (2017)** - "A Multi-Horizon Quantile Recurrent Forecaster" - LSTM quantile regression
12. **Gasthaus et al. (2019)** - "Probabilistic Forecasting with Spline Quantile" - Quantile networks

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
