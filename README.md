# Inventory CVaR Optimization with Probabilistic Forecasting

[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive framework for demand forecasting with uncertainty quantification and CVaR-optimal inventory decisions. This project compares traditional statistical methods with state-of-the-art deep learning approaches for the newsvendor problem.

## 📋 Overview

This project implements and compares multiple approaches for probabilistic demand forecasting and risk-aware inventory optimization:

### Traditional Methods
- **Conformal Prediction + CVaR**: Distribution-free prediction intervals with finite-sample coverage guarantees
- **Normal Assumption + CVaR**: Parametric Gaussian assumption with empirical variance estimation
- **Quantile Regression + CVaR**: Direct quantile estimation with conformal calibration
- **Sample Average Approximation (SAA)**: Standard operations research benchmark
- **Expected Value**: Risk-neutral baseline

### Deep Learning Methods
- **LSTM Quantile Regression**: Recurrent neural network with quantile outputs
- **Transformer Quantile Regression**: Self-attention mechanism for temporal dependencies
- **Deep Ensemble**: Multiple neural networks for uncertainty estimation
- **MC Dropout LSTM**: Bayesian approximation via Monte Carlo dropout

All methods use **CVaR (Conditional Value-at-Risk)** optimization via the Rockafellar-Uryasev formulation for risk-aware inventory decisions.

## 🏗️ Project Structure

```
inventory_cvar_project/
├── configs/
│   ├── __init__.py
│   └── config.py              # Centralized configuration
├── src/
│   ├── data/
│   │   ├── __init__.py
│   │   └── loader.py          # Data loading and preprocessing
│   ├── models/
│   │   ├── __init__.py
│   │   ├── base.py            # Abstract base classes
│   │   ├── traditional.py     # Traditional forecasting models
│   │   └── deep_learning.py   # Deep learning models
│   ├── optimization/
│   │   ├── __init__.py
│   │   └── cvar.py            # CVaR optimization
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── metrics.py         # Evaluation metrics and statistical tests
│   └── visualization/
│       ├── __init__.py
│       └── plots.py           # Visualization utilities
├── scripts/
│   ├── run_experiment.py      # Main experiment runner
│   └── run_multi_store_experiment.py  # Multi-store evaluation
├── tests/                     # Unit tests
├── notebooks/                 # Jupyter notebooks for exploration
├── results/                   # Output directory
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

```bash
# Single store-item experiment
python scripts/run_experiment.py --output results/

# Multi-store experiment for robust evaluation
python scripts/run_multi_store_experiment.py \
    --stores 1,2,3,4,5 \
    --items 1,2,3,4,5,6,7,8,9,10 \
    --output results/multi_store/

# Use GPU if available
python scripts/run_experiment.py --device cuda --epochs 100
```

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

## 📊 Key Results

Our experiments on the Store Item Demand Forecasting dataset show:

| Method | Coverage | Mean Cost | CVaR-90 | Service Level |
|--------|----------|-----------|---------|---------------|
| Conformal_CVaR | 93.2% | $310.39 | $436.52 | 94.8% |
| Normal_CVaR | 93.2% | $310.69 | $436.35 | 94.8% |
| SAA | N/A | $317.95 | $432.95 | 95.6% |
| MCDropout_LSTM | 94.5% | $331.60 | $447.94 | 92.6% |
| Transformer_QR | 96.2% | $356.20 | $461.30 | 95.1% |

**Key Findings:**
- Traditional methods (Conformal, Normal, SAA) achieve lower tail risk than deep learning methods
- Conformal Prediction provides reliable coverage guarantees
- All comparisons are statistically significant (p < 0.001)

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

### Forecasting Metrics
- **Coverage**: Proportion of actual values within prediction intervals
- **Average Interval Width**: Mean width of prediction intervals
- **MAE/RMSE**: Point prediction accuracy

### Inventory Metrics
- **Mean Cost**: Average daily newsvendor cost
- **CVaR-90/95**: Expected cost in the worst 10%/5% of days
- **Service Level**: Proportion of days without stockouts
- **Total Cost**: Cumulative cost over test period

## 📚 References

1. Rockafellar & Uryasev (2000) "Optimization of conditional value-at-risk"
2. Romano et al. (2019) "Conformalized Quantile Regression"
3. Lakshminarayanan et al. (2017) "Simple and Scalable Predictive Uncertainty Estimation using Deep Ensembles"
4. Gal & Ghahramani (2016) "Dropout as a Bayesian Approximation"
5. Ban & Rudin (2019) "The Big Data Newsvendor"

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
