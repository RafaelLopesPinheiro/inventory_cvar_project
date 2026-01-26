# SPO/End-to-End Baseline Implementation

This document describes the implementation of the **SPO (Smart "Predict, then Optimize") / End-to-End baseline** for inventory CVaR optimization, a critical competitor to traditional predict-then-optimize approaches.

## Overview

### What is SPO/End-to-End Learning?

Traditional machine learning for decision-making follows a two-stage approach:
1. **Predict**: Train a model to minimize prediction error (MSE, MAE, etc.)
2. **Optimize**: Use predictions in a downstream optimization problem

**SPO/End-to-End Learning** directly optimizes the downstream decision loss:
- **Loss Function**: Actual decision cost (newsvendor cost) instead of prediction error
- **Objective**: Minimize inventory costs directly, not forecast accuracy
- **Decision-Focused**: Learns predictions that lead to better inventory decisions

### Key References
- Elmachtoub & Grigas (2017) "Smart 'Predict, then Optimize'"
- Donti et al. (2017) "Task-based End-to-end Model Learning"

## Implementation Details

### Model Architecture

**File**: `src/models/deep_learning.py`

The SPO baseline consists of three main components:

#### 1. SPOLoss (Decision-Focused Loss Function)
```python
class SPOLoss(nn.Module):
    """
    Optimizes newsvendor cost directly instead of prediction error.

    Loss = ordering_cost * q + holding_cost * max(0, q-d)
           + stockout_cost * max(0, d-q)
    """
```

**Key Features**:
- Takes predicted quantiles as input
- Computes order quantities from predictions
- Calculates actual newsvendor cost against true demand
- Backpropagates through decision cost (not prediction error)

#### 2. SPONet (Neural Network Architecture)
```python
class SPONet(nn.Module):
    """
    LSTM-based network outputting quantile predictions.
    Same architecture as LSTM_QR but trained with SPO loss.
    """
```

**Architecture**:
- Input: Time series sequences (28 days default)
- LSTM layers: 2 layers with 64 hidden units
- Output: 3 quantiles [2.5%, 50%, 97.5%]
- Dropout: 0.2 for regularization

#### 3. SPOEndToEnd (Main Model Class)
```python
class SPOEndToEnd(BaseDeepLearningForecaster):
    """
    Full SPO/End-to-End model with training and prediction.
    """
```

**Key Parameters**:
- `hidden_size`: 64 (LSTM hidden dimension)
- `num_layers`: 2 (LSTM depth)
- `epochs`: 50-100 (training iterations)
- `ordering_cost`: $10 (cost per unit ordered)
- `holding_cost`: $2 (cost per unit held)
- `stockout_cost`: $50 (cost per unit short)
- `beta`: 0.90 (CVaR level)

## How to Run

### 1. Prerequisites

Ensure dependencies are installed:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn scipy torch
```

### 2. Run Expanding Window Experiment with SPO

```bash
# Run experiment with default settings
python scripts/run_expanding_window_experiment.py

# Custom settings
python scripts/run_expanding_window_experiment.py \
    --epochs 50 \
    --output results/expanding_window_spo \
    --device cpu
```

**What it does**:
- Loads time series data with expanding windows
- Trains 11 models including SPO baseline:
  - **Traditional**: Conformal Prediction, Normal Assumption, Quantile Regression, SAA
  - **Deep Learning**: LSTM-QR, Transformer-QR, TFT
  - **Decision-Focused**: SPO/End-to-End ⭐
- Computes CVaR-optimal order quantities
- Evaluates all performance metrics
- Saves results to CSV files

### 3. Generate Visualizations

```bash
python scripts/visualize_expanding_spo_results.py
```

**Generates 5 key visualizations**:
1. **spo_comparison.png** - SPO vs all other methods over time
2. **spo_ranking.png** - Performance rankings highlighting SPO position
3. **method_categories_with_spo.png** - Traditional vs DL vs SPO comparison
4. **comprehensive_comparison.png** - All metrics side-by-side
5. **spo_summary_table.png** - Comprehensive results table

## Results Interpretation

### Expected Outcomes

The SPO baseline should demonstrate:

1. **Competitive Decision Costs**
   - Mean cost comparable or better than predict-then-optimize methods
   - Lower CVaR (tail risk) in some scenarios

2. **Trade-offs**
   - May have slightly higher prediction error (MAE/RMSE)
   - But lower actual inventory costs (the true objective)
   - Demonstrates value of decision-focused learning

3. **When SPO Excels**
   - High-cost shortage scenarios (high stockout costs)
   - Risk-averse settings (CVaR optimization)
   - When prediction accuracy ≠ decision quality

### Performance Metrics

**Inventory Metrics** (Primary):
- `Mean_Cost`: Average daily newsvendor cost
- `CVaR-90`: Expected cost in worst 10% of days (tail risk)
- `CVaR-95`: Expected cost in worst 5% of days
- `Service_Level`: Proportion of days without stockouts

**Forecast Metrics** (Secondary):
- `MAE`: Mean Absolute Error (prediction accuracy)
- `RMSE`: Root Mean Squared Error
- `Coverage`: Prediction interval coverage rate
- `Interval_Width`: Average width of prediction intervals

### Reading the Results

**Key Question**: *Does SPO achieve lower costs despite possibly higher prediction error?*

Example interpretation:
```
SPO Mean Cost: $142.50 ± $12.30
SPO CVaR-90: $185.20 ± $18.50
SPO MAE: 15.80 ± 2.40

LSTM-QR Mean Cost: $148.30 ± $13.20
LSTM-QR CVaR-90: $195.40 ± $19.30
LSTM-QR MAE: 14.20 ± 2.10
```

**Interpretation**: SPO has slightly higher MAE (+11%), but achieves lower mean cost (-4%) and CVaR-90 (-5%), demonstrating that optimizing for decisions > optimizing for predictions.

## File Structure

### New Files Added

```
inventory_cvar_project/
├── src/models/
│   └── deep_learning.py          # SPOLoss, SPONet, SPOEndToEnd classes
├── scripts/
│   ├── run_expanding_window_experiment.py    # Main experiment runner
│   └── visualize_expanding_spo_results.py    # Visualization generator
├── SPO_BASELINE_README.md        # This file
└── results/expanding_window_spo/  # Output directory
    ├── expanding_window_all.csv           # Detailed results
    ├── expanding_window_aggregated.csv    # Summary statistics
    └── visualizations/                     # Generated plots
        ├── spo_comparison.png
        ├── spo_ranking.png
        ├── method_categories_with_spo.png
        ├── comprehensive_comparison.png
        └── spo_summary_table.png
```

## Technical Details

### SPO Loss Computation

The SPO loss directly computes the newsvendor cost:

```python
# Order quantity from median prediction
order_qty = point_pred

# Compute costs
overage = max(0, order_qty - actual_demand)
underage = max(0, actual_demand - order_qty)

cost = ordering_cost * order_qty +
       holding_cost * overage +
       stockout_cost * underage
```

**Gradient Flow**:
- Gradients backpropagate through the cost function
- Model learns to minimize decision cost, not prediction error
- Encourages predictions that lead to better inventory decisions

### Conformal Calibration

Like other models, SPO uses conformal calibration:
1. Train on training set with SPO loss
2. Compute prediction intervals on calibration set
3. Adjust intervals to achieve target coverage (95%)

This ensures valid uncertainty quantification while maintaining decision-focused learning.

### Comparison to Traditional Approaches

| Aspect | Traditional | SPO/End-to-End |
|--------|-------------|----------------|
| **Loss Function** | MSE, MAE, Quantile Loss | Newsvendor Cost |
| **Optimization** | Two-stage (predict → optimize) | End-to-end |
| **Objective** | Minimize prediction error | Minimize decision cost |
| **Advantages** | Better forecasts | Better decisions |
| **Best For** | General forecasting | Specific decision problems |

## Troubleshooting

### Common Issues

**1. CUDA Out of Memory**
```bash
# Use CPU instead
python scripts/run_expanding_window_experiment.py --device cpu
```

**2. Slow Training**
```bash
# Reduce epochs
python scripts/run_expanding_window_experiment.py --epochs 30
```

**3. Import Errors**
```bash
# Ensure all dependencies installed
pip install -r requirements.txt  # If available
```

## Next Steps

### Experimentation Ideas

1. **Vary Cost Parameters**
   - Increase stockout_cost to see SPO advantage grow
   - Test different cost ratios

2. **Different CVaR Levels**
   - Try beta=0.95 for more risk-averse decisions
   - Compare SPO performance across risk levels

3. **Multiple Stores/Items**
   - Run on different store-item combinations
   - Assess generalization of SPO benefits

4. **Hybrid Approaches**
   - Combine SPO with conformal prediction
   - Ensemble SPO with traditional methods

### Code Modifications

To modify SPO behavior, edit `src/models/deep_learning.py`:

```python
# Change architecture
spo_model = SPOEndToEnd(
    hidden_size=128,      # Increase capacity
    num_layers=3,         # Deeper network
    epochs=100,           # More training
    ...
)

# Adjust cost parameters
spo_model = SPOEndToEnd(
    ordering_cost=5.0,    # Lower ordering cost
    stockout_cost=100.0,  # Higher penalty
    ...
)
```

## Citation

If using this SPO implementation, please cite:

```bibtex
@article{elmachtoub2017smart,
  title={Smart "Predict, then Optimize"},
  author={Elmachtoub, Adam N and Grigas, Paul},
  journal={arXiv preprint arXiv:1710.08005},
  year={2017}
}
```

## Contact & Support

For questions or issues with the SPO baseline:
- Check existing visualizations in `results/expanding_window_spo/`
- Review model training logs
- Compare against traditional baselines in results CSVs

---

**Status**: ✅ Fully implemented and ready to run
**Complexity**: Intermediate (requires understanding of decision-focused learning)
**Runtime**: ~30-60 minutes for full experiment (depends on hardware and epochs)
