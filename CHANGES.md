# Changes: Rolling Window Split + TFT Model

This document summarizes the changes made to implement rolling window cross-validation and add a Temporal Fusion Transformer (TFT) model.

## Summary

- **Rolling Window Split**: Changed from fixed train/test split to rolling window cross-validation for more robust evaluation
- **TFT Model**: Added Temporal Fusion Transformer as a new deep learning model for comparison
- Both changes maintain backward compatibility with existing code

## Key Features

### 1. Rolling Window Cross-Validation

Instead of a single fixed split (train: 2013-2014, cal: 2015-2016, test: 2017), the system now supports:

- Multiple overlapping time windows
- Each window trains on past data and predicts the next month
- Configurable window sizes and step sizes
- Aggregated results across all windows for robust evaluation

**Default Configuration:**
- Initial training: 730 days (2 years)
- Calibration: 365 days (1 year)
- Test window: 30 days (1 month)
- Step size: 30 days (1 month)

### 2. Temporal Fusion Transformer (TFT)

A state-of-the-art deep learning model that combines:
- **Gated Residual Networks (GRN)**: For flexible feature processing
- **LSTM layers**: For local temporal processing
- **Multi-head attention**: For capturing long-range dependencies
- **Quantile outputs**: For probabilistic forecasting
- **Conformal calibration**: For coverage guarantees

## Files Modified

### Core Modules

1. **src/data/loader.py**
   - Added `RollingWindowSplit` dataclass
   - Added `create_rolling_window_splits()` function
   - Added `prepare_rolling_sequence_data()` function
   - Added `load_and_prepare_rolling_data()` convenience function

2. **src/data/__init__.py**
   - Exported new rolling window functions and classes

3. **src/models/deep_learning.py**
   - Added `GatedResidualNetwork` neural network module
   - Added `VariableSelectionNetwork` module (not currently used, for future enhancement)
   - Added `TemporalFusionTransformerNet` PyTorch model
   - Added `TemporalFusionTransformer` forecaster class

4. **src/models/__init__.py**
   - Exported `TemporalFusionTransformer`

### Configuration

5. **configs/config.py**
   - Added `TFTConfig` dataclass with TFT hyperparameters
   - Added `RollingWindowConfig` dataclass with rolling window settings
   - Updated `ExperimentConfig` to include both new configs

### Experiment Scripts

6. **scripts/run_experiment.py**
   - Added TFT import
   - Added TFT to `run_deep_learning_methods()` function
   - Updated `print_key_findings()` to include TFT in DL methods list
   - Updated command-line arguments to set TFT epochs

7. **scripts/run_rolling_window_experiment.py** (NEW)
   - Complete script for running rolling window experiments
   - Runs all models (traditional + DL + TFT) on each window
   - Aggregates results across windows with mean and std
   - Generates comprehensive CSV outputs

### Testing

8. **test_rolling_window_tft.py** (NEW)
   - Quick validation script to verify implementation
   - Tests imports, configuration, and model structure
   - No data dependencies required

9. **CHANGES.md** (THIS FILE)
   - Documentation of all changes

## Usage

### Fixed Split with TFT (Original Mode)

```bash
python scripts/run_experiment.py --epochs 100 --output results/fixed_split
```

This runs the original fixed split experiment but now includes TFT as the 5th deep learning model.

### Rolling Window Mode (New)

```bash
python scripts/run_rolling_window_experiment.py --epochs 50 --output results/rolling_window
```

This runs the rolling window experiment:
- Creates multiple train/cal/test splits by sliding a window through time
- Evaluates all models on each window
- Outputs aggregated statistics across windows

### Quick Test

```bash
python test_rolling_window_tft.py
```

Verifies that all new code imports and initializes correctly.

## Configuration Options

### TFT Model Parameters (in configs/config.py)

```python
@dataclass
class TFTConfig:
    alpha: float = 0.05           # Coverage level (95% intervals)
    hidden_size: int = 64         # Hidden dimension size
    num_heads: int = 4            # Number of attention heads
    num_layers: int = 2           # Number of attention layers
    dropout: float = 0.1          # Dropout rate
    learning_rate: float = 0.001  # Learning rate
    epochs: int = 100             # Training epochs
    batch_size: int = 32          # Batch size
```

### Rolling Window Parameters (in configs/config.py)

```python
@dataclass
class RollingWindowConfig:
    enabled: bool = False             # Enable rolling windows
    initial_train_days: int = 730     # Initial training period (days)
    calibration_days: int = 365       # Calibration period (days)
    test_window_days: int = 30        # Test prediction window (days)
    step_days: int = 30               # Step size for rolling (days)
```

## Model Comparison

The project now includes **10 models total**:

### Traditional Methods (5)
1. Conformal Prediction + CVaR
2. Normal Assumption + CVaR
3. Quantile Regression + CVaR
4. Sample Average Approximation (SAA)
5. Expected Value

### Deep Learning Methods (5)
1. LSTM Quantile Regression
2. Transformer Quantile Regression
3. Deep Ensemble
4. MC Dropout LSTM
5. **Temporal Fusion Transformer (TFT)** ← NEW

## Expected Outputs

### Rolling Window Experiment

1. **rolling_window_all.csv**: Complete results for every method in every window
2. **rolling_window_aggregated.csv**: Mean and std across all windows for each method

### Metrics Reported

- **Inventory Metrics**: Mean Cost, CVaR-90, CVaR-95, Service Level
- **Forecast Metrics**: Coverage, Interval Width, MAE, RMSE, MAPE

## Technical Details

### TFT Architecture

```
Input (batch, seq_len, features)
    ↓
Input Embedding (Linear)
    ↓
Positional Encoding
    ↓
LSTM (local processing)
    ↓
Multi-head Attention + GRN (×num_layers)
    ↓
Global Average Pooling
    ↓
Quantile Head (3 outputs: lower, median, upper)
```

### Rolling Window Flow

```
Data: 2013 ──────────────────────────────────────→ 2017

Window 0: [Train: 730d][Cal: 365d][Test: 30d]
Window 1:     [Train: 730d][Cal: 365d][Test: 30d]
Window 2:         [Train: 730d][Cal: 365d][Test: 30d]
...
```

## Backward Compatibility

- All existing code continues to work unchanged
- Original `run_experiment.py` still uses fixed splits by default
- TFT is automatically included in regular experiments
- Rolling windows only activate when using `run_rolling_window_experiment.py`

## Performance Notes

- TFT is more computationally intensive than LSTM/Transformer
- Rolling window experiments take N×(time per window) where N = number of windows
- Recommended: Start with `--epochs 50` for rolling windows to manage runtime
- For quick testing: Use `--epochs 10` with `--output results/test`

## Future Enhancements

Potential improvements for future work:

1. Use `VariableSelectionNetwork` for interpretability
2. Add attention weights visualization
3. Implement hyperparameter tuning for TFT
4. Support multi-step ahead forecasting (beyond 30 days)
5. Add parallel processing for rolling window experiments
6. Include forecast horizon as a tunable parameter

## References

- **TFT Paper**: Lim et al. (2021) "Temporal Fusion Transformers for Interpretable Multi-horizon Time Series Forecasting"
- **Conformal Prediction**: Vovk et al. (2005)
- **CVaR Optimization**: Rockafellar & Uryasev (2000)
