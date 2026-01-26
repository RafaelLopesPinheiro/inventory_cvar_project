# Seer (Perfect Foresight Oracle) Baseline

## Overview

The **Seer** is a perfect foresight oracle that represents the **theoretical upper bound** of performance in inventory optimization. It's not a practical forecasting model - it's a benchmark that shows what's achievable with perfect knowledge of future demand.

## What Makes Seer Special?

### Perfect Knowledge
- **"Predicts"** by using actual future demand values
- Zero prediction error (MAE = 0, RMSE = 0)
- Perfect certainty (zero-width prediction intervals)

### Optimal Decisions
- Orders exactly the demand that will occur
- No excess inventory (holding cost = $0)
- No stockouts (shortage cost = $0)
- Only pays ordering costs

### Theoretical Limit
Shows the **best possible performance** achievable with:
- Perfect forecasting
- Optimal decision-making
- No uncertainty

## Why Add the Seer?

### 1. **Performance Ceiling**
Answers: "How much better could we possibly do?"

Example:
```
Seer CVaR-90:     $110
Best Model:       $132
Gap:              20%

→ There's room for 20% improvement with better forecasting
```

### 2. **Context for Improvements**
Helps interpret whether a 5% improvement is:
- **Significant**: If gap to Seer is 3%, we're near the limit
- **Trivial**: If gap to Seer is 50%, plenty of room exists

### 3. **Value of Information**
Quantifies the **value of perfect information**:
- Cost with best model - Cost with Seer = VPI
- Shows maximum ROI from better forecasting

## How Seer Works

### Mathematical Formulation

**Traditional Models:**
```
1. Predict: ŷ ≈ y (minimize MSE/MAE)
2. Optimize: q* = argmin E[Loss(q, ŷ)]
```

**Seer Oracle:**
```
1. "Predict": ŷ = y (perfect knowledge)
2. Optimize: q* = y (order exactly the demand)
3. Cost = ordering_cost × y (minimal cost)
```

### Implementation

```python
class Seer(BaseForecaster):
    def predict_with_actuals(self, X, y_actual):
        # Perfect predictions = actual demand
        point_pred = y_actual.copy()

        # Zero-width intervals (perfect certainty)
        lower = y_actual.copy()
        upper = y_actual.copy()

        return PredictionResult(
            point=point_pred,
            lower=lower,
            upper=upper
        )

    def compute_order_quantities(self, y_actual):
        # With perfect foresight, order = demand
        return y_actual.copy()
```

## Usage in Experiments

### Running Experiments with Seer

```bash
# Run expanding window experiment (includes Seer)
python scripts/run_expanding_window_experiment.py --epochs 50
```

The Seer will be automatically included as one of the baselines.

### Interpreting Results

**Results CSV:**
```
Method,Mean_Cost,CVaR-90,MAE
Seer_Oracle,110.50,125.30,0.00
SPO_EndToEnd,132.40,158.20,12.50
Conformal_CVaR,135.80,162.40,14.20
...
```

**Key Metrics:**
- **Mean_Cost**: Minimum achievable (ordering cost only)
- **CVaR-90**: Best possible tail risk
- **MAE**: Always 0.00 (perfect predictions)

### Visualization Features

The Seer is highlighted across all visualizations:

1. **Time Series Plots**
   - Gold color with star markers (⭐)
   - Thickest line to stand out

2. **Bar Charts**
   - Gold bars with diagonal hatching (///)
   - Shows the performance ceiling

3. **Rankings**
   - Always #1 (by definition)
   - Highlighted with gold background

4. **Summary Table**
   - Light gold background
   - Italic text to indicate "oracle"

## Practical Insights

### Interpreting the Gap

**Small Gap (< 10%)**
```
Best Model CVaR-90:  $118
Seer CVaR-90:        $110
Gap:                 7.3%

→ We're close to optimal!
→ Limited room for improvement
→ Focus on robustness, not accuracy
```

**Large Gap (> 30%)**
```
Best Model CVaR-90:  $150
Seer CVaR-90:        $110
Gap:                 36%

→ Significant room for improvement
→ Better forecasting could reduce costs by ~$40
→ Investment in better models is justified
```

### Value of Perfect Information (VPI)

VPI = Cost(Best Model) - Cost(Seer)

**Example:**
```
Mean Cost (Current):  $135
Mean Cost (Seer):     $110
VPI:                  $25 per period

Annual Savings (365 days): $25 × 365 = $9,125
```

This is the **maximum** you could save with perfect forecasting.

### Decision Framework

Use the Seer to guide investment decisions:

| Gap to Seer | Recommendation |
|-------------|----------------|
| < 5% | Near optimal - focus on other issues |
| 5-15% | Good performance - incremental improvements |
| 15-30% | Moderate gap - consider better models |
| > 30% | Large gap - significant ROI opportunity |

## Technical Details

### Cost Breakdown with Seer

**Newsvendor Cost Components:**
```
Total Cost = Ordering + Holding + Stockout

With Seer:
- Ordering = ordering_cost × demand    ✓ (unavoidable)
- Holding = holding_cost × max(0, q-d) = 0  (q = d exactly)
- Stockout = stockout_cost × max(0, d-q) = 0  (q = d exactly)
```

**Result:** Minimum achievable cost = ordering cost only

### Why Intervals Have Zero Width

**Traditional Model:**
```
Lower: point_pred - calibration_adjustment
Upper: point_pred + calibration_adjustment
Width: 2 × calibration_adjustment > 0
```

**Seer:**
```
Lower: actual_demand
Upper: actual_demand
Width: 0 (perfect certainty)
```

No uncertainty → No interval width

### Statistical Properties

- **Unbiased**: E[prediction error] = 0
- **Minimum Variance**: Var(prediction error) = 0
- **Perfect Calibration**: All predictions within interval
- **Perfect Sharpness**: Interval width = 0

## Limitations & Caveats

### ⚠️ Not a Real Model

The Seer is **not** a practical forecasting method:
- ✗ Cannot be deployed in production
- ✗ Requires future knowledge (impossible)
- ✗ Only useful as a benchmark

### ⚠️ Theoretical, Not Achievable

Real models will **always** have a gap to Seer:
- Demand is inherently uncertain
- Perfect forecasting is impossible
- Some costs are unavoidable

### ⚠️ Cost Structure Dependent

The Seer's advantage depends on:
- **High stockout costs**: Larger gap to imperfect models
- **Low holding costs**: Less penalty for over-ordering
- **Cost symmetry**: Gap may be smaller

## Comparison with Other Baselines

| Baseline | Purpose | Gap to Seer |
|----------|---------|-------------|
| **Seer** | Theoretical optimum | 0% (by definition) |
| **SPO** | Decision-focused learning | Small (~5-15%) |
| **Conformal** | Robust intervals | Moderate (~10-20%) |
| **LSTM-QR** | Deep learning | Moderate (~15-25%) |
| **Expected Value** | Simple baseline | Large (~30-50%) |

## Testing

Run the automated test:

```bash
python test_seer_implementation.py
```

**Expected Output:**
```
✅ ALL TESTS PASSED!

Key Properties:
  ✓ Perfect predictions (point = actual demand)
  ✓ Zero-width prediction intervals (perfect certainty)
  ✓ Optimal order quantities (= actual demand)
  ✓ Minimum achievable cost (ordering cost only)
```

## Example Use Cases

### 1. Model Selection

**Question:** Should I use LSTM or Transformer?

**Answer with Seer:**
```
Seer CVaR-90:        $110
LSTM CVaR-90:        $128 (gap: 16%)
Transformer CVaR-90: $132 (gap: 20%)

→ LSTM is closer to optimal
→ Choose LSTM
```

### 2. ROI Analysis

**Question:** Is investing in better forecasting worth it?

**Answer with Seer:**
```
Current Model Cost:  $140/day
Seer Cost:           $110/day
Max Savings:         $30/day = $10,950/year

Investment Budget:   $50,000
Payback Period:      4.6 years

→ If improvements get you halfway to Seer ($5,475/year)
→ Payback in 9.1 years - may not be worth it
```

### 3. Performance Benchmarking

**Question:** How good is our 10% improvement?

**Answer with Seer:**
```
Old Model:           $150 (gap to Seer: 36%)
New Model:           $135 (gap to Seer: 23%)
Improvement:         10%

→ Closed 13% of the gap to optimal
→ Still 23% room for improvement
→ Good progress, but not at the limit
```

## Research Applications

### Academic Studies

Use Seer to:
1. **Bound Performance**: Report gaps to oracle in papers
2. **Compare Methods**: Fair comparison across studies
3. **Value Decomposition**: Separate forecast vs decision quality

### Practical Deployment

Use Seer to:
1. **Set Expectations**: Communicate realistic improvements
2. **Budget Planning**: Quantify maximum ROI
3. **Model Evaluation**: Assess if model is "good enough"

## Summary

The Seer (Perfect Foresight Oracle) provides:

✅ **Theoretical upper bound** of performance
✅ **Context** for interpreting improvements
✅ **Value of information** quantification
✅ **Decision framework** for investments

⚠️ **Not** a practical model
⚠️ **Not** achievable in reality
⚠️ Only useful as a **benchmark**

Use the Seer to answer:
- "How much better could we do?"
- "Is this improvement significant?"
- "Is investment in forecasting justified?"

---

**Implementation**: Fully integrated in experiments
**Visualization**: Highlighted in gold (⭐) across all plots
**Testing**: Automated test script available
**Status**: ✅ Ready to use
