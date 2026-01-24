#!/usr/bin/env python
"""
Quick test to verify Seer (Perfect Foresight Oracle) implementation.

This script tests the Seer model to ensure it works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

print("="*70)
print("SEER (PERFECT FORESIGHT ORACLE) - IMPLEMENTATION TEST")
print("="*70)

# Test 1: Import check
print("\n[1/4] Testing imports...")
try:
    from src.models import Seer, PredictionResult
    print("✓ Seer imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Model initialization
print("\n[2/4] Testing model initialization...")
try:
    seer = Seer(alpha=0.05, random_state=42)
    print(f"✓ Seer initialized successfully")
    print(f"  Type: Oracle/Perfect Foresight Baseline")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Synthetic data generation
print("\n[3/4] Generating synthetic data...")
try:
    np.random.seed(42)

    # Generate synthetic time series
    n_train = 100
    n_cal = 50
    n_test = 30
    n_features = 10

    # Training data
    X_train = np.random.randn(n_train, n_features).astype(np.float32)
    y_train = np.random.uniform(20, 100, n_train).astype(np.float32)

    # Calibration data
    X_cal = np.random.randn(n_cal, n_features).astype(np.float32)
    y_cal = np.random.uniform(20, 100, n_cal).astype(np.float32)

    # Test data
    X_test = np.random.randn(n_test, n_features).astype(np.float32)
    y_test = np.random.uniform(20, 100, n_test).astype(np.float32)

    print(f"✓ Generated synthetic data")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Cal: {X_cal.shape}, {y_cal.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")
except Exception as e:
    print(f"✗ Data generation failed: {e}")
    sys.exit(1)

# Test 4: Model "training" and prediction
print("\n[4/4] Testing model 'training' and prediction...")
try:
    print("'Training' Seer model (no actual training needed)...")
    seer.fit(X_train, y_train, X_cal, y_cal)
    print("✓ Model 'training' completed")

    print("Making perfect predictions...")
    predictions = seer.predict_with_actuals(X_test, y_test)
    print(f"✓ Predictions generated")
    print(f"  Point predictions: {predictions.point.shape}")
    print(f"  Lower bounds: {predictions.lower.shape}")
    print(f"  Upper bounds: {predictions.upper.shape}")
    print(f"  Has intervals: {predictions.has_intervals}")

    # Verify predictions are perfect
    assert np.allclose(predictions.point, y_test), "Point predictions should equal actual demand"
    assert np.allclose(predictions.lower, y_test), "Lower bounds should equal actual demand"
    assert np.allclose(predictions.upper, y_test), "Upper bounds should equal actual demand"
    print("✓ Verified: Predictions are perfect (match actual demand)")

    # Verify prediction intervals have zero width
    interval_widths = predictions.upper - predictions.lower
    assert np.allclose(interval_widths, 0), "Interval widths should be zero (perfect certainty)"
    print("✓ Verified: Prediction intervals have zero width (perfect certainty)")

    # Test order quantities
    print("\nTesting optimal order quantities...")
    order_quantities = seer.compute_order_quantities(
        y_test,
        ordering_cost=10.0,
        holding_cost=2.0,
        stockout_cost=50.0
    )

    # With perfect foresight, order qty = demand
    assert np.allclose(order_quantities, y_test), "Order quantities should equal demand"
    print("✓ Verified: Order quantities equal demand (optimal with perfect foresight)")

    # Print sample results
    print("\nSample predictions (first 5):")
    print(f"{'Actual':<10} {'Predicted':<12} {'Lower':<10} {'Upper':<10} {'Width':<10} {'Order Qty':<10}")
    print("-" * 70)
    for i in range(min(5, len(y_test))):
        width = predictions.upper[i] - predictions.lower[i]
        print(f"{y_test[i]:<10.2f} {predictions.point[i]:<12.2f} "
              f"{predictions.lower[i]:<10.2f} {predictions.upper[i]:<10.2f} "
              f"{width:<10.2f} {order_quantities[i]:<10.2f}")

    # Compute costs with perfect foresight
    print("\nComputing newsvendor costs with perfect foresight...")
    from src.optimization import newsvendor_loss

    costs = newsvendor_loss(
        order_quantities, y_test,
        ordering_cost=10.0,
        holding_cost=2.0,
        stockout_cost=50.0
    )

    mean_cost = np.mean(costs)
    print(f"✓ Mean cost with perfect foresight: ${mean_cost:.2f}")
    print(f"  (This is the theoretical minimum achievable cost)")

    # With perfect foresight, holding and stockout costs should be zero
    # Only ordering costs remain
    expected_cost = 10.0 * np.mean(y_test)
    assert np.isclose(mean_cost, expected_cost, rtol=0.01), "Cost should be ordering cost only"
    print(f"✓ Verified: Cost = ordering_cost * demand = ${expected_cost:.2f}")

except Exception as e:
    print(f"✗ Testing failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nSeer (Perfect Foresight Oracle) implementation is working correctly.")
print("\nKey Properties:")
print("  ✓ Perfect predictions (point = actual demand)")
print("  ✓ Zero-width prediction intervals (perfect certainty)")
print("  ✓ Optimal order quantities (= actual demand)")
print("  ✓ Minimum achievable cost (ordering cost only)")
print("\nThe Seer provides the theoretical upper bound for performance.")
print("Use it to measure the gap between current models and perfection!")
print("="*70 + "\n")
