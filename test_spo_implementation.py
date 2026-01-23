#!/usr/bin/env python
"""
Quick test to verify SPO/End-to-End implementation.

This script tests the SPO model on synthetic data to ensure it works correctly.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np

print("="*70)
print("SPO/END-TO-END BASELINE - IMPLEMENTATION TEST")
print("="*70)

# Test 1: Import check
print("\n[1/4] Testing imports...")
try:
    from src.models import SPOEndToEnd, PredictionResult
    print("✓ SPOEndToEnd imported successfully")
except ImportError as e:
    print(f"✗ Import failed: {e}")
    print("\nPlease install dependencies:")
    print("  pip install numpy torch")
    sys.exit(1)

# Test 2: Model initialization
print("\n[2/4] Testing model initialization...")
try:
    model = SPOEndToEnd(
        alpha=0.05,
        sequence_length=28,
        hidden_size=32,  # Smaller for testing
        num_layers=2,
        epochs=5,  # Few epochs for testing
        batch_size=16,
        ordering_cost=10.0,
        holding_cost=2.0,
        stockout_cost=50.0,
        beta=0.90,
        random_state=42,
        device='cpu'
    )
    print(f"✓ SPOEndToEnd initialized successfully")
    print(f"  Architecture: {model.num_layers} layers, {model.hidden_size} hidden units")
    print(f"  Cost params: order=${model.ordering_cost}, hold=${model.holding_cost}, stockout=${model.stockout_cost}")
except Exception as e:
    print(f"✗ Initialization failed: {e}")
    sys.exit(1)

# Test 3: Synthetic data generation
print("\n[3/4] Generating synthetic data...")
try:
    np.random.seed(42)

    # Generate synthetic time series
    n_samples = 200
    seq_length = 28
    n_features = 10

    # Training data
    X_train = np.random.randn(n_samples, seq_length, n_features).astype(np.float32)
    y_train = np.random.uniform(20, 100, n_samples).astype(np.float32)

    # Calibration data
    X_cal = np.random.randn(50, seq_length, n_features).astype(np.float32)
    y_cal = np.random.uniform(20, 100, 50).astype(np.float32)

    # Test data
    X_test = np.random.randn(30, seq_length, n_features).astype(np.float32)
    y_test = np.random.uniform(20, 100, 30).astype(np.float32)

    print(f"✓ Generated synthetic data")
    print(f"  Train: {X_train.shape}, {y_train.shape}")
    print(f"  Cal: {X_cal.shape}, {y_cal.shape}")
    print(f"  Test: {X_test.shape}, {y_test.shape}")
except Exception as e:
    print(f"✗ Data generation failed: {e}")
    sys.exit(1)

# Test 4: Model training and prediction
print("\n[4/4] Testing model training and prediction...")
try:
    print("Training SPO model (this may take a minute)...")
    model.fit(X_train, y_train, X_cal, y_cal)
    print("✓ Model training completed")

    print("Making predictions...")
    predictions = model.predict(X_test)
    print(f"✓ Predictions generated")
    print(f"  Point predictions: {predictions.point.shape}")
    print(f"  Lower bounds: {predictions.lower.shape}")
    print(f"  Upper bounds: {predictions.upper.shape}")
    print(f"  Has intervals: {predictions.has_intervals}")

    # Verify prediction structure
    assert len(predictions.point) == len(y_test), "Point predictions length mismatch"
    assert len(predictions.lower) == len(y_test), "Lower bounds length mismatch"
    assert len(predictions.upper) == len(y_test), "Upper bounds length mismatch"
    assert predictions.has_intervals, "Prediction intervals missing"

    # Verify prediction intervals are valid
    assert np.all(predictions.lower <= predictions.upper), "Invalid intervals: lower > upper"

    print("✓ All validation checks passed")

    # Print sample results
    print("\nSample predictions (first 5):")
    print(f"{'Actual':<10} {'Predicted':<12} {'Lower':<10} {'Upper':<10} {'Width':<10}")
    print("-" * 60)
    for i in range(min(5, len(y_test))):
        width = predictions.upper[i] - predictions.lower[i]
        print(f"{y_test[i]:<10.2f} {predictions.point[i]:<12.2f} "
              f"{predictions.lower[i]:<10.2f} {predictions.upper[i]:<10.2f} "
              f"{width:<10.2f}")

except Exception as e:
    print(f"✗ Training/prediction failed: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# Success!
print("\n" + "="*70)
print("✅ ALL TESTS PASSED!")
print("="*70)
print("\nSPO/End-to-End implementation is working correctly.")
print("\nNext steps:")
print("  1. Run full experiment: python scripts/run_expanding_window_experiment.py")
print("  2. Generate visualizations: python scripts/visualize_expanding_spo_results.py")
print("  3. Review documentation: cat SPO_BASELINE_README.md")
print("="*70 + "\n")
