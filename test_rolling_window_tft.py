#!/usr/bin/env python
"""
Quick test script to verify rolling window splits and TFT model work.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

print("Testing rolling window splits and TFT model...")
print("=" * 70)

# Test 1: Import all new modules
print("\n[TEST 1] Importing modules...")
try:
    from src.data import (
        load_and_prepare_rolling_data,
        RollingWindowSplit,
        prepare_rolling_sequence_data
    )
    from src.models import TemporalFusionTransformer
    from configs import get_default_config
    print("✓ All imports successful")
except Exception as e:
    print(f"✗ Import failed: {e}")
    sys.exit(1)

# Test 2: Check config has TFT and rolling window settings
print("\n[TEST 2] Checking configuration...")
try:
    config = get_default_config()
    assert hasattr(config, 'tft'), "Missing TFT config"
    assert hasattr(config, 'rolling_window'), "Missing rolling_window config"
    print(f"✓ TFT config: hidden_size={config.tft.hidden_size}, num_heads={config.tft.num_heads}")
    print(f"✓ Rolling window config: enabled={config.rolling_window.enabled}")
except Exception as e:
    print(f"✗ Config check failed: {e}")
    sys.exit(1)

# Test 3: Verify rolling window data loading (dry run)
print("\n[TEST 3] Testing rolling window data structure...")
try:
    # Just check if the function exists and has correct signature
    import inspect
    sig = inspect.signature(load_and_prepare_rolling_data)
    params = list(sig.parameters.keys())
    expected_params = ['filepath', 'store_id', 'item_id', 'lag_periods',
                      'rolling_windows', 'initial_train_days', 'calibration_days',
                      'test_window_days', 'step_days']
    for param in expected_params:
        assert param in params, f"Missing parameter: {param}"
    print(f"✓ Rolling window function has correct signature")
except Exception as e:
    print(f"✗ Function signature check failed: {e}")
    sys.exit(1)

# Test 4: Verify TFT model structure
print("\n[TEST 4] Testing TFT model structure...")
try:
    import torch
    # Create a dummy TFT model
    tft = TemporalFusionTransformer(
        alpha=0.05,
        sequence_length=28,
        hidden_size=32,
        num_heads=4,
        num_layers=1,
        dropout=0.1,
        learning_rate=0.001,
        epochs=1,
        batch_size=16,
        random_state=42,
        device="cpu"
    )
    print(f"✓ TFT model created successfully")
    print(f"  - Hidden size: {tft.hidden_size}")
    print(f"  - Num heads: {tft.num_heads}")
    print(f"  - Num layers: {tft.num_layers}")
except Exception as e:
    print(f"✗ TFT model creation failed: {e}")
    sys.exit(1)

# Test 5: Check if data file exists
print("\n[TEST 5] Checking data file...")
try:
    data_path = "train.csv"
    if os.path.exists(data_path):
        file_size = os.path.getsize(data_path) / (1024 * 1024)  # MB
        print(f"✓ Data file found: {data_path} ({file_size:.1f} MB)")
    else:
        print(f"⚠ Data file not found at {data_path} (this is OK for testing)")
except Exception as e:
    print(f"⚠ Data check warning: {e}")

print("\n" + "=" * 70)
print("All tests passed! ✓")
print("\nYou can now run:")
print("  1. Fixed split with TFT: python scripts/run_experiment.py --epochs 10")
print("  2. Rolling window: python scripts/run_rolling_window_experiment.py --epochs 10")
print("=" * 70)
