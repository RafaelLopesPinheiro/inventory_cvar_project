"""
Unit tests for forecasting models.
"""

import pytest
import numpy as np
import sys
import os

# Add project to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.models import (
    ConformalPrediction,
    NormalAssumption,
    QuantileRegression,
    SampleAverageApproximation,
    ExpectedValue,
    PredictionResult,
)


@pytest.fixture
def sample_data():
    """Generate sample data for testing."""
    np.random.seed(42)
    n_train = 200
    n_cal = 100
    n_test = 50
    n_features = 5
    
    X_train = np.random.randn(n_train, n_features)
    y_train = np.random.poisson(20, n_train).astype(float)
    
    X_cal = np.random.randn(n_cal, n_features)
    y_cal = np.random.poisson(20, n_cal).astype(float)
    
    X_test = np.random.randn(n_test, n_features)
    y_test = np.random.poisson(20, n_test).astype(float)
    
    return X_train, y_train, X_cal, y_cal, X_test, y_test


class TestConformalPrediction:
    """Tests for ConformalPrediction model."""
    
    def test_fit(self, sample_data):
        X_train, y_train, X_cal, y_cal, _, _ = sample_data
        
        model = ConformalPrediction(alpha=0.05)
        model.fit(X_train, y_train, X_cal, y_cal)
        
        assert model.is_fitted
        assert model.q_conformal is not None
        assert model.q_conformal > 0
    
    def test_predict(self, sample_data):
        X_train, y_train, X_cal, y_cal, X_test, _ = sample_data
        
        model = ConformalPrediction(alpha=0.05)
        model.fit(X_train, y_train, X_cal, y_cal)
        
        predictions = model.predict(X_test)
        
        assert isinstance(predictions, PredictionResult)
        assert len(predictions.point) == len(X_test)
        assert predictions.has_intervals
        assert all(predictions.lower <= predictions.upper)
    
    def test_coverage(self, sample_data):
        X_train, y_train, X_cal, y_cal, X_test, y_test = sample_data
        
        model = ConformalPrediction(alpha=0.05)
        model.fit(X_train, y_train, X_cal, y_cal)
        
        predictions = model.predict(X_test)
        
        # Check coverage (should be close to 95%)
        coverage = np.mean((y_test >= predictions.lower) & (y_test <= predictions.upper))
        assert coverage >= 0.80  # Allow some slack for small sample


class TestNormalAssumption:
    """Tests for NormalAssumption model."""
    
    def test_fit(self, sample_data):
        X_train, y_train, X_cal, y_cal, _, _ = sample_data
        
        model = NormalAssumption(alpha=0.05)
        model.fit(X_train, y_train, X_cal, y_cal)
        
        assert model.is_fitted
        assert model.sigma is not None
        assert model.sigma > 0
    
    def test_predict(self, sample_data):
        X_train, y_train, X_cal, y_cal, X_test, _ = sample_data
        
        model = NormalAssumption(alpha=0.05)
        model.fit(X_train, y_train, X_cal, y_cal)
        
        predictions = model.predict(X_test)
        
        assert isinstance(predictions, PredictionResult)
        assert len(predictions.point) == len(X_test)
        assert predictions.has_intervals


class TestQuantileRegression:
    """Tests for QuantileRegression model."""
    
    def test_fit(self, sample_data):
        X_train, y_train, X_cal, y_cal, _, _ = sample_data
        
        model = QuantileRegression(alpha=0.05)
        model.fit(X_train, y_train, X_cal, y_cal)
        
        assert model.is_fitted
        assert model.calibration_adjustment >= 0
    
    def test_predict(self, sample_data):
        X_train, y_train, X_cal, y_cal, X_test, _ = sample_data
        
        model = QuantileRegression(alpha=0.05)
        model.fit(X_train, y_train, X_cal, y_cal)
        
        predictions = model.predict(X_test)
        
        assert isinstance(predictions, PredictionResult)
        assert predictions.has_intervals


class TestSAA:
    """Tests for SampleAverageApproximation model."""
    
    def test_fit(self, sample_data):
        X_train, y_train, X_cal, y_cal, _, _ = sample_data
        
        model = SampleAverageApproximation()
        model.fit(X_train, y_train, X_cal, y_cal)
        
        assert model.is_fitted
        assert model.residuals is not None
        assert len(model.residuals) == len(y_cal)
    
    def test_predict(self, sample_data):
        X_train, y_train, X_cal, y_cal, X_test, _ = sample_data
        
        model = SampleAverageApproximation()
        model.fit(X_train, y_train, X_cal, y_cal)
        
        predictions = model.predict(X_test)
        
        assert isinstance(predictions, PredictionResult)
        assert len(predictions.point) == len(X_test)
        assert not predictions.has_intervals  # SAA doesn't provide intervals
    
    def test_compute_order_quantities(self, sample_data):
        X_train, y_train, X_cal, y_cal, X_test, _ = sample_data
        
        model = SampleAverageApproximation()
        model.fit(X_train, y_train, X_cal, y_cal)
        
        orders = model.compute_order_quantities(X_test)
        
        assert len(orders) == len(X_test)
        assert all(orders >= 0)


class TestExpectedValue:
    """Tests for ExpectedValue model."""
    
    def test_fit_predict(self, sample_data):
        X_train, y_train, X_cal, y_cal, X_test, _ = sample_data
        
        model = ExpectedValue()
        model.fit(X_train, y_train, X_cal, y_cal)
        
        predictions = model.predict(X_test)
        
        assert isinstance(predictions, PredictionResult)
        assert len(predictions.point) == len(X_test)
        assert not predictions.has_intervals


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
