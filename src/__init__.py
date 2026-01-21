"""
Inventory CVaR Optimization Package

A comprehensive toolkit for demand forecasting with uncertainty quantification
and CVaR-optimal inventory decisions.

Modules:
- data: Data loading and preprocessing
- models: Traditional and deep learning forecasting models
- optimization: CVaR optimization for inventory decisions
- evaluation: Metrics and statistical testing
- visualization: Plotting utilities
"""

__version__ = "1.0.0"
__author__ = "Rafael Lopes"

from . import data
from . import models
from . import optimization
from . import evaluation
from . import visualization
