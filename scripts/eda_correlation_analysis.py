#!/usr/bin/env python
"""
Exploratory Data Analysis: Correlation Impact on Inventory Models

This script analyzes various types of correlations in demand data and their
potential impact on forecasting and inventory optimization models.

Analysis includes:
1. Temporal autocorrelation (ACF/PACF)
2. Cross-SKU demand correlations
3. Feature-target correlations
4. Seasonality and trend decomposition
5. Heteroscedasticity detection
6. Stationarity tests
7. Distribution analysis
8. Residual diagnostics (post-model)

Usage:
    python eda_correlation_analysis.py --data train.csv --output results/eda
"""

import argparse
import logging
import os
import sys
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.signal import periodogram

# Statistical tests
from statsmodels.tsa.stattools import adfuller, kpss, acf, pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.stats.diagnostic import acorr_ljungbox, het_breuschpagan
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.stats.stattools import durbin_watson

# Add src to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.data import load_raw_data, filter_store_item, create_all_features

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Set plot style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


# =============================================================================
# 1. TEMPORAL AUTOCORRELATION ANALYSIS
# =============================================================================

def analyze_autocorrelation(
    series: pd.Series,
    max_lags: int = 60,
    output_dir: str = "results/eda"
) -> Dict:
    """
    Analyze temporal autocorrelation in demand series.
    
    Impact on models:
    - High autocorrelation → LSTM/Transformer can exploit temporal patterns
    - Violates exchangeability assumption in conformal prediction
    - Affects rolling window validity
    """
    logger.info("Analyzing temporal autocorrelation...")
    
    results = {}
    
    # Compute ACF and PACF
    acf_values = acf(series.dropna(), nlags=max_lags, fft=True)
    pacf_values = pacf(series.dropna(), nlags=max_lags)
    
    # Significant lags (outside 95% CI)
    ci = 1.96 / np.sqrt(len(series))
    significant_acf_lags = np.where(np.abs(acf_values[1:]) > ci)[0] + 1
    significant_pacf_lags = np.where(np.abs(pacf_values[1:]) > ci)[0] + 1
    
    results['acf_values'] = acf_values
    results['pacf_values'] = pacf_values
    results['significant_acf_lags'] = significant_acf_lags[:10].tolist()  # Top 10
    results['significant_pacf_lags'] = significant_pacf_lags[:10].tolist()
    
    # Ljung-Box test for autocorrelation
    lb_test = acorr_ljungbox(series.dropna(), lags=[7, 14, 28], return_df=True)
    results['ljung_box_pvalues'] = lb_test['lb_pvalue'].to_dict()
    results['has_significant_autocorr'] = any(lb_test['lb_pvalue'] < 0.05)
    
    # Durbin-Watson statistic (for lag-1 autocorrelation)
    dw_stat = durbin_watson(series.dropna())
    results['durbin_watson'] = dw_stat
    results['dw_interpretation'] = (
        "Positive autocorr" if dw_stat < 1.5 else
        "No autocorr" if 1.5 <= dw_stat <= 2.5 else
        "Negative autocorr"
    )
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Temporal Autocorrelation Analysis', fontsize=14, fontweight='bold')
    
    # Time series plot
    ax = axes[0, 0]
    ax.plot(series.values[-365:], linewidth=0.8)
    ax.set_title('Demand Time Series (Last 365 Days)')
    ax.set_xlabel('Day')
    ax.set_ylabel('Demand')
    ax.grid(alpha=0.3)
    
    # ACF plot
    ax = axes[0, 1]
    plot_acf(series.dropna(), lags=max_lags, ax=ax, alpha=0.05)
    ax.set_title(f'Autocorrelation Function (ACF)\nSignificant lags: {results["significant_acf_lags"][:5]}')
    
    # PACF plot
    ax = axes[1, 0]
    plot_pacf(series.dropna(), lags=max_lags, ax=ax, alpha=0.05)
    ax.set_title(f'Partial Autocorrelation Function (PACF)\nSignificant lags: {results["significant_pacf_lags"][:5]}')
    
    # Lag plot (scatter of y_t vs y_{t-1})
    ax = axes[1, 1]
    ax.scatter(series.iloc[:-1].values, series.iloc[1:].values, alpha=0.3, s=10)
    ax.set_xlabel('Demand(t)')
    ax.set_ylabel('Demand(t+1)')
    ax.set_title(f'Lag-1 Scatter Plot\nDurbin-Watson: {dw_stat:.3f} ({results["dw_interpretation"]})')
    
    # Add correlation line
    z = np.polyfit(series.iloc[:-1].dropna(), series.iloc[1:].dropna(), 1)
    p = np.poly1d(z)
    x_line = np.linspace(series.min(), series.max(), 100)
    ax.plot(x_line, p(x_line), 'r-', linewidth=2, label=f'ρ={acf_values[1]:.3f}')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'autocorrelation_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: autocorrelation_analysis.png")
    
    return results


# =============================================================================
# 2. CROSS-SKU CORRELATION ANALYSIS
# =============================================================================

def analyze_cross_sku_correlation(
    df_raw: pd.DataFrame,
    store_ids: List[int],
    item_ids: List[int],
    output_dir: str = "results/eda"
) -> Dict:
    """
    Analyze demand correlations across different SKUs.
    
    Impact on models:
    - High cross-correlation → opportunity for transfer learning
    - Suggests hierarchical/grouped forecasting approaches
    - Affects portfolio-level inventory optimization
    """
    logger.info("Analyzing cross-SKU correlations...")
    
    results = {}
    
    # Create pivot table: date × SKU
    demand_matrix = []
    sku_labels = []
    
    for store_id in store_ids:
        for item_id in item_ids:
            df_sku = filter_store_item(df_raw, store_id, item_id)
            if len(df_sku) > 365:
                df_sku = df_sku.set_index('date')['sales']
                demand_matrix.append(df_sku)
                sku_labels.append(f"S{store_id}_I{item_id}")
    
    if len(demand_matrix) < 2:
        logger.warning("Not enough SKUs for cross-correlation analysis")
        return results
    
    # Align all series
    demand_df = pd.concat(demand_matrix, axis=1)
    demand_df.columns = sku_labels
    demand_df = demand_df.dropna()
    
    # Compute correlation matrix
    corr_matrix = demand_df.corr()
    results['correlation_matrix'] = corr_matrix.to_dict()
    
    # Summary statistics
    corr_values = corr_matrix.values[np.triu_indices_from(corr_matrix.values, k=1)]
    results['mean_correlation'] = np.mean(corr_values)
    results['median_correlation'] = np.median(corr_values)
    results['std_correlation'] = np.std(corr_values)
    results['max_correlation'] = np.max(corr_values)
    results['min_correlation'] = np.min(corr_values)
    
    # Identify highly correlated pairs
    high_corr_pairs = []
    for i in range(len(sku_labels)):
        for j in range(i+1, len(sku_labels)):
            if abs(corr_matrix.iloc[i, j]) > 0.7:
                high_corr_pairs.append({
                    'sku1': sku_labels[i],
                    'sku2': sku_labels[j],
                    'correlation': corr_matrix.iloc[i, j]
                })
    results['high_correlation_pairs'] = high_corr_pairs
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Cross-SKU Correlation Analysis', fontsize=14, fontweight='bold')
    
    # Correlation heatmap
    ax = axes[0, 0]
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdYlBu_r',
                center=0, ax=ax, vmin=-1, vmax=1, square=True,
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Demand Correlation Matrix')
    
    # Correlation distribution
    ax = axes[0, 1]
    ax.hist(corr_values, bins=20, edgecolor='black', alpha=0.7)
    ax.axvline(np.mean(corr_values), color='red', linestyle='--', 
               label=f'Mean: {np.mean(corr_values):.3f}')
    ax.axvline(np.median(corr_values), color='green', linestyle='--',
               label=f'Median: {np.median(corr_values):.3f}')
    ax.set_xlabel('Correlation')
    ax.set_ylabel('Frequency')
    ax.set_title('Distribution of Pairwise Correlations')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Demand time series comparison (top correlated pair)
    ax = axes[1, 0]
    if high_corr_pairs:
        top_pair = max(high_corr_pairs, key=lambda x: x['correlation'])
        ax.plot(demand_df[top_pair['sku1']].values[-180:], label=top_pair['sku1'], alpha=0.7)
        ax.plot(demand_df[top_pair['sku2']].values[-180:], label=top_pair['sku2'], alpha=0.7)
        ax.set_title(f"Highest Correlated Pair (ρ={top_pair['correlation']:.3f})")
    else:
        ax.plot(demand_df.iloc[:, 0].values[-180:], label=sku_labels[0], alpha=0.7)
        ax.plot(demand_df.iloc[:, 1].values[-180:], label=sku_labels[1], alpha=0.7)
        ax.set_title("Sample SKU Comparison")
    ax.set_xlabel('Day')
    ax.set_ylabel('Demand')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Clustermap (hierarchical clustering of SKUs)
    ax = axes[1, 1]
    from scipy.cluster.hierarchy import linkage, dendrogram
    from scipy.spatial.distance import squareform
    
    # Convert correlation to distance
    dist_matrix = 1 - np.abs(corr_matrix.values)
    np.fill_diagonal(dist_matrix, 0)
    
    # Hierarchical clustering
    linkage_matrix = linkage(squareform(dist_matrix), method='ward')
    dendrogram(linkage_matrix, labels=sku_labels, ax=ax, leaf_rotation=90)
    ax.set_title('SKU Clustering by Demand Similarity')
    ax.set_ylabel('Distance (1 - |correlation|)')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'cross_sku_correlation.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: cross_sku_correlation.png")
    
    return results


# =============================================================================
# 3. FEATURE-TARGET CORRELATION ANALYSIS
# =============================================================================

def analyze_feature_correlations(
    df: pd.DataFrame,
    feature_cols: List[str],
    target_col: str = 'sales',
    output_dir: str = "results/eda"
) -> Dict:
    """
    Analyze correlations between features and target variable.
    
    Impact on models:
    - High correlation → feature is predictive
    - Multicollinearity → unstable coefficients
    - Low correlation → feature may not help
    """
    logger.info("Analyzing feature-target correlations...")
    
    results = {}
    
    # Feature-target correlations
    correlations = {}
    for col in feature_cols:
        if col in df.columns:
            corr = df[col].corr(df[target_col])
            correlations[col] = corr
    
    results['feature_target_correlations'] = correlations
    
    # Sort by absolute correlation
    sorted_corr = sorted(correlations.items(), key=lambda x: abs(x[1]), reverse=True)
    results['top_features'] = sorted_corr[:10]
    results['bottom_features'] = sorted_corr[-5:]
    
    # Feature-feature correlation matrix (multicollinearity)
    feature_df = df[feature_cols].dropna()
    feature_corr_matrix = feature_df.corr()
    
    # Find highly correlated feature pairs (multicollinearity)
    multicollinear_pairs = []
    for i in range(len(feature_cols)):
        for j in range(i+1, len(feature_cols)):
            if feature_cols[i] in feature_corr_matrix.index and feature_cols[j] in feature_corr_matrix.columns:
                corr_val = feature_corr_matrix.loc[feature_cols[i], feature_cols[j]]
                if abs(corr_val) > 0.8:
                    multicollinear_pairs.append({
                        'feature1': feature_cols[i],
                        'feature2': feature_cols[j],
                        'correlation': corr_val
                    })
    
    results['multicollinear_pairs'] = multicollinear_pairs
    results['has_multicollinearity'] = len(multicollinear_pairs) > 0
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    fig.suptitle('Feature Correlation Analysis', fontsize=14, fontweight='bold')
    
    # Feature-target correlation bar chart
    ax = axes[0, 0]
    sorted_features = [x[0] for x in sorted_corr]
    sorted_values = [x[1] for x in sorted_corr]
    colors = ['green' if v > 0 else 'red' for v in sorted_values]
    ax.barh(range(len(sorted_features)), sorted_values, color=colors, alpha=0.7)
    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=8)
    ax.set_xlabel('Correlation with Demand')
    ax.set_title('Feature-Target Correlations')
    ax.axvline(0, color='black', linewidth=0.5)
    ax.grid(alpha=0.3, axis='x')
    
    # Feature correlation heatmap
    ax = axes[0, 1]
    sns.heatmap(feature_corr_matrix, cmap='RdYlBu_r', center=0, ax=ax,
                xticklabels=True, yticklabels=True,
                cbar_kws={'label': 'Correlation'})
    ax.set_title('Feature-Feature Correlation Matrix')
    plt.setp(ax.get_xticklabels(), rotation=45, ha='right', fontsize=7)
    plt.setp(ax.get_yticklabels(), fontsize=7)
    
    # Scatter plots for top 2 features
    ax = axes[1, 0]
    if len(sorted_corr) >= 1:
        top_feature = sorted_corr[0][0]
        ax.scatter(df[top_feature], df[target_col], alpha=0.3, s=10)
        ax.set_xlabel(top_feature)
        ax.set_ylabel('Demand')
        ax.set_title(f'Top Feature vs Demand (ρ={sorted_corr[0][1]:.3f})')
        
        # Add trend line
        z = np.polyfit(df[top_feature].dropna(), df.loc[df[top_feature].notna(), target_col], 1)
        p = np.poly1d(z)
        x_line = np.linspace(df[top_feature].min(), df[top_feature].max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2)
    ax.grid(alpha=0.3)
    
    # Variance Inflation Factor (VIF) approximation
    ax = axes[1, 1]
    # Compute R² of each feature regressed on others (simplified VIF)
    vif_approx = {}
    for col in feature_cols[:10]:  # Limit to top 10
        if col in feature_df.columns:
            other_cols = [c for c in feature_cols if c != col and c in feature_df.columns]
            if other_cols:
                X = feature_df[other_cols].values
                y = feature_df[col].values
                # Simple correlation-based R²
                r_squared = 1 - (1 / (1 + np.mean([feature_corr_matrix.loc[col, c]**2 
                                                    for c in other_cols if c in feature_corr_matrix.columns])))
                vif_approx[col] = 1 / (1 - min(r_squared, 0.99))
    
    if vif_approx:
        vif_sorted = sorted(vif_approx.items(), key=lambda x: x[1], reverse=True)
        ax.barh([x[0] for x in vif_sorted], [x[1] for x in vif_sorted], alpha=0.7)
        ax.axvline(5, color='orange', linestyle='--', label='VIF=5 (moderate)')
        ax.axvline(10, color='red', linestyle='--', label='VIF=10 (high)')
        ax.set_xlabel('Variance Inflation Factor (approx)')
        ax.set_title('Multicollinearity Detection')
        ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'feature_correlations.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: feature_correlations.png")
    
    return results


# =============================================================================
# 4. SEASONALITY AND TREND ANALYSIS
# =============================================================================

def analyze_seasonality(
    series: pd.Series,
    freq: int = 7,  # Weekly seasonality
    output_dir: str = "results/eda"
) -> Dict:
    """
    Analyze seasonality and trend components.
    
    Impact on models:
    - Strong seasonality → time features crucial
    - Trend → may need differencing or detrending
    - Affects stationarity assumptions
    """
    logger.info("Analyzing seasonality and trend...")
    
    results = {}
    
    # Ensure series has no NaN
    series_clean = series.dropna()
    
    # Seasonal decomposition (additive)
    try:
        decomposition = seasonal_decompose(series_clean, model='additive', period=freq)
        results['seasonal_strength'] = 1 - (np.var(decomposition.resid.dropna()) / 
                                            np.var(decomposition.observed.dropna()))
        results['trend_strength'] = np.var(decomposition.trend.dropna()) / np.var(decomposition.observed.dropna())
    except Exception as e:
        logger.warning(f"Seasonal decomposition failed: {e}")
        decomposition = None
        results['seasonal_strength'] = None
        results['trend_strength'] = None
    
    # Periodogram analysis (frequency domain)
    freqs, power = periodogram(series_clean.values, fs=1.0)
    
    # Find dominant frequencies
    top_freq_idx = np.argsort(power)[-5:][::-1]
    dominant_periods = []
    for idx in top_freq_idx:
        if freqs[idx] > 0:
            period = 1 / freqs[idx]
            if period < len(series_clean) / 2:  # Valid period
                dominant_periods.append({
                    'period': period,
                    'power': power[idx]
                })
    results['dominant_periods'] = dominant_periods[:3]
    
    # Day-of-week effect
    if 'date' not in series.index.names:
        series_with_date = series.copy()
        series_with_date.index = pd.to_datetime(series_with_date.index)
    else:
        series_with_date = series
    
    dow_effect = series_with_date.groupby(series_with_date.index.dayofweek).mean()
    results['day_of_week_effect'] = dow_effect.to_dict()
    results['dow_variation'] = (dow_effect.max() - dow_effect.min()) / dow_effect.mean()
    
    # Month effect
    month_effect = series_with_date.groupby(series_with_date.index.month).mean()
    results['monthly_effect'] = month_effect.to_dict()
    results['monthly_variation'] = (month_effect.max() - month_effect.min()) / month_effect.mean()
    
    # Create visualization
    fig, axes = plt.subplots(3, 2, figsize=(14, 14))
    fig.suptitle('Seasonality and Trend Analysis', fontsize=14, fontweight='bold')
    
    # Original series
    ax = axes[0, 0]
    ax.plot(series_clean.values, linewidth=0.5)
    ax.set_title('Original Time Series')
    ax.set_xlabel('Day')
    ax.set_ylabel('Demand')
    ax.grid(alpha=0.3)
    
    # Seasonal decomposition
    if decomposition is not None:
        ax = axes[0, 1]
        ax.plot(decomposition.trend.values, linewidth=1)
        ax.set_title(f'Trend Component (Strength: {results["trend_strength"]:.3f})')
        ax.set_xlabel('Day')
        ax.grid(alpha=0.3)
        
        ax = axes[1, 0]
        ax.plot(decomposition.seasonal.values[:freq*4], linewidth=1)
        ax.set_title(f'Seasonal Component (Strength: {results["seasonal_strength"]:.3f})')
        ax.set_xlabel('Day')
        ax.grid(alpha=0.3)
        
        ax = axes[1, 1]
        ax.plot(decomposition.resid.values, linewidth=0.5, alpha=0.7)
        ax.set_title('Residual Component')
        ax.set_xlabel('Day')
        ax.grid(alpha=0.3)
    
    # Day-of-week effect
    ax = axes[2, 0]
    days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']
    ax.bar(days, dow_effect.values, alpha=0.7, edgecolor='black')
    ax.axhline(dow_effect.mean(), color='red', linestyle='--', label='Mean')
    ax.set_title(f'Day-of-Week Effect (Variation: {results["dow_variation"]:.1%})')
    ax.set_ylabel('Mean Demand')
    ax.legend()
    ax.grid(alpha=0.3, axis='y')
    
    # Periodogram
    ax = axes[2, 1]
    ax.semilogy(1/freqs[1:50], power[1:50])  # Plot as period, not frequency
    ax.set_xlabel('Period (days)')
    ax.set_ylabel('Power (log scale)')
    ax.set_title('Periodogram (Frequency Analysis)')
    
    # Mark dominant periods
    for p in dominant_periods[:3]:
        ax.axvline(p['period'], color='red', linestyle='--', alpha=0.5)
        ax.annotate(f"{p['period']:.1f}d", (p['period'], ax.get_ylim()[1]*0.5),
                   fontsize=8, color='red')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'seasonality_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: seasonality_analysis.png")
    
    return results


# =============================================================================
# 5. STATIONARITY TESTS
# =============================================================================

def analyze_stationarity(
    series: pd.Series,
    output_dir: str = "results/eda"
) -> Dict:
    """
    Test for stationarity using ADF and KPSS tests.
    
    Impact on models:
    - Non-stationary → need differencing
    - Affects validity of statistical tests
    - Important for conformal prediction assumptions
    """
    logger.info("Testing stationarity...")
    
    results = {}
    series_clean = series.dropna()
    
    # Augmented Dickey-Fuller test
    # H0: Series has unit root (non-stationary)
    adf_result = adfuller(series_clean, autolag='AIC')
    results['adf_statistic'] = adf_result[0]
    results['adf_pvalue'] = adf_result[1]
    results['adf_critical_values'] = adf_result[4]
    results['adf_is_stationary'] = adf_result[1] < 0.05
    
    # KPSS test
    # H0: Series is stationary
    kpss_result = kpss(series_clean, regression='c', nlags='auto')
    results['kpss_statistic'] = kpss_result[0]
    results['kpss_pvalue'] = kpss_result[1]
    results['kpss_critical_values'] = kpss_result[3]
    results['kpss_is_stationary'] = kpss_result[1] > 0.05
    
    # Combined interpretation
    if results['adf_is_stationary'] and results['kpss_is_stationary']:
        results['stationarity_conclusion'] = "Stationary"
    elif not results['adf_is_stationary'] and not results['kpss_is_stationary']:
        results['stationarity_conclusion'] = "Non-stationary"
    else:
        results['stationarity_conclusion'] = "Trend-stationary or difference-stationary"
    
    # Rolling statistics
    window = 30
    rolling_mean = series_clean.rolling(window=window).mean()
    rolling_std = series_clean.rolling(window=window).std()
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Stationarity Analysis - Conclusion: {results["stationarity_conclusion"]}', 
                 fontsize=14, fontweight='bold')
    
    # Original series with rolling statistics
    ax = axes[0, 0]
    ax.plot(series_clean.values, label='Original', alpha=0.5, linewidth=0.5)
    ax.plot(rolling_mean.values, label=f'Rolling Mean ({window}d)', linewidth=2)
    ax.plot(rolling_std.values, label=f'Rolling Std ({window}d)', linewidth=2)
    ax.set_title('Series with Rolling Statistics')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Distribution over time (split into quarters)
    ax = axes[0, 1]
    n_quarters = 4
    quarter_size = len(series_clean) // n_quarters
    for i in range(n_quarters):
        start = i * quarter_size
        end = (i + 1) * quarter_size
        quarter_data = series_clean.iloc[start:end]
        ax.hist(quarter_data, bins=30, alpha=0.5, label=f'Q{i+1}', density=True)
    ax.set_title('Distribution by Time Period')
    ax.set_xlabel('Demand')
    ax.set_ylabel('Density')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Test results summary
    ax = axes[1, 0]
    ax.axis('off')
    
    summary_text = f"""
    STATIONARITY TEST RESULTS
    ========================
    
    Augmented Dickey-Fuller Test:
    - H0: Series has unit root (non-stationary)
    - Test Statistic: {results['adf_statistic']:.4f}
    - p-value: {results['adf_pvalue']:.4f}
    - Critical Values:
        1%: {results['adf_critical_values']['1%']:.4f}
        5%: {results['adf_critical_values']['5%']:.4f}
        10%: {results['adf_critical_values']['10%']:.4f}
    - Conclusion: {'Reject H0 (Stationary)' if results['adf_is_stationary'] else 'Fail to reject H0 (Non-stationary)'}
    
    KPSS Test:
    - H0: Series is stationary
    - Test Statistic: {results['kpss_statistic']:.4f}
    - p-value: {results['kpss_pvalue']:.4f}
    - Conclusion: {'Fail to reject H0 (Stationary)' if results['kpss_is_stationary'] else 'Reject H0 (Non-stationary)'}
    
    OVERALL: {results['stationarity_conclusion']}
    """
    ax.text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
            verticalalignment='center', transform=ax.transAxes)
    
    # First difference (if non-stationary)
    ax = axes[1, 1]
    diff_series = series_clean.diff().dropna()
    ax.plot(diff_series.values, linewidth=0.5)
    ax.set_title('First Difference of Series')
    ax.set_xlabel('Day')
    ax.set_ylabel('Δ Demand')
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'stationarity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: stationarity_analysis.png")
    
    return results


# =============================================================================
# 6. HETEROSCEDASTICITY ANALYSIS
# =============================================================================

def analyze_heteroscedasticity(
    series: pd.Series,
    output_dir: str = "results/eda"
) -> Dict:
    """
    Test for heteroscedasticity (non-constant variance).
    
    Impact on models:
    - Heteroscedasticity → prediction intervals should vary
    - Fixed-width conformal intervals are suboptimal
    - Consider quantile regression or adaptive methods
    """
    logger.info("Analyzing heteroscedasticity...")
    
    results = {}
    series_clean = series.dropna()
    
    # Rolling variance analysis
    window = 30
    rolling_var = series_clean.rolling(window=window).var()
    
    # Correlation between level and variance
    rolling_mean = series_clean.rolling(window=window).mean()
    level_var_corr = rolling_mean.corr(rolling_var)
    results['level_variance_correlation'] = level_var_corr
    
    # Simple heteroscedasticity test
    # Regress squared residuals on level
    residuals = series_clean - series_clean.mean()
    squared_residuals = residuals ** 2
    
    # Breusch-Pagan-like test (simplified)
    from scipy.stats import pearsonr
    corr, pvalue = pearsonr(series_clean.values[:-1], squared_residuals.values[1:])
    results['bp_correlation'] = corr
    results['bp_pvalue'] = pvalue
    results['has_heteroscedasticity'] = pvalue < 0.05
    
    # Coefficient of variation by quantile
    n_quantiles = 4
    quantile_cv = {}
    for i in range(n_quantiles):
        q_low = series_clean.quantile(i / n_quantiles)
        q_high = series_clean.quantile((i + 1) / n_quantiles)
        mask = (series_clean >= q_low) & (series_clean < q_high)
        subset = series_clean[mask]
        if len(subset) > 10:
            cv = subset.std() / subset.mean() if subset.mean() > 0 else np.nan
            quantile_cv[f'Q{i+1}'] = cv
    results['quantile_cv'] = quantile_cv
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'Heteroscedasticity Analysis - Detected: {results["has_heteroscedasticity"]}', 
                 fontsize=14, fontweight='bold')
    
    # Rolling variance over time
    ax = axes[0, 0]
    ax.plot(rolling_var.values, linewidth=1)
    ax.set_title('Rolling Variance (30-day window)')
    ax.set_xlabel('Day')
    ax.set_ylabel('Variance')
    ax.grid(alpha=0.3)
    
    # Level vs Variance scatter
    ax = axes[0, 1]
    ax.scatter(rolling_mean.values, rolling_var.values, alpha=0.3, s=10)
    ax.set_xlabel('Rolling Mean')
    ax.set_ylabel('Rolling Variance')
    ax.set_title(f'Level vs Variance (ρ={level_var_corr:.3f})')
    ax.grid(alpha=0.3)
    
    # Add trend line if correlation is significant
    if abs(level_var_corr) > 0.3:
        valid_idx = ~(np.isnan(rolling_mean) | np.isnan(rolling_var))
        z = np.polyfit(rolling_mean[valid_idx], rolling_var[valid_idx], 1)
        p = np.poly1d(z)
        x_line = np.linspace(rolling_mean.min(), rolling_mean.max(), 100)
        ax.plot(x_line, p(x_line), 'r-', linewidth=2)
    
    # Residual squared vs fitted
    ax = axes[1, 0]
    ax.scatter(series_clean.values[:-1], squared_residuals.values[1:], alpha=0.3, s=10)
    ax.set_xlabel('Demand Level')
    ax.set_ylabel('Squared Residuals')
    ax.set_title(f'Breusch-Pagan Test (p={results["bp_pvalue"]:.4f})')
    ax.grid(alpha=0.3)
    
    # CV by quantile
    ax = axes[1, 1]
    if quantile_cv:
        ax.bar(quantile_cv.keys(), quantile_cv.values(), alpha=0.7, edgecolor='black')
        ax.set_xlabel('Demand Quantile')
        ax.set_ylabel('Coefficient of Variation')
        ax.set_title('Variability by Demand Level')
        ax.grid(alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'heteroscedasticity_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: heteroscedasticity_analysis.png")
    
    return results


# =============================================================================
# 7. DISTRIBUTION ANALYSIS
# =============================================================================

def analyze_distribution(
    series: pd.Series,
    output_dir: str = "results/eda"
) -> Dict:
    """
    Analyze the distribution of demand.
    
    Impact on models:
    - Normal → Gaussian assumption valid
    - Heavy tails → need robust methods, CVaR important
    - Skewness → affects quantile estimation
    """
    logger.info("Analyzing demand distribution...")
    
    results = {}
    series_clean = series.dropna()
    
    # Basic statistics
    results['mean'] = series_clean.mean()
    results['median'] = series_clean.median()
    results['std'] = series_clean.std()
    results['skewness'] = series_clean.skew()
    results['kurtosis'] = series_clean.kurtosis()
    results['min'] = series_clean.min()
    results['max'] = series_clean.max()
    
    # Percentiles
    percentiles = [5, 10, 25, 50, 75, 90, 95, 99]
    results['percentiles'] = {p: series_clean.quantile(p/100) for p in percentiles}
    
    # Normality tests
    # Shapiro-Wilk (for small samples)
    if len(series_clean) <= 5000:
        shapiro_stat, shapiro_p = stats.shapiro(series_clean.sample(min(5000, len(series_clean))))
    else:
        shapiro_stat, shapiro_p = stats.shapiro(series_clean.sample(5000))
    results['shapiro_statistic'] = shapiro_stat
    results['shapiro_pvalue'] = shapiro_p
    
    # Jarque-Bera test
    jb_stat, jb_p = stats.jarque_bera(series_clean)
    results['jarque_bera_statistic'] = jb_stat
    results['jarque_bera_pvalue'] = jb_p
    
    results['is_normal'] = shapiro_p > 0.05 and jb_p > 0.05
    
    # Fit distributions
    distributions = ['norm', 'lognorm', 'gamma', 'expon', 'poisson']
    best_fit = None
    best_aic = np.inf
    
    dist_results = {}
    for dist_name in distributions:
        try:
            if dist_name == 'poisson':
                # Poisson requires integer data
                params = (series_clean.mean(),)
                log_likelihood = np.sum(stats.poisson.logpmf(series_clean.astype(int), *params))
            else:
                dist = getattr(stats, dist_name)
                params = dist.fit(series_clean)
                log_likelihood = np.sum(dist.logpdf(series_clean, *params))
            
            k = len(params)
            aic = 2 * k - 2 * log_likelihood
            dist_results[dist_name] = {'params': params, 'aic': aic}
            
            if aic < best_aic:
                best_aic = aic
                best_fit = dist_name
        except Exception as e:
            logger.warning(f"Failed to fit {dist_name}: {e}")
    
    results['distribution_fits'] = dist_results
    results['best_fit_distribution'] = best_fit
    
    # Create visualization
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Demand Distribution Analysis', fontsize=14, fontweight='bold')
    
    # Histogram with fitted distributions
    ax = axes[0, 0]
    ax.hist(series_clean, bins=50, density=True, alpha=0.7, edgecolor='black', label='Data')
    
    x_range = np.linspace(series_clean.min(), series_clean.max(), 100)
    
    # Plot normal fit
    mu, sigma = stats.norm.fit(series_clean)
    ax.plot(x_range, stats.norm.pdf(x_range, mu, sigma), 'r-', linewidth=2, label='Normal fit')
    
    # Plot best fit if different
    if best_fit and best_fit != 'norm' and best_fit in dist_results:
        if best_fit != 'poisson':
            dist = getattr(stats, best_fit)
            params = dist_results[best_fit]['params']
            ax.plot(x_range, dist.pdf(x_range, *params), 'g--', linewidth=2, label=f'{best_fit} fit')
    
    ax.set_xlabel('Demand')
    ax.set_ylabel('Density')
    ax.set_title(f'Distribution (Best fit: {best_fit})')
    ax.legend()
    ax.grid(alpha=0.3)
    
    # Q-Q plot (normal)
    ax = axes[0, 1]
    stats.probplot(series_clean, dist="norm", plot=ax)
    ax.set_title(f'Q-Q Plot (Normal)\nSkewness: {results["skewness"]:.3f}, Kurtosis: {results["kurtosis"]:.3f}')
    ax.grid(alpha=0.3)
    
    # Box plot
    ax = axes[1, 0]
    bp = ax.boxplot(series_clean, vert=True, patch_artist=True)
    bp['boxes'][0].set_facecolor('lightblue')
    ax.set_ylabel('Demand')
    ax.set_title('Box Plot (Outlier Detection)')
    ax.grid(alpha=0.3, axis='y')
    
    # Add statistics annotation
    stats_text = f"Mean: {results['mean']:.2f}\nMedian: {results['median']:.2f}\nStd: {results['std']:.2f}"
    ax.text(1.15, results['median'], stats_text, fontsize=9, verticalalignment='center')
    
    # CDF comparison
    ax = axes[1, 1]
    sorted_data = np.sort(series_clean)
    empirical_cdf = np.arange(1, len(sorted_data) + 1) / len(sorted_data)
    ax.plot(sorted_data, empirical_cdf, 'b-', linewidth=2, label='Empirical CDF')
    
    # Theoretical CDF (normal)
    ax.plot(sorted_data, stats.norm.cdf(sorted_data, mu, sigma), 'r--', linewidth=2, label='Normal CDF')
    
    ax.set_xlabel('Demand')
    ax.set_ylabel('Cumulative Probability')
    ax.set_title('CDF Comparison')
    ax.legend()
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'distribution_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: distribution_analysis.png")
    
    return results


# =============================================================================
# 8. IMPACT SUMMARY
# =============================================================================

def create_impact_summary(
    all_results: Dict,
    output_dir: str = "results/eda"
) -> pd.DataFrame:
    """
    Create a summary of how correlations impact different models.
    """
    logger.info("Creating impact summary...")
    
    impact_data = []
    
    # Autocorrelation impact
    if 'autocorrelation' in all_results:
        acr = all_results['autocorrelation']
        impact_data.append({
            'Factor': 'Temporal Autocorrelation',
            'Finding': f"DW={acr.get('durbin_watson', 'N/A'):.3f}" if acr.get('durbin_watson') else 'N/A',
            'Impact_Conformal': 'High' if acr.get('has_significant_autocorr') else 'Low',
            'Impact_LSTM': 'Positive' if acr.get('has_significant_autocorr') else 'Neutral',
            'Impact_Traditional': 'Moderate',
            'Recommendation': 'Use time-aware conformal or EnbPI' if acr.get('has_significant_autocorr') else 'Standard methods OK'
        })
    
    # Cross-SKU correlation impact
    if 'cross_sku' in all_results:
        csk = all_results['cross_sku']
        impact_data.append({
            'Factor': 'Cross-SKU Correlation',
            'Finding': f"Mean ρ={csk.get('mean_correlation', 'N/A'):.3f}" if csk.get('mean_correlation') else 'N/A',
            'Impact_Conformal': 'Low',
            'Impact_LSTM': 'Transfer learning opportunity' if csk.get('mean_correlation', 0) > 0.5 else 'Independent OK',
            'Impact_Traditional': 'Low',
            'Recommendation': 'Consider hierarchical models' if csk.get('mean_correlation', 0) > 0.5 else 'Independent modeling OK'
        })
    
    # Feature correlation impact
    if 'features' in all_results:
        fcr = all_results['features']
        impact_data.append({
            'Factor': 'Feature-Target Correlation',
            'Finding': f"Top feature ρ={fcr.get('top_features', [(None, 0)])[0][1]:.3f}" if fcr.get('top_features') else 'N/A',
            'Impact_Conformal': 'Determines interval width',
            'Impact_LSTM': 'Feature selection important',
            'Impact_Traditional': 'RF handles well',
            'Recommendation': 'Remove low-correlation features' if fcr.get('top_features') else 'N/A'
        })
    
    # Stationarity impact
    if 'stationarity' in all_results:
        sta = all_results['stationarity']
        impact_data.append({
            'Factor': 'Stationarity',
            'Finding': sta.get('stationarity_conclusion', 'N/A'),
            'Impact_Conformal': 'Critical - exchangeability' if 'Non-stationary' in sta.get('stationarity_conclusion', '') else 'OK',
            'Impact_LSTM': 'Can handle non-stationarity',
            'Impact_Traditional': 'May need differencing',
            'Recommendation': 'Use differencing or adaptive methods' if 'Non-stationary' in sta.get('stationarity_conclusion', '') else 'Standard methods OK'
        })
    
    # Heteroscedasticity impact
    if 'heteroscedasticity' in all_results:
        het = all_results['heteroscedasticity']
        impact_data.append({
            'Factor': 'Heteroscedasticity',
            'Finding': 'Detected' if het.get('has_heteroscedasticity') else 'Not detected',
            'Impact_Conformal': 'Use CQR not standard CP' if het.get('has_heteroscedasticity') else 'Standard CP OK',
            'Impact_LSTM': 'Quantile regression recommended',
            'Impact_Traditional': 'Normal assumption invalid',
            'Recommendation': 'Use adaptive/quantile methods' if het.get('has_heteroscedasticity') else 'Fixed intervals OK'
        })
    
    # Distribution impact
    if 'distribution' in all_results:
        dist = all_results['distribution']
        impact_data.append({
            'Factor': 'Distribution Shape',
            'Finding': f"Best fit: {dist.get('best_fit_distribution', 'N/A')}, Skew={dist.get('skewness', 0):.2f}",
            'Impact_Conformal': 'Distribution-free (robust)',
            'Impact_LSTM': 'Loss function choice',
            'Impact_Traditional': 'Normal assumption may fail' if not dist.get('is_normal') else 'Normal OK',
            'Recommendation': 'Use quantile regression' if dist.get('skewness', 0) > 1 else 'Standard methods OK'
        })
    
    # Seasonality impact
    if 'seasonality' in all_results:
        sea = all_results['seasonality']
        impact_data.append({
            'Factor': 'Seasonality',
            'Finding': f"Strength={sea.get('seasonal_strength', 0):.3f}" if sea.get('seasonal_strength') else 'N/A',
            'Impact_Conformal': 'Time features crucial',
            'Impact_LSTM': 'Can capture automatically',
            'Impact_Traditional': 'Need seasonal features',
            'Recommendation': 'Include time features' if sea.get('seasonal_strength', 0) > 0.3 else 'Minimal seasonality'
        })
    
    # Create DataFrame
    impact_df = pd.DataFrame(impact_data)
    
    # Save to CSV
    impact_df.to_csv(os.path.join(output_dir, 'correlation_impact_summary.csv'), index=False)
    logger.info("✓ Saved: correlation_impact_summary.csv")
    
    # Create visualization
    fig, ax = plt.subplots(figsize=(16, 8))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(
        cellText=impact_df.values,
        colLabels=impact_df.columns,
        cellLoc='center',
        loc='center',
        bbox=[0, 0, 1, 1]
    )
    
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1, 2)
    
    # Style header
    for i in range(len(impact_df.columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
    
    # Alternate row colors
    for i in range(1, len(impact_df) + 1):
        for j in range(len(impact_df.columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
    
    plt.title('Correlation Impact on Inventory Models', fontsize=14, fontweight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, 'correlation_impact_summary.png'), dpi=300, bbox_inches='tight')
    plt.close()
    logger.info("✓ Saved: correlation_impact_summary.png")
    
    return impact_df


# =============================================================================
# MAIN FUNCTION
# =============================================================================

def run_full_eda(
    filepath: str,
    store_ids: List[int] = [1, 2, 3],
    item_ids: List[int] = [1, 2, 3, 4, 5],
    primary_store: int = 1,
    primary_item: int = 1,
    output_dir: str = "results/eda"
) -> Dict:
    """
    Run complete EDA analysis.
    
    Parameters
    ----------
    filepath : str
        Path to data file.
    store_ids : List[int]
        Store IDs for cross-SKU analysis.
    item_ids : List[int]
        Item IDs for cross-SKU analysis.
    primary_store : int
        Primary store for detailed analysis.
    primary_item : int
        Primary item for detailed analysis.
    output_dir : str
        Output directory for plots.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info("=" * 70)
    logger.info("EXPLORATORY DATA ANALYSIS: CORRELATION IMPACT")
    logger.info("=" * 70)
    
    # Load data
    logger.info("\nLoading data...")
    df_raw = load_raw_data(filepath)
    
    # Filter primary SKU
    df_primary = filter_store_item(df_raw, primary_store, primary_item)
    df_primary = df_primary.set_index('date').sort_index()
    demand_series = df_primary['sales']
    
    # Create features for feature analysis
    df_features, feature_cols = create_all_features(
        filter_store_item(df_raw, primary_store, primary_item)
    )
    
    logger.info(f"\nPrimary SKU: Store {primary_store}, Item {primary_item}")
    logger.info(f"Time range: {demand_series.index.min()} to {demand_series.index.max()}")
    logger.info(f"Observations: {len(demand_series)}")
    
    all_results = {}
    
    # 1. Autocorrelation Analysis
    logger.info("\n" + "=" * 50)
    logger.info("1. TEMPORAL AUTOCORRELATION")
    logger.info("=" * 50)
    all_results['autocorrelation'] = analyze_autocorrelation(demand_series, output_dir=output_dir)
    
    # 2. Cross-SKU Correlation
    logger.info("\n" + "=" * 50)
    logger.info("2. CROSS-SKU CORRELATION")
    logger.info("=" * 50)
    all_results['cross_sku'] = analyze_cross_sku_correlation(
        df_raw, store_ids, item_ids, output_dir=output_dir
    )
    
    # 3. Feature-Target Correlation
    logger.info("\n" + "=" * 50)
    logger.info("3. FEATURE-TARGET CORRELATION")
    logger.info("=" * 50)
    all_results['features'] = analyze_feature_correlations(
        df_features, feature_cols, output_dir=output_dir
    )
    
    # 4. Seasonality Analysis
    logger.info("\n" + "=" * 50)
    logger.info("4. SEASONALITY AND TREND")
    logger.info("=" * 50)
    all_results['seasonality'] = analyze_seasonality(demand_series, output_dir=output_dir)
    
    # 5. Stationarity Tests
    logger.info("\n" + "=" * 50)
    logger.info("5. STATIONARITY TESTS")
    logger.info("=" * 50)
    all_results['stationarity'] = analyze_stationarity(demand_series, output_dir=output_dir)
    
    # 6. Heteroscedasticity Analysis
    logger.info("\n" + "=" * 50)
    logger.info("6. HETEROSCEDASTICITY")
    logger.info("=" * 50)
    all_results['heteroscedasticity'] = analyze_heteroscedasticity(demand_series, output_dir=output_dir)
    
    # 7. Distribution Analysis
    logger.info("\n" + "=" * 50)
    logger.info("7. DISTRIBUTION ANALYSIS")
    logger.info("=" * 50)
    all_results['distribution'] = analyze_distribution(demand_series, output_dir=output_dir)
    
    # 8. Impact Summary
    logger.info("\n" + "=" * 50)
    logger.info("8. IMPACT SUMMARY")
    logger.info("=" * 50)
    impact_df = create_impact_summary(all_results, output_dir=output_dir)
    
    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("EDA SUMMARY")
    logger.info("=" * 70)
    print(impact_df.to_string(index=False))
    
    # Key findings
    logger.info("\n" + "=" * 70)
    logger.info("KEY FINDINGS & RECOMMENDATIONS")
    logger.info("=" * 70)
    
    findings = []
    
    if all_results['autocorrelation'].get('has_significant_autocorr'):
        findings.append("⚠️  Significant autocorrelation detected → Consider EnbPI or time-aware conformal")
    
    if all_results['stationarity'].get('stationarity_conclusion') == 'Non-stationary':
        findings.append("⚠️  Non-stationary series → May need differencing or adaptive methods")
    
    if all_results['heteroscedasticity'].get('has_heteroscedasticity'):
        findings.append("⚠️  Heteroscedasticity detected → Use Conformalized Quantile Regression (CQR)")
    
    if not all_results['distribution'].get('is_normal'):
        findings.append("⚠️  Non-normal distribution → Normal assumption methods may underperform")
    
    if all_results['seasonality'].get('seasonal_strength', 0) > 0.3:
        findings.append("✓  Strong seasonality → Time features are important")
    
    if all_results['cross_sku'].get('mean_correlation', 0) > 0.5:
        findings.append("✓  High cross-SKU correlation → Consider hierarchical forecasting")
    
    if not findings:
        findings.append("✓  Data appears well-behaved for standard methods")
    
    for finding in findings:
        logger.info(finding)
    
    logger.info("\n" + "=" * 70)
    logger.info("EDA COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nAll visualizations saved to: {output_dir}/")
    
    return all_results


def main():
    parser = argparse.ArgumentParser(description="EDA: Correlation Impact Analysis")
    parser.add_argument("--data", type=str, default="train.csv", help="Data file path")
    parser.add_argument("--output", type=str, default="results/eda", help="Output directory")
    parser.add_argument("--stores", type=str, default="1,2,3", help="Store IDs for cross-SKU analysis")
    parser.add_argument("--items", type=str, default="1,2,3,4,5", help="Item IDs for cross-SKU analysis")
    parser.add_argument("--primary-store", type=int, default=1, help="Primary store for detailed analysis")
    parser.add_argument("--primary-item", type=int, default=1, help="Primary item for detailed analysis")
    
    args = parser.parse_args()
    
    store_ids = [int(s) for s in args.stores.split(",")]
    item_ids = [int(i) for i in args.items.split(",")]
    
    run_full_eda(
        filepath=args.data,
        store_ids=store_ids,
        item_ids=item_ids,
        primary_store=args.primary_store,
        primary_item=args.primary_item,
        output_dir=args.output
    )


if __name__ == "__main__":
    main()