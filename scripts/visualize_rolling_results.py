"""
Visualization script for rolling window experiment results.
 
This script creates comprehensive visualizations to understand:
1. Model performance across time windows
2. Cost evolution and risk metrics
3. Forecast quality comparison
4. Trade-offs between different approaches
"""
 
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import sys
 
# Set style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")
 
def load_results(results_dir: str = "results/rolling_window"):
    """Load rolling window results."""
    all_results_path = Path(results_dir) / "rolling_window_all.csv"
    if not all_results_path.exists():
        print(f"Error: {all_results_path} not found!")
        print("Please run: python scripts/run_rolling_window_experiment.py first")
        sys.exit(1)
 
    df_all = pd.read_csv(all_results_path)
 
    # Convert date columns
    df_all['test_start'] = pd.to_datetime(df_all['test_start'])
    df_all['test_end'] = pd.to_datetime(df_all['test_end'])
 
    # Compute aggregation directly from df_all (more reliable than loading CSV)
    # This ensures we have the proper multi-level column structure
    df_agg = df_all.groupby('Method').agg({
        'Mean_Cost': ['mean', 'std'],
        'CVaR-90': ['mean', 'std'],
        'CVaR-95': ['mean', 'std'],
        'Service_Level': ['mean', 'std'],
        'Coverage': ['mean', 'std'],
        'Interval_Width': ['mean', 'std'],
        'MAE': ['mean', 'std'],
        'RMSE': ['mean', 'std'],
        'MAPE': ['mean', 'std']
    }).round(2)
 
    return df_all, df_agg
 
 
def plot_cost_evolution(df_all, output_dir):
    """Plot how costs evolve across time windows for each method."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('Cost and Risk Evolution Across Time Windows', fontsize=16, y=1.00)
 
    # Get unique methods and colors
    methods = df_all['Method'].unique()
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
 
    # 1. Mean Cost Evolution
    ax = axes[0, 0]
    for i, method in enumerate(methods):
        method_data = df_all[df_all['Method'] == method].sort_values('window_idx')
        ax.plot(method_data['window_idx'], method_data['Mean_Cost'],
                marker='o', label=method, color=colors[i], linewidth=2, markersize=5)
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Mean Cost ($)', fontsize=12)
    ax.set_title('Mean Cost Evolution', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
 
    # 2. CVaR-90 Evolution
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        method_data = df_all[df_all['Method'] == method].sort_values('window_idx')
        ax.plot(method_data['window_idx'], method_data['CVaR-90'],
                marker='s', label=method, color=colors[i], linewidth=2, markersize=5)
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('CVaR-90 ($)', fontsize=12)
    ax.set_title('CVaR-90 (Tail Risk) Evolution', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
 
    # 3. Service Level Evolution
    ax = axes[1, 0]
    for i, method in enumerate(methods):
        method_data = df_all[df_all['Method'] == method].sort_values('window_idx')
        ax.plot(method_data['window_idx'], method_data['Service_Level'] * 100,
                marker='^', label=method, color=colors[i], linewidth=2, markersize=5)
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Service Level (%)', fontsize=12)
    ax.set_title('Service Level Evolution', fontsize=13, fontweight='bold')
    ax.axhline(y=95, color='red', linestyle='--', linewidth=1.5, alpha=0.7, label='Target 95%')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
 
    # 4. MAE Evolution
    ax = axes[1, 1]
    for i, method in enumerate(methods):
        method_data = df_all[df_all['Method'] == method].sort_values('window_idx')
        ax.plot(method_data['window_idx'], method_data['MAE'],
                marker='d', label=method, color=colors[i], linewidth=2, markersize=5)
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('MAE', fontsize=12)
    ax.set_title('Forecast Error (MAE) Evolution', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)
 
    plt.tight_layout()
    save_path = Path(output_dir) / "cost_evolution_over_time.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
 
 
def plot_model_comparison_bars(df_agg, output_dir):
    """Create bar plots comparing model performance."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle('Model Performance Comparison (Averaged Across 23 Windows)', fontsize=16, y=1.00)
 
    # Parse aggregated results
    methods = df_agg.index.tolist()
 
    # 1. Mean Cost
    ax = axes[0, 0]
    mean_cost = df_agg[('Mean_Cost', 'mean')].values
    std_cost = df_agg[('Mean_Cost', 'std')].values
    bars = ax.bar(range(len(methods)), mean_cost, yerr=std_cost, capsize=5,
                   color=plt.cm.tab10(np.linspace(0, 1, len(methods))), alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Cost ($)', fontsize=11)
    ax.set_title('Average Mean Cost', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    # Highlight best
    best_idx = np.argmin(mean_cost)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
 
    # 2. CVaR-90
    ax = axes[0, 1]
    cvar90 = df_agg[('CVaR-90', 'mean')].values
    std_cvar90 = df_agg[('CVaR-90', 'std')].values
    bars = ax.bar(range(len(methods)), cvar90, yerr=std_cvar90, capsize=5,
                   color=plt.cm.tab10(np.linspace(0, 1, len(methods))), alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('CVaR-90 ($)', fontsize=11)
    ax.set_title('Average CVaR-90 (Tail Risk)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    best_idx = np.argmin(cvar90)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
 
    # 3. Service Level
    ax = axes[0, 2]
    service = df_agg[('Service_Level', 'mean')].values * 100
    std_service = df_agg[('Service_Level', 'std')].values * 100
    bars = ax.bar(range(len(methods)), service, yerr=std_service, capsize=5,
                   color=plt.cm.tab10(np.linspace(0, 1, len(methods))), alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Service Level (%)', fontsize=11)
    ax.set_title('Average Service Level', fontsize=12, fontweight='bold')
    ax.axhline(y=95, color='red', linestyle='--', linewidth=2, alpha=0.7)
    ax.grid(axis='y', alpha=0.3)
 
    # 4. Coverage
    ax = axes[1, 0]
    coverage = df_agg[('Coverage', 'mean')].values * 100
    std_coverage = df_agg[('Coverage', 'std')].values * 100
    # Remove NaN for SAA
    valid_idx = ~np.isnan(coverage)
    bars = ax.bar(np.arange(len(methods))[valid_idx], coverage[valid_idx],
                   yerr=std_coverage[valid_idx], capsize=5,
                   color=plt.cm.tab10(np.linspace(0, 1, len(methods)))[valid_idx], alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Coverage (%)', fontsize=11)
    ax.set_title('Prediction Interval Coverage', fontsize=12, fontweight='bold')
    ax.axhline(y=95, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target 95%')
    ax.legend(fontsize=9)
    ax.grid(axis='y', alpha=0.3)
 
    # 5. Interval Width
    ax = axes[1, 1]
    width = df_agg[('Interval_Width', 'mean')].values
    std_width = df_agg[('Interval_Width', 'std')].values
    valid_idx = ~np.isnan(width)
    bars = ax.bar(np.arange(len(methods))[valid_idx], width[valid_idx],
                   yerr=std_width[valid_idx], capsize=5,
                   color=plt.cm.tab10(np.linspace(0, 1, len(methods)))[valid_idx], alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Interval Width', fontsize=11)
    ax.set_title('Average Prediction Interval Width', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
 
    # 6. MAE
    ax = axes[1, 2]
    mae = df_agg[('MAE', 'mean')].values
    std_mae = df_agg[('MAE', 'std')].values
    bars = ax.bar(range(len(methods)), mae, yerr=std_mae, capsize=5,
                   color=plt.cm.tab10(np.linspace(0, 1, len(methods))), alpha=0.8)
    ax.set_xticks(range(len(methods)))
    ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('MAE', fontsize=11)
    ax.set_title('Average Forecast Error (MAE)', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    best_idx = np.argmin(mae)
    bars[best_idx].set_edgecolor('red')
    bars[best_idx].set_linewidth(3)
 
    plt.tight_layout()
    save_path = Path(output_dir) / "model_comparison_bars.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
 
 
def plot_risk_return_tradeoff(df_agg, output_dir):
    """Plot risk-return tradeoff: Mean Cost vs CVaR-90."""
    fig, ax = plt.subplots(figsize=(12, 8))
 
    methods = df_agg.index.tolist()
    mean_cost = df_agg[('Mean_Cost', 'mean')].values
    cvar90 = df_agg[('CVaR-90', 'mean')].values
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
 
    # Scatter plot
    for i, method in enumerate(methods):
        ax.scatter(mean_cost[i], cvar90[i], s=300, color=colors[i],
                  alpha=0.7, edgecolor='black', linewidth=2, label=method)
 
    # Add annotations
    for i, method in enumerate(methods):
        ax.annotate(method, (mean_cost[i], cvar90[i]),
                   xytext=(10, 10), textcoords='offset points',
                   fontsize=10, fontweight='bold',
                   bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
 
    ax.set_xlabel('Mean Cost ($) - Lower is Better', fontsize=13, fontweight='bold')
    ax.set_ylabel('CVaR-90 ($) - Lower is Better', fontsize=13, fontweight='bold')
    ax.set_title('Risk-Return Tradeoff: Mean Cost vs Tail Risk (CVaR-90)',
                fontsize=15, fontweight='bold')
    ax.grid(alpha=0.3)
 
    # Add quadrants
    median_cost = np.median(mean_cost)
    median_cvar = np.median(cvar90)
    ax.axvline(median_cost, color='gray', linestyle='--', alpha=0.5)
    ax.axhline(median_cvar, color='gray', linestyle='--', alpha=0.5)
 
    # Label quadrants
    ax.text(0.02, 0.98, 'Low Cost\nHigh Risk', transform=ax.transAxes,
           fontsize=11, verticalalignment='top', alpha=0.6, style='italic')
    ax.text(0.98, 0.98, 'High Cost\nHigh Risk', transform=ax.transAxes,
           fontsize=11, verticalalignment='top', horizontalalignment='right',
           alpha=0.6, style='italic')
    ax.text(0.02, 0.02, 'Low Cost\nLow Risk ⭐', transform=ax.transAxes,
           fontsize=11, verticalalignment='bottom', alpha=0.6, style='italic',
           fontweight='bold')
    ax.text(0.98, 0.02, 'High Cost\nLow Risk', transform=ax.transAxes,
           fontsize=11, verticalalignment='bottom', horizontalalignment='right',
           alpha=0.6, style='italic')
 
    plt.tight_layout()
    save_path = Path(output_dir) / "risk_return_tradeoff.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
 
 
def plot_coverage_vs_width(df_agg, output_dir):
    """Plot coverage vs interval width tradeoff."""
    fig, ax = plt.subplots(figsize=(12, 8))
 
    methods = df_agg.index.tolist()
    coverage = df_agg[('Coverage', 'mean')].values * 100
    width = df_agg[('Interval_Width', 'mean')].values
    colors = plt.cm.tab10(np.linspace(0, 1, len(methods)))
 
    # Filter out NaN (SAA doesn't have intervals)
    valid_idx = ~np.isnan(coverage) & ~np.isnan(width)
 
    # Scatter plot
    for i, method in enumerate(methods):
        if valid_idx[i]:
            ax.scatter(width[i], coverage[i], s=300, color=colors[i],
                      alpha=0.7, edgecolor='black', linewidth=2, label=method)
            ax.annotate(method, (width[i], coverage[i]),
                       xytext=(10, 10), textcoords='offset points',
                       fontsize=10, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor=colors[i], alpha=0.3))
 
    ax.set_xlabel('Average Interval Width - Lower is Better', fontsize=13, fontweight='bold')
    ax.set_ylabel('Coverage (%) - Target: 95%', fontsize=13, fontweight='bold')
    ax.set_title('Forecast Quality: Coverage vs Interval Width Tradeoff',
                fontsize=15, fontweight='bold')
    ax.axhline(95, color='red', linestyle='--', linewidth=2, alpha=0.7, label='Target Coverage')
    ax.grid(alpha=0.3)
    ax.legend(fontsize=10)
 
    # Ideal region
    ax.fill_between([0, 30], 93, 97, alpha=0.1, color='green',
                    label='Ideal Region (93-97% coverage)')
 
    plt.tight_layout()
    save_path = Path(output_dir) / "coverage_vs_width.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
 
 
def plot_method_categories(df_agg, output_dir):
    """Compare traditional vs deep learning methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Traditional Methods vs Deep Learning Methods', fontsize=16, y=1.02)
 
    # Categorize methods
    traditional = ['Conformal_CVaR', 'Normal_CVaR', 'QuantileReg_CVaR', 'SAA']
    deep_learning = ['LSTM_QR', 'Transformer_QR', 'TFT']
 
    # Extract data
    trad_data = df_agg.loc[df_agg.index.isin(traditional)]
    dl_data = df_agg.loc[df_agg.index.isin(deep_learning)]
 
    metrics = [
        ('Mean_Cost', 'Mean Cost ($)'),
        ('CVaR-90', 'CVaR-90 ($)'),
        ('MAE', 'MAE')
    ]
 
    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]
 
        # Traditional methods
        trad_mean = trad_data[(metric, 'mean')].values
        trad_methods = trad_data.index.tolist()
 
        # DL methods
        dl_mean = dl_data[(metric, 'mean')].values
        dl_methods = dl_data.index.tolist()
 
        x_trad = np.arange(len(trad_methods))
        x_dl = np.arange(len(dl_methods)) + len(trad_methods) + 1
 
        ax.bar(x_trad, trad_mean, color='steelblue', alpha=0.8, label='Traditional')
        ax.bar(x_dl, dl_mean, color='coral', alpha=0.8, label='Deep Learning')
 
        # Set x-ticks
        all_positions = np.concatenate([x_trad, x_dl])
        all_labels = trad_methods + dl_methods
        ax.set_xticks(all_positions)
        ax.set_xticklabels(all_labels, rotation=45, ha='right', fontsize=9)
 
        ax.set_ylabel(label, fontsize=11)
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.legend(fontsize=10)
        ax.grid(axis='y', alpha=0.3)
 
        # Add separator
        separator_x = len(trad_methods) + 0.5
        ax.axvline(separator_x, color='black', linestyle='--', linewidth=2, alpha=0.5)
 
    plt.tight_layout()
    save_path = Path(output_dir) / "traditional_vs_deep_learning.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
 
 
def create_summary_table(df_agg, output_dir):
    """Create a formatted summary table image."""
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.axis('tight')
    ax.axis('off')
 
    # Prepare data
    methods = df_agg.index.tolist()
 
    # Select key metrics
    table_data = []
    for method in methods:
        row = [
            method,
            f"{df_agg.loc[method, ('Mean_Cost', 'mean')]:.2f} ± {df_agg.loc[method, ('Mean_Cost', 'std')]:.2f}",
            f"{df_agg.loc[method, ('CVaR-90', 'mean')]:.2f} ± {df_agg.loc[method, ('CVaR-90', 'std')]:.2f}",
            f"{df_agg.loc[method, ('CVaR-95', 'mean')]:.2f} ± {df_agg.loc[method, ('CVaR-95', 'std')]:.2f}",
            f"{df_agg.loc[method, ('Service_Level', 'mean')]*100:.1f}%",
            f"{df_agg.loc[method, ('Coverage', 'mean')]*100:.1f}%" if not np.isnan(df_agg.loc[method, ('Coverage', 'mean')]) else "N/A",
            f"{df_agg.loc[method, ('MAE', 'mean')]:.2f} ± {df_agg.loc[method, ('MAE', 'std')]:.2f}",
        ]
        table_data.append(row)
 
    columns = ['Method', 'Mean Cost', 'CVaR-90', 'CVaR-95', 'Service\nLevel', 'Coverage', 'MAE']
 
    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])
 
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1, 2.5)
 
    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')
 
    # Alternate row colors
    for i in range(1, len(methods) + 1):
        for j in range(len(columns)):
            if i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')
 
    plt.title('Performance Summary Across 23 Rolling Windows\n(Mean ± Std)',
             fontsize=14, fontweight='bold', pad=20)
 
    save_path = Path(output_dir) / "summary_table.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()
 
 
def main():
    """Main visualization pipeline."""
    print("\n" + "="*70)
    print("ROLLING WINDOW RESULTS VISUALIZATION")
    print("="*70)
 
    # Load results
    print("\n[1/7] Loading results...")
    df_all, df_agg = load_results()
    print(f"✓ Loaded {len(df_all)} records across {df_all['window_idx'].nunique()} windows")
    print(f"✓ {len(df_agg)} methods evaluated")
 
    output_dir = Path("results/rolling_window/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)
 
    # Create visualizations
    print("\n[2/7] Creating cost evolution plots...")
    plot_cost_evolution(df_all, output_dir)
 
    print("\n[3/7] Creating model comparison bars...")
    plot_model_comparison_bars(df_agg, output_dir)
 
    print("\n[4/7] Creating risk-return tradeoff plot...")
    plot_risk_return_tradeoff(df_agg, output_dir)
 
    print("\n[5/7] Creating coverage vs width plot...")
    plot_coverage_vs_width(df_agg, output_dir)
 
    print("\n[6/7] Creating traditional vs DL comparison...")
    plot_method_categories(df_agg, output_dir)
 
    print("\n[7/7] Creating summary table...")
    create_summary_table(df_agg, output_dir)
 
    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. cost_evolution_over_time.png - Temporal trends")
    print("  2. model_comparison_bars.png - Performance metrics")
    print("  3. risk_return_tradeoff.png - Cost vs CVaR scatter")
    print("  4. coverage_vs_width.png - Forecast quality")
    print("  5. traditional_vs_deep_learning.png - Method comparison")
    print("  6. summary_table.png - Results table")
    print("="*70 + "\n")
 
 
if __name__ == "__main__":
    main()