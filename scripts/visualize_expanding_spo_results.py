"""
Visualization script for expanding window experiment results with SPO baseline.

This script creates comprehensive visualizations highlighting SPO performance:
1. SPO vs other models performance
2. Cost evolution and risk metrics
3. Decision-focused learning benefits
4. Comprehensive model comparison
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

def load_results(results_dir: str = "results/expanding_window_spo"):
    """Load expanding window results."""
    all_results_path = Path(results_dir) / "expanding_window_all.csv"
    if not all_results_path.exists():
        print(f"Error: {all_results_path} not found!")
        print("Please run: python scripts/run_expanding_window_experiment.py first")
        sys.exit(1)

    df_all = pd.read_csv(all_results_path)

    # Convert date columns
    df_all['test_start'] = pd.to_datetime(df_all['test_start'])
    df_all['test_end'] = pd.to_datetime(df_all['test_end'])

    # Compute aggregation
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


def plot_spo_comparison(df_all, df_agg, output_dir):
    """Highlight SPO performance vs other methods."""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    fig.suptitle('SPO/End-to-End vs Predict-then-Optimize Methods', fontsize=16, y=1.00)

    methods = df_all['Method'].unique()

    # Create color map: SPO in red, others in different colors
    colors = []
    for method in methods:
        if 'SPO' in method:
            colors.append('red')
        elif method in ['Conformal_CVaR', 'Normal_CVaR', 'QuantileReg_CVaR', 'SAA']:
            colors.append('steelblue')
        else:
            colors.append('coral')

    # 1. Mean Cost Evolution
    ax = axes[0, 0]
    for i, method in enumerate(methods):
        method_data = df_all[df_all['Method'] == method].sort_values('window_idx')
        linewidth = 3 if 'SPO' in method else 2
        linestyle = '-' if 'SPO' in method else '--'
        ax.plot(method_data['window_idx'], method_data['Mean_Cost'],
                marker='o', label=method, color=colors[i],
                linewidth=linewidth, linestyle=linestyle, markersize=6)
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('Mean Cost ($)', fontsize=12)
    ax.set_title('Mean Cost Evolution (SPO in Red)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    # 2. CVaR-90 Evolution
    ax = axes[0, 1]
    for i, method in enumerate(methods):
        method_data = df_all[df_all['Method'] == method].sort_values('window_idx')
        linewidth = 3 if 'SPO' in method else 2
        linestyle = '-' if 'SPO' in method else '--'
        ax.plot(method_data['window_idx'], method_data['CVaR-90'],
                marker='s', label=method, color=colors[i],
                linewidth=linewidth, linestyle=linestyle, markersize=6)
    ax.set_xlabel('Window Index', fontsize=12)
    ax.set_ylabel('CVaR-90 ($)', fontsize=12)
    ax.set_title('CVaR-90 Evolution (SPO in Red)', fontsize=13, fontweight='bold')
    ax.legend(loc='best', fontsize=9)
    ax.grid(alpha=0.3)

    # 3. Bar comparison - Mean Cost
    ax = axes[1, 0]
    mean_costs = df_agg[('Mean_Cost', 'mean')].values
    std_costs = df_agg[('Mean_Cost', 'std')].values
    method_names = df_agg.index.tolist()

    bars = ax.bar(range(len(method_names)), mean_costs, yerr=std_costs,
                  capsize=5, color=colors, alpha=0.8)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Mean Cost ($)', fontsize=11)
    ax.set_title('Average Mean Cost Across Windows', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    # Highlight SPO
    for i, method in enumerate(method_names):
        if 'SPO' in method:
            bars[i].set_edgecolor('darkred')
            bars[i].set_linewidth(3)

    # 4. Bar comparison - CVaR-90
    ax = axes[1, 1]
    cvar90 = df_agg[('CVaR-90', 'mean')].values
    std_cvar90 = df_agg[('CVaR-90', 'std')].values

    bars = ax.bar(range(len(method_names)), cvar90, yerr=std_cvar90,
                  capsize=5, color=colors, alpha=0.8)
    ax.set_xticks(range(len(method_names)))
    ax.set_xticklabels(method_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('CVaR-90 ($)', fontsize=11)
    ax.set_title('Average CVaR-90 Across Windows', fontsize=12, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)

    for i, method in enumerate(method_names):
        if 'SPO' in method:
            bars[i].set_edgecolor('darkred')
            bars[i].set_linewidth(3)

    plt.tight_layout()
    save_path = Path(output_dir) / "spo_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_spo_ranking(df_agg, output_dir):
    """Show SPO ranking among all methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('SPO/End-to-End Performance Ranking', fontsize=16, y=1.02)

    methods = df_agg.index.tolist()

    metrics = [
        ('Mean_Cost', 'Mean Cost ($)', False),  # False = lower is better
        ('CVaR-90', 'CVaR-90 ($)', False),
        ('MAE', 'MAE', False)
    ]

    for idx, (metric, label, higher_better) in enumerate(metrics):
        ax = axes[idx]

        values = df_agg[(metric, 'mean')].values
        stds = df_agg[(metric, 'std')].values

        # Sort by metric
        if higher_better:
            sorted_idx = np.argsort(values)[::-1]
        else:
            sorted_idx = np.argsort(values)

        sorted_methods = [methods[i] for i in sorted_idx]
        sorted_values = values[sorted_idx]
        sorted_stds = stds[sorted_idx]

        # Color: SPO in red, others in gray
        colors = ['red' if 'SPO' in m else 'steelblue' for m in sorted_methods]

        bars = ax.barh(range(len(sorted_methods)), sorted_values,
                      xerr=sorted_stds, capsize=5, color=colors, alpha=0.7)
        ax.set_yticks(range(len(sorted_methods)))
        ax.set_yticklabels(sorted_methods, fontsize=10)
        ax.set_xlabel(label, fontsize=11)
        ax.set_title(f'Ranking by {label}', fontsize=12, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)

        # Add rank numbers
        for i, (method, value) in enumerate(zip(sorted_methods, sorted_values)):
            rank = i + 1
            ax.text(-0.5, i, f'#{rank}', ha='right', va='center',
                   fontweight='bold', fontsize=10)

            # Highlight SPO rank
            if 'SPO' in method:
                ax.axhline(i, color='red', linestyle=':', alpha=0.3, linewidth=2)

    plt.tight_layout()
    save_path = Path(output_dir) / "spo_ranking.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_method_categories_with_spo(df_agg, output_dir):
    """Compare traditional, DL, and SPO methods."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Traditional vs Deep Learning vs SPO (Decision-Focused)', fontsize=16, y=1.02)

    # Categorize methods
    traditional = ['Conformal_CVaR', 'Normal_CVaR', 'QuantileReg_CVaR', 'SAA']
    deep_learning = ['LSTM_QR', 'Transformer_QR', 'TFT']
    spo_methods = ['SPO_EndToEnd']

    metrics = [
        ('Mean_Cost', 'Mean Cost ($)'),
        ('CVaR-90', 'CVaR-90 ($)'),
        ('MAE', 'MAE')
    ]

    for idx, (metric, label) in enumerate(metrics):
        ax = axes[idx]

        # Extract data for each category
        trad_data = df_agg.loc[df_agg.index.isin(traditional), (metric, 'mean')].values
        dl_data = df_agg.loc[df_agg.index.isin(deep_learning), (metric, 'mean')].values
        spo_data = df_agg.loc[df_agg.index.isin(spo_methods), (metric, 'mean')].values

        # Compute category averages
        categories = []
        values = []
        colors_cat = []

        if len(trad_data) > 0:
            categories.append('Traditional\n(Predict-then-\nOptimize)')
            values.append(np.mean(trad_data))
            colors_cat.append('steelblue')

        if len(dl_data) > 0:
            categories.append('Deep Learning\n(Predict-then-\nOptimize)')
            values.append(np.mean(dl_data))
            colors_cat.append('coral')

        if len(spo_data) > 0:
            categories.append('SPO\n(Decision-\nFocused)')
            values.append(np.mean(spo_data))
            colors_cat.append('red')

        bars = ax.bar(range(len(categories)), values, color=colors_cat, alpha=0.8,
                     edgecolor='black', linewidth=2)
        ax.set_xticks(range(len(categories)))
        ax.set_xticklabels(categories, fontsize=11, fontweight='bold')
        ax.set_ylabel(label, fontsize=12)
        ax.set_title(label, fontsize=13, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Add value labels on bars
        for i, (bar, val) in enumerate(zip(bars, values)):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'${val:.2f}' if 'Cost' in label or 'CVaR' in label else f'{val:.2f}',
                   ha='center', va='bottom', fontweight='bold', fontsize=11)

    plt.tight_layout()
    save_path = Path(output_dir) / "method_categories_with_spo.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def plot_comprehensive_comparison(df_agg, output_dir):
    """Comprehensive bar chart comparison."""
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('Comprehensive Model Comparison with SPO Baseline', fontsize=18, y=1.00)

    methods = df_agg.index.tolist()

    # SPO color coding
    colors = ['red' if 'SPO' in m else ('steelblue' if m in ['Conformal_CVaR', 'Normal_CVaR', 'QuantileReg_CVaR', 'SAA'] else 'coral') for m in methods]

    metrics_config = [
        ('Mean_Cost', 'Mean Cost ($)', axes[0, 0]),
        ('CVaR-90', 'CVaR-90 ($)', axes[0, 1]),
        ('CVaR-95', 'CVaR-95 ($)', axes[0, 2]),
        ('Service_Level', 'Service Level (%)', axes[1, 0], True),  # True for percentage
        ('MAE', 'MAE', axes[1, 1]),
        ('RMSE', 'RMSE', axes[1, 2])
    ]

    for config in metrics_config:
        if len(config) == 3:
            metric, label, ax = config
            is_percentage = False
        else:
            metric, label, ax, is_percentage = config

        values = df_agg[(metric, 'mean')].values
        stds = df_agg[(metric, 'std')].values

        if is_percentage:
            values = values * 100
            stds = stds * 100

        bars = ax.bar(range(len(methods)), values, yerr=stds, capsize=5,
                     color=colors, alpha=0.8, edgecolor='black', linewidth=1)
        ax.set_xticks(range(len(methods)))
        ax.set_xticklabels(methods, rotation=45, ha='right', fontsize=9)
        ax.set_ylabel(label, fontsize=11, fontweight='bold')
        ax.set_title(label, fontsize=12, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)

        # Highlight best and SPO
        if metric != 'Service_Level':
            best_idx = np.argmin(values)
        else:
            best_idx = np.argmax(values)

        bars[best_idx].set_edgecolor('green')
        bars[best_idx].set_linewidth(3)

        # Extra highlight for SPO
        for i, method in enumerate(methods):
            if 'SPO' in method:
                bars[i].set_linewidth(3)
                bars[i].set_edgecolor('darkred')

    plt.tight_layout()
    save_path = Path(output_dir) / "comprehensive_comparison.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def create_spo_summary_table(df_agg, output_dir):
    """Create summary table with SPO highlighted."""
    fig, ax = plt.subplots(figsize=(16, 10))
    ax.axis('tight')
    ax.axis('off')

    methods = df_agg.index.tolist()

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
            f"{df_agg.loc[method, ('RMSE', 'mean')]:.2f} ± {df_agg.loc[method, ('RMSE', 'std')]:.2f}",
        ]
        table_data.append(row)

    columns = ['Method', 'Mean Cost', 'CVaR-90', 'CVaR-95', 'Service\nLevel', 'Coverage', 'MAE', 'RMSE']

    table = ax.table(cellText=table_data, colLabels=columns, cellLoc='center',
                    loc='center', bbox=[0, 0, 1, 1])

    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 2.2)

    # Style header
    for i in range(len(columns)):
        table[(0, i)].set_facecolor('#4472C4')
        table[(0, i)].set_text_props(weight='bold', color='white')

    # Highlight rows
    for i in range(1, len(methods) + 1):
        method = methods[i-1]
        for j in range(len(columns)):
            if 'SPO' in method:
                table[(i, j)].set_facecolor('#FFE6E6')  # Light red for SPO
                table[(i, j)].set_text_props(weight='bold')
            elif i % 2 == 0:
                table[(i, j)].set_facecolor('#F0F0F0')

    plt.title('Performance Summary: Expanding Window with SPO Baseline\n(Mean ± Std)',
             fontsize=15, fontweight='bold', pad=20)

    save_path = Path(output_dir) / "spo_summary_table.png"
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved: {save_path}")
    plt.close()


def main():
    """Main visualization pipeline."""
    print("\n" + "="*70)
    print("EXPANDING WINDOW WITH SPO RESULTS VISUALIZATION")
    print("="*70)

    # Load results
    print("\n[1/6] Loading results...")
    df_all, df_agg = load_results()
    print(f"✓ Loaded {len(df_all)} records across {df_all['window_idx'].nunique()} windows")
    print(f"✓ {len(df_agg)} methods evaluated (including SPO)")

    output_dir = Path("results/expanding_window_spo/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    # Create visualizations
    print("\n[2/6] Creating SPO comparison plots...")
    plot_spo_comparison(df_all, df_agg, output_dir)

    print("\n[3/6] Creating SPO ranking plots...")
    plot_spo_ranking(df_agg, output_dir)

    print("\n[4/6] Creating method category comparison with SPO...")
    plot_method_categories_with_spo(df_agg, output_dir)

    print("\n[5/6] Creating comprehensive comparison...")
    plot_comprehensive_comparison(df_agg, output_dir)

    print("\n[6/6] Creating summary table...")
    create_spo_summary_table(df_agg, output_dir)

    # Print SPO performance summary
    print("\n" + "="*70)
    print("SPO/END-TO-END PERFORMANCE SUMMARY")
    print("="*70)

    if 'SPO_EndToEnd' in df_agg.index:
        spo_stats = df_agg.loc['SPO_EndToEnd']
        print(f"\nSPO Mean Cost: ${spo_stats[('Mean_Cost', 'mean')]:.2f} ± ${spo_stats[('Mean_Cost', 'std')]:.2f}")
        print(f"SPO CVaR-90: ${spo_stats[('CVaR-90', 'mean')]:.2f} ± ${spo_stats[('CVaR-90', 'std')]:.2f}")
        print(f"SPO CVaR-95: ${spo_stats[('CVaR-95', 'mean')]:.2f} ± ${spo_stats[('CVaR-95', 'std')]:.2f}")
        print(f"SPO MAE: {spo_stats[('MAE', 'mean')]:.2f} ± {spo_stats[('MAE', 'std')]:.2f}")

        # Compare to best traditional and DL methods
        traditional_methods = ['Conformal_CVaR', 'Normal_CVaR', 'QuantileReg_CVaR']
        dl_methods = ['LSTM_QR', 'Transformer_QR', 'TFT']

        trad_in_results = [m for m in traditional_methods if m in df_agg.index]
        dl_in_results = [m for m in dl_methods if m in df_agg.index]

        if trad_in_results:
            best_trad_cvar = df_agg.loc[trad_in_results, ('CVaR-90', 'mean')].min()
            print(f"\nBest Traditional CVaR-90: ${best_trad_cvar:.2f}")
            print(f"SPO vs Best Traditional: {((spo_stats[('CVaR-90', 'mean')] / best_trad_cvar - 1) * 100):+.2f}%")

        if dl_in_results:
            best_dl_cvar = df_agg.loc[dl_in_results, ('CVaR-90', 'mean')].min()
            print(f"Best DL CVaR-90: ${best_dl_cvar:.2f}")
            print(f"SPO vs Best DL: {((spo_stats[('CVaR-90', 'mean')] / best_dl_cvar - 1) * 100):+.2f}%")

    print("\n" + "="*70)
    print("✅ VISUALIZATION COMPLETE!")
    print("="*70)
    print(f"\nAll visualizations saved to: {output_dir}/")
    print("\nGenerated files:")
    print("  1. spo_comparison.png - SPO vs other methods")
    print("  2. spo_ranking.png - Performance rankings")
    print("  3. method_categories_with_spo.png - Category comparison")
    print("  4. comprehensive_comparison.png - All metrics")
    print("  5. spo_summary_table.png - Results table")
    print("="*70 + "\n")


if __name__ == "__main__":
    main()
