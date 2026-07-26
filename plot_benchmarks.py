"""
Generate plots from benchmark results.

Requires matplotlib and seaborn:
    pip install matplotlib seaborn

Usage:
    python plot_benchmarks.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

# Set style
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams.update({
    'font.size': 11,
    'axes.titlesize': 13,
    'axes.labelsize': 12,
    'figure.dpi': 150,
})


def plot_threshold_sweep(csv_path: Path, output_dir: Path) -> None:
    """Plot accuracy vs threshold for each mode."""
    import csv
    thresholds = []
    img_acc = []
    cos_acc = []
    hybrid_acc = []
    
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            thresholds.append(float(row['threshold']))
            img_acc.append(float(row['img_only_accuracy']))
            cos_acc.append(float(row['cosine_only_accuracy']))
            hybrid_acc.append(float(row['hybrid_accuracy']))
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(thresholds, img_acc, marker='o', label='IMG Only', linewidth=2, color='#6366f1')
    ax.plot(thresholds, cos_acc, marker='s', label='Cosine Only', linewidth=2, color='#10b981')
    ax.plot(thresholds, hybrid_acc, marker='^', label='Hybrid', linewidth=2, color='#f59e0b')
    
    ax.set_xlabel('Threshold')
    ax.set_ylabel('Accuracy')
    ax.set_title('Threshold Sweep: Accuracy by Mode')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    
    plt.tight_layout()
    output_path = output_dir / 'threshold_sweep.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_robustness(csv_path: Path, output_dir: Path) -> None:
    """Plot accuracy vs severity for each mode and strategy."""
    import csv
    from collections import defaultdict
    
    data = defaultdict(lambda: defaultdict(list))
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode = row['mode']
            strategy = row['strategy']
            severity = float(row['severity'])
            correct = int(row['correct'])
            data[mode][strategy].append((severity, correct))
    
    # Compute accuracy per (mode, strategy, severity)
    plot_data = defaultdict(lambda: defaultdict(dict))
    for mode, strategies in data.items():
        for strategy, items in strategies.items():
            by_sev = defaultdict(list)
            for sev, correct in items:
                by_sev[sev].append(correct)
            for sev, corrects in by_sev.items():
                plot_data[mode][strategy][sev] = sum(corrects) / len(corrects)
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    modes = ['img_only', 'cosine_only', 'hybrid']
    titles = ['IMG Only', 'Cosine Only', 'Hybrid']
    colors = ['#6366f1', '#10b981', '#f59e0b']
    
    for ax, mode, title, color in zip(axes, modes, titles, colors):
        for strategy, sev_acc in plot_data[mode].items():
            sevs = sorted(sev_acc.keys())
            accs = [sev_acc[s] for s in sevs]
            ax.plot(sevs, accs, marker='o', label=strategy, linewidth=2)
        ax.set_xlabel('Severity')
        ax.set_title(title)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1.05)
    
    axes[0].set_ylabel('Accuracy')
    plt.tight_layout()
    output_path = output_dir / 'robustness_by_mode.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def plot_mode_comparison_bar(csv_path: Path, output_dir: Path) -> None:
    """Plot overall accuracy comparison bar chart."""
    import csv
    from collections import defaultdict
    
    mode_correct = defaultdict(list)
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            mode_correct[row['mode']].append(int(row['correct']))
    
    modes = []
    accuracies = []
    for mode in ['img_only', 'cosine_only', 'hybrid']:
        if mode in mode_correct:
            modes.append(mode.replace('_', ' ').title())
            accuracies.append(sum(mode_correct[mode]) / len(mode_correct[mode]))
    
    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(modes, accuracies, color=['#6366f1', '#10b981', '#f59e0b'], edgecolor='black', linewidth=0.5)
    
    # Add value labels on bars
    for bar, acc in zip(bars, accuracies):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                f'{acc:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax.set_ylabel('Accuracy')
    ax.set_title('Overall Accuracy by Mode')
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_path = output_dir / 'mode_comparison_bar.png'
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {output_path}")


def main() -> None:
    output_dir = Path('benchmark_plots')
    output_dir.mkdir(exist_ok=True)
    
    # Plot threshold sweep
    threshold_csv = Path('threshold_sweep.csv')
    if threshold_csv.exists():
        plot_threshold_sweep(threshold_csv, output_dir)
    else:
        print(f"Warning: {threshold_csv} not found, skipping threshold sweep plot")
    
    # Plot robustness
    robustness_csv = Path('robustness_by_mode.csv')
    if robustness_csv.exists():
        plot_robustness(robustness_csv, output_dir)
    else:
        print(f"Warning: {robustness_csv} not found, skipping robustness plot")
    
    # Plot mode comparison
    mode_csv = Path('benchmark_mode_comparison.csv')
    if mode_csv.exists():
        plot_mode_comparison_bar(mode_csv, output_dir)
    else:
        print(f"Warning: {mode_csv} not found, skipping mode comparison plot")
    
    print(f"\nAll plots saved to {output_dir}/")


if __name__ == "__main__":
    main()
