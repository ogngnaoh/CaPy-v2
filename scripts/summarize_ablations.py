"""Ablation summary generation (FR-9.2).

Usage:
    python scripts/summarize_ablations.py
    python scripts/summarize_ablations.py --input results/ablation_runs.jsonl --output-dir results
"""

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from src.utils.logging import get_logger

logger = get_logger(__name__)

# Ordered config list for consistent display
CONFIG_ORDER = ["B0", "B1", "B2", "B3", "B4", "B5", "B6", "T1"]

# Key metrics to include in summary CSV
KEY_METRICS = [
    "compound/mean_R@10",
    "compound/mol->morph/R@10",
    "compound/morph->mol/R@10",
    "compound/mol->expr/R@10",
    "compound/expr->mol/R@10",
    "compound/morph->expr/R@10",
    "compound/expr->morph/R@10",
    "mean_R@10",
    "best_epoch",
    "val_loss",
]


def load_runs(jsonl_path: Path) -> pd.DataFrame:
    """Load ablation_runs.jsonl into DataFrame."""
    runs = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if line:
                runs.append(json.loads(line))
    if not runs:
        raise ValueError(f"No runs found in {jsonl_path}")
    return pd.DataFrame(runs)


def compute_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean +/- std per metric, grouped by config.

    Returns DataFrame with columns: config, metric, mean, std, count.
    """
    # Determine which key metrics exist in the data
    available_metrics = [m for m in KEY_METRICS if m in df.columns]
    # Also include alignment/uniformity metrics
    extra_metrics = [c for c in df.columns if c.startswith(("align_", "uniform_"))]
    all_metrics = available_metrics + extra_metrics

    rows = []
    for config in CONFIG_ORDER:
        config_df = df[df["config"] == config]
        if config_df.empty:
            continue
        for metric in all_metrics:
            if metric not in config_df.columns:
                continue
            values = config_df[metric].dropna()
            if values.empty:
                continue
            rows.append(
                {
                    "config": config,
                    "metric": metric,
                    "mean": values.mean(),
                    "std": values.std(ddof=1) if len(values) > 1 else float("nan"),
                    "count": len(values),
                }
            )
    return pd.DataFrame(rows)


def run_statistical_tests(df: pd.DataFrame) -> list[dict]:
    """Welch's t-test: T1 vs B4, T1 vs B5, T1 vs B6 on compound/mean_R@10.

    Applies Bonferroni correction (3 comparisons, alpha=0.05).
    """
    metric = "compound/mean_R@10"
    t1_vals = df[df["config"] == "T1"][metric].dropna().values
    if len(t1_vals) < 2:
        logger.warning("T1 has fewer than 2 runs, cannot perform t-test")
        return []

    comparisons = []
    n_comparisons = 3
    for baseline in ["B4", "B5", "B6"]:
        bl_vals = df[df["config"] == baseline][metric].dropna().values
        if len(bl_vals) < 2:
            logger.warning(
                f"{baseline} has fewer than 2 runs, skipping t-test for T1 vs {baseline}"
            )
            continue
        t_stat, p_val = stats.ttest_ind(t1_vals, bl_vals, equal_var=False)
        p_corrected = min(p_val * n_comparisons, 1.0)
        stars = ""
        if p_val < 0.001:
            stars = "***"
        elif p_val < 0.01:
            stars = "**"
        elif p_val < 0.05:
            stars = "*"
        comparisons.append(
            {
                "comparison": f"T1 vs {baseline}",
                "t1_mean": t1_vals.mean(),
                "t1_std": t1_vals.std(ddof=1),
                "bl_mean": bl_vals.mean(),
                "bl_std": bl_vals.std(ddof=1),
                "diff": t1_vals.mean() - bl_vals.mean(),
                "t_stat": t_stat,
                "p_value": p_val,
                "p_corrected": p_corrected,
                "significant": p_corrected < 0.05,
                "stars": stars,
            }
        )
    return comparisons


def generate_csv(summary_df: pd.DataFrame, output_path: Path) -> None:
    """Generate ablation_summary.csv — pivoted with config as rows, metrics as columns."""
    if summary_df.empty:
        logger.warning("No summary data, skipping CSV generation")
        return

    # Create a pivoted view: config × metric with "mean ± std" strings
    rows = []
    for config in CONFIG_ORDER:
        config_data = summary_df[summary_df["config"] == config]
        if config_data.empty:
            continue
        row = {"config": config}
        for _, r in config_data.iterrows():
            metric = r["metric"]
            if np.isnan(r["std"]):
                row[metric] = f"{r['mean']:.4f}"
            else:
                row[metric] = f"{r['mean']:.4f} ± {r['std']:.4f}"
        rows.append(row)

    out_df = pd.DataFrame(rows)
    out_df.to_csv(output_path, index=False)
    logger.info(f"Saved summary CSV: {output_path} ({len(out_df)} rows)")


def generate_latex(
    df: pd.DataFrame, comparisons: list[dict], output_path: Path
) -> None:
    """Generate LaTeX comparison table with significance stars."""
    metric = "compound/mean_R@10"

    # Build lookup of comparisons
    comp_lookup = {}
    for c in comparisons:
        baseline = c["comparison"].split(" vs ")[1]
        comp_lookup[baseline] = c

    lines = [
        r"\begin{table}[htbp]",
        r"\centering",
        r"\caption{Ablation comparison: compound mean R@10 (\%) with Welch's t-test (T1 vs bi-modal).}",
        r"\label{tab:ablation}",
        r"\begin{tabular}{lccc}",
        r"\toprule",
        r"Config & compound mean R@10 & $p$-value & Sig. \\",
        r"\midrule",
    ]

    for config in CONFIG_ORDER:
        config_df = df[df["config"] == config]
        if config_df.empty or metric not in config_df.columns:
            continue
        values = config_df[metric].dropna()
        if values.empty:
            continue

        mean_val = values.mean() * 100
        std_val = values.std(ddof=1) * 100 if len(values) > 1 else float("nan")

        if np.isnan(std_val):
            val_str = f"{mean_val:.1f}"
        else:
            val_str = f"{mean_val:.1f} $\\pm$ {std_val:.1f}"

        if config in comp_lookup:
            c = comp_lookup[config]
            p_str = f"{c['p_value']:.2e}"
            stars = c["stars"]
        elif config == "T1":
            p_str = "---"
            stars = ""
        else:
            p_str = "---"
            stars = ""

        lines.append(f"{config} & {val_str} & {p_str} & {stars} \\\\")

    lines.extend(
        [
            r"\bottomrule",
            r"\end{tabular}",
            r"\end{table}",
        ]
    )

    output_path.write_text("\n".join(lines) + "\n")
    logger.info(f"Saved LaTeX table: {output_path}")


def generate_barplot(df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of compound/mean_R@10 per config with std error bars."""
    metric = "compound/mean_R@10"
    # Compute per-config stats
    configs_present = [c for c in CONFIG_ORDER if c in df["config"].values]
    means = []
    stds = []
    colors = []

    color_map = {
        "B0": "#999999",
        "B1": "#999999",
        "B2": "#999999",
        "B3": "#999999",
        "B4": "#4C72B0",
        "B5": "#5A9BD5",
        "B6": "#7EC8E3",
        "T1": "#55A868",
    }

    for config in configs_present:
        values = df[df["config"] == config][metric].dropna()
        means.append(values.mean() * 100)
        stds.append(values.std(ddof=1) * 100 if len(values) > 1 else 0)
        colors.append(color_map.get(config, "#888888"))

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(configs_present))
    ax.bar(x, means, yerr=stds, capsize=4, color=colors, edgecolor="black", linewidth=0.5)

    # Random baseline reference line
    random_r10 = df["compound/random_R@10"].dropna().iloc[0] * 100 if "compound/random_R@10" in df.columns else 5.6
    ax.axhline(y=random_r10, color="red", linestyle="--", linewidth=1, alpha=0.7, label=f"Random ({random_r10:.1f}%)")

    ax.set_xticks(x)
    ax.set_xticklabels(configs_present)
    ax.set_ylabel("Compound Mean R@10 (%)")
    ax.set_title("Ablation: Compound Mean R@10 by Configuration")
    ax.legend()
    ax.set_ylim(bottom=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved bar plot: {output_path}")


def print_summary(df: pd.DataFrame, comparisons: list[dict]) -> None:
    """Print formatted comparison table to stdout."""
    metric = "compound/mean_R@10"

    print("\n" + "=" * 80)
    print("ABLATION SUMMARY — compound mean R@10")
    print("=" * 80)

    # Per-config summary
    print(f"\n{'Config':<8} {'Seeds':<6} {'mean R@10 (%)':<20} {'± std':<12}")
    print("-" * 50)

    for config in CONFIG_ORDER:
        config_df = df[df["config"] == config]
        if config_df.empty or metric not in config_df.columns:
            continue
        values = config_df[metric].dropna()
        n = len(values)
        mean_val = values.mean() * 100
        std_val = values.std(ddof=1) * 100 if n > 1 else float("nan")
        std_str = f"± {std_val:.2f}" if not np.isnan(std_val) else "N/A"
        print(f"{config:<8} {n:<6} {mean_val:<20.2f} {std_str:<12}")

    # Statistical comparisons
    if comparisons:
        print(f"\n{'Comparison':<16} {'Diff (%)':<12} {'t-stat':<10} {'p-value':<12} {'p-corrected':<14} {'Sig':<6}")
        print("-" * 70)
        for c in comparisons:
            diff_pct = c["diff"] * 100
            sig_str = f"{c['stars']}" if c["significant"] else "ns"
            print(
                f"{c['comparison']:<16} {diff_pct:>+8.2f}     {c['t_stat']:<10.3f} {c['p_value']:<12.2e} {c['p_corrected']:<14.2e} {sig_str:<6}"
            )

        # Bonferroni note
        print(f"\nBonferroni correction: α = 0.05/{len(comparisons)} = {0.05/len(comparisons):.4f}")

    print("=" * 80 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Ablation summary generation (FR-9.2)")
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("results/ablation_runs.jsonl"),
        help="Path to ablation_runs.jsonl",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("results"),
        help="Output directory for summary files",
    )
    args = parser.parse_args()

    logger.info(f"Loading ablation runs from {args.input}")
    df = load_runs(args.input)
    logger.info(f"Loaded {len(df)} runs across {df['config'].nunique()} configs")

    # Compute summary statistics
    summary_df = compute_summary(df)

    # Run statistical tests
    comparisons = run_statistical_tests(df)

    # Generate outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)

    generate_csv(summary_df, args.output_dir / "ablation_summary.csv")
    generate_latex(df, comparisons, args.output_dir / "ablation_comparison.tex")
    generate_barplot(df, args.output_dir / "ablation_barplot.png")

    # Print to stdout
    print_summary(df, comparisons)

    logger.info("Ablation summary complete")


if __name__ == "__main__":
    main()
