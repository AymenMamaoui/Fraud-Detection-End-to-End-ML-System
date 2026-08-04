import sys
from pathlib import Path
import matplotlib.pyplot as plt
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp, chi2_contingency

from src import config
from src.data import load_data


def detect_numeric_drift(ref, cur):
    """Kolmogorov-Smirnov test for a numeric feature."""
    ref = ref.dropna()
    cur = cur.dropna()
    if len(ref) == 0 or len(cur) == 0:
        return None
    stat, p_value = ks_2samp(ref, cur)
    return {"test": "KS", "statistic": stat, "p_value": p_value}


def detect_categorical_drift(ref, cur):
    """Chi-squared test for a categorical feature."""
    # Align categories across both periods
    ref_counts = ref.value_counts()
    cur_counts = cur.value_counts()
    all_categories = ref_counts.index.union(cur_counts.index)

    ref_freq = ref_counts.reindex(all_categories, fill_value=0)
    cur_freq = cur_counts.reindex(all_categories, fill_value=0)

    table = np.array([ref_freq.values, cur_freq.values])
    # Drop columns that are all zeros (safety)
    table = table[:, table.sum(axis=0) > 0]
    if table.shape[1] < 2:
        return None

    stat, p_value, _, _ = chi2_contingency(table)
    return {"test": "Chi2", "statistic": stat, "p_value": p_value}


def generate_drift_report(threshold=0.05):
    """
    Compare an early (reference) period against a later (current) period,
    simulating the drift a deployed model would face over time.
    Uses TransactionDT for a chronological split, consistent with training.
    """
    print("Loading data...")
    df = load_data()
    df = df.sort_values("TransactionDT").reset_index(drop=True)

    split_idx = len(df) // 2
    reference = df.iloc[:split_idx]
    current = df.iloc[split_idx:]

    print(f"Reference period: {len(reference):,} transactions")
    print(f"Current period:   {len(current):,} transactions\n")

    numeric_features = ["TransactionAmt"] + config.C_FEATURES
    categorical_features = config.CATEGORICAL_FEATURES

    results = []

    for col in numeric_features:
        if col in df.columns:
            r = detect_numeric_drift(reference[col], current[col])
            if r:
                results.append({"feature": col, **r})

    for col in categorical_features:
        if col in df.columns:
            r = detect_categorical_drift(reference[col], current[col])
            if r:
                results.append({"feature": col, **r})

    report = pd.DataFrame(results)
    report["drift_detected"] = report["p_value"] < threshold
    report = report.sort_values("p_value").reset_index(drop=True)

    # Summary
    n_drifted = report["drift_detected"].sum()
    print("=" * 60)
    print("DATA DRIFT REPORT")
    print("=" * 60)
    print(report.to_string(index=False))
    print("=" * 60)
    print(f"Features with drift detected: {n_drifted} / {len(report)}")
    print(f"Drift threshold (p-value): {threshold}")

    figures_dir = config.ROOT / "notebooks" / "figures"
    plot_drift(reference, current, report, figures_dir)

    return report

def plot_drift(reference, current, report, output_dir, top_n=6):
    """
    Plot reference vs current distributions for the most-drifted features,
    so the drift is visible at a glance rather than read from p-values.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Take the top drifted features (lowest p-value = strongest evidence)
    top_features = report.sort_values("p_value").head(top_n)["feature"].tolist()

    n = len(top_features)
    ncols = 3
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(15, 4 * nrows))
    axes = axes.flatten()

    for i, feature in enumerate(top_features):
        ax = axes[i]
        ref_vals = reference[feature]
        cur_vals = current[feature]

        # Numeric vs categorical handling
        if ref_vals.dtype.kind in "biufc":  # numeric
            # Clip extreme values for readability (99th percentile)
            upper = np.nanpercentile(pd.concat([ref_vals, cur_vals]), 99)
            bins = np.linspace(0, upper, 40)
            ax.hist(ref_vals.clip(upper=upper), bins=bins, alpha=0.6,
                    label="Reference", density=True, color="steelblue")
            ax.hist(cur_vals.clip(upper=upper), bins=bins, alpha=0.6,
                    label="Current", density=True, color="darkorange")
        else:  # categorical
            top_cats = (
                pd.concat([ref_vals, cur_vals]).value_counts().head(8).index
            )
            ref_prop = ref_vals.value_counts(normalize=True).reindex(top_cats, fill_value=0)
            cur_prop = cur_vals.value_counts(normalize=True).reindex(top_cats, fill_value=0)
            x = np.arange(len(top_cats))
            ax.bar(x - 0.2, ref_prop.values, 0.4, label="Reference", color="steelblue")
            ax.bar(x + 0.2, cur_prop.values, 0.4, label="Current", color="darkorange")
            ax.set_xticks(x)
            ax.set_xticklabels(top_cats, rotation=45, ha="right", fontsize=8)

        ax.set_title(feature)
        ax.legend(fontsize=8)

    # Hide any unused subplots
    for j in range(n, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    out_path = output_dir / "drift_distributions.png"
    plt.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close()
    print(f"\nDrift visualization saved to {out_path}")

if __name__ == "__main__":
    generate_drift_report()