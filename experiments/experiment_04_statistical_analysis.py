
import hashlib
import json
import platform
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "experiment_03"
RESULTS = ROOT / "results" / "experiment_04"
FIGURES = ROOT / "figures" / "experiment_04"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 20261025
N_BOOTSTRAP = 10000
rng = np.random.default_rng(SEED)

same_file = SOURCE / "same_phase_controls.csv"
different_file = SOURCE / "different_phase_pairs.csv"

same = pd.read_csv(same_file)
different = pd.read_csv(different_file)

# Treat each independent synthetic variant as one unit.
same_long = same.melt(
    id_vars=["variant_id", "phase"],
    value_vars=["abs_delta_minus6", "abs_delta_plus6"],
    value_name="absolute_difference"
)

same_means = (
    same_long.groupby("variant_id")["absolute_difference"]
    .mean()
    .rename("same_phase_mean")
)

different_means = (
    different.groupby("variant_id")["absolute_score_difference"]
    .mean()
    .rename("different_phase_mean")
)

paired = pd.concat(
    [same_means, different_means], axis=1
).dropna().sort_index()

assert len(paired) == 60
assert paired.index.is_unique

paired["difference"] = (
    paired["different_phase_mean"]
    - paired["same_phase_mean"]
)

paired.to_csv(
    RESULTS / "per_variant_paired_comparison.csv"
)

values = paired["difference"].to_numpy()
n = len(values)

# Resample independent variants, not individual phase pairs.
bootstrap_indices = rng.integers(
    0, n, size=(N_BOOTSTRAP, n)
)
bootstrap_means = values[bootstrap_indices].mean(axis=1)

ci_low, ci_high = np.quantile(
    bootstrap_means, [0.025, 0.975]
)

test = wilcoxon(
    paired["different_phase_mean"],
    paired["same_phase_mean"],
    alternative="greater",
    zero_method="wilcox"
)

mean_same = float(paired["same_phase_mean"].mean())
mean_different = float(paired["different_phase_mean"].mean())

summary = {
    "experiment": "04_variant_level_statistical_analysis",
    "n_independent_variants": n,
    "bootstrap_iterations": N_BOOTSTRAP,
    "random_seed": SEED,
    "mean_same_phase_difference": mean_same,
    "mean_different_phase_difference": mean_different,
    "ratio_of_condition_means": (
        mean_different / mean_same
        if mean_same != 0 else None
    ),
    "mean_paired_difference": float(values.mean()),
    "bootstrap_95_ci": [
        float(ci_low), float(ci_high)
    ],
    "wilcoxon_statistic": float(test.statistic),
    "wilcoxon_one_sided_p": float(test.pvalue),
    "variants_with_larger_different_phase_effect": int(
        (values > 0).sum()
    ),
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "pandas_version": pd.__version__,
    "scipy_version": scipy.__version__,
    "source_sha256": {
        file.name: hashlib.sha256(
            file.read_bytes()
        ).hexdigest()
        for file in [same_file, different_file]
    },
    "script_sha256": hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest(),
    "interpretation_limitations": [
        "Synthetic variants only",
        "Same-phase and different-phase shift distances differ",
        "Window context and token position are not fully controlled",
        "Statistical significance cannot establish causality",
        "No biological ground-truth labels"
    ]
}

(RESULTS / "statistical_summary.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

# Presentation-ready paired comparison figure.
fig, ax = plt.subplots(figsize=(8, 5))

for _, row in paired.iterrows():
    ax.plot(
        [0, 1],
        [row["same_phase_mean"],
         row["different_phase_mean"]],
        color="gray",
        alpha=0.22,
        linewidth=0.8
    )

ax.plot(
    [0, 1],
    [mean_same, mean_different],
    marker="o",
    linewidth=3,
    color="black",
    label="Mean across variants"
)

ax.set_xticks([0, 1])
ax.set_xticklabels([
    "Same phase\n(±6 nt)",
    "Different phases\n(1–5 nt)"
])
ax.set_ylabel("Mean absolute score difference")
ax.set_title("Paired Window-Shift Comparison Across 60 Variants")
ax.legend()
ax.grid(axis="y", alpha=0.2)
fig.tight_layout()

fig.savefig(
    FIGURES / "paired_variant_comparison.png",
    dpi=300
)
fig.savefig(
    FIGURES / "paired_variant_comparison.pdf"
)
plt.close(fig)

print("\nSTEP 07 — STATISTICAL ANALYSIS COMPLETE")
print("Independent variants:", n)
print("Same-phase mean:", round(mean_same, 6))
print("Different-phase mean:", round(mean_different, 6))
print("Ratio:", round(summary["ratio_of_condition_means"], 3))
print("Mean paired difference:", round(values.mean(), 6))
print("95% bootstrap CI:", (round(ci_low, 6), round(ci_high, 6)))
print("One-sided Wilcoxon p-value:", test.pvalue)
print("Variants with larger different-phase effect:",
      summary["variants_with_larger_different_phase_effect"])
print("Results:", RESULTS)
print("Figures:", FIGURES)
