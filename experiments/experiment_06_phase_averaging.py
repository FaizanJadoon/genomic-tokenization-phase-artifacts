
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "experiment_03" / "all_control_scores.csv"
RESULTS = ROOT / "results" / "experiment_06"
FIGURES = ROOT / "figures" / "experiment_06"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 20261025
N_BOOTSTRAP = 10000
rng = np.random.default_rng(SEED)

print("Step 09: Training-free phase averaging")

df = pd.read_csv(SOURCE)

assert len(df) == 1080
assert df["variant_id"].nunique() == 60
assert not df.duplicated(
    ["variant_id", "phase", "offset"]
).any()

# Each variant has six phases at three anchor offsets.
counts = df.groupby(
    ["variant_id", "offset"]
)["phase"].nunique()

assert (counts == 6).all()

# Compute the training-free corrected scores.
corrected = df.groupby(
    ["variant_id", "offset"]
)["score"].agg(
    phase_mean="mean",
    phase_median="median"
).reset_index()

corrected.to_csv(
    RESULTS / "phase_averaged_scores.csv",
    index=False
)

# Compare adjacent anchors: -6 -> 0 and 0 -> +6.
# For raw scores, compare the same phase at both anchors.
raw_wide = df.pivot(
    index=["variant_id", "phase"],
    columns="offset",
    values="score"
)

raw_changes = pd.DataFrame({
    "variant_id": raw_wide.index.get_level_values(0),
    "phase": raw_wide.index.get_level_values(1),
    "change_minus6_to_0": (
        raw_wide[0] - raw_wide[-6]
    ).abs().to_numpy(),
    "change_0_to_plus6": (
        raw_wide[6] - raw_wide[0]
    ).abs().to_numpy()
})

raw_changes["mean_change"] = raw_changes[
    ["change_minus6_to_0", "change_0_to_plus6"]
].mean(axis=1)

raw_per_variant = raw_changes.groupby(
    "variant_id"
)["mean_change"].mean()

# Repeat the same anchor comparisons after phase averaging.
corrected_wide = corrected.pivot(
    index="variant_id",
    columns="offset",
    values="phase_mean"
)

corrected_per_variant = (
    (
        (corrected_wide[0] - corrected_wide[-6]).abs()
        + (corrected_wide[6] - corrected_wide[0]).abs()
    ) / 2
)

comparison = pd.DataFrame({
    "raw_mean_absolute_change": raw_per_variant,
    "corrected_mean_absolute_change": corrected_per_variant
})

comparison["improvement"] = (
    comparison["raw_mean_absolute_change"]
    - comparison["corrected_mean_absolute_change"]
)

comparison.to_csv(
    RESULTS / "per_variant_stability.csv"
)

assert len(comparison) == 60
assert not comparison.isna().any().any()

raw_mean = float(
    comparison["raw_mean_absolute_change"].mean()
)
corrected_mean = float(
    comparison["corrected_mean_absolute_change"].mean()
)

reduction_pct = (
    100 * (raw_mean - corrected_mean) / raw_mean
)

improvements = comparison["improvement"].to_numpy()
n = len(improvements)

indices = rng.integers(
    0, n, size=(N_BOOTSTRAP, n)
)

bootstrap = improvements[indices].mean(axis=1)
ci_low, ci_high = np.quantile(
    bootstrap, [0.025, 0.975]
)

test = wilcoxon(
    comparison["raw_mean_absolute_change"],
    comparison["corrected_mean_absolute_change"],
    alternative="greater"
)

summary = {
    "experiment": "06_training_free_phase_averaging",
    "independent_variants": n,
    "phases_per_anchor": 6,
    "anchor_offsets": [-6, 0, 6],
    "method": "arithmetic_mean_of_six_phase_scores",
    "raw_mean_absolute_anchor_change": raw_mean,
    "corrected_mean_absolute_anchor_change": corrected_mean,
    "relative_reduction_percent": float(reduction_pct),
    "mean_paired_improvement": float(
        improvements.mean()
    ),
    "bootstrap_95_ci": [
        float(ci_low), float(ci_high)
    ],
    "bootstrap_iterations": N_BOOTSTRAP,
    "one_sided_wilcoxon_p": float(test.pvalue),
    "variants_with_improvement": int(
        (improvements > 0).sum()
    ),
    "seed": SEED,
    "source_sha256": hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest(),
    "script_sha256": hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest(),
    "limitations": [
        "Synthetic sequences only",
        "Overlapping sequence windows",
        "Averaging can reduce variability by construction",
        "No independent biological ground-truth labels",
        "No evidence yet of improved variant classification",
        "Phase averaging is evaluated for stability only"
    ]
}

(RESULTS / "metadata.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

fig, ax = plt.subplots(figsize=(7, 6))

ax.scatter(
    comparison["raw_mean_absolute_change"],
    comparison["corrected_mean_absolute_change"],
    alpha=0.7
)

limit = float(
    comparison[
        [
            "raw_mean_absolute_change",
            "corrected_mean_absolute_change"
        ]
    ].max().max()
) * 1.05

ax.plot(
    [0, limit], [0, limit],
    linestyle="--", color="gray",
    label="No change"
)

ax.set_xlim(0, limit)
ax.set_ylim(0, limit)
ax.set_xlabel("Raw mean absolute anchor change")
ax.set_ylabel("Phase-averaged mean absolute anchor change")
ax.set_title("Training-Free Phase Averaging: Stability Comparison")
ax.legend()
ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    FIGURES / "phase_averaging_stability.png",
    dpi=300
)
fig.savefig(
    FIGURES / "phase_averaging_stability.pdf"
)
plt.close(fig)

print("\nEXPERIMENT 6 COMPLETE")
print("Variants:", n)
print("Raw mean absolute change:", round(raw_mean, 6))
print("Corrected mean absolute change:", round(corrected_mean, 6))
print("Relative reduction (%):", round(reduction_pct, 2))
print("Mean paired improvement:", round(improvements.mean(), 6))
print(
    "95% bootstrap CI:",
    (round(float(ci_low), 6), round(float(ci_high), 6))
)
print("One-sided Wilcoxon p-value:", test.pvalue)
print(
    "Variants improved:",
    summary["variants_with_improvement"]
)
print("Results:", RESULTS)
print("Figures:", FIGURES)
