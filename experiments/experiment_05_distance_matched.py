
import hashlib
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results" / "experiment_03" / "all_control_scores.csv"
RESULTS = ROOT / "results" / "experiment_05"
FIGURES = ROOT / "figures" / "experiment_05"

RESULTS.mkdir(parents=True, exist_ok=True)
FIGURES.mkdir(parents=True, exist_ok=True)

SEED = 20261025
N_BOOTSTRAP = 10000
rng = np.random.default_rng(SEED)

print("Step 08: Distance-matched tokenization control")

df = pd.read_csv(SOURCE)

assert len(df) == 1080
assert df["variant_id"].nunique() == 60

records = []

for variant_id, group in df.groupby("variant_id"):
    group = group.sort_values("window_start")

    assert len(group) == 18
    assert group["window_start"].is_unique

    scores = dict(
        zip(group["window_start"], group["score"])
    )

    for distance in (5, 6, 7):
        for start in sorted(scores):
            end = start + distance

            if end not in scores:
                continue

            records.append({
                "variant_id": int(variant_id),
                "window_start_a": int(start),
                "window_start_b": int(end),
                "shift_distance": distance,
                "same_phase": distance % 6 == 0,
                "absolute_score_difference": abs(
                    scores[start] - scores[end]
                )
            })

pairs = pd.DataFrame(records)

assert len(pairs) == 60 * (13 + 12 + 11)

pairs.to_csv(
    RESULTS / "distance_matched_pairs.csv",
    index=False
)

per_variant = pairs.pivot_table(
    index="variant_id",
    columns="shift_distance",
    values="absolute_score_difference",
    aggfunc="mean"
)

per_variant.columns = [
    f"distance_{distance}"
    for distance in per_variant.columns
]

assert len(per_variant) == 60
assert not per_variant.isna().any().any()

per_variant["different_phase_mean"] = (
    per_variant["distance_5"]
    + per_variant["distance_7"]
) / 2

per_variant["same_phase_mean"] = (
    per_variant["distance_6"]
)

per_variant["paired_contrast"] = (
    per_variant["different_phase_mean"]
    - per_variant["same_phase_mean"]
)

per_variant.to_csv(
    RESULTS / "per_variant_distance_matched.csv"
)

values = per_variant["paired_contrast"].to_numpy()
n = len(values)

indices = rng.integers(
    0, n, size=(N_BOOTSTRAP, n)
)

bootstrap_means = values[indices].mean(axis=1)

ci_low, ci_high = np.quantile(
    bootstrap_means, [0.025, 0.975]
)

test = wilcoxon(
    per_variant["different_phase_mean"],
    per_variant["same_phase_mean"],
    alternative="greater"
)

distance_means = {
    str(distance): float(
        per_variant[f"distance_{distance}"].mean()
    )
    for distance in (5, 6, 7)
}

summary = {
    "experiment": "05_distance_matched_control",
    "independent_variants": n,
    "total_window_pairs": len(pairs),
    "shift_distances": [5, 6, 7],
    "mean_absolute_differences": distance_means,
    "mean_paired_contrast": float(values.mean()),
    "bootstrap_95_ci": [
        float(ci_low),
        float(ci_high)
    ],
    "bootstrap_iterations": N_BOOTSTRAP,
    "seed": SEED,
    "one_sided_wilcoxon_p": float(test.pvalue),
    "variants_with_positive_contrast": int(
        (values > 0).sum()
    ),
    "source_sha256": hashlib.sha256(
        SOURCE.read_bytes()
    ).hexdigest(),
    "script_sha256": hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest(),
    "limitations": [
        "Shift distances are closely matched, not identical",
        "Phase and shift distance cannot be fully separated",
        "Window context and token position still change",
        "Synthetic sequences without biological labels",
        "Exploratory analysis; not independent confirmation"
    ]
}

(RESULTS / "metadata.json").write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

fig, ax = plt.subplots(figsize=(8, 5))

distances = [5, 6, 7]
means = [
    distance_means[str(d)]
    for d in distances
]

ax.plot(distances, means, marker="o", linewidth=2)
ax.set_xticks(distances)
ax.set_xlabel("Window-shift distance (nucleotides)")
ax.set_ylabel("Mean absolute score difference")
ax.set_title("Distance-Matched Phase Sensitivity Control")
ax.grid(alpha=0.2)

fig.tight_layout()

fig.savefig(
    FIGURES / "distance_matched_control.png",
    dpi=300
)
fig.savefig(
    FIGURES / "distance_matched_control.pdf"
)

plt.close(fig)

print("\nEXPERIMENT 5 COMPLETE")
print("Variants:", n)
print("Window pairs:", len(pairs))

for distance in distances:
    print(
        f"Distance {distance}: "
        f"{distance_means[str(distance)]:.6f}"
    )

print("Mean paired contrast:", round(values.mean(), 6))
print(
    "95% bootstrap CI:",
    (round(float(ci_low), 6),
     round(float(ci_high), 6))
)
print("One-sided Wilcoxon p-value:", test.pvalue)
print(
    "Positive contrasts:",
    summary["variants_with_positive_contrast"]
)
print("Results:", RESULTS)
print("Figures:", FIGURES)
