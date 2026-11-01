
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.stats import wilcoxon


# ==================================================
# STEP 16 — EXPERIMENT 13
# Robustness across six individual-phase baselines
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results" / "experiment_11"
    / "clinvar_anchor_scores.csv"
)

RESULT_DIR = ROOT / "results" / "experiment_13"
FIGURE_DIR = ROOT / "figures" / "experiment_13"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = RESULT_DIR / "phase_baseline_comparison.csv"
SUMMARY = RESULT_DIR / "phase_robustness.json"

SEED = 20261025
BOOTSTRAPS = 2000
PHASES = 6

rng = np.random.default_rng(SEED)

print("STEP 16 — Six-phase baseline robustness")


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def save_figure(name):
    plt.tight_layout()

    plt.savefig(
        FIGURE_DIR / f"{name}.png",
        dpi=300,
        bbox_inches="tight"
    )

    plt.savefig(
        FIGURE_DIR / f"{name}.pdf",
        bbox_inches="tight"
    )

    plt.close()


# ==================================================
# 1. Load and validate existing clinical scores
# ==================================================

with INPUT.open(
    "r",
    newline="",
    encoding="utf-8"
) as handle:
    rows = list(csv.DictReader(handle))

assert len(rows) == 1000

keys = [
    (
        row["chromosome"],
        row["position_grch38_1based"],
        row["ref"],
        row["alt"]
    )
    for row in rows
]

assert len(set(keys)) == 1000

labels = np.array([
    row["clinical_label"].strip().lower()
    for row in rows
])

assert set(labels) == {"benign", "pathogenic"}

assert np.sum(labels == "benign") == 500
assert np.sum(labels == "pathogenic") == 500

print("Variants:", len(rows))


# ==================================================
# 2. Reconstruct scores for three window anchors
# ==================================================

def read_phase_matrix(prefix):
    return np.array([
        [
            float(row[f"{prefix}{phase}"])
            for phase in range(PHASES)
        ]
        for row in rows
    ], dtype=np.float64)


minus6 = read_phase_matrix("minus6_phase_")
center = np.array([
    [
        float(row[f"phase_{phase}_score"])
        for phase in range(PHASES)
    ]
    for row in rows
], dtype=np.float64)
plus6 = read_phase_matrix("plus6_phase_")

assert minus6.shape == (1000, 6)
assert center.shape == (1000, 6)
assert plus6.shape == (1000, 6)

assert np.isfinite(minus6).all()
assert np.isfinite(center).all()
assert np.isfinite(plus6).all()


# ==================================================
# 3. Calculate each individual phase's instability
# ==================================================

# For each phase and variant, calculate the mean
# absolute score change at the two anchor shifts.

individual_changes = (
    np.abs(minus6 - center)
    + np.abs(plus6 - center)
) / 2

assert individual_changes.shape == (1000, 6)

# Training-free six-phase averaging.

average_minus6 = minus6.mean(axis=1)
average_center = center.mean(axis=1)
average_plus6 = plus6.mean(axis=1)

averaged_changes = (
    np.abs(average_minus6 - average_center)
    + np.abs(average_plus6 - average_center)
) / 2

assert averaged_changes.shape == (1000,)

# Verify against the saved Experiment 11 results.

saved_changes = np.array([
    float(
        row["average_mean_absolute_anchor_change"]
    )
    for row in rows
])

assert np.allclose(
    averaged_changes,
    saved_changes,
    atol=1e-5
)

print("Previous stability results reproduced: PASS")


# ==================================================
# 4. Compare averaged scores against all six phases
# ==================================================

improvements = (
    individual_changes
    - averaged_changes[:, None]
)

phase_means = individual_changes.mean(axis=0)

average_mean = float(
    averaged_changes.mean()
)

mean_improvements = improvements.mean(axis=0)

relative_reductions = np.array([
    (
        100 * mean_improvements[i] / phase_means[i]
        if phase_means[i] > 0
        else np.nan
    )
    for i in range(PHASES)
])

positive_counts = (
    improvements > 0
).sum(axis=0)


# ==================================================
# 5. Paired Wilcoxon tests for all six baselines
# ==================================================

raw_p_values = np.array([
    wilcoxon(
        individual_changes[:, phase],
        averaged_changes,
        alternative="greater",
        zero_method="wilcox"
    ).pvalue
    for phase in range(PHASES)
], dtype=np.float64)

# Holm family-wise correction for six comparisons.

order = np.argsort(raw_p_values)

adjusted_p_values = np.empty(PHASES)

running_max = 0.0

for rank, phase in enumerate(order):
    adjusted = min(
        1.0,
        raw_p_values[phase] * (PHASES - rank)
    )

    running_max = max(
        running_max,
        adjusted
    )

    adjusted_p_values[phase] = running_max


# ==================================================
# 6. Paired, class-stratified bootstrap
# ==================================================

benign_indices = np.flatnonzero(
    labels == "benign"
)

pathogenic_indices = np.flatnonzero(
    labels == "pathogenic"
)

bootstrap_means = np.empty(
    (BOOTSTRAPS, PHASES),
    dtype=np.float64
)

for iteration in range(BOOTSTRAPS):
    sampled_benign = rng.choice(
        benign_indices,
        size=len(benign_indices),
        replace=True
    )

    sampled_pathogenic = rng.choice(
        pathogenic_indices,
        size=len(pathogenic_indices),
        replace=True
    )

    indices = np.concatenate([
        sampled_benign,
        sampled_pathogenic
    ])

    bootstrap_means[iteration] = (
        improvements[indices].mean(axis=0)
    )

ci_lower = np.percentile(
    bootstrap_means,
    2.5,
    axis=0
)

ci_upper = np.percentile(
    bootstrap_means,
    97.5,
    axis=0
)


# ==================================================
# 7. Prepare complete per-phase comparison
# ==================================================

results = []

for phase in range(PHASES):
    result = {
        "phase": phase,
        "individual_mean_absolute_change": float(
            phase_means[phase]
        ),
        "averaged_mean_absolute_change": average_mean,
        "mean_paired_improvement": float(
            mean_improvements[phase]
        ),
        "relative_reduction_percent": float(
            relative_reductions[phase]
        ),
        "variants_improved": int(
            positive_counts[phase]
        ),
        "improvement_ci95_lower": float(
            ci_lower[phase]
        ),
        "improvement_ci95_upper": float(
            ci_upper[phase]
        ),
        "wilcoxon_p": float(
            raw_p_values[phase]
        ),
        "holm_adjusted_p": float(
            adjusted_p_values[phase]
        )
    }

    results.append(result)

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(results[0].keys())
    )

    writer.writeheader()
    writer.writerows(results)


# ==================================================
# 8. Analyze benign and pathogenic subgroups
# ==================================================

subgroups = {}

for label in ("benign", "pathogenic"):
    mask = labels == label

    subgroup_individual = (
        individual_changes[mask].mean(axis=0)
    )

    subgroup_averaged = float(
        averaged_changes[mask].mean()
    )

    subgroups[label] = {
        "variants": int(mask.sum()),
        "individual_phase_changes": [
            float(value)
            for value in subgroup_individual
        ],
        "averaged_change": subgroup_averaged,
        "improvement_against_each_phase": [
            float(value - subgroup_averaged)
            for value in subgroup_individual
        ]
    }


# ==================================================
# 9. Figure: Stability of all six baselines
# ==================================================

plt.figure(figsize=(9, 5))

plt.bar(
    [f"Phase {i}" for i in range(PHASES)],
    phase_means
)

plt.axhline(
    average_mean,
    linestyle="--",
    linewidth=2,
    label=(
        "Six-phase average "
        f"({average_mean:.3f})"
    )
)

plt.ylabel("Mean absolute score change")
plt.title("Clinical score stability across all phases")
plt.legend()

save_figure("all_phase_stability_comparison")


# ==================================================
# 10. Figure: Improvement confidence intervals
# ==================================================

plt.figure(figsize=(8, 5))

positions = np.arange(PHASES)

plt.errorbar(
    positions,
    mean_improvements,
    yerr=[
        mean_improvements - ci_lower,
        ci_upper - mean_improvements
    ],
    fmt="o",
    capsize=5
)

plt.axhline(
    0,
    linestyle="--",
    linewidth=1
)

plt.xticks(
    positions,
    [f"Phase {i}" for i in range(PHASES)]
)

plt.ylabel(
    "Individual-phase instability minus "
    "averaged instability"
)

plt.title(
    "Paired stability improvements with "
    "95% bootstrap CIs"
)

save_figure("phase_improvement_confidence_intervals")


# ==================================================
# 11. Save reproducibility metadata
# ==================================================

all_phase_means_greater = bool(
    np.all(
        phase_means > average_mean
    )
)

all_ci_positive = bool(
    np.all(ci_lower > 0)
)

all_holm_significant = bool(
    np.all(adjusted_p_values < 0.05)
)

summary = {
    "experiment": "13_phase_robustness",
    "research_step": 16,
    "seed": SEED,
    "variants": len(rows),
    "anchor_shifts_nt": [-6, 0, 6],
    "phases": PHASES,
    "bootstrap_replicates": BOOTSTRAPS,
    "bootstrap_method": (
        "Paired, class-stratified variant-level "
        "bootstrap"
    ),
    "multiple_testing_correction": "Holm",
    "averaged_mean_absolute_change": average_mean,
    "individual_phase_mean_changes": [
        float(value)
        for value in phase_means
    ],
    "all_phase_means_greater_than_average": (
        all_phase_means_greater
    ),
    "all_improvement_ci95_lower_above_zero": (
        all_ci_positive
    ),
    "all_holm_adjusted_p_below_0_05": (
        all_holm_significant
    ),
    "phase_comparisons": results,
    "clinical_subgroups": subgroups,
    "input_sha256": sha256(INPUT),
    "output_sha256": sha256(OUTPUT),
    "script_sha256": sha256(Path(__file__)),
    "limitations": [
        "Averaging is a smoothing operation; "
        "reduced numerical variability alone "
        "does not establish biological validity",
        "Six-phase averaging requires six model "
        "evaluations per anchor",
        "Comparisons use the same variants and "
        "scores as previous experiments",
        "Window shifts also change sequence context",
        "Shared genes or loci can create dependence "
        "between variants",
        "These are exploratory follow-up comparisons"
    ]
}

SUMMARY.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)


# ==================================================
# 12. Final terminal report
# ==================================================

print("\nSTEP 16 COMPLETE")
print(
    "Six-phase averaged instability:",
    round(average_mean, 6)
)

for result in results:
    print(
        f"Phase {result['phase']}: "
        f"instability="
        f"{result['individual_mean_absolute_change']:.6f}, "
        f"reduction="
        f"{result['relative_reduction_percent']:.2f}%, "
        f"improved="
        f"{result['variants_improved']}/1000, "
        f"Holm p="
        f"{result['holm_adjusted_p']:.3e}"
    )

print(
    "Better than all phase means:",
    all_phase_means_greater
)

print(
    "All bootstrap CIs positive:",
    all_ci_positive
)

print(
    "All Holm-adjusted p-values < 0.05:",
    all_holm_significant
)

print("Results:", OUTPUT)
print("Summary:", SUMMARY)
print("Figures:", FIGURE_DIR)
