
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
# STEP 15 — EXPERIMENT 12
# Distance-matched clinical phase controls
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results" / "experiment_11"
    / "clinvar_anchor_scores.csv"
)

RESULT_DIR = ROOT / "results" / "experiment_12"
FIGURE_DIR = ROOT / "figures" / "experiment_12"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = RESULT_DIR / "distance_matched_variants.csv"
SUMMARY = RESULT_DIR / "distance_matched_analysis.json"

SEED = 20261025
BOOTSTRAPS = 2000

rng = np.random.default_rng(SEED)

print("STEP 15 — Distance-matched clinical controls")


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
# 1. Load previously generated model scores
# ==================================================

with INPUT.open(
    "r",
    newline="",
    encoding="utf-8"
) as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)

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

print("Variants:", len(rows))


# ==================================================
# 2. Reconstruct 18 consecutive window positions
# ==================================================

# Window starts relative to BASE_START=48:
#
# minus6 anchor: -6, -5, -4, -3, -2, -1
# center anchor:  0,  1,  2,  3,  4,  5
# plus6 anchor:   6,  7,  8,  9, 10, 11

columns = (
    [
        f"minus6_phase_{i}"
        for i in range(6)
    ]
    + [
        f"phase_{i}_score"
        for i in range(6)
    ]
    + [
        f"plus6_phase_{i}"
        for i in range(6)
    ]
)

scores = np.array([
    [
        float(row[column])
        for column in columns
    ]
    for row in rows
], dtype=np.float64)

assert scores.shape == (1000, 18)
assert np.isfinite(scores).all()

# Verify alignment with previously saved stability data.
raw_anchor_change = (
    np.abs(scores[:, 0] - scores[:, 6])
    + np.abs(scores[:, 12] - scores[:, 6])
) / 2

saved_anchor_change = np.array([
    float(row["raw_mean_absolute_anchor_change"])
    for row in rows
])

assert np.allclose(
    raw_anchor_change,
    saved_anchor_change,
    atol=1e-5
)

print("Existing score alignment: PASS")


# ==================================================
# 3. Compare window shifts of 5, 6, and 7 nt
# ==================================================

distance_scores = {}

for distance in (5, 6, 7):
    differences = np.abs(
        scores[:, distance:]
        - scores[:, :-distance]
    )

    # Average all valid pairs per variant.
    # Variants, not overlapping window pairs,
    # are the units used in statistical tests.
    distance_scores[distance] = (
        differences.mean(axis=1)
    )

    print(
        f"Distance {distance} nt: "
        f"{differences.shape[1]} pairs/variant, "
        f"mean change "
        f"{distance_scores[distance].mean():.6f}"
    )


# ==================================================
# 4. Construct per-variant matched contrast
# ==================================================

different_phase = (
    distance_scores[5]
    + distance_scores[7]
) / 2

same_phase = distance_scores[6]

contrast = different_phase - same_phase

mean_different = float(different_phase.mean())
mean_same = float(same_phase.mean())
mean_contrast = float(contrast.mean())

positive_count = int(
    np.sum(contrast > 0)
)

print("\nDifferent-phase mean:", mean_different)
print("Same-phase mean:", mean_same)
print("Mean paired contrast:", mean_contrast)
print("Positive contrasts:", positive_count)


# ==================================================
# 5. Paired Wilcoxon test
# ==================================================

test = wilcoxon(
    different_phase,
    same_phase,
    alternative="greater",
    zero_method="wilcox"
)


# ==================================================
# 6. Stratified bootstrap confidence interval
# ==================================================

benign_indices = np.flatnonzero(
    labels == "benign"
)

pathogenic_indices = np.flatnonzero(
    labels == "pathogenic"
)

bootstrap_means = np.empty(
    BOOTSTRAPS,
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
        contrast[indices].mean()
    )

ci95 = [
    float(np.percentile(bootstrap_means, 2.5)),
    float(np.percentile(bootstrap_means, 97.5))
]


# ==================================================
# 7. Analyze clinical subgroups
# ==================================================

subgroups = {}

for label in ("benign", "pathogenic"):
    mask = labels == label

    subgroups[label] = {
        "variants": int(mask.sum()),
        "same_phase_mean": float(
            same_phase[mask].mean()
        ),
        "different_phase_mean": float(
            different_phase[mask].mean()
        ),
        "mean_paired_contrast": float(
            contrast[mask].mean()
        ),
        "positive_contrasts": int(
            np.sum(contrast[mask] > 0)
        )
    }


# ==================================================
# 8. Save per-variant results
# ==================================================

fields = [
    "chromosome",
    "position_grch38_1based",
    "ref",
    "alt",
    "clinical_label",
    "distance_5_change",
    "distance_6_change",
    "distance_7_change",
    "different_phase_mean",
    "same_phase_mean",
    "paired_contrast"
]

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=fields
    )

    writer.writeheader()

    for i, row in enumerate(rows):
        writer.writerow({
            "chromosome": row["chromosome"],
            "position_grch38_1based": (
                row["position_grch38_1based"]
            ),
            "ref": row["ref"],
            "alt": row["alt"],
            "clinical_label": row["clinical_label"],
            "distance_5_change": (
                float(distance_scores[5][i])
            ),
            "distance_6_change": (
                float(distance_scores[6][i])
            ),
            "distance_7_change": (
                float(distance_scores[7][i])
            ),
            "different_phase_mean": (
                float(different_phase[i])
            ),
            "same_phase_mean": (
                float(same_phase[i])
            ),
            "paired_contrast": float(contrast[i])
        })


# ==================================================
# 9. Publication figures
# ==================================================

plt.figure(figsize=(7, 5))

plt.bar(
    ["5 nt", "6 nt", "7 nt"],
    [
        distance_scores[5].mean(),
        distance_scores[6].mean(),
        distance_scores[7].mean()
    ]
)

plt.ylabel("Mean absolute score difference")
plt.xlabel("Window shift distance")
plt.title("Distance-matched clinical score changes")

save_figure("distance_matched_score_changes")


plt.figure(figsize=(7, 5))

plt.hist(
    contrast,
    bins=45
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1,
    label="No phase-associated difference"
)

plt.axvline(
    mean_contrast,
    linewidth=2,
    label=f"Mean contrast: {mean_contrast:.3f}"
)

plt.xlabel(
    "Different-phase minus same-phase "
    "mean absolute change"
)

plt.ylabel("Number of variants")
plt.title("Per-variant distance-matched contrast")
plt.legend()

save_figure("distance_matched_paired_contrasts")


# ==================================================
# 10. Save summary and reproducibility metadata
# ==================================================

summary = {
    "experiment": "12_clinvar_distance_controls",
    "research_step": 15,
    "variants": len(rows),
    "seed": SEED,
    "window_start_offsets": list(range(-6, 12)),
    "distance_nt": [5, 6, 7],
    "pairs_per_variant": {
        "5": 13,
        "6": 12,
        "7": 11
    },
    "mean_absolute_change": {
        str(distance): float(
            distance_scores[distance].mean()
        )
        for distance in (5, 6, 7)
    },
    "different_phase_mean": mean_different,
    "same_phase_mean": mean_same,
    "mean_paired_contrast": mean_contrast,
    "positive_contrasts": positive_count,
    "contrast_ci95": ci95,
    "wilcoxon_one_sided_p": float(test.pvalue),
    "bootstrap_replicates": BOOTSTRAPS,
    "bootstrap_method": (
        "Paired, class-stratified variant-level "
        "bootstrap"
    ),
    "subgroups": subgroups,
    "input_sha256": sha256(INPUT),
    "output_sha256": sha256(OUTPUT),
    "script_sha256": sha256(Path(__file__)),
    "limitations": [
        "The 5-, 6-, and 7-nt shifts differ "
        "slightly in displacement distance",
        "Window shifting changes flanking sequence "
        "context and token positions",
        "These comparisons support phase association "
        "but do not establish an isolated causal "
        "tokenization-phase effect",
        "Overlapping window pairs are aggregated "
        "within each variant before inference",
        "Shared genes or loci can create dependence "
        "between variants",
        "The data were already used for prior "
        "stability analyses; this is a follow-up "
        "control, not independent replication"
    ]
}

SUMMARY.write_text(
    json.dumps(
        summary,
        indent=2
    ),
    encoding="utf-8"
)


# ==================================================
# 11. Final report
# ==================================================

print("\nSTEP 15 COMPLETE")

for distance in (5, 6, 7):
    print(
        f"Mean change at {distance} nt:",
        round(
            float(distance_scores[distance].mean()),
            6
        )
    )

print(
    "Mean paired contrast:",
    round(mean_contrast, 6)
)

print("95% bootstrap CI:", ci95)
print("Wilcoxon p-value:", test.pvalue)
print("Positive contrasts:", positive_count)
print("Results:", OUTPUT)
print("Summary:", SUMMARY)
print("Figures:", FIGURE_DIR)
