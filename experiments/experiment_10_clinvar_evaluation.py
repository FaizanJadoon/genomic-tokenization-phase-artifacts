
import csv
import hashlib
import json
import platform
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import scipy
import sklearn
from sklearn.metrics import (
    roc_auc_score,
    roc_curve,
    average_precision_score,
)


# ==========================================================
# STEP 13 — EXPERIMENT 10
# Clinical-label discrimination and paired uncertainty
# ==========================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results" / "experiment_09"
    / "clinvar_phase_scores.csv"
)

RESULT_DIR = ROOT / "results" / "experiment_10"
FIGURE_DIR = ROOT / "figures" / "experiment_10"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SUMMARY_FILE = RESULT_DIR / "clinical_evaluation.json"
METRICS_FILE = RESULT_DIR / "clinical_metrics.csv"
BOOTSTRAP_FILE = RESULT_DIR / "bootstrap_results.csv"

SEED = 20261025
BOOTSTRAPS = 2000
rng = np.random.default_rng(SEED)

print("STEP 13 — Clinical validation and statistical analysis")


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


def confidence_interval(values):
    return [
        float(np.percentile(values, 2.5)),
        float(np.percentile(values, 97.5))
    ]


# ==========================================================
# 1. Load clinical scores
# ==========================================================

with INPUT.open(
    "r",
    newline="",
    encoding="utf-8"
) as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
    columns = reader.fieldnames

if len(rows) != 1000:
    raise ValueError(
        f"Expected 1000 variants; found {len(rows)}"
    )

phase_columns = [
    f"phase_{i}_score"
    for i in range(6)
]

required = (
    phase_columns
    + [
        "raw_phase0_score",
        "phase_mean_score",
        "phase_range",
        "phase_sign_disagreement"
    ]
)

missing = [
    name for name in required
    if name not in columns
]

if missing:
    raise ValueError(
        f"Missing required columns: {missing}"
    )


# ==========================================================
# 2. Detect the original ClinVar label column
# ==========================================================

preferred_columns = [
    "clinical_significance",
    "clinvar_significance",
    "clinical_label",
    "classification",
    "significance",
    "label",
    "phenotype"
]

candidates = []

for column in columns:
    observed = {
        str(row[column]).strip().lower()
        for row in rows
    }

    if observed == {"benign", "pathogenic"}:
        candidates.append(column)

if not candidates:
    raise ValueError(
        "Could not identify a Benign/Pathogenic label "
        "column. Available columns: "
        + ", ".join(columns)
    )

label_column = next(
    (
        column
        for column in preferred_columns
        if column in candidates
    ),
    candidates[0]
)

labels = np.array([
    1 if row[label_column].strip().lower()
    == "pathogenic" else 0
    for row in rows
], dtype=np.int8)

benign_count = int(np.sum(labels == 0))
pathogenic_count = int(np.sum(labels == 1))

if benign_count == 0 or pathogenic_count == 0:
    raise ValueError("Both clinical classes are required")

print("Detected label column:", label_column)
print("Benign:", benign_count)
print("Pathogenic:", pathogenic_count)


# ==========================================================
# 3. Extract and validate scores
# ==========================================================

phase_scores = np.array([
    [
        float(row[column])
        for column in phase_columns
    ]
    for row in rows
], dtype=np.float64)

raw_scores = np.array([
    float(row["raw_phase0_score"])
    for row in rows
], dtype=np.float64)

corrected_scores = np.array([
    float(row["phase_mean_score"])
    for row in rows
], dtype=np.float64)

if not (
    np.isfinite(phase_scores).all()
    and np.isfinite(raw_scores).all()
    and np.isfinite(corrected_scores).all()
):
    raise ValueError("Nonfinite model scores detected")

if not np.allclose(
    raw_scores,
    phase_scores[:, 0],
    atol=1e-6
):
    raise ValueError("Raw scores do not match phase 0")

if not np.allclose(
    corrected_scores,
    phase_scores.mean(axis=1),
    atol=1e-6
):
    raise ValueError("Phase averages do not match")

# Score = log P(ALT) - log P(REF).
# Lower values mean the ALT is less likely than REF.
# Prespecify -score as the pathogenicity ranking.
# No label-based orientation optimization is performed.

predictors = {
    "Raw phase 0": -raw_scores,
    "Six-phase average": -corrected_scores
}

for phase in range(6):
    predictors[f"Phase {phase}"] = (
        -phase_scores[:, phase]
    )


# ==========================================================
# 4. Full-dataset metrics
# ==========================================================

metrics = []

for method, prediction in predictors.items():
    auc = float(
        roc_auc_score(labels, prediction)
    )

    ap = float(
        average_precision_score(
            labels,
            prediction
        )
    )

    metrics.append({
        "method": method,
        "auroc": auc,
        "average_precision": ap
    })

metrics_by_name = {
    item["method"]: item
    for item in metrics
}

raw_auc = metrics_by_name["Raw phase 0"]["auroc"]
corrected_auc = (
    metrics_by_name["Six-phase average"]["auroc"]
)

raw_ap = (
    metrics_by_name["Raw phase 0"][
        "average_precision"
    ]
)

corrected_ap = (
    metrics_by_name["Six-phase average"][
        "average_precision"
    ]
)

auc_difference = corrected_auc - raw_auc
ap_difference = corrected_ap - raw_ap

print("\nObserved clinical-label discrimination")
print("Raw AUROC:", round(raw_auc, 6))
print("Averaged AUROC:", round(corrected_auc, 6))
print("AUROC difference:", round(auc_difference, 6))
print("Raw AP:", round(raw_ap, 6))
print("Averaged AP:", round(corrected_ap, 6))
print("AP difference:", round(ap_difference, 6))


# ==========================================================
# 5. Paired, class-stratified bootstrap
# ==========================================================

benign_indices = np.flatnonzero(labels == 0)
pathogenic_indices = np.flatnonzero(labels == 1)

bootstrap_rows = []

print("\nRunning", BOOTSTRAPS, "paired bootstraps...")

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

    sample_labels = labels[indices]

    raw_prediction = -raw_scores[indices]
    corrected_prediction = -corrected_scores[indices]

    sample_raw_auc = roc_auc_score(
        sample_labels,
        raw_prediction
    )

    sample_corrected_auc = roc_auc_score(
        sample_labels,
        corrected_prediction
    )

    sample_raw_ap = average_precision_score(
        sample_labels,
        raw_prediction
    )

    sample_corrected_ap = average_precision_score(
        sample_labels,
        corrected_prediction
    )

    bootstrap_rows.append({
        "iteration": iteration + 1,
        "raw_auroc": sample_raw_auc,
        "corrected_auroc": sample_corrected_auc,
        "delta_auroc": (
            sample_corrected_auc - sample_raw_auc
        ),
        "raw_ap": sample_raw_ap,
        "corrected_ap": sample_corrected_ap,
        "delta_ap": (
            sample_corrected_ap - sample_raw_ap
        )
    })

    if (iteration + 1) % 500 == 0:
        print(
            f"Completed {iteration + 1}/{BOOTSTRAPS}"
        )


def bootstrap_column(name):
    return np.array([
        row[name]
        for row in bootstrap_rows
    ], dtype=np.float64)


raw_auc_ci = confidence_interval(
    bootstrap_column("raw_auroc")
)

corrected_auc_ci = confidence_interval(
    bootstrap_column("corrected_auroc")
)

delta_auc_ci = confidence_interval(
    bootstrap_column("delta_auroc")
)

raw_ap_ci = confidence_interval(
    bootstrap_column("raw_ap")
)

corrected_ap_ci = confidence_interval(
    bootstrap_column("corrected_ap")
)

delta_ap_ci = confidence_interval(
    bootstrap_column("delta_ap")
)


# ==========================================================
# 6. Phase variability diagnostics
# ==========================================================

sign_disagreement = np.array([
    int(row["phase_sign_disagreement"])
    for row in rows
])

phase_range = np.array([
    float(row["phase_range"])
    for row in rows
])

if not np.allclose(
    phase_range,
    np.ptp(phase_scores, axis=1),
    atol=1e-6
):
    raise ValueError("Phase ranges do not match")

actual_disagreement = (
    (phase_scores.min(axis=1) < 0)
    & (phase_scores.max(axis=1) > 0)
).astype(int)

if not np.array_equal(
    sign_disagreement,
    actual_disagreement
):
    raise ValueError(
        "Phase sign-disagreement values do not match"
    )

disagreement_count = int(
    sign_disagreement.sum()
)

mean_phase_range = float(
    phase_range.mean()
)


# ==========================================================
# 7. Figure: ROC curves
# ==========================================================

plt.figure(figsize=(7, 6))

for method in [
    "Raw phase 0",
    "Six-phase average"
]:
    fpr, tpr, _ = roc_curve(
        labels,
        predictors[method]
    )

    auc = metrics_by_name[method]["auroc"]

    plt.plot(
        fpr,
        tpr,
        linewidth=2,
        label=f"{method} (AUC={auc:.3f})"
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    linewidth=1,
    label="Chance"
)

plt.xlabel("False-positive rate")
plt.ylabel("True-positive rate")
plt.title("ClinVar label discrimination")
plt.legend(loc="lower right")
save_figure("clinvar_roc_comparison")


# ==========================================================
# 8. Figure: Individual phase AUROCs
# ==========================================================

phase_auc = [
    metrics_by_name[f"Phase {phase}"]["auroc"]
    for phase in range(6)
]

plt.figure(figsize=(8, 5))

plt.bar(
    [f"Phase {i}" for i in range(6)],
    phase_auc
)

plt.axhline(
    corrected_auc,
    linestyle="--",
    linewidth=2,
    label="Six-phase average"
)

plt.axhline(
    0.5,
    linestyle=":",
    linewidth=1,
    label="Chance"
)

plt.ylim(0, 1)
plt.ylabel("AUROC")
plt.title("Individual phases versus phase averaging")
plt.legend()
save_figure("phase_auroc_comparison")


# ==========================================================
# 9. Figure: Paired bootstrap AUROC differences
# ==========================================================

plt.figure(figsize=(8, 5))

plt.hist(
    bootstrap_column("delta_auroc"),
    bins=40
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1,
    label="No difference"
)

plt.axvline(
    auc_difference,
    linewidth=2,
    label=f"Observed difference: {auc_difference:.4f}"
)

plt.xlabel("AUROC: six-phase average minus phase 0")
plt.ylabel("Bootstrap replicates")
plt.title("Paired bootstrap uncertainty")
plt.legend()
save_figure("paired_bootstrap_auc_difference")


# ==========================================================
# 10. Figure: Score distributions by clinical label
# ==========================================================

plt.figure(figsize=(9, 5))

plt.boxplot(
    [
        -raw_scores[labels == 0],
        -raw_scores[labels == 1],
        -corrected_scores[labels == 0],
        -corrected_scores[labels == 1]
    ],
    tick_labels=[
        "Raw\nBenign",
        "Raw\nPathogenic",
        "Average\nBenign",
        "Average\nPathogenic"
    ],
    showfliers=False
)

plt.ylabel(
    "Pathogenicity-oriented model score (-log ratio)"
)

plt.title(
    "Clinical score distributions "
    "(outliers hidden for display)"
)

save_figure("clinical_score_distributions")


# ==========================================================
# 11. Save all metrics and bootstrap samples
# ==========================================================

with METRICS_FILE.open(
    "w",
    newline="",
    encoding="utf-8"
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "method",
            "auroc",
            "average_precision"
        ]
    )
    writer.writeheader()
    writer.writerows(metrics)

with BOOTSTRAP_FILE.open(
    "w",
    newline="",
    encoding="utf-8"
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=list(bootstrap_rows[0].keys())
    )
    writer.writeheader()
    writer.writerows(bootstrap_rows)


# ==========================================================
# 12. Save analysis summary and provenance
# ==========================================================

summary = {
    "experiment": "10_clinvar_evaluation",
    "research_step": 13,
    "seed": SEED,
    "bootstrap_replicates": BOOTSTRAPS,
    "bootstrap_method": (
        "Paired variant-level stratified bootstrap "
        "with replacement within each clinical class"
    ),
    "label_column": label_column,
    "positive_class": "Pathogenic",
    "negative_class": "Benign",
    "sample_size": len(rows),
    "benign_count": benign_count,
    "pathogenic_count": pathogenic_count,
    "score_orientation": (
        "Negative of log P(ALT) minus log P(REF); "
        "fixed before evaluation"
    ),
    "raw": {
        "auroc": raw_auc,
        "auroc_ci95": raw_auc_ci,
        "average_precision": raw_ap,
        "average_precision_ci95": raw_ap_ci
    },
    "phase_average": {
        "auroc": corrected_auc,
        "auroc_ci95": corrected_auc_ci,
        "average_precision": corrected_ap,
        "average_precision_ci95": corrected_ap_ci
    },
    "paired_difference_average_minus_raw": {
        "auroc": auc_difference,
        "auroc_ci95": delta_auc_ci,
        "average_precision": ap_difference,
        "average_precision_ci95": delta_ap_ci
    },
    "individual_phase_aurocs": {
        f"phase_{i}": phase_auc[i]
        for i in range(6)
    },
    "phase_sign_disagreements": (
        disagreement_count
    ),
    "phase_sign_disagreement_fraction": (
        disagreement_count / len(rows)
    ),
    "mean_phase_range": mean_phase_range,
    "input_sha256": sha256(INPUT),
    "script_sha256": sha256(Path(__file__)),
    "python_version": platform.python_version(),
    "numpy_version": np.__version__,
    "scipy_version": scipy.__version__,
    "sklearn_version": sklearn.__version__,
    "limitations": [
        "ClinVar pathogenicity labels are not direct "
        "molecular-effect measurements",
        "The benchmark is artificially balanced "
        "and is not representative of clinical prevalence",
        "No model training or label-based score "
        "orientation selection was performed",
        "Bootstrap assumes variant-level independence; "
        "shared genes or loci may create dependence",
        "AUROC and AP do not establish clinical utility",
        "The pretrained model is not a validated "
        "clinical pathogenicity classifier",
        "Six-phase averaging also changes sequence "
        "window context and token positions"
    ]
}

SUMMARY_FILE.write_text(
    json.dumps(
        summary,
        indent=2
    ),
    encoding="utf-8"
)


# ==========================================================
# 13. Final terminal report
# ==========================================================

print("\nSTEP 13 COMPLETE")
print("Clinical variants:", len(rows))
print("Sign disagreements:", disagreement_count)
print("Mean phase range:", round(mean_phase_range, 6))

print("\nRaw phase-0 AUROC:", round(raw_auc, 6))
print("Raw AUROC 95% CI:", raw_auc_ci)

print(
    "Six-phase average AUROC:",
    round(corrected_auc, 6)
)
print("Average AUROC 95% CI:", corrected_auc_ci)

print(
    "Paired AUROC difference:",
    round(auc_difference, 6)
)
print("Difference 95% CI:", delta_auc_ci)

print("\nRaw average precision:", round(raw_ap, 6))
print(
    "Six-phase average precision:",
    round(corrected_ap, 6)
)
print("AP difference 95% CI:", delta_ap_ci)

print("\nSummary:", SUMMARY_FILE)
print("Metrics:", METRICS_FILE)
print("Bootstrap results:", BOOTSTRAP_FILE)
print("Figures:", FIGURE_DIR)
