
import csv
import hashlib
import importlib.metadata
import json
import platform
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


# =====================================================
# STEP 20 — REPRODUCIBILITY AND PUBLICATION AUDIT
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

PROCESSED = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
MODEL = ROOT / "models" / "nt-v2-50m"

AUDIT_DIR = RESULTS / "publication_audit"
AUDIT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = AUDIT_DIR / "reproducibility_audit.json"

ORIGINAL = PROCESSED / "clinvar_high_confidence_snvs.csv"
HELDOUT = PROCESSED / "clinvar_heldout_grch38_sequences.csv"

ORIGINAL_SUMMARY = (
    RESULTS / "experiment_11" / "stability_analysis.json"
)

EXPERIMENT_A = (
    RESULTS / "main_experiment_A" / "fixed_core_analysis.json"
)

EXPERIMENT_B = (
    RESULTS / "main_experiment_B" / "heldout_validation.json"
)

EXPERIMENT_C = (
    RESULTS / "main_experiment_C" / "compute_benchmark.json"
)

HELDOUT_SCORES = (
    RESULTS / "main_experiment_B" / "heldout_anchor_scores.csv"
)

SEQUENCE_METADATA = (
    PROCESSED / "clinvar_heldout_sequences_metadata.json"
)

checks = {}
warnings = []
manifest = {}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path):
    with path.open(
        "r", newline="", encoding="utf-8-sig"
    ) as handle:
        return list(csv.DictReader(handle))


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def check(name, condition):
    checks[name] = bool(condition)
    print(
        ("PASS" if condition else "FAIL"),
        name,
    )


def locus(row):
    return (
        str(row["chromosome"]),
        int(row["position_grch38_1based"]),
    )


def version(package):
    try:
        return importlib.metadata.version(package)
    except importlib.metadata.PackageNotFoundError:
        return None


print("STEP 20 — Reproducibility audit")


# =====================================================
# 1. Required research artifacts
# =====================================================

required = {
    "original_cohort": ORIGINAL,
    "heldout_cohort": HELDOUT,
    "original_stability_summary": ORIGINAL_SUMMARY,
    "main_experiment_A": EXPERIMENT_A,
    "main_experiment_B": EXPERIMENT_B,
    "main_experiment_C": EXPERIMENT_C,
    "heldout_scores": HELDOUT_SCORES,
    "sequence_metadata": SEQUENCE_METADATA,
    "model_config": MODEL / "config.json",
    "tokenizer_vocabulary": MODEL / "vocab.txt",
}

for name, path in required.items():
    check(f"artifact_{name}", path.is_file())

    if path.is_file():
        manifest[name] = {
            "path": str(path),
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
        }

if not all(path.is_file() for path in required.values()):
    raise RuntimeError(
        "Required artifacts missing; audit cannot continue"
    )


# =====================================================
# 2. Cohort separation and sequence integrity
# =====================================================

original = read_csv(ORIGINAL)
heldout = read_csv(HELDOUT)
scored = read_csv(HELDOUT_SCORES)

original_loci = {locus(row) for row in original}
heldout_loci = {locus(row) for row in heldout}
scored_loci = {locus(row) for row in scored}

check("original_cohort_size_1000", len(original) == 1000)
check("heldout_cohort_size_1000", len(heldout) == 1000)
check("heldout_unique_loci", len(heldout_loci) == 1000)
check(
    "original_heldout_locus_disjoint",
    not (original_loci & heldout_loci),
)
check("scored_variants_1000", len(scored) == 1000)
check(
    "scored_loci_match_heldout",
    scored_loci == heldout_loci,
)

labels = Counter(
    row["clinical_label"] for row in heldout
)

check(
    "heldout_class_balance",
    labels == Counter({
        "Benign": 500,
        "Pathogenic": 500,
    }),
)

sequence_failures = 0

for row in heldout:
    sequence = row["reference_sequence_240nt"].upper()

    if (
        len(sequence) != 240
        or set(sequence) - set("ACGT")
        or sequence[120] != row["ref"]
        or row["reference_match"] != "PASS"
    ):
        sequence_failures += 1

check(
    "heldout_reference_integrity",
    sequence_failures == 0,
)


# =====================================================
# 3. Recompute primary held-out result
# =====================================================

raw = np.array([
    float(row["raw_mean_absolute_anchor_change"])
    for row in scored
])

averaged = np.array([
    float(row["average_mean_absolute_anchor_change"])
    for row in scored
])

improvement = raw - averaged

check(
    "all_stability_values_finite",
    np.isfinite(raw).all()
    and np.isfinite(averaged).all(),
)

heldout_summary = read_json(EXPERIMENT_B)
original_summary = read_json(ORIGINAL_SUMMARY)

recomputed_reduction = (
    100.0 * improvement.mean() / raw.mean()
)

check(
    "heldout_reduction_recomputed",
    np.isclose(
        recomputed_reduction,
        heldout_summary["relative_reduction_percent"],
        atol=1e-6,
    ),
)

check(
    "heldout_improvement_recomputed",
    np.isclose(
        improvement.mean(),
        heldout_summary["mean_paired_improvement"],
        atol=1e-6,
    ),
)

check(
    "heldout_positive_improvement_count",
    int(np.sum(improvement > 0))
    == heldout_summary["variants_with_positive_improvement"],
)

check(
    "heldout_ci_excludes_zero",
    heldout_summary["paired_improvement_ci95"][0] > 0,
)

phase_checks = []

for row in scored:
    center = np.array([
        float(row[f"center_phase_{i}"])
        for i in range(6)
    ])

    minus = np.array([
        float(row[f"minus6_phase_{i}"])
        for i in range(6)
    ])

    plus = np.array([
        float(row[f"plus6_phase_{i}"])
        for i in range(6)
    ])

    baseline = (
        abs(minus[0] - center[0])
        + abs(plus[0] - center[0])
    ) / 2.0

    corrected = (
        abs(minus.mean() - center.mean())
        + abs(plus.mean() - center.mean())
    ) / 2.0

    phase_checks.append(
        np.isclose(
            baseline,
            float(row["raw_mean_absolute_anchor_change"]),
            atol=1e-5,
        )
        and np.isclose(
            corrected,
            float(row["average_mean_absolute_anchor_change"]),
            atol=1e-5,
        )
    )

check(
    "all_heldout_scores_reproduce_stability",
    all(phase_checks),
)

baseline_results = heldout_summary["all_six_phase_baselines"]

check(
    "all_six_baselines_improve",
    all(
        baseline_results[str(i)]["mean_paired_improvement"] > 0
        for i in range(6)
    ),
)

check(
    "all_six_holm_adjusted_p_below_0_05",
    all(
        baseline_results[str(i)]["holm_adjusted_p"] < 0.05
        for i in range(6)
    ),
)


# =====================================================
# 4. Benchmark integrity and clinical claim
# =====================================================

benchmark = read_json(EXPERIMENT_C)

check(
    "benchmark_latency_ratio_consistent",
    np.isclose(
        benchmark["median_latency_ratio"],
        (
            benchmark["six_phase"][
                "median_milliseconds_per_variant"
            ]
            / benchmark["single_phase"][
                "median_milliseconds_per_variant"
            ]
        ),
        atol=1e-8,
    ),
)

check(
    "benchmark_phase0_numerical_agreement",
    benchmark["maximum_phase0_score_difference"] < 1e-4,
)

auc = heldout_summary["exploratory_clinical_auroc"]
auc_ci = auc["delta_ci95"]

check(
    "clinical_auroc_improvement_not_established",
    auc_ci[0] <= 0 <= auc_ci[1],
)


# =====================================================
# 5. Sequence-source provenance
# =====================================================

source_counts = Counter()

for row in heldout:
    identifier = row.get(
        "ensembl_sequence_id", ""
    ).strip()

    if identifier.upper().startswith("UCSC:"):
        source_counts["UCSC"] += 1
    elif identifier:
        source_counts["Other_or_Ensembl"] += 1
    else:
        source_counts["Unknown"] += 1

sequence_metadata = read_json(SEQUENCE_METADATA)

if len(source_counts) > 1:
    warnings.append(
        "Held-out reference sequences use mixed "
        "Ensembl/UCSC provenance. Manuscript and "
        "metadata must disclose both providers."
    )

if source_counts.get("Unknown", 0):
    warnings.append(
        "Some reference-sequence provider IDs are missing."
    )

check(
    "all_sequence_sources_identifiable",
    source_counts.get("Unknown", 0) == 0,
)

metadata_text = json.dumps(
    sequence_metadata
).lower()

if (
    source_counts.get("Other_or_Ensembl", 0)
    and "ucsc" in metadata_text
    and "ensembl" not in metadata_text
):
    warnings.append(
        "Sequence metadata may describe UCSC alone "
        "despite previously retrieved Ensembl records."
    )


# =====================================================
# 6. Model and software environment
# =====================================================

weight_files = sorted(
    path for path in MODEL.iterdir()
    if path.is_file()
    and path.suffix in (".bin", ".safetensors")
)

check("model_weight_files_present", bool(weight_files))

for path in weight_files:
    manifest[f"model_weight_{path.name}"] = {
        "path": str(path),
        "size_bytes": path.stat().st_size,
        "sha256": sha256(path),
    }

environment = {
    "python": platform.python_version(),
    "operating_system": platform.platform(),
    "numpy": version("numpy"),
    "scipy": version("scipy"),
    "scikit_learn": version("scikit-learn"),
    "torch": version("torch"),
    "transformers": version("transformers"),
    "matplotlib": version("matplotlib"),
    "tokenizers": version("tokenizers"),
}

# =====================================================
# 7. Audit summary
# =====================================================

original_reduction = (
    original_summary["relative_reduction_percent"]
)

heldout_reduction = (
    heldout_summary["relative_reduction_percent"]
)

report = {
    "research_step": 20,
    "audit_time_utc": datetime.now(
        timezone.utc
    ).isoformat(),
    "paper_title": (
        "Tokenization-Phase Artifacts in Genomic "
        "Foundation Models: Training-Free Correction "
        "of Single-Nucleotide Variant Effect Scores"
    ),
    "checks": checks,
    "passed_checks": sum(checks.values()),
    "failed_checks": [
        key for key, passed in checks.items()
        if not passed
    ],
    "original_stability_reduction_percent": (
        original_reduction
    ),
    "heldout_stability_reduction_percent": (
        heldout_reduction
    ),
    "heldout_mean_paired_improvement": float(
        improvement.mean()
    ),
    "heldout_improvement_ci95": (
        heldout_summary["paired_improvement_ci95"]
    ),
    "heldout_variants_improved": int(
        np.sum(improvement > 0)
    ),
    "benchmark_latency_ratio": (
        benchmark["median_latency_ratio"]
    ),
    "reference_source_counts": dict(source_counts),
    "warnings": warnings,
    "software_environment": environment,
    "artifact_manifest": manifest,
    "limitations_for_manuscript": [
        "Same ClinVar release for both cohorts",
        "Held-out positions are disjoint, "
        "but genes are not guaranteed disjoint",
        "Anchor shifts also alter flanking context",
        "No demonstrated improvement in clinical AUROC",
        "Model evaluated on one local GPU",
        "Single pretrained model architecture",
        "Clinical annotations are proxy labels "
        "rather than direct functional measurements",
    ],
}

OUTPUT.write_text(
    json.dumps(report, indent=2),
    encoding="utf-8",
)

print("\nSTEP 20 — AUDIT COMPLETE")
print("Passed checks:", sum(checks.values()))
print("Total checks:", len(checks))
print("Failed checks:", report["failed_checks"])
print("Original reduction:", round(original_reduction, 3))
print("Held-out reduction:", round(heldout_reduction, 3))
print("Latency ratio:", round(benchmark["median_latency_ratio"], 3))
print("Reference sources:", dict(source_counts))

for warning in warnings:
    print("WARNING:", warning)

print("Report:", OUTPUT)
