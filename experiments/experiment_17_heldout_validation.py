
import csv
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from scipy.stats import wilcoxon
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForMaskedLM, EsmTokenizer

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =====================================================
# STEP 18 — MAIN EXPERIMENT B3
# Held-out validation of tokenization-phase stability
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed"
    / "clinvar_heldout_grch38_sequences.csv"
)

ORIGINAL = (
    ROOT / "data" / "processed"
    / "clinvar_high_confidence_snvs.csv"
)

MODEL_DIR = ROOT / "models" / "nt-v2-50m"

RESULT_DIR = ROOT / "results" / "main_experiment_B"
FIGURE_DIR = ROOT / "figures" / "main_experiment_B"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = RESULT_DIR / "heldout_anchor_scores.csv"
SUMMARY = RESULT_DIR / "heldout_validation.json"

SEED = 20261026
BATCH_SIZE = 2
BOOTSTRAPS = 2000

VARIANT_INDEX = 120
WINDOW_LENGTH = 144
BASE_START = 48

PHASES = 6
ANCHORS = {
    "center": 0,
    "minus6": -6,
    "plus6": 6,
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("STEP 18 — Main Experiment B3")
print("Held-out genomic variant validation")


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def variant_key(row):
    return (
        str(row["chromosome"]),
        int(row["position_grch38_1based"]),
        row["ref"],
        row["alt"],
    )


def variant_locus(row):
    return (
        str(row["chromosome"]),
        int(row["position_grch38_1based"]),
    )


# =====================================================
# 1. Verify frozen held-out cohort
# =====================================================

with INPUT.open(
    "r", newline="", encoding="utf-8-sig"
) as handle:
    reader = csv.DictReader(handle)
    input_fields = list(reader.fieldnames)
    variants = list(reader)

with ORIGINAL.open(
    "r", newline="", encoding="utf-8-sig"
) as handle:
    original = list(csv.DictReader(handle))

if len(variants) != 1000:
    raise RuntimeError("Expected 1000 held-out variants")

original_loci = {
    variant_locus(row)
    for row in original
}

heldout_loci = {
    variant_locus(row)
    for row in variants
}

if len(heldout_loci) != 1000:
    raise RuntimeError("Duplicate held-out loci")

if original_loci & heldout_loci:
    raise RuntimeError("Original and held-out loci overlap")

expected_keys = {
    variant_key(row)
    for row in variants
}

labels = Counter(
    row["clinical_label"]
    for row in variants
)

if labels != Counter({
    "Benign": 500,
    "Pathogenic": 500,
}):
    raise RuntimeError("Unexpected clinical-label balance")

for row in variants:
    sequence = row["reference_sequence_240nt"].upper()

    if (
        len(sequence) != 240
        or set(sequence) - set("ACGT")
        or sequence[VARIANT_INDEX] != row["ref"]
        or row["reference_match"] != "PASS"
    ):
        raise RuntimeError(
            f"Reference validation failed: {variant_key(row)}"
        )

print("Held-out variants:", len(variants))
print("Overlap with original loci:", 0)
print("Reference validation: PASS")


# =====================================================
# 2. Define score columns and resumable output
# =====================================================

score_fields = [
    f"{anchor}_phase_{phase}"
    for anchor in ANCHORS
    for phase in range(PHASES)
]

extra_fields = score_fields + [
    "center_phase_mean",
    "minus6_phase_mean",
    "plus6_phase_mean",
    "center_phase_range",
    "center_sign_disagreement",
    "raw_mean_absolute_anchor_change",
    "average_mean_absolute_anchor_change",
    "stability_improvement",
]

extra_fields += [
    f"phase_{phase}_anchor_change"
    for phase in range(PHASES)
]

fields = input_fields + extra_fields

completed = {}

if OUTPUT.exists():
    with OUTPUT.open(
        "r", newline="", encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != fields:
            raise RuntimeError(
                "Existing output columns do not match"
            )

        for row in reader:
            key = variant_key(row)

            if key not in expected_keys:
                raise RuntimeError(
                    f"Unexpected saved variant: {key}"
                )

            if key in completed:
                raise RuntimeError(
                    f"Duplicate saved variant: {key}"
                )

            for field in extra_fields:
                value = float(row[field])

                if not np.isfinite(value):
                    raise RuntimeError(
                        f"Nonfinite saved value: {field}"
                    )

            completed[key] = row

pending = [
    row
    for row in variants
    if variant_key(row) not in completed
]

print("Previously scored:", len(completed))
print("Remaining:", len(pending))


# =====================================================
# 3. Load identical pretrained model and tokenizer
# =====================================================

if pending:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU unavailable")

    device = "cuda"

    print("GPU:", torch.cuda.get_device_name(0))
    print("Loading nucleotide transformer...")

    tokenizer = EsmTokenizer(
        vocab_file=str(MODEL_DIR / "vocab.txt")
    )

    model = AutoModelForMaskedLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32,
    )

    model = model.to(device)
    model.eval()

    if tokenizer.mask_token_id is None:
        raise RuntimeError("Mask token unavailable")

    print("Model ready")


# =====================================================
# 4. Prepare identical 144nt phase/anchor windows
# =====================================================

def prepare_variant(variant):
    reference = (
        variant["reference_sequence_240nt"].upper()
    )

    alternate = (
        reference[:VARIANT_INDEX]
        + variant["alt"]
        + reference[VARIANT_INDEX + 1:]
    )

    prepared = []

    for anchor_name, shift in ANCHORS.items():
        for phase in range(PHASES):
            start = BASE_START + shift + phase
            end = start + WINDOW_LENGTH

            ref_window = reference[start:end]
            alt_window = alternate[start:end]

            if (
                len(ref_window) != WINDOW_LENGTH
                or len(alt_window) != WINDOW_LENGTH
            ):
                raise RuntimeError("Invalid sequence window")

            ref_ids = tokenizer(
                ref_window,
                return_tensors="pt",
            )["input_ids"][0]

            alt_ids = tokenizer(
                alt_window,
                return_tensors="pt",
            )["input_ids"][0]

            if ref_ids.shape != alt_ids.shape:
                raise RuntimeError(
                    "Reference/alternate token lengths differ"
                )

            differences = (
                ref_ids != alt_ids
            ).nonzero(as_tuple=True)[0]

            if len(differences) != 1:
                raise RuntimeError(
                    "Expected exactly one changed token: "
                    f"{variant_key(variant)}, "
                    f"{anchor_name}, phase {phase}"
                )

            position = int(differences[0])

            masked = ref_ids.clone()
            masked[position] = tokenizer.mask_token_id

            prepared.append({
                "anchor": anchor_name,
                "phase": phase,
                "masked": masked,
                "position": position,
                "ref_token": int(ref_ids[position]),
                "alt_token": int(alt_ids[position]),
            })

    return prepared


# =====================================================
# 5. Score all three anchors and all six phases
# =====================================================

if not OUTPUT.exists():
    with OUTPUT.open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writeheader()

inference_start = time.perf_counter()

for batch_start in range(
    0, len(pending), BATCH_SIZE
):
    batch = pending[
        batch_start:batch_start + BATCH_SIZE
    ]

    prepared = []

    for variant in batch:
        prepared.extend(prepare_variant(variant))

    masked_ids = torch.stack([
        item["masked"]
        for item in prepared
    ]).to(device)

    positions = torch.tensor(
        [item["position"] for item in prepared],
        dtype=torch.long,
        device=device,
    )

    ref_tokens = torch.tensor(
        [item["ref_token"] for item in prepared],
        dtype=torch.long,
        device=device,
    )

    alt_tokens = torch.tensor(
        [item["alt_token"] for item in prepared],
        dtype=torch.long,
        device=device,
    )

    with torch.inference_mode():
        logits = model(
            input_ids=masked_ids
        ).logits

        indices = torch.arange(
            len(prepared),
            device=device,
        )

        selected = logits[
            indices, positions
        ].float()

        log_probs = torch.log_softmax(
            selected, dim=-1
        )

        scores = (
            log_probs[indices, alt_tokens]
            - log_probs[indices, ref_tokens]
        ).cpu().numpy()

    verified_rows = []

    for local_index, variant in enumerate(batch):
        first = local_index * len(ANCHORS) * PHASES
        last = first + len(ANCHORS) * PHASES

        variant_scores = scores[first:last]

        if len(variant_scores) != 18:
            raise RuntimeError("Expected 18 scores per variant")

        matrices = {}

        for anchor_index, anchor in enumerate(ANCHORS):
            start = anchor_index * PHASES

            matrices[anchor] = np.asarray(
                variant_scores[start:start + PHASES],
                dtype=np.float64,
            )

        center = matrices["center"]
        minus6 = matrices["minus6"]
        plus6 = matrices["plus6"]

        per_phase_change = (
            np.abs(minus6 - center)
            + np.abs(plus6 - center)
        ) / 2.0

        raw_change = float(per_phase_change[0])

        average_change = float(
            (
                abs(minus6.mean() - center.mean())
                + abs(plus6.mean() - center.mean())
            ) / 2.0
        )

        output_row = dict(variant)

        for anchor in ANCHORS:
            for phase in range(PHASES):
                output_row[
                    f"{anchor}_phase_{phase}"
                ] = float(matrices[anchor][phase])

            output_row[
                f"{anchor}_phase_mean"
            ] = float(matrices[anchor].mean())

        output_row["center_phase_range"] = float(
            center.max() - center.min()
        )

        output_row["center_sign_disagreement"] = int(
            center.min() < 0 < center.max()
        )

        output_row[
            "raw_mean_absolute_anchor_change"
        ] = raw_change

        output_row[
            "average_mean_absolute_anchor_change"
        ] = average_change

        output_row["stability_improvement"] = (
            raw_change - average_change
        )

        for phase in range(PHASES):
            output_row[
                f"phase_{phase}_anchor_change"
            ] = float(per_phase_change[phase])

        verified_rows.append(output_row)

    with OUTPUT.open(
        "a", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=fields
        )
        writer.writerows(verified_rows)

    count = (
        len(completed)
        + batch_start
        + len(batch)
    )

    if count % 40 == 0 or count == 1000:
        print(f"Scored {count}/1000 variants")

if pending:
    torch.cuda.synchronize()

scoring_seconds = time.perf_counter() - inference_start


# =====================================================
# 6. Verify completed held-out scores
# =====================================================

with OUTPUT.open(
    "r", newline="", encoding="utf-8"
) as handle:
    reader = csv.DictReader(handle)

    if reader.fieldnames != fields:
        raise RuntimeError("Final output header mismatch")

    results = list(reader)

if len(results) != 1000:
    raise RuntimeError("Incomplete validation results")

if {
    variant_key(row)
    for row in results
} != expected_keys:
    raise RuntimeError("Held-out variant set mismatch")

raw = np.array([
    float(row["raw_mean_absolute_anchor_change"])
    for row in results
])

averaged = np.array([
    float(row["average_mean_absolute_anchor_change"])
    for row in results
])

improvement = raw - averaged

if not (
    np.isfinite(raw).all()
    and np.isfinite(averaged).all()
    and np.isfinite(improvement).all()
):
    raise RuntimeError("Nonfinite stability values")

clinical_labels = np.array([
    int(row["clinical_label"] == "Pathogenic")
    for row in results
])

benign_indices = np.where(
    clinical_labels == 0
)[0]

pathogenic_indices = np.where(
    clinical_labels == 1
)[0]

if (
    len(benign_indices) != 500
    or len(pathogenic_indices) != 500
):
    raise RuntimeError("Class balance changed")


# =====================================================
# 7. Prespecified class-stratified bootstrap
# =====================================================

rng = np.random.default_rng(SEED)

bootstrap_indices = []

for _ in range(BOOTSTRAPS):
    indices = np.concatenate([
        rng.choice(
            benign_indices,
            size=len(benign_indices),
            replace=True,
        ),
        rng.choice(
            pathogenic_indices,
            size=len(pathogenic_indices),
            replace=True,
        ),
    ])

    bootstrap_indices.append(indices)


def paired_mean_ci(values):
    means = np.array([
        values[indices].mean()
        for indices in bootstrap_indices
    ])

    return [
        float(np.percentile(means, 2.5)),
        float(np.percentile(means, 97.5)),
    ]


raw_mean = float(raw.mean())
averaged_mean = float(averaged.mean())

reduction_percent = (
    100.0 * (raw_mean - averaged_mean) / raw_mean
    if raw_mean > 0 else None
)

primary_p = float(
    wilcoxon(
        raw,
        averaged,
        alternative="greater",
        zero_method="wilcox",
    ).pvalue
)


# =====================================================
# 8. All six individual-phase baselines
# =====================================================

baseline_comparisons = {}
raw_pvalues = []

for phase in range(PHASES):
    baseline = np.array([
        float(row[f"phase_{phase}_anchor_change"])
        for row in results
    ])

    paired_difference = baseline - averaged

    pvalue = float(
        wilcoxon(
            baseline,
            averaged,
            alternative="greater",
            zero_method="wilcox",
        ).pvalue
    )

    raw_pvalues.append(pvalue)

    baseline_mean = float(baseline.mean())

    baseline_comparisons[str(phase)] = {
        "baseline_mean_instability": baseline_mean,
        "averaged_mean_instability": averaged_mean,
        "mean_paired_improvement": float(
            paired_difference.mean()
        ),
        "relative_reduction_percent": (
            100.0 * (
                baseline_mean - averaged_mean
            ) / baseline_mean
            if baseline_mean > 0 else None
        ),
        "paired_improvement_ci95": paired_mean_ci(
            paired_difference
        ),
        "wilcoxon_one_sided_p": pvalue,
        "variants_improved": int(
            np.sum(paired_difference > 0)
        ),
    }

# Holm familywise adjustment across six comparisons.
order = np.argsort(raw_pvalues)
adjusted = np.empty(PHASES, dtype=float)
running_max = 0.0

for rank, phase in enumerate(order):
    candidate = min(
        1.0,
        (PHASES - rank) * raw_pvalues[phase],
    )

    running_max = max(running_max, candidate)
    adjusted[phase] = running_max

for phase in range(PHASES):
    baseline_comparisons[str(phase)][
        "holm_adjusted_p"
    ] = float(adjusted[phase])


# =====================================================
# 9. Phase disagreement and subgroup analysis
# =====================================================

phase_ranges = np.array([
    float(row["center_phase_range"])
    for row in results
])

sign_disagreements = np.array([
    int(row["center_sign_disagreement"])
    for row in results
])

subgroups = {}

for label in ("Benign", "Pathogenic"):
    mask = np.array([
        row["clinical_label"] == label
        for row in results
    ])

    subgroup_raw = float(raw[mask].mean())
    subgroup_avg = float(averaged[mask].mean())

    subgroups[label] = {
        "count": int(mask.sum()),
        "raw_mean_instability": subgroup_raw,
        "averaged_mean_instability": subgroup_avg,
        "mean_paired_improvement": (
            subgroup_raw - subgroup_avg
        ),
        "relative_reduction_percent": (
            100.0 * (
                subgroup_raw - subgroup_avg
            ) / subgroup_raw
            if subgroup_raw > 0 else None
        ),
        "sign_disagreements": int(
            sign_disagreements[mask].sum()
        ),
    }


# =====================================================
# 10. Exploratory clinical AUROC
# =====================================================

# Direction fixed in advance:
# more negative ALT-vs-REF likelihood = higher
# putative deleteriousness.

phase0_scores = -np.array([
    float(row["center_phase_0"])
    for row in results
])

mean_scores = -np.array([
    float(row["center_phase_mean"])
    for row in results
])

auc_phase0 = float(
    roc_auc_score(
        clinical_labels,
        phase0_scores,
    )
)

auc_averaged = float(
    roc_auc_score(
        clinical_labels,
        mean_scores,
    )
)

auc_deltas = np.array([
    roc_auc_score(
        clinical_labels[indices],
        mean_scores[indices],
    )
    - roc_auc_score(
        clinical_labels[indices],
        phase0_scores[indices],
    )
    for indices in bootstrap_indices
])

auc_delta_ci = [
    float(np.percentile(auc_deltas, 2.5)),
    float(np.percentile(auc_deltas, 97.5)),
]


# =====================================================
# 11. Publication figures
# =====================================================

plt.figure(figsize=(7, 5))

plt.boxplot(
    [raw, averaged],
    tick_labels=[
        "Phase 0",
        "Six-phase mean",
    ],
    showfliers=False,
)

plt.ylabel("Mean absolute anchor-shift score change")
plt.title("Held-out ClinVar score stability")
plt.tight_layout()

for extension in ("png", "pdf"):
    plt.savefig(
        FIGURE_DIR / f"heldout_stability.{extension}",
        dpi=300,
        bbox_inches="tight",
    )

plt.close()


# =====================================================
# 12. Reproducibility metadata
# =====================================================

summary = {
    "experiment": "17_heldout_validation",
    "research_step": 18,
    "validation_type": (
        "Held-out position-disjoint variants "
        "from the same ClinVar release"
    ),
    "seed": SEED,
    "variants": len(results),
    "benign": 500,
    "pathogenic": 500,
    "overlap_with_original_loci": 0,
    "model": "Nucleotide Transformer v2 50M",
    "window_length_nt": WINDOW_LENGTH,
    "variant_index_zero_based": VARIANT_INDEX,
    "anchor_shifts_nt": [-6, 0, 6],
    "phases_per_anchor": PHASES,
    "score": (
        "log P(ALT token) - log P(REF token)"
    ),
    "primary_endpoint": (
        "Mean absolute anchor-shift score change"
    ),
    "primary_comparison": (
        "Phase 0 versus six-phase mean"
    ),
    "raw_mean_absolute_anchor_change": raw_mean,
    "average_mean_absolute_anchor_change": averaged_mean,
    "mean_paired_improvement": float(
        improvement.mean()
    ),
    "relative_reduction_percent": reduction_percent,
    "paired_improvement_ci95": paired_mean_ci(
        improvement
    ),
    "wilcoxon_one_sided_p": primary_p,
    "variants_with_positive_improvement": int(
        np.sum(improvement > 0)
    ),
    "mean_center_phase_range": float(
        phase_ranges.mean()
    ),
    "center_sign_disagreements": int(
        sign_disagreements.sum()
    ),
    "all_six_phase_baselines": baseline_comparisons,
    "clinical_subgroups": subgroups,
    "exploratory_clinical_auroc": {
        "score_direction": (
            "Negative ALT-vs-REF log likelihood"
        ),
        "phase0": auc_phase0,
        "six_phase_mean": auc_averaged,
        "delta": auc_averaged - auc_phase0,
        "delta_ci95": auc_delta_ci,
    },
    "new_scoring_wall_seconds": scoring_seconds,
    "batch_size_variants": BATCH_SIZE,
    "dtype": "float32",
    "input_sha256": sha256(INPUT),
    "original_cohort_sha256": sha256(ORIGINAL),
    "output_sha256": sha256(OUTPUT),
    "script_sha256": sha256(Path(__file__)),
    "model_config_sha256": sha256(
        MODEL_DIR / "config.json"
    ),
    "tokenizer_vocab_sha256": sha256(
        MODEL_DIR / "vocab.txt"
    ),
    "bootstrap_method": (
        "2000 class-stratified paired "
        "variant-level resamples"
    ),
    "multiple_testing": (
        "Holm adjustment for six "
        "phase-baseline comparisons"
    ),
    "limitations": [
        "The held-out cohort comes from "
        "the same ClinVar release.",
        "Position-disjoint variants can "
        "share genes and genomic regions.",
        "Anchor shifts also change flanking "
        "sequence context and token position.",
        "Variant-level bootstrap intervals "
        "do not account for gene clustering.",
        "Six-phase averaging requires six "
        "model evaluations per anchor.",
        "Clinical labels are not direct "
        "molecular effect measurements.",
        "Clinical AUROC is exploratory and "
        "does not establish clinical utility.",
    ],
}

SUMMARY.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)


# =====================================================
# 13. Final report
# =====================================================

print("\nSTEP 18 — MAIN EXPERIMENT B COMPLETE")
print("Held-out variants:", len(results))
print("Total scores:", len(results) * 18)
print("Original-locus overlap:", 0)

print(
    "Raw instability:",
    round(raw_mean, 6),
)

print(
    "Averaged instability:",
    round(averaged_mean, 6),
)

print(
    "Stability reduction (%):",
    round(reduction_percent, 3),
)

print(
    "Mean paired improvement:",
    round(float(improvement.mean()), 6),
)

print(
    "95% bootstrap CI:",
    paired_mean_ci(improvement),
)

print(
    "Wilcoxon p-value:",
    primary_p,
)

print(
    "Variants improved:",
    int(np.sum(improvement > 0)),
)

print(
    "Mean phase range:",
    round(float(phase_ranges.mean()), 6),
)

print(
    "Sign disagreements:",
    int(sign_disagreements.sum()),
)

print("\nAll six phase baselines:")

for phase, result in baseline_comparisons.items():
    print(
        f"Phase {phase}: "
        f"reduction="
        f"{result['relative_reduction_percent']:.2f}%, "
        f"adjusted_p="
        f"{result['holm_adjusted_p']:.3e}"
    )

print("\nExploratory clinical AUROC:")
print("Phase 0:", round(auc_phase0, 6))
print("Six-phase:", round(auc_averaged, 6))
print("AUROC delta 95% CI:", auc_delta_ci)

print("\nResults:", OUTPUT)
print("Summary:", SUMMARY)
print("Figures:", FIGURE_DIR)
