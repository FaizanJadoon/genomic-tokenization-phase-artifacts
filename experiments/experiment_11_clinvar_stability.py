
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import torch
from scipy.stats import wilcoxon
from transformers import AutoModelForMaskedLM, EsmTokenizer


# =====================================================
# STEP 14 — EXPERIMENT 11
# Clinical variant score stability across window anchors
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "results" / "experiment_09"
    / "clinvar_phase_scores.csv"
)

MODEL_DIR = ROOT / "models" / "nt-v2-50m"

RESULT_DIR = ROOT / "results" / "experiment_11"
FIGURE_DIR = ROOT / "figures" / "experiment_11"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = RESULT_DIR / "clinvar_anchor_scores.csv"
SUMMARY = RESULT_DIR / "stability_analysis.json"

SEED = 20261025
BATCH_SIZE = 4
BOOTSTRAPS = 2000

VARIANT_INDEX = 120
WINDOW_LENGTH = 144
BASE_START = 48

PHASES = 6
ANCHORS = {
    "minus6": -6,
    "plus6": 6
}

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

print("STEP 14 — Clinical variant score stability")


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
        row["chromosome"],
        row["position_grch38_1based"],
        row["ref"],
        row["alt"]
    )


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


# =====================================================
# 1. Load previously scored clinical variants
# =====================================================

with INPUT.open(
    "r",
    newline="",
    encoding="utf-8"
) as handle:
    reader = csv.DictReader(handle)
    variants = list(reader)
    input_fields = reader.fieldnames

assert len(variants) == 1000

expected_keys = {
    variant_key(row)
    for row in variants
}

assert len(expected_keys) == 1000

for row in variants:
    sequence = row["reference_sequence_240nt"].upper()

    assert len(sequence) == 240
    assert sequence[VARIANT_INDEX] == row["ref"]
    assert row["reference_match"] == "PASS"

    center_scores = np.array([
        float(row[f"phase_{i}_score"])
        for i in range(PHASES)
    ])

    assert np.allclose(
        center_scores.mean(),
        float(row["phase_mean_score"]),
        atol=1e-6
    )

print("Clinical variants:", len(variants))


# =====================================================
# 2. Prepare resumable output
# =====================================================

new_fields = [
    f"{anchor}_phase_{phase}"
    for anchor in ANCHORS
    for phase in range(PHASES)
]

new_fields += [
    "raw_minus6",
    "raw_center",
    "raw_plus6",
    "average_minus6",
    "average_center",
    "average_plus6",
    "raw_mean_absolute_anchor_change",
    "average_mean_absolute_anchor_change",
    "stability_improvement"
]

fields = input_fields + [
    field
    for field in new_fields
    if field not in input_fields
]

completed = {}

if OUTPUT.exists():
    with OUTPUT.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != fields:
            raise ValueError(
                "Existing output header does not match"
            )

        for row in reader:
            key = variant_key(row)

            if key in completed:
                raise ValueError(
                    f"Duplicate saved variant: {key}"
                )

            for field in new_fields:
                float(row[field])

            completed[key] = row

pending = [
    row
    for row in variants
    if variant_key(row) not in completed
]

print("Previously completed:", len(completed))
print("Remaining:", len(pending))


# =====================================================
# 3. Load model only when scoring is required
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
        torch_dtype=torch.float32
    )

    model = model.to(device)
    model.eval()

    assert tokenizer.mask_token_id is not None

    print("Model ready")


# =====================================================
# 4. Prepare alternate anchor windows
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
                raise ValueError("Invalid sequence window")

            ref_ids = tokenizer(
                ref_window,
                return_tensors="pt"
            )["input_ids"][0]

            alt_ids = tokenizer(
                alt_window,
                return_tensors="pt"
            )["input_ids"][0]

            if ref_ids.shape != alt_ids.shape:
                raise ValueError("Token lengths differ")

            differences = (
                ref_ids != alt_ids
            ).nonzero(as_tuple=True)[0]

            if len(differences) != 1:
                raise ValueError(
                    "Expected one changed token: "
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
                "alt_token": int(alt_ids[position])
            })

    return prepared


# =====================================================
# 5. Score missing variants
# =====================================================

if not OUTPUT.exists():
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


for batch_start in range(
    0,
    len(pending),
    BATCH_SIZE
):
    batch = pending[
        batch_start:batch_start + BATCH_SIZE
    ]

    prepared = []

    for variant in batch:
        prepared.extend(
            prepare_variant(variant)
        )

    masked_ids = torch.stack([
        item["masked"]
        for item in prepared
    ]).to(device)

    positions = torch.tensor(
        [item["position"] for item in prepared],
        dtype=torch.long,
        device=device
    )

    ref_tokens = torch.tensor(
        [item["ref_token"] for item in prepared],
        dtype=torch.long,
        device=device
    )

    alt_tokens = torch.tensor(
        [item["alt_token"] for item in prepared],
        dtype=torch.long,
        device=device
    )

    with torch.inference_mode():
        logits = model(
            input_ids=masked_ids
        ).logits

        indices = torch.arange(
            len(prepared),
            device=device
        )

        selected = logits[
            indices,
            positions
        ].float()

        log_probs = torch.log_softmax(
            selected,
            dim=-1
        )

        scores = (
            log_probs[indices, alt_tokens]
            - log_probs[indices, ref_tokens]
        ).cpu().tolist()

    verified_rows = []

    for local_index, variant in enumerate(batch):
        first = local_index * len(ANCHORS) * PHASES
        last = first + len(ANCHORS) * PHASES

        variant_scores = scores[first:last]

        minus6_scores = variant_scores[:PHASES]
        plus6_scores = variant_scores[PHASES:]

        center_scores = [
            float(variant[f"phase_{i}_score"])
            for i in range(PHASES)
        ]

        raw_minus6 = minus6_scores[0]
        raw_center = center_scores[0]
        raw_plus6 = plus6_scores[0]

        avg_minus6 = float(np.mean(minus6_scores))
        avg_center = float(np.mean(center_scores))
        avg_plus6 = float(np.mean(plus6_scores))

        raw_change = (
            abs(raw_minus6 - raw_center)
            + abs(raw_plus6 - raw_center)
        ) / 2

        avg_change = (
            abs(avg_minus6 - avg_center)
            + abs(avg_plus6 - avg_center)
        ) / 2

        output_row = dict(variant)

        for phase in range(PHASES):
            output_row[
                f"minus6_phase_{phase}"
            ] = minus6_scores[phase]

            output_row[
                f"plus6_phase_{phase}"
            ] = plus6_scores[phase]

        output_row.update({
            "raw_minus6": raw_minus6,
            "raw_center": raw_center,
            "raw_plus6": raw_plus6,
            "average_minus6": avg_minus6,
            "average_center": avg_center,
            "average_plus6": avg_plus6,
            "raw_mean_absolute_anchor_change": raw_change,
            "average_mean_absolute_anchor_change": avg_change,
            "stability_improvement": raw_change - avg_change
        })

        verified_rows.append(output_row)

    with OUTPUT.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=fields
        )
        writer.writerows(verified_rows)

    count = (
        len(completed)
        + batch_start
        + len(batch)
    )

    if count % 40 == 0 or count == len(variants):
        print(f"Scored {count}/1000 variants")


# =====================================================
# 6. Load and verify complete output
# =====================================================

with OUTPUT.open(
    "r",
    newline="",
    encoding="utf-8"
) as handle:
    final_rows = list(csv.DictReader(handle))

assert len(final_rows) == 1000

final_keys = [
    variant_key(row)
    for row in final_rows
]

assert len(set(final_keys)) == 1000
assert set(final_keys) == expected_keys

raw_changes = np.array([
    float(row["raw_mean_absolute_anchor_change"])
    for row in final_rows
])

avg_changes = np.array([
    float(row["average_mean_absolute_anchor_change"])
    for row in final_rows
])

improvements = raw_changes - avg_changes

assert np.isfinite(raw_changes).all()
assert np.isfinite(avg_changes).all()
assert np.isfinite(improvements).all()


# =====================================================
# 7. Paired statistical evaluation
# =====================================================

raw_mean = float(raw_changes.mean())
avg_mean = float(avg_changes.mean())
mean_improvement = float(improvements.mean())

if raw_mean > 0:
    reduction_percent = float(
        100 * (raw_mean - avg_mean) / raw_mean
    )
else:
    reduction_percent = None

test = wilcoxon(
    raw_changes,
    avg_changes,
    alternative="greater",
    zero_method="wilcox"
)

rng = np.random.default_rng(SEED)

bootstrap_means = np.empty(
    BOOTSTRAPS,
    dtype=np.float64
)

for iteration in range(BOOTSTRAPS):
    indices = rng.integers(
        0,
        len(improvements),
        size=len(improvements)
    )

    bootstrap_means[iteration] = (
        improvements[indices].mean()
    )

ci95 = [
    float(np.percentile(bootstrap_means, 2.5)),
    float(np.percentile(bootstrap_means, 97.5))
]

improved_count = int(
    np.sum(improvements > 0)
)


# =====================================================
# 8. Clinical-class subgroup analysis
# =====================================================

subgroups = {}

for label in ("Benign", "Pathogenic"):
    indices = np.array([
        row["clinical_label"].strip().lower()
        == label.lower()
        for row in final_rows
    ])

    subgroup_raw = float(
        raw_changes[indices].mean()
    )

    subgroup_avg = float(
        avg_changes[indices].mean()
    )

    subgroups[label] = {
        "count": int(indices.sum()),
        "raw_mean_absolute_change": subgroup_raw,
        "averaged_mean_absolute_change": subgroup_avg,
        "mean_improvement": subgroup_raw - subgroup_avg
    }


# =====================================================
# 9. Publication figures
# =====================================================

plt.figure(figsize=(7, 5))

plt.boxplot(
    [raw_changes, avg_changes],
    tick_labels=[
        "Original phase 0",
        "Six-phase average"
    ],
    showfliers=False
)

plt.ylabel("Mean absolute score change")
plt.title("Score stability across ±6-nt anchor shifts")
save_figure("clinical_anchor_stability")


plt.figure(figsize=(7, 5))

plt.hist(
    improvements,
    bins=45
)

plt.axvline(
    0,
    linestyle="--",
    linewidth=1,
    label="No improvement"
)

plt.axvline(
    mean_improvement,
    linewidth=2,
    label=f"Mean improvement: {mean_improvement:.3f}"
)

plt.xlabel(
    "Raw score change minus averaged score change"
)
plt.ylabel("Number of variants")
plt.title("Per-variant stability improvement")
plt.legend()

save_figure("clinical_stability_improvements")


# =====================================================
# 10. Reproducibility metadata and final report
# =====================================================

summary = {
    "experiment": "11_clinvar_stability",
    "research_step": 14,
    "seed": SEED,
    "variants": len(final_rows),
    "anchor_shifts_nt": [-6, 0, 6],
    "phases_per_anchor": PHASES,
    "window_length_nt": WINDOW_LENGTH,
    "score": "log P(ALT token) - log P(REF token)",
    "raw_baseline": "Phase 0 at each anchor",
    "correction": "Mean of six phases at each anchor",
    "raw_mean_absolute_anchor_change": raw_mean,
    "average_mean_absolute_anchor_change": avg_mean,
    "mean_paired_improvement": mean_improvement,
    "relative_reduction_percent": reduction_percent,
    "paired_improvement_ci95": ci95,
    "wilcoxon_one_sided_p": float(test.pvalue),
    "variants_with_positive_improvement": improved_count,
    "subgroups": subgroups,
    "input_sha256": sha256(INPUT),
    "output_sha256": sha256(OUTPUT),
    "script_sha256": sha256(Path(__file__)),
    "model_config_sha256": sha256(
        MODEL_DIR / "config.json"
    ),
    "limitations": [
        "Anchor shifts change flanking sequence context "
        "and token positions, not only tokenization phase",
        "Phase averaging requires six model evaluations "
        "per anchor instead of one",
        "This analysis measures numerical stability, "
        "not pathogenicity prediction accuracy",
        "Variants may be dependent because of shared "
        "genes or genomic loci",
        "The bootstrap resamples variants independently",
        "The center-anchor scores are reused from "
        "Experiment 09"
    ]
}

SUMMARY.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)

print("\nSTEP 14 COMPLETE")
print("Variants:", len(final_rows))
print("Raw mean absolute change:", round(raw_mean, 6))
print("Averaged mean absolute change:", round(avg_mean, 6))
print("Mean paired improvement:", round(mean_improvement, 6))
print("Relative reduction (%):", reduction_percent)
print("95% bootstrap CI:", ci95)
print("Wilcoxon p-value:", test.pvalue)
print("Variants improved:", improved_count)
print("Results:", OUTPUT)
print("Summary:", SUMMARY)
print("Figures:", FIGURE_DIR)
