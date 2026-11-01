
import csv
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import roc_auc_score
from transformers import AutoModelForMaskedLM, EsmTokenizer


# ==================================================
# STEP 17 — MAIN EXPERIMENT A
# Fixed-core phase and flank controls
# ==================================================

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "nt-v2-50m"

INPUT = (
    ROOT / "data" / "processed"
    / "clinvar_grch38_sequences.csv"
)

RESULT_DIR = ROOT / "results" / "main_experiment_A"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = RESULT_DIR / "fixed_core_scores.csv"
SUMMARY = RESULT_DIR / "fixed_core_analysis.json"

SEED = 20261025
VARIANT_INDEX = 120
CORE_START = 48
CORE_LENGTH = 144
FLANK_LENGTH = 6
INPUT_LENGTH = 150
PHASES = 6
VARIANTS_PER_BATCH = 4
BOOTSTRAPS = 2000
DEVICE = "cuda"

CONDITIONS = ("genomic", "rotated")

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


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


def rotate_flank(sequence):
    return sequence[3:] + sequence[:3]


print("STEP 17 — Main Experiment A")
print("Fixed-core phase and flank controls")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU is required")


# ==================================================
# 1. Load and validate ClinVar sequences
# ==================================================

with INPUT.open(
    "r",
    newline="",
    encoding="utf-8-sig"
) as handle:
    variants = list(csv.DictReader(handle))

assert len(variants) == 1000

assert len({
    variant_key(row)
    for row in variants
}) == 1000

for row in variants:
    sequence = row["reference_sequence_240nt"].upper()

    assert len(sequence) == 240
    assert row["reference_match"] == "PASS"
    assert sequence[VARIANT_INDEX] == row["ref"].upper()
    assert row["ref"].upper() in "ACGT"
    assert row["alt"].upper() in "ACGT"
    assert row["ref"].upper() != row["alt"].upper()

    assert row["clinical_label"].lower() in (
        "benign",
        "pathogenic"
    )


# ==================================================
# 2. Prepare resumable output
# ==================================================

score_fields = [
    f"{condition}_phase_{phase}"
    for condition in CONDITIONS
    for phase in range(PHASES)
]

fields = [
    "chromosome",
    "position_grch38_1based",
    "ref",
    "alt",
    "clinical_label"
] + score_fields

completed = {}

if OUTPUT.exists():
    with OUTPUT.open(
        "r",
        newline="",
        encoding="utf-8"
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != fields:
            raise RuntimeError(
                "Existing result header differs. "
                "Investigate before proceeding."
            )

        for row in reader:
            key = variant_key(row)

            if key in completed:
                raise RuntimeError(
                    f"Duplicate result: {key}"
                )

            for field in score_fields:
                assert np.isfinite(float(row[field]))

            completed[key] = row

pending = [
    row for row in variants
    if variant_key(row) not in completed
]

print("Variants:", len(variants))
print("Previously scored:", len(completed))
print("Remaining:", len(pending))
print("GPU:", torch.cuda.get_device_name(0))


# ==================================================
# 3. Load original model and tokenizer
# ==================================================

if pending:
    tokenizer = EsmTokenizer(
        vocab_file=str(MODEL_DIR / "vocab.txt")
    )

    model = AutoModelForMaskedLM.from_pretrained(
        str(MODEL_DIR),
        trust_remote_code=True,
        local_files_only=True,
        torch_dtype=torch.float32
    ).to(DEVICE)

    model.eval()

    assert tokenizer.mask_token_id is not None

    print("Model loaded")


# ==================================================
# 4. Build fixed-core experimental windows
# ==================================================

def prepare_variant(row):
    reference = (
        row["reference_sequence_240nt"].upper()
    )

    core_ref = reference[
        CORE_START:CORE_START + CORE_LENGTH
    ]

    core_alt = list(core_ref)

    core_alt[
        VARIANT_INDEX - CORE_START
    ] = row["alt"].upper()

    core_alt = "".join(core_alt)

    upstream = reference[
        CORE_START - FLANK_LENGTH:CORE_START
    ]

    downstream = reference[
        CORE_START + CORE_LENGTH:
        CORE_START + CORE_LENGTH + FLANK_LENGTH
    ]

    assert len(core_ref) == CORE_LENGTH
    assert len(upstream) == FLANK_LENGTH
    assert len(downstream) == FLANK_LENGTH

    prepared = []

    for condition in CONDITIONS:
        if condition == "genomic":
            left = upstream
            right = downstream
        else:
            left = rotate_flank(upstream)
            right = rotate_flank(downstream)

        for phase in range(PHASES):
            prefix = left[
                FLANK_LENGTH - phase:
            ]

            suffix = right[
                :FLANK_LENGTH - phase
            ]

            ref_window = (
                prefix + core_ref + suffix
            )

            alt_window = (
                prefix + core_alt + suffix
            )

            assert len(ref_window) == INPUT_LENGTH
            assert len(alt_window) == INPUT_LENGTH

            ref_ids = tokenizer(
                ref_window,
                return_tensors="pt"
            )["input_ids"][0]

            alt_ids = tokenizer(
                alt_window,
                return_tensors="pt"
            )["input_ids"][0]

            assert ref_ids.shape == alt_ids.shape

            changed = (
                ref_ids != alt_ids
            ).nonzero(as_tuple=True)[0]

            if len(changed) != 1:
                raise RuntimeError(
                    "Expected exactly one changed token: "
                    f"{variant_key(row)}, "
                    f"{condition}, phase {phase}"
                )

            position = int(changed[0])

            masked = ref_ids.clone()

            masked[position] = (
                tokenizer.mask_token_id
            )

            prepared.append({
                "condition": condition,
                "phase": phase,
                "masked_ids": masked,
                "position": position,
                "ref_token": int(ref_ids[position]),
                "alt_token": int(alt_ids[position])
            })

    return prepared


# ==================================================
# 5. Score all variants using FP32 GPU inference
# ==================================================

if pending and not OUTPUT.exists():
    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as handle:
        csv.DictWriter(
            handle,
            fieldnames=fields
        ).writeheader()

if pending:
    for start in range(
        0,
        len(pending),
        VARIANTS_PER_BATCH
    ):
        batch = pending[
            start:start + VARIANTS_PER_BATCH
        ]

        prepared = [
            item
            for row in batch
            for item in prepare_variant(row)
        ]

        masked = torch.stack([
            item["masked_ids"]
            for item in prepared
        ]).to(DEVICE)

        positions = torch.tensor(
            [
                item["position"]
                for item in prepared
            ],
            device=DEVICE
        )

        ref_tokens = torch.tensor(
            [
                item["ref_token"]
                for item in prepared
            ],
            device=DEVICE
        )

        alt_tokens = torch.tensor(
            [
                item["alt_token"]
                for item in prepared
            ],
            device=DEVICE
        )

        with torch.inference_mode():
            logits = model(
                input_ids=masked
            ).logits

            indices = torch.arange(
                len(prepared),
                device=DEVICE
            )

            selected_logits = logits[
                indices,
                positions
            ].float()

            log_probs = torch.log_softmax(
                selected_logits,
                dim=-1
            )

            scores = (
                log_probs[indices, alt_tokens]
                - log_probs[indices, ref_tokens]
            ).cpu().tolist()

        output_rows = []

        for j, variant in enumerate(batch):
            output_row = {
                field: variant[field]
                for field in fields
                if field not in score_fields
            }

            segment = scores[
                j * 12:(j + 1) * 12
            ]

            assert len(segment) == 12

            for c, condition in enumerate(
                CONDITIONS
            ):
                for phase in range(PHASES):
                    output_row[
                        f"{condition}_phase_{phase}"
                    ] = segment[
                        c * PHASES + phase
                    ]

            output_rows.append(output_row)

        with OUTPUT.open(
            "a",
            newline="",
            encoding="utf-8"
        ) as handle:
            writer = csv.DictWriter(
                handle,
                fieldnames=fields
            )

            writer.writerows(output_rows)

        total = (
            len(completed)
            + start
            + len(batch)
        )

        if (
            total % 40 == 0
            or total == len(variants)
        ):
            print(
                f"Scored {total}/"
                f"{len(variants)} variants"
            )

    del model
    torch.cuda.empty_cache()


# ==================================================
# 6. Validate completed scores
# ==================================================

with OUTPUT.open(
    "r",
    newline="",
    encoding="utf-8"
) as handle:
    results = list(csv.DictReader(handle))

assert len(results) == len(variants)

assert len({
    variant_key(row)
    for row in results
}) == len(variants)

assert {
    variant_key(row)
    for row in results
} == {
    variant_key(row)
    for row in variants
}

results.sort(key=variant_key)

labels = np.array([
    int(
        row["clinical_label"].lower()
        == "pathogenic"
    )
    for row in results
])

assert labels.sum() == 500

matrices = {}

for condition in CONDITIONS:
    matrix = np.array([
        [
            float(
                row[f"{condition}_phase_{phase}"]
            )
            for phase in range(PHASES)
        ]
        for row in results
    ])

    assert matrix.shape == (1000, 6)
    assert np.isfinite(matrix).all()

    matrices[condition] = matrix


# ==================================================
# 7. Quantify phase and flank sensitivity
# ==================================================

pairs = [
    (a, b)
    for a in range(PHASES)
    for b in range(a + 1, PHASES)
]

phase_effect = np.mean([
    np.mean([
        np.abs(
            matrix[:, a] - matrix[:, b]
        )
        for a, b in pairs
    ], axis=0)
    for matrix in matrices.values()
], axis=0)

flank_effect = np.abs(
    matrices["genomic"]
    - matrices["rotated"]
).mean(axis=1)

contrast = phase_effect - flank_effect

raw_flank_change = np.abs(
    matrices["genomic"][:, 0]
    - matrices["rotated"][:, 0]
)

averaged_flank_change = np.abs(
    matrices["genomic"].mean(axis=1)
    - matrices["rotated"].mean(axis=1)
)

stability_improvement = (
    raw_flank_change
    - averaged_flank_change
)


# ==================================================
# 8. Exploratory clinical AUROC
# ==================================================

# Negative score orientation is fixed here.
# This is NOT independent clinical validation.

auc = {}

for condition, matrix in matrices.items():
    raw_auc = roc_auc_score(
        labels,
        -matrix[:, 0]
    )

    averaged_auc = roc_auc_score(
        labels,
        -matrix.mean(axis=1)
    )

    auc[condition] = {
        "phase0": float(raw_auc),
        "six_phase_mean": float(averaged_auc),
        "delta": float(
            averaged_auc - raw_auc
        )
    }


# ==================================================
# 9. Paired class-stratified bootstrap
# ==================================================

rng = np.random.default_rng(SEED)

benign = np.flatnonzero(labels == 0)
pathogenic = np.flatnonzero(labels == 1)

boot_contrast = []
boot_stability = []

boot_auc_delta = {
    condition: []
    for condition in CONDITIONS
}

for iteration in range(BOOTSTRAPS):
    indices = np.concatenate([
        rng.choice(
            benign,
            len(benign),
            replace=True
        ),
        rng.choice(
            pathogenic,
            len(pathogenic),
            replace=True
        )
    ])

    boot_contrast.append(
        float(contrast[indices].mean())
    )

    boot_stability.append(
        float(
            stability_improvement[
                indices
            ].mean()
        )
    )

    for condition, matrix in matrices.items():
        sampled = matrix[indices]
        sampled_labels = labels[indices]

        delta = (
            roc_auc_score(
                sampled_labels,
                -sampled.mean(axis=1)
            )
            - roc_auc_score(
                sampled_labels,
                -sampled[:, 0]
            )
        )

        boot_auc_delta[
            condition
        ].append(float(delta))


def confidence_interval(values):
    return [
        float(value)
        for value in np.percentile(
            values,
            [2.5, 97.5]
        )
    ]


# ==================================================
# 10. Save reproducibility metadata
# ==================================================

summary = {
    "experiment": (
        "main_A_fixed_core_phase_flank_controls"
    ),
    "research_step": 17,
    "model": (
        "InstaDeepAI/"
        "nucleotide-transformer-v2-50m-multi-species"
    ),
    "expected_model_revision_not_verified": (
        "81b29e5786726d891dbf929404ef20adca5b36f1"
    ),
    "precision": "float32",
    "scoring_method": (
        "Masked log P(ALT 6-mer) "
        "minus log P(REF 6-mer)"
    ),
    "seed": SEED,
    "variants": len(results),
    "total_scores": (
        len(results)
        * PHASES
        * len(CONDITIONS)
    ),
    "core_start_zero_based": CORE_START,
    "core_length_nt": CORE_LENGTH,
    "input_length_nt": INPUT_LENGTH,
    "flank_conditions": [
        "Real genomic flanks",
        "Each 6nt genomic flank rotated by 3nt"
    ],
    "mean_phase_effect": float(
        phase_effect.mean()
    ),
    "mean_flank_effect": float(
        flank_effect.mean()
    ),
    "mean_phase_minus_flank_effect": float(
        contrast.mean()
    ),
    "phase_minus_flank_ci95": (
        confidence_interval(boot_contrast)
    ),
    "variants_phase_effect_greater": int(
        (contrast > 0).sum()
    ),
    "mean_raw_flank_change": float(
        raw_flank_change.mean()
    ),
    "mean_averaged_flank_change": float(
        averaged_flank_change.mean()
    ),
    "mean_stability_improvement": float(
        stability_improvement.mean()
    ),
    "stability_improvement_ci95": (
        confidence_interval(boot_stability)
    ),
    "exploratory_clinical_auroc": auc,
    "exploratory_auroc_delta_ci95": {
        condition: confidence_interval(values)
        for condition, values
        in boot_auc_delta.items()
    },
    "input_sha256": sha256(INPUT),
    "output_sha256": sha256(OUTPUT),
    "script_sha256": sha256(Path(__file__)),
    "model_config_sha256": sha256(
        MODEL_DIR / "config.json"
    ),
    "tokenizer_vocab_sha256": sha256(
        MODEL_DIR / "vocab.txt"
    ),
    "limitations": [
        "Phase comparisons retain the same "
        "144nt core but alter distal flank "
        "allocation.",
        "Rotated flanks are artificial "
        "sequence-context perturbations.",
        "The design does not uniquely establish "
        "a causal tokenization-phase effect.",
        "ClinVar labels are not direct "
        "molecular functional measurements.",
        "This is not independent clinical "
        "validation.",
        "Variants may be correlated through "
        "shared genes or loci.",
        "Results from 150nt inputs must not "
        "be directly compared with previous "
        "144nt score magnitudes."
    ]
}

SUMMARY.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8"
)


# ==================================================
# 11. Final report
# ==================================================

print("\nSTEP 17 COMPLETE")

print(
    "Variants:",
    len(results)
)

print(
    "Total scores:",
    summary["total_scores"]
)

print(
    "Mean phase effect:",
    round(summary["mean_phase_effect"], 6)
)

print(
    "Mean flank effect:",
    round(summary["mean_flank_effect"], 6)
)

print(
    "Phase minus flank contrast:",
    round(
        summary["mean_phase_minus_flank_effect"],
        6
    )
)

print(
    "Contrast 95% CI:",
    summary["phase_minus_flank_ci95"]
)

print(
    "Raw flank sensitivity:",
    round(summary["mean_raw_flank_change"], 6)
)

print(
    "Averaged flank sensitivity:",
    round(summary["mean_averaged_flank_change"], 6)
)

print(
    "Stability improvement 95% CI:",
    summary["stability_improvement_ci95"]
)

print(
    "Exploratory clinical AUROC:",
    auc
)

print(
    "AUROC delta 95% CIs:",
    summary["exploratory_auroc_delta_ci95"]
)

print("Results:", OUTPUT)
print("Summary:", SUMMARY)
