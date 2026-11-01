
import csv
import hashlib
import json
import random
import statistics
from pathlib import Path

import torch
from transformers import AutoModelForMaskedLM, EsmTokenizer


# ==================================================
# STEP 12 — EXPERIMENT 09
# Clinical variant scoring across six phases
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

MODEL_DIR = ROOT / "models" / "nt-v2-50m"

INPUT = (
    ROOT / "data" / "processed"
    / "clinvar_grch38_sequences.csv"
)

RESULT_DIR = ROOT / "results" / "experiment_09"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = RESULT_DIR / "clinvar_phase_scores.csv"
METADATA = RESULT_DIR / "experiment_metadata.json"

SEED = 20261025
VARIANT_INDEX = 120
WINDOW_LENGTH = 144
BASE_START = 48
PHASES = 6
VARIANTS_PER_BATCH = 4

DEVICE = "cuda"

random.seed(SEED)
torch.manual_seed(SEED)

if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA GPU not available. "
        "This experiment requires the configured RTX 3080."
    )

print("STEP 12 — Clinical variant phase scoring")
print("Device:", torch.cuda.get_device_name(0))


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


# ==================================================
# 1. Load verified ClinVar sequences
# ==================================================

with INPUT.open(
    "r",
    newline="",
    encoding="utf-8"
) as handle:
    variants = list(csv.DictReader(handle))

assert len(variants) == 1000

keys = [variant_key(row) for row in variants]
assert len(set(keys)) == len(keys)

for row in variants:
    sequence = row["reference_sequence_240nt"].upper()

    assert len(sequence) == 240
    assert row["reference_match"] == "PASS"
    assert sequence[VARIANT_INDEX] == row["ref"]
    assert row["ref"] in "ACGT"
    assert row["alt"] in "ACGT"
    assert row["ref"] != row["alt"]

print("Verified clinical variants:", len(variants))


# ==================================================
# 2. Define output fields and resume support
# ==================================================

score_fields = [
    f"phase_{phase}_score"
    for phase in range(PHASES)
]

additional_fields = (
    score_fields
    + [
        "raw_phase0_score",
        "phase_mean_score",
        "phase_median_score",
        "phase_std_score",
        "phase_range",
        "phase_sign_disagreement"
    ]
)

fields = list(variants[0].keys()) + [
    name
    for name in additional_fields
    if name not in variants[0]
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
                "Existing output has an unexpected header. "
                "Do not overwrite it without investigating."
            )

        for row in reader:
            key = variant_key(row)

            if key in completed:
                raise ValueError(
                    f"Duplicate completed variant: {key}"
                )

            for field in score_fields:
                float(row[field])

            completed[key] = row

pending = [
    row
    for row in variants
    if variant_key(row) not in completed
]

print("Previously scored:", len(completed))
print("Remaining variants:", len(pending))


# ==================================================
# 3. Load pinned local model and tokenizer
# ==================================================

print("Loading local nucleotide transformer...")

tokenizer = EsmTokenizer(
    vocab_file=str(MODEL_DIR / "vocab.txt")
)

model = AutoModelForMaskedLM.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float32
)

model = model.to(DEVICE)
model.eval()

if tokenizer.mask_token_id is None:
    raise RuntimeError("Tokenizer has no mask token")

print("Model loaded successfully")


# ==================================================
# 4. Build masked reference/alternate pairs
# ==================================================

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

    for phase in range(PHASES):
        start = BASE_START + phase
        end = start + WINDOW_LENGTH

        ref_window = reference[start:end]
        alt_window = alternate[start:end]

        if len(ref_window) != WINDOW_LENGTH:
            raise ValueError("Invalid reference window")

        if len(alt_window) != WINDOW_LENGTH:
            raise ValueError("Invalid alternate window")

        ref_ids = tokenizer(
            ref_window,
            return_tensors="pt"
        )["input_ids"][0]

        alt_ids = tokenizer(
            alt_window,
            return_tensors="pt"
        )["input_ids"][0]

        if ref_ids.shape != alt_ids.shape:
            raise ValueError(
                "REF/ALT token sequence length differs"
            )

        differences = (
            ref_ids != alt_ids
        ).nonzero(as_tuple=True)[0]

        if len(differences) != 1:
            raise ValueError(
                "Expected exactly one changed token at "
                f"{variant_key(variant)}, phase {phase}; "
                f"observed {len(differences)}"
            )

        token_position = int(differences[0])

        ref_token = int(ref_ids[token_position])
        alt_token = int(alt_ids[token_position])

        masked_ids = ref_ids.clone()

        masked_ids[token_position] = (
            tokenizer.mask_token_id
        )

        prepared.append({
            "phase": phase,
            "masked_ids": masked_ids,
            "token_position": token_position,
            "ref_token": ref_token,
            "alt_token": alt_token
        })

    return prepared


# ==================================================
# 5. Initialize output
# ==================================================

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


# ==================================================
# 6. Score batches using FP32 GPU inference
# ==================================================

for batch_start in range(
    0,
    len(pending),
    VARIANTS_PER_BATCH
):
    batch = pending[
        batch_start:
        batch_start + VARIANTS_PER_BATCH
    ]

    all_prepared = []

    for variant in batch:
        all_prepared.extend(
            prepare_variant(variant)
        )

    masked_batch = torch.stack([
        item["masked_ids"]
        for item in all_prepared
    ]).to(DEVICE)

    positions = torch.tensor(
        [
            item["token_position"]
            for item in all_prepared
        ],
        dtype=torch.long,
        device=DEVICE
    )

    ref_tokens = torch.tensor(
        [
            item["ref_token"]
            for item in all_prepared
        ],
        dtype=torch.long,
        device=DEVICE
    )

    alt_tokens = torch.tensor(
        [
            item["alt_token"]
            for item in all_prepared
        ],
        dtype=torch.long,
        device=DEVICE
    )

    with torch.inference_mode():
        logits = model(
            input_ids=masked_batch
        ).logits

        indices = torch.arange(
            len(all_prepared),
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

        ref_log_probs = log_probs[
            indices,
            ref_tokens
        ]

        alt_log_probs = log_probs[
            indices,
            alt_tokens
        ]

        scores = (
            alt_log_probs - ref_log_probs
        ).cpu().tolist()

    verified_rows = []

    for local_index, variant in enumerate(batch):
        first = local_index * PHASES
        last = first + PHASES

        phase_scores = scores[first:last]

        assert len(phase_scores) == PHASES

        mean_score = statistics.mean(
            phase_scores
        )

        median_score = statistics.median(
            phase_scores
        )

        std_score = statistics.pstdev(
            phase_scores
        )

        score_range = (
            max(phase_scores)
            - min(phase_scores)
        )

        sign_disagreement = int(
            min(phase_scores) < 0
            and max(phase_scores) > 0
        )

        output_row = dict(variant)

        for phase, score in enumerate(
            phase_scores
        ):
            output_row[
                f"phase_{phase}_score"
            ] = score

        output_row.update({
            "raw_phase0_score": phase_scores[0],
            "phase_mean_score": mean_score,
            "phase_median_score": median_score,
            "phase_std_score": std_score,
            "phase_range": score_range,
            "phase_sign_disagreement": (
                sign_disagreement
            )
        })

        verified_rows.append(output_row)

    # Write only after the entire batch succeeds.
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

    completed_count = (
        len(completed)
        + batch_start
        + len(batch)
    )

    if (
        completed_count % 40 == 0
        or completed_count == len(variants)
    ):
        print(
            f"Scored {completed_count}/"
            f"{len(variants)} variants"
        )


# ==================================================
# 7. Validate complete scoring output
# ==================================================

with OUTPUT.open(
    "r",
    newline="",
    encoding="utf-8"
) as handle:
    final_rows = list(csv.DictReader(handle))

assert len(final_rows) == len(variants)

final_keys = [
    variant_key(row)
    for row in final_rows
]

assert len(set(final_keys)) == len(final_keys)
assert set(final_keys) == set(keys)

sign_disagreements = sum(
    int(row["phase_sign_disagreement"])
    for row in final_rows
)

mean_phase_range = statistics.mean(
    float(row["phase_range"])
    for row in final_rows
)


# ==================================================
# 8. Save reproducibility metadata
# ==================================================

metadata = {
    "experiment": "09_clinvar_phase_scores",
    "research_step": 12,
    "model": (
        "InstaDeepAI/"
        "nucleotide-transformer-v2-50m-multi-species"
    ),
    "expected_model_revision": (
        "81b29e5786726d891dbf929404ef20adca5b36f1"
    ),
    "model_source": str(MODEL_DIR),
    "precision": "float32",
    "device": torch.cuda.get_device_name(0),
    "seed": SEED,
    "reference_assembly": "GRCh38",
    "variants": len(final_rows),
    "phases_per_variant": PHASES,
    "total_scores": len(final_rows) * PHASES,
    "window_length": WINDOW_LENGTH,
    "variant_index_zero_based": VARIANT_INDEX,
    "phase_window_starts": [
        BASE_START + phase
        for phase in range(PHASES)
    ],
    "scoring_method": (
        "Masked-token log probability of alternate "
        "6-mer minus reference 6-mer"
    ),
    "raw_score": "phase_0_score",
    "correction": (
        "Arithmetic mean of six phase scores"
    ),
    "phase_sign_disagreements": (
        sign_disagreements
    ),
    "mean_phase_range": mean_phase_range,
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
        "ClinVar labels are clinical classifications, "
        "not direct molecular-effect measurements",
        "The model is not a validated clinical "
        "pathogenicity classifier",
        "Phase averaging is training-free and does "
        "not guarantee better clinical discrimination",
        "Shifting windows also changes flanking "
        "sequence context",
        "The expected model revision is recorded "
        "but not independently verified by this script"
    ]
}

METADATA.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


# ==================================================
# 9. Report
# ==================================================

print("\nSTEP 12 COMPLETE")
print("Variants scored:", len(final_rows))
print("Total phase scores:", len(final_rows) * PHASES)
print("Phase sign disagreements:", sign_disagreements)
print("Mean phase score range:", round(mean_phase_range, 6))
print("Results:", OUTPUT)
print("Metadata:", METADATA)
