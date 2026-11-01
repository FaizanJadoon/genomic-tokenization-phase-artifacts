
import csv
import hashlib
import json
import random
import statistics
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoModelForMaskedLM, EsmTokenizer


# =====================================================
# STEP 19 — MAIN EXPERIMENT C
# Computational cost of six-phase SNV scoring
# =====================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed"
    / "clinvar_heldout_grch38_sequences.csv"
)

MODEL_DIR = ROOT / "models" / "nt-v2-50m"

RESULT_DIR = ROOT / "results" / "main_experiment_C"
RESULT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT = RESULT_DIR / "compute_benchmark.json"

SEED = 20261026
N_VARIANTS = 120
BATCH_SIZE = 4
REPEATS = 3

VARIANT_INDEX = 120
BASE_START = 48
WINDOW_LENGTH = 144

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(
            lambda: handle.read(1024 * 1024),
            b"",
        ):
            digest.update(chunk)
    return digest.hexdigest()


print("STEP 19 — Computational-cost benchmark")

with INPUT.open(
    "r", newline="", encoding="utf-8-sig"
) as handle:
    variants = list(csv.DictReader(handle))

if len(variants) != 1000:
    raise RuntimeError("Unexpected held-out cohort size")

# Deterministic, balanced sample from the held-out cohort.
rng = random.Random(SEED)

benign = [
    row for row in variants
    if row["clinical_label"] == "Benign"
]

pathogenic = [
    row for row in variants
    if row["clinical_label"] == "Pathogenic"
]

selected = (
    rng.sample(benign, N_VARIANTS // 2)
    + rng.sample(pathogenic, N_VARIANTS // 2)
)

rng.shuffle(selected)

for row in selected:
    sequence = row["reference_sequence_240nt"].upper()
    if (
        len(sequence) != 240
        or sequence[VARIANT_INDEX] != row["ref"]
        or set(sequence) - set("ACGT")
    ):
        raise RuntimeError("Invalid reference sequence")

if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU unavailable")

device = "cuda"

print("GPU:", torch.cuda.get_device_name(0))
print("Benchmark variants:", len(selected))
print("Variant batch size:", BATCH_SIZE)
print("Repeated measurements:", REPEATS)

load_start = time.perf_counter()

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

torch.cuda.synchronize()

load_seconds = time.perf_counter() - load_start
print("Model loaded")


def prepare_variant(variant, phases):
    reference = variant["reference_sequence_240nt"].upper()

    alternate = (
        reference[:VARIANT_INDEX]
        + variant["alt"]
        + reference[VARIANT_INDEX + 1:]
    )

    prepared = []

    for phase in phases:
        start = BASE_START + phase
        end = start + WINDOW_LENGTH

        ref_window = reference[start:end]
        alt_window = alternate[start:end]

        ref_ids = tokenizer(
            ref_window, return_tensors="pt"
        )["input_ids"][0]

        alt_ids = tokenizer(
            alt_window, return_tensors="pt"
        )["input_ids"][0]

        differences = (
            ref_ids != alt_ids
        ).nonzero(as_tuple=True)[0]

        if len(differences) != 1:
            raise RuntimeError(
                "Expected one changed token"
            )

        position = int(differences[0])

        masked = ref_ids.clone()
        masked[position] = tokenizer.mask_token_id

        prepared.append((
            masked,
            position,
            int(ref_ids[position]),
            int(alt_ids[position]),
        ))

    return prepared


def score_batch(batch, phases):
    prepared = []

    for variant in batch:
        prepared.extend(
            prepare_variant(variant, phases)
        )

    masked_ids = torch.stack([
        item[0] for item in prepared
    ]).to(device)

    positions = torch.tensor(
        [item[1] for item in prepared],
        dtype=torch.long,
        device=device,
    )

    ref_tokens = torch.tensor(
        [item[2] for item in prepared],
        dtype=torch.long,
        device=device,
    )

    alt_tokens = torch.tensor(
        [item[3] for item in prepared],
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

        selected_logits = logits[
            indices, positions
        ].float()

        log_probs = torch.log_softmax(
            selected_logits,
            dim=-1,
        )

        scores = (
            log_probs[indices, alt_tokens]
            - log_probs[indices, ref_tokens]
        )

    return scores.cpu().numpy().reshape(
        len(batch), len(phases)
    )


def benchmark(mode, phases):
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    baseline_allocated = (
        torch.cuda.memory_allocated()
    )

    start = time.perf_counter()
    collected = []

    for offset in range(
        0, len(selected), BATCH_SIZE
    ):
        batch = selected[
            offset:offset + BATCH_SIZE
        ]

        scores = score_batch(batch, phases)

        collected.append(scores)

    torch.cuda.synchronize()

    elapsed = time.perf_counter() - start

    all_scores = np.concatenate(
        collected, axis=0
    )

    peak_allocated = (
        torch.cuda.max_memory_allocated()
    )

    peak_reserved = (
        torch.cuda.max_memory_reserved()
    )

    return {
        "mode": mode,
        "seconds": elapsed,
        "milliseconds_per_variant": (
            1000.0 * elapsed / len(selected)
        ),
        "variants_per_second": (
            len(selected) / elapsed
        ),
        "peak_allocated_mib": (
            peak_allocated / 1048576
        ),
        "peak_reserved_mib": (
            peak_reserved / 1048576
        ),
        "peak_incremental_allocated_mib": (
            max(
                0,
                peak_allocated - baseline_allocated,
            ) / 1048576
        ),
        "scores": all_scores,
    }


# =====================================================
# Warm up both modes before measuring
# =====================================================

warmup = selected[:BATCH_SIZE]

for _ in range(2):
    score_batch(warmup, [0])
    score_batch(warmup, list(range(6)))

torch.cuda.synchronize()

print("GPU warmup complete")


# =====================================================
# Timed repetitions, alternating mode order
# =====================================================

records = {
    "single_phase": [],
    "six_phase": [],
}

maximum_score_difference = 0.0

for repeat in range(REPEATS):
    order = (
        ["single_phase", "six_phase"]
        if repeat % 2 == 0
        else ["six_phase", "single_phase"]
    )

    outputs = {}

    for mode in order:
        phases = (
            [0]
            if mode == "single_phase"
            else list(range(6))
        )

        result = benchmark(mode, phases)

        outputs[mode] = result

        records[mode].append({
            key: value
            for key, value in result.items()
            if key != "scores"
        })

        print(
            f"Repeat {repeat + 1}, {mode}: "
            f"{result['seconds']:.3f}s, "
            f"{result['milliseconds_per_variant']:.3f} "
            "ms/variant"
        )

    # Phase 0 must be numerically identical
    # in both scoring configurations.
    raw_scores = outputs[
        "single_phase"
    ]["scores"][:, 0]

    six_phase0_scores = outputs[
        "six_phase"
    ]["scores"][:, 0]

    difference = float(
        np.max(
            np.abs(
                raw_scores - six_phase0_scores
            )
        )
    )

    maximum_score_difference = max(
        maximum_score_difference,
        difference,
    )

    if not np.allclose(
        raw_scores,
        six_phase0_scores,
        rtol=1e-5,
        atol=1e-5,
    ):
        raise RuntimeError(
            "Phase-0 scoring differs between modes"
        )


# =====================================================
# Summarize timing and memory
# =====================================================

def summarize(mode):
    runs = records[mode]

    return {
        "repeats": runs,
        "median_milliseconds_per_variant": (
            statistics.median(
                run["milliseconds_per_variant"]
                for run in runs
            )
        ),
        "median_variants_per_second": (
            statistics.median(
                run["variants_per_second"]
                for run in runs
            )
        ),
        "maximum_peak_allocated_mib": max(
            run["peak_allocated_mib"]
            for run in runs
        ),
        "maximum_peak_reserved_mib": max(
            run["peak_reserved_mib"]
            for run in runs
        ),
        "maximum_incremental_allocated_mib": max(
            run["peak_incremental_allocated_mib"]
            for run in runs
        ),
    }


single = summarize("single_phase")
six = summarize("six_phase")

latency_ratio = (
    six["median_milliseconds_per_variant"]
    / single["median_milliseconds_per_variant"]
)

summary = {
    "experiment": "18_compute_benchmark",
    "research_step": 19,
    "model": "Nucleotide Transformer v2 50M",
    "gpu": torch.cuda.get_device_name(0),
    "torch_version": torch.__version__,
    "cuda_version": torch.version.cuda,
    "dtype": "float32",
    "variants": N_VARIANTS,
    "batch_size_variants": BATCH_SIZE,
    "repeats": REPEATS,
    "window_length_nt": WINDOW_LENGTH,
    "single_phase_scores_per_variant": 1,
    "six_phase_scores_per_variant": 6,
    "model_load_seconds": load_seconds,
    "single_phase": single,
    "six_phase": six,
    "median_latency_ratio": latency_ratio,
    "maximum_phase0_score_difference": (
        maximum_score_difference
    ),
    "timing_scope": (
        "Warm-model end-to-end sequence tokenization, "
        "GPU transfer, forward inference, score "
        "extraction and CPU result transfer"
    ),
    "input_sha256": sha256(INPUT),
    "model_config_sha256": sha256(
        MODEL_DIR / "config.json"
    ),
    "tokenizer_vocab_sha256": sha256(
        MODEL_DIR / "vocab.txt"
    ),
    "script_sha256": sha256(Path(__file__)),
    "limitations": [
        "One GPU and one batch-size configuration",
        "120 sampled held-out variants",
        "Three repeated timing measurements",
        "Model loading excluded from main latency",
        "GPU timings may vary with thermal state "
        "and competing workloads",
        "Peak reserved memory includes allocator caching",
        "Batched six-phase inference may not cost "
        "exactly six times the wall-clock latency",
    ],
}

OUTPUT.write_text(
    json.dumps(summary, indent=2),
    encoding="utf-8",
)

print("\nSTEP 19 COMPLETE")

print(
    "Single-phase median ms/variant:",
    round(
        single["median_milliseconds_per_variant"],
        3,
    ),
)

print(
    "Six-phase median ms/variant:",
    round(
        six["median_milliseconds_per_variant"],
        3,
    ),
)

print(
    "Latency ratio:",
    round(latency_ratio, 3),
)

print(
    "Single-phase peak allocated MiB:",
    round(
        single["maximum_peak_allocated_mib"],
        2,
    ),
)

print(
    "Six-phase peak allocated MiB:",
    round(
        six["maximum_peak_allocated_mib"],
        2,
    ),
)

print(
    "Maximum phase-0 score difference:",
    maximum_score_difference,
)

print("Summary:", OUTPUT)
