
import hashlib
import json
import platform
import random
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import transformers
from transformers import AutoModelForMaskedLM, EsmTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "nt-v2-50m"
RESULT_DIR = ROOT / "results" / "experiment_02"
FIGURE_DIR = ROOT / "figures" / "experiment_02"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

SEED = 20261025
N_VARIANTS = 60
VARIANT_POS = 120
WINDOW_LENGTH = 144
MODEL_REVISION = "81b29e5786726d891dbf929404ef20adca5b36f1"

rng = random.Random(SEED)
torch.manual_seed(SEED)
start_time = time.time()

print("Step 05: Multi-variant phase evaluation")
print("Loading model...")

tokenizer = EsmTokenizer(
    vocab_file=str(MODEL_DIR / "vocab.txt")
)

model = AutoModelForMaskedLM.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float32
).to("cuda").eval()

records = []
variants = []

for variant_id in range(N_VARIANTS):
    sequence = "".join(
        rng.choices("ACGT", k=240)
    )

    ref = sequence[VARIANT_POS]
    alt = rng.choice(
        [base for base in "ACGT" if base != ref]
    )

    variants.append({
        "variant_id": variant_id,
        "sequence": sequence,
        "variant_position_zero_based": VARIANT_POS,
        "ref": ref,
        "alt": alt
    })

    masked_batches = []
    token_pairs = []
    phase_info = []

    for phase in range(6):
        window_start = 48 + phase
        window_end = window_start + WINDOW_LENGTH
        local_pos = VARIANT_POS - window_start

        ref_seq = sequence[window_start:window_end]
        alt_seq = (
            ref_seq[:local_pos]
            + alt
            + ref_seq[local_pos + 1:]
        )

        ref_ids = tokenizer(
            ref_seq, return_tensors="pt"
        )["input_ids"]

        alt_ids = tokenizer(
            alt_seq, return_tensors="pt"
        )["input_ids"]

        differences = (
            (ref_ids != alt_ids)
            .nonzero(as_tuple=False)[:, 1]
            .tolist()
        )

        assert len(differences) == 1, (
            f"Unexpected token changes: variant "
            f"{variant_id}, phase {phase}"
        )

        token_pos = differences[0]
        ref_id = ref_ids[0, token_pos].item()
        alt_id = alt_ids[0, token_pos].item()

        masked = ref_ids.clone()
        masked[0, token_pos] = tokenizer.mask_token_id

        masked_batches.append(masked)
        token_pairs.append((token_pos, ref_id, alt_id))
        phase_info.append((phase, window_start, local_pos))

    batch = torch.cat(masked_batches, dim=0).to("cuda")

    with torch.inference_mode():
        logits = model(input_ids=batch).logits
        log_probs = torch.log_softmax(
            logits.float(), dim=-1
        )

    for row, (phase, window_start, local_pos) in enumerate(phase_info):
        token_pos, ref_id, alt_id = token_pairs[row]

        score = (
            log_probs[row, token_pos, alt_id]
            - log_probs[row, token_pos, ref_id]
        ).item()

        records.append({
            "variant_id": variant_id,
            "ref": ref,
            "alt": alt,
            "phase": phase,
            "position_in_6mer": local_pos % 6,
            "window_start": window_start,
            "ref_token": tokenizer.convert_ids_to_tokens(ref_id),
            "alt_token": tokenizer.convert_ids_to_tokens(alt_id),
            "score": score
        })

    if (variant_id + 1) % 10 == 0:
        print(f"Completed {variant_id + 1}/{N_VARIANTS} variants")

scores_df = pd.DataFrame(records)
variants_df = pd.DataFrame(variants)

summary = scores_df.groupby("variant_id")["score"].agg(
    mean_score="mean",
    min_score="min",
    max_score="max",
    std_score=lambda x: x.std(ddof=0)
).reset_index()

summary["score_range"] = (
    summary["max_score"] - summary["min_score"]
)

sign_flips = scores_df.groupby("variant_id")["score"].agg(
    lambda x: bool((x > 0).any() and (x < 0).any())
)

summary["sign_flip"] = summary["variant_id"].map(sign_flips)

scores_df.to_csv(
    RESULT_DIR / "all_phase_scores.csv", index=False
)
variants_df.to_csv(
    RESULT_DIR / "synthetic_variants.csv", index=False
)
summary.to_csv(
    RESULT_DIR / "variant_summary.csv", index=False
)

fig, ax = plt.subplots(figsize=(8, 5))
ax.hist(summary["score_range"], bins=15, edgecolor="black")
ax.set_xlabel("Score range across six phases")
ax.set_ylabel("Number of variants")
ax.set_title("Phase-Associated Score Variation: 60 Synthetic SNVs")
fig.tight_layout()
fig.savefig(
    FIGURE_DIR / "score_range_distribution.png", dpi=300
)
fig.savefig(
    FIGURE_DIR / "score_range_distribution.pdf"
)
plt.close(fig)

script_hash = hashlib.sha256(
    Path(__file__).read_bytes()
).hexdigest()

metadata = {
    "experiment": "02_multi_variant_phase_evaluation",
    "experiment_type": "exploratory_synthetic",
    "model": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "model_revision": MODEL_REVISION,
    "random_seed": SEED,
    "number_of_variants": N_VARIANTS,
    "number_of_scores": len(records),
    "window_length": WINDOW_LENGTH,
    "scoring_method": "masked_token_log_probability_ALT_minus_REF",
    "precision": "float32",
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "python_version": platform.python_version(),
    "gpu": torch.cuda.get_device_name(0),
    "script_sha256": script_hash,
    "mean_score_range": float(summary["score_range"].mean()),
    "median_score_range": float(summary["score_range"].median()),
    "sign_flip_count": int(summary["sign_flip"].sum()),
    "sign_flip_fraction": float(summary["sign_flip"].mean()),
    "runtime_seconds": round(time.time() - start_time, 2),
    "limitations": [
        "Synthetic random DNA sequences",
        "Shifted windows have different flanking bases",
        "No biological ground-truth labels",
        "No independent tokenizer or model control"
    ]
}

(RESULT_DIR / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("\nEXPERIMENT 2 COMPLETE")
print("Variants:", N_VARIANTS)
print("Scores:", len(records))
print("Mean score range:", round(metadata["mean_score_range"], 6))
print("Sign flips:", metadata["sign_flip_count"])
print("Results:", RESULT_DIR)
print("Figures:", FIGURE_DIR)
