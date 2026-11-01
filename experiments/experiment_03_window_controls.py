
import hashlib
import json
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
import transformers
from transformers import AutoModelForMaskedLM, EsmTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "nt-v2-50m"
INPUT_DIR = ROOT / "results" / "experiment_02"
RESULT_DIR = ROOT / "results" / "experiment_03"
FIGURE_DIR = ROOT / "figures" / "experiment_03"

RESULT_DIR.mkdir(parents=True, exist_ok=True)
FIGURE_DIR.mkdir(parents=True, exist_ok=True)

WINDOW_LENGTH = 144
SEED = 20261025
torch.manual_seed(SEED)
started = time.time()

print("Step 06: Same-phase window-shift controls")

variants = pd.read_csv(
    INPUT_DIR / "synthetic_variants.csv"
)
previous_scores = pd.read_csv(
    INPUT_DIR / "all_phase_scores.csv"
)

assert len(variants) == 60
assert len(previous_scores) == 360

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

for _, variant in variants.iterrows():
    variant_id = int(variant["variant_id"])
    sequence = str(variant["sequence"])
    variant_pos = int(variant["variant_position_zero_based"])
    ref = str(variant["ref"])
    alt = str(variant["alt"])

    assert sequence[variant_pos] == ref

    inputs = []
    details = []

    for phase in range(6):
        for offset in (-6, 0, 6):
            window_start = 48 + phase + offset
            window_end = window_start + WINDOW_LENGTH

            assert 0 <= window_start
            assert window_end <= len(sequence)

            local_pos = variant_pos - window_start
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

            changed = (
                (ref_ids != alt_ids)
                .nonzero(as_tuple=False)[:, 1]
                .tolist()
            )

            assert len(changed) == 1

            token_pos = changed[0]
            ref_id = ref_ids[0, token_pos].item()
            alt_id = alt_ids[0, token_pos].item()

            masked = ref_ids.clone()
            masked[0, token_pos] = tokenizer.mask_token_id

            inputs.append(masked)
            details.append({
                "variant_id": variant_id,
                "phase": phase,
                "offset": offset,
                "window_start": window_start,
                "position_in_6mer": local_pos % 6,
                "token_position": token_pos,
                "ref_id": ref_id,
                "alt_id": alt_id
            })

    batch = torch.cat(inputs, dim=0).to("cuda")

    with torch.inference_mode():
        logits = model(input_ids=batch).logits
        log_probs = torch.log_softmax(
            logits.float(), dim=-1
        )

    for row, detail in enumerate(details):
        token_pos = detail.pop("token_position")
        ref_id = detail.pop("ref_id")
        alt_id = detail.pop("alt_id")

        score = (
            log_probs[row, token_pos, alt_id]
            - log_probs[row, token_pos, ref_id]
        ).item()

        detail["score"] = score
        records.append(detail)

    if (variant_id + 1) % 10 == 0:
        print(f"Completed {variant_id + 1}/60 variants")

df = pd.DataFrame(records)
assert len(df) == 1080

# Verify exact agreement with Experiment 2.
baseline = df[df["offset"] == 0].copy()

merged = baseline.merge(
    previous_scores[
        ["variant_id", "phase", "score"]
    ],
    on=["variant_id", "phase"],
    suffixes=("_new", "_previous"),
    validate="one_to_one"
)

assert len(merged) == 360

max_difference = (
    merged["score_new"]
    - merged["score_previous"]
).abs().max()

assert max_difference < 1e-4, (
    f"Reproducibility check failed: {max_difference}"
)

# Within-phase control: compare shifts of +/-6.
wide = df.pivot(
    index=["variant_id", "phase"],
    columns="offset",
    values="score"
)

controls = pd.DataFrame({
    "variant_id": wide.index.get_level_values(0),
    "phase": wide.index.get_level_values(1),
    "baseline_score": wide[0].to_numpy(),
    "minus6_score": wide[-6].to_numpy(),
    "plus6_score": wide[6].to_numpy()
})

controls["abs_delta_minus6"] = (
    controls["minus6_score"]
    - controls["baseline_score"]
).abs()

controls["abs_delta_plus6"] = (
    controls["plus6_score"]
    - controls["baseline_score"]
).abs()

# Different-phase comparisons from the original experiment.
phase_pairs = []

for variant_id, group in baseline.groupby("variant_id"):
    group = group.sort_values("phase")
    rows = group.to_dict("records")

    for i in range(len(rows)):
        for j in range(i + 1, len(rows)):
            phase_pairs.append({
                "variant_id": int(variant_id),
                "phase_a": rows[i]["phase"],
                "phase_b": rows[j]["phase"],
                "absolute_score_difference": abs(
                    rows[i]["score"] - rows[j]["score"]
                )
            })

phase_pairs_df = pd.DataFrame(phase_pairs)

same_phase_differences = pd.concat([
    controls["abs_delta_minus6"],
    controls["abs_delta_plus6"]
], ignore_index=True)

different_phase_differences = (
    phase_pairs_df["absolute_score_difference"]
)

df.to_csv(
    RESULT_DIR / "all_control_scores.csv",
    index=False
)
controls.to_csv(
    RESULT_DIR / "same_phase_controls.csv",
    index=False
)
phase_pairs_df.to_csv(
    RESULT_DIR / "different_phase_pairs.csv",
    index=False
)

fig, ax = plt.subplots(figsize=(9, 5))

ax.hist(
    same_phase_differences,
    bins=25,
    alpha=0.6,
    density=True,
    label="Same phase: +/-6 nt"
)

ax.hist(
    different_phase_differences,
    bins=25,
    alpha=0.6,
    density=True,
    label="Different phases"
)

ax.set_xlabel("Absolute difference in variant score")
ax.set_ylabel("Density")
ax.set_title(
    "Exploratory Comparison of Window-Shift Effects"
)
ax.legend()
fig.tight_layout()

fig.savefig(
    FIGURE_DIR / "window_control_comparison.png",
    dpi=300
)
fig.savefig(
    FIGURE_DIR / "window_control_comparison.pdf"
)
plt.close(fig)

metadata = {
    "experiment": "03_same_phase_window_controls",
    "model": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "model_revision": "81b29e5786726d891dbf929404ef20adca5b36f1",
    "seed": SEED,
    "variants": len(variants),
    "scores": len(df),
    "window_length": WINDOW_LENGTH,
    "same_phase_offsets": [-6, 0, 6],
    "max_baseline_reproduction_difference": float(max_difference),
    "mean_abs_same_phase_difference": float(
        same_phase_differences.mean()
    ),
    "mean_abs_different_phase_difference": float(
        different_phase_differences.mean()
    ),
    "torch_version": torch.__version__,
    "transformers_version": transformers.__version__,
    "python_version": platform.python_version(),
    "gpu": torch.cuda.get_device_name(0),
    "script_sha256": hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest(),
    "runtime_seconds": round(time.time() - started, 2),
    "limitations": [
        "Same-phase and different-phase shifts have different distances",
        "Shifting windows changes flanking context and token position",
        "Synthetic DNA only; no biological ground truth",
        "This control does not establish tokenization causality"
    ]
}

(RESULT_DIR / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("\nEXPERIMENT 3 COMPLETE")
print("Total scores:", len(df))
print("Baseline reproduction max difference:", max_difference)
print(
    "Mean absolute same-phase difference:",
    round(metadata["mean_abs_same_phase_difference"], 6)
)
print(
    "Mean absolute different-phase difference:",
    round(metadata["mean_abs_different_phase_difference"], 6)
)
print("Results:", RESULT_DIR)
print("Figures:", FIGURE_DIR)
