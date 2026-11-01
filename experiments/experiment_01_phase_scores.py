import random
import torch
from pathlib import Path
from transformers import AutoModelForMaskedLM, EsmTokenizer

ROOT = Path(__file__).resolve().parents[1]
MODEL_DIR = ROOT / "models" / "nt-v2-50m"

random.seed(42)
torch.manual_seed(42)

sequence = "".join(random.choices("ACGT", k=240))
variant_pos = 120
ref_base = sequence[variant_pos]
alt_base = next(base for base in "ACGT" if base != ref_base)

tokenizer = EsmTokenizer(vocab_file=str(MODEL_DIR / "vocab.txt"))

print("Loading model on GPU...")
model = AutoModelForMaskedLM.from_pretrained(
    str(MODEL_DIR),
    trust_remote_code=True,
    local_files_only=True,
    torch_dtype=torch.float32
).to("cuda").eval()

print(f"Variant: {ref_base}>{alt_base} at index {variant_pos}")
print("-" * 65)

scores = []

for phase in range(6):
    start = 48 + phase
    end = start + 144

    ref_sequence = sequence[start:end]
    local_pos = variant_pos - start

    alt_sequence = (
        ref_sequence[:local_pos]
        + alt_base
        + ref_sequence[local_pos + 1:]
    )

    ref_ids = tokenizer(ref_sequence, return_tensors="pt")["input_ids"]
    alt_ids = tokenizer(alt_sequence, return_tensors="pt")["input_ids"]

    differences = (
        (ref_ids != alt_ids)
        .nonzero(as_tuple=False)[:, 1]
        .tolist()
    )

    assert len(differences) == 1, "Expected exactly one changed token"

    token_pos = differences[0]
    ref_token_id = ref_ids[0, token_pos].item()
    alt_token_id = alt_ids[0, token_pos].item()

    masked_ids = ref_ids.clone()
    masked_ids[0, token_pos] = tokenizer.mask_token_id

    with torch.inference_mode():
        logits = model(
            input_ids=masked_ids.to("cuda")
        ).logits[0, token_pos]

        log_probs = torch.log_softmax(logits.float(), dim=-1)

        score = (
            log_probs[alt_token_id]
            - log_probs[ref_token_id]
        ).item()

    scores.append(score)

    print(
        f"Phase {phase} | "
        f"6-mer position {local_pos % 6} | "
        f"Score {score:+.6f}"
    )

print("-" * 65)
print(f"Mean score: {sum(scores) / len(scores):+.6f}")
print(f"Score range: {max(scores) - min(scores):.6f}")
print(f"Score standard deviation: {torch.tensor(scores).std(unbiased=False).item():.6f}")
print("Experiment completed successfully.")
import json
import pandas as pd

output_dir = ROOT / "results" / "experiment_01"
output_dir.mkdir(parents=True, exist_ok=True)

records = [
    {"phase": p, "position_in_6mer": (variant_pos - 48 - p) % 6, "score": s}
    for p, s in enumerate(scores)
]
pd.DataFrame(records).to_csv(output_dir / "phase_scores.csv", index=False)

metadata = {
    "model": "InstaDeepAI/nucleotide-transformer-v2-50m-multi-species",
    "model_revision": "81b29e5786726d891dbf929404ef20adca5b36f1",
    "random_seed": 42,
    "sequence": sequence,
    "variant_position_zero_based": variant_pos,
    "reference_base": ref_base,
    "alternate_base": alt_base,
    "window_length": 144,
    "scoring_method": "masked_token_log_probability_ALT_minus_REF",
    "precision": "float32",
    "torch_version": torch.__version__,
    "phase_scores": records,
    "mean_score": sum(scores) / len(scores),
    "score_range": max(scores) - min(scores),
    "score_std": torch.tensor(scores).std(unbiased=False).item()
}
(output_dir / "experiment_metadata.json").write_text(
    json.dumps(metadata, indent=2), encoding="utf-8"
)
print("Results saved to:", output_dir)
import matplotlib.pyplot as plt

figure_dir = ROOT / "figures" / "experiment_01"
figure_dir.mkdir(parents=True, exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 4.5))
positions = [(variant_pos - 48 - p) % 6 for p in range(6)]

ax.plot(positions, scores, marker="o", linewidth=2)
ax.axhline(0, color="gray", linestyle="--", linewidth=1)
ax.set_xticks(range(6))
ax.set_xlabel("SNV position within 6-mer")
ax.set_ylabel("ALT - REF masked log probability")
ax.set_title("Exploratory SNV Score Variation Across Tokenization Phases")
ax.grid(alpha=0.2)
fig.tight_layout()

fig.savefig(figure_dir / "phase_score_variation.png", dpi=300)
fig.savefig(figure_dir / "phase_score_variation.pdf")
plt.close(fig)

print("Research figures saved to:", figure_dir)
