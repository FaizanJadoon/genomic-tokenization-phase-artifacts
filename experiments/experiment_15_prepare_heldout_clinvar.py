
import csv
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path


# ==================================================
# STEP 18 — MAIN EXPERIMENT B1
# Held-out ClinVar cohort preparation
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

SOURCE = (
    ROOT / "data" / "raw"
    / "clinvar_GRCh38.vcf.gz"
)

ORIGINAL = (
    ROOT / "data" / "processed"
    / "clinvar_high_confidence_snvs.csv"
)

OUTPUT_DIR = ROOT / "data" / "processed"

OUTPUT = (
    OUTPUT_DIR / "clinvar_heldout_snvs.csv"
)

METADATA = (
    OUTPUT_DIR / "clinvar_heldout_metadata.json"
)

SEED = 20261026
MAX_PER_CLASS = 500

ALLOWED_REVIEW = {
    "criteria_provided,_multiple_submitters,_no_conflicts",
    "reviewed_by_expert_panel",
    "practice_guideline",
}

CLASSES = ("Pathogenic", "Benign")

FIELDS = [
    "chromosome",
    "position_grch38_1based",
    "variant_id",
    "ref",
    "alt",
    "clinical_label",
    "review_status",
    "clinical_allele_id",
]

EXPECTED_SOURCE_SHA256 = (
    "1dd89965cdd851bca72aa8aa6ffaca6c07f410c52"
    "f04c95c0cc446b4b75f2610"
)

# Independent reproducible random streams for each class.
rngs = {
    "Pathogenic": random.Random(SEED),
    "Benign": random.Random(SEED + 1),
}


def sha256(path):
    digest = hashlib.sha256()

    with path.open("rb") as handle:
        for block in iter(
            lambda: handle.read(1024 * 1024),
            b""
        ):
            digest.update(block)

    return digest.hexdigest()


def locus(row):
    return (
        str(row["chromosome"]),
        int(row["position_grch38_1based"]),
    )


def variant_key(row):
    return (
        str(row["chromosome"]),
        int(row["position_grch38_1based"]),
        row["ref"],
        row["alt"],
    )


print("STEP 18 — Held-out ClinVar selection")

if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

if not ORIGINAL.exists():
    raise FileNotFoundError(ORIGINAL)

if OUTPUT.exists() or METADATA.exists():
    raise RuntimeError(
        "Held-out output already exists. "
        "Do not overwrite a frozen validation cohort."
    )


# ==================================================
# 1. Verify the original VCF
# ==================================================

print("Verifying source VCF checksum...")

source_hash = sha256(SOURCE)

if source_hash != EXPECTED_SOURCE_SHA256:
    raise RuntimeError(
        "Source VCF checksum differs from the "
        "original research dataset."
    )

print("Source VCF checksum: PASS")


# ==================================================
# 2. Load previously used variants
# ==================================================

with ORIGINAL.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as handle:
    original_rows = list(csv.DictReader(handle))

assert len(original_rows) == 1000

original_keys = {
    variant_key(row)
    for row in original_rows
}

original_loci = {
    locus(row)
    for row in original_rows
}

assert len(original_keys) == 1000

print("Previously evaluated variants:", len(original_keys))
print("Previously evaluated loci:", len(original_loci))


# ==================================================
# 3. Scan full ClinVar VCF
# ==================================================

reservoirs = {
    label: []
    for label in CLASSES
}

eligible_seen = Counter()
counts = Counter()
metadata_lines = []

# Prevent duplicate genomic coordinates within
# the held-out cohort, even if alternate alleles differ.
seen_heldout_loci = set()

print("Scanning complete ClinVar VCF...")

with gzip.open(
    SOURCE,
    "rt",
    encoding="utf-8",
) as handle:

    for line in handle:
        if line.startswith("##"):
            if line.startswith((
                "##fileDate=",
                "##source=",
                "##reference=",
            )):
                metadata_lines.append(line.strip())

            continue

        if line.startswith("#"):
            continue

        counts["total_records"] += 1

        fields = line.rstrip("\n").split("\t")

        if len(fields) < 8:
            counts["malformed_records"] += 1
            continue

        chrom, pos, record_id, ref, alt = fields[:5]

        if chrom not in {
            str(i) for i in range(1, 23)
        }:
            counts["excluded_non_autosomal"] += 1
            continue

        if (
            len(ref) != 1
            or len(alt) != 1
            or ref not in "ACGT"
            or alt not in "ACGT"
            or ref == alt
        ):
            counts["excluded_non_snv"] += 1
            continue

        info = {}

        for item in fields[7].split(";"):
            if "=" in item:
                key, value = item.split("=", 1)
                info[key] = value

        label = info.get("CLNSIG", "")
        review = info.get("CLNREVSTAT", "")

        if label not in CLASSES:
            counts["excluded_classification"] += 1
            continue

        if review not in ALLOWED_REVIEW:
            counts["excluded_review_status"] += 1
            continue

        position = int(pos)
        coordinate = (chrom, position)

        if coordinate in original_loci:
            counts["excluded_original_locus"] += 1
            continue

        if coordinate in seen_heldout_loci:
            counts["excluded_duplicate_heldout_locus"] += 1
            continue

        seen_heldout_loci.add(coordinate)

        row = {
            "chromosome": chrom,
            "position_grch38_1based": position,
            "variant_id": record_id,
            "ref": ref,
            "alt": alt,
            "clinical_label": label,
            "review_status": review,
            "clinical_allele_id": info.get("ALLELEID", ""),
        }

        eligible_seen[label] += 1

        sample = reservoirs[label]
        rng = rngs[label]

        if len(sample) < MAX_PER_CLASS:
            sample.append(row)
        else:
            index = rng.randrange(
                eligible_seen[label]
            )

            if index < MAX_PER_CLASS:
                sample[index] = row

        if counts["total_records"] % 500000 == 0:
            print(
                "Records scanned:",
                counts["total_records"],
            )

print("Full GZIP stream: PASS")


# ==================================================
# 4. Freeze held-out cohort
# ==================================================

for label in CLASSES:
    if len(reservoirs[label]) != MAX_PER_CLASS:
        raise RuntimeError(
            f"Insufficient held-out {label} variants"
        )

selected = (
    reservoirs["Pathogenic"]
    + reservoirs["Benign"]
)

selected.sort(
    key=lambda row: (
        int(row["chromosome"]),
        int(row["position_grch38_1based"]),
        row["ref"],
        row["alt"],
    )
)

selected_keys = {
    variant_key(row)
    for row in selected
}

selected_loci = {
    locus(row)
    for row in selected
}

assert len(selected) == 1000
assert len(selected_keys) == 1000
assert len(selected_loci) == 1000

assert not (
    selected_keys & original_keys
)

assert not (
    selected_loci & original_loci
)

selected_counts = Counter(
    row["clinical_label"]
    for row in selected
)

assert selected_counts["Pathogenic"] == 500
assert selected_counts["Benign"] == 500


# ==================================================
# 5. Write frozen cohort
# ==================================================

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8",
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=FIELDS,
    )

    writer.writeheader()
    writer.writerows(selected)


# ==================================================
# 6. Save provenance and validation checks
# ==================================================

metadata = {
    "experiment": "15_prepare_heldout_clinvar",
    "research_step": 18,
    "purpose": (
        "Prespecified held-out variant-level "
        "validation cohort"
    ),
    "source": "NCBI ClinVar GRCh38 VCF",
    "source_filename": SOURCE.name,
    "source_sha256": source_hash,
    "vcf_metadata": metadata_lines,
    "full_gzip_stream_validation": "PASS",
    "seed": SEED,
    "sampling_method": (
        "Class-specific reservoir sampling "
        "with independent RNG streams"
    ),
    "maximum_per_class": MAX_PER_CLASS,
    "allowed_review_status": sorted(
        ALLOWED_REVIEW
    ),
    "eligible_heldout_counts": dict(
        eligible_seen
    ),
    "filter_counts": dict(counts),
    "selected_counts": dict(
        selected_counts
    ),
    "selected_total": len(selected),
    "previously_evaluated_variants": len(
        original_keys
    ),
    "previously_evaluated_loci": len(
        original_loci
    ),
    "overlapping_variants": len(
        selected_keys & original_keys
    ),
    "overlapping_loci": len(
        selected_loci & original_loci
    ),
    "reference_sequence_verified": False,
    "original_dataset_sha256": sha256(
        ORIGINAL
    ),
    "heldout_dataset_sha256": sha256(
        OUTPUT
    ),
    "script_sha256": sha256(
        Path(__file__)
    ),
    "validation_plan": {
        "primary_endpoint": (
            "Mean absolute score change "
            "across +/-6nt anchor shifts, "
            "single-phase versus six-phase mean"
        ),
        "primary_comparison": (
            "Paired per-variant instability "
            "reduction from phase averaging"
        ),
        "secondary_endpoints": [
            "Six-phase score range",
            "Phase sign disagreement",
            "Phase-baseline robustness",
            "Clinical AUROC, exploratory only",
        ],
        "analysis_unit": "Variant",
        "confidence_interval": (
            "95% paired, class-stratified bootstrap"
        ),
        "interpretation": (
            "Stability is the primary outcome. "
            "No clinical accuracy gain is assumed."
        ),
    },
    "limitations": [
        "Held-out variants originate from "
        "the same ClinVar release.",
        "Variant exclusion does not guarantee "
        "gene- or locus-region independence.",
        "Gene and ascertainment confounding "
        "may remain.",
        "GRCh38 reference alleles must be "
        "verified before scoring.",
        "Clinical classifications are not "
        "direct functional measurements.",
        "This is not independent-source "
        "validation.",
    ],
}

METADATA.write_text(
    json.dumps(
        metadata,
        indent=2,
    ),
    encoding="utf-8",
)


# ==================================================
# 7. Final report
# ==================================================

print("\nSTEP 18 — COHORT SELECTION COMPLETE")

print(
    "Eligible held-out pathogenic:",
    eligible_seen["Pathogenic"],
)

print(
    "Eligible held-out benign:",
    eligible_seen["Benign"],
)

print(
    "Selected pathogenic:",
    selected_counts["Pathogenic"],
)

print(
    "Selected benign:",
    selected_counts["Benign"],
)

print(
    "Overlap with original variants:",
    len(selected_keys & original_keys),
)

print(
    "Overlap with original loci:",
    len(selected_loci & original_loci),
)

print("Total held-out variants:", len(selected))
print("Dataset:", OUTPUT)
print("Metadata:", METADATA)
