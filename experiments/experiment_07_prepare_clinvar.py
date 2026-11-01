
import csv
import gzip
import hashlib
import json
import random
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "data" / "raw" / "clinvar_GRCh38.vcf.gz"
OUTPUT = ROOT / "data" / "processed"
OUTPUT.mkdir(parents=True, exist_ok=True)

SEED = 20261025
MAX_PER_CLASS = 500
rng = random.Random(SEED)

ALLOWED_REVIEW = {
    "criteria_provided,_multiple_submitters,_no_conflicts",
    "reviewed_by_expert_panel",
    "practice_guideline",
}

CLASSES = {"Pathogenic", "Benign"}

samples = {label: [] for label in CLASSES}
seen = Counter()
counts = Counter()
vcf_metadata = []

print("Step 10: ClinVar validation and SNV extraction")
print("Reading complete compressed VCF...")

with gzip.open(SOURCE, "rt", encoding="utf-8") as handle:
    for line in handle:
        if line.startswith("##"):
            if line.startswith((
                "##fileDate=",
                "##source=",
                "##reference="
            )):
                vcf_metadata.append(line.strip())
            continue

        if line.startswith("#"):
            continue

        counts["total_records"] += 1
        fields = line.rstrip("\n").split("\t")

        if len(fields) < 8:
            counts["malformed_records"] += 1
            continue

        chrom, pos, record_id, ref, alt = fields[:5]
        info_text = fields[7]

        if chrom not in {str(i) for i in range(1, 23)}:
            counts["excluded_non_autosomal"] += 1
            continue

        if (
            len(ref) != 1
            or len(alt) != 1
            or ref not in "ACGT"
            or alt not in "ACGT"
            or ref == alt
        ):
            counts["excluded_non_snv_or_multiallelic"] += 1
            continue

        info = {}

        for item in info_text.split(";"):
            if "=" in item:
                key, value = item.split("=", 1)
                info[key] = value

        label = info.get("CLNSIG", "")
        review = info.get("CLNREVSTAT", "")

        if label not in CLASSES:
            counts["excluded_other_classification"] += 1
            continue

        if review not in ALLOWED_REVIEW:
            counts["excluded_review_status"] += 1
            continue

        counts[f"eligible_{label}"] += 1

        row = {
            "chromosome": chrom,
            "position_grch38_1based": int(pos),
            "variant_id": record_id,
            "ref": ref,
            "alt": alt,
            "clinical_label": label,
            "review_status": review,
            "clinical_allele_id": info.get("ALLELEID", ""),
        }

        # Reservoir sampling gives each eligible record
        # an equal chance without loading the full VCF.
        seen[label] += 1
        reservoir = samples[label]

        if len(reservoir) < MAX_PER_CLASS:
            reservoir.append(row)
        else:
            index = rng.randrange(seen[label])
            if index < MAX_PER_CLASS:
                reservoir[index] = row

        if counts["total_records"] % 500000 == 0:
            print(
                "Records scanned:",
                counts["total_records"]
            )

# Reaching this point means gzip decompression
# completed, including its stream integrity checks.
print("Full GZIP stream: PASS")

selected = samples["Pathogenic"] + samples["Benign"]
selected.sort(
    key=lambda row: (
        int(row["chromosome"]),
        row["position_grch38_1based"],
        row["ref"],
        row["alt"]
    )
)

dataset_path = OUTPUT / "clinvar_high_confidence_snvs.csv"

with dataset_path.open(
    "w", newline="", encoding="utf-8"
) as handle:
    writer = csv.DictWriter(
        handle,
        fieldnames=[
            "chromosome",
            "position_grch38_1based",
            "variant_id",
            "ref",
            "alt",
            "clinical_label",
            "review_status",
            "clinical_allele_id"
        ]
    )
    writer.writeheader()
    writer.writerows(selected)

source_metadata_path = (
    ROOT / "data" / "raw"
    / "clinvar_GRCh38_download_metadata.json"
)

source_metadata = json.loads(
    source_metadata_path.read_text(encoding="utf-8")
)

metadata = {
    "experiment": "07_clinvar_dataset_preparation",
    "source": "NCBI ClinVar GRCh38 VCF",
    "source_filename": SOURCE.name,
    "source_sha256": source_metadata["sha256"],
    "full_gzip_stream_validation": "PASS",
    "vcf_metadata": vcf_metadata,
    "seed": SEED,
    "sampling_method": "reservoir_sampling_per_class",
    "maximum_records_per_class": MAX_PER_CLASS,
    "eligible_counts": dict(counts),
    "selected_counts": {
        label: len(samples[label])
        for label in sorted(CLASSES)
    },
    "selected_total": len(selected),
    "reference_alleles_verified_against_fasta": False,
    "limitations": [
        "GRCh38 reference alleles require FASTA verification",
        "Clinical classifications are not direct molecular effect measurements",
        "Benchmark may contain gene and ascertainment confounding",
        "Sampled variants require independent biological evaluation"
    ],
    "script_sha256": hashlib.sha256(
        Path(__file__).read_bytes()
    ).hexdigest(),
    "dataset_sha256": hashlib.sha256(
        dataset_path.read_bytes()
    ).hexdigest()
}

metadata_path = (
    OUTPUT / "clinvar_high_confidence_metadata.json"
)

metadata_path.write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8"
)

print("\nCLINVAR PREPARATION COMPLETE")
print("Total records scanned:", counts["total_records"])
print("Eligible pathogenic:", seen["Pathogenic"])
print("Eligible benign:", seen["Benign"])
print("Selected pathogenic:", len(samples["Pathogenic"]))
print("Selected benign:", len(samples["Benign"]))
print("Selected total:", len(selected))
print("Dataset:", dataset_path)
print("Metadata:", metadata_path)
