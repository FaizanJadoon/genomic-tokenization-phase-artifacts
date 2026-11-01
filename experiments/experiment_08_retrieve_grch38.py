
import csv
import hashlib
import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed"
    / "clinvar_high_confidence_snvs.csv"
)

OUTPUT = (
    ROOT / "data" / "processed"
    / "clinvar_grch38_sequences.csv"
)

METADATA = (
    ROOT / "data" / "processed"
    / "clinvar_grch38_sequences_metadata.json"
)

ENDPOINT = "https://rest.ensembl.org/sequence/region/human"

BATCH_SIZE = 25
FLANK_LEFT = 120
FLANK_RIGHT = 119

print("Step 11: GRCh38 sequence retrieval and verification")


def sha256(path):
    h = hashlib.sha256()

    with path.open("rb") as f:
        for block in iter(
            lambda: f.read(1024 * 1024),
            b""
        ):
            h.update(block)

    return h.hexdigest()


# --------------------------------------------------
# 1. Load selected ClinVar variants
# --------------------------------------------------

with INPUT.open(
    encoding="utf-8",
    newline=""
) as f:
    variants = list(csv.DictReader(f))

assert len(variants) == 1000

fields = list(variants[0].keys()) + [
    "region_start",
    "region_end",
    "reference_sequence_240nt",
    "reference_match",
    "ensembl_sequence_id"
]


def variant_key(row):
    return (
        row["chromosome"],
        row["position_grch38_1based"],
        row["ref"],
        row["alt"]
    )


# --------------------------------------------------
# 2. Resume previously verified sequences
# --------------------------------------------------

completed = {}

if OUTPUT.exists():
    with OUTPUT.open(
        encoding="utf-8",
        newline=""
    ) as f:
        for row in csv.DictReader(f):
            key = variant_key(row)

            assert row["reference_match"] == "PASS"
            assert len(row["reference_sequence_240nt"]) == 240
            assert (
                row["reference_sequence_240nt"][120]
                == row["ref"]
            )
            assert key not in completed

            completed[key] = row

print("Previously verified:", len(completed))

pending = [
    variant
    for variant in variants
    if variant_key(variant) not in completed
]

print("Remaining variants:", len(pending))


# --------------------------------------------------
# 3. Retrieve genomic regions with retries
# --------------------------------------------------

def retrieve_batch(batch):
    regions = []

    for variant in batch:
        pos = int(
            variant["position_grch38_1based"]
        )

        assert pos > FLANK_LEFT

        region = (
            f"{variant['chromosome']}:"
            f"{pos - FLANK_LEFT}.."
            f"{pos + FLANK_RIGHT}:1"
        )

        regions.append(region)

    # Deduplicate regions before requesting them.
    # Different variants can share the same position.
    regions = list(dict.fromkeys(regions))

    payload = json.dumps({
        "regions": regions
    }).encode("utf-8")

    for attempt in range(5):
        request = urllib.request.Request(
            ENDPOINT,
            data=payload,
            headers={
                "Content-Type": "application/json",
                "Accept": "application/json",
                "User-Agent": "Faizan-Paper2-Research/1.0"
            },
            method="POST"
        )

        try:
            with urllib.request.urlopen(
                request,
                timeout=90
            ) as response:
                return json.load(response)

        except (
            urllib.error.HTTPError,
            urllib.error.URLError,
            TimeoutError
        ) as error:

            if isinstance(
                error,
                urllib.error.HTTPError
            ):
                if error.code not in (
                    429, 500, 502, 503, 504
                ):
                    raise

            if attempt == 4:
                raise

            delay = 2 ** attempt + 1

            print(
                f"Retrying in {delay}s: {error}"
            )

            time.sleep(delay)

    raise RuntimeError(
        "Batch retrieval failed"
    )


# --------------------------------------------------
# 4. Validate Ensembl sequence identifiers
# --------------------------------------------------

id_pattern = re.compile(
    r"^chromosome:GRCh38:(\d+):(\d+):(\d+):1$"
)


# --------------------------------------------------
# 5. Initialize output if necessary
# --------------------------------------------------

if not OUTPUT.exists():
    with OUTPUT.open(
        "w",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )
        writer.writeheader()


# --------------------------------------------------
# 6. Download and verify remaining variants
# --------------------------------------------------

for index in range(
    0,
    len(pending),
    BATCH_SIZE
):
    batch = pending[
        index:index + BATCH_SIZE
    ]

    response = retrieve_batch(batch)

    returned = {}

    for item in response:
        match = id_pattern.fullmatch(
            item["id"]
        )

        if match is None:
            raise ValueError(
                "Unexpected assembly or sequence ID: "
                f"{item['id']}"
            )

        chrom, start, end = match.groups()

        region_key = (
            chrom,
            int(start),
            int(end)
        )

        # FIX: Accept repeated genomic regions
        # only when their sequences agree.
        if region_key in returned:
            previous_sequence = (
                returned[region_key]["seq"].upper()
            )

            current_sequence = (
                item["seq"].upper()
            )

            if previous_sequence != current_sequence:
                raise ValueError(
                    "Conflicting sequences for region: "
                    f"{region_key}"
                )

            continue

        returned[region_key] = item

    # Verify that every requested region was returned.
    expected_regions = {
        (
            variant["chromosome"],
            int(
                variant["position_grch38_1based"]
            ) - FLANK_LEFT,
            int(
                variant["position_grch38_1based"]
            ) + FLANK_RIGHT
        )
        for variant in batch
    }

    if set(returned) != expected_regions:
        missing = expected_regions - set(returned)
        unexpected = set(returned) - expected_regions

        raise ValueError(
            f"Region mismatch. "
            f"Missing: {missing}. "
            f"Unexpected: {unexpected}"
        )

    verified_rows = []

    for variant in batch:
        pos = int(
            variant["position_grch38_1based"]
        )

        region_key = (
            variant["chromosome"],
            pos - FLANK_LEFT,
            pos + FLANK_RIGHT
        )

        item = returned[region_key]

        sequence = item["seq"].upper()

        if len(sequence) != 240:
            raise ValueError(
                "Unexpected sequence length: "
                f"{item['id']}"
            )

        if sequence[120] != variant["ref"]:
            raise ValueError(
                f"Reference mismatch at "
                f"{variant['chromosome']}:{pos}. "
                f"ClinVar={variant['ref']}, "
                f"Genome={sequence[120]}"
            )

        verified_rows.append({
            **variant,
            "region_start": pos - FLANK_LEFT,
            "region_end": pos + FLANK_RIGHT,
            "reference_sequence_240nt": sequence,
            "reference_match": "PASS",
            "ensembl_sequence_id": item["id"]
        })

    # Save only fully verified batches.
    with OUTPUT.open(
        "a",
        newline="",
        encoding="utf-8"
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=fields
        )

        writer.writerows(verified_rows)

    completed_count = (
        len(completed)
        + index
        + len(batch)
    )

    print(
        f"Verified {completed_count}/"
        f"{len(variants)}"
    )

    time.sleep(1.2)


# --------------------------------------------------
# 7. Final dataset verification
# --------------------------------------------------

with OUTPUT.open(
    encoding="utf-8",
    newline=""
) as f:
    final_rows = list(csv.DictReader(f))

assert len(final_rows) == len(variants)

expected_keys = {
    variant_key(v)
    for v in variants
}

actual_keys = [
    variant_key(row)
    for row in final_rows
]

assert len(actual_keys) == len(set(actual_keys))
assert set(actual_keys) == expected_keys

assert all(
    row["reference_match"] == "PASS"
    and len(row["reference_sequence_240nt"]) == 240
    and row["reference_sequence_240nt"][120] == row["ref"]
    for row in final_rows
)


# --------------------------------------------------
# 8. Save reproducibility metadata
# --------------------------------------------------

metadata = {
    "experiment": "08_grch38_sequence_retrieval",
    "reference_assembly": "GRCh38",
    "source": ENDPOINT,
    "batch_size": BATCH_SIZE,
    "sequence_length": 240,
    "variant_offset_zero_based": 120,
    "total_variants": len(final_rows),
    "reference_matches": len(final_rows),
    "reference_mismatches": 0,
    "input_sha256": sha256(INPUT),
    "output_sha256": sha256(OUTPUT),
    "script_sha256": sha256(Path(__file__)),
    "limitations": [
        "Sequences obtained from Ensembl GRCh38",
        "Ensembl release is not pinned",
        "Clinical labels are not direct molecular effect measurements"
    ]
}

METADATA.write_text(
    json.dumps(
        metadata,
        indent=2
    ),
    encoding="utf-8"
)


# --------------------------------------------------
# 9. Final report
# --------------------------------------------------

print("\nSTEP 11 COMPLETE")
print("Variants verified:", len(final_rows))
print("Reference mismatches: 0")
print("Output:", OUTPUT)
print("Metadata:", METADATA)
