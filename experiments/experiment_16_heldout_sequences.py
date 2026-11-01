
import csv
import hashlib
import json
import random
import time
from collections import Counter
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
import truststore

truststore.inject_into_ssl()


# ==================================================
# STEP 18 — MAIN EXPERIMENT B2
# Retrieve and validate held-out GRCh38 sequences
# ==================================================

ROOT = Path(__file__).resolve().parents[1]

INPUT = (
    ROOT / "data" / "processed"
    / "clinvar_heldout_snvs.csv"
)

OUTPUT = (
    ROOT / "data" / "processed"
    / "clinvar_heldout_grch38_sequences.csv"
)

METADATA = (
    ROOT / "data" / "processed"
    / "clinvar_heldout_sequences_metadata.json"
)

SEED = 20261026
VARIANT_INDEX = 120
SEQUENCE_LENGTH = 240
MAX_ATTEMPTS = 8
REQUEST_TIMEOUT = 45
REQUEST_PAUSE = 0.15

random.seed(SEED)

SOURCE_FIELDS = [
    "chromosome",
    "position_grch38_1based",
    "variant_id",
    "ref",
    "alt",
    "clinical_label",
    "review_status",
    "clinical_allele_id",
]

EXTRA_FIELDS = [
    "region_start",
    "region_end",
    "reference_sequence_240nt",
    "reference_match",
    "ensembl_sequence_id",
]

FIELDS = SOURCE_FIELDS + EXTRA_FIELDS


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
        str(row["chromosome"]),
        int(row["position_grch38_1based"]),
        row["ref"],
        row["alt"],
    )


def get_region(row):
    position = int(
        row["position_grch38_1based"]
    )

    start = position - VARIANT_INDEX
    end = start + SEQUENCE_LENGTH - 1

    if start < 1:
        raise ValueError(
            f"Invalid GRCh38 region for "
            f"{variant_key(row)}"
        )

    return start, end



def fetch_sequence(row):
    chrom = row["chromosome"]
    start, end = get_region(row)

    # UCSC uses zero-based, half-open coordinates.
    # Our stored coordinates are one-based, inclusive.
    ucsc_start = start - 1
    ucsc_end = end

    region = f"chr{chrom}:{ucsc_start}-{ucsc_end}"

    url = (
        "https://api.genome.ucsc.edu/getData/sequence"
        f"?genome=hg38;chrom=chr{chrom}"
        f";start={ucsc_start};end={ucsc_end}"
    )

    last_error = None

    for attempt in range(1, 5):
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "academic-genomics-validation/1.0",
            },
        )

        try:
            with urlopen(request, timeout=30) as response:
                payload = json.load(response)

            sequence = payload["dna"].upper()

            if len(sequence) != SEQUENCE_LENGTH:
                raise ValueError(
                    f"Unexpected sequence length for {region}: "
                    f"{len(sequence)}"
                )

            if set(sequence) - set("ACGT"):
                raise ValueError(
                    f"Non-ACGT bases in {region}"
                )

            observed_ref = sequence[VARIANT_INDEX]

            if observed_ref != row["ref"].upper():
                raise ValueError(
                    f"GRCh38 REF mismatch at {region}: "
                    f"expected {row['ref']}, "
                    f"observed {observed_ref}"
                )

            time.sleep(0.25)

            return {
                "region_start": start,
                "region_end": end,
                "reference_sequence_240nt": sequence,
                "reference_match": "PASS",
                "ensembl_sequence_id": (
                    f"UCSC:hg38:{region}"
                ),
            }

        except HTTPError as error:
            last_error = error

            if error.code not in (429, 500, 502, 503, 504):
                raise

        except (
            URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ) as error:
            last_error = error

        except (KeyError, ValueError):
            raise

        print(
            f"UCSC retry {attempt}/4 for {region}: "
            f"{last_error}"
        )

        if attempt < 4:
            time.sleep(min(2 ** attempt, 12))

    raise RuntimeError(
        f"UCSC retrieval failed for {region}"
    ) from last_error

    chrom = row["chromosome"]
    start, end = get_region(row)

    region = f"{chrom}:{start}..{end}:1"

    url = (
        "https://rest.ensembl.org/"
        f"sequence/region/human/{region}"
        "?coord_system_version=GRCh38"
    )

    last_error = None

    for attempt in range(1, MAX_ATTEMPTS + 1):
        request = Request(
            url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
                "User-Agent": (
                    "academic-genomics-validation/1.0"
                ),
            },
        )

        try:
            with urlopen(
                request,
                timeout=REQUEST_TIMEOUT,
            ) as response:
                payload = json.load(response)

            sequence = payload["seq"].upper()

            if len(sequence) != SEQUENCE_LENGTH:
                raise ValueError(
                    "Unexpected reference sequence length: "
                    f"{len(sequence)} for {region}"
                )

            if any(
                base not in "ACGT"
                for base in sequence
            ):
                raise ValueError(
                    "Reference sequence contains "
                    f"non-ACGT bases: {region}"
                )

            expected_ref = row["ref"].upper()

            observed_ref = sequence[VARIANT_INDEX]

            if observed_ref != expected_ref:
                raise ValueError(
                    "GRCh38 reference mismatch: "
                    f"{variant_key(row)}; "
                    f"observed {observed_ref}"
                )

            time.sleep(REQUEST_PAUSE)

            return {
                "region_start": start,
                "region_end": end,
                "reference_sequence_240nt": sequence,
                "reference_match": "PASS",
                "ensembl_sequence_id": payload.get(
                    "id", region
                ),
            }

        except HTTPError as error:
            last_error = error

            if error.code not in (
                429, 500, 502, 503, 504
            ):
                raise RuntimeError(
                    f"Non-retryable HTTP {error.code} "
                    f"for {region}"
                ) from error

            retry_after = error.headers.get(
                "Retry-After"
            )

            try:
                delay = float(retry_after)
            except (TypeError, ValueError):
                delay = min(
                    2 ** attempt,
                    45
                )

            delay = max(
                0.5,
                min(delay, 60)
            )

        except (
            URLError,
            TimeoutError,
            ConnectionError,
            json.JSONDecodeError,
        ) as error:
            last_error = error
            delay = min(
                2 ** attempt,
                45
            )

        except (KeyError, ValueError):
            # Data-integrity problems must not be
            # silently retried or skipped.
            raise

        print(
            f"Retry {attempt}/{MAX_ATTEMPTS} "
            f"for {region}: {last_error}"
        )

        if attempt < MAX_ATTEMPTS:
            time.sleep(delay)

    raise RuntimeError(
        f"Reference retrieval failed for {region} "
        f"after {MAX_ATTEMPTS} attempts"
    ) from last_error


# ==================================================
# 1. Load frozen held-out cohort
# ==================================================

print("STEP 18 — Held-out GRCh38 sequence retrieval")

with INPUT.open(
    "r",
    newline="",
    encoding="utf-8-sig",
) as handle:
    reader = csv.DictReader(handle)

    if reader.fieldnames != SOURCE_FIELDS:
        raise RuntimeError(
            "Unexpected held-out cohort columns"
        )

    variants = list(reader)

assert len(variants) == 1000

keys = [
    variant_key(row)
    for row in variants
]

assert len(set(keys)) == 1000

loci = {
    (
        row["chromosome"],
        int(row["position_grch38_1based"]),
    )
    for row in variants
}

assert len(loci) == 1000

for row in variants:
    assert row["ref"] in "ACGT"
    assert row["alt"] in "ACGT"
    assert row["ref"] != row["alt"]
    assert row["clinical_label"] in (
        "Benign",
        "Pathogenic",
    )

print("Held-out variants:", len(variants))


# ==================================================
# 2. Resume previously verified sequences
# ==================================================

completed = {}

if OUTPUT.exists():
    with OUTPUT.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as handle:
        reader = csv.DictReader(handle)

        if reader.fieldnames != FIELDS:
            raise RuntimeError(
                "Existing output has unexpected columns"
            )

        for row in reader:
            key = variant_key(row)

            if key in completed:
                raise RuntimeError(
                    f"Duplicate completed variant: {key}"
                )

            if key not in set(keys):
                raise RuntimeError(
                    f"Unexpected variant in output: {key}"
                )

            sequence = (
                row["reference_sequence_240nt"]
                .upper()
            )

            assert len(sequence) == SEQUENCE_LENGTH
            assert row["reference_match"] == "PASS"
            assert sequence[VARIANT_INDEX] == row["ref"]
            assert set(sequence) <= set("ACGT")

            start, end = get_region(row)

            assert int(row["region_start"]) == start
            assert int(row["region_end"]) == end

            completed[key] = row

pending = [
    row
    for row in variants
    if variant_key(row) not in completed
]

print("Previously completed:", len(completed))
print("Remaining:", len(pending))


# ==================================================
# 3. Create output only if needed
# ==================================================

if not OUTPUT.exists():
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


# ==================================================
# 4. Retrieve, verify and save incrementally
# ==================================================

for index, row in enumerate(pending, start=1):
    region_data = fetch_sequence(row)

    output_row = dict(row)
    output_row.update(region_data)

    with OUTPUT.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=FIELDS,
        )
        writer.writerow(output_row)

    total = len(completed) + index

    if (
        total % 25 == 0
        or total == len(variants)
    ):
        print(
            f"Verified {total}/"
            f"{len(variants)} variants"
        )


# ==================================================
# 5. Validate the completed output
# ==================================================

with OUTPUT.open(
    "r",
    newline="",
    encoding="utf-8",
) as handle:
    reader = csv.DictReader(handle)

    assert reader.fieldnames == FIELDS

    results = list(reader)

assert len(results) == 1000

result_keys = [
    variant_key(row)
    for row in results
]

assert len(set(result_keys)) == 1000
assert set(result_keys) == set(keys)

for row in results:
    sequence = row["reference_sequence_240nt"]

    assert len(sequence) == 240
    assert set(sequence) <= set("ACGT")
    assert row["reference_match"] == "PASS"
    assert sequence[120] == row["ref"]

    start, end = get_region(row)

    assert int(row["region_start"]) == start
    assert int(row["region_end"]) == end

    assert end - start + 1 == 240

counts = Counter(
    row["clinical_label"]
    for row in results
)

assert counts["Benign"] == 500
assert counts["Pathogenic"] == 500


# ==================================================
# 6. Save provenance and validation metadata
# ==================================================

metadata = {
    "experiment": "16_heldout_sequences",
    "research_step": 18,
    "reference_assembly": "GRCh38",
    "reference_service": "UCSC Genome Browser REST API",
    "endpoint": (
        "https://api.genome.ucsc.edu/getData/sequence"
    ),
    "coordinate_system_version": "GRCh38",
    "sequence_length": SEQUENCE_LENGTH,
    "variant_index_zero_based": VARIANT_INDEX,
    "variants": len(results),
    "verified_reference_matches": len(results),
    "reference_mismatches": 0,
    "non_acgt_sequences": 0,
    "clinical_counts": dict(counts),
    "seed": SEED,
    "input_sha256": sha256(INPUT),
    "output_sha256": sha256(OUTPUT),
    "script_sha256": sha256(Path(__file__)),
    "verification": (
        "Reference base at zero-based index 120 "
        "matches the ClinVar REF allele"
    ),
    "limitations": [
        "Reference sequences were obtained "
        "from Ensembl REST, not a locally "
        "downloaded GRCh38 FASTA.",
        "Held-out variants originate from "
        "the same ClinVar release as the "
        "original benchmark.",
        "Sequence verification does not "
        "establish functional pathogenicity.",
        "The source is a live API; reference "
        "service software may change over time.",
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

print("\nSTEP 18 — SEQUENCES VERIFIED")

print("Variants:", len(results))
print("Benign:", counts["Benign"])
print("Pathogenic:", counts["Pathogenic"])
print("Reference matches:", len(results))
print("Reference mismatches:", 0)
print("Non-ACGT sequences:", 0)
print("Dataset:", OUTPUT)
print("Metadata:", METADATA)
