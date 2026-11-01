# Research datasets

## ClinVar

Source: NCBI ClinVar GRCh38 VCF
VCF file date: 2026-10-19
Local filename: data/raw/clinvar_GRCh38.vcf.gz

SHA-256:
1dd89965cdd851bca72aa8aa6ffaca6c07f410c52f04c95c0cc446b4b75f2610

Historical release must be used; the live ClinVar download
may contain a different dataset.

Original cohort:
500 benign and 500 pathogenic SNVs.
Seed: 20261025.

Held-out cohort:
500 benign and 500 pathogenic SNVs.
Seed: 20261026.

The cohorts have zero overlapping variants and genomic positions,
but originate from the same ClinVar release.

## Reference sequences

Genome assembly: GRCh38.
Original cohort: Ensembl REST API.
Held-out cohort: Ensembl REST API and UCSC Genome Browser REST API.

All 2,000 reference alleles were verified against retrieved sequences.
Reference service versions were not pinned.

Processed data and original source files remain local.
