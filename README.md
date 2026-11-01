# Tokenization-Phase Artifacts in Genomic Foundation Models

**Training-Free Correction of Single-Nucleotide Variant Effect Scores**

Research repository for investigating tokenization-phase artifacts
in genomic foundation models and evaluating six-phase score averaging.

## Research objective

Evaluate whether shifting tokenization phase changes SNV scores,
and whether averaging scores across six phases improves stability.

The primary evaluation concerns score stability, not clinical accuracy.

## Model

InstaDeepAI/nucleotide-transformer-v2-50m-multi-species

Pinned revision:
81b29e5786726d891dbf929404ef20adca5b36f1

## Datasets

NCBI ClinVar GRCh38, release dated 2026-10-19.

Original cohort: 1,000 variants.
Held-out cohort: 1,000 position-disjoint variants.

See `data/README.md` for provenance and checksums.

## Original environment

- Windows, Python 3.12.4
- PyTorch 2.9.1+cu128
- CUDA 12.8
- NVIDIA GeForce RTX 3080
- Transformers 4.46.3
- FP32 model inference

## Installation

Create a Python 3.12 virtual environment and install dependencies
from `requirements.txt`.

CUDA-enabled PyTorch may require installation using the appropriate
official PyTorch wheel index.

Download the pinned pretrained model:

    python download_model.py

Obtain the historical ClinVar VCF matching the checksum in
`data/README.md` and place it at:

    data/raw/clinvar_GRCh38.vcf.gz

## Reproduction

The repository contains 19 numbered experiment scripts.

The existing master entry point is:

    python reproduce_paper2.py

It reruns selected core scoring experiments and the reproducibility
audit using previously prepared datasets.

Other modes:

    python reproduce_paper2.py --verify-only
    python reproduce_paper2.py --report-only

IMPORTANT: The current master entry point does not recreate all
upstream datasets or execute every numbered experiment.

A full clean-machine reproduction has not yet been verified.

## Outputs

- `results/`: experimental results and analysis
- `figures/`: research figures
- `experiments/`: numbered experiment scripts
- `reproduce_paper2.py`: core rerun and report generator

## Limitations

- The held-out cohort comes from the same ClinVar release.
- Validation is position-disjoint, not gene-disjoint.
- Reference sequence service versions were not pinned.
- Numerical score stability is not proof of clinical utility.
- Runtime benchmarking depends on hardware and software.
- Exact independent reproduction remains to be tested.

## Reproducibility

The original dataset checksums, model revision, random seeds,
software environment and experiment scripts are documented.

This repository is intended to support independent research
verification. It does not claim that every experiment has already
been reproduced on a separate computer.
