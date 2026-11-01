PAPER 2 — REPRODUCIBILITY + HTML REPORT PACKAGE

1. Extract BOTH files into your existing project root:
   D:\research\Faizan\Paper-2\
   This folder should already contain .venv, experiments, models, data, results, figures.

2. Double-click RUN_PAPER2_REPORT.bat.
   This runs original stability, Experiment A, Experiment B, Experiment C,
   then the reproducibility audit, and generates an offline HTML report.

3. The report opens automatically when the batch file finishes:
   D:\research\Faizan\Paper-2\results\publication_report\index.html
   Each run is also archived in a timestamped folder with logs and SHA-256 manifest.

COMMAND LINE ALTERNATIVES (from project root):
  .venv\Scripts\python.exe reproduce_paper2.py
  .venv\Scripts\python.exe reproduce_paper2.py --verify-only
  .venv\Scripts\python.exe reproduce_paper2.py --report-only

IMPORTANT
- The FULL run needs existing local pretrained model, input data, dependencies,
  and a working CUDA setup. It may take significant time.
- This does not re-download ClinVar, retrieve new GRCh38 sequences, or redo
  original cohort selection; it reruns the core scoring experiments from
  existing processed data, then audits those artifacts.
- The audit and logs distinguish rerun success from previously stored results.
- GPU benchmarking varies across runs; exact latency need not be bitwise equal.
- Numerical stability is not equivalent to clinical pathogenicity accuracy.
- Experiment A is identified using its saved script SHA-256 when possible.
  If unavailable, a unique filename fallback is used and reported explicitly.
- No synthetic results are generated.
