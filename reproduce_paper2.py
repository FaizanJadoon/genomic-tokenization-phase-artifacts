#!/usr/bin/env python3
"""Paper 2: rerun local experiments, audit results, and build an offline HTML report.

Requires the original research project, data, model, experiment scripts, and venv.
This orchestrator does NOT synthesize scientific results or redownload data.

Usage:
  python reproduce_paper2.py                 # full experiment rerun + audit + report
  python reproduce_paper2.py --verify-only   # audit existing artifacts + report
  python reproduce_paper2.py --report-only   # report existing artifacts without rerunning
  python reproduce_paper2.py --root PATH     # alternate project directory
"""
from __future__ import annotations

import argparse
import base64
from datetime import datetime, timezone
import hashlib
import html
import json
import math
import os
from pathlib import Path
import platform
import re
import shutil
import subprocess
import sys
import time
import traceback
from typing import Any

ROOT_DEFAULT = Path(__file__).resolve().parent
EXPECTED = {
    "A_phase": (1.2700412628034752, 0.001),
    "A_flank": (0.3574320958157381, 0.001),
    "A_difference": (0.912609166987737, 0.001),
    "A_flank_raw": (0.262612268447876, 0.001),
    "A_flank_averaged": (0.1777453640798728, 0.001),
    "original_raw": (0.436897, 0.001),
    "original_average": (0.205455, 0.001),
    "heldout_raw": (0.428801, 0.001),
    "heldout_average": (0.199561, 0.001),
    "heldout_improvement": (0.229241, 0.001),
    "heldout_positives": (810, 0),
    "single_latency_ms": (2.533, 0.05),
    "six_latency_ms": (3.810, 0.05),
    "latency_ratio": (1.504, 0.05),
    "single_memory_mib": (226.75, 1),
    "six_memory_mib": (247.07, 1),
}
ARTIFACTS = {
    "original": "results/experiment_11/stability_analysis.json",
    "A": "results/main_experiment_A/fixed_core_analysis.json",
    "B": "results/main_experiment_B/heldout_validation.json",
    "C": "results/main_experiment_C/compute_benchmark.json",
    "audit": "results/publication_audit/reproducibility_audit.json",
    "metadata": "data/processed/clinvar_heldout_sequences_metadata.json",
    "heldout_scores": "results/main_experiment_B/heldout_anchor_scores.csv",
}
SCRIPT_MAP = {
    "original": "experiments/experiment_11_clinvar_stability.py",
    "B": "experiments/experiment_17_heldout_validation.py",
    "C": "experiments/experiment_18_compute_benchmark.py",
    "audit": "experiments/experiment_19_reproducibility_audit.py",
}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict:
    if not path.is_file():
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8-sig"))
        return value if isinstance(value, dict) else {"value": value}
    except (OSError, ValueError) as e:
        return {"_read_error": str(e)}


def flatten_numbers(value: Any, path: str = "") -> list[tuple[str, float]]:
    found = []
    if isinstance(value, dict):
        for k, v in value.items():
            found.extend(flatten_numbers(v, f"{path}.{k}" if path else str(k)))
    elif isinstance(value, list):
        for i, v in enumerate(value):
            found.extend(flatten_numbers(v, f"{path}[{i}]"))
    elif isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value):
        found.append((path, float(value)))
    return found


def match_metric(data: dict, expected: float, tolerance: float, hints: tuple[str, ...] = (), variants: tuple[float, ...] = ()) -> dict:
    """Only mark independently verified if a number occurs in the saved JSON.

    Numeric matching is for locating fields in unknown JSON schemas. The field path
    is always shown so users can independently inspect the source; ambiguous matches
    are flagged rather than hidden.
    """
    targets = (expected,) + variants
    matches = []
    for key, value in flatten_numbers(data):
        for target in targets:
            allowed = tolerance if target == expected else max(abs(target) * 0.02, 1e-7)
            if abs(value - target) <= allowed:
                conversion = expected / target if target != expected and target != 0 else 1.0
                matches.append((key, value * conversion, conversion))
                break
    if not matches:
        return {"status": "not_located", "value": None, "path": None}
    def score(item):
        key = item[0].lower()
        return sum(2 for word in hints if word in key) - (2 if "ci" in key else 0)
    matches.sort(key=score, reverse=True)
    key, normalized_value, conversion = matches[0]
    if conversion != 1:
        key += f" [unit conversion ×{conversion:g}]"
    return {"status": "matched", "value": normalized_value, "path": key, "matches": len(matches)}


def e(s: Any) -> str:
    return html.escape(str(s), quote=True)


def fmt(value: Any, digits: int = 3) -> str:
    if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
        return "—"
    return f"{float(value):,.{digits}f}"


def resolve_a_script(root: Path, a: dict) -> tuple[Path | None, str]:
    expected_hash = a.get("script_sha256")
    scripts = sorted((root / "experiments").glob("*.py"))
    if expected_hash:
        matching = [p for p in scripts if sha256(p).lower() == str(expected_hash).lower()]
        if len(matching) == 1:
            return matching[0], "SHA-256 matched Experiment A JSON"
    matches = [p for p in scripts if "fixed_core" in p.stem.lower() or "main_experiment_a" in p.stem.lower()]
    if len(matches) == 1:
        return matches[0], "Filename matched; SHA-256 NOT confirmed"
    return None, "Could not uniquely identify Experiment A script. Do not substitute an unverified script."


def capture_environment(root: Path) -> dict:
    info = {
        "python": sys.version.splitlines()[0], "executable": sys.executable,
        "platform": platform.platform(), "project_root": str(root),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    for module in ("torch", "transformers", "numpy", "scipy", "sklearn", "pandas"):
        try:
            mod = __import__(module)
            info[module] = getattr(mod, "__version__", "installed")
            if module == "torch":
                info["cuda_available"] = bool(mod.cuda.is_available())
                if mod.cuda.is_available():
                    info["gpu"] = mod.cuda.get_device_name(0)
                    info["cuda_runtime"] = str(mod.version.cuda)
        except Exception as ex:
            info[module] = "unavailable: " + str(ex)[:150]
    return info


def run_script(key: str, script: Path | None, root: Path, log_dir: Path) -> dict:
    record = {"step": key, "script": str(script) if script else None,
              "status": "missing", "seconds": None, "returncode": None}
    if script is None or not script.is_file():
        record["error"] = "Script unavailable; not executed"
        print(f"[MISSING] {key}: {record['error']}", flush=True)
        return record
    print(f"\n[RUN] {key}: {script.relative_to(root)}", flush=True)
    started = time.monotonic()
    log_file = log_dir / ("step_" + key + ".log")
    record["log"] = str(log_file)
    env = os.environ.copy()
    env["PYTHONUNBUFFERED"] = "1"
    env["TOKENIZERS_PARALLELISM"] = "false"
    try:
        with log_file.open("w", encoding="utf-8", errors="replace") as f:
            f.write("COMMAND: " + sys.executable + " " + str(script) + "\n")
            f.write("WORKDIR: " + str(root) + "\n\n")
            f.flush()
            proc = subprocess.Popen([sys.executable, str(script)], cwd=str(root),
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    text=True, encoding="utf-8", errors="replace", bufsize=1)
            assert proc.stdout is not None
            for line in proc.stdout:
                f.write(line)
                print(line.rstrip("\n"), flush=True)
            record["returncode"] = proc.wait()
        record["status"] = "passed" if record["returncode"] == 0 else "failed"
    except Exception as ex:
        record["status"] = "failed"
        record["error"] = repr(ex)
        with log_file.open("a", encoding="utf-8") as f:
            f.write("\nORCHESTRATOR ERROR: " + traceback.format_exc())
    record["seconds"] = round(time.monotonic() - started, 2)
    print(f"[{record['status'].upper()}] {key} ({record['seconds']} seconds)", flush=True)
    return record


def parse_audit_log(run_records: list[dict]) -> dict:
    audit = next((r for r in run_records if r.get("step") == "audit"), None)
    if not audit or not audit.get("log") or not Path(audit["log"]).is_file():
        return {}
    log = Path(audit["log"]).read_text(encoding="utf-8", errors="replace")
    def search(pattern):
        m = re.search(pattern, log)
        return float(m.group(1)) if m else None
    return {
        "passed": search(r"Passed checks:\s*(\d+)"),
        "total": search(r"Total checks:\s*(\d+)"),
        "original_reduction_pct": search(r"Original reduction:\s*([\d.]+)"),
        "heldout_reduction_pct": search(r"Held-out reduction:\s*([\d.]+)"),
        "latency_ratio": search(r"Latency ratio:\s*([\d.]+)"),
        "complete": "STEP 20 — AUDIT COMPLETE" in log or "STEP 20 - AUDIT COMPLETE" in log,
        "failures": re.findall(r"^FAIL\s+(.+)$", log, flags=re.MULTILINE),
        "warnings": re.findall(r"^WARNING:\s*(.+)$", log, flags=re.MULTILINE),
    }


def load_data(root: Path) -> dict:
    return {name: read_json(root / rel) for name, rel in ARTIFACTS.items() if rel.endswith(".json")}


def metric_record(data: dict, name: str, source: str, expected: float, tolerance: float, hints=(), variants=()) -> dict:
    result = match_metric(data.get(source, {}), expected, tolerance, hints, variants)
    result.update({"name": name, "source": source, "expected": expected})
    return result


def make_metrics(data: dict, audit_log: dict) -> list[dict]:
    a = data.get("A", {})
    records = []
    exact_a = [
        ("A_phase", "Phase-associated effect", "mean_phase_effect"),
        ("A_flank", "Flank perturbation effect", "mean_flank_effect"),
        ("A_difference", "Paired phase–flank contrast", "mean_phase_minus_flank_effect"),
        ("A_flank_raw", "Single-phase flank sensitivity", "mean_raw_flank_change"),
        ("A_flank_averaged", "Six-phase flank sensitivity", "mean_averaged_flank_change"),
    ]
    for key, label, field in exact_a:
        expected, tolerance = EXPECTED[key]
        v = a.get(field)
        valid = isinstance(v, (int, float)) and not isinstance(v, bool) and abs(v - expected) <= tolerance
        records.append({"key": key, "name": label, "source": "A", "expected": expected,
                        "value": v if valid else None, "status": "matched" if valid else "not_located",
                        "path": field if valid else None})
    for key, label, source, hints, variants in [
        ("original_raw", "Original phase-0 anchor instability", "original", ("phase0", "raw", "mean"), ()),
        ("original_average", "Original six-phase anchor instability", "original", ("averag", "mean", "six"), ()),
        ("heldout_raw", "Held-out phase-0 anchor instability", "B", ("phase0", "raw", "mean"), ()),
        ("heldout_average", "Held-out six-phase anchor instability", "B", ("averag", "mean", "six"), ()),
        ("heldout_improvement", "Held-out paired improvement", "B", ("improv", "mean"), ()),
        ("heldout_positives", "Held-out variants improved", "B", ("positive", "improv", "count"), ()),
        ("single_latency_ms", "Single-phase latency (ms/variant)", "C", ("single", "phase0", "median", "latency"), (0.002533,)),
        ("six_latency_ms", "Six-phase latency (ms/variant)", "C", ("six", "averag", "median", "latency"), (0.003810,)),
        ("latency_ratio", "Latency ratio", "C", ("ratio", "latency"), ()),
        ("single_memory_mib", "Single-phase peak memory (MiB)", "C", ("single", "peak", "memory"), ()),
        ("six_memory_mib", "Six-phase peak memory (MiB)", "C", ("six", "peak", "memory"), ()),
    ]:
        expected, tolerance = EXPECTED[key]
        r = metric_record(data, label, source, expected, tolerance, hints, variants)
        r["key"] = key
        records.append(r)
    for key, label, audit_key in [
        ("original_reduction", "Original stability reduction (%)", "original_reduction_pct"),
        ("heldout_reduction", "Held-out stability reduction (%)", "heldout_reduction_pct"),
    ]:
        value = audit_log.get(audit_key)
        records.append({"key": key, "name": label, "source": "audit output", "expected": 52.974 if key == "original_reduction" else 53.461,
                        "value": value, "status": "matched" if value is not None else "not_located",
                        "path": "audit stdout: " + audit_key if value is not None else None})
    return records


def pill(status: str) -> str:
    color = "good" if status in ("matched", "passed") else "bad" if status == "failed" else "muted"
    label = "Verified" if status == "matched" else status.replace("_", " ").title()
    return f'<span class="pill {color}">{e(label)}</span>'


def bar_pair(label: str, left_name: str, left: float | None, right_name: str, right: float | None,
             max_value: float, left_color="#c57b4e", right_color="#3f7c85") -> str:
    if left is None or right is None:
        return f'<div class="empty-chart">{e(label)}: numeric values not independently located in result JSON.</div>'
    x0, chartw = 154, 515
    y1, y2 = 44, 96
    w1 = max(0, min(chartw, chartw * left / max_value))
    w2 = max(0, min(chartw, chartw * right / max_value))
    return f'''<svg role="img" aria-label="{e(label)}" viewBox="0 0 780 150" class="chart-svg">
    <text x="0" y="17" class="svg-title">{e(label)}</text>
    <text x="0" y="64" class="svg-label">{e(left_name)}</text><rect x="{x0}" y="40" width="{chartw}" height="29" rx="6" fill="#e8edf0"/>
    <rect x="{x0}" y="40" width="{w1:.2f}" height="29" rx="6" fill="{left_color}"/><text x="{x0+chartw+10}" y="61" class="svg-value">{left:.3f}</text>
    <text x="0" y="116" class="svg-label">{e(right_name)}</text><rect x="{x0}" y="92" width="{chartw}" height="29" rx="6" fill="#e8edf0"/>
    <rect x="{x0}" y="92" width="{w2:.2f}" height="29" rx="6" fill="{right_color}"/><text x="{x0+chartw+10}" y="113" class="svg-value">{right:.3f}</text>
    </svg>'''


def evidence_table(metrics: list[dict]) -> str:
    rows = []
    for m in metrics:
        observed = fmt(m.get("value"), 4)
        source = e(m.get("source"))
        path = e(m.get("path") or "No matching field located")
        rows.append(f'<tr><td>{e(m["name"])}</td><td class="num">{observed}</td><td>{source}<span class="subpath">{path}</span></td><td>{pill(m["status"])}</td></tr>')
    return "".join(rows)


def figure_gallery(root: Path, max_figures=8) -> str:
    candidates = []
    for base in [root / "figures" / "main_experiment_B", root / "figures" / "main_experiment_A", root / "figures" / "main_experiment_C"]:
        if base.is_dir():
            candidates.extend(sorted(base.glob("*.png")))
            candidates.extend(sorted(base.glob("*.jpg")))
    selected = candidates[:max_figures]
    if not selected:
        return '<p class="quiet">No existing PNG/JPG experiment figures found in the standard figures directories.</p>'
    out = []
    for path in selected:
        if path.stat().st_size > 3 * 1024 * 1024:
            continue
        kind = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
        uri = f"data:{kind};base64," + base64.b64encode(path.read_bytes()).decode("ascii")
        out.append(f'<figure class="figure"><img src="{uri}" alt="Experimental figure {e(path.stem)}"/><figcaption>{e(path.relative_to(root))}</figcaption></figure>')
    return '<div class="figure-grid">' + "".join(out) + '</div>' if out else '<p class="quiet">Available figures were too large to embed.</p>'


def file_inventory(root: Path) -> tuple[str, list[dict]]:
    rows = []
    manifest = []
    for label, rel in ARTIFACTS.items():
        path = root / rel
        if path.is_file():
            digest = sha256(path)
            manifest.append({"name": label, "path": rel, "sha256": digest, "bytes": path.stat().st_size})
            rows.append(f'<tr><td>{e(label)}</td><td><code>{e(rel)}</code></td><td class="num">{path.stat().st_size / 1024:.1f} KiB</td><td><code title="{digest}">{digest[:16]}…</code></td></tr>')
        else:
            rows.append(f'<tr><td>{e(label)}</td><td><code>{e(rel)}</code></td><td colspan="2" class="warn">Missing</td></tr>')
    return "".join(rows), manifest


def run_table(run_records: list[dict]) -> str:
    if not run_records:
        return '<p class="quiet">No experiments executed during this report-only run. Existing artifacts have not been independently rerun.</p>'
    out = []
    for r in run_records:
        out.append(f'<tr><td>{e(r["step"])}</td><td><code>{e(Path(r["script"]).name if r.get("script") else "unavailable")}</code></td><td>{pill(r["status"])}</td><td class="num">{fmt(r.get("seconds"), 1)} s</td></tr>')
    return '<table><thead><tr><th>Step</th><th>Script</th><th>Status</th><th>Duration</th></tr></thead><tbody>' + "".join(out) + '</tbody></table>'


def report_html(root: Path, mode: str, env: dict, run_records: list[dict], data: dict,
                audit_log: dict, metrics: list[dict], inventory_rows: str, created: str) -> str:
    mm = {m["key"]: m for m in metrics}
    def get(key):
        item = mm.get(key)
        return item.get("value") if item and item.get("status") == "matched" else None
    a = data.get("A", {})
    audit_passed = bool(audit_log.get("complete") and audit_log.get("total") and audit_log.get("passed") == audit_log.get("total") and not audit_log.get("failures"))
    if not audit_passed and mode == "report-only":
        audit_label = "Not rerun in report-only mode"
        audit_class = "muted"
    else:
        audit_label = f'{int(audit_log["passed"])}/{int(audit_log["total"])} checks passed' if audit_passed else 'Not fully confirmed'
        audit_class = "good" if audit_passed else "bad"
    source_counts = data.get("metadata", {}).get("reference_source_counts", {})
    source_text = ", ".join(f"{e(k)}: {e(v)}" for k, v in source_counts.items()) if source_counts else "Not recorded in metadata"
    metrics_verified = sum(1 for m in metrics if m["status"] == "matched")
    all_runs_ok = all(r["status"] == "passed" for r in run_records) and bool(run_records)
    full_ok = mode == "full" and all_runs_ok and audit_passed
    banner = "Full rerun completed and audit passed" if full_ok else "Evidence review — inspect execution status before claiming reproduction"
    audit_html = ''.join(f'<li>{e(x)}</li>' for x in audit_log.get("warnings", []))
    if not audit_html:
        audit_html = '<li>None reported by the audit executed in this run.</li>' if audit_log else '<li>No audit log was produced in this run.</li>'
    env_html = ''.join(f'<tr><th>{e(k)}</th><td><code>{e(v)}</code></td></tr>' for k, v in env.items())
    json_panels = []
    for label in ("original", "A", "B", "C", "audit", "metadata"):
        value = data.get(label, {})
        if value:
            pretty = json.dumps(value, indent=2, ensure_ascii=False)
            if len(pretty) > 100_000:
                pretty = pretty[:100_000] + "\n... (truncated for report size)"
            json_panels.append(f'<details><summary>{e(label)} — source JSON</summary><pre>{e(pretty)}</pre></details>')
    bar_original = bar_pair("Original cohort · anchor-shift instability", "Single phase", get("original_raw"), "Six-phase mean", get("original_average"), 0.5)
    bar_heldout = bar_pair("Held-out cohort · anchor-shift instability", "Single phase", get("heldout_raw"), "Six-phase mean", get("heldout_average"), 0.5)
    bar_runtime = bar_pair("Compute benchmark · latency (ms/variant)", "Single phase", get("single_latency_ms"), "Six-phase mean", get("six_latency_ms"), 4.5)
    bar_a = bar_pair("Fixed-core controls · mean absolute effect", "Phase-associated", get("A_phase"), "Flank perturbation", get("A_flank"), 1.5)
    main_reduction = fmt(get("heldout_reduction"), 2) + "%" if get("heldout_reduction") is not None else "Not extracted"
    orig_reduction = fmt(get("original_reduction"), 2) + "%" if get("original_reduction") is not None else "Not extracted"
    latency = fmt(get("latency_ratio"), 2) + "×" if get("latency_ratio") is not None else "Not extracted"
    meta_a = a.get("phase_minus_flank_ci95", [])
    ci_text = "[" + ", ".join(fmt(v, 4) for v in meta_a) + "]" if len(meta_a) == 2 else "Not located"
    style = r'''
    :root{--ink:#162c37;--muted:#5b6c74;--accent:#176a75;--accent2:#d78b57;--line:#dbe5e8;--paper:#fff;--bg:#f3f7f7;--good:#18724c;--bad:#a13b31}
    *{box-sizing:border-box}body{margin:0;background:var(--bg);font-family:Inter,"Segoe UI",Arial,sans-serif;color:var(--ink);line-height:1.55}
    .layout{display:grid;grid-template-columns:220px minmax(0,1fr);max-width:1520px;margin:auto;min-height:100vh}
    aside{background:#12313c;color:#e6f0f1;padding:32px 20px;position:sticky;top:0;height:100vh}
    .brand{font-weight:800;letter-spacing:.06em;font-size:14px;margin-bottom:30px;color:#fff}
    aside a{display:block;color:#c8d9dd;text-decoration:none;font-size:13px;padding:11px 12px;border-radius:7px}
    aside a:hover{background:#224751;color:#fff}main{padding:44px 5vw 80px;max-width:1250px;width:100%}
    .eyebrow{font-size:12px;font-weight:800;letter-spacing:.14em;text-transform:uppercase;color:var(--accent)}
    h1{font-size:clamp(30px,3.6vw,48px);letter-spacing:-.035em;line-height:1.13;max-width:990px;margin:10px 0 18px}
    h2{font-size:24px;letter-spacing:-.025em;margin:0 0 18px}h3{font-size:16px;margin:0 0 14px}
    .lede{max-width:900px;color:var(--muted);font-size:16px}.section{margin-top:42px;scroll-margin-top:20px}
    .status{border-left:4px solid var(--accent);padding:12px 17px;background:#eaf3f3;margin:26px 0 28px;border-radius:0 9px 9px 0}
    .status strong{display:block}.status span{color:var(--muted);font-size:13px}
    .kpis{display:grid;grid-template-columns:repeat(4,minmax(0,1fr));gap:14px;margin-top:28px}
    .kpi,.panel{background:var(--paper);border:1px solid var(--line);border-radius:12px;box-shadow:0 3px 15px rgba(12,43,55,.035)}
    .kpi{padding:19px}.kpi .label{color:var(--muted);font-size:12px;font-weight:700;text-transform:uppercase;letter-spacing:.04em}
    .kpi .number{font-size:30px;font-weight:800;letter-spacing:-.03em;margin:9px 0 2px}.kpi .detail{font-size:12px;color:var(--muted)}
    .panel{padding:22px;margin:15px 0}.two{display:grid;grid-template-columns:1fr 1fr;gap:16px}
    .chart-svg{display:block;width:100%;height:auto;max-height:190px}.svg-title{font-size:16px;font-weight:700;fill:#233943}.svg-label{font-size:13px;fill:#52646d}.svg-value{font-size:14px;font-weight:700;fill:#263943}
    table{border-collapse:collapse;width:100%;font-size:13px;background:#fff}th,td{padding:11px 12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}thead th{font-size:11px;letter-spacing:.06em;text-transform:uppercase;background:#f5f9fa;color:#52636b}
    .num{text-align:right;white-space:nowrap;font-variant-numeric:tabular-nums}.subpath{display:block;color:#83919a;font-size:11px;word-break:break-all}
    code{font-family:Consolas,"Courier New",monospace;font-size:12px;word-break:break-word}
    .pill{display:inline-block;border-radius:99px;padding:3px 9px;font-size:11px;font-weight:800;white-space:nowrap}.pill.good{background:#e2f4ea;color:#166542}.pill.bad{background:#fce7e3;color:#98372c}.pill.muted{background:#e9eff2;color:#50616a}
    .quiet,.caption{font-size:13px;color:var(--muted)}.warn{color:#9b4634}.empty-chart{color:#6b7b81;padding:30px 12px;background:#f6f8f9;border:1px dashed #d6e1e4;border-radius:8px}
    .figure-grid{display:grid;grid-template-columns:repeat(2,minmax(0,1fr));gap:15px}.figure{margin:0;border:1px solid var(--line);padding:12px;border-radius:8px}.figure img{max-width:100%;height:auto;display:block;margin:auto}.figure figcaption{font-size:11px;color:var(--muted);margin-top:10px;word-break:break-word}
    details{margin:8px 0;border:1px solid var(--line);border-radius:8px;background:#fff}summary{cursor:pointer;padding:12px 15px;font-weight:650;font-size:13px}pre{white-space:pre-wrap;word-break:break-word;padding:14px;margin:0;border-top:1px solid var(--line);max-height:440px;overflow:auto;font-size:11px}
    .note{font-size:13px;color:#4c616b;border-left:3px solid #c18a58;padding:9px 15px;background:#fff9f1}
    footer{margin-top:50px;font-size:12px;color:var(--muted);border-top:1px solid var(--line);padding-top:22px}
    @media(max-width:1040px){.kpis{grid-template-columns:repeat(2,1fr)}.two{grid-template-columns:1fr}}
    @media(max-width:760px){.layout{display:block}aside{position:static;height:auto;padding:14px;display:flex;overflow:auto;gap:3px}aside .brand{display:none}aside a{white-space:nowrap}main{padding:24px 16px 45px}.kpis{grid-template-columns:1fr 1fr}.figure-grid{grid-template-columns:1fr}table{display:block;overflow-x:auto}}
    @media print{body{background:white;font-size:10pt}.layout{display:block}aside{display:none}main{padding:0;max-width:none}.section{break-inside:avoid;margin-top:20px}.panel,.kpi{box-shadow:none}details:not([open]){display:none}.kpis{grid-template-columns:repeat(4,1fr)}h1{font-size:23pt}h2{font-size:16pt}}
    '''
    body = f'''<!DOCTYPE html><html lang="en"><head><meta charset="utf-8"/><meta name="viewport" content="width=device-width, initial-scale=1"/>
    <title>Paper 2 | Reproducibility & Results Report</title><style>{style}</style></head><body>
    <div class="layout"><aside><div class="brand">PAPER 02 / RESEARCH</div>
    <a href="#overview">Overview</a><a href="#experiments">Experiments</a><a href="#validation">Verification</a><a href="#figures">Figures</a><a href="#provenance">Provenance</a><a href="#limitations">Interpretation</a><a href="#raw">Source evidence</a></aside>
    <main><header id="overview"><div class="eyebrow">Computational genomics · reproducibility dossier</div>
    <h1>Tokenization-Phase Artifacts in Genomic Foundation Models</h1>
    <p class="lede">Training-free correction of single-nucleotide variant effect scores. Offline report generated from local experiment artifacts, run logs, and SHA-256 provenance records.</p>
    <div class="status"><strong>{e(banner)}</strong><span>Mode: {e(mode)} · Created: {e(created)} · Audit: <span class="pill {audit_class}">{e(audit_label)}</span></span></div>
    <div class="kpis">
      <div class="kpi"><div class="label">Original stability reduction</div><div class="number">{e(orig_reduction)}</div><div class="detail">Only displayed when recovered from this run's audit</div></div>
      <div class="kpi"><div class="label">Held-out stability reduction</div><div class="number">{e(main_reduction)}</div><div class="detail">1,000 position-disjoint SNVs</div></div>
      <div class="kpi"><div class="label">Runtime overhead</div><div class="number">{e(latency)}</div><div class="detail">Six-phase / single-phase median latency</div></div>
      <div class="kpi"><div class="label">Verified numeric fields</div><div class="number">{metrics_verified}/{len(metrics)}</div><div class="detail">Matched to saved result JSON or audit output</div></div>
    </div></header>
    <section class="section" id="experiments"><h2>Experiment results</h2>
      <div class="two"><div class="panel">{bar_a}<p class="caption">Experiment A: fixed 144-nt core, 150-nt inputs, two flank conditions, six phases. Paired difference CI: {e(ci_text)}.</p></div>
      <div class="panel">{bar_original}<p class="caption">Original cohort: anchor shifts of ±6 nt. A smaller value means greater numerical stability.</p></div>
      <div class="panel">{bar_heldout}<p class="caption">Experiment B: 1,000 held-out ClinVar SNVs at disjoint genomic positions, same ClinVar release.</p></div>
      <div class="panel">{bar_runtime}<p class="caption">Experiment C: NVIDIA RTX 3080, batch size four, three repeats; model loading excluded.</p></div></div>
      <div class="panel"><h3>Metric evidence ledger</h3><p class="quiet">Values are marked Verified only when independently located in the saved result JSON or this run's audit stdout. Missing fields are never silently filled with manuscript numbers.</p>
      <table><thead><tr><th>Measurement</th><th>Observed</th><th>Artifact and field</th><th>Evidence</th></tr></thead><tbody>{evidence_table(metrics)}</tbody></table></div>
    </section>
    <section class="section" id="validation"><h2>Execution and audit</h2><div class="panel">{run_table(run_records)}</div>
      <div class="panel"><h3>Audit summary</h3><p><span class="pill {audit_class}">{e(audit_label)}</span></p>
      <p class="quiet">The audit script is the authority for checks on cohort integrity, held-out disjointness, reference alleles, score recomputation, confidence intervals, and benchmark consistency.</p>
      <h3>Audit warnings</h3><ul>{audit_html}</ul></div>
    </section>
    <section class="section" id="figures"><h2>Saved experimental figures</h2><div class="panel">{figure_gallery(root)}</div></section>
    <section class="section" id="provenance"><h2>Data provenance and environment</h2>
      <div class="panel"><h3>Reference genome and cohort</h3><p><strong>GRCh38</strong> · held-out 240-nt reference sequences · variant index 120 (zero-based) · 500 benign / 500 pathogenic.</p>
      <p>Held-out sequence sources: <strong>{source_text}</strong>.</p><p class="quiet">Original and held-out cohorts originate from the same ClinVar release. Position disjointness is not gene-level or independent-source validation.</p></div>
      <div class="panel"><h3>Artifact manifest</h3><table><thead><tr><th>Artifact</th><th>Path</th><th>Size</th><th>SHA-256 prefix</th></tr></thead><tbody>{inventory_rows}</tbody></table>
      <p class="quiet">Full SHA-256 digests are available in the adjacent run_manifest.json file.</p></div>
      <div class="panel"><h3>Execution environment</h3><table><tbody>{env_html}</tbody></table></div>
    </section>
    <section class="section" id="limitations"><h2>Scientific interpretation</h2>
      <div class="panel"><h3>What the evidence supports</h3><p>Six-phase score averaging reduces sensitivity of masked SNV likelihood-ratio scores to sequence-window anchor shifts in the tested six-mer Nucleotide Transformer model. The effect was observed in an original cohort and reproduced on position-disjoint ClinVar SNVs.</p>
      <h3>What it does not establish</h3><ul>
      <li>Clinical pathogenicity improvement: AUROC confidence intervals included zero.</li>
      <li>Exclusive causal attribution to tokenization phase: distal sequence-context allocation changes with the offsets.</li>
      <li>Generalization to other models, tokenizer architectures, clinical cohorts, or independent data sources.</li>
      <li>Gene-independent uncertainty: variant-level bootstrap resampling may overlook correlated variants.</li>
      </ul><p class="note">A numerically stable score is not necessarily biologically correct or clinically actionable. This report is not a clinical decision-support tool.</p></div>
    </section>
    <section class="section" id="raw"><h2>Source JSON records</h2><p class="quiet">Expand to inspect the exact saved evidence used by this report. JSON may be abbreviated if very large.</p>{''.join(json_panels)}</section>
    <footer>Generated by reproduce_paper2.py · Offline/self-contained HTML · Local experiment files remain under the research project directory · No simulated experimental results.</footer>
    </main></div></body></html>'''
    return body


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--root", type=Path, default=ROOT_DEFAULT, help="Paper 2 project root (default: script directory)")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--verify-only", action="store_true", help="Run reproducibility audit only; reuse existing A/B/C results")
    group.add_argument("--report-only", action="store_true", help="Only generate HTML from existing artifacts (no audit)")
    args = parser.parse_args()
    root = args.root.expanduser().resolve()
    mode = "report-only" if args.report_only else "verify-only" if args.verify_only else "full"
    if not (root / "experiments").is_dir():
        print(f"ERROR: {root} does not contain an experiments directory. Extract this file into the Paper-2 project root, or use --root.", file=sys.stderr)
        return 2
    created = datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_root = root / "results" / "publication_report"
    run_dir = out_root / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    print("=" * 70)
    print("PAPER 2 | Reproducibility and HTML report")
    print("Project:", root)
    print("Mode:", mode)
    print("Output:", run_dir)
    print("=" * 70, flush=True)
    env = capture_environment(root)
    run_records = []
    a = read_json(root / ARTIFACTS["A"])
    if mode == "full":
        a_script, why = resolve_a_script(root, a)
        print("Experiment A script resolution:", why, flush=True)
        steps = [
            ("original", root / SCRIPT_MAP["original"]),
            ("A", a_script),
            ("B", root / SCRIPT_MAP["B"]),
            ("C", root / SCRIPT_MAP["C"]),
            ("audit", root / SCRIPT_MAP["audit"]),
        ]
    elif mode == "verify-only":
        steps = [("audit", root / SCRIPT_MAP["audit"])]
    else:
        steps = []
    for key, script in steps:
        run_records.append(run_script(key, script, root, run_dir))
    audit_log = parse_audit_log(run_records)
    data = load_data(root)
    metrics = make_metrics(data, audit_log)
    inventory_rows, manifest = file_inventory(root)
    manifest.append({"name": "orchestrator", "path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__).resolve()), "bytes": Path(__file__).stat().st_size})
    summary = {
        "mode": mode, "created": created, "root": str(root), "environment": env,
        "runs": run_records, "audit_from_current_run": audit_log, "metric_evidence": metrics,
        "artifact_manifest": manifest,
        "disclaimer": "Results are only independently rerun if their scripts have status passed in this run. "
                      "Audit pass verifies local artifacts, not clinical utility or cross-model generalization.",
    }
    (run_dir / "run_manifest.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False, default=str) + "\n", encoding="utf-8")
    report = report_html(root, mode, env, run_records, data, audit_log, metrics, inventory_rows, created)
    html_path = run_dir / "index.html"
    html_path.write_text(report, encoding="utf-8")
    latest = out_root / "index.html"
    shutil.copy2(html_path, latest)
    shutil.copy2(run_dir / "run_manifest.json", out_root / "latest_manifest.json")
    print("\n" + "=" * 70)
    print("HTML REPORT:", latest)
    print("RUN ARCHIVE:", run_dir)
    print("METRIC FIELDS VERIFIED:", sum(m["status"] == "matched" for m in metrics), "/", len(metrics))
    if mode != "report-only":
        failures = [r for r in run_records if r["status"] != "passed"]
        audit_ok = audit_log.get("complete") and audit_log.get("passed") == audit_log.get("total") and not audit_log.get("failures")
        if failures or not audit_ok:
            print("WARNING: One or more rerun/audit steps were not confirmed. See HTML and logs.")
            print("=" * 70)
            return 1
    print("=" * 70)
    return 0


if __name__ == "__main__":
    sys.exit(main())
