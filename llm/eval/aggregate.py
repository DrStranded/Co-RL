#!/usr/bin/env python
"""Merge the vLLM-run outputs into one CSV row (paper 7-benchmark suite).

Reads from a RUN_DIR laid out by run_eval_all.sh:

    RUN_DIR/
      lm_eval/results.json        (and per-task subdir)
      lcb/lcb_v6.json

      run.log

Writes CSV with columns matching paper §4.2 main table + appendix:

    ckpt, revision, gsm8k, math_500, amc, humaneval, mbpp, gpqa_d, lcb_v6

Missing benchmarks (skipped runs / failed) emit "NA" in their cell so the
row stays alignable; the CSV is intended to append-write across many ckpts.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path


# (csv column, lm-eval task key, metric key)
# Notes:
#   - minerva_math500 (no underscore) is the actual lm-eval task name. The
#     custom math_500_chat (chat-friendly 0-shot \boxed{}) is preferred when
#     present — see _PREFER below.
#   - math_500: math_verify (sympy-based equivalence). "exact_match,none" is
#     too strict (e.g. "1/2" vs "0.5" miscount as wrong).
#   - gpqa_d: custom task gpqa_diamond_boxed (utils.process_results_gpqa) — extracts
#     \boxed{A-D} exactly like CoMAS (maslab/evaluation.py), so numbers are comparable.
#     Native gpqa flexible-extract misses \boxed{A} (37% of RL-ckpt answers → wrong 0).
#   - gsm8k: flexible-extract, NOT strict-match. strict-match only matches the native
#     "#### N" format; RL ckpts trained to emit \boxed{N} score ~0 under strict (e.g.
#     heter-Qwen: strict 0.34 vs flexible 0.81). flexible-extract = last number in the
#     completion, which catches the digits inside \boxed{N}. Same lesson as gpqa above.
#   - mbpp metric key is `pass_at_1` (underscore), not `pass@1`.
#   - humaneval/mbpp _instruct variants override the default when present
#     (chat-aware extractor handles markdown code fences).
_LM_EVAL_TASKS = [
    ("gsm8k",      "gsm8k",                       "exact_match,flexible-extract"),
    ("math_500",   "minerva_math500",             "exact_match,none"),
    ("amc",        "amc23",                       "exact_match,avg8"),
    ("humaneval",  "humaneval",                   "pass@1,create_test"),
    ("mbpp",       "mbpp",                        "pass_at_1,none"),
    ("gpqa_d",     "gpqa_diamond_boxed",          "exact_match,none"),
]


def _read_lm_eval(run_dir: Path) -> dict[str, float | None]:
    """lm-eval-harness writes results.json with a `results` dict keyed by task name.
    Multiple results*.json files may coexist (e.g. main 10-task suite + a follow-up
    humaneval_instruct run). Merge all of them; newer files overwrite per task key."""
    out: dict[str, float | None] = {c: None for c, _, _ in _LM_EVAL_TASKS}
    candidates = list(run_dir.rglob("results*.json"))
    if not candidates:
        return out
    candidates.sort(key=lambda p: p.stat().st_mtime)  # oldest → newest
    res: dict = {}
    for p in candidates:
        try:
            data = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        for task_key, task_node in (data.get("results") or {}).items():
            res[task_key] = task_node  # later (newer) wins

    # Prefer chat-aware / chat-friendly variants when present.
    _PREFER = {
        "humaneval":       "humaneval_instruct",   # chat-aware extractor
        "mbpp":            "mbpp_instruct",        # chat-aware extractor
        "minerva_math500": "math_500_chat",        # 0-shot chat-friendly prompt (better for all models)
    }
    _SKIP = ("stderr", "sample_len", "alias", "bypass", "n,")
    for csv_col, task_key, metric_key in _LM_EVAL_TASKS:
        preferred = _PREFER.get(task_key)
        if preferred and preferred in res:
            node = res[preferred]
        else:
            node = res.get(task_key)
        if not node:
            continue
        val = node.get(metric_key)
        if val is None:
            # Fallback: first numeric metric, skipping sample-count / alias keys.
            for k, v in node.items():
                if isinstance(v, (int, float)) and not any(s in k for s in _SKIP):
                    val = v
                    break
        if isinstance(val, (int, float)):
            out[csv_col] = float(val)
    return out


def _read_external(path: Path, default_key: str) -> float | None:
    if not path.exists():
        return None
    try:
        data = json.loads(path.read_text())
    except json.JSONDecodeError:
        return None
    score = data.get("score")
    return float(score) if isinstance(score, (int, float)) else None


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run_dir", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--revision", default="")
    ap.add_argument("--out_csv", required=True)
    args = ap.parse_args()

    run_dir = Path(args.run_dir)
    if not run_dir.is_dir():
        print(f"ERROR: run_dir not found: {run_dir}", file=sys.stderr)
        return 2

    lm_eval = _read_lm_eval(run_dir / "lm_eval")
    lcb     = _read_external(run_dir / "lcb"      / "lcb_v6.json",   "lcb_v6")

    row = {
        "ckpt":      args.model,
        "revision":  args.revision,
        "gsm8k":     lm_eval["gsm8k"],
        "math_500":  lm_eval["math_500"],
        "amc":       lm_eval["amc"],
        "humaneval": lm_eval["humaneval"],
        "mbpp":      lm_eval["mbpp"],
        "gpqa_d":    lm_eval["gpqa_d"],
        "lcb_v6":    lcb,
    }

    # Format: 4 decimal places, "NA" for None.
    def _fmt(v: float | None) -> str:
        return "NA" if v is None else f"{v:.4f}"

    headers = list(row.keys())
    formatted = {k: (v if k in ("ckpt", "revision") else _fmt(v)) for k, v in row.items()}

    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    new_file = not out_csv.exists()
    with out_csv.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        if new_file:
            w.writeheader()
        w.writerow(formatted)

    # Echo a one-liner with the headline metrics.
    print(
        f"ckpt={args.model} rev={args.revision or '-'}  "
        f"gsm8k={formatted['gsm8k']} math500={formatted['math_500']} "
        f"amc={formatted['amc']} "
        f"humaneval={formatted['humaneval']} gpqa_d={formatted['gpqa_d']}"
    )
    print(f"row appended → {out_csv}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
