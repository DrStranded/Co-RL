#!/usr/bin/env python
"""LiveCodeBench v6 wrapper for run_eval_all.sh.

Strategy: shell out to LCB's own `lcb_runner.runner.main` (which already
supports vLLM via `--model_provider vllm`) and then read its output JSON to
normalize into our standard schema:

    {"benchmark": "lcb_v6", "score": float, "n": int, "raw_path": str}

Notes on LCB:
  - LCB writes outputs under `<LCB_DIR>/output/<model_name>/...`.
  - `--release_version release_v6` selects the v6 test set (~1055 problems).
  - LCB uses its own sandbox for code execution; sandbox = subprocess on the
    same machine, so HF_ALLOW_CODE_EVAL is honored implicitly.

Caveats:
  - LCB is sensitive to model name / chat template. If `--model` is a local
    path, LCB's name handling may strip slashes — pass repo-style names where
    possible.
  - Pass-rate metric LCB reports: `pass@1` averaged over problems. We surface
    that as `score`.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1] / "external_repos" / "LiveCodeBench"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="HF repo or local path")
    ap.add_argument("--revision", default=None)
    ap.add_argument("--release_version", default="release_v6")
    ap.add_argument("--output", required=True, help="normalized JSON output path")
    ap.add_argument("--max_model_len", default="4096")
    ap.add_argument("--gpu_mem", default="0.9")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--n", type=int, default=1, help="pass@n samples per problem")
    args = ap.parse_args()

    if not REPO_DIR.exists():
        print(f"ERROR: LiveCodeBench repo not found at {REPO_DIR}.", file=sys.stderr)
        print("       Re-run eval/setup_external.sh.", file=sys.stderr)
        return 2

    # Build the LCB runner command.
    # NOTE: LCB has NO `--model_provider` flag — it routes by name via lm_styles.py.
    # Models must be registered in lm_styles.py first (see external_repos/LiveCodeBench
    # patches in the eval setup docs).
    cmd = [
        sys.executable, "-m", "lcb_runner.runner.main",
        "--model", args.model,
        "--scenario", "codegeneration",
        "--release_version", args.release_version,
        "--n", str(args.n),
        "--evaluate",
    ]
    env = os.environ.copy()
    if args.revision:
        env["HF_REVISION"] = args.revision
    env["VLLM_GPU_MEMORY_UTILIZATION"] = str(args.gpu_mem)
    env["VLLM_MAX_MODEL_LEN"] = str(args.max_model_len)
    if args.limit:
        cmd += ["--num_process_evaluate", "1"]

    print("[lcb] cwd:", REPO_DIR)
    print("[lcb] cmd:", " ".join(cmd))
    rc = subprocess.call(cmd, cwd=str(REPO_DIR), env=env)
    if rc != 0:
        # Non-fatal: write an NA result and exit 0 so aggregation still runs.
        # LCB integration is fragile (model registry in lm_styles.py; gated tokenizers
        # hardcoded in prompt files; datasets script-loader compat). See setup docs.
        print(f"[lcb] runner exited {rc} — writing NA result and continuing.", file=sys.stderr)
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(
            {"benchmark": "lcb_v6", "score": None, "n": 0,
             "error": f"LCB runner exited {rc}; likely model not in lm_styles.py "
                      "or dataset/tokenizer load issue."},
            indent=2,
        ))
        return 0

    # Find the LCB metric JSON. LCB writes `output/<model>/Scenario.codegeneration_*_eval_all.json`.
    # We look for the most recent one.
    out_root = REPO_DIR / "output"
    matches = sorted(
        out_root.rglob("*_eval_all.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not matches:
        print("[lcb] could not locate *_eval_all.json under output/", file=sys.stderr)
        return 3

    raw = json.loads(matches[0].read_text())
    # LCB schema: list of {"problem_id", "pass@1", ...} OR a top-level dict
    # with "pass@1" aggregated. We try both.
    if isinstance(raw, dict) and "pass@1" in raw:
        score = float(raw["pass@1"])
        n = int(raw.get("num_problems", 0)) or 0
    elif isinstance(raw, list):
        scored = [d for d in raw if "pass@1" in d]
        n = len(scored)
        score = sum(float(d["pass@1"]) for d in scored) / n if n else 0.0
    else:
        print("[lcb] unexpected JSON shape; raw saved as-is", file=sys.stderr)
        score = 0.0
        n = 0

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(
        json.dumps(
            {"benchmark": "lcb_v6", "score": score, "n": n, "raw_path": str(matches[0])},
            indent=2,
        )
    )
    print(f"[lcb] score={score:.4f} (n={n}) → {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
