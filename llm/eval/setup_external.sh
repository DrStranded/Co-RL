#!/usr/bin/env bash
# Clone, patch, and install the external evaluation harnesses next to this script.
# Run once before eval/run_eval_all.sh.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
mkdir -p "$HERE/external_repos"
cd "$HERE/external_repos"
[ -d lm-evaluation-harness ] || git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness.git
[ -d LiveCodeBench ]         || git clone --depth 1 https://github.com/LiveCodeBench/LiveCodeBench.git
# Apply the bundled patches (idempotent; `patch --forward` skips if applied).
patch --forward -p1 -d lm-evaluation-harness < "$HERE/patches/lmeval_gemma_u2581.patch"
patch --forward -p1 -d lm-evaluation-harness < "$HERE/patches/lmeval_mbpp_lang_tag.patch"
patch --forward -p1 -d LiveCodeBench         < "$HERE/patches/livecodebench_register_baselines.patch"
# Install both so `lm_eval` is on PATH and the LiveCodeBench runner is importable.
# math-verify grades AMC / MATH-500. It pulls antlr4 4.13, which conflicts with
# the 4.7 the training stack pins, so install it in the evaluation environment
# only -- never alongside training.
pip install math-verify
pip install -e lm-evaluation-harness
pip install -e LiveCodeBench

echo "external harnesses ready (cloned, patched, installed) under $HERE/external_repos"
