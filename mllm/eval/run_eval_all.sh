#!/usr/bin/env bash
# Evaluate ONE MLLM checkpoint on all 4 benchmarks -> 1-row CSV (append-safe).
# Benchmarks: MathVision / MathVerse / MathVista / We-Math / CoreCognition.
#
# Usage:
#   bash eval/run_eval_all.sh --model <ckpt> --tag <name> [--csv <path>] \
#        [--prompt answer|boxed] [--gpu 0] [--limit N] [--out_dir <dir>]
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"   # repo root

OUT_ROOT="${OUT_ROOT:-./data/mllm_eval}"
MODEL="" TAG="" CSV="" PROMPT="answer" GPU="0" LIMIT="0" OUTDIR=""
while [ $# -gt 0 ]; do case "$1" in
  --model) MODEL="$2"; shift 2;;
  --tag) TAG="$2"; shift 2;;
  --csv) CSV="$2"; shift 2;;
  --prompt) PROMPT="$2"; shift 2;;
  --gpu) GPU="$2"; shift 2;;
  --limit) LIMIT="$2"; shift 2;;
  --out_dir) OUTDIR="$2"; shift 2;;
  *) echo "unknown arg $1"; exit 1;;
esac; done
[ -n "$MODEL" ] || { echo "need --model"; exit 1; }
[ -n "$TAG" ] || TAG="$(basename "$MODEL")"
TS="$(date +%Y%m%d_%H%M%S)"
OUTDIR="${OUTDIR:-work_dirs/eval/${TAG}_${TS}}"
CSV="${CSV:-$OUTDIR/results.csv}"
mkdir -p "$OUTDIR"

# benchmark -> jsonl path (mathvista uses local testmini.jsonl; others data.jsonl)
declare -A DATA=(
  [mathvision]="$OUT_ROOT/mathvision/data.jsonl"
  [mathverse]="$OUT_ROOT/mathverse/data.jsonl"
  [mathvista]="$OUT_ROOT/mathvista/testmini.jsonl"
  [wemath]="$OUT_ROOT/wemath/data.jsonl"
  [corecognition]="$OUT_ROOT/corecognition/data.jsonl"
)
ORDER=(mathvision mathverse mathvista wemath)
[ -n "${ONLY:-}" ] && ORDER=($ONLY)   # e.g. ONLY=corecognition for a single bench

for b in "${ORDER[@]}"; do
  d="${DATA[$b]}"
  if [ ! -f "$d" ]; then echo "!! missing data for $b: $d (run: python eval/prepare_benchmarks.py $b)"; continue; fi
  if [ -f "$OUTDIR/$b.json" ] && python -c "
import json,sys
o=json.load(open('$OUTDIR/$b.json'))
sys.exit(0 if o.get('n',0)>0 and len(o.get('samples',[]))==o['n'] else 1)" 2>/dev/null; then
    echo "==== [$b] complete, skipping ===="; continue
  fi
  echo "==== [$b] ===="
  CUDA_VISIBLE_DEVICES="$GPU" python eval/eval_mllm.py \
    --model "$MODEL" --data "$d" --image_dir "$(dirname "$d")" \
    --out "$OUTDIR/$b.json" --prompt "$PROMPT" --limit "$LIMIT" \
    --max_tokens "${MAX_TOKENS:-16384}" --max_model_len "${MAX_MODEL_LEN:-24576}" 2>&1 | tee -a "$OUTDIR/run.log"
done

# aggregate 4 json -> 1 CSV row (append-safe)
python eval/aggregate_row.py --tag "$TAG" --model "$MODEL" --out_dir "$OUTDIR" --csv "$CSV"
echo ">>> DONE  $TAG  -> $CSV"
