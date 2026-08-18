#!/usr/bin/env bash
# =============================================================================
# prepare_data.sh
# -----------------------------------------------------------------------------
# Builds everything the training + eval scripts read, into LOCAL paths (no
# shared-filesystem defaults). Two stages:
#
#   STAGE=train : preprocess BOTH big-tier training datasets once each
#                 (cap<=1024 long side, RGB, prune, save_to_disk) so training
#                 skips the slow per-rank image map. Uses MAX_SAMPLES=8000.
#                   MMR1/MMR1-Math-RL-Data-v0        -> $PRE_ROOT/mmr1_8k
#                   lmms-lab/multimodal-open-r1-8k-verified -> $PRE_ROOT/openr1_8k
#
#   STAGE=eval  : build the final 4-bench eval set via eval/prepare_benchmarks.py
#                   MathVision  (MathLLMs/MathVision, 3040)   <- fetched from HF
#                   MathVerse   (AI4Math/MathVerse testmini)  <- fetched from HF
#                   We-Math     (We-Math/We-Math testmini)    <- fetched from HF
#                   MathVista   (1000)                        <- LOCAL, bundled
#
#   STAGE=all   : train + eval   (default)
#
# Env overrides:
#   PYTHON        interpreter                     (default: python)
#   STAGE         train | eval | all              (default: all)
#   MAX_SAMPLES   train truncation                (default: 8000)
#   PRE_ROOT      preprocessed-train output root  (default: $REPO/data/mllm_pre)
#   OUT_ROOT      4-bench eval output root        (default: $REPO/data/mllm_eval)
#   HF_HOME       hub cache (shared w/ prefetch_models.sh)
#   HF_TOKEN      only if a source dataset is gated (these two are public)
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-python}"
STAGE="${STAGE:-all}"
MAX_SAMPLES="${MAX_SAMPLES:-8000}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

PRE_ROOT="${PRE_ROOT:-$REPO_ROOT/data/mllm_pre}"
OUT_ROOT="${OUT_ROOT:-$REPO_ROOT/data/mllm_eval}"

# Prevent the ~/.local user-site torch stub from shadowing the env's torch.
export PYTHONNOUSERSITE=1

log() { printf '[prepare-data] %s\n' "$*"; }
die() { printf '[prepare-data][ERROR] %s\n' "$*" >&2; exit 1; }

command -v "$PYTHON" >/dev/null 2>&1 || die "python interpreter '$PYTHON' not found"
cd "$REPO_ROOT"

# --- bundled MathVista (LOCAL) ------------------------------------------------
# MathVista is the one benchmark NOT fetched from HF. Two files live under
# data/mathvista/ (both reference images under data/mathvista/images/):
#   testmini_150.jsonl  -> 150-subset, used for IN-LOOP eval during training
#   testmini.jsonl      -> full 1000, used for the final 4-bench eval
# Schema per line: {"problem": str, "image": "images/<x>.png" (relative),
#                   "solution": str, "qtype": "mcq"|"free"}
MV_DIR="$REPO_ROOT/data/mathvista"
MV_INLOOP="$MV_DIR/testmini_150.jsonl"   # -> MLLM_EVAL_PATH (training in-loop)
MV_FULL="$MV_DIR/testmini.jsonl"         # -> $OUT_ROOT/mathvista (final 4-bench)

require_mathvista_inloop() {
  if [ ! -f "$MV_INLOOP" ]; then
    cat >&2 <<EOF
[prepare-data][ERROR] Missing bundled MathVista in-loop set:
    $MV_INLOOP
The training scripts need this for IN-LOOP eval / checkpoint selection, and the
preprocessor loads it too (so it must exist before STAGE=train runs).

Fix, put the MathVista testmini files under data/mathvista/ :
    data/mathvista/testmini_150.jsonl   (150-subset, in-loop)
    data/mathvista/testmini.jsonl       (full 1000, final 4-bench)
    data/mathvista/images/...           (images referenced by both jsonls)
Source: MathVista testmini (1000).
To make the 150-subset from the 1000:
    head -n 150 data/mathvista/testmini.jsonl > data/mathvista/testmini_150.jsonl
EOF
    exit 1
  fi
}

# ---------------------------------------------------------------------------
# STAGE: train , preprocess both datasets
# ---------------------------------------------------------------------------
run_train_stage() {
  require_mathvista_inloop
  mkdir -p "$PRE_ROOT"

  # Force MLLM_EVAL_PATH to the REAL local in-loop set BEFORE calling the
  # preprocessor. The tool defaults MLLM_EVAL_PATH to a NAS path when unset and
  # then tries to OPEN it -> crash. Pointing it at the
  # bundled 150 both fixes that and forces the "train on ALL of train" branch
  # (so save_to_disk holds the full capped train, no 150-holdout carve).
  export MLLM_EVAL_PATH="$MV_INLOOP"
  export MLLM_EVAL_IMAGE_DIR="$MV_DIR"
  export MAX_SAMPLES

  # (dataset id, output subdir)
  local pairs=(
    "MMR1/MMR1-Math-RL-Data-v0|mmr1_8k"
    "lmms-lab/multimodal-open-r1-8k-verified|openr1_8k"
  )
  for p in "${pairs[@]}"; do
    local ds="${p%%|*}" out_sub="${p##*|}"
    local out="$PRE_ROOT/$out_sub"
    log ">> preprocess: $ds  (MAX_SAMPLES=$MAX_SAMPLES)  ->  $out"
    MAX_SAMPLES="$MAX_SAMPLES" "$PYTHON" tools/preprocess_mllm_dataset.py "$ds" "$out"
  done
  log "training data ready under: $PRE_ROOT"
  log "   MMR1    -> $PRE_ROOT/mmr1_8k"
  log "   open_r1 -> $PRE_ROOT/openr1_8k"
}

# ---------------------------------------------------------------------------
# STAGE: eval , build 4-bench set
# ---------------------------------------------------------------------------
run_eval_stage() {
  mkdir -p "$OUT_ROOT"
  # The bundled MathVista dir ships only the jsonl -- the 150 images it points
  # at are rebuilt (with per-row verification) if absent. Without this the first
  # in-loop eval dies on images/1.png.
  if [ ! -f "$MV_DIR/images/1.png" ]; then
    log ">> materialising MathVista eval images (tools/materialise_mathvista_images.py)"
    "$PYTHON" "$REPO_ROOT/tools/materialise_mathvista_images.py"
  fi
  export OUT_ROOT

  # MathVista: pre-place $OUT_ROOT/mathvista as a symlink to the bundled dir so
  # prepare_benchmarks.py's mathvista() sees it already exists and skips its
  # own download step. prepare_benchmarks.py takes ONE target per call (a
  # single bench name or "all"), so we drive it explicitly per bench.
  local benches=(mathvision mathverse wemath mathvista)
  if [ -e "$OUT_ROOT/mathvista" ]; then
    log "MathVista: $OUT_ROOT/mathvista already present (mathvista step will no-op)."
  elif [ -d "$MV_DIR" ] && { [ -f "$MV_FULL" ] || [ -f "$MV_INLOOP" ]; }; then
    ln -s "$MV_DIR" "$OUT_ROOT/mathvista"
    log "MathVista: symlinked bundled $MV_DIR -> $OUT_ROOT/mathvista (mathvista step will no-op)."
  else
    # No bundled MathVista -> DROP it from the list (its mathvista() would try
    # the hardcoded NAS path and crash on open()). Fetch only the 3 HF benches.
    benches=(mathvision mathverse wemath)
    log "WARNING: no bundled MathVista at $MV_DIR, fetching only the 3 HF benches."
    log "         Final 4-bench mathvista row will be missing until you add $MV_FULL."
  fi

  # Fetch each bench (mathvista no-ops when the symlink exists).
  for b in "${benches[@]}"; do
    log ">> eval/prepare_benchmarks.py $b  (OUT_ROOT=$OUT_ROOT)"
    "$PYTHON" eval/prepare_benchmarks.py "$b"
  done
  log "4-bench eval set ready under: $OUT_ROOT"
  log "   mathvision/data.jsonl, mathverse/data.jsonl, wemath/data.jsonl, mathvista/testmini.jsonl"
}

# --- dispatch -----------------------------------------------------------------
case "$STAGE" in
  train) run_train_stage;;
  eval)  run_eval_stage;;
  # eval stage first: it materialises the MathVista images that the
  # training stage's in-loop eval set points at.
  all)   run_eval_stage; run_train_stage;;
  *) die "STAGE must be train|eval|all (got '$STAGE')";;
esac

# ---------------------------------------------------------------------------
# Env-var contract the TRAINING scripts need (set these before launching a run)
# ---------------------------------------------------------------------------
cat <<EOF

============================================================================
 DONE.  Export these before launching training (trainers/train_mllm_single.py):
----------------------------------------------------------------------------
 # (1) preprocessed train set, pick ONE per run:
 export MLLM_PRE_DIR=$PRE_ROOT/mmr1_8k        # or: $PRE_ROOT/openr1_8k
 # (2) in-loop eval set (MathVista-150) for checkpoint selection:
 export MLLM_EVAL_PATH=$MV_INLOOP
 export MLLM_EVAL_IMAGE_DIR=$MV_DIR
 # (optional) debug truncation of train:
 # export MAX_SAMPLES=$MAX_SAMPLES

 # For the FINAL 4-bench eval (eval/run_eval_all.sh) instead export:
 export OUT_ROOT=$OUT_ROOT
============================================================================
EOF
