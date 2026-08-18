#!/usr/bin/env bash
# =============================================================================
# prefetch_models.sh
# -----------------------------------------------------------------------------
# Model weights are NOT committed. Training/eval reference models by HF id and
# read them from the HF hub cache ($HF_HOME/hub). This script pulls them there.
#
# Where egress is available, prefetch is
# mostly a speed/cache optimization, training would download on first use too.
# On an air-gapped / offline cluster: run THIS script
# on a LOGIN node that has network, then on the compute node export
#     HF_HUB_OFFLINE=1   (and point HF_HOME at the same cache)
# so every later `from_pretrained(<id>)` resolves purely from the local cache.
#
# Model tiers:
#   big   (default)  = the 3 big-tier ids the main experiments use:
#                        Qwen/Qwen2.5-VL-7B-Instruct     (ungated)
#                        OpenGVLab/InternVL3_5-8B-HF     (ungated, apache-2.0)
#                        google/gemma-3-12b-it           (GATED, see below)
#   smoke            = the small-tier variants used for MAX_STEPS=2 smokes:
#                        Qwen/Qwen2.5-VL-3B-Instruct
#                        OpenGVLab/InternVL3_5-2B-HF
#                        google/gemma-3-4b-it            (GATED)
#   all              = big + smoke
#
# NOTE ON COUNT: The model list below is exactly the
# 3 big-tier ids and that is the authoritative REQUIRED set (default here). The
# smoke-tier trio is opt-in so you don't pull ~100GB+ you didn't ask for.
#
# GATED MODELS (gemma-*): you must (1) accept the Google Gemma license on the
# model's HF page while logged in, and (2) provide a read-scope token via
# HF_TOKEN. Without a token the gemma pulls are SKIPPED (non-fatal) so the
# ungated models still download.  (Gemma is
# model and gating is typically a non-issue.)
#
# Env overrides:
#   HF_HOME               cache root (default: $HOME/.cache/huggingface)
#   HF_TOKEN              read token; required only for gated (gemma) ids
#   MLLM_PREFETCH_TIER    big | smoke | all         (default: big)
#   MLLM_MODELS           space-separated id list; OVERRIDES tier entirely
#   HF_XET_HIGH_PERFORMANCE / HF_HUB_ENABLE_HF_TRANSFER  (optional, if installed)
# =============================================================================
set -euo pipefail

log() { printf '[prefetch] %s\n' "$*"; }
die() { printf '[prefetch][ERROR] %s\n' "$*" >&2; exit 1; }

# --- HF cache -----------------------------------------------------------------
export HF_HOME="${HF_HOME:-$HOME/.cache/huggingface}"
mkdir -p "$HF_HOME"
log "HF_HOME=$HF_HOME (models land in \$HF_HOME/hub)"

# --- CLI (prefer `huggingface-cli`, fall back to the newer `hf`) --------------
if command -v huggingface-cli >/dev/null 2>&1; then
  HF_DL=(huggingface-cli download)
elif command -v hf >/dev/null 2>&1; then
  HF_DL=(hf download)
else
  die "neither 'huggingface-cli' nor 'hf' found. Install huggingface_hub (in requirements.txt)."
fi
log "using: ${HF_DL[*]} <id>"

# --- token (exported so the CLI/library picks it up automatically) ------------
if [ -n "${HF_TOKEN:-}" ]; then
  export HF_TOKEN
  export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"   # older lib name
  TOKEN_ARGS=(--token "$HF_TOKEN")
  log "HF_TOKEN is set (gated models enabled)."
else
  TOKEN_ARGS=()
  log "HF_TOKEN not set: gated (gemma) models will be SKIPPED."
fi

# --- model lists --------------------------------------------------------------
BIG=( "Qwen/Qwen2.5-VL-7B-Instruct" "OpenGVLab/InternVL3_5-8B-HF" "google/gemma-3-12b-it" )
SMOKE=( "Qwen/Qwen2.5-VL-3B-Instruct" "OpenGVLab/InternVL3_5-2B-HF" "google/gemma-3-4b-it" )

if [ -n "${MLLM_MODELS:-}" ]; then
  # shellcheck disable=SC2206
  MODELS=( ${MLLM_MODELS} )
  log "model set: explicit MLLM_MODELS override (${#MODELS[@]} ids)"
else
  case "${MLLM_PREFETCH_TIER:-big}" in
    big)   MODELS=( "${BIG[@]}" );;
    smoke) MODELS=( "${SMOKE[@]}" );;
    all)   MODELS=( "${BIG[@]}" "${SMOKE[@]}" );;
    *) die "MLLM_PREFETCH_TIER must be big|smoke|all (got '${MLLM_PREFETCH_TIER}')";;
  esac
  log "model set: tier='${MLLM_PREFETCH_TIER:-big}' (${#MODELS[@]} ids)"
fi

is_gated() { case "$1" in google/gemma-*) return 0;; *) return 1;; esac; }

# --- download loop ------------------------------------------------------------
OK=(); SKIPPED=(); FAILED=()
for m in "${MODELS[@]}"; do
  if is_gated "$m" && [ ${#TOKEN_ARGS[@]} -eq 0 ]; then
    log ">> SKIP (gated, no HF_TOKEN): $m"
    log "        accept license at https://huggingface.co/$m then re-run with HF_TOKEN set."
    SKIPPED+=("$m")
    continue
  fi
  log ">> downloading: $m"
  if "${HF_DL[@]}" "$m" "${TOKEN_ARGS[@]}"; then
    OK+=("$m")
  else
    log "!! FAILED: $m"
    FAILED+=("$m")
  fi
done

# --- summary ------------------------------------------------------------------
log "-------------------------------------------------------------"
log "downloaded (${#OK[@]}): ${OK[*]:-none}"
[ ${#SKIPPED[@]} -eq 0 ] || log "skipped   (${#SKIPPED[@]}): ${SKIPPED[*]}  (gated, set HF_TOKEN)"
[ ${#FAILED[@]}  -eq 0 ] || log "FAILED    (${#FAILED[@]}): ${FAILED[*]}"
log "cache: $HF_HOME/hub"
log "offline clusters: on the compute node set  HF_HUB_OFFLINE=1  and reuse this HF_HOME."
[ ${#FAILED[@]} -eq 0 ] || die "one or more downloads failed (see above)."
