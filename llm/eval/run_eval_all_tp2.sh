#!/usr/bin/env bash
# =============================================================================
# projects/eval/run_eval_all.sh
#
# Plan A driver: 1 ckpt → 13 benchmark → 1 CSV row.
# Tensor-parallel-2 variant of run_eval_all.sh, for 7B/8B checkpoints.
#
# Four sequential vLLM runs:
#   Run 1: lm-eval-harness — GSM8K, MATH-500, AMC23, HumanEval, MBPP, GPQA-Diamond
#   Run 2: LiveCodeBench v6   (external repo)
# → aggregate.py → results.csv (1 row, 13 col)
#
# Usage:
#   bash projects/eval/run_eval_all.sh \
#       --model <hf_repo_or_local_path> \
#       [--revision step-170] \
#       [--out_dir work_dirs/eval] \
#       [--gpu 0] \
#       [--max_model_len 4096] \
#       [--gpu_mem 0.9] \
#       [--skip_lm_eval] [--skip_lcb]
#       [--limit N]      # debug: first N examples per task
#
# Env vars:
#   EVAL_ENV_NAME  conda env name (activate it first; the script does not)
#   HF_HOME        HF cache dir
#
# One checkpoint takes ~3-4 h (7B/8B model, 2 x 80GB GPUs).
# =============================================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXT_REPOS_DIR="$SCRIPT_DIR/external_repos"
CUSTOM_TASKS_DIR="$SCRIPT_DIR/lm_eval_custom_tasks"

# --- defaults ---
MODEL=""
REVISION=""
OUT_DIR="$REPO_ROOT/llm/work_dirs/eval"
GPU=""
MAX_MODEL_LEN="4096"
GPU_MEM="0.9"
LIMIT=""
CSV=""                                    # if set, aggregate.py appends here instead of RUN_DIR/results.csv
TASKS_OVERRIDE=""                         # if set, replaces the default LM_EVAL_TASKS list (subset runs)
CHAT_TEMPLATE=0                           # default OFF — base models like Qwen2.5-3B mis-handle chat tokens
SKIP_LM_EVAL=0
SKIP_LCB=0

# --- arg parse ---
while [ $# -gt 0 ]; do
    case "$1" in
        --model)         MODEL="$2"; shift 2;;
        --revision)      REVISION="$2"; shift 2;;
        --out_dir)       OUT_DIR="$2"; shift 2;;
        --gpu)           GPU="$2"; shift 2;;
        --max_model_len) MAX_MODEL_LEN="$2"; shift 2;;
        --gpu_mem)       GPU_MEM="$2"; shift 2;;
        --limit)         LIMIT="$2"; shift 2;;
        --csv)           CSV="$2"; shift 2;;
        --tasks)         TASKS_OVERRIDE="$2"; shift 2;;
        --chat_template) CHAT_TEMPLATE=1; shift;;
        --skip_lm_eval)  SKIP_LM_EVAL=1; shift;;
        --skip_lcb)      SKIP_LCB=1; shift;;
        -h|--help)
            sed -n '2,40p' "$0"; exit 0;;
        *) echo "unknown arg: $1" >&2; exit 1;;
    esac
done

if [ -z "$MODEL" ]; then
    echo "ERROR: --model required" >&2
    exit 1
fi
if [ -n "$GPU" ]; then
    export CUDA_VISIBLE_DEVICES="$GPU"
fi

# Run id: model basename + optional revision + timestamp.
MODEL_TAG=$(echo "$MODEL" | tr '/:' '__')
REV_TAG="${REVISION:+_$REVISION}"
TS=$(date +%Y%m%d_%H%M%S)
RUN_DIR="$OUT_DIR/${MODEL_TAG}${REV_TAG}_${TS}"
mkdir -p "$RUN_DIR"
echo "RUN_DIR=$RUN_DIR"

# --- log everything ---
exec > >(tee -a "$RUN_DIR/run.log") 2>&1

bold() { printf "\n\033[1m===== %s =====\033[0m\n" "$*"; }

bold "Config"
echo "  MODEL          $MODEL"
echo "  REVISION       ${REVISION:-(none)}"
echo "  GPU            ${CUDA_VISIBLE_DEVICES:-(all)}"
echo "  MAX_MODEL_LEN  $MAX_MODEL_LEN"
echo "  GPU_MEM        $GPU_MEM"
echo "  OUT_DIR        $RUN_DIR"
[ -n "$LIMIT" ] && echo "  LIMIT          $LIMIT (debug)"
nvidia-smi --query-gpu=index,name,memory.free --format=csv,noheader || true

# Build a common vLLM model_args string for lm-eval and the external runners.
VLLM_ARGS="pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=$GPU_MEM,max_model_len=$MAX_MODEL_LEN,trust_remote_code=True,tensor_parallel_size=2"
[ -n "$REVISION" ] && VLLM_ARGS="$VLLM_ARGS,revision=$REVISION"

# =============================================================================
# Run 1: lm-eval-harness  ×  10 task
# =============================================================================
if [ "$SKIP_LM_EVAL" = "0" ]; then
    bold "Run 1/4  lm-eval-harness × 10 task"
    LM_EVAL_OUT="$RUN_DIR/lm_eval"
    mkdir -p "$LM_EVAL_OUT"

    # NOTES on task choices:
    # - minerva_math500 (no underscore) is the actual lm-eval task name.
    # - We also run math_500_chat (custom 0-shot \boxed{} prompt) — works much better for chat
    #   models (Gemma3 0.28 → 0.76) AND base models (Qwen 0.39 → 0.59). aggregate.py prefers it.
    # - For chat/instruct models: humaneval_instruct + mbpp_instruct (chat-aware extractor;
    #   default humaneval/mbpp fail on instruct outputs because they wrap code in markdown).
    # - GPQA default `max_gen_toks` is too short (256), instruct models get truncated before
    #   emitting an answer letter → [invalid]. Override via --gen_kwargs below.
    if [ "$CHAT_TEMPLATE" = "1" ]; then
        LM_EVAL_TASKS="gsm8k,math_500_chat,humaneval_instruct,mbpp_instruct,gpqa_diamond_boxed,amc23"
    else
        LM_EVAL_TASKS="gsm8k,math_500_chat,humaneval,mbpp,gpqa_diamond_boxed,amc23"
    fi
    # --tasks overrides the full list (e.g. math-only re-runs: gsm8k,math_500_chat,amc23)
    [ -n "$TASKS_OVERRIDE" ] && LM_EVAL_TASKS="$TASKS_OVERRIDE"

    EXTRA=()
    [ -n "$LIMIT" ] && EXTRA+=(--limit "$LIMIT")

    # `--apply_chat_template`: ONLY for instruct/chat ckpts. Base models like Qwen2.5-3B
    # *have* a chat_template in their tokenizer but aren't trained for it — forcing it
    # tanks HumanEval/MBPP/MMLU. Opt-in via --chat_template flag.
    [ "$CHAT_TEMPLATE" = "1" ] && EXTRA+=(--apply_chat_template)

    # HumanEval / MBPP need unsafe code flag + env var.
    export HF_ALLOW_CODE_EVAL=1

    lm_eval \
        --model vllm \
        --model_args "$VLLM_ARGS" \
        --tasks "$LM_EVAL_TASKS" \
        --include_path "$CUSTOM_TASKS_DIR" \
        --batch_size auto \
        --confirm_run_unsafe_code \
        --gen_kwargs "max_gen_toks=3072,do_sample=True,temperature=0.6,top_p=0.95" \
        --output_path "$LM_EVAL_OUT" \
        --log_samples \
        "${EXTRA[@]}"
else
    bold "Run 1/4  lm-eval-harness  [SKIPPED]"
fi

# =============================================================================
# Run 2: LiveCodeBench v6
# =============================================================================
if [ "$SKIP_LCB" = "0" ]; then
    bold "Run 2/4  LiveCodeBench v6"
    LCB_OUT="$RUN_DIR/lcb"
    mkdir -p "$LCB_OUT"
    python "$SCRIPT_DIR/external/livecodebench_runner.py" \
        --model "$MODEL" \
        ${REVISION:+--revision "$REVISION"} \
        --release_version release_v6 \
        --output "$LCB_OUT/lcb_v6.json" \
        --max_model_len "$MAX_MODEL_LEN" \
        --gpu_mem "$GPU_MEM" \
        ${LIMIT:+--limit "$LIMIT"}
else
    bold "Run 2/4  LiveCodeBench v6  [SKIPPED]"
fi

# =============================================================================
# Aggregate
# =============================================================================
bold "Aggregate → CSV"
LOCAL_CSV="$RUN_DIR/results.csv"
python "$SCRIPT_DIR/aggregate.py" \
    --run_dir "$RUN_DIR" \
    --model "$MODEL" \
    ${REVISION:+--revision "$REVISION"} \
    --out_csv "$LOCAL_CSV"

# Also append to the shared CSV if --csv was provided. aggregate.py is
# append-safe (writes header iff file is new), so this is just a 2nd call.
if [ -n "$CSV" ]; then
    mkdir -p "$(dirname "$CSV")"
    python "$SCRIPT_DIR/aggregate.py" \
        --run_dir "$RUN_DIR" \
        --model "$MODEL" \
        ${REVISION:+--revision "$REVISION"} \
        --out_csv "$CSV"
    echo "ALSO appended → $CSV"
fi

cat "$LOCAL_CSV"
echo
echo "DONE → $LOCAL_CSV"
