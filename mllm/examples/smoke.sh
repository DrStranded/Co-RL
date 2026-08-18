#!/usr/bin/env bash
# ── PREFLIGHT (<10 min) ─────────────────────────────────────────────
# Validates the built image end-to-end BEFORE you commit to a ~15h full run:
#   env gate -> model download/load -> flash_attention_2 (proves the FA resolver worked) ->
#   vLLM colocate generation -> 2 ZeRO-3 optimizer steps -> the eval/grade pipeline once.
# A shrunk GT-style run: MAX_STEPS=2, MAX_SAMPLES=16, tiny generations + eval.
# NOT a repro run, HP here are deliberately reduced for speed; use the real launchers for paper numbers.
#
# Usage:  export HF_TOKEN=hf_...   MLLM_ENV_READY=1
#         export MLLM_PRE_DIR=... MLLM_EVAL_PATH=... MLLM_EVAL_IMAGE_DIR=...
#         bash examples/smoke.sh
# TIP: point MLLM_EVAL_PATH at a ~16-line subset of testmini_150.jsonl to keep the eval step tiny, e.g.:
#         head -n 16 data/mathvista/testmini_150.jsonl > /tmp/mathvista_smoke.jsonl
#         export MLLM_EVAL_PATH=/tmp/mathvista_smoke.jsonl MLLM_EVAL_IMAGE_DIR=data/mathvista
set -euo pipefail
[ "${MLLM_ENV_READY:-0}" = "1" ] || { echo "[mllm] ERROR: env not activated. Activate the environment and 'export MLLM_ENV_READY=1' (see README)." >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"
# HF cache: local, overridable, no NAS default.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"; mkdir -p "$HF_HUB_CACHE"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"; mkdir -p "$HF_DATASETS_CACHE"
# wandb: OFFLINE (a preflight should never touch the network for tracking). --report_to none below = no run created.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-co-rl-mllm}"
export DISABLE_MLFLOW_INTEGRATION=TRUE
# HF token: required (hf read-scope token). Qwen2.5-VL is ungated, so a plain read token suffices for the smoke.
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN (hf read-scope token)}"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# Data/eval paths: required, no NAS default (see setup/prepare_data.sh + bundled data/mathvista).
export MAX_SAMPLES="${MAX_SAMPLES:-16}"
export MLLM_PRE_DIR="${MLLM_PRE_DIR:?set MLLM_PRE_DIR = preprocessed open_r1_8k dir (see setup/prepare_data.sh)}"
export MLLM_EVAL_PATH="${MLLM_EVAL_PATH:?set MLLM_EVAL_PATH = mathvista jsonl (a ~16-line subset is ideal for the smoke)}"
export MLLM_EVAL_IMAGE_DIR="${MLLM_EVAL_IMAGE_DIR:?set MLLM_EVAL_IMAGE_DIR = mathvista image root (data/mathvista)}"
DATASET="lmms-lab/multimodal-open-r1-8k-verified"
MODEL="${MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
TS="$(date +%Y%m%d_%H%M%S)"; RUN="smoke_qwen25vl7b_gt_${TS}"
BASE_OUT="work_dirs/co-rl-mllm/$RUN"; mkdir -p "$BASE_OUT"
# Reduced-for-speed knobs (override if needed). max_steps=2, tiny group + short completions.
export MAX_STEPS="${MAX_STEPS:-2}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file trainers/accelerate_zero3_offload.yaml \
    --num_processes ${NUM_PROC:-8} --main_process_port 19468 --gradient_accumulation_steps ${GA:-1} \
    trainers/train_mllm_single.py \
    --model_name_or_path "$MODEL" --train_dataset "$DATASET" \
    --output_dir "$BASE_OUT" --run_config "$RUN" \
    --learning_rate 1e-6 \
    --per_device_train_batch_size ${BS:-1} --gradient_accumulation_steps ${GA:-1} \
    --num_train_epochs 1 --max_steps ${MAX_STEPS} \
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' --warmup_ratio 0.03 \
    --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length ${SMOKE_MAXLEN:-256} --num_generations ${NUM_GEN:-4} --temperature 1.0 \
    --use_vllm --vllm_mode colocate --vllm_max_model_length 4096 \
    --vllm_gpu_memory_utilization ${VLLM_MEM:-0.55} --vllm_importance_sampling_mode token_truncate \
    --logging_steps 1 --save_strategy steps --save_steps ${SAVE_STEPS:-2} \
    --save_only_model true \
    --save_total_limit 1 \
    --eval_strategy steps --eval_steps ${EVAL_STEPS:-2} --eval_on_start true \
    --num_generations_eval 1 --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 --beta 0 --loss_type bnpo --scale_rewards group \
    --seed 42 --data_seed 42 --report_to none \
    --attn_implementation flash_attention_2 --trust_remote_code --bf16 true 2>&1 | tee -a "$BASE_OUT/train.log"
echo "[mllm] SMOKE PASSED, image validated (env, FA2, vLLM colocate, ZeRO-3 step, eval/grade). Safe to launch a full training run."
