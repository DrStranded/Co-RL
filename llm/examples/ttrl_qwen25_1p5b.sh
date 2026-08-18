#!/usr/bin/env bash
# MATH-Level345, qwen25_1p5b, Qwen/Qwen2.5-1.5B, TTRL (self-label majority vote, no ground truth)
# entry: trainers/self_rewarding/train_un_grpo.py
# Illustrative settings, not the paper's configuration: a small model pair, a
# short rollout budget, and a step cap so the run finishes quickly on 4 GPUs.
# The configuration behind the reported numbers is given in the paper.
# Requires the environment active + LLM_ENV_READY=1.
# smoke: MAX_STEPS=1 bash examples/math345_qwen25_1p5b_ttrl.sh
set -euo pipefail
[ "${LLM_ENV_READY:-0}" = "1" ] || { echo "[llm] ERROR: env not activated (see README)." >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"; mkdir -p "$HF_HUB_CACHE"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"; mkdir -p "$HF_DATASETS_CACHE"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-co-learning}"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN (hf read-scope token)}"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
DATASET="${DATASET:-q1716523669/MATH-Level345}"
MODEL="Qwen/Qwen2.5-1.5B"
VLLM_MEM="${VLLM_MEM:-0.45}"
TS="$(date +%Y%m%d_%H%M%S)"; RUN="math345_qwen25_1p5b_ttrl_${TS}"
BASE_OUT="${LLM_OUT_ROOT:-work_dirs}/co-learning/$RUN"; mkdir -p "$BASE_OUT"
MAXSTEPS_ARG="--max_steps ${MAX_STEPS:-50}"   # example run; unset for a full epoch
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file trainers/accelerate_zero3.yaml \
    --num_processes ${NUM_PROC:-8} --main_process_port 20402 --gradient_accumulation_steps ${GA:-64} \
    trainers/self_rewarding/train_un_grpo.py  \
    --model_name_or_path "$MODEL" --train_dataset "$DATASET" \
    --output_dir "$BASE_OUT" --run_config "$RUN" \
    --learning_rate 1e-6 \
    --per_device_train_batch_size ${BS:-3} --gradient_accumulation_steps ${GA:-64} \
    --num_train_epochs 1 ${MAXSTEPS_ARG} \
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' --warmup_ratio 0.03 \
    --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length 1024 --num_generations 8 --temperature 1.0 \
    --use_vllm --vllm_mode colocate --vllm_gpu_memory_utilization "$VLLM_MEM" \
    --vllm_importance_sampling_mode token_truncate \
    --logging_steps 1 --save_strategy steps --save_steps ${SAVE_STEPS:-10} --save_only_model true \
    --eval_strategy steps --eval_steps ${EVAL_STEPS:-10} --eval_on_start true \
    --num_generations_eval 1 --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 --beta 0 --loss_type bnpo --scale_rewards group \
    --seed 42 --data_seed 42 --report_to wandb \
    --bf16 true 2>&1 | tee -a "$BASE_OUT/train.log"
