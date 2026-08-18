#!/usr/bin/env bash
# open_r1 / mmr1, qwen25vl3b, Qwen/Qwen2.5-VL-3B-Instruct, ttrl (unmaj, self-label majority-vote), the pinned stack (torch2.9/vllm0.11.2/tf4.57.0/ds0.18), ZeRO-3 (no offload).
# Requires the environment active + MLLM_ENV_READY=1.
# NOTE small tier: ZeRO-3 WITHOUT optimizer offload, EB=64 via bs=1 x ga=8 x 8gpu, vllm_mem 0.45, attn=flash_attention_2.
# Dataset is chosen by MLLM_PRE_DIR (openr1_8k vs mmr1_8k) -- see README.
# smoke: MAX_STEPS=1 MAX_SAMPLES=64 bash examples/openr1_qwen25vl3b_ttrl.sh
# Illustrative settings, not the paper's configuration: a smaller model pair, a
# short rollout budget, and a step cap. See the paper for the reported setup.
set -euo pipefail
[ "${MLLM_ENV_READY:-0}" = "1" ] || { echo "[mllm] ERROR: env not activated. Activate the environment and 'export MLLM_ENV_READY=1' (see README)." >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"
# HF cache: local, overridable, no NAS default.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"; mkdir -p "$HF_HUB_CACHE"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"; mkdir -p "$HF_DATASETS_CACHE"
# wandb: OFFLINE by default (no API key needed); `wandb sync <dir>` later to upload.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-co-rl-mllm}"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN (hf read-scope token)}"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# Data/eval paths: required, no NAS default (see setup/prepare_data.sh + bundled data/mathvista).
export MAX_SAMPLES="${MAX_SAMPLES:-8000}"
export MLLM_PRE_DIR="${MLLM_PRE_DIR:?set MLLM_PRE_DIR = preprocessed openr1_8k or mmr1_8k dir (see setup/prepare_data.sh)}"
export MLLM_EVAL_PATH="${MLLM_EVAL_PATH:?set MLLM_EVAL_PATH = mathvista/testmini_150.jsonl (bundled under data/mathvista)}"
export MLLM_EVAL_IMAGE_DIR="${MLLM_EVAL_IMAGE_DIR:?set MLLM_EVAL_IMAGE_DIR = mathvista image root (data/mathvista)}"
DATASET="lmms-lab/multimodal-open-r1-8k-verified"
MODEL="Qwen/Qwen2.5-VL-3B-Instruct"
VLLM_MEM="${VLLM_MEM:-0.45}"
TS="$(date +%Y%m%d_%H%M%S)"; RUN="openr1_qwen25vl3b_ttrl_${TS}"
BASE_OUT="${MLLM_OUT_ROOT:-work_dirs}/co-rl-mllm/$RUN"; mkdir -p "$BASE_OUT"
MAXSTEPS_ARG="--max_steps ${MAX_STEPS:-50}"   # example run; unset for a full epoch
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7}" accelerate launch \
    --config_file trainers/accelerate_zero3.yaml \
    --num_processes ${NUM_PROC:-8} --main_process_port 19412 --gradient_accumulation_steps ${GA:-8} \
    trainers/train_mllm_single.py \
    --model_name_or_path "$MODEL" --train_dataset "$DATASET" \
    --output_dir "$BASE_OUT" --run_config "$RUN" \
    --learning_rate 1e-6 \
    --per_device_train_batch_size ${BS:-1} --gradient_accumulation_steps ${GA:-8} \
    --num_train_epochs 1 ${MAXSTEPS_ARG} \
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' --warmup_ratio 0.03 \
    --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}' \
    --max_completion_length 512 --num_generations 4 --temperature 1.0 \
    --use_vllm --vllm_mode colocate --vllm_max_model_length 4096 \
    --vllm_gpu_memory_utilization "$VLLM_MEM" --vllm_importance_sampling_mode token_truncate \
    --logging_steps 1 --save_strategy steps --save_steps ${SAVE_STEPS:-20} \
    --save_only_model true \
    --save_total_limit 100 \
    --eval_strategy steps --eval_steps ${EVAL_STEPS:-20} --eval_on_start true \
    --num_generations_eval 1 --per_device_eval_batch_size 1 \
    --adam_beta2 0.95 --beta 0 --loss_type bnpo --scale_rewards group \
    --self_labeling --self_consistency_threshold 0.0 \
    --seed 42 --data_seed 42 --report_to wandb --wandb_project co-rl-mllm \
    --attn_implementation flash_attention_2 --bf16 true 2>&1 | tee -a "$BASE_OUT/train.log"
