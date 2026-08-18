#!/usr/bin/env bash
# MATH-Level345, heter_qwen25_3b_x_llama32_3b, Qwen2.5-1.5B x Llama-3.2-1B-Instruct, Co-RL (Different family)
# entry: trainers/co_rl/train_co_rl.py
# Illustrative settings, not the paper's configuration: a small model pair, a
# short rollout budget, and a step cap so the run finishes quickly on 4 GPUs.
# The configuration behind the reported numbers is given in the paper.
# Requires the environment active + LLM_ENV_READY=1. Llama is gated: accept its HF license first.
# smoke: MAX_STEPS=1 bash examples/heter_qwen25_3b_x_llama32_3b.sh
set -euo pipefail
[ "${LLM_ENV_READY:-0}" = "1" ] || { echo "[llm] ERROR: env not activated (see README)." >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"; mkdir -p "$HF_HUB_CACHE"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"; mkdir -p "$HF_DATASETS_CACHE"
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-co-learning}"
export DISABLE_MLFLOW_INTEGRATION=TRUE
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN (hf read-scope token; gated models also need license acceptance)}"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
DATASET="${DATASET:-q1716523669/MATH-Level345}"
MODEL_A="${MODEL_A:-Qwen/Qwen2.5-1.5B}"
MODEL_B="${MODEL_B:-meta-llama/Llama-3.2-1B-Instruct}"
VLLM_MEM="${VLLM_MEM:-0.45}"
TS="$(date +%Y%m%d_%H%M%S)"; RUN="different_family_${TS}"
BASE_OUT="${LLM_OUT_ROOT:-work_dirs}/co-learning/$RUN"; RDV_DIR="$BASE_OUT/rdv"; mkdir -p "$BASE_OUT" "$RDV_DIR"
MAXSTEPS_ARG="--max_steps ${MAX_STEPS:-50}"   # example run; unset for a full epoch
COMMON=(--train_dataset "$DATASET" --learning_rate 1e-6
    --per_device_train_batch_size ${BS:-2} --gradient_accumulation_steps ${GA:-32}
    --num_train_epochs 1 ${MAXSTEPS_ARG}
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' --warmup_ratio 0.03
    --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --max_completion_length 1024 --num_generations 8 --temperature 1.0
    --use_vllm --vllm_mode colocate --vllm_importance_sampling_mode token_truncate
    --logging_steps 1 --save_strategy steps --save_steps ${SAVE_STEPS:-10} --save_only_model true
    --eval_strategy steps --eval_steps ${EVAL_STEPS:-10} --eval_on_start true
    --num_generations_eval 1 --per_device_eval_batch_size 1
    --adam_beta2 0.95 --beta 0 --loss_type bnpo --scale_rewards group --self_consistency_threshold 0.0
    --seed 42 --data_seed 42 --report_to wandb --bf16 true
    --rendezvous_dir "$RDV_DIR" --run_config "$RUN")
launch_group () {  # group gpus my_model peer_model port outdir
    local grp="$1" gpus="$2" my="$3" peer="$4" port="$5" out="$6"
    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch --config_file trainers/accelerate_zero3.yaml \
        --num_processes 2 --main_process_port "$port" --gradient_accumulation_steps ${GA:-32} \
        trainers/co_rl/train_co_rl.py --group "$grp" \
        --model_name_or_path "$my" --peer_model_name_or_path "$peer" \
        --output_dir "$out" --vllm_gpu_memory_utilization "$VLLM_MEM" \
        "${COMMON[@]}" 2>&1 | tee -a "$out/train.log"
}
mkdir -p "$BASE_OUT/model_a" "$BASE_OUT/model_b"
launch_group A "0,1" "$MODEL_A" "$MODEL_B" 20417 "$BASE_OUT/model_a" & PID_A=$!
launch_group B "2,3" "$MODEL_B" "$MODEL_A" 20418 "$BASE_OUT/model_b" & PID_B=$!
trap 'kill $PID_A $PID_B 2>/dev/null || true' INT TERM
wait $PID_A; RC_A=$?; wait $PID_B; RC_B=$?
echo "[llm] group A rc=$RC_A  group B rc=$RC_B"; exit $(( RC_A || RC_B ))
