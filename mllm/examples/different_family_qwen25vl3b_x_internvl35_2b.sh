#!/usr/bin/env bash
# open_r1, heter co-learn, A=Qwen/Qwen2.5-VL-3B-Instruct (flash_attention_2) x B=OpenGVLab/InternVL3_5-2B-HF (flash_attention_2)
# the pinned stack (torch2.9/vllm0.11.2/tf4.57.0/ds0.18), ZeRO-3+optim-offload, 4+4 GPUs.
# Requires the environment active + MLLM_ENV_READY=1. bs/ga/vllm/steps overridable via env.
# smoke: MAX_STEPS=1 MAX_SAMPLES=64 bash examples/heter_qwen25vl7b_x_internvl35_8b_openr1.sh
# Illustrative settings, not the paper's configuration: a smaller model pair, a
# short rollout budget, and a step cap. See the paper for the reported setup.
set -euo pipefail
[ "${MLLM_ENV_READY:-0}" = "1" ] || { echo "[mllm] ERROR: env not activated. Activate the environment and 'export MLLM_ENV_READY=1' (see README)." >&2; exit 1; }
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"; REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"; cd "$REPO_ROOT"
# HF cache: local, overridable, no NAS default.
export HF_HUB_CACHE="${HF_HUB_CACHE:-$HOME/.cache/huggingface/hub}"; mkdir -p "$HF_HUB_CACHE"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-$HOME/.cache/huggingface/datasets}"; mkdir -p "$HF_DATASETS_CACHE"
# wandb: OFFLINE by default (no API key needed); `wandb sync <dir>` later to upload. We keep `--report_to wandb`
# below so metrics land in a local offline run dir; to disable tracking entirely, change it to `--report_to none`.
export WANDB_MODE="${WANDB_MODE:-offline}"
export WANDB_PROJECT="${WANDB_PROJECT:-co-rl-mllm}"
export DISABLE_MLFLOW_INTEGRATION=TRUE
# HF token: required (hf read-scope token; Gemma models also need license acceptance on their HF page).
export HF_TOKEN="${HF_TOKEN:?set HF_TOKEN (hf read-scope token; Gemma needs license acceptance)}"; export HUGGING_FACE_HUB_TOKEN="$HF_TOKEN"
# Data/eval paths: required, no NAS default (see setup/prepare_data.sh + bundled data/mathvista).
export MAX_SAMPLES="${MAX_SAMPLES:-8000}"
export MLLM_PRE_DIR="${MLLM_PRE_DIR:?set MLLM_PRE_DIR = preprocessed open_r1_8k dir (see setup/prepare_data.sh)}"
export MLLM_EVAL_PATH="${MLLM_EVAL_PATH:?set MLLM_EVAL_PATH = mathvista/testmini_150.jsonl (bundled under data/mathvista)}"
export MLLM_EVAL_IMAGE_DIR="${MLLM_EVAL_IMAGE_DIR:?set MLLM_EVAL_IMAGE_DIR = mathvista image root (data/mathvista)}"
DATASET="lmms-lab/multimodal-open-r1-8k-verified"
MODEL_A="Qwen/Qwen2.5-VL-3B-Instruct"; MODEL_B="OpenGVLab/InternVL3_5-2B-HF"
VLLM_MEM_A="${VLLM_MEM:-0.45}"; VLLM_MEM_B="${VLLM_MEM:-0.45}"
TS="$(date +%Y%m%d_%H%M%S)"; RUN="heter_qwen25vl7b_x_internvl35_8b_openr1_${TS}"
BASE_OUT="work_dirs/co-rl-mllm/$RUN"; RDV_DIR="${BASE_OUT}/rdv"
rm -rf "$RDV_DIR"; mkdir -p "$BASE_OUT/model_a" "$BASE_OUT/model_b" "$RDV_DIR"
COMMON=(
    --learning_rate 1e-6 --per_device_train_batch_size ${BS:-2} --gradient_accumulation_steps ${GA:-8}
    --train_dataset "$DATASET" --num_train_epochs 1
    --lr_scheduler_type cosine_with_min_lr --lr_scheduler_kwargs '{"min_lr_rate": 0.1}' --warmup_ratio 0.03
    --gradient_checkpointing --gradient_checkpointing_kwargs '{"use_reentrant": false}'
    --max_completion_length 512 --num_generations 4 --temperature 1.0
    --use_vllm --vllm_mode colocate --vllm_max_model_length 4096 --vllm_importance_sampling_mode token_truncate
    --logging_steps 1 --save_strategy steps --save_steps ${SAVE_STEPS:-50} --save_only_model true --save_total_limit ${SAVE_LIMIT:-100}
    --eval_strategy steps --eval_steps ${EVAL_STEPS:-50} --eval_on_start true
    --num_generations_eval 1 --per_device_eval_batch_size 1
    --adam_beta2 0.95 --beta 0 --loss_type bnpo --scale_rewards group --self_consistency_threshold 0.0
    --seed 42 --data_seed 42 --report_to wandb --wandb_project co-rl-mllm
    --rendezvous_dir "$RDV_DIR" --run_config "$RUN" --bf16 true --trust_remote_code
)
[ -n "${MAX_STEPS:-}" ] && COMMON+=(--max_steps "$MAX_STEPS")
launch_group () {
    local grp="$1" gpus="$2" my="$3" peer="$4" port="$5" out="$6" mem="$7" attn="$8"
    CUDA_VISIBLE_DEVICES="$gpus" accelerate launch --config_file trainers/accelerate_zero3_offload.yaml \
        --num_processes 2 --main_process_port "$port" --gradient_accumulation_steps ${GA:-8} \
        trainers/train_mllm_co_rl.py --group "$grp" \
        --model_name_or_path "$my" --peer_model_name_or_path "$peer" \
        --output_dir "$out" --vllm_gpu_memory_utilization "$mem" --attn_implementation "$attn" \
        "${COMMON[@]}" 2>&1 | tee -a "$out/train.log"
}
launch_group A "0,1,2,3" "$MODEL_A" "$MODEL_B" 19470 "$BASE_OUT/model_a" "$VLLM_MEM_A" "flash_attention_2" & PID_A=$!
launch_group B "4,5,6,7" "$MODEL_B" "$MODEL_A" 19471 "$BASE_OUT/model_b" "$VLLM_MEM_B" "flash_attention_2" & PID_B=$!
cleanup() { kill "$PID_A" "$PID_B" 2>/dev/null || true; }; trap cleanup EXIT INT TERM
wait -n "$PID_A" "$PID_B"; EC=$?; cleanup; wait 2>/dev/null || true; exit "$EC"
