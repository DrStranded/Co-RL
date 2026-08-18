<h1 align="center">
    Co-RL: Vision-Language Experiments
</h1>

<p align="center">
    Vision-language experiments of "Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in Multi-agent RL" on open-r1 and MMR1.
</p>

Methods: GT-Reward, TTRL, and Co-RL (Different family). Models: Qwen2.5-VL-3B
and 7B, InternVL3.5-2B and 8B, Gemma-3-4B and 12B. Representative launchers
under `examples/`, plus the four-benchmark evaluation suite under `eval/`.

## ⚙️ Configuration

```bash
pip install -r requirements.txt     # Python 3.11, CUDA 12.8
bash patches/apply_patches.sh       # three small vLLM/TRL fixes; each patch says what it does
python tools/verify.py              # asserts the stack matches the paper runs
```

The freeze is a full pip freeze of the environment behind the paper numbers
(torch 2.9.0+cu128, vllm 0.11.2, transformers 4.57.0, flash-attn 2.8.3,
patched TRL). Export `MLLM_VIT_ATTN_FIX=1` for every Qwen2.5-VL run, and
`MLLM_ENV_READY=1` once the environment is active.

Prepare models and data once:

```bash
export HF_TOKEN=...                 # Gemma models are gated
bash setup/prefetch_models.sh
bash setup/prepare_data.sh          # writes data/mllm_pre/{mmr1_8k, openr1_8k}
```

Each launcher selects its training set through `MLLM_PRE_DIR`.

## 🚀 Training

A ten-minute preflight of the whole stack:

```bash
bash examples/smoke.sh
```

Representative launchers (8 GPUs per run, 4+4 for the pair; other paper runs
swap the model, dataset, or method):

```bash
bash examples/ttrl_qwen25vl3b.sh                    # TTRL baseline
bash examples/different_family_qwen25vl3b_x_internvl35_2b.sh   # Co-RL (Different family)
```

Every run uses an effective batch of 64 prompts per step, 8 rollouts per
prompt, 1 epoch, seed 42.

## 📊 Evaluation

Score a checkpoint on MathVision, MathVerse, MathVista, and We-Math (greedy
decoding, 16k generation budget, rule-based grading, boxed prompt for every
cell; reported as the four-benchmark average):

```bash
export VLLM_WORKER_MULTIPROC_METHOD=spawn
python eval/prepare_benchmarks.py all
bash eval/run_eval_all.sh --model [checkpoint] --prompt boxed --gpu 0 --out_dir [dir]
```

For multiple choice the grader accepts the option letter or that option's
value.
