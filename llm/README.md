<h1 align="center">
    Co-RL: Text-LLM Experiments
</h1>

<p align="center">
    Text-LLM experiments of "Co-RL: Unsupervised Reasoning Emerges from Diverse Cohort in Multi-agent RL" on MATH-Level345.
</p>

Methods: GT-Reward, TTRL, RENT, Intuitor, and Co-RL in three settings:
Co-RL (Same family), Co-RL (Different family), and Co-RL (Different
family+) — the last adds data decoupling through rephrased training prompts. Models: Qwen2.5-3B and 7B,
Llama-3.2-3B, Llama-3.1-8B, Qwen3-1.7B. Representative launchers under
`examples/`.

## ⚙️ Configuration

```bash
pip install -r requirements.txt      # Python 3.12, CUDA 12.8
python tools/verify.py               # asserts the stack matches the paper runs
```

`requirements.txt` is a full pip freeze of the environment behind the paper
numbers (torch 2.9.0+cu128, vllm 0.11.2, transformers 4.57.0, deepspeed
0.18.0, patched TRL). The three-agent runs use a newer stack, frozen
separately in `requirements-n3.txt`.

Training prompts load from the HuggingFace dataset `q1716523669/MATH-Level345`.
The MATH-500 evaluation set is bundled at `data/math500/test.json`.

## 🚀 Training

All launchers need `HF_TOKEN` for gated models (Llama). A one-step smoke:

```bash
MAX_STEPS=1 HF_TOKEN=... bash examples/ttrl_qwen25_1p5b.sh
```

Representative launchers (8 GPUs; other paper runs swap the model or method):

```bash
bash examples/ttrl_qwen25_1p5b.sh          # TTRL baseline
bash examples/same_family_qwen25_1p5b.sh                  # Co-RL (Same family)
bash examples/different_family_qwen25_1p5b_x_llama32_1b.sh    # Co-RL (Different family)
```

Co-RL (Different family+) reuses the heterogeneous launcher with the
DeepSeek-rephrased dataset selected for one agent. `examples/n3_ring_qwen25_1p5b_x_llama32_1b_x_qwen3_0p6b.sh` runs the
three-agent setting, where votes pass along a directed ring.

## 📊 Evaluation

In-loop MATH-500 accuracy is logged every 10 steps and selects the best
checkpoint. Final numbers use the seven-benchmark suite (GSM8K, MATH-500,
AMC23 avg@8, HumanEval, MBPP, GPQA-Diamond, LiveCodeBench v6; T=0.6/top-p
0.95, last-`\boxed{}` extraction, chat template for chat-lineage models):

The evaluation harnesses live in their own environment: `math-verify` is
required there to grade AMC and MATH-500, and must not be installed alongside
training (it pulls a conflicting antlr4).

```bash
bash eval/setup_external.sh        # once: clone + patch the external harnesses
bash eval/run_eval_all.sh --model [checkpoint] --chat_template --out_dir [dir]
```
