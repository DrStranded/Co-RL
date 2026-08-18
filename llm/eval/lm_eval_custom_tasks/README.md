# lm-eval custom tasks

Custom tasks that aren't in lm-eval-harness main:

| Task        | Dataset                              | N   | Grader                                  |
|-------------|--------------------------------------|----:|-----------------------------------------|
| `amc23`     | `AI-MO/aimo-validation-amc`          |  83 | `math_verify.verify` (latex/sympy), avg@8 |

## Usage

Always pass `--include_path llm/eval/lm_eval_custom_tasks` so lm-eval can discover the yamls:

```bash
lm_eval \
    --model vllm \
    --model_args "pretrained=$MODEL,dtype=bfloat16,gpu_memory_utilization=0.9" \
    --tasks amc23 \
    --include_path llm/eval/lm_eval_custom_tasks \
    --batch_size auto \
    --output_path /tmp/out
```

## Notes on grading

- We extract the **last** `\boxed{...}` from the completion. CoT chains commonly emit several
  intermediate boxes — only the last is graded.
- AMC: `math_verify.parse + verify`; falls back to float / string compare on parse failure.
- Both graders are case-/whitespace-tolerant on the string fallback.

## Reproducibility caveats (paper)

- Datasets are public on HF — anyone running `setup.sh` then `run_eval_all.sh` reproduces.
