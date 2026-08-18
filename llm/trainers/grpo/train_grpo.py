import os
import json
import shutil
import sys
import wandb

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# Shared modules (`dataset`, `verifiers`, `comas`) live in trainers/co_rl; add it
# to the path so `from comas.code_reward import passes_tests` resolves (shared
# code-exec utility, not duplicated).
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "co_rl"))
from verifiers.qwen.qwen_math_parser import extract_answer
from verifiers.qwen.math_grade import grade_answer

import torch.nn as _nn
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel as _PreTrainedModel
from dataset import DAPO_DATASET, OPSD_DATASET, MATH_LEVEL345_DATASET, MATH_LEVEL12345_DATASET, COREWARDING_MATH_ORIGINAL, COREWARDING_MATH_REPHRASED, COMAS_BLENDED, COMAS_MATH, load_dataset

# Gemma-3 + ZeRO-3 fix: _init_weights zeroes weight[padding_idx] for nn.Embedding,
# Under ZeRO-3 non-rank-0 sees size-0 shards -> IndexError. Gemma-3 sets padding_idx and triggers it.
_orig_init_weights = _PreTrainedModel._init_weights

def _safe_init_weights(self, module):
    if isinstance(module, _nn.Embedding) and module.weight.data.numel() == 0:
        return
    return _orig_init_weights(self, module)

_PreTrainedModel._init_weights = _safe_init_weights

from trl import (
    GRPOTrainer,
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)
from dataclasses import dataclass, field



@dataclass
class CustomScriptArguments(ScriptArguments):
    """Extended script arguments with GRPO-specific options."""

    run_config: str = field(
        default=None,
        metadata={
            "help": "Run name for this experiment. Will be used for both the output directory "
            "(appended to output_dir) and WandB run name. If not specified, will generate "
            "automatic name based on hyperparameters."
        },
    )
    wandb_entity: str = field(
        default=None,
        metadata={"help": "WandB entity (username or team name) to log runs under."},
    )
    wandb_project: str = field(
        default="grpo-training",
        metadata={"help": "WandB project name to log runs under."},
    )
    train_dataset: str = field(
        default=OPSD_DATASET,
        metadata={
            "help": "Dataset to use for GRPO training.",
            "choices": [OPSD_DATASET, DAPO_DATASET, MATH_LEVEL345_DATASET, MATH_LEVEL12345_DATASET, COREWARDING_MATH_ORIGINAL, COREWARDING_MATH_REPHRASED, COMAS_BLENDED, COMAS_MATH],
        },
    )


def extract_boxed_answer(text):
    """Extract the math answer from a model completion.

    Returns the answer string (e.g. ``\\frac{1}{2}``, ``42``) or ``None`` if no
    parseable answer is found. Backed by Qwen2.5-Math's parser, which handles
    \\boxed{}, "answer is X", ``\\boxed`` without braces, and many fallbacks.
    """
    return extract_answer(text, "math")


def _get_text(completion):
    # TRL wraps completions as [{"role": "assistant", "content": "..."}] for conversational prompts
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def reward_correctness(completions, solution, **kwargs):
    """Reward function: 1.0 if completion's parsed answer is sympy-equivalent to
    the (peer-supplied or ground-truth) solution, else 0.0.

    `solution` here can be:
      - train mode: peer's pseudo-label (from majority vote), possibly the
        sentinel `_UNLABELED_SENTINEL` for prompts the peer dropped, sentinel
        cannot match any parsed answer, so reward is 0 for those.
      - eval mode: dataset's real ground-truth solution (eval branch in trainer
        skips the cross-labeling override).

    Uses qwen's `grade_answer` (sympy + latex2sympy2) so equivalent forms like
    `1/2` vs `\\frac{1}{2}` vs `0.5` all count as correct. Slower than string
    equality (~10-100ms per check) but eliminates spurious negative rewards.

    CoMAS task routing (gt baseline): `task`=="coding" -> run the asserts in the
    persistent `test_code` column (completion must pass them, supervised, uses
    the asserts' expected outputs); math/science -> the boxed+sympy path above.
    """
    tasks = kwargs.get("task")
    test_codes = kwargs.get("test_code")
    n = len(completions)
    tasks = tasks if tasks is not None else ["math"] * n
    test_codes = test_codes if test_codes is not None else [""] * n

    rewards = []
    for i, (completion, ground_truth) in enumerate(zip(completions, solution)):
        if tasks[i] == "coding":
            from comas.code_reward import passes_tests
            rewards.append(1.0 if passes_tests(_get_text(completion), test_codes[i]) else 0.0)
        else:
            pred_answer = extract_boxed_answer(_get_text(completion))
            if pred_answer is not None and grade_answer(pred_answer, ground_truth):
                rewards.append(1.0)
            else:
                rewards.append(0.0)
    return rewards



from transformers.trainer_callback import TrainerCallback


class BestKeeperCallback(TrainerCallback):
    """DeepSpeed-compatible substitute for `load_best_model_at_end=True`,
    which HF Trainer rejects (trainer.py:5547) when combined with
    DeepSpeed/FSDP + `save_only_model=True`. On every save, if the latest
    eval metric beat the prior best, hardlink the just-written checkpoint
    to `$output_dir/best_model/` (0 byte / 0 time via inode refcount;
    survives ring-buffer deletion of the source ckpt).
    """

    def __init__(self, metric_name="eval_reward", greater_is_better=True):
        self.metric_name = metric_name
        self.greater_is_better = greater_is_better
        self.best = None
        self.last_metrics = {}

    def on_evaluate(self, args, state, control, metrics=None, **kw):
        if metrics:
            self.last_metrics = metrics

    def on_save(self, args, state, control, **kw):
        if not state.is_world_process_zero:
            return
        v = self.last_metrics.get(self.metric_name)
        if v is None:
            # TRL merges reward metrics into the logged dict (state.log_history),
            # not into the metrics handed to on_evaluate; fall back to the most
            # recent logged value so best-ckpt tracking actually fires.
            for entry in reversed(state.log_history):
                if self.metric_name in entry:
                    v = entry[self.metric_name]
                    break
        if v is None:
            return
        better = self.best is None or (
            (v > self.best) if self.greater_is_better else (v < self.best)
        )
        if not better:
            return
        self.best = v
        src = os.path.join(args.output_dir, f"checkpoint-{state.global_step}")
        dst = os.path.join(args.output_dir, "best_model")
        if not os.path.exists(src):
            return
        if os.path.exists(dst):
            shutil.rmtree(dst)
        shutil.copytree(src, dst, copy_function=os.link)
        with open(os.path.join(args.output_dir, "best_metric.json"), "w") as f:
            json.dump(
                {"step": state.global_step, "metric": self.metric_name, "value": float(v)},
                f, indent=2,
            )


if __name__ == "__main__":
    parser = TrlParser((CustomScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # WandB Run Name & Output Directory
    ################
    # Format learning rate (e.g., 2e-5 -> "2e-5")
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")

    # Get number of processes from environment (set by accelerate launch)
    num_processes = int(os.environ.get("WORLD_SIZE", 1))

    # Calculate effective batch size
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    # Use custom run_config if provided, otherwise generate automatic name
    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_lr{lr_str}_bs{effective_batch_size}"
        # Append run_config to output_dir if it doesn't already end with it
        if not training_args.output_dir.endswith(script_args.run_config):
            from pathlib import Path

            training_args.output_dir = str(Path(training_args.output_dir) / script_args.run_config)
    else:
        # Extract model name from path
        model_name = model_args.model_name_or_path.split("/")[-1]

        # Create concise run name
        full_wandb_run_name = (
            f"GRPO_{model_name}_"
            f"lr{lr_str}_"
            f"bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_"
            f"temp{training_args.temperature}"
        )

    # Print configuration info
    print(f"\n{'='*80}")
    print(f"RUN CONFIGURATION")
    print(f"{'='*80}")
    print(f"WandB Run Name: {full_wandb_run_name}")
    print(f"Output Directory: {training_args.output_dir}")
    print(f"Num Generations: {training_args.num_generations}")
    print(f"Temperature: {training_args.temperature}")
    print(f"Max Completion Length: {training_args.max_completion_length}")
    print(f"{'='*80}\n")

    ################
    # WandB Initialization
    ################
    # Only initialize wandb on main process (LOCAL_RANK 0 or not set)
    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "model_name": model_args.model_name_or_path,
                "learning_rate": training_args.learning_rate,
                "per_device_train_batch_size": training_args.per_device_train_batch_size,
                "gradient_accumulation_steps": training_args.gradient_accumulation_steps,
                "effective_batch_size": effective_batch_size,
                "num_train_epochs": training_args.num_train_epochs,
                "num_generations": training_args.num_generations,
                "max_completion_length": training_args.max_completion_length,
                "temperature": training_args.temperature,
                "beta": training_args.beta,
                "use_peft": model_args.use_peft,
                "lora_r": model_args.lora_r if model_args.use_peft else None,
                "lora_alpha": model_args.lora_alpha if model_args.use_peft else None,
                "gradient_checkpointing": training_args.gradient_checkpointing,
                "num_processes": num_processes,
                "loss_type": training_args.loss_type,
                "scale_rewards": training_args.scale_rewards,
                "steps_per_generation": training_args.steps_per_generation,
                "vllm_importance_sampling_correction": training_args.vllm_importance_sampling_correction,
                "adam_beta2": training_args.adam_beta2,
                "lr_scheduler_type": training_args.lr_scheduler_type,
                "lr_scheduler_kwargs": training_args.lr_scheduler_kwargs,
                "warmup_ratio": training_args.warmup_ratio,
                "max_grad_norm": training_args.max_grad_norm,
                "weight_decay": training_args.weight_decay,
                "eval_steps": training_args.eval_steps,
                "num_generations_eval": training_args.num_generations_eval,
                "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
                "seed": training_args.seed,
                "data_seed": training_args.data_seed,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    # Determine dtype
    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16,
                "bf16": torch.bfloat16,
                "float16": torch.float16,
                "fp16": torch.float16,
                "float32": torch.float32,
                "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

    print(f"\n{'='*80}")
    print(f"Loading model with dtype: {model_dtype}")
    print(f"Using attention implementation: {model_args.attn_implementation or 'flash_attention_2'}")
    print(f"{'='*80}\n")

    model_kwargs = dict(
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        attn_implementation=model_args.attn_implementation or "flash_attention_2",
        torch_dtype=model_dtype,
    )

    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config

    training_args.model_init_kwargs = model_kwargs

    tokenizer = AutoTokenizer.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    ################
    # Dataset
    ################
    # Load and preprocess dataset
    train_dataset, eval_dataset = load_dataset(script_args.train_dataset)

    ################
    # Training
    ################
    trainer = GRPOTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_correctness,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=get_peft_config(model_args),
    )

    trainer.add_callback(BestKeeperCallback())

    trainer.train()

    # Save model
    trainer.save_model(training_args.output_dir)
