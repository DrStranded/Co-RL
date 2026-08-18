"""Entry point for one half of a cross-supervised GRPO data-parallel run.

Launched twice in parallel (once per group) by `run_co_dp_lora.sh`. Each launch
is an independent accelerate world bound to its own CUDA_VISIBLE_DEVICES and
master port. The two launches coordinate solely through the file rendezvous
directory (`--rendezvous_dir`) to exchange pseudo-labels every generation step.
"""

import os
import json
import shutil
import datetime as _dt
from dataclasses import dataclass, field

import wandb
import torch.distributed as _td
import torch.nn as _nn
from transformers import AutoTokenizer
from transformers.modeling_utils import PreTrainedModel as _PreTrainedModel

from co_rl_trainer import CoRLTrainer
from co_label_utils import extract_boxed_answer, grade_answer
from dataset import DAPO_DATASET, MATH_LEVEL12345_DATASET, MATH_LEVEL345_DATASET, OPSD_DATASET, load_dataset
from rendezvous import Rendezvous

# Gemma-3 + ZeRO-3 fix: _init_weights zeroes weight[padding_idx] for nn.Embedding,
# Under ZeRO-3 non-rank-0 sees size-0 shards -> IndexError. Gemma-3 sets padding_idx and triggers it.
_orig_init_weights = _PreTrainedModel._init_weights

def _safe_init_weights(self, module):
    if isinstance(module, _nn.Embedding) and module.weight.data.numel() == 0:
        return
    return _orig_init_weights(self, module)

_PreTrainedModel._init_weights = _safe_init_weights

# NCCL watchdog: PyTorch default collective timeout is 30 min.
# In co_rl, rank 0 (main process) blocks in `rendezvous.exchange_n_way()`
# polling peer-group files between train steps; meanwhile rank 1 is parked on
# the in-group `broadcast_object_list` (an NCCL collective) waiting for rank 0
# to reach it. If peer groups stagger their eval (Gemma-3-4B eval ≈ 2x Qwen 3B
# eval at G=1/250 prompts), rank 0's file wait can exceed 30 min, and rank 1's
# NCCL broadcast watchdog kills the whole group.
# Set via monkey-patch because neither `accelerate launch` (1.12) nor the YAML
# expose a default-timeout knob, and PyTorch 2.9 has no env-var override for
# `default_pg_timeout`. Kept at the PyTorch default (30 min) so a genuinely hung
# run fails fast rather than wasting hours; if the rdv-wait > 30 min crash
# (2026-05-26 group B) reappears, raise this instead of masking it.
_NCCL_PG_TIMEOUT = _dt.timedelta(minutes=30)
_orig_init_pg = _td.init_process_group


def _patched_init_pg(*args, **kwargs):
    kwargs.setdefault("timeout", _NCCL_PG_TIMEOUT)
    return _orig_init_pg(*args, **kwargs)


_td.init_process_group = _patched_init_pg

from trl import (
    GRPOConfig,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class CoRLScriptArguments(ScriptArguments):
    """Script arguments for co_rl (single-model, one-group-per-launch)."""

    group: str = field(
        default=None,
        metadata={"help": "Single uppercase letter (A/B/C/...) identifying this launch's group."},
    )
    peers: str = field(
        default=None,
        metadata={
            "help": "Comma-separated list of peer group names for N-way (N≥3) co-learning, "
            "e.g. 'B,C' when --group A. If omitted, defaults to the legacy 2-way pair "
            "(A↔B)."
        },
    )
    rendezvous_dir: str = field(
        default=None,
        metadata={"help": "Directory shared between groups for pseudo-label exchange."},
    )
    peer_model_name_or_path: str = field(
        default=None,
        metadata={
            "help": "Peer group's model id (for logging only; peer is launched separately). "
            "For N-way pass a comma-separated list, e.g. 'meta-llama/Llama-3.2-3B,google/gemma-3-4b-it'."
        },
    )
    run_config: str = field(
        default=None,
        metadata={"help": "Run name prefix for this experiment."},
    )
    wandb_entity: str = field(default=None, metadata={"help": "WandB entity."})
    wandb_project: str = field(default="co_rl", metadata={"help": "WandB project name."})
    train_dataset: str = field(
        default=OPSD_DATASET,
        metadata={
            "help": "Dataset to use for training. Group A always uses this. "
            "Group B (and N-way C/D/...) also use this UNLESS "
            "--train_dataset_per_group is set for that group's letter.",
        },
    )
    train_dataset_per_group: str = field(
        default=None,
        metadata={
            "help": "Optional per-group dataset override for data-side cross-view "
            "(Co-rewarding-I replication on co_rl infra). "
            "Format: 'B=coreward/math_rephrased' or 'B=coreward/math_rephrased,C=...' "
            "(comma-separated 'GROUP=dataset' pairs). Group letters NOT listed fall "
            "back to --train_dataset. Group A is always --train_dataset (cannot be "
            "overridden via this flag, pass --train_dataset for A). For Co-I "
            "replication: --train_dataset coreward/math_original "
            "--train_dataset_per_group B=coreward/math_rephrased. Row-index "
            "alignment between paired parquets is REQUIRED (same underlying "
            "problem at position i in both datasets), rendezvous payload[i] "
            "carries the MV of model-i's view of problem-i."
        },
    )
    self_consistency_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Minimum top-answer frequency for a pseudo-label to be accepted. "
            "0.0 accepts the plurality winner; 0.5 requires a strict majority."
        },
    )
    log_oracle_accuracy: bool = field(
        default=True,
        metadata={"help": "Log how often pseudo-labels match real ground truth (diagnostic only)."},
    )


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
    """
    # Per-example task routing (CoMAS coding data). `task`/`test_code` arrive as
    # reward kwargs (dataset columns). For coding, `ground_truth` is the peer's
    # pseudo-label = an output-tuple string, so we re-run THIS completion's code on
    # the same inputs (parsed from the persistent `test_code` column) and compare
    # output tuples by equality. Math/science keep the exact original sympy path.
    tasks = kwargs.get("task")
    test_codes = kwargs.get("test_code")
    n = len(completions)
    tasks = tasks if tasks is not None else ["math"] * n
    test_codes = test_codes if test_codes is not None else [""] * n

    rewards = []
    for i, (completion, ground_truth) in enumerate(zip(completions, solution)):
        # r = 1[extracted answer == the peer's voted answer], 0 otherwise.
        if tasks[i] == "coding":
            from comas.code_reward import extract_calls, voting_answer
            fn, call_inputs = extract_calls(test_codes[i])
            my_answer = voting_answer(_get_text(completion), fn, call_inputs)
            hit = my_answer is not None and my_answer == ground_truth
        else:
            pred_answer = extract_boxed_answer(_get_text(completion))
            hit = pred_answer is not None and grade_answer(pred_answer, ground_truth)
        rewards.append(1.0 if hit else 0.0)
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
    parser = TrlParser((CoRLScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    # `script_args.group` is one of {'A', 'B', 'C', ...}. For 2-way the legacy
    # rule (must be in {A,B}) holds; for N-way (N≥3) any single uppercase letter is OK.
    if not (isinstance(script_args.group, str) and len(script_args.group) == 1 and script_args.group.isupper()):
        raise ValueError(
            f"--group must be a single uppercase letter (A, B, C, ...); got {script_args.group!r}"
        )

    # `--peers` is the comma-separated cohort in ring order, excluding this
    # group, e.g. `--group A --peers B,C` at N=3: A is supervised by B, B by C,
    # C by A. With two agents it defaults to the other group.
    if script_args.peers:
        peer_groups = [p.strip() for p in script_args.peers.split(",") if p.strip()]
    else:
        if script_args.group not in ("A", "B"):
            raise ValueError(
                f"For N>2 launches you must pass --peers explicitly. "
                f"Got --group {script_args.group!r} with no --peers."
            )
        peer_groups = ["B" if script_args.group == "A" else "A"]
    if script_args.group in peer_groups:
        raise ValueError(
            f"--group {script_args.group!r} must not appear in --peers {peer_groups!r}"
        )

    # Each group uses an offset `seed` so the groups' vLLM/torch RNG diverge.
    # Without this, all groups' accelerate worlds set torch.manual_seed(seed +
    # process_index) with identical (seed, process_index) pairs, producing byte-
    # identical vLLM rollouts and forcing peer_agreement -> 1 (cross-supervision
    # degenerates into self-vote).
    # IMPORTANT: do NOT also bump `data_seed`. `data_seed` is the
    # transformers-convention sampler seed; ALL groups must iterate the dataset
    # in identical order so that `gathered_answers[g*G:(g+1)*G]` corresponds to
    # the SAME prompt across groups (required for cross-supervision to be meaningful).
    # Offset by group index (A=0, B=+1, C=+2, ...).
    seed_offset = ord(script_args.group) - ord("A")
    if seed_offset > 0:
        if training_args.data_seed is None:
            training_args.data_seed = training_args.seed
        training_args.seed += seed_offset
    if script_args.rendezvous_dir is None:
        raise ValueError("--rendezvous_dir is required for co_rl.")

    ################
    # WandB
    ################
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )

    model_short = model_args.model_name_or_path.split("/")[-1]
    peer_short = (
        script_args.peer_model_name_or_path.split("/")[-1]
        if script_args.peer_model_name_or_path
        else "unknown"
    )

    if script_args.run_config:
        full_wandb_run_name = f"{script_args.run_config}_group{script_args.group}_lr{lr_str}_bs{effective_batch_size}"
    else:
        full_wandb_run_name = (
            f"CoRL_{model_short}_x_{peer_short}_group{script_args.group}_"
            f"lr{lr_str}_bs{effective_batch_size}_"
            f"gen{training_args.num_generations}_"
            f"temp{training_args.temperature}_"
            f"sct{script_args.self_consistency_threshold}"
        )

    print(f"\n{'='*80}")
    print(f"CO-RL (group {script_args.group}) CONFIGURATION")
    print(f"{'='*80}")
    print(f"This model   : {model_args.model_name_or_path}")
    print(f"Supervised by: {peer_groups[0]}  (N={len(peer_groups) + 1}-way co-learning)")
    print(f"Peer models  : {script_args.peer_model_name_or_path}")
    print(f"Rendezvous   : {script_args.rendezvous_dir}")
    print(f"WandB run    : {full_wandb_run_name}")
    print(f"Output dir   : {training_args.output_dir}")
    print(f"SCT          : {script_args.self_consistency_threshold}")
    print(f"World size   : {num_processes}")
    print(f"{'='*80}\n")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "group": script_args.group,
                "model": model_args.model_name_or_path,
                "peer_model": script_args.peer_model_name_or_path,
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
                "data_seed": training_args.data_seed,
                "self_consistency_threshold": script_args.self_consistency_threshold,
                "vllm_gpu_memory_utilization": training_args.vllm_gpu_memory_utilization,
                "seed": training_args.seed,
            },
        )

    ################
    # Model & Tokenizer
    ################
    import torch

    if hasattr(model_args, "torch_dtype") and model_args.torch_dtype is not None:
        if isinstance(model_args.torch_dtype, str):
            dtype_map = {
                "bfloat16": torch.bfloat16, "bf16": torch.bfloat16,
                "float16": torch.float16, "fp16": torch.float16,
                "float32": torch.float32, "fp32": torch.float32,
            }
            model_dtype = dtype_map.get(model_args.torch_dtype.lower(), torch.bfloat16)
        else:
            model_dtype = model_args.torch_dtype
    elif hasattr(model_args, "dtype") and model_args.dtype is not None:
        model_dtype = model_args.dtype
    else:
        model_dtype = torch.bfloat16

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
    # Dataset, two groups use the same seed/world_size so RepeatSampler
    # yields identical index sequences, ensuring both groups train on the
    # same prompts at every generation step (required for cross-labeling).
    #
    # Co-rewarding-I replication on co_rl infra: group A may load the
    # ORIGINAL parquet and group B the REPHRASED parquet (same underlying
    # problems, different surface form, identical answers, row-aligned).
    # `--train_dataset_per_group` overrides this group's dataset; if absent
    # for this letter, fall back to `--train_dataset`.
    ################
    _ds_for_this_group = script_args.train_dataset
    if script_args.train_dataset_per_group:
        # Parse 'B=coreward/math_rephrased,C=...' into a dict.
        _per_group_map = {}
        for piece in script_args.train_dataset_per_group.split(","):
            piece = piece.strip()
            if not piece:
                continue
            if "=" not in piece:
                raise ValueError(
                    f"--train_dataset_per_group entries must be 'GROUP=dataset'; "
                    f"got {piece!r}"
                )
            grp, name = piece.split("=", 1)
            grp, name = grp.strip(), name.strip()
            if grp == "A":
                raise ValueError(
                    f"--train_dataset_per_group cannot override group A; pass "
                    f"--train_dataset for group A's dataset (got A={name!r})"
                )
            _per_group_map[grp] = name
        if script_args.group in _per_group_map:
            _ds_for_this_group = _per_group_map[script_args.group]
            print(
                f"[dataset] group {script_args.group} using per-group override: "
                f"{_ds_for_this_group} (vs --train_dataset {script_args.train_dataset})"
            )
    train_dataset, eval_dataset = load_dataset(_ds_for_this_group)

    ################
    # PEFT
    ################
    peft_config = get_peft_config(model_args)

    ################
    # Rendezvous
    ################
    rendezvous = Rendezvous(
        rendezvous_dir=script_args.rendezvous_dir,
        my_group_name=script_args.group,
        peers=peer_groups,
    )

    ################
    # Training
    ################
    trainer = CoRLTrainer(
        model=model_args.model_name_or_path,
        reward_funcs=reward_correctness,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=tokenizer,
        peft_config=peft_config,
        my_group_name=script_args.group,
        rendezvous=rendezvous,
        self_consistency_threshold=script_args.self_consistency_threshold,
        log_oracle_accuracy=script_args.log_oracle_accuracy,
    )

    trainer.add_callback(BestKeeperCallback())

    trainer.train()
    trainer.save_model(training_args.output_dir)
