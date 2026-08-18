"""Single-model VLM GRPO baseline (Phase 3 sanity, no cross-supervision).

Trains one VLM with standard GRPO using the **dataset's ground-truth
solution** as the reward target, no peer rendezvous, no pseudo-labels,
no `<group>` arg. Use this to verify the VLM training loop is healthy
and that inline eval produces reasonable numbers before flipping on
cross-supervision (Phase 4 via `train_mllm_co_rl.py`).

Shares with `train_mllm_co_rl.py`:
  - `AutoProcessor` (VLM)
  - `dataset.load_dataset(--train_dataset)`  (R1-V prompt + image)
  - `reward_correctness` (math_verify grader, `<answer>` extractor)

Does NOT share:
  - Rendezvous (not needed, no peer)
  - `--group` / `--peer_model_name_or_path` / `--rendezvous_dir`
  - Dual-seed bump (no peer to diverge from)
  - `CoRLTrainer`, uses vanilla `GRPOTrainer`
"""

import os
import glob
import json
import shutil
from dataclasses import dataclass, field

import wandb
import torch.nn as _nn
from transformers import AutoProcessor
from transformers.modeling_utils import PreTrainedModel as _PreTrainedModel

from co_label_utils import extract_boxed_answer
from mcq_grade import grade_mcq_or_math
from dataset import CLEVR_COUNTING_DATASET, GEOQA_DATASET, load_dataset


# Gemma-3 + ZeRO-3 fix: PreTrainedModel._init_weights for nn.Embedding does
# `module.weight.data[module.padding_idx].zero_()`. Under ZeRO-3, non-rank-0
# processes see size-0 weight shards because deepspeed.zero.GatheredParameters
# only materializes on modifier_rank=0. Indexing into a size-0 tensor crashes
# with `IndexError: index 0 is out of bounds for dimension 0 with size 0`.
# Qwen2.5-VL embedding has padding_idx=None so its base init never hits this
# branch; Gemma-3 sets padding_idx and crashes.
_orig_init_weights = _PreTrainedModel._init_weights


def _safe_init_weights(self, module):
    if isinstance(module, _nn.Embedding) and module.weight.data.numel() == 0:
        return
    return _orig_init_weights(self, module)


_PreTrainedModel._init_weights = _safe_init_weights


# Gemma-3 + batched-prompt fix: TRL's `_tokenize_prompts` calls the processor's
# `apply_chat_template` on the whole batch at once, and only passes `padding=True`
# when it detects the transformers 5.3.0 processor bug (transformers#44514):
#
#     needs_padding_workaround = Version("5.3.0") <= Version(transformers.__version__) < Version("5.4.0")
#
# The same class of bug is present in the Gemma-3 processor on the pinned 4.57.x, so
# that version guard misses it. Gemma3Processor builds `token_type_ids` with
#     array_ids = np.array(text_inputs["input_ids"])          # processing_gemma3.py
# which raises
#     ValueError: setting an array element with a sequence. The requested array has an
#     inhomogeneous shape after 1 dimensions.
# the moment the batch contains prompts of differing length. Qwen2.5-VL and InternVL
# never take that numpy path, which is why only Gemma runs crash (at step 0, before
# any training happens). Verified directly on google/gemma-3-4b-it: the processor
# fails with padding=False and succeeds with padding=True.
#
# Fix: force `padding=True` for Gemma-3 processors by wrapping apply_chat_template.
# Padding here is harmless for the caller, TRL unpads via the attention mask right
# after (`needs_padding_workaround` branch), and for the non-Gemma processors this
# wrapper is a no-op.
from transformers.models.gemma3.processing_gemma3 import Gemma3Processor as _Gemma3Processor

_orig_gemma3_apply_chat_template = _Gemma3Processor.apply_chat_template


def _gemma3_apply_chat_template_padded(self, *args, **kwargs):
    if not (kwargs.get("tokenize", True) and kwargs.get("return_dict", False)):
        return _orig_gemma3_apply_chat_template(self, *args, **kwargs)
    # Pad so the processor's np.array() over input_ids sees a rectangular batch...
    kwargs.setdefault("padding", True)
    out = _orig_gemma3_apply_chat_template(self, *args, **kwargs)
    # ...then UNPAD before returning. This is essential: TRL only strips padding when
    # its own `needs_padding_workaround` version check fires (transformers 5.3.0), which
    # it does not on the pinned 4.57.x. Without unpadding here, TRL would take the
    # padded ids as the literal prompt, feeding pad tokens into the model, which shows
    # up as generations that never emit <end_of_turn> and run to max_completion_length
    # (clipped_ratio 1.0, reward 0). Undoing it here keeps the contract identical to the
    # unpadded call the caller expects.
    if "attention_mask" in out:
        masks = out["attention_mask"]  # keep the ORIGINAL mask: every field is unpadded against it
        out["input_ids"] = [
            [tok for tok, m in zip(ids, mask, strict=True) if m]
            for ids, mask in zip(out["input_ids"], masks, strict=True)
        ]
        if "token_type_ids" in out:
            out["token_type_ids"] = [
                [t for t, m in zip(tt, mask, strict=True) if m]
                for tt, mask in zip(out["token_type_ids"], masks, strict=True)
            ]
        out["attention_mask"] = [[1] * len(ids) for ids in out["input_ids"]]
    return out


_Gemma3Processor.apply_chat_template = _gemma3_apply_chat_template_padded


def _fix_vit_attn_backend_for_odd_head_dims():
    """Stop vLLM handing Qwen2.5-VL's vision tower to a kernel that cannot run it.

    vLLM 0.11.2 drops a flag. `Qwen2_5_VisionAttention.__init__` does

        self.attn_backend, self.flash_attn_varlen_func = (
            maybe_get_vit_flash_attn_backend(self.attn_backend, self.use_upstream_fa, ...))

    but `maybe_get_vit_flash_attn_backend` returns only two values. On CUDA it
    promotes XFORMERS -> FLASH_ATTN and sets `use_upstream_fa = True` *inside*
    itself; the caller never receives that, and only the ROCm and XPU branches
    below repair it. So the tower ends up with attn_backend=FLASH_ATTN and
    use_upstream_fa=False, which routes it to vLLM's bundled FA2 -- built only
    for head dims that are multiples of 32. Qwen2.5-VL's ViT is 1280/16 = 80:

        RuntimeError: This flash attention build does not support headdim
        not being a multiple of 32.

    (An earlier version of this patched `get_vit_attn_backend` instead. That was
    a no-op: measured on an A100, it already returns XFORMERS for head_size 80.
    The promotion downstream is what undoes it.)

    Fix: suppress the promotion. When the platform picked something other than
    FLASH_ATTN, keep it -- the tower then uses the xformers wrapper, which has
    no head-dim restriction. Backends that legitimately chose FLASH_ATTN (Gemma
    and InternVL head dims are multiples of 32) fall through untouched.

    Patched on `vllm.attention.layer` before vLLM imports any model, so the
    `from ... import` in qwen2_5_vl.py binds the wrapped version.
    """
    from vllm.attention import layer as _layer
    from vllm.attention.backends.registry import AttentionBackendEnum

    _orig = _layer.maybe_get_vit_flash_attn_backend

    def _patched(attn_backend, use_upstream_fa, attn_backend_override=None):
        if attn_backend is not AttentionBackendEnum.FLASH_ATTN:
            print(
                f"[vit-attn-fix] keeping {attn_backend.name} for the vision tower "
                "instead of promoting to FLASH_ATTN (vLLM would then use its "
                "bundled FA2, which rejects head dims that are not multiples of 32)",
                flush=True,
            )
            return attn_backend, None
        return _orig(attn_backend, use_upstream_fa,
                     attn_backend_override=attn_backend_override)

    _layer.maybe_get_vit_flash_attn_backend = _patched


if os.environ.get("MLLM_VIT_ATTN_FIX") == "1":
    _fix_vit_attn_backend_for_odd_head_dims()


from trl import (
    GRPOConfig,
    GRPOTrainer,
    ModelConfig,
    ScriptArguments,
    TrlParser,
    get_kbit_device_map,
    get_peft_config,
    get_quantization_config,
)


@dataclass
class MllmSingleScriptArguments(ScriptArguments):
    """Script arguments for Phase 3 single-model VLM baseline."""

    run_config: str = field(default=None, metadata={"help": "Run name prefix."})
    wandb_entity: str = field(default=None, metadata={"help": "WandB entity."})
    wandb_project: str = field(default="co-rl-mllm", metadata={"help": "WandB project name."})
    train_dataset: str = field(
        default=CLEVR_COUNTING_DATASET,
        metadata={
            # No `choices` whitelist: `dataset.load_dataset` is the single source
            # of truth for supported sources (GeoQA/CLEVR + the `_SPECS` datasets)
            # and raises a clear ValueError for anything unknown.
            "help": "Dataset to use for training (see dataset._SPECS for sources).",
        },
    )
    self_labeling: bool = field(
        default=False,
        metadata={
            "help": "Self-rewarding mode: train against the per-prompt majority-vote "
            "pseudo-label over the N rollouts (no ground truth), via "
            "SelfLabelingGRPOTrainer. Eval still uses the dataset's GT solution."
        },
    )
    self_consistency_threshold: float = field(
        default=0.0,
        metadata={
            "help": "Min top-answer frequency for a prompt's pseudo-label to be "
            "accepted (else the group gets reward 0). Only used with --self_labeling."
        },
    )


def _get_text(completion):
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def reward_correctness(completions, solution, **kwargs):
    """1.0 iff completion's `<answer>` content matches GT (math-equivalent, or
    same option letter for multiple-choice answers, see `mcq_grade`)."""
    rewards = []
    for completion, ground_truth in zip(completions, solution):
        pred_answer = extract_boxed_answer(_get_text(completion))
        if pred_answer is not None and grade_mcq_or_math(pred_answer, ground_truth):
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


class OptimStatePrunerCallback(TrainerCallback):
    """Keep every checkpoint's weights, but only the newest checkpoint's
    optimizer state. With `save_only_model=false` (what the GT small-tier
    launchers use, so a crashed run can resume) ZeRO-3 writes a full fp32
    optimizer shard set per checkpoint: ~12 bytes/param, i.e. ~36 GB per
    checkpoint for a 3B model. Over a 961-step epoch at save_steps=20 that
    is ~1.7 TB of state, of which only the newest copy can ever be resumed
    from. On every save, drop the resume-only files from all older
    checkpoints; the model weights, config and trainer_state stay untouched
    so every checkpoint remains fully evaluatable.

    Opt-in via `MLLM_PRUNE_OPTIM_STATE=1` so the repo default is unchanged.
    """

    def on_save(self, args, state, control, **kw):
        if not state.is_world_process_zero:
            return
        ckpts = sorted(
            glob.glob(os.path.join(args.output_dir, "checkpoint-*")),
            key=lambda p: int(p.rsplit("-", 1)[1]),
        )
        # best_model is a hardlink copy taken while its source was still the
        # newest ckpt, so it carries a full optimizer set of its own. It is
        # only ever used for eval, so prune it too.
        stale = ckpts[:-1] + [os.path.join(args.output_dir, "best_model")]

        def _reclaimed(path):
            # best_model is a hardlink copy, so its bytes are still referenced by the
            # source checkpoint. Only count files whose last link is the one being
            # removed, else the log overstates the saving by a whole optimizer set.
            st = os.stat(path)
            return st.st_size if st.st_nlink == 1 else 0

        freed = 0
        for ckpt in stale:
            for pat in ("global_step*", "optimizer*.pt", "scheduler.pt", "rng_state*.pth", "latest"):
                for path in glob.glob(os.path.join(ckpt, pat)):
                    if os.path.isdir(path):
                        for root, _, files in os.walk(path):
                            freed += sum(_reclaimed(os.path.join(root, f)) for f in files)
                        shutil.rmtree(path)
                    else:
                        freed += _reclaimed(path)
                        os.remove(path)
        if freed:
            print(f"[prune-optim] step {state.global_step}: freed {freed / 2**30:.2f} GiB of "
                  f"optimizer state; resume state now exists only for {os.path.basename(ckpts[-1])}",
                  flush=True)


if __name__ == "__main__":
    parser = TrlParser((MllmSingleScriptArguments, GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()

    ################
    # WandB
    ################
    lr_str = f"{training_args.learning_rate:.0e}".replace("e-0", "e-")
    num_processes = int(os.environ.get("WORLD_SIZE", 1))
    effective_batch_size = (
        training_args.per_device_train_batch_size * training_args.gradient_accumulation_steps * num_processes
    )
    model_short = model_args.model_name_or_path.split("/")[-1]
    full_wandb_run_name = (
        script_args.run_config
        or f"MllmSingle_{model_short}_lr{lr_str}_bs{effective_batch_size}"
    )

    print(f"\n{'='*80}")
    print(f"MLLM-SINGLE (Phase 3 baseline) CONFIGURATION")
    print(f"{'='*80}")
    print(f"Model        : {model_args.model_name_or_path}")
    print(f"Dataset      : {script_args.train_dataset}")
    print(f"WandB run    : {full_wandb_run_name}")
    print(f"Output dir   : {training_args.output_dir}")
    print(f"World size   : {num_processes}")
    print(f"{'='*80}\n")

    if os.environ.get("LOCAL_RANK", "0") == "0":
        wandb.init(
            entity=script_args.wandb_entity,
            project=script_args.wandb_project,
            name=full_wandb_run_name,
            config={
                "model": model_args.model_name_or_path,
                "train_dataset": script_args.train_dataset,
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
                "lr_scheduler_type": training_args.lr_scheduler_type,
                "warmup_ratio": training_args.warmup_ratio,
                "eval_steps": training_args.eval_steps,
                "num_generations_eval": training_args.num_generations_eval,
                "per_device_eval_batch_size": training_args.per_device_eval_batch_size,
                "vllm_gpu_memory_utilization": training_args.vllm_gpu_memory_utilization,
                "seed": training_args.seed,
            },
        )

    ################
    # Model & Processor
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
        # `use_cache` deliberately omitted, see train_mllm_co_rl.py for rationale.
    )
    quantization_config = get_quantization_config(model_args)
    if quantization_config is not None:
        model_kwargs["device_map"] = get_kbit_device_map()
        model_kwargs["quantization_config"] = quantization_config
    training_args.model_init_kwargs = model_kwargs

    processor = AutoProcessor.from_pretrained(
        model_args.model_name_or_path,
        revision=model_args.model_revision,
        trust_remote_code=model_args.trust_remote_code,
        padding_side="left",
    )
    if processor.tokenizer.pad_token is None:
        processor.tokenizer.pad_token = processor.tokenizer.eos_token

    # InternVL3.5-HF crashes step 0 with "Image features and image tokens do not
    # match: tokens: 3328, features 256" because InternVLProcessor defaults
    # crop_to_patches=True → 1 image → up to 13 tiles in pixel_values; TRL's
    # split_pixel_values_by_grid only handles Qwen image_grid_thw / Gemma
    # image_position_ids, returns the batch unchanged, then split_tensor_dict
    # naively chunks pixel_values by shape[0]/num_chunks and drops most tiles.
    # Fix: force no-tiling on processor instance + class-level kwargs defaults.
    # Lossy for high-detail images, OK for GeoQA (geometry diagrams <300px
    # already tile to 1 patch with the processor's default min/max_patches=1/12).
    if "internvl" in model_args.model_name_or_path.lower():
        if hasattr(processor, "image_processor"):
            if hasattr(processor.image_processor, "crop_to_patches"):
                processor.image_processor.crop_to_patches = False
            if hasattr(processor.image_processor, "max_patches"):
                processor.image_processor.max_patches = 1
            if hasattr(processor.image_processor, "min_patches"):
                processor.image_processor.min_patches = 1
        try:
            from transformers.models.internvl.processing_internvl import InternVLProcessorKwargs
            InternVLProcessorKwargs._defaults["images_kwargs"]["crop_to_patches"] = False
        except Exception:
            pass

    # Gemma3-IT uses <end_of_turn> (id=106) as the turn terminator, but HF
    # tokenizer.eos_token_id still returns 1 (<eos>). vLLM never sees 106 as a
    # stop signal and generates until max_completion_length; TRL likewise marks
    # every completion as clipped. Fix: patch both the tokenizer and
    # generation_kwargs so both TRL and vLLM agree on the stop token set.
    _model_name_lower = model_args.model_name_or_path.lower()
    if "gemma-3" in _model_name_lower or "gemma3" in _model_name_lower:
        _GEMMA3_EOT_ID = 106  # <end_of_turn>
        processor.tokenizer.eos_token_id = _GEMMA3_EOT_ID
        processor.tokenizer.eos_token = "<end_of_turn>"
        _existing = training_args.generation_kwargs or {}
        training_args.generation_kwargs = {**_existing, "stop_token_ids": [1, _GEMMA3_EOT_ID]}

    train_dataset, eval_dataset = load_dataset(script_args.train_dataset)

    trainer_kwargs = dict(
        model=model_args.model_name_or_path,
        reward_funcs=reward_correctness,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        processing_class=processor,
        peft_config=get_peft_config(model_args),
    )
    if script_args.self_labeling:
        # Self-rewarding: reward = match the N-rollout majority vote (no GT in train).
        from self_label_mllm_trainer import SelfLabelingGRPOTrainer
        print(f"[unmaj] self-labeling ON (threshold={script_args.self_consistency_threshold})")
        trainer = SelfLabelingGRPOTrainer(
            self_consistency_threshold=script_args.self_consistency_threshold,
            **trainer_kwargs,
        )
    else:
        trainer = GRPOTrainer(**trainer_kwargs)

    trainer.add_callback(BestKeeperCallback())
    if os.environ.get("MLLM_PRUNE_OPTIM_STATE") == "1":
        trainer.add_callback(OptimStatePrunerCallback())

    trainer.train()
    trainer.save_model(training_args.output_dir)
