"""GRPO trainer for intrinsic single-model rewards (SC + Entropy).

Computes per-token entropy `H(p_t) = -sum_v p_v log p_v` (RENT, arXiv 2505.22660)
or per-token `KL(U || p_t) = -log V - mean_v(log p_v)` (Intuitor, arXiv 2505.19590
Eq. 2) via a chunked no-grad forward pass over (prompt+completion) sequences.
Aggregates to a per-rollout scalar reward (sign chosen so GRPO advantage
normalization maximizes it), stashes the scalar into `inputs[i]["intrinsic_reward"]`,
and delegates to `super()._calculate_rewards` so the closure-bound passthrough
reward functions (`make_reward_entropy` / `make_reward_self_certainty`) can
surface the values to TRL.

Eval mode short-circuits: no extra forward, no `intrinsic_reward` stash. The
reward closure detects missing `intrinsic_reward` and falls back to binary
equality against the dataset's `solution`.

## Memory & speed

Full-vocabulary logits are required (mode-seeking KL is sensitive to long-tail
zeros; top-K shortcut would distort the signal). Naive single-batch forward over
B*G ≈ 96 rollouts x T=3072 x V=152k x 4B = ~180 GB -> OOM even on Blackwell 96GB.
Solution: chunked forward (default `intrinsic_chunk_size=4` sequences per chunk,
~7.5 GB peak per chunk for fp32 logits + log_softmax). 24 chunks per training
step on a per-device-batch of 96.

Time cost: +30-40% step time vs vanilla GRPO. Acceptable for paper baselines.
"""

from __future__ import annotations

import math
import os
import sys

import torch
import torch.nn.functional as F
from trl import GRPOTrainer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from intrinsic_rewards import aggregate_per_seq_reward


class IntrinsicRewardTrainer(GRPOTrainer):
    """GRPO trainer that computes per-rollout intrinsic reward (entropy or KL(U||p))
    via a chunked forward pass and stashes a single scalar per rollout into
    `inputs[i]["intrinsic_reward"]`.

    Args:
        intrinsic_reward_type (`str`):
            `'entropy'` (RENT, Prabhudesai 2025) or `'self_certainty'` (Intuitor,
            Zhao 2025). The trainer computes the matching per-token quantity
            (`H(p_t)` or `KL(U||p_t)`) and aggregates to per-sequence reward.
        intrinsic_chunk_size (`int`, *optional*, defaults to `4`):
            How many rollouts per chunked forward batch. Lower this if OOM
            (each chunk's peak memory scales with `chunk_size * T * V * 4` for
            fp32 logits, plus the same for the log_softmax tensor).
    """

    def __init__(
        self,
        *args,
        intrinsic_reward_type: str,
        intrinsic_chunk_size: int = 4,
        **kwargs,
    ):
        assert intrinsic_reward_type in ("entropy", "self_certainty"), (
            f"intrinsic_reward_type must be 'entropy' or 'self_certainty', got {intrinsic_reward_type!r}"
        )
        super().__init__(*args, **kwargs)
        self.intrinsic_reward_type = intrinsic_reward_type
        self.intrinsic_chunk_size = intrinsic_chunk_size

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        # Eval-mode short-circuit: closures will detect missing `intrinsic_reward`
        # and fall back to binary equality against ground truth.
        if not self.model.training:
            return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # ---- Train mode: chunked forward -> per-rollout scalar reward ----
        per_rollout_rewards = self._compute_intrinsic_rewards(prompts, completion_ids_list)

        # Stash per-rollout (one float per rollout) into inputs[i] so TRL's
        # _calculate_rewards reward_kwargs builder forwards it to our closure.
        # Writes self_answers
        # per inputs[i] before super()).
        for i, r in enumerate(per_rollout_rewards):
            inputs[i]["intrinsic_reward"] = float(r)

        # Diagnostic metrics. _metrics is a defaultdict(list) on parent.
        mode = "train"
        metrics = self._metrics[mode]
        if per_rollout_rewards:
            avg = sum(per_rollout_rewards) / len(per_rollout_rewards)
            metrics[f"intrinsic/{self.intrinsic_reward_type}/avg"].append(avg)

        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

    # -------------------------------------------------------------------------
    # Core: chunked forward + per-token value + per-sequence aggregation
    # -------------------------------------------------------------------------

    def _compute_intrinsic_rewards(self, prompts, completion_ids_list):
        """Compute per-rollout intrinsic reward scalar.

        Args:
            prompts: list of decoded prompt text (length B = num rollouts).
            completion_ids_list: list of completion token-id lists (length B).

        Returns:
            list[float]: per-rollout reward, signed for GRPO maximization.
        """
        tokenizer = self.processing_class
        device = next(self.model.parameters()).device
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id

        # Normalize prompts to string form, then re-tokenize.
        # `prompts` is `[x["prompt"] for x in inputs]` from parent, could be
        # either list[str] (text dataset) or list[list[dict]] (conversational
        # chat dataset). For conversational we apply the chat template with
        # add_generation_prompt=True to match the generation-time prompt
        # construction (so the (prompt+completion) sequence aligns with what
        # the model actually saw at rollout time).
        prompts_text = self._normalize_prompts_to_text(prompts)
        prompt_encs = tokenizer(prompts_text, add_special_tokens=False)
        prompt_ids_list = prompt_encs["input_ids"]

        B = len(prompts)
        if B == 0:
            return []

        prompt_lens = [len(p) for p in prompt_ids_list]
        completion_lens = [len(c) for c in completion_ids_list]
        max_p = max(prompt_lens)
        max_c = max(completion_lens)
        L = max_p + max_c

        # Build padded sequences:
        #   input_ids: (B, L), left-padded prompt, right-padded completion
        #   attention_mask: (B, L), 1 over valid tokens
        #   completion_mask: (B, max_c), 1 over valid completion positions
        input_ids = torch.full((B, L), pad_id, dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, L), dtype=torch.long, device=device)
        completion_mask = torch.zeros((B, max_c), dtype=torch.long, device=device)

        for i, (p, c) in enumerate(zip(prompt_ids_list, completion_ids_list)):
            p_len = len(p)
            c_len = len(c)
            # Left-pad prompt: place p at positions [max_p - p_len, max_p)
            input_ids[i, max_p - p_len : max_p] = torch.tensor(p, dtype=torch.long, device=device)
            attention_mask[i, max_p - p_len : max_p] = 1
            # Right-pad completion: place c at positions [max_p, max_p + c_len)
            input_ids[i, max_p : max_p + c_len] = torch.tensor(c, dtype=torch.long, device=device)
            attention_mask[i, max_p : max_p + c_len] = 1
            completion_mask[i, :c_len] = 1

        # Chunked forward -> (B, max_c) per-token value tensor
        per_token_value = self._chunked_forward_per_token(
            input_ids,
            attention_mask,
            logits_to_keep=max_c,
            chunk_size=self.intrinsic_chunk_size,
        )

        # Aggregate per rollout
        per_token_value_cpu = per_token_value.tolist()
        completion_mask_cpu = completion_mask.cpu().tolist()
        rewards = [
            aggregate_per_seq_reward(
                per_token_value_cpu[i],
                completion_mask_cpu[i],
                self.intrinsic_reward_type,
            )
            for i in range(B)
        ]
        return rewards

    def _normalize_prompts_to_text(self, prompts):
        """Coerce `prompts` (list of strings OR list of conversational message lists)
        into list[str] suitable for the tokenizer.

        TRL passes `prompts = [x["prompt"] for x in inputs]` from the parent's
        `_generate_and_score_completions`. For chat datasets each prompt is a
        list of `{"role": ..., "content": ...}` dicts; we apply
        `tokenizer.apply_chat_template(..., add_generation_prompt=True)` to
        materialize the exact text the model saw at generation time. For
        plain-text datasets we pass through.
        """
        if not prompts:
            return []
        first = prompts[0]
        if isinstance(first, str):
            return list(prompts)
        # Conversational: list of message dicts per prompt
        tokenizer = self.processing_class
        if not hasattr(tokenizer, "apply_chat_template"):
            raise ValueError(
                f"prompts are conversational (type={type(first)!r}) but tokenizer "
                "has no apply_chat_template; cannot proceed."
            )
        return [
            tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True)
            for p in prompts
        ]

    def _chunked_forward_per_token(self, input_ids, attention_mask, logits_to_keep, chunk_size):
        """Chunked no-grad forward returning per-token entropy or KL(U||p).

        Mirrors `_get_per_token_logps_and_entropies` parent's chunking + temperature
        scaling, but computes full-vocab quantities instead of the selective logp.

        Returns:
            torch.Tensor: shape (B, logits_to_keep), on CPU. Per-token H(p_t) if
            self.intrinsic_reward_type == 'entropy', or KL(U||p_t) if
            'self_certainty'. Both quantities are non-negative.
        """
        B = input_ids.size(0)
        log_V = None
        out_chunks = []

        # `model_kwarg_keys` is set by parent in __init__; tells us if model
        # supports the `logits_to_keep` short-circuit (saves compute by only
        # producing logits for the last K positions instead of full seq).
        supports_logits_to_keep = "logits_to_keep" in self.model_kwarg_keys

        for start in range(0, B, chunk_size):
            ids = input_ids[start : start + chunk_size]
            mask = attention_mask[start : start + chunk_size]

            with torch.no_grad():
                model_kwargs = {"input_ids": ids, "attention_mask": mask, "use_cache": False}
                if supports_logits_to_keep:
                    # +1 because the last logit is for the position AFTER the
                    # completion ends and gets sliced off below.
                    model_kwargs["logits_to_keep"] = logits_to_keep + 1
                logits = self.model(**model_kwargs).logits  # (chunk, L', V)

                # Drop the trailing position (next-token after completion).
                logits = logits[:, :-1, :]
                # Slice to only completion positions.
                logits = logits[:, -logits_to_keep:, :]  # (chunk, logits_to_keep, V)
                # Temperature scale to match the distribution that produced the
                # rollouts (parent does the same in `_get_per_token_logps_and_entropies`).
                logits = logits / self.temperature

                # log_softmax over vocab dimension.
                log_p = F.log_softmax(logits, dim=-1)  # (chunk, logits_to_keep, V), fp32
                if log_V is None:
                    log_V = math.log(logits.size(-1))
                # Free logits, log_p has the info we need from here on.
                del logits

                if self.intrinsic_reward_type == "entropy":
                    # H(p_t) = -sum_v p_v log p_v = -sum_v exp(log_p) * log_p
                    p = log_p.exp()
                    value = -(p * log_p).sum(dim=-1)  # (chunk, logits_to_keep), >= 0
                    del p
                else:  # self_certainty
                    # KL(U||p) = -log V - mean_v(log p_v)
                    value = -log_V - log_p.mean(dim=-1)  # (chunk, logits_to_keep), >= 0
                del log_p

                out_chunks.append(value.cpu())

        return torch.cat(out_chunks, dim=0)  # (B, logits_to_keep) on CPU
