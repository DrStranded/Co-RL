"""Intrinsic single-model rewards: Entropy (RENT) & Self-Certainty (Intuitor).

Self-supervised baselines from Co-rewarding (arXiv 2508.00410, ICLR 2026) Table 1.
Originally from:
  - Intuitor (Self-Certainty): Zhao et al. 2025, "Learning to reason without external
    rewards" (arXiv 2505.19590). Eq. 2:
        Self-certainty(o|q) = 1/|o|, sum_i KL(U || p_πθ(·|q, o_<i))
    NOTE the direction: KL(U || p) is *mode-seeking*, exploding when p has near-zero
    entries (i.e. the model is concentrated on a few tokens). This is paper's
    deliberate choice over KL(p || U), which is mode-covering and equivalent up
    to a constant to entropy.
  - RENT (Entropy): Prabhudesai et al. 2025, "Maximizing confidence alone improves
    reasoning" (arXiv 2505.22660). §3.3:
        ℛ(y) = -H(π(x)) per token; per-sequence reward = -mean_t H(p_t)
    Sign convention: maximize negative entropy = minimize entropy = increase
    confidence.

NOTE: These two rewards are NOT affine-equivalent. KL(U||p) - (-H(p)) is a non-linear
function of p. Treat them as two distinct signals in the paper main table.

## Per-token quantities computed by the trainer

The trainer (`IntrinsicRewardTrainer`) does a chunked no-grad forward over
prompt+completion sequences and emits two per-token tensors:

  entropy_per_token[b, t] = H(p_t) = -sum_v p_t(v) log p_t(v)   (positive)
  kl_u_p_per_token[b, t]  = KL(U || p_t) = -log V - mean_v(log p_t(v))   (positive)

Per-sequence rewards (mean over valid completion tokens):

  r_entropy = -mean_over_mask(entropy_per_token, completion_mask)
              # negative; maximize == lower entropy == higher confidence
  r_sc      =  mean_over_mask(kl_u_p_per_token, completion_mask)
              # positive; maximize == mode-seeking concentration

## Trainer-side aggregation API (used by IntrinsicRewardTrainer)

`aggregate_per_seq_reward(per_token_value, completion_mask, reward_type)`
returns a Python `float` per rollout, ready to stash into `inputs[i]`.

## Reward function closures (passthrough)

The closure-bound `make_reward_entropy()` / `make_reward_self_certainty()` just
forward the trainer-computed per-rollout scalar from `inputs[i]["intrinsic_reward"]`.
In eval mode, the trainer skips the forward pass; the closures detect missing
`intrinsic_reward` kwargs and fall back to binary equality against the dataset's
`solution`, matching baseline `reward_correctness`.
"""

from __future__ import annotations

import os
import sys
from typing import Sequence

# Resolve self_label_utils import for eval-mode binary fallback (extract_boxed_answer,
# grade_answer). We avoid hard-coding the import path so the closures work from any cwd.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from self_label_utils import extract_boxed_answer, grade_answer


# ----------------------------------------------------------------------------
# Per-token quantity -> per-sequence reward (used by trainer, *not* by closures)
# ----------------------------------------------------------------------------

def aggregate_per_seq_reward(
    per_token_value: Sequence[float],
    completion_mask: Sequence[int],
    reward_type: str,
) -> float:
    """Aggregate per-token entropy or KL(U||p) into a per-sequence scalar reward.

    Args:
        per_token_value: length-T sequence of per-token H(p_t) (if reward_type='entropy')
            or KL(U||p_t) (if reward_type='self_certainty'). All values are
            non-negative.
        completion_mask: length-T sequence of 0/1 (1 = valid completion token,
            0 = padding or post-EOS).
        reward_type: 'entropy' or 'self_certainty'.

    Returns:
        float: scalar reward for this rollout, signed so that *larger = better*
        in GRPO advantage normalization:
          - 'entropy':         r = -mean_t(H_t)   (negative; up = more confident)
          - 'self_certainty':  r =  mean_t(KL_t)  (positive; up = more concentrated)

        Returns 0.0 if the mask is all zero (no valid tokens).
    """
    total = 0.0
    n = 0
    for v, m in zip(per_token_value, completion_mask):
        if m:
            total += v
            n += 1
    if n == 0:
        return 0.0
    mean_val = total / n
    if reward_type == "entropy":
        return -mean_val  # maximize -H == minimize H
    elif reward_type == "self_certainty":
        return mean_val  # maximize KL(U||p) == more peaked p
    else:
        raise ValueError(f"reward_type must be 'entropy' or 'self_certainty', got {reward_type!r}")


# ----------------------------------------------------------------------------
# Eval-mode binary fallback (shared by both closures)
# ----------------------------------------------------------------------------

def _get_text(completion):
    """Extract assistant text from completion (chat format or string)."""
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def _eval_binary_against_gt(completions, solution):
    """Eval-mode reward: 1.0 if extract(completion) sympy-equals solution, else 0.0."""
    rewards = []
    for completion, gt in zip(completions, solution):
        text = _get_text(completion)
        pred = extract_boxed_answer(text) if text else None
        rewards.append(1.0 if (pred is not None and grade_answer(pred, gt)) else 0.0)
    return rewards


# ----------------------------------------------------------------------------
# Reward function closures (passed to GRPOTrainer as reward_funcs)
# ----------------------------------------------------------------------------

def make_reward_entropy():
    """Closure for `--reward_type entropy` (RENT, Prabhudesai et al. 2025).

    The trainer computes the per-rollout entropy reward and stashes it as
    `inputs[i]["intrinsic_reward"]` (a scalar float, sign already set so larger
    is better). This closure is a passthrough that surfaces those values to TRL.

    In eval mode the trainer skips the chunked forward; `intrinsic_reward` is
    absent and we fall back to binary equality against `solution`, matching the
    baseline `reward_correctness` semantics.
    """

    def reward_entropy(completions, intrinsic_reward=None, solution=None,
                       log_metric=None, **kwargs):
        if intrinsic_reward is None:
            return _eval_binary_against_gt(completions, solution)
        rewards = list(intrinsic_reward)  # already signed by trainer
        if log_metric is not None and rewards:
            log_metric("entropy/avg_reward", sum(rewards) / len(rewards))
            # rewards are -mean(H); negate back for "average entropy H" diagnostic
            log_metric("entropy/avg_H", -sum(rewards) / len(rewards))
        return rewards

    return reward_entropy


def make_reward_self_certainty():
    """Closure for `--reward_type self_certainty` (Intuitor, Zhao et al. 2025).

    Same passthrough pattern as `make_reward_entropy`: trainer computes per-rollout
    KL(U||p) reward, stashes as `inputs[i]["intrinsic_reward"]`, closure surfaces it.
    """

    def reward_self_certainty(completions, intrinsic_reward=None, solution=None,
                              log_metric=None, **kwargs):
        if intrinsic_reward is None:
            return _eval_binary_against_gt(completions, solution)
        rewards = list(intrinsic_reward)  # already positive KL values
        if log_metric is not None and rewards:
            log_metric("self_certainty/avg_reward", sum(rewards) / len(rewards))
            log_metric("self_certainty/avg_kl_u_p", sum(rewards) / len(rewards))
        return rewards

    return reward_self_certainty
