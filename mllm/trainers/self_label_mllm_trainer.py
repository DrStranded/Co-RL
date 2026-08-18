"""Self-supervised GRPO with majority-vote pseudo-labels, multimodal (VLM) variant.

Port of self_rewarding/self_label_trainer.py to co-rl-mllm. The reward-side
logic (majority-vote over the N rollouts → pseudo-label → reward) is
model-agnostic: it operates on the decoded completion text + `solution`, never
on pixel_values. So this is byte-identical to the text version EXCEPT the import
is `co_label_utils` (this project's self-contained copy) instead of
`self_label_utils`. Per repo convention (self-contained, share-by-copy) we copy
rather than bridge-import.

Replaces the dataset's `solution` with a per-prompt majority-vote pseudo-label
computed across the N rollouts, then delegates to the parent reward path.
Groups below `self_consistency_threshold` get a sentinel that no parsed answer
can match, so every rollout in the group receives reward 0.
"""

from accelerate.utils import gather_object
from trl import GRPOTrainer

from co_label_utils import (
    _UNLABELED_SENTINEL,
    _majority_vote,
)
from mcq_grade import extract_and_normalize_mcq, grade_mcq_or_math


class SelfLabelingGRPOTrainer(GRPOTrainer):
    """
    GRPO trainer that replaces the ground-truth `solution` with a majority-vote
    pseudo-label computed across the N rollouts of each prompt, then delegates
    reward computation to the parent class.

    Args:
        self_consistency_threshold (`float`, *optional*, defaults to `0.0`):
            Minimum fraction (over parseable rollouts) that the top answer must
            reach for a prompt group to be labeled. `0.0` accepts the plurality
            winner. When a group fails the threshold, every rollout in it receives
            a sentinel `solution` so the accuracy reward evaluates to `0.0`.
        log_oracle_accuracy (`bool`, *optional*, defaults to `True`):
            If `True`, also log how often the pseudo-label matches the real
            `solution` from the dataset (metric `self_labeling/pseudo_label_matches_gt`).
            Purely diagnostic, the real label does not influence training.
    """

    def __init__(
        self,
        *args,
        self_consistency_threshold: float = 0.0,
        log_oracle_accuracy: bool = True,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.self_consistency_threshold = self_consistency_threshold
        self.log_oracle_accuracy = log_oracle_accuracy

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        # Eval-mode short-circuit. In eval we want pass@1 accuracy against the
        # dataset's real solution, not against a self-majority pseudo-label.
        # Skipping the self-labeling path means:
        #   1. inputs[i]["solution"] keeps its dataset value (not overwritten),
        #      so the parent's reward path (reward_correctness) compares the
        #      completion against ground truth via grade_answer.
        #   2. self.num_generations (=train value, e.g. 8) is not used to divide
        #      N_global, parent uses num_generations_eval (typically 1), so
        #      the divisibility assertion here would crash in eval (e.g. 4 % 8).
        #   3. self_labeling/* metrics are intentionally not logged in eval
        #      because they have no meaning without majority-vote pseudo-labels.
        if not self.model.training:
            return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # A prompt's N rollouts are grouped *contiguously in the global batch* (after
        # cross-rank concatenation), but a single rank only holds a slice of that batch -
        # its local slice length is not necessarily a multiple of num_generations. We
        # therefore all-gather the parsed answers, compute pseudo-labels globally, and
        # each rank writes back only its own slice.
        G = self.num_generations
        N_local = len(inputs)
        world_size = self.accelerator.num_processes
        rank = self.accelerator.process_index
        N_global = N_local * world_size
        assert N_global % G == 0, (
            f"global batch {N_global} (local {N_local} x world {world_size}) "
            f"not divisible by num_generations {G}"
        )
        num_groups = N_global // G
        mode = "train"

        local_answers = [extract_and_normalize_mcq(c) for c in completions]
        local_real_solutions = [inp.get("solution") for inp in inputs]

        if world_size > 1:
            gathered_answers = gather_object(local_answers)
            gathered_real_solutions = gather_object(local_real_solutions)
        else:
            gathered_answers = local_answers
            gathered_real_solutions = local_real_solutions
        assert len(gathered_answers) == N_global, (
            f"gather_object returned {len(gathered_answers)} items, expected {N_global}"
        )

        num_parseable = sum(a is not None for a in gathered_answers)
        num_labeled = 0
        top_freq_sum = 0.0
        num_oracle_matches = 0
        pseudo_labels_global = []

        for g in range(num_groups):
            lo, hi = g * G, (g + 1) * G
            label, top_freq = _majority_vote(
                gathered_answers[lo:hi], self.self_consistency_threshold
            )
            top_freq_sum += top_freq

            if label is None:
                pseudo = _UNLABELED_SENTINEL
            else:
                pseudo = label
                num_labeled += 1
                if self.log_oracle_accuracy:
                    gt_real = gathered_real_solutions[lo]
                    if gt_real is not None and grade_mcq_or_math(label, gt_real):
                        num_oracle_matches += 1

            pseudo_labels_global.extend([pseudo] * G)

        my_slice = pseudo_labels_global[rank * N_local : (rank + 1) * N_local]
        for i, pseudo in enumerate(my_slice):
            inputs[i]["solution"] = pseudo

        metrics = self._metrics[mode]
        metrics["self_labeling/fraction_labeled"].append(num_labeled / num_groups)
        metrics["self_labeling/top_frequency_mean"].append(top_freq_sum / num_groups)
        metrics["self_labeling/parseable_fraction"].append(num_parseable / N_global)
        if self.log_oracle_accuracy:
            oracle = (num_oracle_matches / num_labeled) if num_labeled > 0 else 0.0
            metrics["self_labeling/pseudo_label_matches_gt"].append(oracle)

        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
