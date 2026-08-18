"""Cross-supervised multimodal GRPO with data-parallel split: each group trains one VLM.

Multimodal sibling of `co_rl/co_rl_trainer.py`. Identical
cross-supervision logic (string-level peer pseudo-label exchange via
file rendezvous, majority-vote labeling, eval-mode short-circuit). The
**only differences** from co_rl:

  1. This trainer's `processing_class` is an `AutoProcessor` (not
     `AutoTokenizer`) at construction time, the parent `GRPOTrainer`
     handles the VLM forward path / image tokenization transparently
     so this trainer body needs no image-specific code.
  2. `_majority_vote` (from `co_label_utils`) + MCQ-aware
     `extract_and_normalize_mcq` / `normalize_mcq` (from `mcq_grade`) use R1-V
     `<answer>` tags; option-letter answers cluster by bare letter so
     `"C"` / `"C. text"` / `"option C"` vote together (instead of fragmenting).
  3. No 4-regime / disagree / naive reward variants, only binary
     cross-supervision.

Cross-supervision mechanism (unchanged from co_rl):

Two accelerate worlds run in parallel (group A on CUDA_VISIBLE_DEVICES=0..N-1,
group B on N..2N-1). Each group is a standard `GRPOTrainer` with a single
override: `_calculate_rewards` computes this group's pseudo-labels, exchanges
them with the peer group via a file rendezvous, and injects the peer's
pseudo-labels into `inputs[i]["solution"]` before delegating to the parent
reward path.

This override is the *only* coupling between the two groups. Generation,
forward, backward, and DS->vLLM weight sync all happen independently
inside each group, so the two groups run in genuine parallel across
disjoint GPUs.
"""

from accelerate.utils import broadcast_object_list, gather_object
from trl import GRPOTrainer
from trl.trainer.grpo_trainer import RepeatSampler

from co_label_utils import (
    _UNLABELED_SENTINEL,
    _majority_vote,
)
from mcq_grade import extract_and_normalize_mcq, grade_mcq_or_math, normalize_mcq


class CoRLTrainer(GRPOTrainer):
    """
    Args:
        my_group_name (`str`):
            `'A'` or `'B'`. Identifies which half of the run this process belongs to.
        rendezvous (`Rendezvous`):
            File-based communicator to the peer group. Only the main process of
            each group calls `exchange()`; the rest receive via broadcast.
        self_consistency_threshold (`float`, *optional*, defaults to `0.0`):
            Minimum top-answer frequency (over parseable rollouts per prompt group)
            for this group's pseudo-label to be accepted. `0.0` takes the plurality
            winner. Groups below the threshold are labeled with `_UNLABELED_SENTINEL`
            so the peer's accuracy reward evaluates to 0.0 for every rollout in them.
        log_oracle_accuracy (`bool`, *optional*, defaults to `True`):
            Log how often this group's pseudo-label matches the dataset's real
            `solution` (metric `co_labeling/oracle_accuracy_me`). Purely diagnostic;
            the real label never influences training.
    """

    def __init__(
        self,
        *args,
        my_group_name: str,
        rendezvous,
        self_consistency_threshold: float = 0.0,
        log_oracle_accuracy: bool = True,
        **kwargs,
    ):
        assert my_group_name in ("A", "B"), f"my_group_name must be 'A' or 'B', got {my_group_name!r}"
        super().__init__(*args, **kwargs)
        self.my_group_name = my_group_name
        self.rendezvous = rendezvous
        self.self_consistency_threshold = self_consistency_threshold
        self.log_oracle_accuracy = log_oracle_accuracy
        # Rendezvous counter advances once per call to `_calculate_rewards` in
        # train mode (i.e., once per train generation step), NOT per training
        # step. `_calculate_rewards` is only invoked inside
        # `_generate_and_score_completions`, which the parent calls every
        # `steps_per_generation * num_iterations` training steps. Eval mode
        # short-circuits before touching rendezvous, so no eval counter is needed.
        self._gen_counter_train = 0

    def _get_train_sampler(self, dataset=None):
        """Pin the data-shuffle seed to `data_seed` so both co-learn groups
        iterate the dataset in IDENTICAL order.

        Group B offsets `args.seed` (+1) to diverge generation RNG. The parent
        `GRPOTrainer._get_train_sampler` builds its `RepeatSampler` with
        `seed=self.args.seed` (NOT `data_seed`), so the seed offset would also
        reshuffle B's data, making group A's prompt-group `g` a DIFFERENT prompt
        than group B's prompt-group `g`. Cross-supervision then pairs peer
        pseudo-labels with unrelated prompts (peer_agreement collapses to noise,
        reward → 0). Both groups share `data_seed`, so pinning the sampler to it
        keeps prompt order in lockstep while `seed` still diverges generation.

        Body mirrors the parent's sampler exactly except for `seed=`. Keep it
        aligned with `GRPOTrainer._get_train_sampler` if that ever changes.
        """
        if dataset is None:
            dataset = self.train_dataset
        sampler_seed = self.args.data_seed if self.args.data_seed is not None else self.args.seed
        return RepeatSampler(
            data_source=dataset,
            mini_repeat_count=self.num_generations,
            batch_size=self.args.generation_batch_size // self.num_generations,
            repeat_count=self.num_iterations * self.args.steps_per_generation,
            shuffle=self.shuffle_dataset,
            seed=sampler_seed,
        )

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        # Eval-mode short-circuit. In eval we want pass@1 accuracy on the
        # validation set against the **dataset's real solution**, not against
        # a peer-supplied pseudo-label. Skipping the cross-labeling path means:
        #   1. inputs[i]["solution"] keeps its dataset value (not overwritten),
        #      so the parent's reward path (reward_correctness) compares the
        #      completion against ground truth via grade_answer.
        #   2. self.rendezvous is never touched in eval, so the two groups do
        #      not need to be in lockstep during eval (one can finish first).
        #   3. self._gen_counter_train is not advanced by eval, so train-mode
        #      rendezvous alignment with the peer survives any number of eval
        #      runs interleaved between train steps.
        # The "co_labeling/*" metrics are intentionally not logged in eval mode
        # because they have no meaning without cross-labeling. The parent path
        # logs reward stats automatically into `eval/rewards/...` via trl.
        if not self.model.training:
            return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)

        # ---- Train mode: cross-labeling + peer rendezvous (original path) ----
        # A prompt's N rollouts are grouped contiguously in the global batch (after
        # cross-rank concatenation), but a single rank only holds a slice of that
        # batch, its local slice length is not necessarily a multiple of
        # num_generations. We therefore all-gather parsed answers within our group,
        # compute pseudo-labels globally, exchange them with the peer group, and
        # each rank writes back only its own slice of the peer's pseudo-labels.
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

        # ---- 1. Gather my group's answers and the dataset's real solutions ----
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

        # ---- 2. Majority vote my pseudo-labels over my own G rollouts per prompt ----
        my_pseudo = []
        num_labeled_me = 0
        num_oracle_me = 0
        for g in range(num_groups):
            lo, hi = g * G, (g + 1) * G
            label, _ = _majority_vote(gathered_answers[lo:hi], self.self_consistency_threshold)
            if label is None:
                my_pseudo.append(_UNLABELED_SENTINEL)
            else:
                my_pseudo.append(label)
                num_labeled_me += 1
                if self.log_oracle_accuracy:
                    gt = normalize_mcq(gathered_real_solutions[lo])
                    if gt is not None and gt == label:
                        num_oracle_me += 1

        # ---- 3. Exchange pseudo-labels with peer group via file rendezvous ----
        # Only the main process of each group touches the filesystem; the rest
        # receive peer's pseudo-labels via in-group broadcast.
        # NB: only train-mode rendezvous (eval short-circuits before this).
        gc = self._gen_counter_train
        self._gen_counter_train += 1

        if self.accelerator.is_main_process:
            peer_pseudo = self.rendezvous.exchange(mode=mode, counter=gc, payload=my_pseudo)
            # Sanity: peer must send same number of prompt groups.
            if len(peer_pseudo) != num_groups:
                raise RuntimeError(
                    f"peer sent {len(peer_pseudo)} pseudo-labels for {mode} gc={gc}, "
                    f"expected {num_groups}, groups out of sync"
                )
            object_list = [peer_pseudo]
        else:
            object_list = [None]
        # Broadcast a single-element list containing the peer_pseudo list.
        # (broadcast_object_list modifies the list inplace.)
        broadcast_object_list(object_list, from_process=0)
        peer_pseudo = object_list[0]

        # ---- 4. Cross-labeling metrics ----
        metrics = self._metrics[mode]
        num_labeled_peer = sum(1 for p in peer_pseudo if p != _UNLABELED_SENTINEL)
        both_labeled = sum(
            1 for a, b in zip(my_pseudo, peer_pseudo)
            if a != _UNLABELED_SENTINEL and b != _UNLABELED_SENTINEL
        )
        peer_agree = sum(
            1 for a, b in zip(my_pseudo, peer_pseudo)
            if a != _UNLABELED_SENTINEL and b != _UNLABELED_SENTINEL and a == b
        )
        metrics["co_labeling/peer_agreement"].append(
            peer_agree / both_labeled if both_labeled > 0 else 0.0
        )
        metrics["co_labeling/labeled_fraction_me"].append(num_labeled_me / num_groups)
        metrics["co_labeling/labeled_fraction_peer"].append(num_labeled_peer / num_groups)
        metrics["co_labeling/both_labeled_fraction"].append(both_labeled / num_groups)
        if self.log_oracle_accuracy:
            metrics["co_labeling/oracle_accuracy_me"].append(
                num_oracle_me / num_labeled_me if num_labeled_me > 0 else 0.0
            )

        # ---- DEBUG (env CO_DEBUG_PEER=1): diagnose the peer_agreement anomaly ----
        import os as _os
        if _os.environ.get("CO_DEBUG_PEER") == "1" and self.accelerator.is_main_process:
            import json as _json
            # (a) group purity: are a prompt-group's G rollouts actually ONE prompt?
            #     tests the "G rollouts are contiguous after gather_object" assumption.
            pure = 0
            for g in range(num_groups):
                sols = {str(s) for s in gathered_real_solutions[g * G:(g + 1) * G]}
                if len(sols) == 1:
                    pure += 1
            # (b) string vs semantic peer agreement (does surface form hide agreement?)
            s_agree = sem_agree = both = 0
            for a, b in zip(my_pseudo, peer_pseudo):
                if a != _UNLABELED_SENTINEL and b != _UNLABELED_SENTINEL:
                    both += 1
                    s_agree += int(a == b)
                    sem_agree += int(grade_mcq_or_math(a, b))
            rec = {
                "grp": self.my_group_name, "gc": gc, "num_groups": num_groups, "G": G,
                "group_purity": pure / num_groups,
                "string_agree": (s_agree / both) if both else None,
                "semantic_agree": (sem_agree / both) if both else None,
                "examples": [
                    {
                        "real_set": list({str(s) for s in gathered_real_solutions[g*G:(g+1)*G]})[:5],
                        "my_pseudo": my_pseudo[g], "peer_pseudo": peer_pseudo[g],
                        "rollout_answers": [str(x) for x in gathered_answers[g*G:(g+1)*G]],
                    }
                    for g in range(min(3, num_groups))
                ],
            }
            with open(_os.environ.get("CO_DEBUG_FILE", "/tmp/co_debug_peer.jsonl"), "a") as _f:
                _f.write(_json.dumps(rec, ensure_ascii=False) + "\n")

        # ---- 5. Inject peer's pseudo-labels into this rank's local slice ----
        # Expand per-prompt-group label into per-rollout labels (G copies each),
        # then take this rank's [rank * N_local, (rank + 1) * N_local) slice.
        peer_expanded = []
        for label in peer_pseudo:
            peer_expanded.extend([label] * G)
        my_slice = peer_expanded[rank * N_local : (rank + 1) * N_local]
        for i, label in enumerate(my_slice):
            inputs[i]["solution"] = label

        # ---- 6. Delegate to parent for the actual reward function call ----
        # Parent will gather rewards_per_func across my group (not across the peer
        # group, the two groups have disjoint process groups). Group-internal
        # gather + group-internal advantage normalization is exactly what GRPO
        # semantics call for: each model normalizes its own rewards.
        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
