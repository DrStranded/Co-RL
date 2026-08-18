"""Cross-supervised GRPO with data-parallel split: each group trains one model.

N≥2 accelerate worlds run in parallel (group A on CUDA_VISIBLE_DEVICES=0..,
group B on next slice, group C ..., each on disjoint GPUs). Each group is a
standard `GRPOTrainer` with a single override: `_calculate_rewards` computes
this group's pseudo-labels, exchanges them with the *other N-1* groups via a
file rendezvous, then injects a single supervision label per prompt:

  - 2-way (N=2): the peer's pseudo-label IS the supervision.
  - N-way (N≥3): majority-vote over the N-1 peer pseudo-labels. **Ties (no
    strict majority) -> UNLABELED**, matching the tie-discard protocol.

This override is the *only* coupling between groups. Generation, forward,
backward, and DS->vLLM weight sync all happen independently inside each
group, so the groups run in genuine parallel across disjoint GPUs.
"""

import time
from collections import Counter

from accelerate.utils import broadcast_object_list, gather_object
from trl import GRPOTrainer

from co_label_utils import (
    _UNLABELED_SENTINEL,
    _extract_and_normalize,
    _get_text,
    _majority_vote,
    normalize_answer,
)
# CoMAS coding data: the votable "answer" of a code completion is its run-output
# tuple (string). Imported only for the task=="coding" branch; math is untouched.
from comas.code_reward import extract_calls as _comas_extract_calls
from comas.code_reward import voting_answer as _comas_voting_answer


def _peer_majority_vote(peer_labels: list[str]) -> str:
    """N-way pseudo-label aggregator: majority over N-1 peer voted answers.

    Args:
        peer_labels: one voted answer per peer, possibly `_UNLABELED_SENTINEL`
            from a peer whose own internal SC vote failed.

    Returns:
        The strict-majority answer string, or `_UNLABELED_SENTINEL` if:
          - all peers are unlabeled, OR
          - there is no strict plurality (ties at the top -> discard).

    `_majority_vote` (sibling) treats `None` as "no answer"; we map sentinel
    to `None` for the same semantics.
    """
    valid = [p for p in peer_labels if p != _UNLABELED_SENTINEL]
    if not valid:
        return _UNLABELED_SENTINEL
    counts = Counter(valid).most_common()
    if len(counts) >= 2 and counts[0][1] == counts[1][1]:
        # tie at the top, discard on ties
        return _UNLABELED_SENTINEL
    return counts[0][0]






class CoRLTrainer(GRPOTrainer):
    """
    Args:
        my_group_name (`str`):
            Group identifier (e.g. `'A'`, `'B'`, `'C'`). Identifies which world this
            process belongs to.
        rendezvous (`Rendezvous`):
            File-based communicator to the peer group(s). Only the main process of
            each group touches the filesystem; the rest receive via broadcast.
            For N-way co-learning the `Rendezvous` must be constructed with
            `peers=[...]` listing the other N-1 group names.
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
        super().__init__(*args, **kwargs)
        self.my_group_name = my_group_name
        self.rendezvous = rendezvous
        self.self_consistency_threshold = self_consistency_threshold
        self.log_oracle_accuracy = log_oracle_accuracy
        # `peers` is sourced from the Rendezvous instance, single source of truth.
        # `peers` is sourced from the Rendezvous instance, single source of truth.
        # It excludes this group and keeps the declared order, so `peers[0]` is
        # this agent's predecessor round the ring. With two agents that is simply
        # the other agent, and they supervise each other.
        self.peers = list(rendezvous.peers)
        self.n_way = len(self.peers) + 1
        # Rendezvous counter advances once per call to `_calculate_rewards` in
        # train mode (i.e., once per train generation step), NOT per training
        # step. `_calculate_rewards` is only invoked inside
        # `_generate_and_score_completions`, which the parent calls every
        # `steps_per_generation * num_iterations` training steps. Eval mode
        # short-circuits before touching rendezvous, so no eval counter is needed.
        self._gen_counter_train = 0

    def _calculate_rewards(self, inputs, prompts, completions, completion_ids_list):
        # Eval-mode short-circuit. In eval we want pass@1 accuracy on the
        # 150-prompt validation set against the **dataset's real solution**,
        # not against a peer-supplied pseudo-label. Skipping the cross-labeling
        # path means:
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
        # Task-routed votable answer: math/science -> normalized boxed answer
        # (unchanged); coding -> run the completion's code on the test inputs and
        # use the output-tuple string (so the existing vote/exchange/reward chain
        # works unchanged, just with a different "answer" representation).
        def _votable_answer(completion, inp):
            if inp.get("task") == "coding":
                fn, call_inputs = _comas_extract_calls(inp.get("test_code", ""))
                return _comas_voting_answer(_get_text(completion), fn, call_inputs)
            return _extract_and_normalize(completion)

        local_answers = [_votable_answer(c, inp) for c, inp in zip(completions, inputs)]
        local_real_solutions = [inp.get("solution") for inp in inputs]
        local_tasks = [inp.get("task", "math") for inp in inputs]
        if world_size > 1:
            gathered_answers = gather_object(local_answers)
            gathered_real_solutions = gather_object(local_real_solutions)
            gathered_tasks = gather_object(local_tasks)
        else:
            gathered_answers = local_answers
            gathered_real_solutions = local_real_solutions
            gathered_tasks = local_tasks
        assert len(gathered_answers) == N_global, (
            f"gather_object returned {len(gathered_answers)} items, expected {N_global}"
        )

        # ---- 2. Majority vote my pseudo-labels over my own G rollouts per prompt ----
        my_pseudo = []
        num_labeled_me = 0
        num_oracle_me = 0
        # Normalized ground truth per prompt group, or None where unavailable
        # (coding tasks, unparseable solutions). Diagnostics only: this list is
        # never written into `inputs` and never reaches the reward function.
        gt_answers: list = []
        for g in range(num_groups):
            lo, hi = g * G, (g + 1) * G
            label, _ = _majority_vote(gathered_answers[lo:hi], self.self_consistency_threshold)
            # Same normalization the oracle check below uses, so the two
            # diagnostics agree on what "matches the truth" means.
            gt_answers.append(
                normalize_answer(gathered_real_solutions[lo])
                if gathered_tasks[lo] != "coding" else None
            )
            if label is None:
                my_pseudo.append(_UNLABELED_SENTINEL)
            else:
                my_pseudo.append(label)
                num_labeled_me += 1
                # Oracle accuracy is a math-only diagnostic; for coding the real
                # "solution" is test asserts, not a sympy-comparable answer, so skip.
                if self.log_oracle_accuracy and gathered_tasks[lo] != "coding":
                    gt = gt_answers[g]
                    if gt is not None and gt == label:
                        num_oracle_me += 1

        # ---- 3. Exchange pseudo-labels with peer group(s) via file rendezvous ----
        # Only the main process of each group touches the filesystem; the rest
        # receive peer's pseudo-labels via in-group broadcast.
        # NB: only train-mode rendezvous (eval short-circuits before this).
        # peer_pseudos_by_name: dict[peer_name -> list[pseudo per prompt group]]
        gc = self._gen_counter_train
        self._gen_counter_train += 1

        if self.accelerator.is_main_process:
            # Diagnostic: log wall time spent waiting on peers' files. This is the
            # interval where rank 0 is in file-poll while rank 1 is blocked on the
            # next NCCL collective (broadcast_object_list below). If this number
            # gets close to the NCCL collective timeout (default 30 min, patched to
            # 2 h in train_co_rl.py), rank 1's watchdog will fire and kill
            # this group, root cause of the 2026-05-26 group B crash. Print on
            # every step so we can see the per-step distribution in train.log.
            _rdv_start = time.perf_counter()
            if self.n_way == 2:
                # Legacy 2-way path, keep byte-identical with pre-N-way runs.
                peer_pseudo = self.rendezvous.exchange(mode=mode, counter=gc, payload=my_pseudo)
                if len(peer_pseudo) != num_groups:
                    raise RuntimeError(
                        f"peer sent {len(peer_pseudo)} pseudo-labels for {mode} gc={gc}, "
                        f"expected {num_groups}, groups out of sync"
                    )
                peer_pseudos_by_name = {self.peers[0]: peer_pseudo}
            else:
                # N-way (N≥3) path. exchange_n_way returns dict[peer_name -> payload].
                peer_pseudos_by_name = self.rendezvous.exchange_n_way(
                    mode=mode, counter=gc, payload=my_pseudo,
                )
                for peer_name, peer_pseudo in peer_pseudos_by_name.items():
                    if len(peer_pseudo) != num_groups:
                        raise RuntimeError(
                            f"peer {peer_name!r} sent {len(peer_pseudo)} pseudo-labels "
                            f"for {mode} gc={gc}, expected {num_groups}, groups out of sync"
                        )
            _rdv_elapsed = time.perf_counter() - _rdv_start
            print(
                f"[rdv-timing] group {self.my_group_name} {mode} gc={gc}: "
                f"rendezvous wait = {_rdv_elapsed:.2f}s",
                flush=True,
            )
            # Also surface as a wandb-tracked metric (logged below via _metrics).
            self._metrics[mode]["co_labeling/rendezvous_wait_seconds"].append(_rdv_elapsed)
            object_list = [peer_pseudos_by_name]
        else:
            object_list = [None]
        # Broadcast within-group: payload is small (a dict of lists of short strings).
        broadcast_object_list(object_list, from_process=0)
        peer_pseudos_by_name = object_list[0]

        # ---- 4. Supervision pseudo-label ----
        # Votes pass along a directed ring, A <- B <- C <- A: each agent is
        # rewarded against the voted answer of its predecessor, `peers[0]`.
        teacher = self.peers[0]
        supervision_pseudo = list(peer_pseudos_by_name[teacher])

        # ---- 4b. Cross-labeling metrics ----
        metrics = self._metrics[mode]

        # Per-peer fraction labeled + pairwise (me, peer) agreement.
        for peer_name in self.peers:
            peer_pseudo = peer_pseudos_by_name[peer_name]
            num_labeled_peer = sum(1 for p in peer_pseudo if p != _UNLABELED_SENTINEL)
            both_labeled = sum(
                1 for a, b in zip(my_pseudo, peer_pseudo)
                if a != _UNLABELED_SENTINEL and b != _UNLABELED_SENTINEL
            )
            peer_agree = sum(
                1 for a, b in zip(my_pseudo, peer_pseudo)
                if a != _UNLABELED_SENTINEL and b != _UNLABELED_SENTINEL and a == b
            )
            agree_rate = (peer_agree / both_labeled) if both_labeled > 0 else 0.0
            labeled_rate = num_labeled_peer / num_groups
            if self.n_way == 2:
                # Preserve legacy metric names for back-compat (existing wandb dashboards).
                metrics["co_labeling/peer_agreement"].append(agree_rate)
                metrics["co_labeling/labeled_fraction_peer"].append(labeled_rate)
                metrics["co_labeling/both_labeled_fraction"].append(both_labeled / num_groups)
            else:
                metrics[f"co_labeling/peer_agreement/{peer_name}"].append(agree_rate)
                metrics[f"co_labeling/labeled_fraction_peer/{peer_name}"].append(labeled_rate)

        metrics["co_labeling/labeled_fraction_me"].append(num_labeled_me / num_groups)

        # N-way supervision-level metrics (only meaningful when N≥3).
        if self.n_way >= 3:
            num_supervised = sum(1 for s in supervision_pseudo if s != _UNLABELED_SENTINEL)
            metrics["co_labeling/supervision_fraction"].append(num_supervised / num_groups)
            # Tie-discard rate among groups where ALL peers were labeled
            # (otherwise "tie" can be confused with "missing peer").
            all_peers_labeled_groups = [
                g for g in range(num_groups)
                if all(peer_pseudos_by_name[p][g] != _UNLABELED_SENTINEL for p in self.peers)
            ]
            ties = sum(
                1 for g in all_peers_labeled_groups
                if supervision_pseudo[g] == _UNLABELED_SENTINEL
            )
            metrics["co_labeling/peer_tie_rate"].append(
                ties / len(all_peers_labeled_groups) if all_peers_labeled_groups else 0.0
            )

            # P(all N models produced the same answer). Without this the tie
            # rate and the pairwise agreements only bound the no-majority rate
            # by inclusion-exclusion; an offline attempt on the first full run
            # could pin it no tighter than [0.00, 0.30]. One counter closes it.
            fully_labeled = [
                g for g in all_peers_labeled_groups
                if my_pseudo[g] != _UNLABELED_SENTINEL
            ]
            unanimous = sum(
                1 for g in fully_labeled
                if all(peer_pseudos_by_name[p][g] == my_pseudo[g] for p in self.peers)
            )
            metrics["co_labeling/unanimous_rate"].append(
                unanimous / len(fully_labeled) if fully_labeled else 0.0
            )

            # Does the supervision signal contain the truth? Diagnostic only -
            # `solution_gt` never reaches the reward function.
            if self.log_oracle_accuracy and gt_answers is not None:
                hits, seen_sup = 0, 0
                for g in range(num_groups):
                    sup = supervision_pseudo[g]
                    if sup == _UNLABELED_SENTINEL or gt_answers[g] is None:
                        continue
                    seen_sup += 1
                    if gt_answers[g] == sup:
                        hits += 1
                metrics["co_labeling/supervision_contains_truth"].append(
                    hits / seen_sup if seen_sup else 0.0
                )

        if self.log_oracle_accuracy:
            metrics["co_labeling/oracle_accuracy_me"].append(
                num_oracle_me / num_labeled_me if num_labeled_me > 0 else 0.0
            )

        # ---- 5. Inject supervision labels into this rank's local slice ----
        # Expand per-prompt-group label into per-rollout labels (G copies each),
        # then take this rank's [rank * N_local, (rank + 1) * N_local) slice.
        sup_expanded = []
        for label in supervision_pseudo:
            sup_expanded.extend([label] * G)
        my_slice = sup_expanded[rank * N_local : (rank + 1) * N_local]
        for i, label in enumerate(my_slice):
            inputs[i]["solution"] = label

        # ---- 6. Delegate to parent for the actual reward function call ----
        # Parent will gather rewards_per_func across my group (not across the peer
        # group, the two groups have disjoint process groups). Group-internal
        # gather + group-internal advantage normalization is exactly what GRPO
        # semantics call for: each model normalizes its own rewards.
        return super()._calculate_rewards(inputs, prompts, completions, completion_ids_list)
