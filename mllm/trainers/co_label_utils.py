"""Answer extraction + grading + majority-vote for the vision-language runs.

The text-side counterpart with two substitutions:

1. **Extractor**: `<answer>...</answer>` tags (R1-V baseline) instead of
   `\\boxed{}` (qwen-math convention).
2. **Grader**: routed through `verifiers.math_verify_wrapper`, which grades
   with the same qwen-sympy verifier the text side uses.

Function names are kept identical to the text-side module so the two
trainers stay structurally aligned.

The vision-language runs use binary cross-supervision only.
"""

from collections import Counter

from verifiers.math_verify_wrapper import (
    extract_answer_tag,
    grade_answer,
)
from verifiers.math_verify_wrapper import normalize_answer as _normalize


# Sentinel written into `solution` for prompt groups that fail the
# self-consistency threshold. Cannot match any parseable answer
# (math_verify.parse on this string returns an empty list, grade_answer
# falls through to exact-match, which also fails), so reward evaluates
# to 0 for every rollout in such a group, no reward-function change needed.
_UNLABELED_SENTINEL = "\x00__unlabeled__\x00"


def extract_boxed_answer(text):
    """Extract the answer from a model completion.

    Function named `extract_boxed_answer` (not `extract_answer_tag`) to
    keep the caller-visible signature aligned with co_rl's
    `co_label_utils.py`, the internals differ (R1-V `<answer>` tags vs
    qwen-math `\\boxed{}`) but the calling pattern in train scripts and
    trainer stays identical.

    Args:
        text (`str` or `None`):
            Raw completion text (already unwrapped from conversational
            wrapper by the caller).

    Returns:
        `str` or `None`: extracted answer string, or `None` if no
        `<answer>...</answer>` tag is present or the content is empty.
    """
    return extract_answer_tag(text)


def normalize_answer(answer):
    """Canonical-string form for use as a hash key in majority-vote.

    Args:
        answer (`str` or `None`):
            Raw extracted answer.

    Returns:
        `str` or `None`: lowercase + whitespace-collapsed string, or
        `None` for empty/None input.
    """
    return _normalize(answer)


def _get_text(completion):
    # TRL wraps completions as [{"role": "assistant", "content": "..."}] for conversational prompts
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def _extract_and_normalize(completion):
    """Pipeline used by `_majority_vote`: extract from text, then canonicalize.

    Returns ``None`` when extraction fails OR yields an empty string. The
    extractor returns ``None`` (not ``""``) on miss already, but we guard
    against empty strings here too, `_majority_vote` treats only ``None``
    as "no answer", so empty strings would otherwise inflate the
    denominator and depress `top_frequency` below threshold.
    """
    result = normalize_answer(extract_boxed_answer(_get_text(completion)))
    if result is None or result == "":
        return None
    return result


def _majority_vote(answers, threshold):
    """Group N rollouts of one prompt by canonical answer; return the plurality.

    Args:
        answers (`list[str | None]`):
            One canonicalized parsed answer per rollout in the prompt group.
            ``None`` means the rollout did not produce a parseable answer.
        threshold (`float`):
            Minimum top-answer frequency (over parseable answers) to accept
            the majority as the pseudo-label. ``0.0`` accepts the plurality
            winner.

    Returns:
        `tuple[str | None, float]`: ``(pseudo_label, top_frequency)``.
        ``pseudo_label`` is ``None`` when no rollout parses, or when the top
        frequency is below ``threshold``. ``top_frequency`` is ``0.0`` when
        no rollout parses.

    Note: clustering uses string identity on canonicalized answers, NOT
    semantic equivalence. This is fast (microseconds vs ~1-10ms per
    math_verify parse) and relies on the simple normalizer upstream.
    The downstream reward function (`reward_correctness`) does use
    math_verify semantic equivalence, so any miss here is recovered at
    reward time.
    """
    valid = [a for a in answers if a is not None]
    if not valid:
        return None, 0.0
    counts = Counter(valid)
    top_answer, top_count = counts.most_common(1)[0]
    top_freq = top_count / len(valid)
    if top_freq < threshold:
        return None, top_freq
    return top_answer, top_freq


__all__ = [
    "extract_boxed_answer",
    "normalize_answer",
    "_extract_and_normalize",
    "_majority_vote",
    "_UNLABELED_SENTINEL",
    "grade_answer",  # re-exported for reward_correctness in train script
]
