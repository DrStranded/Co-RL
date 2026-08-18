"""Math answer extraction + grading + 4-regime reward for self_rewarding.

Parallels `co_rl/co_label_utils.py`. Extraction and grading are
powered by the `verifiers/qwen/` package (see NOTICE). Functions here are
kept identical to the co_rl copy so the two trainers stay aligned.
"""

from collections import Counter

from verifiers.qwen.qwen_math_parser import extract_answer
from verifiers.qwen.math_normalize import normalize_answer as _qwen_normalize
from verifiers.qwen.math_grade import grade_answer


# Sentinel written into `solution` for prompt groups that fail the self-
# consistency threshold. Cannot match any parsed answer (qwen.grade_answer
# returns False on this string), so reward evaluates to 0 for every rollout
# in such a group, no reward-function change needed.
_UNLABELED_SENTINEL = "\x00__unlabeled__\x00"


def extract_boxed_answer(text):
    """Extract the math answer from a model completion.

    Returns the answer string (e.g. ``\\frac{1}{2}``, ``42``) or ``None`` if no
    parseable answer is found. Backed by Qwen2.5-Math's parser, which handles
    \\boxed{}, "answer is X", ``\\boxed`` without braces, and many fallbacks.
    """
    return extract_answer(text, "math")


def normalize_answer(answer):
    """Canonicalize an answer for use as a hash key in majority-vote grouping."""
    return _qwen_normalize(answer)


def _get_text(completion):
    # TRL wraps completions as [{"role": "assistant", "content": "..."}] for conversational prompts
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def _extract_and_normalize(completion):
    """Extract from text, then canonicalize. Returns ``None`` on failure or empty."""
    result = normalize_answer(extract_boxed_answer(_get_text(completion)))
    if result is None or result == "":
        return None
    return result


def _majority_vote(answers, threshold):
    """Group N rollouts of one prompt by canonical answer; return the plurality."""
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
    "grade_answer",
    "extract_answer",
]
