"""Math answer extraction + grading for co_rl.

Extraction and grading are powered by the `verifiers/qwen/` package, adapted
from Qwen2.5-Math evaluation/parser.py and a grader composing
Hendrycks-MATH / OpenAI prm800k / Microsoft ToRA / DeepSeek-Math equality
checks. This is the de-facto industry-standard math eval pipeline (lm-eval
-harness's math tasks share the same sympy + latex2sympy2 lineage).

Three things share one verifier here, by design:
1. `_majority_vote` (uses normalized strings as hash keys; canonical-ish form
   from `normalize_answer` keeps equivalent answers in the same bucket without
   paying sympy cost on every cluster decision).
2. `reward_correctness` in `train_co_rl.py` (uses `grade_answer` for true
   sympy equivalence, so train reward is robust to `1/2` vs `\\frac{1}{2}`).
3. `co_rl_trainer._calculate_rewards` eval mode (delegates to the same
   parent reward path, so eval accuracy uses the same `grade_answer`).

Used to importlib-bridge self_rewarding/self_label_trainer.py; now self-contained
per repo "trainer self-contained, share by copy" convention.
"""

from collections import Counter

from verifiers.qwen.qwen_math_parser import extract_answer
from verifiers.qwen.math_normalize import normalize_answer as _qwen_normalize
from verifiers.qwen.math_grade import grade_answer as _qwen_grade_answer

import re as _re

# `3.74 \times 10^{24}` / `3.74\cdot10^{-9}` / `3.74 * 10^24` -> `3.74e24` / `3.74e-9`.
# Lets the qwen grader match sci-notation gold (e.g. CoMAS science `3.74e+24`)
# against natural-LaTeX model output without it the qwen normalizer expands one
# side to full-int and leaves the other as `3.74*10^24`, missing the match.
_LATEX_SCI_PAT = _re.compile(
    r"(-?\s*\d+(?:\.\d+)?)\s*(?:\\times|\\cdot|\*)\s*10\s*\^\s*\{?\s*(-?\d+)\s*\}?"
)
def _normalize_sci_notation(s):
    if s is None or not isinstance(s, str):
        return s
    return _LATEX_SCI_PAT.sub(lambda m: f"{m.group(1).replace(' ', '')}e{int(m.group(2))}", s)


def grade_answer(predicted, gold):
    """Qwen grader + LaTeX-sci-notation rescue path (CoMAS science compat).

    Tries the qwen grader as-is first; if False, retries with sci-notation
    normalized on each side (model may write `\\times 10^{N}`, gold may be
    `Ae±B`). All other forms unchanged, math/coding paths are unaffected.
    """
    if _qwen_grade_answer(predicted, gold):
        return True
    p2 = _normalize_sci_notation(predicted)
    g2 = _normalize_sci_notation(gold)
    if p2 != predicted and _qwen_grade_answer(p2, gold):
        return True
    if g2 != gold and _qwen_grade_answer(predicted, g2):
        return True
    return False


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
    """Canonicalize an answer for use as a hash key in majority-vote grouping.

    Backed by Hendrycks-MATH's normalizer (via qwen): unifies tfrac/dfrac->frac,
    `0.5`->`\\frac{1}{2}`, `a/b`->`\\frac{a}{b}`, strips \\text{}/\\left/\\right/
    units/spaces/percentages, etc. Equivalent latex forms collapse into the same
    string, so `Counter(...)` clustering in `_majority_vote` actually counts
    semantically equal answers as one bucket (without paying sympy cost).

    Returns ``None`` if input is ``None``.
    """
    return _qwen_normalize(answer)


def _get_text(completion):
    # TRL wraps completions as [{"role": "assistant", "content": "..."}] for conversational prompts
    if isinstance(completion, list):
        return completion[-1]["content"] if completion else ""
    return completion


def _extract_and_normalize(completion):
    """Pipeline used by `_majority_vote`: extract from text, then canonicalize.

    Returns ``None`` when extraction fails OR yields an empty string. Qwen's
    parser returns ``''`` (not ``None``) when no parseable answer is found, but
    `_majority_vote` treats only ``None`` as "this rollout has no answer", so
    empty strings would otherwise inflate the denominator and depress
    top_frequency below threshold. Normalize to None at the boundary.
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
            Minimum top-answer frequency (over parseable answers) to accept the
            majority as the pseudo-label. ``0.0`` accepts the plurality winner.

    Returns:
        `tuple[str | None, float]`: ``(pseudo_label, top_frequency)``.
        ``pseudo_label`` is ``None`` when no rollout parses, or when the top
        frequency is below ``threshold``. ``top_frequency`` is ``0.0`` when no
        rollout parses.

    Note: clustering uses string identity on canonicalized answers, NOT sympy
    equivalence. This is fast (microseconds vs ~10-100ms per sympy compare) and
    relies on the qwen normalizer to collapse equivalent latex forms upstream.
    The downstream reward function (`reward_correctness`) does use sympy
    equivalence, so any miss here is recovered at reward time.
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
    "grade_answer",  # exported for reward_correctness in train_co_rl.py
    "extract_answer",  # raw qwen parser, exported for completeness
]
