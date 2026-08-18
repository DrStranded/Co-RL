"""MCQ-aware answer grading for MLLM datasets (open-r1, zwz, GeoQA, …).

Applies to the vision-language runs only; the shared graders
(`co_label_utils.py`, `verifiers/math_verify_wrapper.py`) are left untouched
so they stay aligned with the text-side copies.

Why this exists
---------------
Many MLLM datasets are multiple-choice: the gold solution is an option
*letter* (e.g. `"C"`), but the model commonly echoes the full option text
inside `<answer>` (e.g. `"C. Pythagorean theorem"`). The math grader
(`grade_answer`) scores that as wrong, so a model that picked the right
option still gets reward 0. On `multimodal-open-r1-8k-verified` this drove
Qwen2.5-VL-3B's train reward to ~0.01 (far below the 0.25 MCQ chance floor)
even though ~5/8 sampled completions were actually correct, a grading
bug, not a learning failure.

Design (validated against the real open-r1 gold distribution)
------------------------------------------------------------
`grade_mcq_or_math` runs the existing math/exact grader FIRST, then adds a
*conservative* option-letter match as a fallback (OR semantics). This is
strictly safer than letter-first matching:

  - It can only ADD credit; it never removes a correct math/numeric grade.
  - `mcq_letter` only fires on unambiguous option forms, so it does NOT
    misfire on the many golds where A–E are geometry point labels or compass
    directions (`"D is the midpoint of AC"`, `"N 23.5° E"`, `"O, B, C"`),
    which all correctly route to the math grader.

Known residual limits (accepted, ~0.3% of open-r1):
  - golds like `"90 degrees (D)"` (value + trailing bare paren) are not
    letter-matched (a trailing `"(D)"` rule would risk catching point labels).
  - completions that omit the `<answer>` tag still score 0 (intentional -
    learning the R1-V format is a training objective).

NOT addressed here (flagged separately): `_majority_vote` clusters rollouts
by `normalize_answer`, so `"C. Pythagorean"` and `"C"` hash to different
keys and fragment the vote, this degrades self-label / co-learn pseudo-label
quality on MCQ data and needs its own fix.
"""

import re

from co_label_utils import _get_text, extract_boxed_answer, grade_answer, normalize_answer

# Conservative option-letter patterns. Only forms that are unambiguously an
# A–E choice, never a bare mid-sentence letter (avoids geometry point labels).
_LEAD_ONLY = re.compile(r"^\s*\(?\s*([A-E])\s*\)?\s*$", re.IGNORECASE)            # "C", "(C)", "C)"
_LEAD = re.compile(r"^\s*\(?\s*([A-E])\s*[.):\]]", re.IGNORECASE)                  # "C. text", "A) text"
_PHRASE = re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?\s*([A-E])\b", re.IGNORECASE)  # "answer is C"


def mcq_letter(text):
    """Return the option letter `A`–`E` from `text`, or `None` if `text` is not
    an unambiguous multiple-choice answer.

    Args:
        text (`str` or `None`):
            Candidate answer string (a model's extracted `<answer>` content or a
            gold solution).

    Returns:
        `str` or `None`: the uppercase option letter, or `None` when no
        unambiguous option form is present.
    """
    if not text:
        return None
    text = text.strip()
    for pattern, use_search in ((_LEAD_ONLY, False), (_LEAD, False), (_PHRASE, True)):
        match = pattern.search(text) if use_search else pattern.match(text)
        if match:
            return match.group(1).upper()
    return None


def grade_mcq_or_math(pred, gold):
    """Grade `pred` against `gold`, tolerant of multiple-choice answers.

    Math/exact grading runs first; if it fails and `gold` is an unambiguous
    option letter, the option letters of `pred` and `gold` are compared.

    Args:
        pred (`str` or `None`):
            Extracted prediction (e.g. `<answer>` content).
        gold (`str` or `None`):
            Ground-truth (or pseudo-label) solution.

    Returns:
        `bool`: whether `pred` is judged correct.
    """
    if pred is None or gold is None:
        return False
    if grade_answer(pred, gold):
        return True
    gold_letter = mcq_letter(gold)
    return gold_letter is not None and mcq_letter(pred) == gold_letter


def normalize_mcq(answer):
    """MCQ-aware canonical key for majority-vote clustering / oracle equality.

    An option letter canonicalizes to that bare uppercase letter, so `"C"`,
    `"C. Pythagorean theorem"`, and `"option C"` all hash to the same key and
    cluster together. Non-MCQ answers fall back to the standard normalizer
    (lowercase string), so numeric / free-form answers are unchanged.

    The two key spaces never collide: MCQ keys are a single uppercase `A`–`E`,
    the standard normalizer lowercases everything.

    Args:
        answer (`str` or `None`):
            Raw extracted answer.

    Returns:
        `str` or `None`: canonical clustering key, or `None` for empty/None input.
    """
    if answer is None:
        return None
    letter = mcq_letter(answer)
    if letter is not None:
        return letter
    return normalize_answer(answer)


def extract_and_normalize_mcq(completion):
    """MCQ-aware replacement for `co_label_utils._extract_and_normalize`.

    Extracts the `<answer>` content, then canonicalizes with `normalize_mcq` so
    option-letter rollouts cluster together in majority vote. Returns `None`
    when extraction fails or yields an empty key (same contract as the original,
    which `_majority_vote` relies on to not inflate the denominator).

    Args:
        completion (`str` or `list[dict]`):
            One model rollout (raw text or TRL conversational wrapper).

    Returns:
        `str` or `None`: canonical clustering key, or `None` on no/empty answer.
    """
    result = normalize_mcq(extract_boxed_answer(_get_text(completion)))
    if result is None or result == "":
        return None
    return result
