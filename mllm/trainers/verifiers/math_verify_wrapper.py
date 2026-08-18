"""Grading wrapper for co-rl-mllm (qwen-sympy backend).

Originally designed around HuggingFace `math_verify` (the grader R1-V
baseline uses). CPU verification on 2026-05-15 surfaced two issues that
made math_verify unsuitable for this project:

1. `verify(parse("\\sqrt{2}\\pi"), parse("\\sqrt{2}\\pi"))` returns
   `False` (self-fail on irrational products), produces false-negative
   rewards on geometry answers containing `\\sqrt` and `\\pi`.
2. `latex2sympy2_extended` (a math_verify dependency) pins antlr4 4.13.2,
   which conflicts with the antlr4 4.7.2 this stack pins for `latex2sympy2`.

Switched to the qwen-sympy grader (`verifiers.qwen.math_grade.grade_answer`),
the same grader the text-side trainers use.
It bakes in:
  - Hendrycks-MATH string normalization (tfrac→frac, \\text{}, etc.)
  - LaTeX → sympy equivalence (\\frac{1}{2} == 0.5)
  - Unit word stripping (`degree`, `cm`, `meter`, …) and `^\\circ`
  - Tuple/list element-wise equality with fraction-reduction guards

WARN Unicode `°` is NOT covered by qwen's `_normalize` (only the LaTeX
`^\\circ` form and the word `degree` are). GEOQA pseudo-labels and model
completions both use the raw `°` character, so this wrapper pre-strips it
before delegating to qwen. Same for the raw `π` character.

Three callers share this module, mirroring co_rl's three-way design:

1. `_majority_vote` in `co_label_utils.py` (uses `normalize_answer` as a
   fast hash key for clustering, no sympy cost).
2. `reward_correctness` in `train_mllm_co_rl.py` (uses `grade_answer`
   for true math equivalence, robust to format diffs in peer pseudo-labels).
3. `mllm_co_rl_trainer._calculate_rewards` eval-mode short-circuit
   (delegates to parent reward path, so eval accuracy uses the same
   `grade_answer` against the dataset's ground-truth solution).

File name kept as `math_verify_wrapper.py` for backwards compatibility with
upstream imports (`co_label_utils.py`, `dataset.py`). The public surface
(`extract_answer_tag`, `normalize_answer`, `grade_answer`) is stable.
"""

import re

from verifiers.qwen.math_grade import grade_answer as _qwen_grade
from verifiers.qwen.math_normalize import normalize_answer as _qwen_normalize


# Match <answer>...</answer> tags (R1-V baseline convention). Permissive:
# - case-insensitive
# - DOTALL so the content can span newlines (thinking traces often do)
# - non-greedy `.*?` so multiple <answer> blocks each match independently;
#   we take the last one (most likely the final answer after retries).
_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)


# Unicode unit / symbol pre-strip applied before qwen's normalizer, which
# only handles ASCII unit words and LaTeX `^\circ`. Order matters: replace
# multi-byte symbols with their LaTeX equivalent so qwen's `_normalize`
# downstream picks them up, then drop bare `°` outright.
_UNICODE_FIXUPS = [
    ("π", "\\pi"),
    ("√", "\\sqrt"),
    ("∞", "\\infty"),
    ("°", ""),
    ("度", ""),  # Chinese "degree" suffix occasionally appears in GEOQA gold.
]


def extract_answer_tag(text):
    """Extract content from the last `<answer>...</answer>` tag in `text`.

    R1-V baseline prompts request `<think>...</think><answer>...</answer>`
    format; we extract from the answer tag (not `\\boxed{}` which is the
    math convention used by co_rl).
    """
    if text is None:
        return None
    matches = _ANSWER_TAG_RE.findall(text)
    if not matches:
        return None
    answer = matches[-1].strip()
    if not answer:
        return None
    return answer


def extract_boxed_balanced(text):
    """Return the content of the LAST `\\boxed{...}` in `text`, balancing braces.

    Handles nested braces (e.g. `\\boxed{\\frac{a}{b}}`) which a naive regex
    `\\boxed\\{[^{}]*\\}` cannot. Returns `None` if no `\\boxed{` is present
    or the braces never close.

    Provided here for reuse by math-style projects (co-opd, co_rl) that
    share this wrapper. mllm itself does not call this, R1-V uses
    `<answer>...</answer>` tags, not `\\boxed{}`.
    """
    if text is None:
        return None
    pos = text.rfind("\\boxed{")
    if pos < 0:
        return None
    start = pos + len("\\boxed{")
    depth, i = 1, start
    while i < len(text) and depth > 0:
        if text[i] == "{":
            depth += 1
        elif text[i] == "}":
            depth -= 1
        i += 1
    if depth != 0:
        return None
    return text[start:i - 1].strip() or None


def _unicode_prestrip(s):
    """Replace unicode math symbols with LaTeX / strip bare unit chars."""
    for src, dst in _UNICODE_FIXUPS:
        s = s.replace(src, dst)
    return s


def normalize_answer(answer):
    """Canonical-string normalization for majority-vote hash keys.

    Backed by Hendrycks-MATH's normalizer (via qwen): unifies tfrac/dfrac→frac,
    drops `\\text{}` wrappers, fixes sqrt formatting, etc. String-only
    (no sympy parse), so cheap enough to run on every cluster decision.

    Returns `None` for empty/None input so callers can dedupe against missing
    labels without colliding with literal empty-string predictions.
    """
    if answer is None:
        return None
    answer = _unicode_prestrip(answer)
    out = _qwen_normalize(answer)
    if out is None or not out.strip():
        return None
    return out


def grade_answer(pred, gold):
    """Math-equivalence judgment via qwen-sympy.

    Used by:
      - `reward_correctness` in train script (per-rollout reward)
      - eval-mode reward path (against dataset ground truth)

    Returns `False` for `None` / empty inputs. Falls back to
    case-insensitive exact match if qwen's grader raises (e.g. malformed
    latex from the model).
    """
    if pred is None or gold is None:
        return False
    pred = _unicode_prestrip(pred.strip())
    gold = _unicode_prestrip(gold.strip())
    if not pred or not gold:
        return False
    try:
        return bool(_qwen_grade(pred, gold))
    except Exception:
        return pred.lower() == gold.lower()


__all__ = [
    "extract_answer_tag",
    "extract_boxed_balanced",
    "normalize_answer",
    "grade_answer",
]
