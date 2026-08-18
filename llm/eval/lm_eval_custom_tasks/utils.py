"""Custom-task helpers for the AMC23 / MATH-500 / GPQA lm-eval-harness yamls.

We extract the **last** \\boxed{...} from the model completion and compare to
the gold answer. Comparison strategy:

  - AMC 23: numeric / symbolic (e.g. "1/2", "\\sqrt{3}"). Use `math_verify`
            (sympy-based, latex-aware) for equivalence.

Both `process_results_*` functions return:
    {"exact_match": 1.0 | 0.0}

so they slot into the yaml's `metric_list: [{metric: exact_match, ...}]`.
"""

from __future__ import annotations

import re
from typing import Any


_BOXED_RE = re.compile(r"\\?boxed\{([^{}]+|\{[^}]*\})\}")  # backslash optional: RL ckpts often emit bare `boxed{}`


def _last_boxed(text: str) -> str | None:
    """Return content of the *last* \\boxed{...} in `text`, or None.

    Handles 1-level nested braces (e.g. \\boxed{\\frac{1}{2}}).
    """
    if not text:
        return None
    # Greedy walk to find the last boxed{ and balance braces.
    # Match "boxed{" (NOT "\\boxed{") so we catch both the LaTeX form and the
    # bare `boxed{}` that chat/RL ckpts frequently emit (dropped backslash) —
    # measured ~6% of MATH-500 answers were correct-but-bare and scored 0.
    idx = text.rfind("boxed{")
    if idx == -1:
        # fallback: simple regex (no nesting)
        m = _BOXED_RE.findall(text)
        return m[-1].strip() if m else None
    i = idx + len("boxed{")
    depth = 1
    out = []
    while i < len(text) and depth > 0:
        ch = text[i]
        if ch == "{":
            depth += 1
            out.append(ch)
        elif ch == "}":
            depth -= 1
            if depth == 0:
                break
            out.append(ch)
        else:
            out.append(ch)
        i += 1
    return "".join(out).strip() if out else None


def _flatten_samples(results: Any) -> list[str]:
    """Return the k completions for one doc, whatever nesting lm-eval used.

    With the default `take_first` filter `results` is `[str]`. With
    `take_first_k` it comes back one level deeper, `[[str, ...]]`, and grading
    it directly raises `AttributeError: 'list' object has no attribute 'rfind'`.
    Accept both so the grader does not depend on the filter in the yaml.
    """
    if not results:
        return []
    out: list[str] = []
    for r in results:
        if isinstance(r, (list, tuple)):
            out.extend(str(x) for x in r)
        else:
            out.append(str(r))
    return out



_WARNED = False


def _warn_no_math_verify() -> None:
    """math-verify is missing, so grading falls back to a plain compare."""
    global _WARNED
    if not _WARNED:
        import sys
        print("[warn] math-verify not installed: AMC / MATH-500 fall back to a "
              "plain compare and will read low. Run eval/setup_external.sh.",
              file=sys.stderr)
        _WARNED = True


def process_results_amc(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """AMC-23: numeric / symbolic; use math_verify for equivalence."""
    # avg@k, not results[0]. `repeats: 8` in the yaml asks for eight samples and
    # the paper describes this column as avg@8 (PAPER_OUTLINE_v5 4.3 and the
    # 5.1 caption); grading only the first sample made it a single-sample pass@1
    # at eight times the cost, and left the column with sigma ~= 4.9 points on
    # 83 problems -- an identical rerun moved it 6.0 points.
    target = str(doc.get("answer", "")).strip()
    if not results:
        return {"exact_match": 0.0}

    def _one(completion: str) -> float:
        pred = _last_boxed(completion)
        if pred is None:
            return 0.0
        try:
            from math_verify import parse, verify  # type: ignore
            return float(bool(verify(parse(f"${target}$"), parse(f"${pred}$"))))
        except ImportError:
            _warn_no_math_verify()
            # Fallback: best-effort numeric compare.
        except Exception:
            # Fallback: best-effort numeric compare.
            try:
                return float(float(pred) == float(target))
            except (ValueError, TypeError):
                return float(pred.strip() == target.strip())

    samples = _flatten_samples(results)
    if not samples:
        return {"exact_match": 0.0}
    scores = [_one(c) for c in samples]
    return {"exact_match": sum(scores) / len(scores)}


# ---------------------------------------------------------------------------
# GPQA (choice-based) — aligned with CoMAS (arXiv 2510.08529) maslab/evaluation.py
# so our GPQA numbers are directly comparable to CoMAS/Co-rewarding.
#
# CoMAS extracts the LAST `\boxed{A-D}` (case-insensitive) and exact-matches the
# gold letter. lm-eval's native gpqa filters look for "(A)" / "The answer is A"
# and MISS `\boxed{A}` — which is exactly what \boxed-trained RL ckpts emit
# (measured: 37% of heter-Qwen GPQA answers were \boxed{A} → scored 0). This
# replicates CoMAS's extraction verbatim.
# ---------------------------------------------------------------------------

_BOXED_LETTER_RE = re.compile(r"\\?boxed\{\(?([A-D])\)?\}", re.IGNORECASE)  # backslash optional


def process_docs_gpqa(dataset):
    """Shuffle the 4 choices and set gold letter — same as lm-eval's gpqa
    process_docs, but with a fixed seed per doc for reproducibility, and a
    `\\boxed{}` answer instruction so models emit the CoMAS-style format."""
    import random

    def _preprocess(t):
        if t is None:
            return " "
        return t.strip().replace(" [title]", ". ").replace("  ", " ")

    def _process(doc, idx):
        choices = [
            _preprocess(doc["Incorrect Answer 1"]),
            _preprocess(doc["Incorrect Answer 2"]),
            _preprocess(doc["Incorrect Answer 3"]),
            _preprocess(doc["Correct Answer"]),
        ]
        random.Random(idx).shuffle(choices)
        correct_idx = choices.index(_preprocess(doc["Correct Answer"]))
        return {
            "choice1": choices[0],
            "choice2": choices[1],
            "choice3": choices[2],
            "choice4": choices[3],
            "answer": chr(65 + correct_idx),  # bare letter "A".."D" (CoMAS gold format)
        }

    return dataset.map(_process, with_indices=True)


def process_results_gpqa(doc: dict[str, Any], results: list[str]) -> dict[str, float]:
    """GPQA: extract last \\boxed{A-D} (CoMAS), exact-match the gold letter."""
    completion = results[0] if results else ""
    target = str(doc.get("answer", "")).strip().upper()
    m = _BOXED_LETTER_RE.findall(completion)
    pred = m[-1].strip().upper() if m else ""
    return {"exact_match": float(pred == target and pred != "")}
