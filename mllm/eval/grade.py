#!/usr/bin/env python3
"""Answer extraction + grading for MLLM benchmark eval.

Extract the boxed answer and grade with mathruler (rule-based). We
additionally handle
the R1-V `<answer>...</answer>` format our trained models emit, and an MCQ
option-letter fallback (A-E) for multiple-choice benchmarks.

No GPT / LLM judge, pure rule-based, fast and consistent.
"""
import re
from mathruler.grader import extract_boxed_content, grade_answer

_ANSWER_TAG = re.compile(r"<answer>(.*?)</answer>", re.DOTALL | re.IGNORECASE)
# Bare option letter, e.g. "C", "(C)", "C.", "C)", "answer is C"
_LETTER_ONLY = re.compile(r"^\s*\(?\s*([A-E])\s*\)?[.):\]]?\s*$")
_LETTER_LEAD = re.compile(r"^\s*\(?\s*([A-E])\s*[.):\]]")
_LETTER_PHRASE = re.compile(r"(?:answer|option|choice)\s*(?:is|:)?\s*\(?\s*([A-E])\b", re.IGNORECASE)


def extract_pred(text: str) -> str | None:
    """Pull the model's final answer string. Priority: <answer> tag (our models)
    → \\boxed{} (box-prompted models) → None."""
    if text is None:
        return None
    m = list(_ANSWER_TAG.finditer(text))
    if m:
        return m[-1].group(1).strip()
    boxed = extract_boxed_content(text)  # mathruler: returns "None" string if absent
    if boxed and boxed != "None":
        return boxed.strip()
    return None


def _option_letter(s: str) -> str | None:
    if s is None:
        return None
    s = s.strip()
    for rx in (_LETTER_ONLY, _LETTER_LEAD, _LETTER_PHRASE):
        m = rx.search(s)
        if m:
            return m.group(1).upper()
    return None


_CHOICE = re.compile(r"([A-E])\s*[.:)]\s*([^;\n]+?)(?=\s*(?:;|\n|$|[A-E]\s*[.:)]))")


def parse_choices(question: str) -> dict:
    """Parse MCQ options from a question. Handles 'A. 45; B. 60' (We-Math) and
    'A:40°\\nB:60°' (MathVerse). Returns {letter: value_str}."""
    if not question:
        return {}
    seg = re.split(r"[Cc]hoices?\s*[:：]", question, maxsplit=1)
    seg = seg[1] if len(seg) > 1 else question
    return {m.group(1).upper(): m.group(2).strip() for m in _CHOICE.finditer(seg)}


def grade(pred: str, gold: str, question: str | None = None) -> bool:
    """True iff pred matches gold. mathruler grade_answer first (math/latex/numeric
    aware), then MCQ option-letter fallback when gold is a bare A-E letter. When the
    model answers with the option's *value* (e.g. "90") but gold is the letter ("D"),
    map via the question's option list: credit if pred equals the gold option's value."""
    if pred is None or gold is None:
        return False
    pred, gold = str(pred).strip(), str(gold).strip()
    if grade_answer(pred, gold):
        return True
    gold_letter = _option_letter(gold)
    if gold_letter is not None:
        if _option_letter(pred) == gold_letter:
            return True
        # model emitted a value, not a letter → map to the gold option's value
        gv = parse_choices(question).get(gold_letter)
        if gv and (grade_answer(pred, gv)
                   or grade_answer(pred, re.sub(r"°|%|\\circ|\bcm\b|\bdegrees?\b", "", gv).strip())):
            return True
    return False


def score_response(response: str, gold: str, question: str | None = None) -> bool:
    """End-to-end: extract the answer from a raw model response, grade vs gold."""
    return grade(extract_pred(response), gold, question)


if __name__ == "__main__":
    # self-test (no GPU)
    cases = [
        ("<answer>42</answer>", "42", True),
        ("...so \\boxed{0.5}", "\\frac{1}{2}", True),
        ("The answer is <answer>C</answer>", "C", True),
        ("<answer>(B)</answer>", "B", True),
        ("<answer>D. 90</answer>", "D", True),
        ("\\boxed{60}", "60", True),
        ("<answer>7</answer>", "8", False),
        ("no answer here", "5", False),
        ("<answer>3.14</answer>", "\\pi", False),  # not equal numerically enough
    ]
    ok = 0
    for resp, gold, exp in cases:
        got = score_response(resp, gold)
        flag = "ok" if got == exp else "MISMATCH"
        if got == exp:
            ok += 1
        print(f"  {flag}  resp={resp!r:40s} gold={gold!r:10s} -> {got} (exp {exp})")
    print(f"\n{ok}/{len(cases)} passed")
