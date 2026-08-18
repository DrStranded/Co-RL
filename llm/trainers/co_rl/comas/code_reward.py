"""Unsupervised CODE reward via run-output majority vote (CoMAS coding data).

Self-contained, imported only through a thin task-routed hook in the trainer;
the existing math reward path is untouched.

Mechanism (mirrors our math majority-vote, but the votable "answer" is the code's
OUTPUT SEQUENCE on a fixed set of inputs instead of a boxed math string):

    extract_calls(test_code)          -> (func_name, [input_arg_tuple, ...])
        parse `assert func(args) == expected` ; take ONLY inputs, drop expected
        outputs -> stays unsupervised (no ground truth in the reward).

    run_outputs(code, func, inputs)   -> tuple(canonical output strings)
        sandboxed exec (timeout + stdout swallow); any failure -> marker string,
        so a broken solution simply won't match the majority.

    output_majority(list_of_output_tuples) -> (majority_tuple, [agree_bool,...])
        the most common output tuple is the pseudo-label; each generation agrees
        iff its output tuple equals it. (Same shape as math self-consistency.)

A code completion's reward, in the cross-supervised trainer, is then exactly the
math case with "answer string" replaced by "output tuple".
"""
import ast
import io
import signal
import sys
import contextlib
from collections import Counter

ERR_NOFUNC = "<NOFUNC>"


# ----------------------------------------------------- extract code from output
import re

_FENCE = re.compile(r"```(?:python)?\s*\n(.*?)```", re.DOTALL)


def extract_code(text: str):
    """Pull the code from a model completion.

    Prefers the last ```python ...``` fenced block (models usually put the final
    solution last); falls back to the whole text if no fence. Returns '' if empty.
    """
    blocks = _FENCE.findall(text)
    if blocks:
        return blocks[-1].strip()
    return text.strip()


def voting_answer(completion_text: str, func_name: str, inputs: list, timeout: float = 2.0):
    """The votable 'answer' for a CODE completion = canonical string of its output
    tuple on the fixed inputs. Mirrors `_extract_and_normalize` for math, so the
    existing majority-vote / exchange / reward pipeline works unchanged.

    Returns None (== 'no answer', drops out of the vote like a math None) if the
    completion yields no usable code or an all-error output.
    """
    code = extract_code(completion_text)
    if not code:
        return None
    outs = run_outputs(code, func_name, inputs, timeout=timeout)
    if not outs or _all_err(outs):
        return None
    return repr(outs)


# ---------------------------------------------------------------- input parsing
def extract_calls(test_code: str):
    """Return (func_name, [arg_tuple, ...]) parsed from the test asserts.

    Expected outputs (`== ...`) are deliberately ignored to keep the reward
    unsupervised. Returns (None, []) if nothing parseable.
    """
    try:
        tree = ast.parse(test_code)
    except Exception:
        return None, []
    func_name, inputs = None, []
    for node in ast.walk(tree):
        if isinstance(node, ast.Assert):
            call = node.test.left if isinstance(node.test, ast.Compare) else node.test
            if isinstance(call, ast.Call) and isinstance(call.func, ast.Name):
                func_name = call.func.id
                try:
                    inputs.append(tuple(ast.literal_eval(a) for a in call.args))
                except Exception:
                    pass
    return func_name, inputs


# --------------------------------------------------------------- code execution
class _TimeLimit(Exception):
    pass


@contextlib.contextmanager
def _time_limit(seconds: float):
    def _handler(signum, frame):
        raise _TimeLimit()
    old = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


def run_outputs(code: str, func_name: str, inputs: list, timeout: float = 2.0):
    """Exec `code`, call func_name(*args) per input, return tuple of repr(output).

    Failures (missing func / exception / timeout) become marker strings so the
    completion fails to match the majority rather than crashing the trainer.
    """
    if not func_name or not inputs:
        return tuple()
    outs = []
    for args in inputs:
        _saved_argv = sys.argv
        sys.argv = [func_name]  # neutralize: model code may call argparse/pytest.main()
        try:
            g = {}
            with _time_limit(timeout), contextlib.redirect_stdout(io.StringIO()):
                exec("from typing import *\n" + code, g)
                fn = g.get(func_name)
                if fn is None:
                    outs.append(ERR_NOFUNC)
                    continue
                result = fn(*args)
            outs.append(repr(result))
        except _TimeLimit:
            outs.append("<TIMEOUT>")
        except SystemExit as e:
            # argparse/pytest.main() raise SystemExit; must catch (BaseException, not Exception)
            outs.append(f"<ERR:SystemExit:{e.code}>")
        except Exception as e:
            outs.append(f"<ERR:{type(e).__name__}>")
        finally:
            sys.argv = _saved_argv
    return tuple(outs)


# ------------------------------------------------- supervised (gt) code reward
def passes_tests(completion_text: str, test_code: str, timeout: float = 4.0) -> bool:
    """GT (supervised) code reward: does the completion's code pass the asserts?

    Exec the extracted completion code together with `test_code` (which defines
    `test_*` functions containing `assert func(...) == expected`), then call every
    test function. True iff all run without raising. This USES the asserts'
    expected outputs, the supervised signal that the unsupervised CoMAS reward
    (run_outputs + output_majority) deliberately drops. Any failure (missing func,
    AssertionError, timeout, SystemExit, exception) -> False (-> reward 0).
    """
    code = extract_code(completion_text)
    if not code or not test_code:
        return False
    _saved_argv = sys.argv
    sys.argv = ["pytest"]  # neutralize argparse/pytest.main() in model code
    try:
        g = {}
        with _time_limit(timeout), contextlib.redirect_stdout(io.StringIO()):
            exec("from typing import *\n" + code + "\n" + test_code, g)
            test_fns = [v for k, v in g.items() if k.startswith("test") and callable(v)]
            if not test_fns:
                return False
            for fn in test_fns:
                fn()
        return True
    except BaseException:
        # BaseException, not Exception: CoMAS test_code uses `pytest.raises(...)`,
        # whose "DID NOT RAISE" failure is `_pytest.outcomes.Failed`, a
        # BaseException subclass that `except Exception` misses, so one bad
        # coding completion would otherwise propagate and crash the trainer.
        return False
    finally:
        sys.argv = _saved_argv


# ------------------------------------------------------------- majority voting
def _all_err(t):
    return len(t) > 0 and all(o.startswith("<") for o in t)


def output_majority(output_tuples: list):
    """Given output tuples from G generations, return (majority_tuple, agree_flags).

    All-error tuples never win the vote (they're degenerate). If every tuple is
    degenerate, majority is None and nothing agrees (-> reward 0, like UNLABELED).
    """
    valid = [t for t in output_tuples if t and not _all_err(t)]
    if not valid:
        return None, [False] * len(output_tuples)
    majority, _ = Counter(valid).most_common(1)[0]
    agree = [t == majority for t in output_tuples]
    return majority, agree
