#!/usr/bin/env python3
"""Environment self-check.

Asserts the installed stack matches requirements.txt, the pip freeze of the
environment behind the paper's LLM results. Exit 0 = versions match.

    python tools/verify.py
"""
import importlib
import sys

# (module, expected version or None = "any, just must import")
CHECKS = [
    ("torch", "2.9.0+cu128"),
    ("transformers", "4.57.0"),
    ("vllm", "0.11.2"),
    ("trl", None),              # the patched fork, version string is upstream-ish
    ("accelerate", "1.13.0"),
    ("deepspeed", "0.18.0"),
    ("wandb", "0.25.1"),
    ("sympy", None),
    ("latex2sympy2", None),
    ("flash_attn", None),       # optional: runs can fall back to sdpa
]
OPTIONAL = {"flash_attn"}

def main() -> int:
    fail = False
    for mod, expected in CHECKS:
        try:
            m = importlib.import_module(mod)
            ver = getattr(m, "__version__", "?")
            ok = expected is None or ver == expected
            print(f"{'  ' if ok else ' !'} {mod:20s} {ver}" + ("" if ok else f"  (expected {expected})"))
            if not ok:
                fail = True
        except Exception as e:
            print(f" x {mod:20s} import FAILED: {e}", file=sys.stderr)
            if mod not in OPTIONAL:
                fail = True

    # The three custom GRPOConfig fields these launchers actually pass. If the TRL pin
    # ever drifts to upstream, this is what catches it.
    try:
        from trl import GRPOConfig
        # The last two are LLM-fork-only, they are what distinguishes it from the MLLM
        # fork, and the math_rephrased / DECOUPLED launchers pass --temperature_eval.
        need = {"num_generations_eval", "scale_rewards", "vllm_importance_sampling_mode",
                "temperature_eval", "entropy_coeff"}
        missing = need - set(GRPOConfig.__dataclass_fields__)
        if missing:
            print(f" x {'GRPOConfig fields':20s} MISSING {sorted(missing)}, wrong trl fork. If temperature_eval/entropy_coeff are the ones missing you have the MLLM fork; see README", file=sys.stderr)
            fail = True
        else:
            print(f"   {'GRPOConfig fields':20s} all present")
    except Exception as e:
        print(f" x {'GRPOConfig':20s} {e}", file=sys.stderr)
        fail = True

    # Training environment only. The evaluation harness installs math-verify in a
    # separate environment (eval/setup_external.sh).
    # math-verify must NOT be installed here: it fights the vendored qwen verifier by pulling
    # antlr4 4.13.2, which breaks the qwen-sympy chain.
    try:
        importlib.import_module("math_verify")
        print(" x math_verify is INSTALLED, it breaks the vendored qwen verifier. "
              "pip uninstall math-verify", file=sys.stderr)
        fail = True
    except ImportError:
        print("   math_verify           absent (correct)")

    print("\n" + ("FAILED, fix the above before training." if fail else "OK, stack matches the training env."))
    return 1 if fail else 0

if __name__ == "__main__":
    sys.exit(main())
