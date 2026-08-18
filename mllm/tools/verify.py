"""One-shot install verifier for the vision-language environment.

The expected versions are the frozen environment behind the paper's VLM runs.
Do NOT target torch2.10 / vllm0.18 / transformers5.x, those are the old
the documented versions and are NOT what this repo runs.

Expected versions (frozen):
    torch         2.9.0+cu128    (cu128 runtime)
    trl           1.5.0.dev0     (DrStranded fork, editable)
    transformers  4.57.0
    vllm          0.11.2
    deepspeed     0.18.0
    flash_attn    2.8.3          (cu128 / torch2.9, cxx11abi FALSE)

MLLM extras (should be in the frozen venv):
    qwen-vl-utils 0.0.14, opencv-python-headless, timm, av

Grader path (qwen-sympy via 2-hop wrapper, NOT math-verify):
    co_label_utils.py:25 -> verifiers/math_verify_wrapper.py:44 -> verifiers/qwen/math_grade.py

Two kinds of check:
    kind="pure" , pure-python (versions / registry / extras / grader). Safe to run
                   at DOCKER BUILD time as a build-gate. Guards every optional import
                   so a missing dep is reported, never an uncaught crash.
    kind="gpu"  , RUNTIME-ONLY (needs a live GPU): cuda bf16 matmul, flash-attn forward.

Run modes:
    python tools/verify.py            # all 6 checks (runtime; needs GPU)
    python tools/verify.py --pure     # only the 4 pure-python checks (build-gate; no GPU)
    VERIFY_MODE=build python tools/verify.py   # same as --pure
"""

from __future__ import annotations

import importlib
import inspect
import os
import sys
import traceback

PASS_MARK = "ok"
FAIL_MARK = "FAIL"
# each entry: (name, fn, kind)  where kind ∈ {"pure", "gpu"}
CHECKS: list[tuple[str, callable, str]] = []


def check(name: str, kind: str = "pure"):
    def deco(fn):
        CHECKS.append((name, fn, kind))
        return fn
    return deco


def _dist_version(dist_name: str) -> str:
    """Distribution version via stdlib importlib.metadata (no pkg_resources)."""
    try:
        from importlib.metadata import PackageNotFoundError, version
        return version(dist_name)
    except Exception:
        return "unknown"


# ----- Check 1: torch + CUDA bf16 matmul (RUNTIME-ONLY) -----
@check("[1/6] torch + CUDA (runtime)", kind="gpu")
def _torch_cuda():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available()=False: driver or CUDA toolkit broken"
    cap = torch.cuda.get_device_capability(0)
    name = torch.cuda.get_device_name(0)
    runtime = torch.version.cuda
    print(f"   torch {torch.__version__}, cuda runtime {runtime}, GPU '{name}' cap {cap}")
    assert runtime.startswith("12.8") or runtime.startswith("12.9"), (
        f"cuda runtime {runtime} is not 12.8; the frozen stack is torch2.9.0+cu128"
    )
    x = torch.randn(64, 64, device="cuda", dtype=torch.bfloat16)
    y = x @ x.T
    assert y.shape == (64, 64)
    print(f"   bf16 matmul OK on {name}")


# ----- Check 2: locked version table (pure) -----
@check("[2/6] frozen stack versions", kind="pure")
def _versions():
    # (import name, expected-prefix). trl/torch/flash_attn may carry +local / .devN suffixes.
    specs = [
        ("torch",        "2.9.0"),
        ("trl",          "1.5.0"),
        ("transformers", "4.57.0"),
        ("vllm",         "0.11.2"),
        ("deepspeed",    "0.18.0"),
        ("flash_attn",   "2.8.3"),
    ]
    bad = []
    for pkg, want in specs:
        try:
            mod = importlib.import_module(pkg)
            raw = getattr(mod, "__version__", None) or _dist_version(pkg)
            got = str(raw).split("+")[0].split(".dev")[0]
            ok = got.startswith(want)
        except Exception as e:
            got = f"<import failed: {type(e).__name__}: {e}>"
            ok = False
        mark = PASS_MARK if ok else FAIL_MARK
        print(f"   {mark} {pkg:14s} got {got:26s} want {want}")
        if not ok:
            bad.append(pkg)
    if bad:
        raise AssertionError(
            f"version mismatch or missing on {bad}."
            f" Expected torch2.9.0 / transformers4.57.0 / vllm0.11.2 / deepspeed0.18.0"
            f" / flash-attn2.8.3 / trl1.5.0。"
            f" Never install transformers 5.x / vllm 0.18+ / flash-attn cxx11abi TRUE。"
        )


# ----- Check 3: flash-attn forward (ABI sanity, RUNTIME-ONLY) -----
@check("[3/6] flash-attn forward(ABI, runtime)", kind="gpu")
def _flash_attn_forward():
    import torch
    from flash_attn import flash_attn_func
    q = k = v = torch.randn(1, 4, 1, 64, dtype=torch.float16, device="cuda")
    out = flash_attn_func(q, k, v)
    assert out.shape == q.shape
    print(f"   flash_attn_func OK, out shape {tuple(out.shape)}")


# ----- Check 4: MLLM extras present (pure) -----
@check("[4/6] MLLM misc invariants (the pinned stack)", kind="pure")
def _mllm_extras():
    missing = []
    for import_name, dist_name in [
        ("qwen_vl_utils", "qwen-vl-utils"),
        ("cv2",           "opencv-python-headless"),
        ("timm",          "timm"),
        ("av",            "av"),
    ]:
        try:
            mod = importlib.import_module(import_name)
            ver = getattr(mod, "__version__", None) or _dist_version(dist_name)
            print(f"   {PASS_MARK} {dist_name:24s} {ver}")
        except Exception as e:
            print(f"   {FAIL_MARK} {dist_name:24s} import failed: {type(e).__name__}: {e}")
            missing.append(dist_name)

    # math-verify MUST NOT be present (the grader uses the qwen-sympy 2-hop wrapper)
    try:
        importlib.import_module("math_verify")
        raise AssertionError(
            "math-verify is installed. The grader uses the qwen-sympy wrapper, so it is dead weight, "
            "and worse, it drags in a newer antlr4 that breaks the latex2sympy2 1.9.x path. "
            "Remove: `pip uninstall -y math-verify latex2sympy2_extended`"
        )
    except ImportError:
        print(f"   {PASS_MARK} math-verify absent (expected)")

    if missing:
        raise AssertionError(
            f"missing extras {missing}: the pinned stack includes all of them; check requirements.txt installed fully."
        )


# ----- Check 5: AutoConfig registry has our MLLM model types (pure) -----
@check("[5/6] AutoConfig registry(transformers 4.57.0)", kind="pure")
def _autoconfig_registry():
    from transformers import CONFIG_MAPPING

    needed = ["qwen2_5_vl", "internvl", "gemma3"]
    missing = [m for m in needed if m not in CONFIG_MAPPING]
    for m in needed:
        mark = PASS_MARK if m in CONFIG_MAPPING else FAIL_MARK
        print(f"   {mark} model_type '{m}'")
    if missing:
        raise AssertionError(
            f"transformers 4.57.0 missing {missing}: all should be registered. Check the transformers version."
        )

    # NOTE: qwen3_vl / qwen3_vl_moe were once wrongly assumed to be transformers 5.x-only,
    #       but they exist in 4.57.x already. Asserting their absence would falsely
    #       abort the build gate on the frozen tf4.57.0, so presence is only reported.
    for m in ["qwen3_vl", "qwen3_vl_moe"]:
        if m in CONFIG_MAPPING:
            print(f"   note: model_type '{m}' present (shipped since 4.57.x, not an error)")

    # Genuinely transformers-5.x-only model_types (e.g. gemma4) should be absent;
    # their presence is a WARNING (a 5.x install), not an assert, to keep the build gate alive.
    future_only = ["gemma4"]
    leaked = [m for m in future_only if m in CONFIG_MAPPING]
    if leaked:
        print(f"   WARN registry contains {leaked}: these were added in transformers 5.x; "
              f"the frozen stack expects 4.57.0. Run `pip show transformers` to check.")
    else:
        print(f"   {PASS_MARK} no 5.x-only model_type {future_only} (expected)")


# ----- Check 6: grader 2-hop callable + trl EOS patch (pure) -----
@check("[6/6] grader 2-hop(qwen-sympy)+ trl EOS patch", kind="pure")
def _grader_and_eos_patch():
    # 6a: trl GRPOTrainer EOS patch present (Phi-3.5 / Qwen3 list-eos fix)
    #      a failed trl import must not kill the whole check (trl may be absent in the pure-python build gate).
    try:
        from trl.trainer.grpo_trainer import GRPOTrainer
        src = inspect.getsource(GRPOTrainer.__init__)
        if "self.eos_token_ids" not in src:
            print(f"   WARN trl GRPOTrainer lacks the self.eos_token_ids EOS patch")
            print(f"   WARN Phi-3.5 / Qwen3 list-typed eos_token_id will falsely report clipped_ratio=0.97")
            print(f"   WARN the patch lives in the LLM repo trl/; the DrStranded/trl fork has not synced it. "
                  "MLLM-only runs (Qwen2.5-VL/InternVL/Gemma-3) are unaffected; sync before sharing with LLM.")
        else:
            print(f"   {PASS_MARK} trl EOS patch detected")
    except Exception as e:
        print(f"   WARN trl check skipped (import failed: {type(e).__name__}: {e})")

    # 6b, qwen-sympy 2-hop grader callable
    candidate_paths = [
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "trainers"),
    ]
    wrapper_seen = False
    last_err = None
    for p in candidate_paths:
        wrapper = os.path.join(p, "verifiers", "math_verify_wrapper.py")
        if os.path.isfile(wrapper):
            wrapper_seen = True
            if p not in sys.path:
                sys.path.insert(0, p)
            try:
                from verifiers.math_verify_wrapper import grade_answer
                # a Unicode degree-sign case (what the wrapper exists for)
                ok_unicode = grade_answer("90°", "90")
                # LaTeX equivalence (what qwen-sympy exists for)
                ok_latex = grade_answer("1/2", "\\frac{1}{2}")
                assert ok_unicode and ok_latex, (
                    f"grader wrong: Unicode '90 deg' vs '90' = {ok_unicode}, "
                    f"LaTeX '1/2' vs '\\frac{{1}}{{2}}' = {ok_latex}"
                )
                print(f"   {PASS_MARK} qwen-sympy 2-hop grader callable")
                print(f"     - grade('90°', '90') = True(wrapper Unicode strip)")
                print(f"     - grade('1/2', '\\\\frac{{1}}{{2}}') = True(qwen-sympy LaTeX equiv)")
                return
            except Exception as e:
                last_err = e
                print(f"   WARN grader attempt failed at {p}: {type(e).__name__}: {e}")

    if wrapper_seen:
        raise AssertionError(
            f"grader wrapper found but calling it failed (commonly missing pylatexenc / latex2sympy2 / "
            f"antlr4-python3-runtime, all pinned in requirements.txt). Last error: {last_err}"
        )
    raise AssertionError(
        "grader wrapper not found (trainers/verifiers/math_verify_wrapper.py). "
        "Not running from the mllm/ root, or trainers/verifiers/ is missing."
    )


# ----- Run -----
def main(pure_only: bool = False) -> int:
    selected = [(n, f, k) for (n, f, k) in CHECKS if (k == "pure" or not pure_only)]
    mode = "PURE (build-gate, no GPU)" if pure_only else "FULL (runtime, needs GPU)"
    print("=" * 72)
    print(f"  verify.py, pinned stack, mode: {mode}")
    print("=" * 72)
    fail = 0
    for name, fn, kind in selected:
        print(f"\n{name}")
        try:
            fn()
            print(f"   {PASS_MARK} pass")
        except Exception as e:
            fail += 1
            print(f"   {FAIL_MARK} {type(e).__name__}: {e}")
            traceback.print_exc(limit=2)
    print()
    print("=" * 72)
    if fail == 0:
        print(f"  {PASS_MARK} ALL {len(selected)} CHECKS PASSED")
        print("=" * 72)
        print("Pinned stack verified.")
        return 0
    else:
        print(f"  {FAIL_MARK} {fail}/{len(selected)} CHECKS FAILED")
        print("=" * 72)
        return 1


if __name__ == "__main__":
    _pure = (
        "--pure" in sys.argv
        or os.environ.get("VERIFY_MODE", "").lower() in ("build", "pure")
        or os.environ.get("VERIFY_PURE", "") == "1"
    )
    sys.exit(main(pure_only=_pure))
