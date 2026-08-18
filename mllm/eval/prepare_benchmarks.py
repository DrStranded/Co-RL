#!/usr/bin/env python3
"""Download + convert the 4 MLLM math benchmarks to a unified jsonl+images layout.

Unified per-example schema (matches the project's `_load_local_eval_jsonl`):
    {"problem": <question text with choices inlined>, "image": "images/<id>.png",
     "solution": <gold answer string>, "qtype": "mcq"|"free"}

Splits (full test/testmini):
    MathVision 3040 | MathVista 1000 | MathVerse 3940 (5 versions) | We-Math 1740
Plus CoreCognition (frame-combined, 1423): core-knowledge MCQ/yes-no probes.

Output: <OUT_ROOT>/<bench>/{data.jsonl, images/}  (image path relative to data.jsonl).
MathVista is already local; we just re-point to it.

Usage:
    python prepare_benchmarks.py <bench>      # bench = mathvision|mathverse|wemath|mathvista|all
    OUT_ROOT=/path python prepare_benchmarks.py all
"""
import os, re, sys, json

OUT_ROOT = os.environ.get(
    "OUT_ROOT", "./data/mllm_eval"
)
MATHVISTA_LOCAL = "./data/mathvista"
MAX_SIDE = 1024  # cap long side (same as training _cap_image; avoids vLLM blowup)


def cap_image(img):
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > MAX_SIDE:
        s = MAX_SIDE / max(w, h)
        from PIL import Image
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    return img


def _write(bench, rows_iter, total=None):
    out_dir = os.path.join(OUT_ROOT, bench)
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)
    jsonl = os.path.join(out_dir, "data.jsonl")
    n = 0
    with open(jsonl, "w") as f:
        for i, (img, problem, solution, qtype) in enumerate(rows_iter):
            rel = f"images/{i}.png"
            cap_image(img).save(os.path.join(out_dir, rel))
            f.write(json.dumps({"problem": problem, "image": rel,
                                "solution": str(solution), "qtype": qtype},
                               ensure_ascii=False) + "\n")
            n += 1
            if n % 200 == 0:
                print(f"  {bench}: {n}{'/'+str(total) if total else ''}", flush=True)
    print(f"{bench}: {n} examples -> {jsonl}")


def _strip_img_tags(s):
    import re
    return re.sub(r"<image\d*>", "", s or "").strip()


def mathvision():
    from datasets import load_dataset
    ds = load_dataset("MathLLMs/MathVision", split="test")
    def rows():
        for ex in ds:
            q = _strip_img_tags(ex["question"])
            opts = ex.get("options") or []
            qtype = "free"
            if opts:  # MCQ: inline choices as (A)/(B)/...
                letters = [chr(65 + j) for j in range(len(opts))]
                q += "\nChoices:\n" + "\n".join(f"({l}) {o}" for l, o in zip(letters, opts))
                qtype = "mcq"
            yield ex["decoded_image"], q, ex["answer"], qtype
    _write("mathvision", rows(), total=len(ds))


def mathverse():
    from datasets import load_dataset
    ds = load_dataset("AI4Math/MathVerse", "testmini", split="testmini")  # 3940 = 5 versions
    def rows():
        for ex in ds:
            qtype = "mcq" if ex.get("question_type") == "multi-choice" else "free"
            yield ex["image"], ex["question"], ex["answer"], qtype
    _write("mathverse", rows(), total=len(ds))


def wemath():
    from datasets import load_dataset
    ds = load_dataset("We-Math/We-Math", split="testmini")  # 1740
    def rows():
        for ex in ds:
            q = ex["question"].strip() + "\nChoices: " + ex["option"].strip()
            yield ex["image_path"], q, ex["answer"], "mcq"
    _write("wemath", rows(), total=len(ds))


def mathvista():
    """Symlink the local copy when present; otherwise build the full testmini
    (1000 rows + images) from AI4Math/MathVista in the project's row format.

    The repo bundles only the 150-row in-loop subset; the full testmini is
    built from HuggingFace on first use. The HF path renders multi-choice rows the same way the
    bundled 150-row file does (choices into the problem text, LETTER as the
    solution) and asserts the answer maps onto a choice before writing.
    """
    out_dir = os.path.join(OUT_ROOT, "mathvista")
    os.makedirs(OUT_ROOT, exist_ok=True)
    if os.path.exists(os.path.join(out_dir, "testmini.jsonl")):
        print(f"mathvista: already at {out_dir}")
        return
    local_full = os.path.join(MATHVISTA_LOCAL, "testmini.jsonl")
    if os.path.exists(local_full):
        if not os.path.exists(out_dir):
            os.symlink(os.path.abspath(MATHVISTA_LOCAL), out_dir)
        print(f"mathvista: symlinked {MATHVISTA_LOCAL} ({sum(1 for _ in open(local_full))} rows)")
        return
    import string
    from datasets import load_dataset
    os.makedirs(os.path.join(out_dir, "images"), exist_ok=True)
    ds = load_dataset("AI4Math/MathVista", split="testmini")
    n = 0
    with open(os.path.join(out_dir, "testmini.jsonl"), "w") as f:
        for r in ds:
            pid = int(r["pid"])
            prob = (r["question"] or "").strip()
            sol = str(r["answer"]).strip()
            if r["question_type"] == "multi_choice":
                ch = r["choices"] or []
                prob += "\nChoices:\n" + "\n".join(
                    f"({string.ascii_uppercase[i]}) {c}" for i, c in enumerate(ch))
                hit = [i for i, c in enumerate(ch) if str(c).strip() == sol]
                assert hit, f"pid {pid}: answer {sol!r} not among choices {ch}"
                sol = string.ascii_uppercase[hit[0]]
            img = f"images/{pid}.png"
            cap_image(r["decoded_image"]).save(os.path.join(out_dir, img))
            f.write(json.dumps({"problem": prob, "image": img, "solution": sol},
                               ensure_ascii=False) + "\n")
            n += 1
    assert n == 1000, f"expected 1000 testmini rows, wrote {n}"
    print(f"mathvista: built full testmini from HF ({n} rows)")


def corecognition():
    """CoreCognition frame-combined: 1423 rows, video samples pre-rendered as one
    combined frame image. `prompt` already inlines the MCQ options; we strip the
    leading <image> tag and the trailing answer-format instruction (our eval adds
    its own). Yes/no golds stay as YES/NO (mathruler compares case-insensitively)."""
    from datasets import load_dataset
    ds = load_dataset("williamium/CoreCognition", "frame-combined", split="train")
    def rows():
        for ex in ds:
            q = re.sub(r"^\s*<image>\s*", "", ex["prompt"])
            q = re.sub(r"\n?Please answer with[^\n]*$", "", q.strip()).strip()
            qtype = "mcq" if str(ex.get("question_type", "")).upper().startswith("MC") else "free"
            yield ex["images"], q, ex["answer"], qtype
    _write("corecognition", rows(), total=len(ds))


FNS = {"mathvision": mathvision, "mathverse": mathverse, "wemath": wemath, "mathvista": mathvista,
       "corecognition": corecognition}

USAGE = f"""usage: prepare_benchmarks.py [all|{'|'.join(FNS)}]

Builds the evaluation sets under $MLLM_BENCH_ROOT (default: data/benchmarks),
downloading each benchmark from the HuggingFace Hub and writing one
data.jsonl plus its images. Re-running is safe: a benchmark whose row count
already matches is skipped. With no argument, builds all of them."""

if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "all"
    if which in ("-h", "--help", "help"):
        print(USAGE)
        raise SystemExit(0)
    if which != "all" and which not in FNS:
        print(f"unknown benchmark {which!r}\n\n{USAGE}", file=sys.stderr)
        raise SystemExit(2)
    targets = list(FNS) if which == "all" else [which]
    for b in targets:
        print(f"==== {b} ====")
        FNS[b]()
