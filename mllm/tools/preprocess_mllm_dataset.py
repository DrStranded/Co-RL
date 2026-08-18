"""Offline preprocess: cap + prune a dataset's train split and save_to_disk.

Run once per (dataset, MAX_SAMPLES). Training then loads the result instantly
via MLLM_PRE_DIR (see `dataset.load_dataset`), skipping the slow single-process
image map (full-res decode + resize) that otherwise runs on every launch and
on every DDP rank. The saved set already has capped (<=1024 long side), pruned,
RGB images, so no per-run map is needed. Eval is NOT saved, it stays live from
MLLM_EVAL_PATH (small).

Usage:
    MAX_SAMPLES=8000 python tools/preprocess_mllm_dataset.py <dataset_name> <out_dir>

Example:
    MAX_SAMPLES=8000 python tools/preprocess_mllm_dataset.py \
        williamium/zwz-37k \
        ./data/mllm_pre/zwz37k_8k
"""

import os
import sys

_REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(_REPO, "trainers"))

import dataset as d  # noqa: E402

# A real eval set anywhere keeps load_dataset on its "train on ALL of train"
# branch (no 150-holdout carve), so the saved set is the full capped train.
# The eval content is irrelevant here; we only save train.
_DEFAULT_EVAL = (
    "./"
    "data/mathvista/testmini_150.jsonl"
)


def main():
    if len(sys.argv) != 3:
        print(__doc__)
        sys.exit(1)
    name, out = sys.argv[1], sys.argv[2]

    # Don't recursively consume a preprocessed dir while preprocessing.
    os.environ.pop("MLLM_PRE_DIR", None)
    if not os.environ.get("MLLM_EVAL_PATH"):
        os.environ["MLLM_EVAL_PATH"] = _DEFAULT_EVAL
        os.environ.setdefault(
            "MLLM_EVAL_IMAGE_DIR",
            os.path.dirname(_DEFAULT_EVAL),
        )

    max_samples = os.environ.get("MAX_SAMPLES", "(all)")
    print(f"[preprocess] dataset={name} max_samples={max_samples} -> {out}")
    train, _eval = d.load_dataset(name)
    os.makedirs(os.path.dirname(out.rstrip("/")), exist_ok=True)
    train.save_to_disk(out)
    print(f"[preprocess] saved {len(train)} rows -> {out}")


if __name__ == "__main__":
    main()
