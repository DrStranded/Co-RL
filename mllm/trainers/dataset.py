"""R1-V style multimodal dataset loader.

The `<answer>...</answer>` tag
extractor is inlined as a simple regex (`_strip_answer_tag`); no external
grader call is needed at dataset-format time.

Each example carries:
- `prompt` (`list[dict]`): R1-V chat format, no system role, multimodal content
- `image` (`PIL.Image`): the input image (HFImage feature, eager-decoded),
  long side capped to `_MAX_LONG_SIDE` so VLM tiling can't exceed max_model_len
- `solution` (`str`): bare ground-truth answer (e.g. '3', '145°'),
  `<answer>...</answer>` wrapper stripped at load time

R1-V prompt template:
    "{Question} Output the thinking process in <think> </think> and
    final answer in <answer> </answer> tags."

Beyond CLEVR/GEOQA, seven additional sources are supported via `_SPECS`
(zwz-37k, MMFineReason rl/sft, OpenMMReasoner-74K, multimodal-open-r1-8k,
geometry3k, MMR1-Math). Each spec maps the source schema → {prompt, image,
solution}. Sentinels: question="@chat" reads the chat-list `prompt` column;
answer="@reward" reads reward_model["ground_truth"]; answer="@sol_answer"
reads the `<answer>` tag out of the `solution` column.

Env vars (optional):
    MLLM_EVAL_PATH: jsonl eval set, schema `{"problem","image","solution"}`/line
    MLLM_EVAL_IMAGE_DIR: dir for relative image paths in MLLM_EVAL_PATH
    MAX_SAMPLES: truncate train to first N (sanity / debug only)
"""

import json
import os
import re
from pathlib import Path

from PIL import Image
from datasets import Dataset
from datasets import Image as HFImage
from datasets import load_dataset as hf_load_dataset


# Training datasets (HuggingFace Hub IDs)
CLEVR_COUNTING_DATASET = "leonardPKU/clevr_cogen_a_train"
GEOQA_DATASET = "leonardPKU/GEOQA_R1V_Train_8K"

# Additional MLLM training datasets. Each spec maps the source schema → our
# {prompt, image, solution}. Sentinels: question="@chat" reads the chat-list
# `prompt` column (OpenMMReasoner); answer="@reward" reads
# reward_model["ground_truth"]; answer="@sol_answer" reads the `<answer>` tag
# out of the `solution` column (open-r1). MMFineReason is one repo with two
# splits (rl / sft), the `#sft` key selects the sft split via spec["hf_id"].
# "<image>" placeholders in question text are stripped (the prompt builder
# injects the image part separately).
ZWZ_37K = "williamium/zwz-37k"
MMFINEREASON = "OpenDataArena/MMFineReason-1.8M-Qwen3-VL-235B-Thinking"
MMFINEREASON_SFT = MMFINEREASON + "#sft"
OPENMMREASONER = "OpenMMReasoner/OpenMMReasoner-RL-74K"
OPEN_R1_8K = "lmms-lab/multimodal-open-r1-8k-verified"
GEOMETRY3K = "hiyouga/geometry3k"
MMR1_MATH = "MMR1/MMR1-Math-RL-Data-v0"

_OPENMMR_SUBSETS = ["virl39k", "thinklite_vl_hard", "tqa_train",
                    "wemath_standard", "mmk12", "wemath_pro", "algopuzzle"]

_SPECS = {
    ZWZ_37K:          dict(subset="37k", split="train", image="images", question="problem", answer="answer"),
    MMFINEREASON:     dict(split="rl",   image="image",  question="question", answer="answer", mmfr_filter=True),
    MMFINEREASON_SFT: dict(hf_id=MMFINEREASON, split="sft", image="image", question="question", answer="answer"),
    OPENMMREASONER:   dict(concat=_OPENMMR_SUBSETS, split="train", image="images", question="@chat", answer="@reward"),
    OPEN_R1_8K:       dict(split="train", image="image",  question="problem", answer="@sol_answer"),
    GEOMETRY3K:       dict(split="train", image="images", question="problem", answer="answer"),
    MMR1_MATH:        dict(split="train", image="images", question="problem", answer="answer"),
}

_VALIDATION_SIZE = 150
_VALIDATION_SEED = 42

_PROMPT_SUFFIX = (
    " Output the thinking process in <think> </think> and final answer in "
    "<answer> </answer> tags."
)

_ANSWER_TAG_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)

# Cap the longest image side so Qwen2.5-VL's native dynamic-resolution tiling
# can't blow past vllm_max_model_length, and (critically) so full-res image
# bytes don't overflow pyarrow's int32 2GB per-column offset when datasets.map
# combines a writer_batch into one Arrow shard. No-op for InternVL (1 tile) and
# already-small GeoQA diagrams.
_MAX_LONG_SIDE = 1024


def _strip_answer_tag(text):
    """Return content inside <answer>...</answer>, or None if absent."""
    if text is None:
        return None
    m = _ANSWER_TAG_RE.search(str(text))
    return m.group(1).strip() if m else None


def _make_prompt(question_text):
    """R1-V style prompt: no system role, multimodal user content (image + text).

    Content **must** be a list with an explicit `{"type": "image"}` part, both
    Qwen2.5-VL and InternVL3.5 chat templates branch on
    `message['content'] is string`:
      - string content → text is rendered as-is, **no image placeholder emitted**
      - list content   → each `{"type": "image"}` part emits the model's image
        placeholder token(s), required for vLLM mm processing and model forward.
    """
    return [{
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": f"{question_text}{_PROMPT_SUFFIX}"},
        ],
    }]


def _cap_image(img):
    """RGB + downscale so max(w, h) <= _MAX_LONG_SIDE (preserves aspect ratio)."""
    if img.mode != "RGB":
        img = img.convert("RGB")
    w, h = img.size
    long_side = max(w, h)
    if long_side > _MAX_LONG_SIDE:
        scale = _MAX_LONG_SIDE / long_side
        img = img.resize((max(1, round(w * scale)), max(1, round(h * scale))))
    return img


def _convert_to_rgb(example):
    """Ensure image is RGB and capped to `_MAX_LONG_SIDE` on the long edge."""
    example["image"] = _cap_image(example["image"])
    return example


def _load_local_eval_jsonl(jsonl_path, image_dir):
    image_dir = Path(image_dir) if image_dir is not None else None
    records = []
    with open(jsonl_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            img_path = row["image"]
            if image_dir is not None and not os.path.isabs(img_path):
                img_path = image_dir / img_path
            with Image.open(img_path) as im:
                im.load()
                im = _cap_image(im)
            records.append({
                "prompt": _make_prompt(row["problem"]),
                "image": im,
                "solution": row["solution"],
            })
    return Dataset.from_list(records)


def _extract_chat_text(prompt):
    """Pull the user-turn text out of a chat-list `prompt` (OpenMMReasoner).

    `content` may be a plain string or a list of `{type, text}` parts. Returns
    the concatenated user text with any `<image>` placeholder stripped.
    """
    parts = []
    for msg in prompt:
        if msg.get("role") != "user":
            continue
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for p in content:
                if isinstance(p, dict) and p.get("text"):
                    parts.append(p["text"])
    return " ".join(parts).replace("<image>", "").strip()


def _load_spec_dataset(dataset_name):
    """Load + normalize one of the `_SPECS` datasets → {prompt, image, solution}."""
    spec = _SPECS[dataset_name]
    hf_id = spec.get("hf_id", dataset_name)
    split = spec.get("split", "train")

    if "concat" in spec:
        from datasets import concatenate_datasets
        ds = concatenate_datasets(
            [hf_load_dataset(hf_id, name=s, split=split) for s in spec["concat"]]
        )
    elif "subset" in spec:
        ds = hf_load_dataset(hf_id, name=spec["subset"], split=split)
    else:
        ds = hf_load_dataset(hf_id, split=split)

    if spec.get("mmfr_filter"):
        # RL split: keep self-consistent, verifiable, non-degenerate items
        # (both judges agree, and 0 < pass_rate < 1 so reward has variance).
        ds = ds.filter(lambda r: r["is_consistent"] and 0.0 < r["pass_rate"] < 1.0)

    img_f, q_f, a_f = spec["image"], spec["question"], spec["answer"]

    def _fmt(ex):
        img = ex[img_f]
        if isinstance(img, list):
            img = img[0]
        # Cap BEFORE the map writes to Arrow. Full-res images (~2MB each) blow
        # past pyarrow's 2GB int32 offset limit once writer_batch_size rows of
        # image bytes are combined into a single shard (ArrowInvalid: offset
        # overflow). Capping here keeps each image ~200-500KB; also speeds the
        # map and shrinks the on-disk cache. (Distinct from the column prune
        # below, which fixes CPU-RAM OOM, this fixes the Arrow write-size limit.)
        img = _cap_image(img)
        if q_f == "@chat":
            question = _extract_chat_text(ex["prompt"])
        else:
            question = str(ex[q_f]).replace("<image>", "").strip()
        if a_f == "@reward":
            ans = ex["reward_model"]["ground_truth"]
        elif a_f == "@sol_answer":
            # open-r1: `original_answer` is prose; the clean gold lives in the
            # `solution` field's <answer>...</answer> tag. Fall back to the raw
            # original_answer only if no tag is present.
            ans = _strip_answer_tag(ex["solution"]) or ex.get("original_answer", "")
        else:
            ans = ex[a_f]
        return {"prompt": _make_prompt(question), "image": img, "solution": str(ans).strip()}

    # Prune columns the formatter never reads BEFORE the map. zwz carries a SECOND
    # full-res image column (`original_images`, 2250x1500) + bbox/extra_info; left
    # in, every row decodes ~2x the images, and 8 DDP ranks mapping 37k pairs on
    # one node exhaust CPU RAM → OOM-killed mid-map with no traceback. Keep only
    # what `_fmt` reads.
    _keep = {img_f, "prompt" if q_f == "@chat" else q_f}
    if a_f == "@reward":
        _keep.add("reward_model")
    elif a_f == "@sol_answer":
        _keep.update(("solution", "original_answer"))
    else:
        _keep.add(a_f)
    ds = ds.remove_columns([c for c in ds.column_names if c not in _keep])

    # MAX_SAMPLES truncates BEFORE the (image-heavy) map so debug/sanity runs on
    # huge sources (e.g. MMFineReason-sft 1.77M) don't map the whole split.
    _max = os.environ.get("MAX_SAMPLES")
    if _max is not None:
        ds = ds.select(range(min(int(_max), len(ds))))
    # writer_batch_size kept small as a second guard against the 2GB Arrow offset
    # overflow (primary guard is the _cap_image call in _fmt): 100 capped images
    # per shard is comfortably under the int32 offset limit even for large images.
    return ds.map(_fmt, remove_columns=ds.column_names, writer_batch_size=100)


def load_dataset(dataset_name):
    """Load (train, eval) datasets for the given dataset name.

    Args:
        dataset_name (`str`): a `_SPECS` key, or `CLEVR_COUNTING_DATASET` /
            `GEOQA_DATASET`.

    Returns:
        `tuple[Dataset, Dataset]`: each row has `prompt` (list), `image` (PIL),
        `solution` (str, bare answer, `<answer>` wrapper pre-stripped).

    Env vars (optional):
        MLLM_EVAL_PATH: jsonl eval set; if set, train on ALL of train (no holdout).
        MLLM_EVAL_IMAGE_DIR: dir for relative image paths in MLLM_EVAL_PATH.
        MAX_SAMPLES: truncate train to first N (sanity / debug only).
        MLLM_PRE_DIR: opt-in fast path, a pre-capped/pruned train set saved
            offline by tools/preprocess_mllm_dataset.py. Skips the slow
            single-process image map entirely. Eval still comes from
            MLLM_EVAL_PATH (small, live). Unset → default path below, unchanged.
    """
    pre_dir = os.environ.get("MLLM_PRE_DIR")
    if pre_dir:
        from datasets import load_from_disk
        # Provenance: MLLM_PRE_DIR silently overrides `dataset_name`, so the run's
        # name, banner and output_dir can all disagree with what was really trained
        # on. Log the resolved path + row count (the only signal that cannot lie),
        # and let callers pin the expected corpus via MLLM_PRE_REQUIRE.
        require = os.environ.get("MLLM_PRE_REQUIRE")
        if require and require not in pre_dir:
            raise SystemExit(
                f"[run] FATAL: MLLM_PRE_DIR={pre_dir} does not contain "
                f"MLLM_PRE_REQUIRE={require!r} - refusing to train on the wrong corpus"
            )
        train_dataset = load_from_disk(pre_dir)
        print(f"[run] train set: {pre_dir} rows={len(train_dataset)}", flush=True)
        eval_path = os.environ.get("MLLM_EVAL_PATH")
        if eval_path is not None:
            eval_dataset = _load_local_eval_jsonl(eval_path, os.environ.get("MLLM_EVAL_IMAGE_DIR"))
        else:
            split = train_dataset.train_test_split(test_size=_VALIDATION_SIZE, seed=_VALIDATION_SEED)
            train_dataset, eval_dataset = split["train"], split["test"]
        max_samples = os.environ.get("MAX_SAMPLES")
        if max_samples is not None:
            train_dataset = train_dataset.select(range(min(int(max_samples), len(train_dataset))))
        return train_dataset, eval_dataset

    if dataset_name in _SPECS:
        full_train = _load_spec_dataset(dataset_name)
    elif dataset_name in (CLEVR_COUNTING_DATASET, GEOQA_DATASET):
        raw = hf_load_dataset(dataset_name)
        train_split = raw["train"]
        columns = set(train_split.column_names)
        # R1-V datasets standardize on `problem` + `solution` + `image`.
        if not {"image", "problem", "solution"} <= columns:
            raise ValueError(
                f"Dataset '{dataset_name}' must have 'problem'/'solution'/'image'. "
                f"Found columns: {columns}"
            )

        def _format(example):
            # CLEVR/GEOQA store solution as '<answer> X </answer>'. Strip the
            # wrapper so `solution` is the bare gold (e.g. '3', '145°'), matches
            # what reward_correctness extracts from completions.
            raw_sol = str(example["solution"])
            stripped = _strip_answer_tag(raw_sol)
            return {
                "prompt": _make_prompt(example["problem"]),
                "image": example["image"],
                "solution": stripped if stripped is not None else raw_sol.strip(),
            }

        full_train = train_split.map(_format, remove_columns=train_split.column_names)
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported: "
            f"GEOQA/CLEVR + {sorted(_SPECS)}"
        )

    # Decode `image` → PIL (no-op if already) and cap long side. Shared by all
    # sources so Qwen's dynamic-resolution tiling can't exceed max_model_len.
    if not isinstance(full_train.features["image"], HFImage):
        full_train = full_train.cast_column("image", HFImage())
    full_train = full_train.map(_convert_to_rgb, writer_batch_size=100)

    eval_path = os.environ.get("MLLM_EVAL_PATH")
    if eval_path is not None:
        # A real fixed eval set is provided → train on ALL of full_train (no
        # 150-holdout carve, which also avoids train_test_split crashing when
        # MAX_SAMPLES truncates full_train below _VALIDATION_SIZE).
        train_dataset = full_train
        eval_dataset = _load_local_eval_jsonl(eval_path, os.environ.get("MLLM_EVAL_IMAGE_DIR"))
    else:
        # No eval set: carve a 150 holdout from train (seed 42).
        split = full_train.train_test_split(test_size=_VALIDATION_SIZE, seed=_VALIDATION_SEED)
        train_dataset, eval_dataset = split["train"], split["test"]

    max_samples = os.environ.get("MAX_SAMPLES")
    if max_samples is not None:
        train_dataset = train_dataset.select(range(min(int(max_samples), len(train_dataset))))

    return train_dataset, eval_dataset


__all__ = [
    "CLEVR_COUNTING_DATASET",
    "GEOQA_DATASET",
    "ZWZ_37K",
    "MMFINEREASON",
    "MMFINEREASON_SFT",
    "OPENMMREASONER",
    "OPEN_R1_8K",
    "GEOMETRY3K",
    "MMR1_MATH",
    "load_dataset",
]
