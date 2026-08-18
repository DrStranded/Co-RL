"""Dataset loading for co_rl.

Self-contained per repo "trainer self-contained, share by copy" convention.

Set MATH500_EVAL_PATH=data/math500/test.json (relative to repo root) to use
the MATH-500 validation set (industry standard, used by MARTI / SimpleRL-Zoo).
Without this env var, a 150-prompt holdout is carved from the train split.

Co-rewarding-I replication uses a private HF dataset with two row-aligned configs
(q1716523669/MATH-Level345-Rephrased-DeepSeek):
  - config 'original'  (MATH-Level345 questions, verl format)
  - config 'rephrased' (same rows positionally, DeepSeek-rephrased; semantically
                        equivalent question + identical answer)
Both configs share schema and are aligned by row index (extra_info.index). Override
the repo via the COREWARDING_HF_REPO env var. The dataset is private, so HF_TOKEN
must be set in the environment at load time.
"""

import json
import os

from datasets import Dataset
from datasets import load_dataset as hf_load_dataset

OPSD_DATASET = "siyanzhao/Openthoughts_math_30k_opsd"
DAPO_DATASET = "open-r1/DAPO-Math-17k-Processed"
MATH_LEVEL345_DATASET = "q1716523669/MATH-Level345"
MATH_LEVEL12345_DATASET = "q1716523669/MATH-Level12345"

# Co-rewarding-I replication: paired HF-dataset configs (original ↔ rephrased).
# Selected via these sentinel "dataset names"; the actual HF repo/config is
# resolved in _load_coreward at load time (see _COREWARDING_HF_REPO).
COREWARDING_MATH_ORIGINAL = "coreward/math_original"
COREWARDING_MATH_REPHRASED = "coreward/math_rephrased"

# CoMAS (arXiv 2510.08529) training data, we use ONLY their data; our model /
# method / setting are unchanged. Reward is task-routed (see reward_correctness):
# math/science -> qwen math verifier (grade_answer); coding -> run-output majority
# (comas/code_reward). `comas/blended` (5000: 2000 science + 1500 coding + 1500
# math): math + coding both get valid reward; science also routes through the math
# verifier, so its reward is only valid where the science answer is sympy-gradable
# (boxed/numeric). `comas/math` (the 1500 math subset) is fully verifier-native.
COMAS_BLENDED = "comas/blended"
COMAS_MATH = "comas/math"
_COMAS_DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "data", "comas")

_VALIDATION_SIZE = 150
_VALIDATION_SEED = 42

_INSTRUCTION = "Please reason step by step, and put your final answer within \\boxed{}."

_COREWARDING_HF_REPO = os.environ.get("COREWARDING_HF_REPO", "q1716523669/MATH-Level345-Rephrased-DeepSeek")


def _make_prompt(text):
    return [{"role": "user", "content": f"{text}\n {_INSTRUCTION}"}]


def _load_math500_eval(path):
    with open(path) as f:
        data = json.load(f)
    return Dataset.from_list([
        {"prompt": _make_prompt(e["prompt"]), "solution": e["answer"]}
        for e in data
    ])


def _load_comas_json(which: str) -> Dataset:
    """Load CoMAS data (json list of {prompt, answer, task, ...}) -> {prompt, solution}.

    `which` ∈ {'blended' (5000: math+science+coding), 'math' (1500 math subset)}.
    Path from COMAS_DATA_DIR env var, else repo-local data/comas/.
    """
    base = os.environ.get("COMAS_DATA_DIR", _COMAS_DATA_DIR)
    fname = {"blended": "blended_train.json", "math": "math_train.json"}[which]
    path = os.path.join(os.path.expanduser(base), fname)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"CoMAS data not found at {path}. Expected blended_train.json / "
            f"math_train.json under data/comas/ (or set COMAS_DATA_DIR)."
        )
    with open(path) as f:
        data = json.load(f)
    # `task` routes the reward (math/science -> sympy grade; coding -> run-output
    # majority). `test_code` is a SEPARATE persistent column: for coding it holds
    # the test asserts (the trainer extracts call-inputs from it). It must NOT be
    # `solution`, because the trainer overwrites `solution` with the peer pseudo-
    # label during cross-supervision, whereas `test_code` has to survive to the
    # reward fn. Empty string for non-coding rows (keeps the column uniform).
    return Dataset.from_list([
        {
            "prompt": _make_prompt(e["prompt"]),
            "solution": str(e["answer"]),
            "task": e.get("task", "math"),
            "test_code": e["answer"] if e.get("task") == "coding" else "",
        }
        for e in data
    ])


def _load_coreward(which: str) -> Dataset:
    """Load the Co-rewarding verl-format pair from HF and convert to {prompt, solution}.

    `which` ∈ {'original', 'rephrased'} selects the HF config of _COREWARDING_HF_REPO.
    Both configs hold the same rows positionally (extra_info.index), so the original ↔
    rephrased alignment holds end-to-end. The verl `prompt` is a list of {role, content}
    dicts (system + user); we extract the user content, drop the verl system message (its
    'reason step by step + boxed' instruction is duplicated by our own _INSTRUCTION), and
    wrap via _make_prompt. `solution` is taken from `reward_model.ground_truth` (string).
    The dataset is private, so HF_TOKEN must be set in the environment.
    """
    hf = hf_load_dataset(_COREWARDING_HF_REPO, which, split="train",
                         token=os.environ.get("HF_TOKEN"))
    records = []
    for row in hf:
        # `prompt` is a list of dicts; pull the user message.
        msgs = row["prompt"]
        user_content = None
        for m in msgs:
            if m.get("role") == "user":
                user_content = m["content"]
                break
        if user_content is None:
            raise ValueError(f"no user role in row prompt: {msgs}")
        # `reward_model` is a dict with `ground_truth` (string) + `style`.
        solution = row["reward_model"]["ground_truth"]
        records.append({
            "prompt": _make_prompt(user_content),
            "solution": str(solution),
        })
    return Dataset.from_list(records)


def load_dataset(dataset_name):
    # CoMAS data branch, local json, not HF Hub. Same downstream handling
    # (150-prompt val split + optional MATH500_EVAL_PATH override) as the others.
    if dataset_name in (COMAS_BLENDED, COMAS_MATH):
        which = "blended" if dataset_name == COMAS_BLENDED else "math"
        full_train = _load_comas_json(which)
        split = full_train.train_test_split(test_size=_VALIDATION_SIZE, seed=_VALIDATION_SEED)
        train_dataset, eval_dataset = split["train"], split["test"]
        max_samples = os.environ.get("MAX_SAMPLES")
        if max_samples is not None:
            n = min(int(max_samples), len(train_dataset))
            train_dataset = train_dataset.select(range(n))
        math500_path = os.environ.get("MATH500_EVAL_PATH")
        if math500_path is not None:
            eval_dataset = _load_math500_eval(math500_path)
        return train_dataset, eval_dataset

    # Co-rewarding-I branch, paired HF-dataset configs (original ↔ rephrased).
    if dataset_name in (COREWARDING_MATH_ORIGINAL, COREWARDING_MATH_REPHRASED):
        which = "original" if dataset_name == COREWARDING_MATH_ORIGINAL else "rephrased"
        full_train = _load_coreward(which)
        split = full_train.train_test_split(test_size=_VALIDATION_SIZE, seed=_VALIDATION_SEED)
        train_dataset, eval_dataset = split["train"], split["test"]
        max_samples = os.environ.get("MAX_SAMPLES")
        if max_samples is not None:
            n = min(int(max_samples), len(train_dataset))
            train_dataset = train_dataset.select(range(n))
        math500_path = os.environ.get("MATH500_EVAL_PATH")
        if math500_path is not None:
            eval_dataset = _load_math500_eval(math500_path)
        return train_dataset, eval_dataset

    if dataset_name == OPSD_DATASET:
        format_prompt = lambda example: {
            "prompt": _make_prompt(example["problem"]),
            "solution": example["Answer"],
        }
    elif dataset_name == DAPO_DATASET:
        format_prompt = lambda example: {
            "prompt": _make_prompt(example["prompt"]),
            "solution": example["solution"],
        }
    elif dataset_name in (MATH_LEVEL345_DATASET, MATH_LEVEL12345_DATASET):
        format_prompt = lambda example: {
            "prompt": _make_prompt(example["prompt"]),
            "solution": example["answer"],
        }
    else:
        raise ValueError(
            f"Unsupported dataset '{dataset_name}'. Supported datasets: "
            f"'{OPSD_DATASET}', '{DAPO_DATASET}', "
            f"'{MATH_LEVEL345_DATASET}', '{MATH_LEVEL12345_DATASET}', "
            f"'{COREWARDING_MATH_ORIGINAL}', '{COREWARDING_MATH_REPHRASED}'."
        )

    dataset = hf_load_dataset(dataset_name)
    full_train = dataset["train"].map(format_prompt, remove_columns=dataset["train"].column_names)

    split = full_train.train_test_split(test_size=_VALIDATION_SIZE, seed=_VALIDATION_SEED)
    train_dataset, eval_dataset = split["train"], split["test"]

    max_samples = os.environ.get("MAX_SAMPLES")
    if max_samples is not None:
        n = min(int(max_samples), len(train_dataset))
        train_dataset = train_dataset.select(range(n))

    math500_path = os.environ.get("MATH500_EVAL_PATH")
    if math500_path is not None:
        eval_dataset = _load_math500_eval(math500_path)

    return train_dataset, eval_dataset
