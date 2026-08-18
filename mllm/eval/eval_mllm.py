#!/usr/bin/env python3
"""Run one MLLM checkpoint on one benchmark (jsonl+images) → accuracy json.

Inference: vLLM multimodal (Qwen2.5-VL / InternVL3.5 / Gemma3), greedy.
Grading: grade.py (mathruler rule-based + <answer>/boxed extraction + MCQ
letter/value mapping). No LLM judge.

Prompt style:
  --prompt answer : R1-V format our models were trained on (<think>/<answer>), no system role
  --prompt boxed  : "put final answer within \\boxed{}", helpful-assistant system

Usage:
  python eval_mllm.py --model <ckpt> --data <bench>/data.jsonl --out out.json \
      [--image_dir <dir>] [--prompt answer|boxed] [--limit N] [--temperature 0.0]
"""
import os, sys, json, argparse, time

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)  # for `trainers.*`
from grade import extract_pred, grade  # noqa: E402

# EXACT training suffix (trainers/dataset.py), spaces inside tags matter (model was trained on this).
ANSWER_SUFFIX = " Output the thinking process in <think> </think> and final answer in <answer> </answer> tags."
BOXED_SUFFIX = " Please reason step by step, and put your final answer within \\boxed{}."


def cap_image(img, max_side=1024):
    from PIL import Image
    img = img.convert("RGB")
    w, h = img.size
    if max(w, h) > max_side:
        s = max_side / max(w, h)
        img = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BICUBIC)
    return img


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True, help="benchmark jsonl ({problem,image,solution})")
    ap.add_argument("--image_dir", default=None, help="base dir for relative image paths (default: data's dir)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--prompt", choices=["answer", "boxed"], default="answer")
    ap.add_argument("--limit", type=int, default=0, help="first N examples (0=all)")
    ap.add_argument("--temperature", type=float, default=0.0)  # greedy: highest acc + reproducible (swept)
    ap.add_argument("--max_tokens", type=int, default=16384)
    ap.add_argument("--gpu_mem", type=float, default=0.92)
    ap.add_argument("--max_model_len", type=int, default=24576)  # keep 16k of max_tokens a REAL budget
    ap.add_argument("--tp", type=int, default=1, help="tensor parallel size")
    args = ap.parse_args()

    from PIL import Image
    from vllm import LLM, SamplingParams
    from transformers import AutoProcessor

    image_dir = args.image_dir or os.path.dirname(os.path.abspath(args.data))
    rows = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        rows = rows[: args.limit]
    print(f"[eval] model={args.model}  data={args.data}  n={len(rows)}  prompt={args.prompt}", flush=True)

    # Detect model family from config.json (NOT the path, colearn dirs contain both
    # model names, e.g. "qwen25vl3b_x_internvl35_2b", which fools path-based checks).
    from transformers import AutoConfig
    cfg = AutoConfig.from_pretrained(args.model, trust_remote_code=True).to_dict()
    fam = (" ".join(cfg.get("architectures", []) or []) + " " + str(cfg.get("model_type", ""))).lower()
    is_intern = "internvl" in fam
    print(f"[eval] model family: {'InternVL' if is_intern else ('Gemma' if 'gemma' in fam else 'Qwen-VL')}", flush=True)

    if is_intern:
        # disable patch-tiling so image-token count matches (same fix as training).
        # InternVL3.5-2B-HF is transformers-native → AutoProcessor loads directly from
        # the saved ckpt (it has preprocessor/processor_config/chat_template); no need
        # for the legacy custom-modeling patch (which the save_only_model ckpt lacks).
        try:
            from transformers.models.internvl.processing_internvl import InternVLProcessorKwargs
            InternVLProcessorKwargs._defaults["images_kwargs"]["crop_to_patches"] = False
        except Exception as e:
            print(f"[warn] InternVL crop_to_patches patch failed: {e}", flush=True)
    processor = AutoProcessor.from_pretrained(args.model, trust_remote_code=True, padding_side="left")
    suffix = ANSWER_SUFFIX if args.prompt == "answer" else BOXED_SUFFIX
    use_system = args.prompt == "boxed"

    def _build(eager):
        # mm_encoder_attn_backend: vLLM's bundled FA2 kernel rejects a ViT head_dim that
        # is not a multiple of 32 (Qwen2.5-VL ViT = 80). Pinning SDPA for the vision
        # encoder keeps that path off FA2; head_dim%32==0 ViTs (InternVL=128) are
        # unaffected because they never needed the upgrade in the first place.
        return LLM(model=args.model, dtype="bfloat16", trust_remote_code=True,
                   gpu_memory_utilization=args.gpu_mem, max_model_len=args.max_model_len,
                   limit_mm_per_prompt={"image": 1}, tensor_parallel_size=args.tp,
                   mm_encoder_attn_backend="TORCH_SDPA",
                   enforce_eager=eager)
    try:
        llm = _build(False)   # CUDA graphs on: ~1.4x decode throughput
    except Exception as e:
        print(f"[eval] cudagraph engine failed ({type(e).__name__}), falling back to eager", flush=True)
        llm = _build(True)
    sp = SamplingParams(temperature=args.temperature, top_p=0.95,
                        max_tokens=args.max_tokens, seed=0)

    # Build vLLM inputs (prompt text via chat template + PIL via multi_modal_data).
    inputs, golds, questions = [], [], []
    for r in rows:
        img = cap_image(Image.open(os.path.join(image_dir, r["image"])))
        content = [{"type": "image"}, {"type": "text", "text": r["problem"] + suffix}]
        msgs = ([{"role": "system", "content": "You are a helpful assistant."}] if use_system else []) + \
               [{"role": "user", "content": content}]
        prompt = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        inputs.append({"prompt": prompt, "multi_modal_data": {"image": img}})
        golds.append(str(r["solution"]))
        questions.append(str(r["problem"]))

    t0 = time.time()
    outs = llm.generate(inputs, sp)
    dt = time.time() - t0

    n_correct = n_extracted = 0
    samples = []
    for o, gold, q in zip(outs, golds, questions):
        resp = o.outputs[0].text
        pred = extract_pred(resp)
        ok = grade(pred, gold, q)
        n_extracted += pred is not None
        n_correct += ok
        samples.append({"gold": gold, "pred": pred, "ok": ok, "resp": resp})

    n = len(rows)
    res = {"model": args.model, "data": args.data, "prompt": args.prompt,
           "n": n, "n_correct": n_correct, "accuracy": round(n_correct / n, 4) if n else 0.0,
           "n_extracted": n_extracted, "extract_rate": round(n_extracted / n, 4) if n else 0.0,
           "gen_seconds": round(dt, 1), "samples": samples}
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    json.dump(res, open(args.out, "w"), indent=2, ensure_ascii=False)
    print(f"[eval] accuracy={res['accuracy']}  extract_rate={res['extract_rate']}  "
          f"({n_correct}/{n})  {dt:.0f}s  -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
