"""Materialise the 150 MathVista eval images the repo references but does not ship.

data/mathvista/testmini_150.jsonl points at images/1.png..150.png; the images are
not committed, and nothing in setup/ creates them -- the first eval otherwise
dies on images/1.png. This rebuilds them from AI4Math/MathVista (testmini).

The filenames map to MathVista pids 1..150, but the mapping is never assumed:
every row is checked against the jsonl (question must prefix `problem`; the
answer must match, resolving multi-choice letters through the choice list)
before its image is written, and the run aborts if any row disagrees. A
mispaired image would still evaluate and still produce a plausible number --
silently meaningless.

Usage:  python tools/materialise_mathvista_images.py
Needs:  `datasets` (any recent version), network access to HF.
"""
import json, os, sys
from datasets import load_dataset

repo = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
mv = os.path.join(repo, "data", "mathvista")
imgdir = os.path.join(mv, "images")
rows = [json.loads(l) for l in open(os.path.join(mv, "testmini_150.jsonl"))]

want = {}
for r in rows:
    pid = int(os.path.splitext(os.path.basename(r["image"]))[0])
    want[pid] = r
print(f"  need {len(want)} images, pids {min(want)}..{max(want)}")

if all(os.path.exists(os.path.join(mv, r["image"])) for r in rows):
    print("  all present, nothing to do"); sys.exit(0)

os.makedirs(imgdir, exist_ok=True)
ds = load_dataset("AI4Math/MathVista", split="testmini")
pid_to_idx = {int(p): i for i, p in enumerate(ds["pid"])}

bad = []
for pid, r in sorted(want.items()):
    if pid not in pid_to_idx:
        bad.append(f"pid {pid} absent from testmini"); continue
    row = ds[pid_to_idx[pid]]

    # the jsonl problem is the question, plus a rendered choice list for
    # multi_choice rows, so prefix rather than equality
    q = (row["question"] or "").strip()
    if not r["problem"].strip().startswith(q[:60]):
        bad.append(f"pid {pid}: question mismatch\n    hf   : {q[:70]!r}\n    jsonl: {r['problem'][:70]!r}")
        continue

    # free_form keeps the literal answer; multi_choice stores the letter, so
    # resolve it through the choice list before comparing
    sol, ans = str(r["solution"]).strip(), str(row["answer"]).strip()
    if row["question_type"] == "multi_choice":
        ch = row["choices"] or []
        i = ord(sol.upper()) - ord("A")
        if not (0 <= i < len(ch)) or str(ch[i]).strip() != ans:
            bad.append(f"pid {pid}: choice {sol} -> {ch[i] if 0<=i<len(ch) else '??'!r}, hf answer {ans!r}")
            continue
    elif sol != ans:
        bad.append(f"pid {pid}: answer {sol!r} != hf {ans!r}")
        continue

    row["decoded_image"].convert("RGB").save(os.path.join(mv, r["image"]))

if bad:
    print(f"  {len(bad)} rows failed verification; the jsonl and MathVista do not line up:")
    for b in bad[:10]:
        print("   ", b)
    sys.exit(1)
print(f"  wrote {len(want)} verified images to {imgdir}")
