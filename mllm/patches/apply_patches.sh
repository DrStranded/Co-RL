#!/usr/bin/env bash
# Apply the three environment patches into the active venv. Idempotent.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SP="$(python -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
echo "site-packages: $SP"
for pkg in vllm trl; do
    [ -d "$SP/$pkg" ] || { echo "ERROR: $pkg is not installed in this environment. Run 'pip install -r requirements.txt' first." >&2; exit 1; }
done
patch --batch --forward -p1 -d "$SP" < "$HERE/vllm-0.11.2-vit-headdim.patch" || true
patch --batch --forward -p1 -d "$SP" < "$HERE/trl-mm-encoder-attn-backend.patch" || true
patch --batch --forward -p1 -d "$SP" < "$HERE/trl-nan-logprob-tolerance.patch" || true
python - <<'PY'
from pathlib import Path
import sysconfig
sp = Path(sysconfig.get_paths()["purelib"])
checks = [
    (sp/"vllm/model_executor/models/vision.py", "head_size % 32 != 0"),
    (sp/"trl/generation/vllm_generation.py",    "VLLM_MM_ENCODER_ATTN_BACKEND"),
    (sp/"trl/trainer/grpo_trainer.py",          "_fill_nan_logps"),
]
bad = [str(f) for f, needle in checks if needle not in f.read_text(encoding="utf-8")]
if bad:
    raise SystemExit(f"patch verification FAILED for: {bad}")
print("all three patches verified")
PY
