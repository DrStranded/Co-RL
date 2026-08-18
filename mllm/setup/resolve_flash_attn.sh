#!/usr/bin/env bash
# =============================================================================
# resolve_flash_attn.sh
# -----------------------------------------------------------------------------
# flash-attn is NOT committed to this repo (the built .so is ~952MB). This script
# makes it appear, using the cheapest path that works on THIS machine, and prints
# which path it took so a downstream agent can log it.
#
# Target stack (frozen):
#     torch 2.9.0+cu128 / vllm 0.11.2 / transformers 4.57.0 / deepspeed 0.18.0
#   For that stack the official Dao-AILab prebuilt wheel EXISTS:
#     flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
#   ("must compile from source" is only true for torch 2.10, which has cu13-only
#    wheels, we do NOT target 2.10 here.)
#
# Resolution order (each falls through to the next on failure):
#   PATH A  import flash_attn already works at the wanted version   -> use it
#   PATH B  detect torch/cuda/py/abi -> pip install the matching
#           official v2.8.3 release wheel from GitHub                -> install
#   PATH C  no matching wheel (e.g. torch2.10+cu12) or B failed
#           -> compile from source (FLASH_ATTN_CUDA_ARCHS="80;90")   -> build
#
# Env overrides (all optional):
#   PYTHON                interpreter to use            (default: python)
#   FLASH_ATTN_VERSION    FA version to target          (default: 2.8.3)
#   FA_ACCEPT_ANY=1       accept any already-importable FA (skip version match)
#   FA_WHEEL_URL          force PATH B to use this exact wheel URL
#   FLASH_ATTN_CUDA_ARCHS source-build archs            (default: "80;90")
#   MAX_JOBS              parallel build jobs for PATH C (default: min(nproc,16))
#   FA_FORCE_SOURCE=1     skip A+B, go straight to source build
# =============================================================================
set -euo pipefail

PYTHON="${PYTHON:-python}"
FA_VERSION="${FLASH_ATTN_VERSION:-2.8.3}"
FA_ARCHS="${FLASH_ATTN_CUDA_ARCHS:-80;90}"

log() { printf '[resolve-fa] %s\n' "$*"; }
die() { printf '[resolve-fa][ERROR] %s\n' "$*" >&2; exit 1; }

command -v "$PYTHON" >/dev/null 2>&1 || die "python interpreter '$PYTHON' not found"

# --- probe torch (must be installed; FA links against it) --------------------
"$PYTHON" -c 'import torch' 2>/dev/null \
  || die "torch is not importable under '$PYTHON'. Install the stack first (requirements.txt)."

# One probe call -> all wheel-tag components, kept consistent.
#   TORCH_FULL  e.g. 2.9.0
#   TORCH_MM    e.g. 2.9   (major.minor -> matches FA wheel 'torch2.9' tag)
#   CUDA_MAJOR  e.g. 12    (FA wheel uses cuda MAJOR: cu12 / cu13)
#   PY_TAG      e.g. cp312
#   ABI         TRUE|FALSE (cxx11 abi of the torch build)
read -r TORCH_FULL TORCH_MM CUDA_MAJOR PY_TAG ABI < <("$PYTHON" - <<'PY'
import sys, torch
tv = torch.__version__.split('+')[0]
mm = '.'.join(tv.split('.')[:2])
cu = (torch.version.cuda or '0.0').split('.')[0]
py = f"cp{sys.version_info.major}{sys.version_info.minor}"
abi = 'TRUE' if getattr(torch._C, '_GLIBCXX_USE_CXX11_ABI', True) else 'FALSE'
print(tv, mm, cu, py, abi)
PY
)
log "detected: torch=${TORCH_FULL} (torch${TORCH_MM})  cuda_major=${CUDA_MAJOR}  py=${PY_TAG}  cxx11abi=${ABI}"
[ "${CUDA_MAJOR}" != "0" ] || die "torch reports no CUDA build (torch.version.cuda is None). Need a CUDA torch."

# ---------------------------------------------------------------------------
# PATH A: already importable at the wanted version?
# ---------------------------------------------------------------------------
if [ "${FA_FORCE_SOURCE:-0}" != "1" ]; then
  if HAVE_VER="$("$PYTHON" -c 'import flash_attn; print(flash_attn.__version__)' 2>/dev/null)"; then
    if [ "${FA_ACCEPT_ANY:-0}" = "1" ] || [ "${HAVE_VER}" = "${FA_VERSION}" ]; then
      log "PATH A: flash_attn ${HAVE_VER} already installed and importable -> using it."
      log "RESULT: flash-attn ready via PATH A (existing install, v${HAVE_VER})."
      exit 0
    fi
    log "flash_attn ${HAVE_VER} present but != target ${FA_VERSION}; will (re)install. (set FA_ACCEPT_ANY=1 to keep it)"
  else
    log "flash_attn not importable yet -> trying prebuilt wheel."
  fi
fi

# ---------------------------------------------------------------------------
# PATH B: pip install the matching official release wheel
# ---------------------------------------------------------------------------
if [ "${FA_FORCE_SOURCE:-0}" != "1" ]; then
  CU_TAG="cu${CUDA_MAJOR}"                       # cu12 for torch2.9/cu12.x, cu13 for torch2.10
  WHEEL="flash_attn-${FA_VERSION}+${CU_TAG}torch${TORCH_MM}cxx11abi${ABI}-${PY_TAG}-${PY_TAG}-linux_x86_64.whl"
  URL="${FA_WHEEL_URL:-https://github.com/Dao-AILab/flash-attention/releases/download/v${FA_VERSION}/${WHEEL}}"
  # For the frozen stack (torch2.9 / cu12 / cp312 / abiTRUE) this resolves to the
  # known-good asset: flash_attn-2.8.3+cu12torch2.9cxx11abiTRUE-cp312-cp312-linux_x86_64.whl
  log "PATH B: attempting prebuilt wheel:"
  log "        ${URL}"
  if "$PYTHON" -m pip install "${URL}"; then
    if "$PYTHON" -c 'import flash_attn; print("fa", flash_attn.__version__)'; then
      log "RESULT: flash-attn ready via PATH B (official wheel ${WHEEL})."
      exit 0
    fi
    log "wheel installed but flash_attn still not importable -> falling through to source build."
  else
    log "wheel not available / install failed (combo ${CU_TAG}torch${TORCH_MM}abi${ABI} may have no published asset) -> source build."
  fi
fi

# ---------------------------------------------------------------------------
# PATH C: compile from source (last resort)
#   - FA 2.8.3 reads FLASH_ATTN_CUDA_ARCHS (it IGNORES TORCH_CUDA_ARCH_LIST).
#   - --no-build-isolation so it compiles against the installed torch.
#   - 80 = A100 (sm_80), 90 = H100 (sm_90); the 8x80GB target is one of these.
#   - Build is memory-heavy (~2.5GB/job) and slow (30min-3h). Tune MAX_JOBS.
# ---------------------------------------------------------------------------
NPROC="$(nproc 2>/dev/null || echo 4)"
if [ "${NPROC}" -gt 16 ]; then DEF_JOBS=16; else DEF_JOBS="${NPROC}"; fi
MAX_JOBS="${MAX_JOBS:-${DEF_JOBS}}"

log "PATH C: compiling flash-attn ${FA_VERSION} from source."
log "        FLASH_ATTN_CUDA_ARCHS='${FA_ARCHS}'  MAX_JOBS=${MAX_JOBS}  (this can take 30min-3h)"
"$PYTHON" -m pip install -U pip setuptools wheel
"$PYTHON" -m pip install "ninja>=1.11" packaging
FLASH_ATTN_CUDA_ARCHS="${FA_ARCHS}" MAX_JOBS="${MAX_JOBS}" \
  "$PYTHON" -m pip install --no-build-isolation "flash-attn==${FA_VERSION}"

"$PYTHON" -c 'import flash_attn; print("fa", flash_attn.__version__)' \
  || die "source build finished but flash_attn is still not importable."
log "RESULT: flash-attn ready via PATH C (compiled from source, archs ${FA_ARCHS})."
