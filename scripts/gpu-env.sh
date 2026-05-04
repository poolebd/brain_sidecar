#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${ROOT_DIR}/.venv/bin/python"

if [[ -x "${PYTHON_BIN}" ]]; then
  CUDA_LIB_PATH="$("${PYTHON_BIN}" - <<'PY'
import os
paths = []
try:
    import nvidia.cublas.lib
    paths.extend(str(path) for path in getattr(nvidia.cublas.lib, "__path__", []))
except Exception:
    pass
try:
    import nvidia.cuda_nvrtc.lib
    paths.extend(str(path) for path in getattr(nvidia.cuda_nvrtc.lib, "__path__", []))
except Exception:
    pass
try:
    import nvidia.cudnn.lib
    paths.extend(str(path) for path in getattr(nvidia.cudnn.lib, "__path__", []))
except Exception:
    pass
deduped = []
for path in paths:
    if path and path not in deduped:
        deduped.append(path)
print(":".join(deduped))
PY
)"
  if [[ -n "${CUDA_LIB_PATH}" ]]; then
    export LD_LIBRARY_PATH="${CUDA_LIB_PATH}:${LD_LIBRARY_PATH:-}"
  fi
fi
