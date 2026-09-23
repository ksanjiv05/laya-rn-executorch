#!/usr/bin/env bash
# Export the iOS variants of the Laya .pte — RUN THIS ON A MAC.
#
# Produces:
#   laya_xnnpack_int8wo.pte  — int8, XNNPACK CPU backend. Cross-platform, runs on iOS as-is. (reliable)
#   laya_coreml.pte          — Core ML backend (Apple Neural Engine / GPU). Experimental — see notes.
#
# ExecuTorch's Core ML lowering needs Apple's coremltools native runtime, which ONLY exists on macOS,
# so this script cannot be run on Linux.
#
# Usage:
#   ./export_ios.sh              # builds the XNNPACK int8 model (safe iOS default)
#   ./export_ios.sh --coreml     # ALSO attempts the Core ML build (may need the gather fix, see below)
set -euo pipefail

HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"

if [[ "$(uname)" != "Darwin" ]]; then
  echo "ERROR: Core ML export requires macOS. Run this on a Mac."
  echo "(On Linux you can still build the XNNPACK model, which also runs on iOS: "
  echo " python export_laya_pte.py --weight-only --out laya_xnnpack_int8wo.pte )"
  exit 1
fi

# --- 1. Python env (uv recommended; falls back to venv) ---
if command -v uv >/dev/null 2>&1; then
  uv venv --python 3.11 .venv-ios 2>/dev/null || true
  source .venv-ios/bin/activate
  uv pip install "torch>=2.11" "executorch>=1.5" transformers safetensors numpy huggingface_hub "coremltools>=8"
else
  python3.11 -m venv .venv-ios
  source .venv-ios/bin/activate
  pip install "torch>=2.11" "executorch>=1.5" transformers safetensors numpy huggingface_hub "coremltools>=8"
fi

# --- 2. Get the source weights (typed-decisions checkpoint) if missing ---
if [ ! -f "../model-src/hf/typed-decisions/model.safetensors" ]; then
  echo "Downloading the typed-decisions checkpoint from Hugging Face..."
  python - <<'PY'
from huggingface_hub import hf_hub_download
for f in ["typed-decisions/model.safetensors","typed-decisions/encoder/config.json",
          "typed-decisions/rl_agent_config.json","typed-decisions/tokenizer/tokenizer.json",
          "typed-decisions/tokenizer/tokenizer_config.json"]:
    hf_hub_download("convaiinnovations/laya", f, local_dir="../model-src/hf")
PY
fi

# --- 3. XNNPACK int8 (the reliable iOS build — same .pte runs on Android too) ---
echo "==> Exporting XNNPACK int8 (.pte) for iOS..."
python export_laya_pte.py --backend xnnpack --weight-only --out laya_xnnpack_int8wo.pte
echo "    -> laya_xnnpack_int8wo.pte  (ship this on iOS; put it in the app's Documents dir)"

# --- 4. Core ML (optional, experimental) ---
if [[ "${1:-}" == "--coreml" ]]; then
  echo "==> Attempting Core ML (.pte) — Apple Neural Engine / GPU..."
  echo "    NOTE: Laya's decision head does a gather over the option markers; Core ML's"
  echo "    gather_along_axis wants int32 indices. If you hit"
  echo "      ValueError: Op \"aten_gather_default\" ... expects int32 but got ...fp32"
  echo "    apply the one-line fix in COREML_NOTES.md, then re-run."
  python export_laya_pte.py --backend coreml --out laya_coreml.pte \
    && echo "    -> laya_coreml.pte" \
    || { echo "    Core ML export failed — see COREML_NOTES.md for the gather fix."; }
fi

echo "Done. iOS-ready models are in $HERE."
