#!/usr/bin/env bash
# Reproduce the validated environment: torch 2.11 cu128 (Blackwell sm_120) + NeMo @ pinned commit.
# Usage:  NEMO_DIR=./NeMo bash setup.sh
set -euo pipefail

NEMO_REV=dcd715329a1c2dfbe620641e5f0a46ea561ea6bd     # pinned; keep in sync with pyproject.toml
NEMO_DIR=${NEMO_DIR:-$PWD/NeMo}

echo ">> system audio libs (ffmpeg, libsndfile)"
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y --no-install-recommends ffmpeg libsndfile1 git || true
fi

echo ">> NeMo repo (examples/ scripts + streaming-prompt config YAML) @ ${NEMO_REV:0:10}"
[ -d "$NEMO_DIR/.git" ] || git clone https://github.com/NVIDIA/NeMo.git "$NEMO_DIR"
git -C "$NEMO_DIR" fetch origin "$NEMO_REV" 2>/dev/null || git -C "$NEMO_DIR" fetch origin
git -C "$NEMO_DIR" checkout "$NEMO_REV"

echo ">> python env: torch cu128 + core deps (from uv.lock)"
uv sync --frozen

echo ">> NeMo library, pinned (separate step: NeMo declares its own torch index, conflicts with cu128 in one lock)"
uv pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@${NEMO_REV}"

# NOTE: torchcodec is intentionally NOT installed — its native lib lags torch 2.11; datasets
# audio is decoded via soundfile in prepare_*.py (NeMo trains via torchaudio/lhotse, unaffected).

cat <<EOF

Done. Then:
  export NEMO_DIR=$NEMO_DIR
  export HF_HOME=\$PWD/hf VENV=.venv      # HF_HOME needs space/quota for the model cache
  huggingface-cli login                    # avoids HF rate limits
  python prepare_data.py --out \$DATA --rixvox-hours 400
  python prepare_fleurs_test.py --out \$DATA
  DATA=\$DATA bash run_finetune.sh                       # 1 GPU   |  3 GPU: DEVICES=3 LR=2e-4
  MODEL=\$DATA/../ckpt/sv_ft/version_0/checkpoints/sv_ft.nemo bash run_eval.sh
EOF
