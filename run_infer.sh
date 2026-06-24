#!/usr/bin/env bash
# Streaming TRANSCRIPTION (inference) with a Nemotron 3.5 ASR .nemo (base or fine-tuned).
# No reference text needed. Uses the cache-aware streaming script — the low-latency, no-recompute
# path. Pick latency via ATT: [56,0]=80ms (real-time voice agents) ... [56,13]=1.12s (max accuracy).
#
# Provide exactly ONE input:  AUDIO=clip.wav   |   AUDIO_DIR=dir_of_wavs   |   MANIFEST=clips.jsonl
# Audio must be mono 16 kHz wav.
set -euo pipefail
VENV=${VENV:-.venv}; [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"   # dev box: VENV=/root/.venv
NEMO_DIR=${NEMO_DIR:-./NeMo}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

MODEL=${MODEL:-nvidia/nemotron-3.5-asr-streaming-0.6b}   # or your fine-tuned .nemo
LANG=${LANG:-sv-SE}                                       # or "auto" for language detection
ATT=${ATT:-"[56,0]"}
OUT=${OUT:-./infer_out}

ARGS=()
[ -n "${AUDIO:-}" ]     && ARGS+=(audio_file="$AUDIO")
[ -n "${AUDIO_DIR:-}" ] && ARGS+=(audio_dir="$AUDIO_DIR")
[ -n "${MANIFEST:-}" ]  && ARGS+=(dataset_manifest="$MANIFEST")
[ ${#ARGS[@]} -eq 1 ] || { echo "ERROR: set exactly one of AUDIO / AUDIO_DIR / MANIFEST"; exit 1; }

python "$NEMO_DIR/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py" \
  model_path="$MODEL" \
  "${ARGS[@]}" \
  target_lang="$LANG" \
  att_context_size="$ATT" \
  decoder_type=rnnt \
  pad_and_drop_preencoded=true \
  batch_size="${BATCH_SIZE:-8}" \
  cuda=0 \
  strip_lang_tags="${STRIP_LANG_TAGS:-true}" \
  output_path="$OUT"
# strip_lang_tags=false keeps the detected <xx-XX> tag in the output (useful with LANG=auto).
# See docs/inference_and_deploy.md for offline-batch, multi-GPU, mic-demo, and export options.
