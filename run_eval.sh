#!/usr/bin/env bash
# Streaming EVALUATION (WER) of a Nemotron 3.5 ASR .nemo at a chosen latency.
# Needs a manifest WITH reference text. Run the hardest setting ([56,0]=80ms) first, then sweep.
set -euo pipefail
VENV=${VENV:-.venv}; [ -f "$VENV/bin/activate" ] && source "$VENV/bin/activate"   # dev box: VENV=/root/.venv
NEMO_DIR=${NEMO_DIR:-./NeMo}
export HF_HOME=${HF_HOME:-$HOME/.cache/huggingface}

DATA=${DATA:-/workspace/sv_asr/data}
MODEL=${MODEL:?set MODEL=/path/to/finetuned.nemo (or nvidia/nemotron-3.5-asr-streaming-0.6b for baseline)}
MANIFEST=${MANIFEST:-$DATA/fleurs_sv_test_manifest.json}   # honest held-out benchmark
LANG=${LANG:-sv-SE}                                         # or "auto"
ATT=${ATT:-"[56,0]"}                                        # 80ms; also try [56,3], [56,13]

python "$NEMO_DIR/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py" \
  model_path="$MODEL" \
  dataset_manifest="$MANIFEST" \
  target_lang="$LANG" \
  att_context_size="$ATT" \
  decoder_type=rnnt \
  pad_and_drop_preencoded=true \
  batch_size="${BATCH_SIZE:-16}" \
  cuda=0 \
  strip_lang_tags=true \
  output_path="$DATA/eval_out"

# Logs "WER% of streaming mode". For an apples-to-apples vs the model card's normalized numbers,
# normalize hyp+ref identically (lowercase, strip punctuation, keep å ä ö). Baseline to beat
# (FLEURS sv-SE, LangID): 80ms=25.61 ... 1.12s=22.17. Get YOUR baseline by setting
# MODEL=nvidia/nemotron-3.5-asr-streaming-0.6b on the same MANIFEST, then measure the delta.
