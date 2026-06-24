#!/usr/bin/env bash
# Streaming evaluation of the fine-tuned model at deployment latency.
# Reports WER. Run the HARDEST setting ([56,0] = 80ms) first, then sweep.
set -euo pipefail
source /root/.venv/bin/activate
export HF_HOME=${HF_HOME:-/workspace/sv_asr/hfcache}

NEMO_DIR=${NEMO_DIR:-/root/NeMo}
DATA=${DATA:-/workspace/sv_asr/data}
MODEL=${MODEL:?set MODEL=/path/to/finetuned.nemo (or nvidia/nemotron-3.5-asr-streaming-0.6b for baseline)}
MANIFEST=${MANIFEST:-$DATA/fleurs_sv_test_manifest.json}   # honest benchmark
ATT=${ATT:-"[56,0]"}                                       # 80ms; also try [56,3], [56,13]

python "$NEMO_DIR/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py" \
  model_path="$MODEL" \
  dataset_manifest="$MANIFEST" \
  target_lang=sv-SE \
  att_context_size="$ATT" \
  decoder_type=rnnt \
  pad_and_drop_preencoded=true \
  batch_size=8 \
  cuda=0 \
  strip_lang_tags=true \
  output_path="$DATA/eval_out"

# Compare against base-model FLEURS sv-SE WER (LangID): 80ms=25.61 ... 1.12s=22.17.
# Baseline sanity check: run the SAME command with
#   MODEL=nvidia/nemotron-3.5-asr-streaming-0.6b  (set model_path to that name)
# to get the pre-fine-tune number on YOUR manifest, then measure the delta.
