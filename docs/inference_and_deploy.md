# Inference & deployment

The fine-tuned `.nemo` is the **same architecture** as the base, so every inference path below works
on both. Pick the latency/accuracy point at inference time via `att_context_size` — no retraining.

| `att_context_size` | Chunk / latency | Use case |
|---|---|---|
| `[56,0]` | 80 ms (ultra-low) | real-time voice agents |
| `[56,1]` | 160 ms | interactive voice agents |
| `[56,3]` | 320 ms | conversational AI, live captions |
| `[56,6]` | 560 ms | higher accuracy |
| `[56,13]` | 1.12 s | highest accuracy |

## 1. Real-time streaming (the optimized, cache-aware path)
This is the no-recompute streaming path — every frame is processed once.
```bash
MODEL=path/to/sv_ft.nemo AUDIO_DIR=wavs/ LANG=sv-SE ATT="[56,0]" bash run_infer.sh
```
- Script: `NeMo/examples/asr/asr_cache_aware_streaming/speech_to_text_cache_aware_streaming_infer.py`
- Live mic demo: `NeMo/tutorials/asr/Online_ASR_Microphone_Demo_Cache_Aware_Streaming.ipynb`
- `LANG=auto` + `STRIP_LANG_TAGS=false` → model detects the language and appends `<xx-XX>`.

## 2. Offline / batch transcription (max throughput)
For files you don't need *streamed* (e.g. archives), offline batch is fastest:
```bash
python $NEMO_DIR/examples/asr/transcribe_speech.py \
  model_path=sv_ft.nemo dataset_manifest=clips.jsonl \
  batch_size=32 compute_langs=false output_filename=hyp.json
```
Multi-GPU parallel transcription: `NeMo/examples/asr/transcribe_speech_parallel.py`.

## 3. WER evaluation
`run_eval.sh` (wraps the streaming script with a reference manifest) or
`NeMo/examples/asr/speech_to_text_eval.py`.

## 4. Faster decoding (optimizations)
- **CUDA-graph RNNT decoding**: greedy RNNT supports CUDA graphs for lower per-step overhead
  (the log shows `CUDA graphs disabled ...` by default). Enable via the decoding config
  (`model.decoding.greedy.use_cuda_graph_decoder=true`) when supported by your build.
- **bf16** inference; keep audio mono 16 kHz to skip resampling.
- **Export**: `NeMo/examples/asr/export/` exports encoder/decoder to ONNX/TensorRT for serving.

## 5. Production serving
NVIDIA **Riva / Speech NIM** provides gRPC streaming for Nemotron ASR (Ampere→Blackwell, Jetson).
Watch the model card / NIM docs for the release. Until then, the cache-aware streaming script above
is the reference low-latency server-side path.
