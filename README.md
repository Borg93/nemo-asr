# Fine-tuning Nemotron 3.5 ASR on Swedish (streaming RNNT)

Reproducible scripts to fine-tune NVIDIA **`nvidia/nemotron-3.5-asr-streaming-0.6b`**
(prompt-conditioned, cache-aware **streaming** FastConformer-RNNT) on Swedish (`sv-SE`),
evaluate it in streaming mode, and run inference — plus how to adapt the model to a brand-new language.

**Streaming is preserved.** We restore the full pretrained model and change only data + optimizer —
never the encoder — so the fine-tuned `.nemo` still streams at every latency (`[56,0]`=80 ms …
`[56,13]`=1.12 s) via the same cache-aware inference path.

## Setup (reproducible)
```bash
git clone https://github.com/Borg93/nemo-asr.git && cd nemo-asr
export NEMO_DIR=$PWD/NeMo DATA=/path/with/space/data HF_HOME=/path/with/space/hf VENV=.venv
bash setup.sh            # clones NeMo @ the pinned commit, installs torch cu128 + NeMo (uv.lock)
huggingface-cli login    # avoids HF rate limits
```
Pinned: **torch 2.11.0+cu128** (Blackwell sm_120), **NeMo @ `dcd715329a`** (`pyproject.toml` + `uv.lock`).
`torchcodec` is deliberately **not** installed (its native lib lags torch 2.11) — audio is decoded via
`soundfile`. Frozen full-env snapshot in `requirements-frozen.txt`.

## Data
| Dataset | Domain | Text style | Role |
|---|---|---|---|
| `KBLab/rixvox-v2` | Parliamentary (~23,000 h, 16 kHz) | punctuated + cased ✅ | primary (subsampled) — **ODC-BY** |
| `datadriven-company/TTS-Swedish` | LibriVox audiobooks (~40 h, 24 kHz) | punctuated + cased ✅ | supplement (CC0) |

`KTH/nst` is excluded for now (its `text` has dictated `\Punkt`/`\Komma` + silence-tag artifacts).
```bash
python prepare_data.py --out $DATA --rixvox-hours 400   # wav + manifests, target_lang=sv-SE, cased text
python prepare_fleurs_test.py --out $DATA               # FLEURS sv_se held-out benchmark
```

## Train
```bash
DATA=$DATA bash run_finetune.sh                         # 1 GPU
DEVICES=3 LR=2e-4 DATA=$DATA bash run_finetune.sh       # 3× Blackwell, single node (DDP)
```
DDP + `use_distributed_sampler=false` (Lhotse shards by rank). Global batch scales with `DEVICES`,
so scale `LR` (~√N…N); each step sees N× audio (same coverage in fewer steps). Env knobs:
`DEVICES, LR, BATCH_DURATION, MAX_STEPS, WARMUP, VENV, NEMO_DIR, DATA, EXP`.

## Evaluate (streaming, at deployment latency)
```bash
MODEL=$EXP/sv_ft/version_0/checkpoints/sv_ft.nemo ATT="[56,0]" bash run_eval.sh
```
Baseline to beat (base model, FLEURS sv-SE, LangID): `80ms=25.61 … 1.12s=22.17`. Always measure
the base model on the **same** manifest first (`MODEL=nvidia/nemotron-3.5-asr-streaming-0.6b`) for an
apples-to-apples delta. (The raw script WER is inflated vs the card unless you normalize hyp+ref the
same way — lowercase, strip punctuation, keep å ä ö.)

## Inference / deploy
```bash
MODEL=sv_ft.nemo AUDIO_DIR=wavs/ LANG=sv-SE ATT="[56,0]" bash run_infer.sh   # real-time streaming
```
`LANG=auto STRIP_LANG_TAGS=false` → auto language detection with `<xx-XX>` tags. Offline-batch,
multi-GPU transcription, CUDA-graph decoding, ONNX/TensorRT export, and NIM serving:
see **[`docs/inference_and_deploy.md`](docs/inference_and_deploy.md)**.

## Adapt to a NEW language (outside the 40 locales)
Not needed for Swedish. For an unsupported language you train + merge a tokenizer and fine-tune with
`update_tokenizer=true` onto a prompt-dict slot — `merge_tokenizers.py` +
**[`docs/new_language_adaptation.md`](docs/new_language_adaptation.md)**.

## Recipe facts (verified against the shipped config)
- **`sv-SE` = prompt slot 24**; `lang_field: target_lang` drives conditioning → manifests set
  `"target_lang": "sv-SE"` (and `"lang"`), with cased+punctuated `text`.
- The config's `optim.lr: 2.0` is a **NoamAnnealing multiplier** (effective ~6e-4), not a real LR;
  we use **CosineAnnealing, explicit `lr=1e-4`** (and delete `sched.d_model`).
- Config defaults `is_tarred=true`; for a plain JSON manifest pass **`is_tarred=false`**.
- Don't set `att_context_size` during training — one checkpoint serves all latencies.
- Slim checkpoints (`save_top_k=1`, no `last.ckpt`) — each `.ckpt` is ~7 GB (model+optimizer state).

## Validated environment & results
- **Stack:** torch 2.11.0+cu128, NeMo 3.1 (@`dcd715329a`), lhotse 1.33, RTX PRO 6000 Blackwell (96 GB).
- **Throughput:** ~3.3 it/s single-GPU at `batch_duration=300` (~50 GB VRAM, compute-bound at ~89% util).
- **Baseline** (FLEURS `sv_se`, streaming, raw WER): `[56,0]`=37.28, `[56,13]`=34.85.

## Storage gotcha
Audio as 16 kHz wav ≈ 115 MB/h (RixVox full ≈ 2.7 TB). Mind your volume **quota** — a
`Disk quota exceeded` (errno 122) crash means the quota, not the cluster, is full. For big runs use
tarred shards / read HF parquet directly via Lhotse (`type: parquet`), or a RAM disk on big-mem nodes.

## Files
| File | Purpose |
|---|---|
| `setup.sh` | clone NeMo @ pinned commit + install env (uv) |
| `prepare_data.py` / `prepare_fleurs_test.py` | build wav + manifests / FLEURS test |
| `run_finetune.sh` | fine-tune (1→N GPUs) |
| `run_eval.sh` / `run_infer.sh` | streaming WER eval / transcription |
| `merge_tokenizers.py` | tokenizer merge for new-language adaptation |
| `docs/` | inference & deploy; new-language adaptation; **results.md** (run 1 findings) |
| `pyproject.toml` / `uv.lock` / `requirements-frozen.txt` | pinned, reproducible deps |
