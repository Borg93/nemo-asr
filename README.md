# Fine-tuning Nemotron 3.5 ASR on Swedish (streaming RNNT)

Scripts to fine-tune NVIDIA **`nvidia/nemotron-3.5-asr-streaming-0.6b`** (prompt-conditioned,
cache-aware **streaming** FastConformer-RNNT) on Swedish (`sv-SE`), using clean punctuated+cased
corpora, and evaluate it in streaming mode against FLEURS.

**Streaming is preserved.** We restore the full pretrained model and change only data + optimizer —
never the encoder — so the fine-tuned `.nemo` still streams at every latency operating point
(`[56,0]`=80 ms … `[56,13]`=1.12 s) via the same cache-aware inference script.

## Data
| Dataset | Domain | Text style | Role |
|---|---|---|---|
| `KBLab/rixvox-v2` | Parliamentary (~23,000 h, 16 kHz) | punctuated + cased ✅ | primary (subsampled) — **ODC-BY** |
| `datadriven-company/TTS-Swedish` | LibriVox audiobooks (~40 h, 24 kHz) | punctuated + cased ✅ | supplement (CC0) |

`KTH/nst` is intentionally excluded for now: its `text` carries dictated-punctuation artifacts
(`\Punkt`, `\Komma`, silence tags) that would need a regex clean first.

## Quickstart
```bash
# deps (use uv; newest libs). torchcodec is NOT installed — it lags torch 2.11, so we
# decode audio with soundfile instead (NeMo trains via torchaudio/lhotse, unaffected).
uv pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
uv pip install soundfile librosa "datasets>=2.18" Cython packaging
uv pip install "nemo_toolkit[asr] @ git+https://github.com/NVIDIA/NeMo.git@main"
export NEMO_DIR=/path/to/NeMo HF_HOME=/path/with/space/hf
huggingface-cli login   # avoids HF rate limits

python prepare_data.py --out $DATA --rixvox-hours 400      # 1. build wav + manifests (sv-SE tags)
python prepare_fleurs_test.py --out $DATA                  #    honest held-out benchmark
DATA=$DATA bash run_finetune.sh                            # 2. train (1 GPU)
MODEL=$DATA/../ckpt/sv_ft/version_0/checkpoints/sv_ft.nemo bash run_eval.sh   # 3. streaming eval @ 80ms
```

## Multi-GPU (single node, e.g. 3× Blackwell)
```bash
DEVICES=3 LR=2e-4 DATA=$DATA bash run_finetune.sh
```
DDP is already configured; `use_distributed_sampler=false` lets Lhotse shard by rank. Effective
global batch scales with `DEVICES`, so scale `LR` (~√N…N) and note each step now sees N× audio
(reach the same coverage in fewer steps). Data dir must be on storage visible to all ranks.

## Recipe facts (verified against the shipped config)
- **`sv-SE` = prompt slot 24**; `lang_field: target_lang` drives conditioning → every manifest line
  gets `"target_lang": "sv-SE"` (we also set `"lang"`). Use the cased+punctuated `text`.
- The config's `optim.lr: 2.0` is a **NoamAnnealing multiplier** (effective peak ~6e-4), NOT a real LR.
  For fine-tuning we switch to **CosineAnnealing, explicit peak `lr=1e-4`** (delete `sched.d_model`).
- Config defaults `is_tarred=true`; for a plain JSON manifest pass **`is_tarred=false`**.
- Don't set `att_context_size` during training — one checkpoint serves all latencies; pick it at eval.
- Slim checkpoints (`save_top_k=1`, no `last.ckpt`) — each `.ckpt` is ~7 GB (model+optimizer state).

## Validated environment & results
- **Stack:** torch 2.11.0+cu128, NeMo 3.1, lhotse 1.33, on an RTX PRO 6000 Blackwell (96 GB, sm_120).
- **Throughput:** ~3.3 it/s single-GPU at `batch_duration=300` (~49 GB VRAM). Compute-bound at 89% util —
  bigger batches don't speed it up, they only smooth gradients.
- **Baseline** (base model, FLEURS `sv_se`, streaming, *raw* WER): `[56,0]`=37.28%, `[56,13]`=34.85%.
  Raw is inflated vs the card's normalized 25.61/22.17 (model emits cased+punctuated, FLEURS refs are
  lowercased) — what matters is the **before/after delta measured the same way**.

## Storage gotcha
Audio as 16 kHz wav is large (~115 MB/h). RixVox full ≈ 2.7 TB. Mind your volume **quota** (a
`Disk quota exceeded` / errno 122 crash means the quota, not the cluster, is full). For big runs use
tarred shards or read HF parquet directly via Lhotse (`type: parquet`), or a RAM disk on big-memory nodes.

## Files
- `prepare_data.py` — stream RixVox subset + TTS-Swedish → 16 kHz wav + NeMo manifests (`target_lang=sv-SE`).
- `prepare_fleurs_test.py` — FLEURS `sv_se` test manifest (honest benchmark).
- `run_finetune.sh` — fine-tune launcher (1→N GPUs via env vars).
- `run_eval.sh` — streaming WER eval at a chosen `att_context_size`.
