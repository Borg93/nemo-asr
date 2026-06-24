#!/usr/bin/env python
"""Build a NeMo manifest for the FLEURS Swedish test set -- the honest, in-the-wild
benchmark the Nemotron 3.5 ASR model card reports Swedish numbers on.

BASELINE TO BEAT (base model, FLEURS sv-SE, LangID mode):
    80ms = 25.61   160ms = 24.85   320ms = 23.63   560ms = 22.72   1.12s = 22.17  (WER %)

Decodes via soundfile (torchcodec lags torch 2.11). Usage: python prepare_fleurs_test.py
"""
import argparse, io, json, os
import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio

SR, LANG = 16000, "sv-SE"


def decode16k(d):
    src = io.BytesIO(d["bytes"]) if d.get("bytes") else d["path"]
    y, sr = sf.read(src, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(y)


ap = argparse.ArgumentParser()
ap.add_argument("--out", default="/workspace/sv_asr/data")
ap.add_argument("--config", default="sv_se", help="FLEURS config code for Swedish")
args = ap.parse_args()

wav_dir = os.path.join(args.out, "fleurs_test_wavs")
os.makedirs(wav_dir, exist_ok=True)

ds = load_dataset("google/fleurs", args.config, split="test").cast_column("audio", Audio(decode=False))

rows = []
for i, ex in enumerate(ds):
    # FLEURS `transcription` is normalized (lowercase, little punctuation). Fine for a
    # comparable WER number as long as hyp+ref get the same normalization at eval time.
    text = (ex.get("transcription") or ex.get("raw_transcription") or "").strip()
    if not text:
        continue
    arr = decode16k(ex["audio"])
    p = os.path.join(wav_dir, f"fleurs_{i:06d}.wav")
    sf.write(p, arr, SR, subtype="PCM_16")
    rows.append({"audio_filepath": os.path.abspath(p), "duration": round(len(arr) / SR, 3),
                 "text": text, "lang": LANG, "target_lang": LANG})

out = os.path.join(args.out, "fleurs_sv_test_manifest.json")
with open(out, "w") as f:
    for r in rows:
        f.write(json.dumps(r, ensure_ascii=False) + "\n")
print(f"FLEURS sv test: {len(rows)} clips, {sum(r['duration'] for r in rows)/3600:.2f} h -> {out}")
