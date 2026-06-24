#!/usr/bin/env python
"""Prepare Swedish ASR fine-tuning data for Nemotron 3.5 ASR (sv-SE).

Builds NeMo JSON-lines manifests from two CLEAN, punctuated + cased corpora:
  - KBLab/rixvox-v2               (Swedish parliamentary; primary, subsampled by hours)
  - datadriven-company/TTS-Swedish (Swedish LibriVox audiobooks; small supplement)

Both already match Nemotron's output style (cased + punctuation), so NO text
cleaning is needed -- we just use the `text` field as-is. (NST is dropped for now.)

Audio is decoded with soundfile (NOT torchcodec, which lags torch 2.11) and
resampled to 16 kHz mono, written as PCM16 wav. Every clip gets
lang/target_lang="sv-SE", which drives the model's prompt-based language
conditioning (the config reads `target_lang`). A small held-out val split is
carved for val_check; for the HONEST benchmark use prepare_fleurs_test.py.

Set HF_TOKEN in the environment for higher download rate limits.

Usage (smoke):     python prepare_data.py --rixvox-hours 20 --no-tts
Usage (first run): python prepare_data.py --rixvox-hours 40
Usage (full):      python prepare_data.py --rixvox-hours 0   # 0 = no cap (all ~23k h)
"""
import argparse, io, json, os, random
import numpy as np
import soundfile as sf
import librosa
from datasets import load_dataset, Audio

SR = 16000
LANG = "sv-SE"


def decode16k(d):
    """Decode an undecoded HF audio dict {bytes,path} to a 16 kHz mono float32 array."""
    src = io.BytesIO(d["bytes"]) if d.get("bytes") else d["path"]
    y, sr = sf.read(src, dtype="float32", always_2d=False)
    if y.ndim > 1:
        y = y.mean(axis=1)
    if sr != SR:
        y = librosa.resample(y, orig_sr=sr, target_sr=SR)
    return np.ascontiguousarray(y)


def dump(path, rows):
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")  # ensure_ascii=False keeps å ä ö


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="/workspace/sv_asr/data")
    ap.add_argument("--rixvox-hours", type=float, default=40.0,
                    help="approx hours to stream from RixVox-v2; 0 = no cap (full ~23k h)")
    ap.add_argument("--rixvox-min-langprob", type=float, default=0.8)
    ap.add_argument("--min-dur", type=float, default=1.0)
    ap.add_argument("--max-dur", type=float, default=30.0)
    ap.add_argument("--tts-dnsmos-min", type=float, default=2.0)
    ap.add_argument("--no-tts", action="store_true")
    ap.add_argument("--val-clips", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=1234)
    args = ap.parse_args()

    random.seed(args.seed)
    wav_dir = os.path.join(args.out, "wavs")
    os.makedirs(wav_dir, exist_ok=True)
    records = []

    # ---------- RixVox-v2 (streamed subset; never downloads all 23k h unless uncapped) ----------
    cap = args.rixvox_hours * 3600 if args.rixvox_hours > 0 else float("inf")
    print(f"[rixvox] streaming up to {args.rixvox_hours or 'ALL'} h ...", flush=True)
    rv = load_dataset("KBLab/rixvox-v2", split="train", streaming=True)
    rv = rv.cast_column("audio", Audio(decode=False))   # raw bytes -> we decode via soundfile
    sec, n = 0.0, 0
    for ex in rv:
        if sec >= cap:
            break
        text = (ex.get("text") or "").strip()           # cased + punctuated -> use as-is
        if not text or ex.get("is_silence"):
            continue
        lp = ex.get("lang_prob_sv")
        if lp is not None and lp < args.rixvox_min_langprob:
            continue
        try:
            arr = decode16k(ex["audio"])
        except Exception as e:
            continue
        dur = len(arr) / SR
        if dur < args.min_dur or dur > args.max_dur:
            continue
        p = os.path.join(wav_dir, f"rixvox_{n:08d}.wav")
        sf.write(p, arr, SR, subtype="PCM_16")
        records.append({"audio_filepath": os.path.abspath(p), "duration": round(dur, 3),
                        "text": text, "lang": LANG, "target_lang": LANG, "source": "rixvox-v2"})
        sec += dur; n += 1
        if n % 1000 == 0:
            print(f"  rixvox {n} clips, {sec/3600:.1f} h", flush=True)
    print(f"[rixvox] kept {n} clips, {sec/3600:.2f} h", flush=True)

    # ---------- TTS-Swedish (full; audio field is `mp3`, 24k -> 16k) ----------
    if not args.no_tts:
        print("[tts] loading datadriven-company/TTS-Swedish ...", flush=True)
        tts = load_dataset("datadriven-company/TTS-Swedish", split="train", streaming=True)
        tts = tts.cast_column("mp3", Audio(decode=False))
        m, tsec = 0, 0.0
        for ex in tts:
            text = (ex.get("text") or "").strip()
            if not text:
                continue
            dn = ex.get("dnsmos")
            if dn is not None and dn < args.tts_dnsmos_min:
                continue
            try:
                arr = decode16k(ex["mp3"])
            except Exception:
                continue
            dur = len(arr) / SR
            if dur < args.min_dur or dur > args.max_dur:
                continue
            p = os.path.join(wav_dir, f"tts_{m:08d}.wav")
            sf.write(p, arr, SR, subtype="PCM_16")
            records.append({"audio_filepath": os.path.abspath(p), "duration": round(dur, 3),
                            "text": text, "lang": LANG, "target_lang": LANG, "source": "tts-swedish"})
            m += 1; tsec += dur
        print(f"[tts] kept {m} clips, {tsec/3600:.2f} h", flush=True)

    # ---------- shuffle, split, write ----------
    random.shuffle(records)
    val, train = records[:args.val_clips], records[args.val_clips:]
    dump(os.path.join(args.out, "train_manifest.json"), train)
    dump(os.path.join(args.out, "val_manifest.json"), val)
    th = sum(r["duration"] for r in train) / 3600
    vh = sum(r["duration"] for r in val) / 3600
    print(f"\nDONE  train={len(train)} ({th:.1f} h)  val={len(val)} ({vh:.1f} h)")
    print(f"  -> {args.out}/train_manifest.json")
    print(f"  -> {args.out}/val_manifest.json")


if __name__ == "__main__":
    main()
