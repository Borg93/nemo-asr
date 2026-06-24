# Results & lessons

## Run 1 — 76 h pipeline validation (single Blackwell)
**Setup:** 40 h RixVox-v2 (parliamentary) + ~36 h TTS-Swedish (audiobook), `sv-SE`.
AdamW + CosineAnnealing **lr=1e-4**, `batch_duration=300`, bf16, **10 k steps (10 epochs)**.
~3.3 it/s on RTX PRO 6000 Blackwell (~50 min). `init_from_pretrained_model`.

**In-domain val (random RixVox+TTS split, training context `[70,6]`):** improved
`val_wer 0.3828 → 0.3075` (epochs 0→8, then plateaued) — ~20 % relative.

**Held-out FLEURS `sv_se` (streaming, the honest external benchmark):**

| latency | base raw | FT raw | base norm | FT norm |
|---|---|---|---|---|
| `[56,0]` 80 ms | 37.28 | 39.20 | **27.96** | **29.79** |
| `[56,13]` 1.12 s | 34.85 | 37.62 | — | — |

→ **General Swedish REGRESSED** ~1.8 pts normalized (~6.5 % rel). (Normalization = lowercase,
strip punctuation, keep å ä ö. Base ≈ card's 25.61.)

## Lesson: domain over-specialization
The model got better at *parliamentary/audiobook* Swedish but worse at *general* Swedish because:
1. **Domain shift** — RixVox (formal parliamentary) + TTS (audiobook) ≠ FLEURS (general read speech).
2. **Overfit** — 10 epochs over only 76 h; in-domain val had already plateaued.
3. **No replay** — Swedish-only, nothing anchoring the original distribution.

The 76 h run validated the whole pipeline (data→train→stream-eval→deploy) and proved the recipe
mechanics. But FLEURS is the wrong benchmark for *this* data — for parliamentary Swedish the in-domain
gain is the real signal.

## Run 2 plan — "robust everywhere" (chosen target)
Improve in-domain WITHOUT eroding general/other:
- **Data:** balanced mix via Lhotse `input_cfg` weights — RixVox (parliamentary) **+ general Swedish**
  (Common Voice sv, FLEURS-sv *train*) **+ 20–30 % multilingual replay** (other locales) to protect the 39.
- **Recipe:** moderate **lr ~5e-5**, fewer steps / **early-stop on the worse of the two test sets**.
- **Eval:** held-out **RixVox** test (speaker/date-disjoint) **and** FLEURS `sv_se` — track both;
  ship only if FLEURS does not regress and RixVox improves.
- Keep the language tag exact (`target_lang=sv-SE`) and cased+punctuated text.
