# Adapting to a NEW language (outside the 40 locales)

**You do NOT need this for Swedish** — `sv-SE` is already a supported locale (prompt slot 24), so we
just fine-tune with `target_lang=sv-SE` and reuse the base tokenizer (`update_tokenizer` stays false).

This doc is for bootstrapping a language the base checkpoint does **not** cover. Nemotron 3.5 ASR uses
a fixed **prompt dictionary** (128 slots) for language conditioning. ~60 unused slots exist (e.g. the
`speech_to_text_..._streaming_prompt.yaml` dict has `ms-MY: 35`, plus many empty IDs), so you can adapt
toward a new language by reusing/registering a slot.

## Two routes
1. **Reuse a free prompt-dict slot** (recommended): pick an unused/placeholder language ID and train
   the new language onto it (e.g. register `sm-WS` → your language).
2. **Redefine the prompt dictionary**: fine-tune with
   `NeMo/examples/asr/asr_transducer/speech_to_text_rnnt_bpe_prompt.py`, passing the full model config
   (num_layers, d_model, …) so it rebuilds the model with your `prompt_dictionary` and loads weights.

## Recipe (route 1, tokenizer extension)
```bash
# 1. Train a tokenizer for the new language (vocab_size must match the model; register the lang tag)
python $NEMO_DIR/scripts/tokenizers/process_asr_text_tokenizer.py \
  --manifest=train.json --data_root=xx_tokenizer \
  --vocab_size=128 --tokenizer=spe --spe_type=unigram \
  --spe_user_defined_symbols="<xx-XX>"

# 2. Extract the base tokenizer: untar the .nemo -> <prefix>_tokenizer.model, <prefix>_vocab.txt

# 3. Merge base (40-lang) + new (1-lang)  ->  40+1 tokenizer
python merge_tokenizers.py --out merged_tokenizer \
  --tokenizers base_tokenizer.model xx_tokenizer/tokenizer_spe_unigram_v128/tokenizer.model \
  --vocabs     base_vocab.txt       xx_tokenizer/tokenizer_spe_unigram_v128/vocab.txt

# 4. Fine-tune WITH the merged tokenizer (RNNT decoder is re-init'd from scratch)
python $NEMO_DIR/examples/asr/speech_to_text_finetune.py \
  --config-path="../asr/conf/fastconformer/cache_aware_streaming" \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_nemo_model=$HF_CKPT \
  model.train_ds.manifest_filepath=train.json \
  model.validation_ds.manifest_filepath=dev.json \
  model.train_ds.default_prompt_mode=langID \
  model.tokenizer.update_tokenizer=true \
  model.tokenizer.dir=merged_tokenizer \
  model.char_labels.update_labels=false \
  trainer.devices=1 trainer.max_steps=2000 trainer.precision=bf16
```

## Data notes (same as the Swedish recipe)
- Every clip carries `target_lang` / `lang` = the new locale tag.
- Append the language tag after each sentence's terminal punctuation, e.g. `"... test. <xx-XX>"`,
  and keep transcripts punctuated + properly cased (the style the model produces).
- `update_tokenizer=true` forces the RNNT **decoder to train from scratch** — expect to need much
  more data and many more steps than same-language fine-tuning (meaningful output ~40–60 epochs in the
  NVIDIA ms-MY walkthrough, WER still high). Evaluate before/after with `target_lang=<xx-XX>`.

Reference: NVIDIA "Adapt Tokenizer and Nemotron-3.5-asr-streaming Model to New Language" tutorial.
