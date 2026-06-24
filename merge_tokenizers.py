#!/usr/bin/env python
"""Merge SentencePiece tokenizers (+ vocabs) — for adapting Nemotron 3.5 ASR to a NEW language
that is NOT one of the 40 supported locales. (Swedish does NOT need this — sv-SE is slot 24.)

Workflow (see docs/new_language_adaptation.md):
  1. Train a tokenizer for the new language with NeMo's process_asr_text_tokenizer.py
     (--spe_user_defined_symbols="<xx-XX>" to register the language tag).
  2. Extract the base model's tokenizer: untar nemotron-3.5-asr-streaming-0.6b.nemo ->
     <prefix>_tokenizer.model and <prefix>_vocab.txt.
  3. Merge base + new with this script -> a 40+1 tokenizer dir.
  4. Fine-tune with model.tokenizer.update_tokenizer=true model.tokenizer.dir=<merged dir>
     (this re-inits the RNNT decoder; needs more data/steps than same-language fine-tuning).

Usage:
  python merge_tokenizers.py --out merged_tokenizer \
      --tokenizers base_tokenizer.model new_tokenizer.model \
      --vocabs     base_vocab.txt       new_vocab.txt
"""
import argparse, os
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2_model


def merge_spm(tokenizer_paths, out_model):
    base = spm.SentencePieceProcessor(); base.load(tokenizer_paths[0])
    proto = sp_pb2_model.ModelProto(); proto.ParseFromString(base.serialized_model_proto())
    tokens = {p.piece for p in proto.pieces if p.piece.strip()}
    scores = {p.piece: p.score for p in proto.pieces if p.piece.strip()}
    for path in tokenizer_paths[1:]:
        sp = spm.SentencePieceProcessor(); sp.load(path)
        other = sp_pb2_model.ModelProto(); other.ParseFromString(sp.serialized_model_proto())
        for p in other.pieces:
            if not p.piece.strip():
                continue
            if p.piece not in tokens:
                np_ = sp_pb2_model.ModelProto().SentencePiece()
                np_.piece, np_.score = p.piece, p.score
                proto.pieces.append(np_); tokens.add(p.piece); scores[p.piece] = p.score
            elif p.score > scores[p.piece]:                  # duplicate -> keep higher score
                scores[p.piece] = p.score
                for piece in proto.pieces:
                    if piece.piece == p.piece:
                        piece.score = p.score; break
    with open(out_model, "wb") as f:
        f.write(proto.SerializeToString())
    print(f"merged tokenizer -> {out_model}  ({len(proto.pieces)} pieces)")


def merge_vocabs(vocab_paths, out_vocab):
    combined = set()
    for p in vocab_paths:
        with open(p) as f:
            combined |= {ln.strip() for ln in f if ln.strip()}
    with open(out_vocab, "w") as f:
        f.write("\n".join(sorted(combined)))
    print(f"merged vocab -> {out_vocab}  ({len(combined)} entries)")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output tokenizer dir")
    ap.add_argument("--tokenizers", nargs="+", required=True, help="base.model new.model ...")
    ap.add_argument("--vocabs", nargs="+", required=True, help="base_vocab.txt new_vocab.txt ...")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    merge_spm(a.tokenizers, os.path.join(a.out, "tokenizer.model"))
    merge_vocabs(a.vocabs, os.path.join(a.out, "vocab.txt"))
