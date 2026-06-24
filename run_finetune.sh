#!/usr/bin/env bash
# Fine-tune Nemotron 3.5 ASR (prompt-conditioned, cache-aware STREAMING RNNT) on Swedish.
#
# Streaming is preserved: we restore the full pretrained model and only change data+optim,
# so the fine-tuned .nemo still streams at every latency ([56,0]=80ms ... [56,13]=1.12s).
#
# Scales 1 -> N GPUs via env vars (single node, DDP):
#   DEVICES=1                       # today (1 Blackwell)
#   DEVICES=3 LR=2e-4               # tomorrow (3 Blackwell, single node) -- scale LR ~linearly/sqrt
#
# Recipe grounded in the shipped config (fastconformer_transducer_bpe_streaming_prompt.yaml):
#   sv-SE = prompt slot 24; lang_field=target_lang; config lr=2.0 is a Noam MULTIPLIER, so we
#   use an explicit CosineAnnealing peak; config defaults is_tarred=true so we pass is_tarred=false.
set -euo pipefail
source /root/.venv/bin/activate

NEMO_DIR=${NEMO_DIR:-/root/NeMo}                 # existing clone (matches installed nemo)
DATA=${DATA:-/workspace/sv_asr/data}            # output of prepare_data.py (must be visible to all ranks)
EXP=${EXP:-/workspace/sv_asr/ckpt}              # checkpoints -> persistent volume
DEVICES=${DEVICES:-1}                            # number of GPUs on this node
LR=${LR:-1e-4}                                   # 1 GPU: 1e-4 ; 3 GPU: ~2e-4 (scale with global batch)
BATCH_DURATION=${BATCH_DURATION:-300}            # seconds of audio per batch PER GPU (Lhotse dynamic)
MAX_STEPS=${MAX_STEPS:-10000}                    # NB: with N GPUs each step sees N x data
WARMUP=${WARMUP:-1000}
export PYTHONPATH=$NEMO_DIR:${PYTHONPATH:-}
export HF_HOME=${HF_HOME:-/workspace/sv_asr/hfcache}

echo "Launching on $DEVICES GPU(s): lr=$LR batch_duration=${BATCH_DURATION}s/gpu max_steps=$MAX_STEPS"

python "$NEMO_DIR/examples/asr/speech_to_text_finetune.py" \
  --config-path="../asr/conf/fastconformer/cache_aware_streaming" \
  --config-name=fastconformer_transducer_bpe_streaming_prompt.yaml \
  +init_from_pretrained_model=nvidia/nemotron-3.5-asr-streaming-0.6b \
  model.train_ds.manifest_filepath="$DATA/train_manifest.json" \
  model.train_ds.is_tarred=false \
  model.train_ds.batch_duration="$BATCH_DURATION" \
  model.validation_ds.manifest_filepath="$DATA/val_manifest.json" \
  model.optim.lr="$LR" \
  model.optim.sched.name=CosineAnnealing \
  ~model.optim.sched.d_model \
  model.optim.sched.warmup_steps="$WARMUP" \
  model.optim.sched.min_lr=1e-6 \
  trainer.devices="$DEVICES" \
  trainer.num_nodes=1 \
  ++trainer.use_distributed_sampler=false \
  trainer.precision=bf16 \
  trainer.max_steps="$MAX_STEPS" \
  trainer.limit_train_batches=1000 \
  trainer.val_check_interval=1.0 \
  exp_manager.exp_dir="$EXP" \
  exp_manager.name=sv_ft \
  ++exp_manager.use_datetime_version=false \
  exp_manager.checkpoint_callback_params.save_top_k=1 \
  ++exp_manager.checkpoint_callback_params.save_last=false \
  ++exp_manager.checkpoint_callback_params.save_nemo_on_train_end=true

# Multi-GPU notes (single node, e.g. 3x Blackwell):
#  - The config strategy is DDP and use_distributed_sampler=false (required for Lhotse, which
#    shards by rank itself). Just set DEVICES=3.
#  - Effective global batch = DEVICES x batch_duration -> raise LR (~sqrt(N)..N). Start LR=2e-4 for 3 GPUs.
#  - Each step now consumes DEVICES x audio, so you reach the same data coverage in fewer steps;
#    keep MAX_STEPS for more coverage or lower it ~ /DEVICES for the same coverage.
#  - 96 GB/GPU: you can also raise BATCH_DURATION (e.g. 450) for fewer, larger steps.
#  - Data dir must be on shared storage visible to all ranks (single node: /workspace or /dev/shm).
#
# OOM? lower BATCH_DURATION. Smoke test: MAX_STEPS=500.
# Final streaming model -> $EXP/sv_ft/version_0/checkpoints/sv_ft.nemo
