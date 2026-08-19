#!/bin/bash
# Full SFT of Qwen3 (8B / 14B) on the <think>-wrapped OL-NL/DFS dataset.
#
# Adapted from train_qwen_prompt_guided_explanation.sh. Differences from that
# script, all deliberate -- see finetune_qwen3_*_think.sbatch for the matching
# Slurm side:
#
#   * --template qwen3, not deepseek. qwen3 registers as a ReasoningTemplate
#     with thought_words ("<think>", "</think>"). It injects an EMPTY
#     <think></think> pair only when the assistant turn lacks the tags; our
#     dataset already carries them, so nothing is double-wrapped and the
#     reasoning is trained on normally.
#   * --enable_thinking true, passed explicitly. It defaults to true, but
#     passing false would call remove_thought() and silently strip every
#     chain of thought out of the dataset -- the whole point of this run.
#   * --dataset_dir. The old scripts relied on LLaMA-Factory resolving
#     --dataset against a data/ dir relative to cwd. Pointing it at this repo's
#     own data/ is what makes dataset_info.json (and the dataset beside it)
#     actually resolve, from any cwd.
#   * --deepspeed points at LLaMA-Factory's shipped ds_z3_config.json. The old
#     scripts referenced ./ds_config_zero3.json, which no longer exists
#     anywhere on this filesystem.
#   * --learning_rate 1.0e-5, not 8.0e-5. 8e-5 is a LoRA-scale LR; for FULL
#     fine-tuning of an 8-14B model it is ~5-10x the usual range and risks
#     divergence or catastrophic forgetting. Override LR=8.0e-5 to restore the
#     old value.
#   * --warmup_ratio only. The old scripts passed --warmup_steps 100 AND
#     --warmup_ratio 0.1; HF Trainer honours warmup_steps and ignores the
#     ratio, so having both is misleading.

set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LLAMA_FACTORY="${LLAMA_FACTORY:-/scratch/wx2356/LLaMA-Factory}"

# 8B or 14B
QWEN3_SIZE="${QWEN3_SIZE:-8B}"
MODEL_PATH="${MODEL_PATH:-Qwen/Qwen3-${QWEN3_SIZE}}"

DATASET="${DATASET:-codev_sva_ol_dfs_5000_think}"
DATASET_DIR="${DATASET_DIR:-${REPO_DIR}/data}"
OUTPUT_PATH="${OUTPUT_PATH:-/scratch/wx2356/verilogFinetune/output/qwen3-${QWEN3_SIZE}-codev-sva-ol-dfs-think}"
DS_CONFIG_PATH="${DS_CONFIG_PATH:-${LLAMA_FACTORY}/examples/deepspeed/ds_z3_config.json}"

# One process per GPU. Defaults to however many the scheduler actually gave us.
NPROC_PER_NODE="${NPROC_PER_NODE:-$(nvidia-smi -L 2>/dev/null | wc -l)}"
NPROC_PER_NODE="${NPROC_PER_NODE:-4}"
NNODES="${NNODES:-1}"
NODE_RANK="${NODE_RANK:-0}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-12345}"

LR="${LR:-1.0e-5}"
EPOCHS="${EPOCHS:-3}"
CUTOFF_LEN="${CUTOFF_LEN:-16384}"
MICRO_BSZ="${MICRO_BSZ:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-2}"

# Checkpointing. --save_only_model is the important one: without it a full
# fine-tune writes the whole ZeRO-3 optimizer state (fp32 moments + master
# weights, ~13 bytes/param on top of the model) with every checkpoint. The
# first 8B run did exactly that and produced 10 x 107GB = 1.1TB of output for
# a 16GB model, against a 5TB scratch quota -- and 14B checkpoints would be
# ~195GB each. Optimizer state is only needed to RESUME training; evaluating
# or serving the model needs weights alone, which is what the final top-level
# save already contains. Set SAVE_ONLY_MODEL=false if you specifically need
# resumable checkpoints, and budget the disk for it.
SAVE_STEPS="${SAVE_STEPS:-500}"
SAVE_TOTAL_LIMIT="${SAVE_TOTAL_LIMIT:-2}"

for f in "$DS_CONFIG_PATH" "${DATASET_DIR}/dataset_info.json" "${LLAMA_FACTORY}/src/train.py"; do
    [ -f "$f" ] || { echo "ERROR: required file not found: $f" >&2; exit 1; }
done

echo "model=${MODEL_PATH} dataset=${DATASET} gpus=${NPROC_PER_NODE} lr=${LR} out=${OUTPUT_PATH}"

torchrun \
    --nproc_per_node "$NPROC_PER_NODE" \
    --nnodes "$NNODES" \
    --node_rank "$NODE_RANK" \
    --master_addr "$MASTER_ADDR" \
    --master_port "$MASTER_PORT" \
    "${LLAMA_FACTORY}/src/train.py" \
    --deepspeed "$DS_CONFIG_PATH" \
    --stage sft \
    --do_train \
    --use_fast_tokenizer \
    --model_name_or_path "$MODEL_PATH" \
    --dataset "$DATASET" \
    --dataset_dir "$DATASET_DIR" \
    --template qwen3 \
    --enable_thinking true \
    --finetuning_type full \
    --output_dir "$OUTPUT_PATH" \
    --overwrite_cache \
    --overwrite_output_dir \
    --warmup_ratio 0.1 \
    --weight_decay 0.1 \
    --per_device_train_batch_size "$MICRO_BSZ" \
    --gradient_accumulation_steps "$GRAD_ACCUM" \
    --ddp_timeout 180000000 \
    --learning_rate "$LR" \
    --lr_scheduler_type cosine \
    --logging_steps 1 \
    --cutoff_len "$CUTOFF_LEN" \
    --save_steps "$SAVE_STEPS" \
    --save_total_limit "$SAVE_TOTAL_LIMIT" \
    --save_only_model true \
    --plot_loss \
    --num_train_epochs "$EPOCHS" \
    --report_to none \
    --bf16
