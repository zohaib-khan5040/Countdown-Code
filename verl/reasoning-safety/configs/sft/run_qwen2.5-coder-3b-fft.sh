#!/bin/bash
set -x

# Should be in `verl/reasoning_safety`
PROJECT_DIR="$(pwd)"

PROJECT_NAME="verl_reasoning_safety"
TRAIN_PATH="${PROJECT_DIR}/data/sft/train.parquet"
TEST_PATH="${PROJECT_DIR}/data/sft/test.parquet"

MODEL_PATH="Qwen/Qwen2.5-Coder-3B-Instruct"
EXPERIMENT_NAME="sft_qwen25_coder_7b_orig"

if [ "$#" -lt 2 ]; then
    echo "Usage: run_qwen2.5-coder-3b-lora.sh <nproc_per_node> <save_path> [other_configs...]"
    exit 1
fi

nproc_per_node=$1
save_path=$2

# Shift the arguments so $@ refers to the rest
shift 2

torchrun --standalone --nnodes=1 --nproc_per_node=$nproc_per_node \
     -m verl.trainer.fsdp_sft_trainer \
    data.train_files=$TRAIN_PATH \
    data.val_files=$TEST_PATH \
    data.prompt_key=extra_info \
    data.response_key=extra_info \
    optim.lr=1e-4 \
    data.prompt_dict_keys=['question'] \
    +data.response_dict_keys=['answer'] \
    data.micro_batch_size_per_gpu=64 \
    model.partial_pretrain=$MODEL_PATH \
    trainer.default_local_dir=$save_path \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.logger=console \
    trainer.total_epochs=5 \
    model.strategy=fsdp \
    ulysses_sequence_parallel_size=2 \
    use_remove_padding=true \
    $@