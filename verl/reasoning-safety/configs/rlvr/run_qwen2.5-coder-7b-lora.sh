#!/bin/bash
set -x

# Should be in `verl/reasoning_safety`
PROJECT_DIR="$(pwd)"

PROJECT_NAME="verl_reasoning_safety"
TRAIN_PATH="${PROJECT_DIR}/data/grpo/train.parquet"
TEST_PATH="${PROJECT_DIR}/data/grpo/test.parquet"

MODEL_PATH="Qwen/Qwen2.5-Coder-7B-Instruct"
EXPERIMENT_NAME="rlvr_qwen25_coder_7b_orig"

python -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    trainer.val_before_train=True \
    data.filter_overlong_prompts_workers=24 \
    data.train_batch_size=32 \
    data.train_files=$TRAIN_PATH \
    data.val_files=$TEST_PATH \
    data.max_prompt_length=2048 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    data.shuffle=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console","wandb"]' \
    trainer.project_name=$PROJECT_NAME \
    trainer.experiment_name=$EXPERIMENT_NAME \
    trainer.n_gpus_per_node=2 \
    trainer.nnodes=1 \
    trainer.save_freq=40 \
    trainer.test_freq=10 \
    trainer.rollout_data_dir="rollouts/${EXPERIMENT_NAME}" \
    trainer.total_epochs=10 \
    actor_rollout_ref.model.path=$MODEL_PATH \
    actor_rollout_ref.actor.fsdp_config.model_dtype='bf16' \
    actor_rollout_ref.model.use_liger=True \
    actor_rollout_ref.actor.optim.lr=3e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=16 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.8 \
    actor_rollout_ref.rollout.dtype='bfloat16' \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.enforce_eager=False \
    actor_rollout_ref.rollout.free_cache_engine=True \
    actor_rollout_ref.rollout.load_format=safetensors \
    actor_rollout_ref.rollout.layered_summon=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.model_dtype='bf16' \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    reward_model.enable=False \
    reward_model.reward_manager='countdown_code' \
    actor_rollout_ref.model.lora_rank=128 \
    actor_rollout_ref.model.lora_alpha=128 \
    $@
