
# SFT Training
policy_model_name_or_path=qwen2.5_3b_instruct
policy_pretrain_model=Qwen/Qwen2.5-3B-Instruct 

deepspeed --module openrlhf.cli.train_sft \
    --max_len 2048 \
    --dataset /path/to/training_data/policy_sft_${policy_model_name_or_path} \
    --dataset_probs 1.0 \
    --input_key prompt \
    --output_key label \
    --input_template {} \
    --pretrain ${policy_pretrain_model} \
    --train_batch_size 64 \
    --micro_train_batch_size 4 \
    --save_path /path/to/checkpoint/${policy_model_name_or_path}_sft \
    --save_steps 0 \
    --ckpt_path /path/to/checkpoint/${policy_model_name_or_path}_sft \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --logging_steps 5 \
    --eval_steps 50 \
    --zero_stage 2 \
    --max_epochs 1 \
    --bf16 \
    --flash_attn \
    --learning_rate 5e-6 \
    --gradient_checkpointing \
    --adam_offload \
    --overlap_comm

# PPO Training 
policy_pretrain_model=/path/to/sft/checkpoing
reasoning_model=Qwen/Qwen2.5-7B-Instruct
critic_pretrain_model=Qwen/Qwen2.5-3B-Instruct
policy_model_name=qwen2.5_3b_instruct 
reasoning_model_name=qwen2.5_7b_instruct

deepspeed --module openrlhf.cli.train_ppo_kg_adaptive_rag \
    --pretrain ${policy_pretrain_model} \
    --reasoning_model ${reasoning_model} \
    --critic_pretrain ${critic_pretrain_model} \
    --save_path /path/to/checkpoint/${policy_model_name}_${reasoning_model_name}_ppo_tree_rollout \
    --save_steps 10 \
    --ckpt_path /path/to/checkpoint/${policy_model_name}_${reasoning_model_name}_ppo_tree_rollout \
    --save_hf_ckpt \
    --disable_ds_ckpt \
    --logging_steps 1 \
    --eval_steps 5 \
    --num_episodes 1 \
    --rollout_batch_size 64 \
    --micro_rollout_batch_size 4 \
    --n_samples_per_prompt 1 \
    --max_epochs 1 \
    --train_batch_size 1024 \
    --micro_train_batch_size 4 \
    --prompt_max_len 2048 \
    --generate_max_len 160 \
    --zero_stage 2 \
    --bf16 \
    --actor_learning_rate 5e-7 \
    --critic_learning_rate 5e-6 \
    --kl_estimator k1 \
    --init_kl_coef 0.002 \
    --rm_use_recall \
    --prompt_data /path/to/train/data \
    --normalize_reward \
    --flash_attn \
    --gradient_checkpointing \
    --adam_offload \
    --max_samples 200000 \
    --tree_rollout 

