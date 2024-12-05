#!/usr/bin/env bash
# Copyright 2024  Bofeng Huang

set -e

echo "START TIME: $(date)"

# https://github.com/pytorch/audio/issues/1021#issuecomment-726915239
# export OMP_NUM_THREADS="1"

# cuda
export CUDA_VISIBLE_DEVICES="4,5,6,7"

# hf
export HF_HOME="/projects/bhuang/.cache/huggingface"
export TOKENIZERS_PARALLELISM="false"
# export BITSANDBYTES_NOWELCOME="1"
# export HF_HUB_ENABLE_HF_TRANSFER="1"
# export HF_HUB_OFFLINE="1"
# export HF_DATASETS_OFFLINE="1"
# export HF_EVALUATE_OFFLINE="1"

export VLLM_WORKER_MULTIPROC_METHOD=spawn

model_name_or_path=/projects/bhuang/models/llm/pretrained/meta-llama/Llama-3.2-1B-Instruct
# model_name_or_path=/projects/bhuang/models/llm/pretrained/meta-llama/Meta-Llama-3.1-8B-Instruct
# model_name_or_path=bofenghuang/Llama-3-Vgn-8B-Instruct-v0.6
# model_name_or_path=OpenLLM-France/Claire-7B-FR-Instruct-0.1

gpus_per_model=1
model_replicas=4

    # --tasks 'examples/tasks/finetasks/{cf,mcf}/{ara,fra,rus,tur,swa,hin,tel,tha,zho}' \
    # --max_samples 1000
    # --override_batch_size 1 \
    # --model_args vllm,pretrained=${model_name_or_path},pairwise_tokenization=True,dtype=bfloat16 \
    # --model_args vllm,pretrained=${model_name_or_path},tensor_parallel_size=${gpus_per_model},dtype=bfloat16 \
    # --model_args vllm,pretrained=${model_name_or_path},tensor_parallel_size=${gpus_per_model},dtype=bfloat16,data_parallel_size=${model_replicas},pairwise_tokenization=True \
    # --model_args vllm,pretrained=${model_name_or_path},tensor_parallel_size=${gpus_per_model},dtype=bfloat16,data_parallel_size=${model_replicas},pairwise_tokenization=True,max_model_length=4096 \

lighteval accelerate \
    --model_args vllm,pretrained=${model_name_or_path},tensor_parallel_size=${gpus_per_model},dtype=bfloat16,data_parallel_size=${model_replicas},pairwise_tokenization=True,max_model_length=4096 \
    --custom_task lighteval.tasks.multilingual.tasks \
    --tasks examples/tasks/fine_tasks/mcf/fr.txt \
    --output_dir ./results \


echo "END TIME: $(date)"
