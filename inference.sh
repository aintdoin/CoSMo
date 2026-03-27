#!/bin/bash
set -e

export CUDA_VISIBLE_DEVICES=0
MODEL_PATH=/mnt/shared-storage-user/liyafu/models/Llama-3.1-8B-Instruct
DATASET=HotpotQA #HotpotQA, Halueval
DATASET_PATH="data/${DATASET}/test.parquet"  #data/MuSiQue/hops_split/4hop_1000.parquet
MODEL_TEMPLATE=llama #qwen, llama
OUTPUT_DIR="output/inference/$DATASET/$MODEL_TEMPLATE"    #debug
PROMPT_TEMPLATE="cot"  # cot, directly, tot, htp, cod, tale


#cosmo:  /mnt/shared-storage-user/liyafu/runquan/SMIR/checkpoints/SMIR/llama_8b/cosmo/global_step_168/actor/huggingface  qwen_7b
#base:   /mnt/shared-storage-user/liyafu/models/Llama-3.1-8B-Instruct     Qwen2.5-7B-Instruct  
#C3oT:  /mnt/shared-storage-user/liyafu/runquan/SMIR/SFT/output/C3oT_llama/global_step_936  C3oT/global_step_936

# LLM Judge Config
export LLM_JUDGE_API_BASE="http://localhost:8000"
export LLM_JUDGE_MODEL_NAME=/mnt/shared-storage-user/liyafu/models/Qwen2.5-14B-Instruct
export LLM_JUDGE_API_KEY=""
export LLM_JUDGE_MAX_WORKERS=8
export LLM_JUDGE_TIMEOUT=60
TEMPERATURE=0
MAX_OUTPUT=2048
# Run Inference
python3 inference.py \
    --model "$MODEL_PATH" \
    --datasets "$DATASET_PATH" \
    --output-dir "$OUTPUT_DIR" \
    --temperature "$TEMPERATURE" \
    --max-output "$MAX_OUTPUT" \
    --prompt-template "$PROMPT_TEMPLATE" \
    --model-template "$MODEL_TEMPLATE" \
    --judge-api-base "$LLM_JUDGE_API_BASE" \
    --judge-model-name "$LLM_JUDGE_MODEL_NAME" \
    --judge-api-key "$LLM_JUDGE_API_KEY" \
    --judge-max-workers "$LLM_JUDGE_MAX_WORKERS" \
    --judge-timeout "$LLM_JUDGE_TIMEOUT"

