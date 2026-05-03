#!/usr/bin/env bash
set -euo pipefail

# OpenAI-compatible vLLM server used by CoSMo SFT split-merge generation.
# Override the variables below from the shell instead of editing this file.

LOG_DIR="${LOG_DIR:-./logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/LLM_judge_server.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/LLM_judge_server.pid}"

MODEL_PATH="${MODEL_PATH:-Qwen/Qwen2.5-72B-Instruct}"
SERVED_MODEL_NAME="${SERVED_MODEL_NAME:-Qwen/Qwen2.5-72B-Instruct}"
HOST="${HOST:-0.0.0.0}"
PORT="${PORT:-8000}"
MAX_MODEL_LEN="${MAX_MODEL_LEN:-4096}"
GPU_UTIL="${GPU_UTIL:-0.85}"
TENSOR_PARALLEL_SIZE="${TENSOR_PARALLEL_SIZE:-2}"
MAX_NUM_SEQS="${MAX_NUM_SEQS:-64}"
MAX_NUM_BATCHED_TOKENS="${MAX_NUM_BATCHED_TOKENS:-8192}"
DTYPE="${DTYPE:-bfloat16}"
PYTHON_BIN="${PYTHON_BIN:-python3}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"

export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-spawn}"
export CUDA_VISIBLE_DEVICES
GPU_MESSAGE="CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES}"

mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${PID_FILE}")"

echo "Starting vLLM OpenAI server on ${HOST}:${PORT}"
echo "Model path: ${MODEL_PATH}"
echo "Served model name: ${SERVED_MODEL_NAME}"
echo "Tensor parallel size: ${TENSOR_PARALLEL_SIZE} (${GPU_MESSAGE})"
echo "Logs: ${LOG_FILE}"
echo "PID file: ${PID_FILE}"

cmd=(
  "${PYTHON_BIN}" -m vllm.entrypoints.openai.api_server
  --host "${HOST}"
  --port "${PORT}"
  --model "${MODEL_PATH}"
  --served-model-name "${SERVED_MODEL_NAME}"
  --max-model-len "${MAX_MODEL_LEN}"
  --gpu-memory-utilization "${GPU_UTIL}"
  --dtype "${DTYPE}"
  --trust-remote-code
  --tensor-parallel-size "${TENSOR_PARALLEL_SIZE}"
  --enable-prefix-caching
  --max-num-seqs "${MAX_NUM_SEQS}"
  --max-num-batched-tokens "${MAX_NUM_BATCHED_TOKENS}"
)

nohup "${cmd[@]}" > "${LOG_FILE}" 2>&1 &
SERVER_PID=$!
echo "${SERVER_PID}" > "${PID_FILE}"
echo "vLLM server started. PID=${SERVER_PID}"
