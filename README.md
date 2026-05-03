# CoSMo

Code for **Short Chains, Deep Thoughts: Balancing Reasoning Efficiency and Intra-Segment Capability via Split-Merge Optimization**.

## Install

```bash
conda create -n cosmo python=3.10
conda activate cosmo
pip install -r requirements.txt
pip install -e .
```

## Data Format

This repository does not currently include dataset-specific preprocessing scripts. Before running SFT or RL, convert the data to parquet files such as `data/<dataset>/<split>.parquet`.

Use the same parquet row contract across datasets:

```python
{
    "prompt": "Question and optional references/context shown to the model.",
    "response": "<think>\n<seg>...</seg>\n</think>\n<answer>...</answer>",
    "ground_truth": "gold answer",
    "golden_segments": 2,
    "extra_info": {"index": 0}
}
```

`prompt`, `response`, `ground_truth`, and `golden_segments` are required. `golden_segments` must be a positive integer segment target used by both Split-Merge SFT data construction and RL segment-budget reward.

## LLM Judge Deployment

SFT preparation uses an OpenAI-compatible LLM judge/generator. The provided script starts a vLLM server without hard-coded personal paths:

```bash
MODEL_PATH=/path/to/Qwen2.5-72B-Instruct \
SERVED_MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
CUDA_VISIBLE_DEVICES=0,1 \
TENSOR_PARALLEL_SIZE=2 \
PORT=8000 \
bash LLM_judge.sh
```

Then point CoSMo to the service:

```bash
export COSMO_API_BASE=http://localhost:8000/v1
export COSMO_API_KEY=EMPTY
export COSMO_JUDGE_MODEL=Qwen/Qwen2.5-72B-Instruct
```

## Stage 1: SFT

Prepare HotpotQA SFT data:

```bash
python main_sft.py \
  --stage prepare \
  --dataset hotpotqa \
  --input data/hotpotqa/train.parquet \
  --sft-data data/sft/hotpotqa_cosmo.parquet
```

Train the SFT model with the bundled verl FSDP SFT trainer. The default backbone is `Llama-3.1-8B-Instruct`; use `--backbone qwen` for `Qwen2.5-7B-Instruct`. By default `--lora-rank 0` runs full-model SFT so the checkpoint can be loaded directly by the RL stage.

```bash
python main_sft.py \
  --stage train \
  --backbone llama \
  --sft-data data/sft/hotpotqa_cosmo.parquet \
  --output-dir checkpoints/cosmo_sft
```

verl saves step checkpoints under `checkpoints/cosmo_sft/global_step_*`; `main_sft.py` links the latest one to `checkpoints/cosmo_sft/final` for the RL stage.
If you enable LoRA with `--lora-rank > 0`, merge the adapter into the base model before using the checkpoint as `--sft-model`.

## Stage 2: RL

Run GRPO from the SFT checkpoint:

```bash
python main_rl.py \
  --sft-model checkpoints/cosmo_sft/final \
  --train-files data/rl/hotpotqa.parquet data/rl/halueval.parquet \
  --val-files data/rl/hotpotqa_val.parquet data/rl/halueval_val.parquet \
  --output-dir checkpoints/cosmo_rl
```
