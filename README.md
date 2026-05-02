# CoSMo

Code for **Short Chains, Deep Thoughts: Balancing Reasoning Efficiency and Intra-Segment Capability via Split-Merge Optimization**.

CoSMo has two stages:

1. **SFT data construction with Split-Merge Optimization**: Qwen2.5-72B-Instruct acts as the judge and generator. The pipeline keeps every reasoning unit inside XML blocks, `<seg>...</seg>`, then iteratively merges redundant adjacent segments or splits coarse segments.
2. **Structure-aligned RL**: verl/GRPO trains the SFT model with a segment-level budget reward when `golden_segments` is available: `r = r_format + r_correctness - |N - golden_segments|`. Token length inside each segment is not penalized. If `golden_segments` is missing, the reward falls back to standard answer-based GRPO.

## Install

```bash
conda create -n cosmo python=3.10
conda activate cosmo
pip install -r requirements.txt
pip install -e .
```

## Data Format

This repository does not currently include dataset-specific preprocessing scripts. Before running SFT or RL, convert the data to parquet files such as `data/<dataset>/<split>.parquet`.

Currently supported datasets are HotpotQA, 2WikiMultihopQA, MuSiQue, HaluEval, NQ, CRAG, MATH500, and GSM8K. Use the same parquet row contract across datasets:

```python
{
    "prompt": "Question and optional references/context shown to the model.",
    "response": "<think>\n<seg>...</seg>\n</think>\n<answer>...</answer>",
    "answer": "gold answer",
    "ground_truth": "gold answer, optional alias for RL",
    "golden_segments": 2,
    "extra_info": {"index": 0}
}
```

`prompt` is required. `response` is required for `python main_sft.py --stage train`; it can be omitted during `--stage prepare` only if `--generate-missing` is used. RL data should contain `prompt` and either `ground_truth` or `answer`.

`golden_segments` is optional and should be a positive integer segment target. When HotpotQA, 2WikiMultihopQA, or MuSiQue is used as training data, providing `golden_segments` is recommended so Split-Merge SFT and RL can align to the intended hop/segment count. If it is absent, SFT uses the no-gold pairwise split-merge path, and RL degrades to answer-based GRPO.

## LLM Judge Deployment

SFT preparation uses an OpenAI-compatible LLM judge/generator. The provided script starts a vLLM server without hard-coded personal paths:

```bash
MODEL_PATH=/path/to/Qwen2.5-72B-Instruct \
SERVED_MODEL_NAME=Qwen/Qwen2.5-72B-Instruct \
CUDA_VISIBLE_DEVICES=0,1,2,3 \
TENSOR_PARALLEL_SIZE=4 \
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
  --sft-data data/sft/hotpotqa_cosmo.parquet \
  --generate-missing
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

Expected SFT training columns are `prompt` and `response`. If the input rows contain `golden_segments`, `gold_hops`, `hop`, or `hops`, CoSMo aligns the segment count to that target. For HotpotQA, 2WikiMultihopQA, and MuSiQue this uses the golden segment path when the target is provided; for HaluEval, GSM8K, NQ, CRAG, and MATH500 it falls back to iterative pairwise merge then per-segment split until convergence or the iteration limit.

## Stage 2: RL

Run GRPO from the SFT checkpoint on HotpotQA and HaluEval:

```bash
python main_rl.py \
  --sft-model checkpoints/cosmo_sft/final \
  --train-files data/rl/hotpotqa.parquet data/rl/halueval.parquet \
  --val-files data/rl/hotpotqa_val.parquet data/rl/halueval_val.parquet \
  --output-dir checkpoints/cosmo_rl
```

RL data should contain `prompt`, a ground-truth answer column such as `ground_truth` or `answer`, and a segment target such as `golden_segments`, `hop`, or `hops` when available. The CoSMo reward manager enforces the `<think><seg>...</seg></think><answer>...</answer>` format and applies the segment-budget penalty only when a segment target exists; otherwise it falls back to standard GRPO correctness reward.
