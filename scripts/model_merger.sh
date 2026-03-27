export CUDA_VISIBLE_DEVICES=5
LOCAL_DIR="checkpoints/SMIR/llama_8b/cosmo/global_step_168/actor"

python scripts/model_merger.py --local_dir "$LOCAL_DIR"