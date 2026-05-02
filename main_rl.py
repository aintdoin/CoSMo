import argparse
import os
import subprocess
import sys


def hydra_list(values: list[str]) -> str:
    return "[" + ",".join(values) + "]"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoSMo GRPO entry point with segment-budget reward.")
    parser.add_argument("--sft-model", default=os.environ.get("COSMO_SFT_CKPT", "checkpoints/cosmo_sft/final"))
    parser.add_argument("--model-template", choices=["llama", "qwen"], default=os.environ.get("COSMO_MODEL_TEMPLATE", "llama"))
    parser.add_argument("--train-files", nargs="+", default=["data/rl/hotpotqa.parquet", "data/rl/halueval.parquet"])
    parser.add_argument("--val-files", nargs="+", default=["data/rl/hotpotqa_val.parquet", "data/rl/halueval_val.parquet"])
    parser.add_argument("--output-dir", default="checkpoints/cosmo_rl")
    parser.add_argument("--gpus", type=int, default=4)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--val-batch-size", type=int, default=64)
    parser.add_argument("--max-prompt-length", type=int, default=4096)
    parser.add_argument("--max-response-length", type=int, default=2048)
    parser.add_argument("--group-size", type=int, default=8)
    parser.add_argument("--clip-ratio", type=float, default=0.1)
    parser.add_argument("--lr", type=float, default=1e-6)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--segment-tolerance", type=int, default=1)
    parser.add_argument("--segment-penalty-scale", type=float, default=1.0)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    env["STRATEGY"] = "cosmo"
    env["COSMO_SEGMENT_TOLERANCE"] = str(args.segment_tolerance)
    env["COSMO_SEGMENT_PENALTY_SCALE"] = str(args.segment_penalty_scale)

    cmd = [
        sys.executable,
        "-m",
        "verl.trainer.main_ppo",
        f"data.train_files={hydra_list(args.train_files)}",
        f"data.val_files={hydra_list(args.val_files)}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.val_batch_size={args.val_batch_size}",
        f"data.max_prompt_length={args.max_prompt_length}",
        f"data.max_response_length={args.max_response_length}",
        f"data.model_template={args.model_template}",
        f"actor_rollout_ref.model.path={args.sft_model}",
        f"actor_rollout_ref.actor.optim.lr={args.lr}",
        f"actor_rollout_ref.actor.clip_ratio={args.clip_ratio}",
        f"actor_rollout_ref.rollout.n={args.group_size}",
        f"algorithm.adv_estimator=grpo",
        f"reward_model.reward_manager=cosmo",
        f"trainer.n_gpus_per_node={args.gpus}",
        f"trainer.total_epochs={args.epochs}",
        f"trainer.project_name=CoSMo-RL",
        f"trainer.experiment_name=hotpotqa-halueval",
        f"trainer.default_local_dir={args.output_dir}",
    ]
    subprocess.run(cmd, check=True, env=env)


if __name__ == "__main__":
    main()
