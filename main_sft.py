import argparse
import os
from pathlib import Path
import re
import subprocess
import sys


DEFAULT_MODELS = {
    "llama": "meta-llama/Llama-3.1-8B-Instruct",
    "qwen": "Qwen/Qwen2.5-7B-Instruct",
}


def _global_step(path: Path) -> int:
    match = re.fullmatch(r"global_step_(\d+)", path.name)
    return int(match.group(1)) if match else -1


def link_latest_sft_checkpoint(output_dir: str) -> None:
    root = Path(output_dir)
    if not root.exists():
        return

    checkpoints = [path for path in root.iterdir() if path.is_dir() and _global_step(path) >= 0]
    if not checkpoints:
        return

    latest = max(checkpoints, key=_global_step)
    final = root / "final"
    if final.is_symlink():
        final.unlink()
    elif final.exists():
        print(f"Skip final checkpoint link because {final} already exists and is not a symlink.", file=sys.stderr)
        return

    try:
        final.symlink_to(latest.name, target_is_directory=True)
        print(f"Linked {final} -> {latest.name}")
    except OSError as exc:
        print(f"Could not create final checkpoint link for {latest}: {exc}", file=sys.stderr)


def run_prepare(args: argparse.Namespace) -> None:
    cmd = [
        sys.executable,
        "SFT/sft_data.py",
        "--input",
        args.input,
        "--output",
        args.sft_data,
        "--dataset",
        args.dataset,
        "--max-iterations",
        str(args.max_iterations),
        "--chunk-size",
        str(args.chunk_size),
        "--concurrency",
        str(args.concurrency),
        "--max-concurrency",
        str(args.max_concurrency),
    ]
    if args.max_samples:
        cmd += ["--max-samples", str(args.max_samples)]
    subprocess.run(cmd, check=True)


def run_train(args: argparse.Namespace) -> None:
    model_path = args.model_path or DEFAULT_MODELS[args.backbone]
    cmd = [
        sys.executable,
        "-m",
        "torch.distributed.run",
        "--standalone",
        "--nnodes=1",
        f"--nproc_per_node={args.gpus}",
        "-m",
        "verl.trainer.fsdp_sft_trainer",
        f"data.train_files={args.sft_data}",
        f"data.val_files={args.val_data or args.sft_data}",
        f"data.train_batch_size={args.train_batch_size}",
        f"data.micro_batch_size_per_gpu={args.micro_batch_size}",
        f"data.max_length={args.max_length}",
        f"model.partial_pretrain={model_path}",
        f"model.lora_rank={args.lora_rank}",
        f"model.lora_alpha={args.lora_alpha}",
        f"optim.lr={args.lr}",
        f"trainer.total_epochs={args.epochs}",
        f"trainer.default_local_dir={args.output_dir}",
        f"trainer.project_name=CoSMo-SFT",
        f"trainer.experiment_name={args.dataset}-{args.backbone}",
    ]
    subprocess.run(cmd, check=True)
    link_latest_sft_checkpoint(args.output_dir)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CoSMo SFT entry point.")
    parser.add_argument("--stage", choices=["prepare", "train", "all"], default="prepare")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--input", default="data/hotpotqa/train.parquet")
    parser.add_argument("--sft-data", default="data/sft/hotpotqa_cosmo.parquet")
    parser.add_argument("--val-data", default="")
    parser.add_argument("--backbone", choices=sorted(DEFAULT_MODELS), default="llama")
    parser.add_argument("--model-path", default=os.environ.get("COSMO_SFT_MODEL", ""))
    parser.add_argument("--output-dir", default="checkpoints/cosmo_sft")

    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("COSMO_CONCURRENCY", "16")))
    parser.add_argument("--max-concurrency", type=int, default=int(os.environ.get("COSMO_MAX_CONCURRENCY", "64")))

    parser.add_argument("--gpus", type=int, default=2)
    parser.add_argument("--train-batch-size", type=int, default=64)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--max-length", type=int, default=8192)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--lora-rank", type=int, default=0)
    parser.add_argument("--lora-alpha", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.stage in {"prepare", "all"}:
        run_prepare(args)
    if args.stage in {"train", "all"}:
        run_train(args)


if __name__ == "__main__":
    main()
