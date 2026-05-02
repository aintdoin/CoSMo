import argparse
import asyncio
import os
from pathlib import Path

import pandas as pd

from cosmo.async_llm import AsyncOpenAIChatClient
from cosmo.split_merge import process_dataframe


def read_table(path: str) -> pd.DataFrame:
    suffix = Path(path).suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix in {".jsonl", ".json"}:
        return pd.read_json(path, lines=suffix == ".jsonl")
    if suffix == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"Unsupported input format: {path}")


def write_table(df: pd.DataFrame, path: str) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    suffix = output.suffix.lower()
    if suffix == ".parquet":
        df.to_parquet(output, index=False)
    elif suffix == ".jsonl":
        df.to_json(output, orient="records", lines=True, force_ascii=False)
    elif suffix == ".csv":
        df.to_csv(output, index=False)
    else:
        raise ValueError(f"Unsupported output format: {path}")


async def run(args: argparse.Namespace) -> None:
    df = read_table(args.input)
    if args.max_samples:
        df = df.head(args.max_samples)

    async with AsyncOpenAIChatClient(
        api_base=args.api_base,
        api_key=args.api_key,
        model=args.judge_model,
        timeout=args.timeout,
        max_retries=args.max_retries,
        initial_concurrency=args.concurrency,
        max_concurrency=args.max_concurrency,
    ) as client:
        output_df = await process_dataframe(
            df=df,
            client=client,
            dataset=args.dataset,
            max_iterations=args.max_iterations,
            generate_missing=args.generate_missing,
            seed_max_tokens=args.seed_max_tokens,
            chunk_size=args.chunk_size,
        )

    write_table(output_df, args.output)
    status_counts = output_df["status"].value_counts().to_dict() if "status" in output_df else {}
    print(f"wrote {len(output_df)} samples to {args.output}")
    print(f"status: {status_counts}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate CoSMo SFT data with Qwen2.5-72B Split-Merge optimization.")
    parser.add_argument("--input", default="data/hotpotqa/train.jsonl")
    parser.add_argument("--output", default="data/sft/hotpotqa_cosmo.parquet")
    parser.add_argument("--dataset", default="hotpotqa")
    parser.add_argument("--max-samples", type=int, default=0)
    parser.add_argument("--max-iterations", type=int, default=5)
    parser.add_argument("--chunk-size", type=int, default=128)
    parser.add_argument("--generate-missing", action="store_true", help="Use the 72B model to generate seed reasoning if no response exists.")
    parser.add_argument("--seed-max-tokens", type=int, default=2048)

    parser.add_argument("--api-base", default=os.environ.get("COSMO_API_BASE", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("COSMO_API_KEY", "EMPTY"))
    parser.add_argument("--judge-model", default=os.environ.get("COSMO_JUDGE_MODEL", "Qwen/Qwen2.5-72B-Instruct"))
    parser.add_argument("--timeout", type=int, default=int(os.environ.get("COSMO_TIMEOUT", "120")))
    parser.add_argument("--max-retries", type=int, default=int(os.environ.get("COSMO_MAX_RETRIES", "5")))
    parser.add_argument("--concurrency", type=int, default=int(os.environ.get("COSMO_CONCURRENCY", "16")))
    parser.add_argument("--max-concurrency", type=int, default=int(os.environ.get("COSMO_MAX_CONCURRENCY", "64")))
    return parser.parse_args()


def main() -> None:
    asyncio.run(run(parse_args()))


if __name__ == "__main__":
    main()
