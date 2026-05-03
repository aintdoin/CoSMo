import asyncio
import json
import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from .async_llm import AsyncOpenAIChatClient, LLMRequestError
from .prompts import (
    MERGE_GENERATOR_PROMPT,
    MERGE_JUDGE_PROMPT,
    SPLIT_GENERATOR_PROMPT,
    SPLIT_JUDGE_PROMPT,
    render_prompt,
)
from .segments import clean_segment, extract_answer, extract_segments, format_response

if TYPE_CHECKING:
    import pandas as pd


GOLDEN_SEGMENT_KEYS = ("golden_segments",)


@dataclass
class SplitMergeResult:
    segments: List[str]
    answer: str
    iterations: int
    status: str


def has_golden_segments(golden_segments: Optional[int]) -> bool:
    return golden_segments is not None and golden_segments > 0


def coerce_golden_segments(value) -> Optional[int]:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass
    if isinstance(value, str):
        value = value.strip()
        if not value or value.lower() in {"nan", "none", "null"}:
            return None
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            pass
        else:
            return coerce_golden_segments(parsed)
    if isinstance(value, dict):
        for key in GOLDEN_SEGMENT_KEYS:
            if key in value:
                return coerce_golden_segments(value[key])
        return None
    if isinstance(value, (list, tuple, set)):
        return len(value) if len(value) > 0 else None
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def coerce_extra_info(value) -> Dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def extract_golden_segments(row: Dict) -> Optional[int]:
    for key in GOLDEN_SEGMENT_KEYS:
        if key in row:
            value = coerce_golden_segments(row.get(key))
            if value is not None:
                return value
    extra = coerce_extra_info(row.get("extra_info"))
    for key in GOLDEN_SEGMENT_KEYS:
        if key in extra:
            value = coerce_golden_segments(extra.get(key))
            if value is not None:
                return value
    return None


class SplitMergeOptimizer:
    def __init__(self, client: AsyncOpenAIChatClient, max_iterations: int = 5):
        self.client = client
        self.max_iterations = max_iterations

    async def _json_prompt(self, prompt: str) -> Dict:
        data = await self.client.json_chat(
            [
                {"role": "system", "content": "You are a strict JSON-output reasoning editor."},
                {"role": "user", "content": prompt},
            ],
            max_tokens=700,
        )
        if not isinstance(data, dict):
            raise LLMRequestError("judge returned non-object JSON")
        return data

    async def _text_prompt(self, prompt: str) -> str:
        return await self.client.chat(
            [
                {"role": "system", "content": "You are a precise XML-output reasoning editor."},
                {"role": "user", "content": prompt},
            ],
            temperature=0.0,
            max_tokens=900,
        )

    async def judge_merge(self, left: str, right: str) -> bool:
        data = await self._json_prompt(render_prompt(MERGE_JUDGE_PROMPT, segment_1=left, segment_2=right))
        return str(data.get("decision", "")).strip().lower() == "merge"

    async def generate_merge(self, left: str, right: str) -> str:
        text = await self._text_prompt(render_prompt(MERGE_GENERATOR_PROMPT, segment_1=left, segment_2=right))
        segments = extract_segments(text)
        if segments:
            return clean_segment(segments[0])
        return clean_segment(f"{left} {right}")

    async def judge_split(self, segment: str) -> bool:
        data = await self._json_prompt(render_prompt(SPLIT_JUDGE_PROMPT, segment=segment))
        return str(data.get("decision", "")).strip().lower() == "split"

    async def generate_split(self, segment: str) -> List[str]:
        text = await self._text_prompt(render_prompt(SPLIT_GENERATOR_PROMPT, segment=segment))
        segments = extract_segments(text)
        segments = [clean_segment(s) for s in segments if clean_segment(s)]
        if len(segments) >= 2:
            return segments[:2]
        return [clean_segment(segment)]

    async def run_with_gold(self, segments: Iterable[str], golden_segments: int) -> SplitMergeResult:
        current = [clean_segment(s) for s in segments if clean_segment(s)]
        iterations = 0
        status = "unchanged"

        while iterations < self.max_iterations and len(current) != golden_segments:
            modified = False
            if len(current) > golden_segments:
                for idx in range(len(current) - 1):
                    if await self.judge_merge(current[idx], current[idx + 1]):
                        merged = await self.generate_merge(current[idx], current[idx + 1])
                        current = current[:idx] + [merged] + current[idx + 2 :]
                        modified = True
                        status = "merged"
                        break
            elif len(current) < golden_segments:
                for idx, segment in enumerate(current):
                    if await self.judge_split(segment):
                        split = await self.generate_split(segment)
                        if len(split) > 1:
                            current = current[:idx] + split + current[idx + 1 :]
                            modified = True
                            status = "split"
                            break

            iterations += 1
            if not modified:
                status = "local_optimum"
                break

        if len(current) == golden_segments:
            status = "aligned"
        return SplitMergeResult(segments=current, answer="", iterations=iterations, status=status)

    async def refine(self, text: str, answer: str = "", golden_segments: int = 0) -> SplitMergeResult:
        segments = extract_segments(text)
        if not segments:
            return SplitMergeResult(segments=[], answer=answer, iterations=0, status="no_segments")
        if golden_segments <= 0:
            raise ValueError("`golden_segments` must be a positive integer.")
        result = await self.run_with_gold(segments, golden_segments)
        result.answer = answer
        return result


def build_prompt_from_row(row: Dict) -> str:
    return require_text_field(row, "prompt")


def build_seed_text(row: Dict) -> str:
    value = row.get("response")
    if value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip():
        raise ValueError("Input row is missing required `response`.")
    return str(value)


def require_text_field(row: Dict, key: str) -> str:
    value = row.get(key)
    if value is None or (isinstance(value, float) and math.isnan(value)) or not str(value).strip():
        raise ValueError(f"Input row is missing required `{key}`.")
    return str(value)


async def process_row(
    row: Dict,
    client: AsyncOpenAIChatClient,
    dataset: str,
    max_iterations: int,
) -> Dict:
    prompt = build_prompt_from_row(row)
    seed_text = build_seed_text(row)

    ground_truth = require_text_field(row, "ground_truth")
    answer = extract_answer(seed_text, fallback=ground_truth)
    golden_segments = extract_golden_segments(row)
    if not has_golden_segments(golden_segments):
        raise ValueError("Input row is missing required positive integer `golden_segments`.")

    optimizer = SplitMergeOptimizer(client=client, max_iterations=max_iterations)
    try:
        result = await optimizer.refine(seed_text, answer=answer, golden_segments=golden_segments)
    except (LLMRequestError, json.JSONDecodeError) as exc:
        result = SplitMergeResult(
            segments=extract_segments(seed_text),
            answer=answer,
            iterations=0,
            status=f"llm_error:{exc}",
        )

    response = format_response(result.segments, result.answer)
    extra_info = coerce_extra_info(row.get("extra_info", {}))
    if golden_segments is not None:
        extra_info["golden_segments"] = golden_segments

    return {
        "prompt": prompt,
        "response": response,
        "ground_truth": ground_truth,
        "golden_segments": golden_segments,
        "segments": len(result.segments),
        "status": result.status,
        "iterations": result.iterations,
        "extra_info": extra_info,
    }


async def process_dataframe(
    df: "pd.DataFrame",
    client: AsyncOpenAIChatClient,
    dataset: str,
    max_iterations: int = 5,
    chunk_size: int = 128,
) -> "pd.DataFrame":
    import pandas as pd

    rows = df.to_dict(orient="records")
    outputs = []
    for start in range(0, len(rows), chunk_size):
        chunk = rows[start : start + chunk_size]
        tasks = [
            process_row(
                row=row,
                client=client,
                dataset=dataset,
                max_iterations=max_iterations,
            )
            for row in chunk
        ]
        outputs.extend(await asyncio.gather(*tasks))
        print(f"processed {min(start + chunk_size, len(rows))}/{len(rows)}")
    return pd.DataFrame(outputs)
