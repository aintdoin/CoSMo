import asyncio
import json
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Iterable, List, Optional

from .async_llm import AsyncOpenAIChatClient, LLMRequestError
from .prompts import (
    MERGE_GENERATOR_PROMPT,
    MERGE_JUDGE_PROMPT,
    PAIRWISE_MERGE_PROMPT,
    SINGLE_SPLIT_PROMPT,
    SPLIT_GENERATOR_PROMPT,
    SPLIT_JUDGE_PROMPT,
    STRUCTURED_REASONING_SYSTEM_PROMPT,
    render_prompt,
)
from .segments import clean_segment, extract_answer, extract_segments, format_response

if TYPE_CHECKING:
    import pandas as pd


GOLDEN_SEGMENT_KEYS = ("golden_segments", "gold_segments", "gold_hops", "hops", "hop", "num_hops")


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

    async def pairwise_merge_or_keep(self, left: str, right: str) -> List[str]:
        data = await self._json_prompt(render_prompt(PAIRWISE_MERGE_PROMPT, segment_1=left, segment_2=right))
        decision = str(data.get("decision", "")).strip().lower()
        raw_segments = data.get("segments", [])
        if not isinstance(raw_segments, list):
            raw_segments = []
        segments = [clean_segment(s) for s in raw_segments if clean_segment(s)]
        if decision == "merge":
            return [segments[0] if segments else clean_segment(f"{left} {right}")]
        if len(segments) >= 2:
            return segments[:2]
        return [clean_segment(left), clean_segment(right)]

    async def split_or_keep(self, segment: str) -> List[str]:
        data = await self._json_prompt(render_prompt(SINGLE_SPLIT_PROMPT, segment=segment))
        decision = str(data.get("decision", "")).strip().lower()
        raw_segments = data.get("segments", [])
        if not isinstance(raw_segments, list):
            raw_segments = []
        segments = [clean_segment(s) for s in raw_segments if clean_segment(s)]
        if decision == "split" and len(segments) >= 2:
            return segments[:2]
        return [segments[0] if segments else clean_segment(segment)]

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

    async def run_without_gold(self, segments: Iterable[str]) -> SplitMergeResult:
        current = [clean_segment(s) for s in segments if clean_segment(s)]
        status = "unchanged"
        iterations = 0

        for iterations in range(1, self.max_iterations + 1):
            modified = False

            merged: List[str] = []
            idx = 0
            while idx < len(current):
                if idx + 1 >= len(current):
                    merged.append(current[idx])
                    idx += 1
                    continue
                result = await self.pairwise_merge_or_keep(current[idx], current[idx + 1])
                if len(result) == 1:
                    merged.extend(result)
                    idx += 2
                    modified = True
                    status = "merged"
                else:
                    merged.append(result[0])
                    current[idx + 1] = result[1]
                    idx += 1

            split: List[str] = []
            for segment in merged:
                result = await self.split_or_keep(segment)
                if len(result) > 1:
                    modified = True
                    status = "split"
                split.extend(result)

            current = split
            if not modified:
                status = "converged"
                break

        return SplitMergeResult(segments=current, answer="", iterations=iterations, status=status)

    async def refine(self, text: str, answer: str = "", golden_segments: Optional[int] = None) -> SplitMergeResult:
        segments = extract_segments(text)
        if not segments:
            return SplitMergeResult(segments=[], answer=answer, iterations=0, status="no_segments")
        if golden_segments is not None and golden_segments > 0:
            result = await self.run_with_gold(segments, golden_segments)
        else:
            result = await self.run_without_gold(segments)
        result.answer = answer
        return result


def build_prompt_from_row(row: Dict) -> str:
    if row.get("prompt"):
        return str(row["prompt"])
    question = row.get("question", "")
    references = row.get("references", row.get("documents", row.get("context", "")))
    if references:
        return f"References:\n{references}\n\nQuestion:\n{question}"
    return str(question)


def build_seed_text(row: Dict) -> str:
    for key in ("generated_text", "response", "reasoning", "thought", "cot"):
        if row.get(key):
            return str(row[key])
    return ""


async def generate_seed(client: AsyncOpenAIChatClient, prompt: str, max_tokens: int = 2048) -> str:
    return await client.chat(
        [
            {"role": "system", "content": STRUCTURED_REASONING_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        temperature=0.2,
        max_tokens=max_tokens,
    )


async def process_row(
    row: Dict,
    client: AsyncOpenAIChatClient,
    dataset: str,
    max_iterations: int,
    generate_missing: bool,
    seed_max_tokens: int,
) -> Dict:
    prompt = build_prompt_from_row(row)
    seed_text = build_seed_text(row)
    if not seed_text and generate_missing:
        seed_text = await generate_seed(client, prompt, max_tokens=seed_max_tokens)

    answer = extract_answer(seed_text, fallback=row.get("answer", row.get("ground_truth", "")))
    golden_segments = extract_golden_segments(row)
    if not has_golden_segments(golden_segments):
        golden_segments = None

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
        "answer": answer,
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
    generate_missing: bool = False,
    seed_max_tokens: int = 2048,
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
                generate_missing=generate_missing,
                seed_max_tokens=seed_max_tokens,
            )
            for row in chunk
        ]
        outputs.extend(await asyncio.gather(*tasks))
        print(f"processed {min(start + chunk_size, len(rows))}/{len(rows)}")
    return pd.DataFrame(outputs)
