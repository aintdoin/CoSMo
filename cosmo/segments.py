import re
from typing import Iterable, List, Optional


SEGMENT_PATTERN = re.compile(r"<seg>\s*(.*?)\s*</seg>", re.IGNORECASE | re.DOTALL)
THINK_PATTERN = re.compile(r"<think>\s*(.*?)\s*</think>", re.IGNORECASE | re.DOTALL)
ANSWER_PATTERN = re.compile(r"<answer>\s*(.*?)\s*</answer>", re.IGNORECASE | re.DOTALL)
BOXED_PATTERN = re.compile(r"\\boxed\{(.*?)\}", re.DOTALL)


def clean_segment(text: str) -> str:
    text = str(text or "").strip()
    text = re.sub(r"^\s*(?:[-*]|\d+[.)])\s*", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_segments(text: str) -> List[str]:
    """Extract reasoning segments from XML blocks, with numbered-list fallback."""
    if not isinstance(text, str) or not text.strip():
        return []

    think_match = THINK_PATTERN.search(text)
    source = think_match.group(1) if think_match else text

    xml_segments = [clean_segment(s) for s in SEGMENT_PATTERN.findall(source)]
    xml_segments = [s for s in xml_segments if s]
    if xml_segments:
        return xml_segments

    numbered = re.split(r"\n(?=\s*\d+[.)]\s+)", source.strip())
    parsed = [clean_segment(s) for s in numbered if clean_segment(s)]
    if len(parsed) > 1:
        return parsed

    lines = [clean_segment(s) for s in source.splitlines()]
    return [s for s in lines if s]


def count_segments(text: str) -> int:
    return len(extract_segments(text))


def extract_answer(text: str, fallback: Optional[str] = None) -> str:
    if isinstance(text, str):
        answer_match = ANSWER_PATTERN.search(text)
        if answer_match:
            return answer_match.group(1).strip()
        boxed_match = BOXED_PATTERN.search(text)
        if boxed_match:
            return boxed_match.group(1).strip()
    return "" if fallback is None else str(fallback).strip()


def format_segments(segments: Iterable[str]) -> str:
    return "\n".join(f"<seg>{clean_segment(segment)}</seg>" for segment in segments if clean_segment(segment))


def format_response(segments: Iterable[str], answer: str) -> str:
    return f"<think>\n{format_segments(segments)}\n</think>\n<answer>{str(answer).strip()}</answer>"
