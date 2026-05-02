import json
import os
import re
import string
from typing import Any, Iterable, Optional

import torch

from cosmo.segments import ANSWER_PATTERN, SEGMENT_PATTERN
from verl import DataProto


GOLDEN_SEGMENT_KEYS = ("golden_segments", "gold_segments", "gold_hops", "hops", "hop", "num_hops")


def _normalize(text: Any) -> str:
    text = "" if text is None else str(text)
    text = text.lower().replace("_", " ")
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = "".join(" " if ch in string.punctuation else ch for ch in text)
    return " ".join(text.split())


def _as_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple, set)):
        return [str(v) for v in value if v is not None]
    if isinstance(value, str):
        stripped = value.strip()
        if not stripped:
            return []
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            return [stripped]
        if isinstance(parsed, (list, tuple)):
            return [str(v) for v in parsed if v is not None]
        return [str(parsed)]
    return [str(value)]


def _extract_answer(text: str) -> str:
    match = ANSWER_PATTERN.search(text or "")
    return match.group(1).strip() if match else ""


def _strip_xml_tags(text: str) -> str:
    return re.sub(r"<[^>]*>", " ", text or "", flags=re.DOTALL).strip()


def _extract_answer_for_grpo(text: str) -> str:
    tagged_answer = _extract_answer(text)
    if tagged_answer:
        return tagged_answer
    stripped = _strip_xml_tags(text)
    lines = [line.strip() for line in stripped.splitlines() if line.strip()]
    return lines[-1] if lines else " ".join(stripped.split())


def _xml_segment_count(text: str) -> int:
    think_match = re.search(r"<think>\s*(.*?)\s*</think>", text or "", re.IGNORECASE | re.DOTALL)
    content = think_match.group(1) if think_match else (text or "")
    return len([m for m in SEGMENT_PATTERN.finditer(content) if m.group(1).strip()])


def _valid_format(text: str) -> bool:
    if not isinstance(text, str):
        return False
    required_counts = {
        "<think>": len(re.findall(r"<think>", text, re.IGNORECASE)),
        "</think>": len(re.findall(r"</think>", text, re.IGNORECASE)),
        "<answer>": len(re.findall(r"<answer>", text, re.IGNORECASE)),
        "</answer>": len(re.findall(r"</answer>", text, re.IGNORECASE)),
    }
    if any(count != 1 for count in required_counts.values()):
        return False
    if not re.search(r"<think>.*</think>\s*<answer>.*</answer>", text, re.IGNORECASE | re.DOTALL):
        return False
    return _xml_segment_count(text) > 0


def _get_nested_dict(value: Any) -> dict:
    if isinstance(value, dict):
        return value
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _coerce_positive_int(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        if hasattr(value, "item"):
            value = value.item()
    except Exception:
        pass

    if isinstance(value, str):
        stripped = value.strip()
        if not stripped or stripped.lower() in {"nan", "none", "null"}:
            return None
        try:
            parsed = json.loads(stripped)
        except json.JSONDecodeError:
            value = stripped
        else:
            return _coerce_positive_int(parsed)

    if isinstance(value, dict):
        for key in GOLDEN_SEGMENT_KEYS:
            if key in value:
                return _coerce_positive_int(value[key])
        return None

    if isinstance(value, (list, tuple, set)):
        return len(value) if len(value) > 0 else None

    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return None
    return parsed if parsed > 0 else None


def _get_ground_truth(item) -> list[str]:
    nt = item.non_tensor_batch
    candidates: list[Any] = []
    for key in ("ground_truth", "answer", "answers", "target"):
        if key in nt:
            candidates.append(nt.get(key))

    reward_model = _get_nested_dict(nt.get("reward_model"))
    for key in ("ground_truth", "answer", "answers", "target"):
        if key in reward_model:
            candidates.append(reward_model.get(key))

    answers: list[str] = []
    for candidate in candidates:
        answers.extend(_as_list(candidate))
    return [a for a in answers if a]


def _get_golden_segments(item) -> Optional[int]:
    nt = item.non_tensor_batch
    extra_info = _get_nested_dict(nt.get("extra_info"))
    for source in (nt, extra_info):
        for key in GOLDEN_SEGMENT_KEYS:
            if key not in source:
                continue
            parsed = _coerce_positive_int(source.get(key))
            if parsed is not None:
                return parsed
    return None


def _answer_score(prediction: str, ground_truths: Iterable[str]) -> float:
    pred = _normalize(prediction)
    if not pred:
        return 0.0
    for gold in ground_truths:
        norm_gold = _normalize(gold)
        if norm_gold and (pred == norm_gold or norm_gold in pred.split() or norm_gold in pred):
            return 1.0
    return 0.0


class CoSMoRewardManager:
    """CoSMo reward, with answer-only GRPO fallback when no segment target exists."""

    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine
        self.penalty_scale = float(os.environ.get("COSMO_SEGMENT_PENALTY_SCALE", "1.0"))
        self.tolerance = int(os.environ.get("COSMO_SEGMENT_TOLERANCE", "0"))

    def __call__(self, data: DataProto):
        responses = data.batch["responses"]
        device = responses.device
        reward_tensor = torch.zeros_like(responses, dtype=torch.float32, device=device)
        format_reward_tensor = torch.zeros(responses.shape[0], dtype=torch.float32, device=device)
        answer_reward_tensor = torch.zeros(responses.shape[0], dtype=torch.float32, device=device)
        base_reward_tensor = torch.zeros(responses.shape[0], dtype=torch.float32, device=device)

        for i in range(len(data)):
            item = data[i]
            prompt_len = int(item.batch["prompts"].shape[-1])
            valid_resp_len = int(item.batch["attention_mask"][prompt_len:].sum().item())
            if valid_resp_len <= 0:
                continue

            resp_ids = item.batch["responses"][:valid_resp_len]
            response_text = self.tokenizer.decode(resp_ids, skip_special_tokens=True)

            golden_segments = _get_golden_segments(item)
            ground_truths = _get_ground_truth(item)

            if golden_segments is None:
                answer = _extract_answer_for_grpo(response_text)
                answer_score = _answer_score(answer, ground_truths)
                total = answer_score
                format_score = 0.0
                base_score = answer_score
            elif not _valid_format(response_text):
                total = -1.0
                format_score = -1.0
                answer_score = 0.0
                base_score = answer_score
            else:
                format_score = 0.0
                answer = _extract_answer(response_text)
                answer_score = _answer_score(answer, ground_truths)
                current_segments = _xml_segment_count(response_text)
                diff = max(0, abs(current_segments - golden_segments) - self.tolerance)
                structural_penalty = -self.penalty_scale * float(diff)
                total = format_score + answer_score + structural_penalty
                base_score = answer_score

            reward_tensor[i, valid_resp_len - 1] = float(total)
            format_reward_tensor[i] = float(format_score)
            answer_reward_tensor[i] = float(answer_score)
            base_reward_tensor[i] = float(base_score)

        return reward_tensor, format_reward_tensor, answer_reward_tensor, base_reward_tensor
