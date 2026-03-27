# Copyright 2024 PRIME team and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import re
import torch

from verl import DataProto


def _extract(text: str) -> str:
    if not isinstance(text, str):
        return ""
    # Strip any XML-like tags without relying on tag names.
    return re.sub(r"<[^>]*>", "", text, flags=re.DOTALL).strip()


def _norm(text: str) -> str:
    if not isinstance(text, str):
        return ""
    return re.sub(r"\s+", " ", text.strip().lower())


class PrimeRewardManager:
    def __init__(self, tokenizer, num_examine) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine

    def __call__(self, data: DataProto):
        # Keep the same return signature as NaiveRewardManager for trainer compatibility.
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        format_reward_tensor = torch.zeros(data.batch["responses"].shape[0], dtype=torch.float32)
        answer_reward_tensor = torch.zeros(data.batch["responses"].shape[0], dtype=torch.float32)
        base_reward_tensor = torch.zeros(data.batch["responses"].shape[0], dtype=torch.float32)

        for i in range(len(data)):
            item = data[i]
            prompt_len = item.batch["prompts"].shape[-1]
            valid_resp_len = item.batch["attention_mask"][prompt_len:].sum()
            resp_ids = item.batch["responses"][:valid_resp_len]

            resp_text = self.tokenizer.decode(resp_ids, skip_special_tokens=False)
            pred = _norm(_extract(resp_text))
            gt = _norm(item.non_tensor_batch.get("ground_truth", ""))

            score = 1.0 if gt and pred == gt else 0.0
            reward_tensor[i, valid_resp_len - 1] = score
            answer_reward_tensor[i] = score
            base_reward_tensor[i] = score

        return reward_tensor, format_reward_tensor, answer_reward_tensor, base_reward_tensor
