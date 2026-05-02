from cosmo.prompts import STRUCTURED_REASONING_SYSTEM_PROMPT


def get_system_prompt() -> str:
    return STRUCTURED_REASONING_SYSTEM_PROMPT


def wrap_prompt_with_system(user_content: str, model_template: str = "qwen") -> str:
    system_prompt = get_system_prompt()
    if model_template == "llama":
        return f"""<|begin_of_text|><|start_header_id|>system<|end_header_id|>

{system_prompt}<|eot_id|><|start_header_id|>user<|end_header_id|>

{user_content}<|eot_id|><|start_header_id|>assistant<|end_header_id|>

"""
    return f"""<|im_start|>system
{system_prompt}<|im_end|>
<|im_start|>user
{user_content}<|im_end|>
<|im_start|>assistant
"""
