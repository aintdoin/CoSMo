SEGMENT_RULES = """Every <seg> block must be one complete atomic reasoning unit.
An atomic unit may include retrieving evidence, citing it, and drawing the immediate factual deduction.
Do not fragment one continuous inference into multiple tiny segments.
Do not merge distinct logical actions that should remain independently inspectable.
Do not add facts that are absent from the input reasoning, references, or common knowledge needed by the question."""


STRUCTURED_REASONING_SYSTEM_PROMPT = """You are a rigorous reasoning engine. Solve the user query with compact, logically complete reasoning.

Output format is mandatory:
<think>
<seg>First complete logical inference.</seg>
<seg>Second complete logical inference.</seg>
...
</think>
<answer>Final answer only.</answer>

Rules:
1. Each reasoning segment must be enclosed in exactly one XML block: <seg>...</seg>.
2. Never put reasoning outside <think> or outside <seg> blocks.
3. The final answer must be enclosed in <answer>...</answer>.
4. Each <seg> is an atomic reasoning unit: combine evidence retrieval, citation, and immediate deduction when they are one inference.
5. Keep the number of segments aligned with the problem's real logical depth; avoid redundant restatements and avoid hidden logical jumps."""


MERGE_JUDGE_PROMPT = """Role & Objective
You are the CoSMo consistency judge. Decide whether two adjacent XML reasoning segments should be fused into one atomic segment.

Decision criteria:
- Return "merge" when Segment 2 is a paraphrase, a trivial extension, a detail that belongs to the same evidence-to-deduction act, or the pair forms one continuous inference.
- Return "keep" when the two segments perform distinct logical actions, introduce separate evidence, or represent different hops in the derivation.

{rules}

Output JSON only:
{{
  "decision": "merge" | "keep",
  "reason": "brief reason"
}}

Input:
<seg>{segment_1}</seg>
<seg>{segment_2}</seg>"""


MERGE_GENERATOR_PROMPT = """Role & Objective
You are the CoSMo semantic generator. Fuse two adjacent XML reasoning segments into exactly one logically complete segment.

Requirements:
- Preserve all necessary information from both inputs.
- Remove redundancy and paraphrase loops.
- Do not introduce new facts.
- Output exactly one XML segment and nothing else.

{rules}

Input:
<seg>{segment_1}</seg>
<seg>{segment_2}</seg>

Required output:
<seg>...</seg>"""


SPLIT_JUDGE_PROMPT = """Role & Objective
You are the CoSMo consistency judge. Decide whether one XML reasoning segment hides multiple distinct logical actions and should be decomposed.

Decision criteria:
- Return "split" when the segment contains at least two separable hops, such as retrieving one fact and then using it with another fact in a new deduction.
- Return "keep" when the segment is already one complete atomic inference, even if it is detailed.

{rules}

Output JSON only:
{{
  "decision": "split" | "keep",
  "reason": "brief reason"
}}

Input:
<seg>{segment}</seg>"""


SPLIT_GENERATOR_PROMPT = """Role & Objective
You are the CoSMo semantic generator. Decompose one coarse XML reasoning segment into exactly two XML segments.

Requirements:
- Step 1 must logically precede Step 2.
- The two output segments must be complete atomic reasoning units.
- Preserve the original meaning and do not add new facts.
- Output exactly two <seg>...</seg> blocks and nothing else.

{rules}

Input:
<seg>{segment}</seg>

Required output:
<seg>...</seg>
<seg>...</seg>"""


PAIRWISE_MERGE_PROMPT = """Role & Objective
You are the CoSMo split-merge editor. Analyze two adjacent XML reasoning segments and either merge them or keep them separate.

{rules}

Output JSON only:
{{
  "decision": "merge" | "keep",
  "segments": ["one segment if merged, otherwise refined segment 1", "refined segment 2 only when kept"]
}}

Input:
<seg>{segment_1}</seg>
<seg>{segment_2}</seg>"""


SINGLE_SPLIT_PROMPT = """Role & Objective
You are the CoSMo split-merge editor. Analyze one XML reasoning segment and either keep it atomic or split it into two atomic segments.

{rules}

Output JSON only:
{{
  "decision": "split" | "keep",
  "segments": ["original/refined segment if kept", "two segments if split"]
}}

Input:
<seg>{segment}</seg>"""


def render_prompt(template: str, **kwargs) -> str:
    return template.format(rules=SEGMENT_RULES, **kwargs)
