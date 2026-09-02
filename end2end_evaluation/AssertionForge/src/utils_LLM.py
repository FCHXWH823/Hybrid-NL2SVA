# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Filled in for the Hybrid-NL2SVA UART pilot -- upstream ships this file as an
# empty "Add your own LLM client" stub. Callers (gen_plan.py, context_pruner.py,
# design_context_summarizer.py) expect llm_inference(llm_agent, prompt, tag) to
# return a plain string (they call e.g. result.split('\n') directly on it) and
# get_llm(model_name, **llm_args) to return an opaque agent object -- see
# utils_LLM_client.py for that half.
import tiktoken
from saver import saver
from utils_LLM_client import get_llm  # re-exported: gen_plan.py imports both from here

print = saver.log_info

_ENCODING = tiktoken.get_encoding("cl100k_base")


def count_prompt_tokens(text):
    return len(_ENCODING.encode(text))


def llm_inference(llm_agent, prompt, tag="", system_prompt=None):
    """system_prompt (2026-09-02, added): optional, defaults to None (every
    pre-existing call site -- context_pruner.py, design_context_summarizer.py,
    most of gen_plan.py -- is unaffected, still gets a single user-role
    message exactly as before). gen_plan.py's plan-generation call sites pass
    one now, mirroring how Hybrid-NL2SVA's own Stage 1 OL-NL grounding
    (generate_ol_nl_grounding, run_rag_on_fveval_benchmarks.py) puts general
    instructions/constraints/reference material (its SVA operator table
    included) in an actual system message rather than folding everything
    into one big user message -- this file's own llm_inference had no system-
    message concept at all until now, since the original empty stub gave no
    guidance on it and the call-site contract this was written against
    (gen_plan.py's OTHER callers) never needed one."""
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": prompt})
    completion = llm_agent.client.chat.completions.create(
        model=llm_agent.model_name,
        messages=messages,
    )
    result = completion.choices[0].message.content
    prompt_len = len(prompt) + (len(system_prompt) if system_prompt else 0)
    print(f"[llm_inference:{tag}] prompt={prompt_len} chars -> response={len(result)} chars")
    return result
