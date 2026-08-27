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


def llm_inference(llm_agent, prompt, tag=""):
    completion = llm_agent.client.chat.completions.create(
        model=llm_agent.model_name,
        messages=[{"role": "user", "content": prompt}],
    )
    result = completion.choices[0].message.content
    print(f"[llm_inference:{tag}] prompt={len(prompt)} chars -> response={len(result)} chars")
    return result
