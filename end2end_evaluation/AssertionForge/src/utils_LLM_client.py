# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

# Filled in for the Hybrid-NL2SVA UART pilot -- upstream ships this file as an
# empty "Add your own LLM client" stub. Minimal OpenAI-backed implementation,
# reading the same Src/Config.yml key the rest of Hybrid-NL2SVA uses.
import os
import yaml
from openai import OpenAI

_HYBRID_NL2SVA_CONFIG = os.environ.get(
    "HYBRID_NL2SVA_CONFIG", "/home/wx2356/Hybrid-NL2SVA/Src/Config.yml"
)


class LLMAgent:
    """Opaque handle passed around by gen_plan.py/context_pruner.py/
    design_context_summarizer.py -- just bundles a client + model name."""

    def __init__(self, client, model_name):
        self.client = client
        self.model_name = model_name


def get_llm(model_name, **llm_args):
    with open(_HYBRID_NL2SVA_CONFIG) as f:
        config = yaml.safe_load(f)
    client = OpenAI(api_key=config["Openai_API_Key"])
    return LLMAgent(client, model_name)
