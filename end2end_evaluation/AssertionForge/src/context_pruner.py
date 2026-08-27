# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

"""
Simplified tolerant LLM-based Context Pruner for hardware verification prompt generation.
"""
from utils_LLM import llm_inference
from typing import List, Dict
from dataclasses import dataclass
import time
import re
from saver import saver

# Use saver's logging mechanism
print = saver.log_info


# Import the ContextResult class
@dataclass
class ContextResult:
    """Class for storing context generation results"""

    text: str
    score: float
    source_type: str  # e.g., 'rag', 'path', 'motif', 'community'
    metadata: Dict


class LLMContextPruner:
    """
    Uses an LLM to evaluate and prune context candidates for hardware verification prompts.
    Simplified tolerant version.
    """

    def __init__(self, llm_agent, max_contexts_per_type=3, max_total_contexts=10):
        """
        Initialize the LLM-based context pruner.

        Args:
            llm_agent: The LLM inference agent to use for pruning
            max_contexts_per_type: Maximum number of contexts to keep per type
            max_total_contexts: Maximum total contexts to keep overall
        """
        self.llm_agent = llm_agent
        self.max_contexts_per_type = max_contexts_per_type
        self.max_total_contexts = max_total_contexts
        self.min_contexts_per_type = (
            2  # Minimum contexts to select per type if available
        )

    def prune(
        self, contexts: List[ContextResult], query: str, signal_name: str = None
    ) -> List[ContextResult]:
        """
        Use LLM to evaluate and select the most relevant contexts.

        Args:
            contexts: List of context candidates to evaluate
            query: The original verification query
            signal_name: Optional signal name to focus on

        Returns:
            List[ContextResult]: The pruned list of contexts
        """
        if not contexts:
            print(f"No contexts to prune")
            return []

        print(f"Pruning {len(contexts)} contexts using tolerant LLM pruner")

        # Group contexts by type
        contexts_by_type = {}
        for context in contexts:
            if context.source_type not in contexts_by_type:
                contexts_by_type[context.source_type] = []
            contexts_by_type[context.source_type].append(context)

        # For each type, ask LLM to select the best contexts
        selected_contexts = []

        # Process each context type
        for context_type, type_contexts in contexts_by_type.items():
            if not type_contexts:
                continue

            print(f"Evaluating {len(type_contexts)} contexts of type '{context_type}'")

            # If too many contexts, pre-filter by score
            max_eval_contexts = 20  # Maximum to evaluate with LLM
            if len(type_contexts) > max_eval_contexts:
                print(
                    f"Pre-filtering {len(type_contexts)} contexts of type '{context_type}' based on score"
                )
                type_contexts = sorted(
                    type_contexts, key=lambda x: x.score, reverse=True
                )[:max_eval_contexts]

            # Create evaluation prompt
            prompt = self._create_tolerant_prompt(
                type_contexts, query, signal_name, context_type
            )

            try:
                # Call LLM
                start_time = time.time()
                result = self._call_llm(prompt, f"context_eval_{context_type}")
                duration = time.time() - start_time
                print(f"LLM evaluation for {context_type} took {duration:.2f} seconds")

                # Parse results
                selected_indices = self._parse_llm_response(result, len(type_contexts))

                # If no contexts selected, use top scoring ones
                if not selected_indices and type_contexts:
                    print(
                        f"No contexts selected by LLM for {context_type}, using top scoring contexts"
                    )
                    sorted_indices = sorted(
                        range(len(type_contexts)),
                        key=lambda i: type_contexts[i].score,
                        reverse=True,
                    )
                    min_count = min(self.min_contexts_per_type, len(type_contexts))
                    selected_indices = sorted_indices[:min_count]

                # Add selected contexts
                for idx in selected_indices:
                    if idx < len(type_contexts):
                        selected_contexts.append(type_contexts[idx])
                        print(f"Selected context {idx} of type {context_type}")

            except Exception as e:
                print(f"Error during LLM pruning for {context_type}: {str(e)}")
                # Fall back to top scoring contexts
                sorted_contexts = sorted(
                    type_contexts, key=lambda x: x.score, reverse=True
                )
                min_count = min(self.min_contexts_per_type, len(type_contexts))
                selected_for_type = sorted_contexts[:min_count]
                selected_contexts.extend(selected_for_type)
                print(
                    f"Fallback: Selected top {len(selected_for_type)} contexts by score for {context_type}"
                )

        # If we have too many contexts overall, perform final selection
        if len(selected_contexts) > self.max_total_contexts:
            print(
                f"Too many contexts overall ({len(selected_contexts)}), selecting balanced subset"
            )
            selected_contexts = self._select_balanced_subset(selected_contexts)

        print(f"Final selection: {len(selected_contexts)} contexts")
        return selected_contexts

    def _select_balanced_subset(
        self, contexts: List[ContextResult]
    ) -> List[ContextResult]:
        """Select a balanced subset of contexts across different types"""
        by_type = {}
        for ctx in contexts:
            if ctx.source_type not in by_type:
                by_type[ctx.source_type] = []
            by_type[ctx.source_type].append(ctx)

        # Get contexts from each type
        balanced = []
        types = list(by_type.keys())
        min_per_type = max(1, self.max_total_contexts // len(types))

        for type_name in types:
            sorted_contexts = sorted(
                by_type[type_name], key=lambda x: x.score, reverse=True
            )
            balanced.extend(sorted_contexts[:min_per_type])

        # If we still have room, add more from the highest scoring
        if len(balanced) < self.max_total_contexts:
            remaining = [ctx for ctx in contexts if ctx not in balanced]
            remaining = sorted(remaining, key=lambda x: x.score, reverse=True)
            balanced.extend(remaining[: self.max_total_contexts - len(balanced)])

        print(f"Balanced selection: {len(balanced)} contexts across {len(types)} types")
        return balanced[: self.max_total_contexts]

    def _create_tolerant_prompt(
        self,
        contexts: List[ContextResult],
        query: str,
        signal_name: str,
        context_type: str,
    ) -> str:
        """Create a tolerant prompt for context evaluation"""

        signal_info = f" for signal '{signal_name}'" if signal_name else ""
        min_selection = min(self.min_contexts_per_type, len(contexts))
        max_selection = min(self.max_contexts_per_type, len(contexts))

        prompt = f"""You are an expert hardware verification engineer evaluating contexts to be used in generating verification plans{signal_info}.

QUERY: {query}

YOUR TASK: Select between {min_selection} and {max_selection} contexts of type '{context_type}' that could help with verification.

IMPORTANT NOTES:
- Select at least {min_selection} contexts even if they seem only indirectly relevant
- Consider both explicit mentions of '{signal_name}' and general system information
- Partial information about interfaces, protocols, and system behavior is still valuable
- When in doubt, include rather than exclude contexts

CONTEXTS TO EVALUATE:
"""

        for i, context in enumerate(contexts):
            # Truncate very long contexts
            context_text = context.text
            if len(context_text) > 800:
                context_text = context_text[:750] + "... [truncated]"

            prompt += f"\n[CONTEXT {i}]\n{context_text}\n"

            # Add relevant metadata
            if context.metadata:
                metadata_str = ", ".join(
                    f"{k}: {v}"
                    for k, v in context.metadata.items()
                    if k in ['start_node', 'path_length', 'type', 'size']
                )
                if metadata_str:
                    prompt += f"[Metadata: {metadata_str}]\n"

            prompt += "----\n"

        prompt += f"""
SELECTION INSTRUCTIONS:
1. SELECT AT LEAST {min_selection} CONTEXTS, even if only partially relevant
2. Focus on contexts that might help verify {signal_name if signal_name else "the system"}
3. Output your selection using ONLY the format "Selected contexts: [list of indices]" (e.g., "Selected contexts: [0, 2, 5]")

For hardware verification, we need information about:
- Signal connections and dependencies
- Timing requirements
- Protocol details
- State transitions
- Interfaces
"""

        return prompt

    def _call_llm(self, prompt: str, tag: str) -> str:
        """Call the LLM with the given prompt"""
        # Import here to avoid circular imports

        return llm_inference(self.llm_agent, prompt, tag)

    def _parse_llm_response(self, response: str, max_index: int) -> List[int]:
        """Parse the LLM response to extract selected context indices"""
        selected_indices = []

        try:
            # Extract indices from "Selected contexts: [0, 2, 5]" format
            match = re.search(r"Selected contexts:\s*\[(.*?)\]", response)
            if match:
                indices_str = match.group(1)
                if indices_str.strip():
                    selected_indices = [
                        int(idx.strip()) for idx in indices_str.split(",")
                    ]
                    # Filter out invalid indices
                    selected_indices = [
                        idx for idx in selected_indices if 0 <= idx < max_index
                    ]

            # Warning if nothing parsed
            if not selected_indices and response.strip():
                print(
                    f"Failed to parse context indices from response: {response[:100]}..."
                )

        except Exception as e:
            print(f"Error parsing LLM response: {str(e)}")

        return selected_indices
