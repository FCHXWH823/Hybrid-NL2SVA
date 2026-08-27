# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from context_pruner import LLMContextPruner, ContextResult
from utils_LLM import count_prompt_tokens
from config import FLAGS

# Calculate query embedding once
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_transformers import SentenceTransformer
from saver import saver
import networkx.algorithms.community as nx_comm

# from networkx.algorithms import isomorphism  # instead of vf2graph_matcher
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

print = saver.log_info


class DynamicPromptBuilder:
    def __init__(
        self,
        context_generators: Dict[str, object],
        pruning_config: Dict,
        llm_agent,
        context_summarizer=None,  # Add context_summarizer as optional parameter
    ):
        """
        Initialize the prompt builder with provided context generators.

        Args:
            context_generators: Dict of context generators by type
            pruning_config: Configuration for pruning
            llm_agent: Optional LLM agent for LLM-based pruning
            context_summarizer: Optional context summarizer for enhancement
        """
        self.context_generators = context_generators
        self.pruning_config = pruning_config
        self.llm_agent = llm_agent
        self.context_summarizer = context_summarizer  # Store the summarizer

        # Initialize LLM-based pruner if llm_agent is provided
        self.llm_pruner = LLMContextPruner(
            llm_agent=llm_agent,
            max_contexts_per_type=self.pruning_config.get('max_contexts_per_type', 3),
            max_total_contexts=self.pruning_config.get('max_total_contexts', 10),
        )
        print(f"Using LLM-based context pruning")

    def build_prompt(
        self,
        query: str,
        base_prompt: str,
        signal_name: Optional[str] = None,
        enable_context_enhancement: bool = False,  # Add flag for enhancement
    ) -> List[str]:
        """
        Build multiple dynamic prompts using available context generators,
        efficiently distributing contexts across prompts.

        Args:
            query: The query to generate context for
            base_prompt: The base prompt to build upon
            signal_name: Optional signal name for more targeted context generation
            enable_context_enhancement: Whether to enhance context with summarizer

        Returns:
            List[str]: A list of complete prompts with relevant context
        """
        # Check max prompts per signal configuration
        max_prompts_per_signal = FLAGS.max_prompts_per_signal
        max_tokens_per_prompt = FLAGS.max_tokens_per_prompt

        # Calculate enhancement overhead if it will be used
        enhancement_token_overhead = 0
        if enable_context_enhancement and self.context_summarizer and signal_name:
            # Create a simple placeholder dynamic context to estimate the enhancement
            placeholder_context = "This is a placeholder for dynamic context."

            # Calculate the token overhead by measuring before and after enhancement
            before_tokens = count_prompt_tokens(placeholder_context)
            enhanced_context = self.context_summarizer.add_enhanced_context(
                placeholder_context, signal_name
            )
            after_tokens = count_prompt_tokens(enhanced_context)

            # The overhead is the difference between enhanced and original
            enhancement_token_overhead = after_tokens - before_tokens
            print(
                f"Calculated enhancement overhead: {enhancement_token_overhead} tokens"
            )

        # Reserve tokens for LLM response (roughly 25% of max tokens) and for enhancement
        max_context_tokens = (
            int(max_tokens_per_prompt * 0.75) - enhancement_token_overhead
        )

        print(
            f"Building up to {max_prompts_per_signal} prompts with max {max_context_tokens} tokens each"
        )

        all_contexts = []

        # First get relevant nodes if KG retrieval is enabled
        start_nodes = []
        if 'kg_node_retriever' in self.context_generators:
            start_nodes = self.context_generators['kg_node_retriever'].retrieve(
                query, k=FLAGS.dynamic_prompt_settings['rag']['k']
            )
            if signal_name and signal_name not in start_nodes:
                start_nodes.append(signal_name)  # Ensure signal_name is included
            print(f'kg_node_retriever in there and start_nodes={start_nodes}')
        elif signal_name:
            start_nodes = [signal_name]
            print(f'start_nodes={start_nodes}')

        # Get contexts from each enabled generator
        for generator_type, generator in self.context_generators.items():
            print(f"\nProcessing generator: {generator_type}")

            if generator_type == 'rag':
                # RAG doesn't need start_nodes
                contexts = generator.get_contexts(
                    query, k=FLAGS.dynamic_prompt_settings['rag']['k']
                )
                print(f"RAG generator returned {len(contexts)} contexts")

            elif generator_type in [
                'path',
                'motif',
                'community',
                'local_expansion',
                'guided_random_walk',
            ]:
                # These generators need start_nodes
                if start_nodes:
                    if generator_type == 'path':
                        contexts = generator.get_contexts(
                            start_nodes,
                        )
                    else:  # motif or community
                        contexts = generator.get_contexts(start_nodes)

                    print(
                        f"{generator_type} generator returned {len(contexts)} contexts"
                    )
                else:
                    print(
                        f"Warning: No start nodes available for {generator_type} generator"
                    )
                    contexts = []

            elif generator_type == 'kg_node_retriever':
                # Skip the node retriever as it's used for getting start_nodes
                print("Skipping kg_node_retriever as it's used for start nodes")
                contexts = []
                continue

            else:
                print(f"Warning: Unknown generator type {generator_type}")
                contexts = []

            if contexts:
                all_contexts.extend(contexts)
                print(
                    f"Added {len(contexts)} contexts from {generator_type}, total now: {len(all_contexts)}"
                )

                # Print a sample context from each type
                if contexts:
                    sample = contexts[0]
                    print(f"Sample {generator_type} context: {sample.text}")

        # Prune contexts if enabled
        if self.pruning_config['enabled']:
            print(f"\nPruning {len(all_contexts)} contexts...")

            # Use LLM-based pruning if available, otherwise fall back to similarity-based pruning
            if self.llm_pruner and self.llm_agent:
                print(f"Using LLM-based context pruning")
                selected_contexts = self.llm_pruner.prune(
                    all_contexts, query, signal_name
                )
            else:
                assert False  # We should never reach here as per the existing code

            print(f"After pruning: {len(selected_contexts)} contexts remain")

            # Count by type
            type_counts = {}
            for context in selected_contexts:
                if context.source_type not in type_counts:
                    type_counts[context.source_type] = 0
                type_counts[context.source_type] += 1
            print(f"Selected contexts by type: {type_counts}")
        else:
            selected_contexts = all_contexts
            print(f"Pruning disabled, using all {len(selected_contexts)} contexts")

        # Group contexts by type for even distribution
        contexts_by_type = {
            'rag': [],
            'path': [],
            'motif': [],
            'community': [],
            'local_expansion': [],
            'guided_random_walk': [],
        }

        for context in selected_contexts:
            contexts_by_type[context.source_type].append(context)

        # Calculate how many contexts of each type should go in each prompt
        contexts_per_prompt_by_type = {}
        for context_type, contexts in contexts_by_type.items():
            if len(contexts) == 0:
                contexts_per_prompt_by_type[context_type] = [0] * max_prompts_per_signal
                continue

            # Calculate base distribution (minimum each prompt gets)
            base_count = len(contexts) // max_prompts_per_signal
            # Calculate remaining contexts after base distribution
            remainder = len(contexts) % max_prompts_per_signal

            # Distribute the remainder one by one
            distribution = [base_count] * max_prompts_per_signal
            for i in range(remainder):
                distribution[i] += 1

            contexts_per_prompt_by_type[context_type] = distribution

        print(f"Context distribution plan: {contexts_per_prompt_by_type}")

        # Create the base for each prompt
        base_parts = [base_prompt, "\nRelevant Context:"]
        if signal_name:
            base_parts.append(f"\nContexts relevant to signal '{signal_name}':")

        base_prompt_text = "\n".join(base_parts)
        base_token_count = count_prompt_tokens(base_prompt_text)
        print(f"Base prompt token count: {base_token_count}")

        # Prepare the final prompts
        final_prompts = []
        prompt_token_counts = [base_token_count] * max_prompts_per_signal
        prompt_parts = [base_parts.copy() for _ in range(max_prompts_per_signal)]
        contexts_added = [0] * max_prompts_per_signal

        # Function to format a context with metadata
        def format_context(context, context_type):
            if not context or not context.text:
                return ""

            # Format the main content
            formatted_parts = [context.text]

            # Add relevant metadata based on context type
            if context.metadata:
                metadata_parts = []

                if context_type == 'path':
                    relevant_keys = ['path_length', 'start_node']
                elif context_type == 'motif':
                    relevant_keys = ['type', 'start_node', 'metrics']
                elif context_type == 'community':
                    relevant_keys = ['size', 'start_node', 'metrics']
                elif context_type == 'local_expansion':
                    relevant_keys = ['size', 'start_node', 'metrics']
                elif context_type == 'guided_random_walk':
                    relevant_keys = ['size', 'start_node', 'metrics']
                else:  # rag
                    relevant_keys = []

                for key in relevant_keys:
                    if key in context.metadata:
                        value = context.metadata[key]
                        if key == 'metrics':
                            # Format metrics more concisely
                            metrics_str = ', '.join(
                                (f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}")
                                for k, v in value.items()
                            )
                            metadata_parts.append(f"{key}: {metrics_str}")
                        else:
                            metadata_parts.append(f"{key}: {value}")

                if metadata_parts:
                    formatted_parts.append(f"({'; '.join(metadata_parts)})")

            formatted_parts.append("")  # Add blank line between contexts
            return "\n".join(formatted_parts)

        # Function to estimate token count for a context
        def get_context_token_count(context, context_type):
            formatted_text = format_context(context, context_type)
            return count_prompt_tokens(formatted_text)

        # Step 2: Group formatted contexts by type first, then add to prompt
        for prompt_idx in range(max_prompts_per_signal):
            prompt_contexts_by_type = {
                'rag': [],
                'path': [],
                'motif': [],
                'community': [],
                'local_expansion': [],
                'guided_random_walk': [],
            }

            # Process and format all contexts, gathering them by type
            for context_type in [
                'rag',
                'path',
                'motif',
                'community',
                'local_expansion',
                'guided_random_walk',
            ]:
                contexts_of_type = contexts_by_type[context_type].copy()
                target_count = contexts_per_prompt_by_type[context_type][prompt_idx]
                added_to_prompt = 0

                while added_to_prompt < target_count and contexts_of_type:
                    context = contexts_of_type.pop(0)
                    formatted_context = format_context(context, context_type)
                    context_token_count = count_prompt_tokens(formatted_context)

                    if (
                        prompt_token_counts[prompt_idx] + context_token_count
                        <= max_context_tokens
                    ):
                        prompt_contexts_by_type[context_type].append(formatted_context)
                        prompt_token_counts[prompt_idx] += context_token_count
                        contexts_added[prompt_idx] += 1
                        added_to_prompt += 1
                    else:

                        # If this prompt is full, try to add to another prompt
                        alternative_idx = None
                        for alt_idx in range(max_prompts_per_signal):
                            if (
                                alt_idx != prompt_idx
                                and prompt_token_counts[alt_idx] + context_token_count
                                <= max_context_tokens
                            ):
                                alternative_idx = alt_idx
                                break

                        if alternative_idx is not None:
                            formatted_context = format_context(context, context_type)
                            prompt_parts[alternative_idx].append(formatted_context)
                            prompt_token_counts[alternative_idx] += context_token_count
                            contexts_added[alternative_idx] += 1
                            # Adjust distribution counts to reflect reality
                            contexts_per_prompt_by_type[context_type][prompt_idx] -= 1
                            contexts_per_prompt_by_type[context_type][
                                alternative_idx
                            ] += 1
                        else:
                            # If all prompts are too full, we have to skip this context
                            print(
                                f"Warning: Skipping context due to token limits. Type: {context_type}"
                            )
                            contexts_per_prompt_by_type[context_type][prompt_idx] -= 1

                # If we ran out of contexts, adjust the expected counts
                if not contexts_of_type and added_to_prompt < target_count:
                    contexts_per_prompt_by_type[context_type][
                        prompt_idx
                    ] = added_to_prompt

            # Step 3: Now add the type headers and their contexts in order
            for context_type in [
                'rag',
                'path',
                'motif',
                'community',
                'local_expansion',
                'guided_random_walk',
            ]:
                if prompt_contexts_by_type[context_type]:
                    # Only add section header if we have contexts of this type
                    type_header = f"\n{context_type.upper()} Context:"
                    prompt_parts[prompt_idx].append(type_header)

                    # Add all formatted contexts of this type
                    prompt_parts[prompt_idx].extend(
                        prompt_contexts_by_type[context_type]
                    )

        # Build final prompts from prompt parts
        for prompt_idx in range(max_prompts_per_signal):
            # Only include prompts that have contexts
            if contexts_added[prompt_idx] > 0:
                final_prompt = "\n".join(prompt_parts[prompt_idx])
                final_prompts.append(final_prompt)
                print(
                    f"Prompt {prompt_idx+1}: {contexts_added[prompt_idx]} contexts, {prompt_token_counts[prompt_idx]} tokens"
                )

        # If no prompts were created (no contexts), return the base prompt
        if not final_prompts:
            print("No contexts available, returning base prompt only")
            final_prompts = [base_prompt_text]

        print(
            f"Created {len(final_prompts)} final prompts out of maximum {max_prompts_per_signal}"
        )

        if enable_context_enhancement and self.context_summarizer and signal_name:
            enhanced_final_prompts = []
            for prompt in final_prompts:
                enhanced_prompt = self.context_summarizer.add_enhanced_context(
                    prompt, signal_name
                )
                enhanced_final_prompts.append(enhanced_prompt)
            return enhanced_final_prompts
        else:
            return final_prompts

    def _prune_contexts_similarity(
        self,
        contexts: List[ContextResult],
        query: str,
        max_per_type: int = 2,
        min_similarity: float = 0.3,
    ) -> List[ContextResult]:
        """
        Legacy similarity-based pruning method.
        Used as fallback when LLM pruning is not enabled.

        Args:
            contexts: List of contexts to prune
            query: The original query for relevance comparison
            max_per_type: Maximum number of contexts to keep per type
            min_similarity: Minimum similarity threshold for contexts

        Returns:
            List[ContextResult]: Pruned list of contexts
        """
        selected = []

        try:
            print(
                f"  Pruning contexts with max_per_type={max_per_type}, min_similarity={min_similarity}"
            )

            model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
            query_embedding = model.encode([query])[0]

            # Group contexts by type
            for context_type in [
                'rag',
                'path',
                'motif',
                'community',
                'local_expansion',
                'guided_random_walk',
            ]:
                type_contexts = [c for c in contexts if c.source_type == context_type]
                if not type_contexts:
                    print(f"  No {context_type} contexts to prune")
                    continue

                print(f"  Pruning {len(type_contexts)} {context_type} contexts")

                # Calculate similarities and filter by threshold
                scored_contexts = []
                for context in type_contexts:
                    context_embedding = model.encode([context.text])[0]
                    similarity = cosine_similarity(
                        [query_embedding], [context_embedding]
                    )[0][0]
                    print(
                        f"    Context similarity: {similarity:.4f} (threshold: {min_similarity})"
                    )

                    if similarity >= min_similarity:
                        # Combine similarity with original score
                        final_score = (similarity + context.score) / 2
                        scored_contexts.append((context, final_score))
                    else:
                        print(
                            f"    Context filtered out due to low similarity: {similarity:.4f} < {min_similarity}"
                        )

                # Sort by final score and select top contexts
                scored_contexts.sort(key=lambda x: x[1], reverse=True)
                selected_for_type = [
                    context for context, _ in scored_contexts[:max_per_type]
                ]
                selected.extend(selected_for_type)
                print(
                    f"  Selected {len(selected_for_type)}/{len(type_contexts)} {context_type} contexts"
                )

        except Exception as e:
            print(f"Error during context pruning: {str(e)}")
            traceback.print_exc()
            # If pruning fails, return all contexts
            selected = contexts
            print(f"Pruning failed, returning all {len(contexts)} contexts")

        return selected
