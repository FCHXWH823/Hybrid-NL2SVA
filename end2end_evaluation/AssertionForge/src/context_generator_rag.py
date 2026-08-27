# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import List, Dict, Set, Tuple, Optional, DefaultDict
from context_pruner import ContextResult
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from saver import saver

from config import FLAGS
from saver import saver

print = saver.log_info


class RAGContextGenerator:
    """Enhanced RAG context generator with configurable parameters that supports both specification text and RTL code"""

    def __init__(
        self,
        spec_text: str,
        rtl_code: Optional[str] = None,
        chunk_sizes: List[int] = None,
        overlap_ratios: List[float] = None,
    ):
        """
        Initialize RAG context generator with configurable chunking parameters.

        Args:
            spec_text: Specification text to be processed
            rtl_code: RTL code to be processed (optional)
            chunk_sizes: List of chunk sizes to use
            overlap_ratios: List of overlap ratios to use
        """
        self.spec_text = spec_text
        self.rtl_code = rtl_code

        # Use provided parameters or defaults, mimicking original format
        self.chunk_sizes = chunk_sizes or [50, 100, 200]
        self.overlap_ratios = overlap_ratios or [0.2, 0.4]
        self.k = 3  # Default chunks per configuration

        # Get RTL-specific settings from FLAGS
        self.enable_rtl_rag = FLAGS.dynamic_prompt_settings['rag']['enable_rtl']
        self.baseline_full_spec_RTL = FLAGS.dynamic_prompt_settings['rag'].get(
            'baseline_full_spec_RTL', False
        )

        # Validate settings
        if self.baseline_full_spec_RTL:
            # Ensure enable_rtl is True if baseline_full_spec_RTL is True
            assert (
                self.enable_rtl_rag
            ), 'enable_rtl must be True when baseline_full_spec_RTL is True'
            assert (
                rtl_code is not None
            ), 'RTL code must be provided when baseline_full_spec_RTL is True'

        if self.enable_rtl_rag:
            assert rtl_code is not None, 'Should send in RTL code'

        # Initialize spec text retrievers
        self.spec_retrievers = []
        if not self.baseline_full_spec_RTL:
            for chunk_size in self.chunk_sizes:
                for overlap_ratio in self.overlap_ratios:
                    overlap = int(chunk_size * overlap_ratio)
                    retriever = DocRetriever(
                        spec_text,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        source_type='spec',
                    )
                    self.spec_retrievers.append(retriever)

        # Initialize RTL code retrievers if RTL code is provided (only if not using baseline)
        self.rtl_retrievers = []
        if rtl_code and self.enable_rtl_rag and not self.baseline_full_spec_RTL:
            # Use the same chunk sizes and overlap ratios for RTL
            for chunk_size in self.chunk_sizes:
                for overlap_ratio in self.overlap_ratios:
                    overlap = int(chunk_size * overlap_ratio)
                    retriever = DocRetriever(
                        rtl_code,
                        chunk_size=chunk_size,
                        overlap=overlap,
                        source_type='rtl',
                    )
                    self.rtl_retrievers.append(retriever)

    def get_contexts(self, query: str, k: int = None) -> List[ContextResult]:
        """
        Get contexts using multiple retrievers with different configurations.

        Args:
            query: Search query
            k: Number of contexts to retrieve per configuration (overrides default)

        Returns:
            List of ContextResults
        """
        # If using the baseline approach, simply return the concatenated text
        if self.baseline_full_spec_RTL and self.rtl_code:
            # Create a single context with the concatenated spec text and RTL code
            combined_text = f"{self.spec_text}\n\n{self.rtl_code}"
            print('baseline_full_spec_RTL! simple concat')
            return [
                ContextResult(
                    text=combined_text,
                    score=1.0,  # Maximum score
                    source_type='rag',
                    metadata={
                        'chunk_size': len(combined_text.split()),
                        'overlap': 0,
                        'rank': 1,
                        'content_type': 'baseline_full_spec_RTL',
                    },
                )
            ]

        if k is None:
            k = self.k

        contexts = []

        # Get contexts from specification text
        for retriever in self.spec_retrievers:
            try:
                chunks = retriever.retrieve(query, k=k)
                for i, chunk in enumerate(chunks):
                    # Score based on rank and chunk properties
                    rank_score = 1.0 / (i + 1)
                    size_score = 1.0 / (
                        abs(len(chunk.split()) - 100) + 1
                    )  # Peak at ~100 words
                    score = (rank_score + size_score) / 2

                    contexts.append(
                        ContextResult(
                            text=chunk,
                            score=score,
                            source_type='rag',
                            metadata={
                                'chunk_size': retriever.chunk_size,
                                'overlap': retriever.overlap,
                                'rank': i + 1,
                                'content_type': 'spec',
                            },
                        )
                    )
            except Exception as e:
                print(f"Warning: Retriever failed for query '{query}' on spec: {e}")
                continue

        # Get contexts from RTL code if available
        if self.rtl_code and self.enable_rtl_rag:
            for retriever in self.rtl_retrievers:
                try:
                    chunks = retriever.retrieve(query, k=k)
                    for i, chunk in enumerate(chunks):
                        # Score based on rank and chunk properties
                        rank_score = 1.0 / (i + 1)
                        # Use a different ideal size for code chunks
                        size_score = 1.0 / (
                            abs(len(chunk.split()) - 50) + 1
                        )  # Peak at ~50 words for code
                        # Give a slight bonus to RTL chunks (1.1x)
                        score = (rank_score + size_score) / 2 * 1.1

                        contexts.append(
                            ContextResult(
                                text=chunk,
                                score=score,
                                source_type='rag',  # Keep the same source_type for consistent prompt integration
                                metadata={
                                    'chunk_size': retriever.chunk_size,
                                    'overlap': retriever.overlap,
                                    'rank': i + 1,
                                    'content_type': 'rtl',
                                },
                            )
                        )
                except Exception as e:
                    print(f"Warning: Retriever failed for query '{query}' on RTL: {e}")
                    continue

        # Sort by score and limit to top contexts
        contexts.sort(key=lambda x: x.score, reverse=True)
        max_contexts = 10  # Reasonable default limit
        contexts = contexts[:max_contexts]

        return contexts


class DocRetriever:
    def __init__(self, text, chunk_size=100, overlap=20, source_type='spec'):
        # Store parameters as instance variables
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.source_type = source_type
        self.chunks = self._create_chunks(text, chunk_size, overlap)
        self.vectorizer = TfidfVectorizer()
        self.chunk_vectors = self.vectorizer.fit_transform(self.chunks)

    def _create_chunks(self, text, chunk_size, overlap):
        # Special case for code: try to preserve module boundaries and meaningful blocks
        if self.source_type == 'rtl':
            return self._create_code_aware_chunks(text, chunk_size, overlap)
        else:
            # Standard chunking for specification text
            words = text.split()
            chunks = []
            for i in range(0, len(words), chunk_size - overlap):
                chunk = ' '.join(words[i : i + chunk_size])
                chunks.append(chunk)
            return chunks

    def _create_code_aware_chunks(self, code_text, chunk_size, overlap):
        """Create chunks that try to respect code structure"""
        # Split by lines first
        lines = code_text.split('\n')
        chunks = []
        current_chunk_lines = []
        current_word_count = 0

        for line in lines:
            line_words = len(line.split())

            # Try to keep module definitions and important blocks together
            important_start = any(
                marker in line
                for marker in ['module ', 'function ', 'task ', 'always ', 'initial ']
            )
            important_end = (
                'endmodule' in line or 'endfunction' in line or 'endtask' in line
            )

            # If adding this line would exceed chunk size and we're not in the middle of an important block
            if (
                current_word_count + line_words > chunk_size
                and not important_start
                and not important_end
            ):
                # Save current chunk
                if current_chunk_lines:
                    chunks.append('\n'.join(current_chunk_lines))

                # Start new chunk with overlap
                overlap_lines = current_chunk_lines[
                    -min(len(current_chunk_lines), overlap) :
                ]
                current_chunk_lines = overlap_lines
                current_word_count = sum(len(l.split()) for l in overlap_lines)

            # Add current line to chunk
            current_chunk_lines.append(line)
            current_word_count += line_words

        # Add the last chunk
        if current_chunk_lines:
            chunks.append('\n'.join(current_chunk_lines))

        return chunks

    def retrieve(self, query, k=3):
        query_vector = self.vectorizer.transform([query])
        similarities = cosine_similarity(query_vector, self.chunk_vectors).flatten()
        top_k_indices = similarities.argsort()[-k:][::-1]
        return [self.chunks[i] for i in top_k_indices]


class KGNodeRetriever:
    def __init__(self, kg):
        assert kg is not None and 'nodes' in kg
        self.kg = kg
        self.model = SentenceTransformer('paraphrase-MiniLM-L6-v2')
        self.node_embeddings = self._create_node_embeddings()
        self.node_ids = [node['id'] for node in self.kg['nodes']]

    def _create_node_embeddings(self):
        node_texts = [
            f"{node['id']} {' '.join(node['attributes'].values())}"
            for node in self.kg['nodes']
        ]
        return self.model.encode(node_texts)

    def retrieve(self, query, k=3):
        query_embedding = self.model.encode([query])[0]
        similarities = cosine_similarity([query_embedding], self.node_embeddings)[0]
        top_k_indices = similarities.argsort()[-k:][::-1]
        return [self.node_ids[i] for i in top_k_indices]
