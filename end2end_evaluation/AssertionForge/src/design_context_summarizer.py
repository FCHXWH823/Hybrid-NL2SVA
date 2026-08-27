# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from typing import Dict, List, Optional, Any
from utils_LLM import llm_inference
import time
import re
from saver import saver

# Use saver's logging mechanism
print = saver.log_info


class DesignContextSummarizer:
    """
    Class to generate summaries and overviews for design specifications and RTL
    to enhance SVA generation prompts.
    """

    def __init__(self, llm_agent: str = "gpt-4"):
        """
        Initialize with specified LLM agent.

        Args:
            llm_agent: The LLM agent to use for summarization
        """
        self.llm_agent = llm_agent
        self.summary_cache = {}  # Cache for summaries
        self.global_summary = None  # Cache for global design summary
        self.all_signals_summary = None  # Cache for comprehensive signals summary

    def generate_global_summary(
        self, spec_text: str, rtl_text: str, valid_signals: List[str]
    ) -> Dict[str, Any]:
        """
        Generate a comprehensive global summary of the design, including spec and RTL.
        This should be called once and the results cached for future use.

        Args:
            spec_text: The design specification text
            rtl_text: The RTL code
            valid_signals: List of valid signal names

        Returns:
            Dictionary with various summary components
        """
        # Check if we already generated the global summary
        if self.global_summary is not None:
            return self.global_summary

        print("Generating global design summary...")

        # Generate design specification summary
        design_summary = self._generate_design_specification_summary(spec_text)

        # Generate RTL architecture summary
        rtl_summary = self._generate_rtl_architecture_summary(rtl_text)

        # Generate comprehensive signals summary for all valid signals
        signals_summary = self._generate_comprehensive_signals_summary(
            spec_text, rtl_text, valid_signals
        )
        self.all_signals_summary = signals_summary

        # Generate design patterns summary
        patterns_summary = self._generate_design_patterns_summary(spec_text, rtl_text)

        # Cache the result
        self.global_summary = {
            "design_summary": design_summary,
            "rtl_summary": rtl_summary,
            "signals_summary": signals_summary,
            "patterns_summary": patterns_summary,
            "generation_time": time.time(),
        }

        print("Global design summary generated successfully")
        return self.global_summary


    def _generate_design_specification_summary(self, spec_text: str) -> str:
        """Generate a concise summary of the design specification."""
        prompt = f"""
        You are an expert hardware design engineer. Please provide a concise summary (3-5 sentences) 
        of the following hardware design specification. Focus on the main functionality, key components, 
        and architecture. The summary should give a clear high-level understanding of what this design does.
        
        Design Specification:
        {spec_text}
        
        Provide only the summary, with no additional commentary or introduction.
        """

        return self._call_llm(prompt, "design_spec_summary")

    def _generate_rtl_architecture_summary(self, rtl_text: str) -> str:
        """Generate a concise summary of the RTL architecture."""
        prompt = f"""
        You are an expert hardware design engineer. Please provide a concise summary (3-5 sentences)
        of the following RTL code. Focus on the module hierarchy, interfaces, and key architectural features.
        
        RTL Code:
        {rtl_text}
        
        Provide only the RTL architecture summary, with no additional commentary or introduction.
        """

        return self._call_llm(prompt, "rtl_architecture_summary")

    def _generate_comprehensive_signals_summary(
        self, spec_text: str, rtl_text: str, signals: List[str]
    ) -> str:
        """
        Generate a comprehensive summary of all signals in the design, with technical details.

        Args:
            spec_text: The design specification
            rtl_text: The RTL code
            signals: List of valid signal names

        Returns:
            String containing a comprehensive summary of all signals
        """
        signals_str = ", ".join(signals)
        prompt = f"""
        You are an expert hardware verification engineer. Please analyze the following design specification and RTL code
        to provide a comprehensive summary of the signals in the design. For each signal, include details about:
        
        1. Signal name
        2. Signal type (input, output, inout, internal, clock, reset, etc.)
        3. Bit width (e.g., 1-bit, 8-bit, 32-bit)
        4. Functionality and purpose
        5. Key interactions with other signals
        
        Valid Signals: {signals_str}
        
        Design Specification:
        {spec_text}
        
        RTL Code:
        {rtl_text}
        
        Focus on the signals listed above. If the RTL/spec doesn't provide information for a signal, make your best inference.
        Format your response as a list with each signal having its own paragraph that includes all the details mentioned above.
        Be concise yet complete.
        """

        return self._call_llm(prompt, "comprehensive_signals_summary")

    def _generate_design_patterns_summary(self, spec_text: str, rtl_text: str) -> str:
        """Generate a summary of design patterns and protocols in the design."""
        prompt = f"""
        You are an expert hardware design engineer. Please analyze the following design specification and RTL code
        to identify and summarize key design patterns, protocols, or verification-critical structures.
        Examples might include handshaking protocols, state machines, pipelines, arbiters, or clock domain crossings.
        
        Design Specification:
        {spec_text}
        
        RTL Code:
        {rtl_text}
        
        Provide a concise summary (5-10 sentences) of the key design patterns and their verification implications.
        """

        return self._call_llm(prompt, "design_patterns_summary")

    def get_signal_specific_summary(
        self, signal_name: str, spec_text: str, rtl_text: str
    ) -> Dict[str, str]:
        """
        Get a signal-specific summary. This first checks the cache before generating.

        Args:
            signal_name: The name of the signal to focus on
            spec_text: The design specification text
            rtl_text: The RTL code

        Returns:
            Dictionary with signal-specific summary information
        """
        # Check if we already have this signal in the cache
        cache_key = f"signal_{signal_name}"
        if cache_key in self.summary_cache:
            return self.summary_cache[cache_key]

        print(f"Generating detailed summary for signal: {signal_name}")

        # Generate detailed signal description
        prompt = f"""
        You are an expert hardware verification engineer. Please provide a detailed description of
        the signal '{signal_name}' based on the following specification and RTL.
        
        Design Specification:
        {spec_text}
        
        RTL Code:
        {rtl_text}
        
        Include in your description:
        1. The precise function of this signal
        2. Its type (input, output, inout, internal, etc.) and bit width
        3. Its timing characteristics (synchronous/asynchronous, edge-triggered, etc.)
        4. Key relationships with other signals
        5. How it affects or is affected by the overall system behavior
        6. Any special conditions or corner cases related to this signal
        
        Write 3-5 sentences with comprehensive, verification-focused details.
        """

        signal_description = self._call_llm(prompt, f"signal_desc_{signal_name}")

        # Cache the result
        signal_summary = {
            "description": signal_description,
            "generation_time": time.time(),
        }

        self.summary_cache[cache_key] = signal_summary
        return signal_summary

    def add_enhanced_context(
        self, dynamic_context: str, target_signal_name: str
    ) -> str:
        """
        Add enhanced context to an existing dynamic context string.
        Includes summaries for all signals, with special focus on the target signal.

        Args:
            dynamic_context: The original dynamic context
            target_signal_name: The primary signal name for this context

        Returns:
            Enhanced dynamic context with summaries
        """
        if self.global_summary is None:
            print(
                "Warning: Global summary not generated yet, returning original context"
            )
            return dynamic_context

        # Get the cached signal summary for the target signal
        target_signal_summary = self.summary_cache.get(
            f"signal_{target_signal_name}",
            {"description": "No detailed information available for this signal."},
        )

        enhanced_context = f"""
Design Overview:
{self.global_summary["design_summary"]}

RTL Architecture:
{self.global_summary["rtl_summary"]}

Target Signal '{target_signal_name}' Description:
{target_signal_summary["description"]}

All Signals Summary:
{self.all_signals_summary}

Key Design Patterns:
{self.global_summary["patterns_summary"]}

{dynamic_context}
"""
        return enhanced_context

    def _call_llm(self, prompt: str, tag: str) -> str:
        """Call the LLM with the given prompt"""
        # Import here to avoid circular imports
        return llm_inference(self.llm_agent, prompt, tag)
