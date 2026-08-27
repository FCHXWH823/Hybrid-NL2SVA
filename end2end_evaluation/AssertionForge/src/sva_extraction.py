# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import re
from typing import Tuple, List, Dict, Optional, Set
from saver import saver

print = saver.log_info


def extract_svas_strategy1(result: str) -> List[str]:
    """
    Extract SVAs using the strategy of finding blocks wrapped in backticks with SVA prefix.
    """
    sva_blocks = re.findall(r'SVA:?\s*```\s*(.*?)\s*```', result, re.DOTALL)
    svas = []
    for block in sva_blocks:
        individual_svas = re.findall(r'@\(\s*posedge\s+\w+.*?;', block, re.DOTALL)
        svas.extend(individual_svas)
    return svas


def extract_svas_strategy2(result: str) -> List[str]:
    """
    Extract SVAs directly by matching the core SVA pattern in the entire text.
    """
    return re.findall(r'@\(\s*posedge\s+\w+.*?;', result, re.DOTALL)


def extract_svas_strategy3(result: str) -> List[str]:
    """
    Extract SVAs by identifying patterns within specific SVA Plan blocks.
    """
    sva_blocks = re.findall(r'SVA for Plan \d+:.*?```(.*?)```', result, re.DOTALL)
    svas = []
    for block in sva_blocks:
        individual_svas = re.findall(r'@\(\s*posedge\s+\w+.*?;', block, re.DOTALL)
        svas.extend(individual_svas)
    return svas


def extract_svas_strategy4(result: str) -> List[str]:
    """
    Extract SVAs by splitting based on SVA plan headers and searching within those sections.
    """
    sva_blocks = re.split(r'SVA for Plan \d+:', result)[
        1:
    ]  # Skip the first split which is before the first SVA
    svas = []
    for block in sva_blocks:
        match = re.search(r'```(.*?)```', block, re.DOTALL)
        if match:
            sva = match.group(1).strip()
            if sva.startswith('@(posedge'):
                svas.append(sva)
    return svas


def extract_svas_strategy5(result: str) -> List[str]:
    """
    Extract SVAs by identifying SystemVerilog properties within systemverilog code blocks.
    """
    # Find all blocks of text marked as systemverilog
    systemverilog_blocks = re.findall(r'```systemverilog\s(.*?)```', result, re.DOTALL)
    svas = []
    for block in systemverilog_blocks:
        # Look for property declarations followed by assertions
        properties = re.findall(
            r'property\s+\w+\s*;.*?endproperty\s*;?\s*assert\s+property\(.+?\);',
            block,
            re.DOTALL,
        )
        svas.extend(properties)
    return svas


def extract_svas_strategy6(result: str) -> List[str]:
    """
    Extract SVAs from commented assertions within code blocks.
    """
    # Find all code blocks
    code_blocks = re.findall(r'```(.*?)```', result, re.DOTALL)
    svas = []
    for block in code_blocks:
        # Find all commented assert property lines
        assert_lines = re.findall(r'//\s*assert\s+property\s*\((.*?)\);', block)
        for line in assert_lines:
            # Extract the actual SVA content
            sva_match = re.search(r'@\(posedge\s+\w+\)\s*(.*)', line)
            if sva_match:
                svas.append(sva_match.group(0))
    return svas


def extract_svas_strategy7(result: str) -> List[str]:
    """
    Extract SVAs from code blocks without relying on comments.
    """
    # Find all code blocks
    code_blocks = re.findall(r'```(.*?)```', result, re.DOTALL)
    svas = []
    for block in code_blocks:
        # Find all assert property lines
        assert_lines = re.findall(r'assert\s+property\s*\((.*?)\);', block, re.DOTALL)
        svas.extend(assert_lines)
    return svas


def extract_svas_strategy8(result: str) -> List[str]:
    """
    Extract SVAs from structured property and assertion blocks.
    """
    pattern = r'property\s+(\w+);\s*@\s*\(posedge\s+\w+\)\s*(.*?);\s*endproperty\s*assert\s+property\s*\(\1\);'
    matches = re.findall(pattern, result, re.DOTALL)
    return [f"@(posedge clk) {sva_content}" for _, sva_content in matches]


def extract_svas_strategy9(result: str) -> List[str]:
    """
    Extract SystemVerilog Assertions (SVAs) from the provided text.

    Args:
        result (str): The text containing SVAs.

    Returns:
        List[str]: A list of extracted SVAs.
    """
    # Define a regex pattern to match SVAs
    sva_pattern = re.compile(
        r'assert\s+property\s*\(\s*@\(posedge\s+\w+\)\s+.*?;\s*\)',
        re.DOTALL
    )

    # Find all matches in the result
    matches = sva_pattern.findall(result)

    # Clean and return the matches
    svas = [match.strip() for match in matches]
    return svas



def clean_sva(sva: str) -> str:
    """
    Clean an SVA statement by removing comments and extra spaces.
    """
    sva = re.sub(r'/\*.*?\*/', '', sva, flags=re.DOTALL)  # Remove multi-line comments
    sva = re.sub(r'//.*', '', sva)  # Remove single-line comments
    return ' '.join(sva.split())  # Remove newlines and extra spaces


def extract_svas_from_block(block: str) -> List[str]:
    """
    Extract SVAs from a given block of text using multiple parsing strategies.

    Args:
        block (str): Text block containing potential SVAs.

    Returns:
        List[str]: A list of extracted SVAs.
    """
    strategies = {
        "Strategy 1": extract_svas_strategy1,
        "Strategy 2": extract_svas_strategy2,
        "Strategy 3": extract_svas_strategy3,
        "Strategy 4": extract_svas_strategy4,
        # "Strategy 5": extract_svas_strategy5,
        # "Strategy 6": extract_svas_strategy6,
        # "Strategy 7": extract_svas_strategy7,
        # "Strategy 8": extract_svas_strategy8,
        "Strategy 9": extract_svas_strategy9,
    }

    all_extracted_svas = set()
    strategy_counts: Dict[str, int] = {}

    for name, strategy in strategies.items():
        svas = strategy(block)
        strategy_counts[name] = len(svas)
        all_extracted_svas.update(svas)

    # Clean and filter SVAs
    signal_svas = [clean_sva(sva) for sva in all_extracted_svas if sva.strip()]

    # Print strategy effectiveness
    print("\nStrategy Effectiveness:")
    for name, count in strategy_counts.items():
        print(f"{name}: {count} SVAs extracted")
    print(f"Total unique SVAs after cleaning: {len(signal_svas)}")

    return signal_svas
