# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import os
import subprocess
import shutil
import json
import re
from pathlib import Path
import PyPDF2
import logging
from utils import OurTimer
from saver import saver
from config import FLAGS

print = saver.log_info


# Set up logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


def use_KG():
    timer = OurTimer()
    try:
        # Step 1: Set up the query command
        kg_root = FLAGS.KG_root
        query_method = FLAGS.graphrag_method
        query = FLAGS.query

        print(f"Using GraphRAG with the following parameters:")
        print(f"KG Root: {kg_root}")
        print(f"Query Method: {query_method}")
        print(f"Query: {query}")

        command = f"export PYTHONPATH='{FLAGS.graphrag_local_dir}:$PYTHONPATH' && python -m graphrag.query --data {kg_root} --method {query_method} \"{query}\""
        print(command)
        timer.time_and_clear("Setup")

        # Step 2: Execute the query
        print("Executing GraphRAG query...")
        process = subprocess.Popen(
            command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            universal_newlines=True,
        )

        # Step 3: Capture and print the output
        print("Query Output:")
        full_output = ""
        for line in process.stdout:
            print(line.rstrip())
            full_output += line

        return_code = process.wait()
        timer.time_and_clear("Query Execution")

        if return_code != 0:
            print(f"GraphRAG query failed with return code {return_code}")
            raise RuntimeError("GraphRAG query failed")

        # Step 4: Process and return the result
        # Note: This is a simple extraction. You might need to adjust based on the actual output format.
        result = full_output.strip().split("\n")[
            -1
        ]  # Assuming the last line is the answer
        print(f"\nFinal Answer: {result}")

        timer.time_and_clear("Result Processing")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    finally:
        timer.print_durations_log(print_func=print)

    return result
