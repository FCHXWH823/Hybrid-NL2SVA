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
import logging, datetime
from utils import OurTimer, get_ts
from saver import saver
from config import FLAGS
import tiktoken

print = saver.log_info


# Set up logging
logging.basicConfig(
    level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s'
)


def build_KG():
    timer = OurTimer()
    try:
        input_file_path = FLAGS.input_file_path

        base_dir = get_base_dir(input_file_path)
        print(f"Derived base directory: {base_dir}")

        # Step 1: Process input file(s)
        print("Step 1: Processing input file(s)")
        graph_rag_dir = create_directory_structure(base_dir)
        timer.time_and_clear(f'create_directory_structure')

        # Clean up the input folder
        clean_input_folder(os.path.join(graph_rag_dir, 'input'))
        timer.time_and_clear(f'clean_input_folder')

        process_files(input_file_path, graph_rag_dir)
        timer.time_and_clear(f'process_files')

        # Step 2: Initialize GraphRAG
        print("Step 2: Initializing GraphRAG")
        initialize_graphrag(graph_rag_dir)
        timer.time_and_clear(f'initialize_graphrag')

        # Step 3: Update .env file
        print("Step 3: Updating .env file")
        shutil.copy(FLAGS.env_source_path, os.path.join(graph_rag_dir, '.env'))
        print(f"Copied .env from {FLAGS.env_source_path} to {graph_rag_dir}")
        timer.time_and_clear(f'Updating .env file')

        # Step 4: Update settings.yaml file
        print("Step 4: Updating settings.yaml file")
        shutil.copy(
            FLAGS.settings_source_path, os.path.join(graph_rag_dir, 'settings.yaml')
        )
        print(
            f"Copied settings.yaml from {FLAGS.settings_source_path} to {graph_rag_dir}"
        )
        timer.time_and_clear(f'Updating settings.yaml file')

        # New Step: Copy entity extraction prompt
        print("Step 5: Copying entity extraction prompt")
        copy_entity_extraction_prompt(graph_rag_dir)
        timer.time_and_clear(f'Copying entity extraction prompt')

        # Step 6: Run GraphRAG indexing
        print("Step 6: Running GraphRAG indexing")
        return_code = run_graphrag_index(graph_rag_dir)
        if return_code != 0:
            timer.time_and_clear("error")
            timer.print_durations_log(print_func=print)
            raise RuntimeError("GraphRAG indexing failed")

        # New Step: Detect and report log folder
        print("Step 7: Detecting GraphRAG log folder")
        log_folder = detect_graphrag_log_folder(graph_rag_dir)
        if log_folder:
            print(f"GraphRAG log folder detected: {log_folder}")
        else:
            print("No new GraphRAG log folder detected")

        print("Process completed successfully")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        raise
    timer.time_and_clear(f'run_graphrag_index')
    timer.print_durations_log(print_func=print)


def process_files(input_file_path, graph_rag_dir):
    if isinstance(input_file_path, list):
        text_file_paths = []
        total_stats = {'pages': 0, 'size': 0, 'words': 0, 'tokens': 0}
        for file_path in input_file_path:
            result = process_single_file(file_path, graph_rag_dir)
            text_file_paths.append(result[0])
            stats = result[1]
            total_stats['pages'] += stats[0]
            total_stats['size'] += stats[1]
            total_stats['words'] += stats[2]
            total_stats['tokens'] += stats[3]

        print("\nTotal Statistics:")
        print(f"  Total Pages: {total_stats['pages']}")
        print(f"  Total File Size: {total_stats['size']:.2f} MB")
        print(f"  Total Word Count: {total_stats['words']}")
        print(f"  Total Token Count: {total_stats['tokens']}")

        return text_file_paths
    else:
        return [process_single_file(input_file_path, graph_rag_dir)[0]]


def process_single_file(file_path, graph_rag_dir):
    if file_path.lower().endswith('.jsonl'):
        num_pages, file_size = get_jsonl_stats(file_path)
        print(f"JSONL Statistics: {num_pages} pages, {file_size:.2f} MB")
        return parse_jsonl_to_text(file_path, os.path.join(graph_rag_dir, 'input'))
    else:
        num_pages, file_size, word_count, token_count = get_pdf_stats(file_path)
        print(f"PDF Statistics for {os.path.basename(file_path)}:")
        print(f"  Pages: {num_pages}")
        print(f"  File size: {file_size:.2f} MB")
        print(f"  Word count: {word_count}")
        print(f"  Token count: {token_count}")
        return parse_pdf_to_text(file_path, os.path.join(graph_rag_dir, 'input')), (
            num_pages,
            file_size,
            word_count,
            token_count,
        )


def detect_graphrag_log_folder(graph_rag_dir):
    """
    Detect the log folder generated by GraphRAG.

    Args:
    graph_rag_dir (str): Path to the GraphRAG directory

    Returns:
    str: Path to the detected log folder, or None if not found
    """
    output_dir = os.path.join(graph_rag_dir, 'output')
    before_folders = set(os.listdir(output_dir))

    # Run GraphRAG indexing
    run_graphrag_index(graph_rag_dir)

    after_folders = set(os.listdir(output_dir))
    new_folders = after_folders - before_folders

    if not new_folders:
        return None

    # If multiple new folders, find the one closest to the current timestamp
    current_ts = get_ts()
    closest_folder = min(
        new_folders,
        key=lambda f: abs(
            datetime.datetime.strptime(f, '%Y%m%d-%H%M%S')
            - datetime.datetime.strptime(current_ts, '%Y-%m-%dT%H-%M-%S.%f')
        ),
    )

    return os.path.join(output_dir, closest_folder)


def get_base_dir(file_path):
    if isinstance(file_path, list):
        if not file_path:
            raise ValueError("The input file path list is empty.")

        base_dirs = set(os.path.dirname(path) for path in file_path)
        if len(base_dirs) > 1:
            raise ValueError(
                "All input files must be in the same directory. "
                f"Found multiple directories: {', '.join(base_dirs)}"
            )

        return base_dirs.pop()  # Return the single base directory
    elif isinstance(file_path, str):
        return os.path.dirname(file_path)
    else:
        raise TypeError("Input must be a string or a list of strings.")


def get_pdf_stats(pdf_path):
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        num_pages = len(reader.pages)
        file_size = os.path.getsize(pdf_path) / (1024 * 1024)  # Convert to MB

        # Extract text and count words
        full_text = ""
        for page in reader.pages:
            full_text += page.extract_text()

        word_count = len(re.findall(r'\w+', full_text))

        # Count tokens using tiktoken
        encoding = tiktoken.get_encoding(
            "cl100k_base"
        )  # Versatility: It supports multiple encoding schemes used by different OpenAI models (e.g., "cl100k_base" for GPT-4, "p50k_base" for GPT-3).
        token_count = len(encoding.encode(full_text))

    return num_pages, file_size, word_count, token_count


def get_jsonl_stats(jsonl_path):
    with open(jsonl_path, 'r') as file:
        lines = file.readlines()
        num_pages = sum(1 for line in lines if json.loads(line).get('page') is not None)
        file_size = os.path.getsize(jsonl_path) / (1024 * 1024)  # Convert to MB
    return num_pages, file_size


def parse_pdf_to_text(pdf_path, output_dir):
    design_name = os.path.splitext(os.path.basename(pdf_path))[0]
    output_path = os.path.join(output_dir, f"{design_name}.txt")

    with open(pdf_path, 'rb') as pdf_file, open(
        output_path, 'w', encoding='utf-8'
    ) as txt_file:
        reader = PyPDF2.PdfReader(pdf_file)
        for page in reader.pages:
            txt_file.write(page.extract_text())

    return output_path


def clean_input_folder(input_dir):
    """
    Remove all files from the input directory.

    Args:
    input_dir (str): Path to the input directory
    """
    for file in os.listdir(input_dir):
        file_path = os.path.join(input_dir, file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
            elif os.path.isdir(file_path):
                shutil.rmtree(file_path)
        except Exception as e:
            print(f"Failed to delete {file_path}. Reason: {e}")
    print(f"Cleaned up input folder: {input_dir}")


def clean_table(table_content):
    table = re.sub(r'\\text\{([^}]*)\}', r'\1', table_content)
    table = table.replace('\\\\', '\n')
    table = table.replace('&', ' | ')
    table = re.sub(r'\\[a-zA-Z]+', '', table)
    table = re.sub(r'\^\{?2\}?', '²', table)
    table = re.sub(r'\s+', ' ', table).strip()
    return table


def parse_jsonl_to_text(jsonl_path, output_dir):
    design_name = os.path.splitext(os.path.basename(jsonl_path))[0]
    output_path = os.path.join(output_dir, f"{design_name}_processed.txt")

    with open(jsonl_path, 'r') as jsonl_file, open(
        output_path, 'w', encoding='utf-8'
    ) as txt_file:
        for line in jsonl_file:
            json_obj = json.loads(line)

            # Process normal "out" content
            if 'out' in json_obj and json_obj['out'].strip():
                txt_file.write(json_obj['out'] + "\n\n")

            # Process Table 1 if present in raw_out
            if 'raw_out' in json_obj:
                raw_out = json_obj['raw_out']
                if 'Table 1: Pinout description' in raw_out:  # pretty hacky...
                    txt_file.write("Table 1: Pinout description\n\n")
                    table_content = re.search(
                        r'\\begin\{array\}(.*?)\\end\{array\}', raw_out, re.DOTALL
                    )
                    if table_content:
                        cleaned_table = clean_table(table_content.group(1))
                        txt_file.write(cleaned_table + "\n\n")

                # Process other raw_out content if "out" is empty
                elif not json_obj.get('out'):
                    cleaned_raw_out = re.sub(
                        r'<[^>]+>', '', raw_out
                    )  # Remove HTML-like tags
                    cleaned_raw_out = re.sub(
                        r'\\[a-zA-Z]+(\{[^}]*\})?', '', cleaned_raw_out
                    )  # Remove LaTeX commands
                    cleaned_raw_out = cleaned_raw_out.replace('\\n', '\n').strip()
                    if cleaned_raw_out:
                        txt_file.write(cleaned_raw_out + "\n\n")

    return output_path


def create_directory_structure(base_dir):
    assert isinstance(base_dir, str)
    graph_rag_dir = os.path.join(base_dir, f'graph_rag_{FLAGS.design_name}')
    input_dir = os.path.join(graph_rag_dir, 'input')
    os.makedirs(input_dir, exist_ok=True)
    return graph_rag_dir


def initialize_graphrag(graph_rag_dir):
    command = f"export PYTHONPATH='{FLAGS.graphrag_local_dir}' && python -m graphrag.index --init --root {graph_rag_dir}"
    print(command)
    result = subprocess.run(command, shell=True, capture_output=True, text=True)

    if result.returncode != 0:
        if "Project already initialized" in result.stderr:
            print("GraphRAG project already initialized. Skipping initialization.")
        else:
            print("Error during initialization:")
            print(result.stderr)
            raise RuntimeError("GraphRAG initialization failed")
    else:
        print("GraphRAG Initialization Output:")
        print(result.stdout)


def copy_entity_extraction_prompt(graph_rag_dir):
    """
    Copy the entity extraction prompt from the source path to the destination.

    Args:
    graph_rag_dir (str): Path to the GraphRAG directory
    """
    source_path = FLAGS.entity_extraction_prompt_source_path
    destination_path = os.path.join(graph_rag_dir, 'prompts', 'entity_extraction.txt')

    # Ensure the prompts directory exists
    os.makedirs(os.path.dirname(destination_path), exist_ok=True)

    shutil.copy(source_path, destination_path)
    print(f"Copied entity extraction prompt from {source_path} to {destination_path}")


def run_graphrag_index(graph_rag_dir):
    command = f"export PYTHONPATH='{FLAGS.graphrag_local_dir}:$PYTHONPATH' && python -m graphrag.index --root {graph_rag_dir}"
    print(command)
    process = subprocess.Popen(
        command,
        shell=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
        universal_newlines=True,
    )

    print(f"GraphRAG Indexing Output:")
    for line in process.stdout:
        print(f"{line.rstrip()}")  # Remove trailing newline and print

    return_code = process.wait()

    if return_code != 0:
        print(f"GraphRAG indexing failed with return code {return_code}")
    else:
        print(f"GraphRAG indexing completed successfully")

    return return_code


if __name__ == "__main__":
    build_KG()


