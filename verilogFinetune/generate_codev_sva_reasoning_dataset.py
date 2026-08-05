"""
Stages 2+3+4 of the CodeV-SVA OL-NL + two-part decomposition pipeline (see the
plan in /Users/fch/.claude/plans/polished-chasing-wreath.md).

Takes the 5,000-record sample selected by select_codev_sva_sample.py and, for
each record:
  1. Generates a grounded OL-NL statement from its (possibly abstract)
     Question text (generate_ol_nl_explanation.generate_ol_nl -- Stage 2).
  2. Feeds that OL-NL text into the existing, unmodified DFS decomposition
     pipeline (generate_dfs_explanation.walk_tree / render_decomposition_tree,
     sva_graph.render_merge_tree -- Stage 3) to get Part 1 (top-down
     NL-decomposition tree) and Part 2 (bottom-up symbolic derivation).
  3. Reassembles a new chat-format record: the original system/user turns
     unchanged, and a new assistant turn made of the OL-NL statement, Part 1,
     a bridge sentence, Part 2, and the final ```systemverilog``` block
     (Stage 4) -- kept compatible with the FVEval harness's
     utils.parse_code_response(), which only looks at the *last*
     ```systemverilog fence in the response.

Records whose OL-NL statement never fully grounds (every leaf signal named)
within --max-retries are dropped and backfilled with the next eligible-but-
unused record from the source file, so the final dataset always has exactly
--sample-size records.

Each record involves several sequential, independent LLM calls (I/O-bound),
so records are processed concurrently across --workers threads; a lock
guards the shared source-file handle and the checkpoint file. Resumable via
the same load_checkpoint/checkpoint-file pattern used by every other
generation script in this directory.

Usage:
    python verilogFinetune/generate_codev_sva_reasoning_dataset.py \\
        --sample verilogFinetune/data/codev_sva_5000_sample.json \\
        --source verilogFinetune/data/CodeV-SVA-dataset-training-83K.jsonl \\
        --output verilogFinetune/data/codev_sva_ol_dfs_5000.jsonl \\
        --workers 24
"""
import argparse
import json
import os
import sys
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_dfs_explanation import render_decomposition_tree, walk_tree
from generate_ol_nl_explanation import generate_ol_nl, split_user_content
from generate_prompt_guided_explanation import load_operator_context
from select_codev_sva_sample import extract_final_sva
from sva_graph import build_operator_signal_graph, render_merge_tree


def build_line_offsets(path):
    offsets = []
    with open(path, "rb") as file:
        offset = 0
        for line in file:
            offsets.append(offset)
            offset += len(line)
    return offsets


class SourceFile:
    """Thread-safe random-access reader over the source JSONL, plus a cursor
    for finding backfill candidates (both guarded by the same lock, since
    both do a seek+read against the one shared file handle)."""

    def __init__(self, path, offsets):
        self.offsets = offsets
        self.handle = open(path, "rb")
        self.lock = threading.Lock()

    def read_record(self, index):
        with self.lock:
            self.handle.seek(self.offsets[index])
            line = self.handle.readline().decode("utf-8")
        return json.loads(line)

    def next_backfill_candidate(self, start_index, used_indices):
        """Scans forward from start_index for the next record that is both
        unused and passes the same eligibility filter as
        select_codev_sva_sample.py. Returns (index, record, sva)."""
        with self.lock:
            index = start_index
            while index < len(self.offsets):
                if index not in used_indices:
                    self.handle.seek(self.offsets[index])
                    record = json.loads(self.handle.readline().decode("utf-8"))
                    sva = extract_final_sva(record["messages"][2]["content"])
                    if sva is not None and sva.count("assert property") == 1:
                        try:
                            build_operator_signal_graph(sva)
                        except Exception:
                            index += 1
                            continue
                        used_indices.add(index)
                        return index, record, sva
                index += 1
        raise StopIteration("Ran out of backfill candidates")


def assemble_assistant_content(ol_nl, root, parsed_by_id, sva):
    decomposition = "\n".join(render_decomposition_tree(root, parsed_by_id))
    merge = "\n".join(render_merge_tree(root))
    return (
        f"OL NL: {ol_nl}\n\n"
        "***nl-decomposition tree***\n"
        f"{decomposition}\n\n"
        "According to the above natural-language decomposition, we finally derive "
        "the SystemVerilog assertion as follows:\n\n"
        "***operator-merge-sva tree***\n"
        f"{merge}\n\n"
        "```systemverilog\n"
        f"{sva}\n"
        "```"
    )


def process_one(client, model, record, sva, operator_context, max_retries, sv_dir, task_id):
    """Returns the new assistant content, or None if this record can't be
    used (OL-NL never verified equivalent, or the property is a bare signal
    with no operator to decompose). task_id must be unique across concurrent
    callers (the worker passes the record's own index) so parallel JasperGold
    invocations don't collide on temp files under sv_dir."""
    user = record["messages"][1]["content"]
    testbench, question = split_user_content(user)

    root = build_operator_signal_graph(sva)
    if root["type"] != "operator":
        return None  # degenerate: bare signal/sequence, nothing to decompose

    ol_nl, verified = generate_ol_nl(
        client, model, question, sva, testbench, operator_context, max_retries, sv_dir, task_id
    )
    if not verified:
        return None

    parsed_by_id = {}
    walk_tree(client, model, root, ol_nl, operator_context, max_retries, parsed_by_id)
    return assemble_assistant_content(ol_nl, root, parsed_by_id, sva)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample", default="verilogFinetune/data/codev_sva_5000_sample.json")
    parser.add_argument("--source", default="verilogFinetune/data/CodeV-SVA-dataset-training-83K.jsonl")
    parser.add_argument("--output", default="verilogFinetune/data/codev_sva_ol_dfs_5000.jsonl")
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--operators", default="operators.json")
    parser.add_argument("--config", default="Src/Config.yml")
    parser.add_argument("--model", default="o4-mini")
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--limit", type=int, default=None, help="Only process this many records (for smoke-testing)")
    parser.add_argument("--sv-dir", default="verilogFinetune/data/ol_nl_validation_scratch",
                         help="Scratch dir for JasperGold temp files (shared across workers, keyed by task_id)")
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or f"{args.output}.checkpoint.jsonl"

    with open(args.config) as file:
        config = yaml.safe_load(file)
    client = OpenAI(api_key=config["Openai_API_Key"])
    operator_context = load_operator_context(args.operators)

    with open(args.sample) as file:
        sample = json.load(file)
    target_indices = [r["index"] for r in sample["records"]]
    if args.limit is not None:
        target_indices = target_indices[: args.limit]
    used_indices = set(target_indices)

    print(f"Building line-offset index for {args.source} ...")
    offsets = build_line_offsets(args.source)
    source = SourceFile(args.source, offsets)

    completed = {}
    if os.path.exists(checkpoint_path):
        with open(checkpoint_path) as file:
            for line in file:
                result = json.loads(line)
                completed[result["source_index"]] = result
    print(f"Loaded {len(completed)} already-completed records from {checkpoint_path}")

    checkpoint_lock = threading.Lock()
    checkpoint_file = open(checkpoint_path, "a")
    backfill_cursor = [max(target_indices) + 1 if target_indices else 0]
    backfill_lock = threading.Lock()
    dropped_lock = threading.Lock()
    dropped_count = [0]
    progress_lock = threading.Lock()
    progress = [0]
    total = len(target_indices)

    def get_backfill():
        with backfill_lock:
            new_index, new_record, new_sva = source.next_backfill_candidate(
                backfill_cursor[0], used_indices
            )
            backfill_cursor[0] = new_index + 1
            return new_index, new_record, new_sva

    def worker(source_index):
        """Processes one target slot, recursively backfilling on failure
        until it succeeds or backfill candidates run out. Returns nothing;
        writes directly to the checkpoint file under a lock. `completed` is
        keyed by the original source_index (the target slot), never by a
        backfilled replacement's own index -- checked once upfront so a
        resumed run skips slots a prior run already finished."""
        if source_index in completed:
            with progress_lock:
                progress[0] += 1
            return

        index = source_index
        record = source.read_record(index)
        while True:
            sva = extract_final_sva(record["messages"][2]["content"])
            try:
                assistant_content = process_one(
                    client, args.model, record, sva, operator_context, args.max_retries,
                    args.sv_dir, task_id=index,
                )
            except Exception as error:
                print(f"    unexpected error on source_index={index} ({error}), dropping and backfilling")
                assistant_content = None

            if assistant_content is not None:
                result = {
                    "source_index": source_index,  # the ORIGINAL slot this fills, for output ordering
                    "resolved_index": index,
                    "messages": [
                        record["messages"][0],
                        record["messages"][1],
                        {"role": "assistant", "content": assistant_content},
                    ],
                }
                with checkpoint_lock:
                    checkpoint_file.write(json.dumps(result) + "\n")
                    checkpoint_file.flush()
                    completed[source_index] = result
                with progress_lock:
                    progress[0] += 1
                    print(f"[{progress[0]}/{total}] done (source_index={source_index}, resolved_index={index})")
                return

            with dropped_lock:
                dropped_count[0] += 1
            index, record, sva = get_backfill()
            print(f"    backfilling slot source_index={source_index} with resolved_index={index}")

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(worker, index) for index in target_indices]
        for future in as_completed(futures):
            future.result()  # re-raise any worker exception

    checkpoint_file.close()

    final_records = [completed[i]["messages"] for i in target_indices]
    with open(args.output, "w") as file:
        for messages in final_records:
            file.write(json.dumps({"messages": messages}, ensure_ascii=False) + "\n")

    print(f"Wrote {len(final_records)} records to {args.output}")
    print(f"  dropped-and-backfilled: {dropped_count[0]}")


if __name__ == "__main__":
    main()
