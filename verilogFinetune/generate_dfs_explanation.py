"""
Generates a depth-first, node-by-node explanation of each golden SVA's deep
operator/signal syntax tree (sva_graph.py), by literally walking the tree in
Python and making one small, focused LLM call per *operator* node -- rather
than handing the model the whole tree at once and asking it to narrate all of
it in a single completion.

For each (golden SVA, natural-language explanation) pair, this script:
  1. Parses the SVA into its full operator/signal tree with sva_graph.py --
     going through the sequence and Boolean layers too, not stopping at the
     property layer the way sva_parser.py (and the Fig. 12 prompt) does.
  2. Walks that tree depth-first, root first. At each *operator* node, it
     asks the model for exactly four things, given only that node's own
     operator, code, and operand(s):
         (a) the natural-language piece for this node (restated, not chosen)
         (b) why the (given) operator represents that piece
         (c) the sub-piece belonging to operand 1
         (d) the sub-piece belonging to operand 2 (if binary)
     Python then threads (c)/(d) down as the *input* natural-language piece
     for the corresponding child call -- so a child's (a) is never generated
     independently, it's mechanically inherited from its parent's split. This
     is what actually enforces top-down consistency, rather than an
     instruction the model has to remember to follow.
  3. Signal (leaf) nodes never get their own block or LLM call at all: their
     natural-language piece is already sitting verbatim in their parent's
     (c)/(d), so a separate block for them would just repeat that text.
  4. The final "output" text has two parts:
       - a top-down "nl-decomposition tree", one block per operator node
         (operator / natural-language piece / reason), with leaves shown as
         "operator: null" / the signal or literal itself / a fixed reason;
       - a bottom-up "operator-merge-sva tree" showing the same tree read the
         other way: leaves and intermediate code merge upward via each
         operator until the full property expression appears at the end.
     The second tree is pure bookkeeping over data already in sva_graph.py
     (node["code"]) -- no extra LLM calls, just a different rendering.

Falls back to generate_prompt_guided_explanation.build_prompt() (the
property-layer-only Fig. 12 prompt, itself falling back further to the
structure-free prompt) whenever sva_graph.py can't parse a given SVA
(~15% of the corpus; see check_parser_coverage.py) -- in that case the whole
record is generated with a single call, as before.

Usage:
    python verilogFinetune/generate_dfs_explanation.py \\
        --input verilogFinetune/data/qwen_explanation.jsonl \\
        --output verilogFinetune/data/qwen_dfs_explanation_new.jsonl
"""
import argparse
import json
import os
import re
import sys
import time

import yaml
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_prompt_guided_explanation import (
    load_checkpoint,
    load_input_records,
    load_operator_context,
)
from generate_prompt_guided_explanation import build_prompt as build_fallback_prompt
from sva_graph import build_operator_signal_graph, render_merge_tree

NODE_SYSTEM_PROMPT = (
    "You are a helpful bot that explains one node of an already-parsed SystemVerilog "
    "assertion syntax tree at a time, following the requested format exactly."
)

PROMPT_TEMPLATE_NODE = """You are explaining ONE node of a SystemVerilog assertion's already-parsed \
syntax tree. You are not shown the rest of the tree, and you must not assume \
anything about it -- only this node's own operator and operand(s), given below, \
are relevant.

Natural-language piece assigned to this node (this is ground truth -- restate \
it, do not change its meaning or wording):
 {nl_piece}

This node's code:
 {code}

This node's operator: {operator}
{operand_lines}

SVA Operator Context:
 {operator_context}

Output exactly the following labeled lines in plain text (not JSON):

(a) Natural language piece: <restate the natural-language piece given above, unchanged, word for word>
(b) Operator reasoning: <why the operator "{operator}" correctly represents this natural-language piece>
(c) Operand 1 natural language piece: <the part of the natural-language piece above that corresponds to operand 1's code>
{d_line}
"""

_FIELD_PATTERN = re.compile(r"\(([abcd])\)[^:\n]*:\s*(.*?)(?=\n\s*\([abcd]\)|\Z)", re.DOTALL)


def parse_node_response(text):
    """Extract {"a":..., "b":..., "c":..., "d":...} from the model's labeled
    plain-text reply. Missing labels are simply absent from the result."""
    return {letter: content.strip() for letter, content in _FIELD_PATTERN.findall(text)}


def _code_of(node):
    return node["code"] if node["type"] == "operator" else node["label"]


def build_node_prompt(nl_piece, node, operator_context):
    children = node["children"]
    operand_lines = [f"Operand 1 code: {_code_of(children[0])}"]
    d_line = ""
    if len(children) == 2:
        operand_lines.append(f"Operand 2 code: {_code_of(children[1])}")
        d_line = (
            "(d) Operand 2 natural language piece: <the part of the natural-language "
            "piece above that corresponds to operand 2's code>"
        )
    return PROMPT_TEMPLATE_NODE.format(
        nl_piece=nl_piece,
        code=node["code"],
        operator=node["label"],
        operand_lines="\n".join(operand_lines),
        operator_context=operator_context,
        d_line=d_line,
    )


def call_node_llm(client, model, nl_piece, node, operator_context, max_retries):
    """Call the LLM for one operator node, retrying on API failure or on a
    reply missing a required field. After max_retries, degrades to reusing
    nl_piece verbatim for every operand rather than losing the subtree."""
    required = {"a", "b", "c"}
    if len(node["children"]) == 2:
        required.add("d")

    prompt = build_node_prompt(nl_piece, node, operator_context)
    last_error = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": NODE_SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = parse_node_response(completion.choices[0].message.content)
            if required.issubset(parsed):
                return parsed
            print(f"    node reply missing field(s) {required - parsed.keys()}, retrying...")
        except Exception as error:
            last_error = error
            print(f"    node call failed ({error}), retrying...")
        time.sleep(2 ** attempt)

    print(f"    giving up on this node after {max_retries} attempts; "
          f"degrading to a verbatim split (last_error={last_error})")
    parsed = {"a": nl_piece, "b": "(reasoning unavailable -- extraction failed after retries)", "c": nl_piece}
    if "d" in required:
        parsed["d"] = nl_piece
    return parsed


def render_decomposition_tree(node, parsed_by_id, prefix=""):
    """Top-down view: this node's operator/natural-language piece/reason,
    with '├──'/'└──' branches down to its operand(s). A leaf shows
    'operator: null' and its own code as the piece, with a fixed reason --
    there's nothing to call the LLM for at a leaf. Line 0 carries no prefix
    (the caller adds the connector); lines[1:] already have their full
    indent baked in via `prefix`, so parents can splice children in directly."""
    if node["type"] == "signal":
        return [
            "operator: null",
            f"{prefix}    natural language piece: {node['label']}",
            f"{prefix}    reason: direct signal name or value",
        ]

    parsed = parsed_by_id[node["id"]]
    lines = [
        f"operator: {node['label']}",
        f"{prefix}    natural language piece: {parsed['a']}",
        f"{prefix}    reason: {parsed['b']}",
    ]
    children = node["children"]
    for i, child in enumerate(children):
        is_last = i == len(children) - 1
        branch = "└── " if is_last else "├── "
        child_prefix = prefix + "    " + ("    " if is_last else "│   ")
        child_lines = render_decomposition_tree(child, parsed_by_id, child_prefix)
        lines.append(f"{prefix}    {branch}{child_lines[0]}")
        lines.extend(child_lines[1:])
    return lines


def walk_tree(client, model, node, nl_piece, operator_context, max_retries, parsed_by_id):
    """Visits *operator* nodes only, storing each one's parsed a/b/c/d in
    parsed_by_id (keyed by node id) instead of building a flat block list --
    render_decomposition_tree() re-walks the same tree afterward to lay these
    out with proper nesting. Signal/leaf nodes never get an LLM call: their
    piece is already sitting verbatim in their parent's (c)/(d)."""
    parsed = call_node_llm(client, model, nl_piece, node, operator_context, max_retries)
    parsed_by_id[node["id"]] = parsed
    binary = len(node["children"]) == 2

    first_child = node["children"][0]
    if first_child["type"] == "operator":
        walk_tree(client, model, first_child, parsed["c"], operator_context, max_retries, parsed_by_id)
    if binary:
        second_child = node["children"][1]
        if second_child["type"] == "operator":
            walk_tree(client, model, second_child, parsed["d"], operator_context, max_retries, parsed_by_id)


def generate_explanation(client, model, assertion, explanation, operator_context, max_retries):
    """Returns (output_text, used_dfs_tree). When the deep tree parses, walks
    it node by node (many small calls); otherwise falls back to a single call
    with the property-layer/plain Fig. 12 prompt, as generate_prompt_guided_
    explanation.py already does."""
    try:
        root = build_operator_signal_graph(assertion)
    except ValueError:
        prompt, _ = build_fallback_prompt(assertion, explanation, operator_context)
        last_error = None
        for attempt in range(max_retries):
            try:
                completion = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": NODE_SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                )
                return completion.choices[0].message.content.strip(), False
            except Exception as error:
                last_error = error
                wait_seconds = 2 ** attempt
                print(f"  attempt {attempt + 1}/{max_retries} failed ({error}); retrying in {wait_seconds}s")
                time.sleep(wait_seconds)
        raise RuntimeError(f"Failed to generate fallback explanation after {max_retries} attempts") from last_error

    if root["type"] != "operator":
        # Degenerate case: the whole property is a bare signal/sequence with
        # no operator at all (e.g. "assert property (@(posedge clk) req);").
        # There's no operator node to explain, so there's nothing to walk.
        return explanation, True

    parsed_by_id = {}
    walk_tree(client, model, root, explanation, operator_context, max_retries, parsed_by_id)

    decomposition = "\n".join(render_decomposition_tree(root, parsed_by_id))
    merge = "\n".join(render_merge_tree(root))
    output_text = (
        "***nl-decomposition tree***\n"
        f"{decomposition}\n\n"
        "According to the above natural-language decomposition, we finally derive "
        "the SystemVerilog assertion as follows:\n\n"
        "***operator-merge-sva tree***\n"
        f"{merge}"
    )
    return output_text, True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="verilogFinetune/data/qwen_explanation.jsonl",
                         help="Dataset of {instruction: explanation, output: golden_sva} pairs")
    parser.add_argument("--output", default="verilogFinetune/data/qwen_dfs_explanation_new.jsonl",
                         help="Where to write the final JSON array of generated records")
    parser.add_argument("--checkpoint", default=None,
                         help="Line-delimited progress file (default: <output>.checkpoint.jsonl); "
                              "checkpointing is per whole record, not per node")
    parser.add_argument("--operators", default="operators.json",
                         help="Path to the SVA operator quick-reference table")
    parser.add_argument("--config", default="Src/Config.yml",
                         help="Path to the project config.yml containing Openai_API_Key")
    parser.add_argument("--model", default="o4-mini",
                         help="OpenAI model used to generate the DFS explanation")
    parser.add_argument("--start", type=int, default=0, help="Start index into the input dataset")
    parser.add_argument("--limit", type=int, default=None, help="Max number of records to process")
    parser.add_argument("--max-retries", type=int, default=5)
    args = parser.parse_args()

    checkpoint_path = args.checkpoint or f"{args.output}.checkpoint.jsonl"

    with open(args.config) as file:
        config = yaml.safe_load(file)
    client = OpenAI(api_key=config["Openai_API_Key"])

    operator_context = load_operator_context(args.operators)
    records = load_input_records(args.input)

    end = len(records) if args.limit is None else min(len(records), args.start + args.limit)
    indices = range(args.start, end)

    completed = load_checkpoint(checkpoint_path)
    print(f"Loaded {len(completed)} already-completed records from {checkpoint_path}")

    dfs_count = sum(1 for r in completed.values() if r.get("dfs_tree"))
    fallback_count = len(completed) - dfs_count

    with open(checkpoint_path, "a") as checkpoint_file:
        for index in indices:
            if index in completed:
                continue

            record = records[index]
            explanation = record["instruction"]
            golden_sva = record["output"]

            print(f"[{index + 1}/{end}] generating DFS explanation "
                  f"(dfs-tree so far: {dfs_count}, fallback: {fallback_count})...")
            output_text, used_dfs_tree = generate_explanation(
                client, args.model, golden_sva, explanation, operator_context, args.max_retries
            )
            dfs_count += used_dfs_tree
            fallback_count += not used_dfs_tree

            result = {
                "index": index,
                "instruction": explanation,
                "input": record.get("input", ""),
                "output": output_text,
                "dfs_tree": used_dfs_tree,
            }
            checkpoint_file.write(json.dumps(result) + "\n")
            checkpoint_file.flush()
            completed[index] = result

    final_records = [
        {
            "instruction": completed[i]["instruction"],
            "input": completed[i]["input"],
            "output": completed[i]["output"],
        }
        for i in sorted(completed)
        if args.start <= i < end or (args.limit is None and i in completed)
    ]

    with open(args.output, "w") as file:
        for record in final_records:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Wrote {len(final_records)} records (one per line) to {args.output}")
    print(f"  dfs-tree-guided: {dfs_count}, fallback (property-layer or plain Fig. 12): {fallback_count}")


if __name__ == "__main__":
    main()
