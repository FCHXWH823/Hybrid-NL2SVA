"""
Synthesizes the prompt-guided explanation fine-tuning dataset (Section 3.2 / Fig. 12
of TCAD-Hybrid-NL2SVA).

For every (golden SVA, natural-language explanation) pair in an existing
instruction-tuning dataset (e.g. verilogFinetune/data/qwen_explanation.jsonl), this
script prompts an OpenAI reasoning model (o4-mini in the paper) with the exact
"Prompt-guided Explanation Generation Prompt" to produce a step-by-step recursive
derivation of the SVA from its explanation. The result is written in the same
{instruction, input, output} schema used by the other datasets in
verilogFinetune/data/, so it can be registered in data/dataset_info.json and
consumed by LLaMA-Factory as-is.

Usage:
    python verilogFinetune/generate_prompt_guided_explanation.py \\
        --input verilogFinetune/data/qwen_explanation.jsonl \\
        --output verilogFinetune/data/qwen_prompt_guided_explanation_new.jsonl

Re-running with the same --output resumes from the checkpoint file instead of
re-calling the API for entries already generated.

Structure-guided generation
----------------------------
Asking an LLM to both discover an SVA's recursive structure *and* narrate it in
one shot is exactly where it's least reliable, since LLMs aren't dependable SVA
parsers. So for each golden SVA, this script first tries to parse its real
property-layer structure deterministically with sva_parser.py (built on pyslang,
a real SystemVerilog frontend) and feeds that ground-truth structure into the
prompt as a hard constraint the model must follow, rather than asking it to
infer the structure from scratch. The Step 1-5 output format itself is
unchanged, so the fine-tuned model still learns to produce the same narrative
style even though it won't have a parsed tree available at inference time.
The parser covers ~85% of the corpus (see check_parser_coverage.py); whenever
it can't parse an assertion (unsupported construct, or malformed/truncated
source data), this script transparently falls back to the original
structure-free Fig. 12 prompt for that one record.
"""
import argparse
import json
import os
import sys
import time

import yaml
from openai import OpenAI

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sva_parser import UnsupportedSVAConstruct, parse_sva_property, render_parsed_sva

PROMPT_STEPS = """Step 1: Identify the top-level property operator:
• Operator: <chosen_operator>
• Reason: <why you chose it>

Step 2: Split the explanation into fragment(s):
• expression_1: <text>
• expression_2: <text> (omit if unary)
• Reason: <how the number of operands of selected operator guides your split>

Step 3: Process each fragment:
• According to the defined property expression types, determine if the fragment is a
property or a sequence:
    (1) If it corresponds to a property, write "Nested property:" and then
    recursively apply Step 1-4 to it.
    (2) Otherwise, translate it into an SVA sequence, e.g. "##1 req" or
    "$rose(req)".
• Reason: <why you chose those sequence operators>

Step 4: Combine the fragment expressions with the operator into the
property_expression:
• property_expression: <combined_expression>
• Reason: <how you form this expression>

Step 5: Wrap into the final assertion:
• assertion: assert property (<property_expression>);
"""

PROMPT_TEMPLATE = """You are given a SystemVerilog assertion and its natural-language explanation:
Assertion:
 {assertion}
Explanation:
 {explanation}
SVA Operator Context:
 {operator_context}
Simulate the recursive SystemVerilog Assertion construction process and show your
complete reasoning at each step. Do not output JSON—just list the steps in plain text:

""" + PROMPT_STEPS

PROMPT_TEMPLATE_STRUCTURED = """You are given a SystemVerilog assertion and its natural-language explanation:
Assertion:
 {assertion}
Explanation:
 {explanation}
SVA Operator Context:
 {operator_context}

The exact recursive structure of this assertion has already been determined by
directly parsing it (this is ground truth, not a guess). Follow it exactly: use
the given top-level operator, split the explanation into exactly as many
fragments as it has operands, and recurse only where the structure below shows
a nested property rather than a sequence leaf. Do not choose a different
operator or a different number of fragments, even if another structure seems
to fit the explanation better.

Parsed structure:
{parsed_structure}

Using ONLY the structure above, simulate the recursive SystemVerilog Assertion
construction process and show your complete reasoning at each step, aligning
each fragment of the natural-language explanation with the corresponding node
in the structure. Do not output JSON—just list the steps in plain text:

""" + PROMPT_STEPS

SYSTEM_PROMPT = (
    "You are a helpful bot that reconstructs SystemVerilog assertions step by step "
    "from their natural-language explanations, following the requested format exactly."
)


def load_operator_context(operators_path):
    with open(operators_path) as file:
        operators = json.load(file)
    lines = [f"{op}: {explanation}" for op, explanation in operators.items()]
    return "\n".join(lines)


def load_input_records(input_path):
    """Accepts either a JSON array of records or true line-delimited JSONL."""
    with open(input_path) as file:
        text = file.read()
    text = text.strip()
    if not text:
        return []
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def load_checkpoint(checkpoint_path):
    completed = {}
    if not os.path.exists(checkpoint_path):
        return completed
    with open(checkpoint_path) as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            completed[record["index"]] = record
    return completed


def build_prompt(assertion, explanation, operator_context):
    """Try to deterministically parse the assertion's structure; use the
    structure-guided template on success, falling back to the plain Fig. 12
    prompt if the parser can't handle this construct. Returns (prompt, used_structure)."""
    try:
        parsed = parse_sva_property(assertion)
    except UnsupportedSVAConstruct:
        parsed = None

    if parsed is not None:
        prompt = PROMPT_TEMPLATE_STRUCTURED.format(
            assertion=assertion,
            explanation=explanation,
            operator_context=operator_context,
            parsed_structure=render_parsed_sva(parsed),
        )
        return prompt, True

    prompt = PROMPT_TEMPLATE.format(
        assertion=assertion,
        explanation=explanation,
        operator_context=operator_context,
    )
    return prompt, False


def generate_explanation(client, model, assertion, explanation, operator_context, max_retries):
    prompt, used_structure = build_prompt(assertion, explanation, operator_context)
    last_error = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": prompt},
                ],
            )
            return completion.choices[0].message.content.strip(), used_structure
        except Exception as error:
            last_error = error
            wait_seconds = 2 ** attempt
            print(f"  attempt {attempt + 1}/{max_retries} failed ({error}); retrying in {wait_seconds}s")
            time.sleep(wait_seconds)
    raise RuntimeError(f"Failed to generate explanation after {max_retries} attempts") from last_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="verilogFinetune/data/qwen_explanation.jsonl",
                         help="Dataset of {instruction: explanation, output: golden_sva} pairs")
    parser.add_argument("--output", default="verilogFinetune/data/qwen_prompt_guided_explanation_new.jsonl",
                         help="Where to write the final JSON array of generated records")
    parser.add_argument("--checkpoint", default=None,
                         help="Line-delimited progress file (default: <output>.checkpoint.jsonl)")
    parser.add_argument("--operators", default="operators.json",
                         help="Path to the SVA operator quick-reference table")
    parser.add_argument("--config", default="Src/Config.yml",
                         help="Path to the project config.yml containing Openai_API_Key")
    parser.add_argument("--model", default="o4-mini",
                         help="OpenAI model used to generate the prompt-guided explanation")
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

    structured_count = sum(1 for r in completed.values() if r.get("structured"))
    fallback_count = len(completed) - structured_count

    with open(checkpoint_path, "a") as checkpoint_file:
        for index in indices:
            if index in completed:
                continue

            record = records[index]
            explanation = record["instruction"]
            golden_sva = record["output"]

            print(f"[{index + 1}/{end}] generating prompt-guided explanation "
                  f"(structured so far: {structured_count}, fallback: {fallback_count})...")
            output_text, used_structure = generate_explanation(
                client, args.model, golden_sva, explanation, operator_context, args.max_retries
            )
            structured_count += used_structure
            fallback_count += not used_structure

            result = {
                "index": index,
                "instruction": explanation,
                "input": record.get("input", ""),
                "output": output_text,
                "structured": used_structure,
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
    print(f"  structure-guided: {structured_count}, fallback (Fig. 12 only): {fallback_count}")


if __name__ == "__main__":
    main()
