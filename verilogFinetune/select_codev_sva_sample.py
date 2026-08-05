"""
Stage 1 of the CodeV-SVA OL-NL + two-part decomposition pipeline (see the plan
in /Users/fch/.claude/plans/polished-chasing-wreath.md).

Scans CodeV-SVA-dataset-training-83K.jsonl (83,195 chat-format {messages:
[system, user, assistant]} records) and selects a fixed-seed random sample of
--sample-size records whose final SVA:
  1. is the only ```systemverilog``` block after the assistant's </think> tag,
  2. contains exactly one `assert property` (the DFS pipeline decomposes one
     property at a time; multi-property answers are ~0.3% of the corpus and
     are simplest to exclude rather than special-case), and
  3. is parseable by sva_graph.build_operator_signal_graph() -- the deep
     operator/signal tree the rest of the pipeline (Part 1 decomposition,
     Part 2 symbolic derivation) is built on.

Writes the selected line indices (plus a few extracted fields already needed
downstream, so later stages don't have to re-scan all 83K lines) to
--output as a single JSON file.

Usage:
    python verilogFinetune/select_codev_sva_sample.py \\
        --input verilogFinetune/data/CodeV-SVA-dataset-training-83K.jsonl \\
        --output verilogFinetune/data/codev_sva_5000_sample.json \\
        --sample-size 5000
"""
import argparse
import json
import os
import random
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sva_graph import build_operator_signal_graph

CODE_RE = re.compile(r"```systemverilog\s*(.*?)```", re.DOTALL)


def extract_final_sva(assistant_content):
    """Returns the final SVA text, or None if it can't be cleanly extracted."""
    final_part = (
        assistant_content.split("</think>", 1)[1]
        if "</think>" in assistant_content
        else assistant_content
    )
    match = CODE_RE.search(final_part)
    if not match:
        return None
    sva = match.group(1).strip()
    if "assert property" not in sva:
        return None
    return sva


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="verilogFinetune/data/CodeV-SVA-dataset-training-83K.jsonl")
    parser.add_argument("--output", default="verilogFinetune/data/codev_sva_5000_sample.json")
    parser.add_argument("--sample-size", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Scanning {args.input} ...")
    eligible = []  # list of {"index": int, "sva": str}
    total = 0
    no_sva = 0
    multi_property = 0
    parse_fail = 0
    parse_fail_reasons = {}

    with open(args.input) as file:
        for index, line in enumerate(file):
            total += 1
            if total % 10000 == 0:
                print(f"  ...{total} scanned, {len(eligible)} eligible so far")
            record = json.loads(line)
            assistant_content = record["messages"][2]["content"]
            sva = extract_final_sva(assistant_content)
            if sva is None:
                no_sva += 1
                continue
            if len(re.findall(r"\bassert\s+property\b", sva)) != 1:
                multi_property += 1
                continue
            try:
                build_operator_signal_graph(sva)
            except Exception as error:
                parse_fail += 1
                reason = f"{type(error).__name__}: {str(error)[:80]}"
                parse_fail_reasons[reason] = parse_fail_reasons.get(reason, 0) + 1
                continue
            eligible.append({"index": index, "sva": sva})

    print()
    print(f"total records:        {total}")
    print(f"no parseable sva block: {no_sva}")
    print(f"multi-property answers: {multi_property}")
    print(f"sva_graph parse failed: {parse_fail}")
    print(f"eligible:              {len(eligible)} ({len(eligible) / total:.1%})")
    if parse_fail_reasons:
        print("parse failure reasons (top 10):")
        for reason, count in sorted(parse_fail_reasons.items(), key=lambda x: -x[1])[:10]:
            print(f"  {count:5d}  {reason}")

    if len(eligible) < args.sample_size:
        raise RuntimeError(
            f"Only {len(eligible)} eligible records found, need {args.sample_size}"
        )

    random.seed(args.seed)
    sample = random.sample(eligible, args.sample_size)
    sample.sort(key=lambda r: r["index"])

    with open(args.output, "w") as file:
        json.dump(
            {
                "source": args.input,
                "seed": args.seed,
                "sample_size": args.sample_size,
                "eligible_pool_size": len(eligible),
                "records": sample,
            },
            file,
        )

    print()
    print(f"Wrote {len(sample)} sampled records to {args.output}")


if __name__ == "__main__":
    main()
