"""
Scans a dataset of golden SVAs and reports how many sva_parser.py can parse,
broken down by failure reason. No API calls; useful for gauging how often
generate_prompt_guided_explanation.py will fall back to the structure-free
Fig. 12 prompt, and for spotting unsupported constructs worth adding support
for versus malformed/truncated entries in the source data itself.

Usage:
    python verilogFinetune/check_parser_coverage.py \\
        --input verilogFinetune/data/qwen_explanation.jsonl
"""
import argparse
import json
from collections import Counter

from sva_parser import parse_sva_property


def load_records(path):
    with open(path) as file:
        text = file.read().strip()
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return [json.loads(line) for line in text.splitlines() if line.strip()]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="verilogFinetune/data/qwen_explanation.jsonl")
    parser.add_argument("--show-failures", type=int, default=0,
                         help="Print this many example failing SVAs per reason")
    args = parser.parse_args()

    records = load_records(args.input)
    ok = 0
    reasons = Counter()
    examples = {}

    for record in records:
        sva = record["output"]
        try:
            parse_sva_property(sva)
            ok += 1
        except Exception as error:
            reason = str(error)
            reasons[reason] += 1
            examples.setdefault(reason, []).append(sva)

    total = len(records)
    print(f"total: {total}")
    print(f"parsed OK: {ok} ({ok / total:.1%})")
    print(f"failed:    {total - ok} ({(total - ok) / total:.1%})")
    print()
    print("failure reasons (most common first):")
    for reason, count in reasons.most_common():
        print(f"  {count:4d}  {reason}")
        if args.show_failures:
            for sva in examples[reason][: args.show_failures]:
                print(f"        - {sva}")


if __name__ == "__main__":
    main()
