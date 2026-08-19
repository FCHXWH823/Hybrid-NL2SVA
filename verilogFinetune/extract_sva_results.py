"""Distil the full generation log down to just the final SVA per record.

generate_fveval_outputs.py writes everything -- prompts, the whole <think>
reasoning block, the completion -- which is what you want when debugging a run
but is ~100x larger than the answers themselves. This keeps only the extracted
assertion.

Records whose completion had no ```systemverilog fence are written with
sva = null rather than skipped, so the output stays index-aligned with the
input and a missing answer is visible instead of silently absent.
"""

import argparse
import json
import os


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", default="data/fveval_human_qwen3_8b_think.jsonl")
    parser.add_argument("--output", default="data/fveval_human_qwen3_8b_think_sva.jsonl")
    parser.add_argument("--with-ground-truth", action="store_true",
                        help="also carry ground_truth through (needed for scoring)")
    args = parser.parse_args()

    rows = []
    with open(args.input) as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))

    missing = 0
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as out:
        for r in rows:
            sva = r.get("extracted_assertion")
            if not sva:
                missing += 1
            item = {"name": r.get("name"), "sva": sva}
            if args.with_ground_truth:
                item["ground_truth"] = r.get("ground_truth")
            out.write(json.dumps(item, ensure_ascii=False) + "\n")

    src = os.path.getsize(args.input)
    dst = os.path.getsize(args.output)
    print("read   : %s (%.1f MB, %d records)" % (args.input, src / 1024 ** 2, len(rows)))
    print("wrote  : %s (%.1f KB)" % (args.output, dst / 1024))
    print("shrank : %.0fx" % (src / dst if dst else 0))
    if missing:
        print("WARNING: %d record(s) had no assertion; written as null" % missing)


if __name__ == "__main__":
    main()
