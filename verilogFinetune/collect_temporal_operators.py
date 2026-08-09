"""
Scans CodeV-SVA-dataset-training-83K.jsonl (83,195 chat-format records, the
same corpus and eligibility rule as select_codev_sva_sample.py Stage 1) and
summarizes every SVA operator actually seen in the data, to check completeness
of operators.json (currently 11 hand-curated entries) against real usage.

Two passes, since sva_graph.py's structural walk isn't the whole story:

1. Structural: for every SVA whose final ```systemverilog block parses via
   sva_graph.build_operator_signal_graph(), walk every operator node and
   count its label (e.g. "|->", "$past", "##2"). This is precise -- each hit
   comes with the exact code span it was extracted from -- but sva_graph.py
   only has explicit rules for a subset of the grammar; anything it doesn't
   recognize (case properties, accept_on/reject_on/sync_*_on, cover/restrict/
   expect statements, ...) is silently folded into an opaque signal-leaf
   string rather than raised as an error, so those constructs are invisible
   to this pass -- they don't parse-fail, they just never become operator
   nodes. See sva_graph.py's own module docstring.

2. Keyword: a plain regex scan of the same raw SVA text for a curated list of
   SVA-distinctive keyword operators (the ones pass 1 is known to swallow),
   to catch what pass 1 structurally misses. This is approximate (counts
   textual occurrences, not confirmed operator usage) and is reported
   separately, not merged into the structural counts.

Operator tokens are split into "temporal" (SVA/sequence/property-layer:
##N, |->, |=>, $past, s_eventually, ...) vs "non-temporal" (plain Boolean/
arithmetic/relational/bitwise: ==, &&, +, ...) via an exclusion list of the
latter -- anything not in that list is treated as temporal, so previously
unknown temporal operators show up automatically instead of needing to be
anticipated in advance.

Usage:
    python verilogFinetune/collect_temporal_operators.py \\
        --input verilogFinetune/data/CodeV-SVA-dataset-training-83K.jsonl \\
        --output operators_discovered.json \\
        --existing operators.json
"""
import argparse
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "sva_tree"))
from select_codev_sva_sample import extract_final_sva
from sva_graph import build_operator_signal_graph, dfs_nodes

# Plain Boolean/arithmetic/relational/bitwise operators -- anything collected
# by the structural pass that is NOT in this set is treated as "temporal"
# (SVA/sequence/property-layer). Kept as an exclusion list, not an allow
# list, so operators this list's author didn't think of still surface.
NON_TEMPORAL_OPERATORS = {
    "==", "!=", "===", "!==", "==?", "!=?",
    "<", ">", "<=", ">=",
    "&&", "||", "!",
    "+", "-", "*", "/", "%", "**",
    "&", "|", "^", "~", "^~", "~^", "~&", "~|",
    "<<", ">>", "<<<", ">>>",
    "=", "+=", "-=",
}

# SVA-distinctive keyword operators known to NOT become their own operator
# node under sva_graph.py's current rule set (case/if-else properties,
# abort properties, verification-layer statements) -- scanned for as raw
# text in pass 2 rather than structurally. Word-boundary matched.
KEYWORD_SCAN_OPERATORS = [
    "accept_on", "reject_on", "sync_accept_on", "sync_reject_on",
    "cover property", "restrict property", "expect",
    "case",
]

CODE_RE = re.compile(r"```systemverilog\s*(.*?)```", re.DOTALL)


def classify(label):
    return "non_temporal" if label in NON_TEMPORAL_OPERATORS else "temporal"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="verilogFinetune/data/CodeV-SVA-dataset-training-83K.jsonl")
    parser.add_argument("--output", default="operators_discovered.json")
    parser.add_argument("--existing", default="operators.json",
                         help="Path to the current hand-curated operator table, for a coverage diff")
    parser.add_argument("--limit", type=int, default=None, help="Only scan the first N records (debugging)")
    args = parser.parse_args()

    with open(args.existing) as file:
        existing_operators = set(json.load(file).keys())

    total = 0
    no_sva = 0
    multi_property = 0
    parse_failed = 0
    parsed_ok = 0

    # label -> {"occurrences": int, "svas_containing": int, "example": str}
    temporal = {}
    non_temporal = {}
    # keyword -> raw text occurrence count (pass 2)
    keyword_hits = {kw: 0 for kw in KEYWORD_SCAN_OPERATORS}

    print(f"Scanning {args.input} ...")
    with open(args.input) as file:
        for index, line in enumerate(file):
            if args.limit is not None and index >= args.limit:
                break
            total += 1
            if total % 10000 == 0:
                print(f"  ...{total} scanned")

            record = json.loads(line)
            assistant_content = record["messages"][2]["content"]
            sva = extract_final_sva(assistant_content)
            if sva is None:
                no_sva += 1
                continue
            if len(re.findall(r"\bassert\s+property\b", sva)) != 1:
                multi_property += 1
                continue

            for keyword in KEYWORD_SCAN_OPERATORS:
                if re.search(rf"\b{re.escape(keyword)}\b", sva):
                    keyword_hits[keyword] += 1

            try:
                root = build_operator_signal_graph(sva)
            except Exception:
                parse_failed += 1
                continue
            parsed_ok += 1

            seen_this_sva = set()
            for node in dfs_nodes(root):
                if node["type"] != "operator":
                    continue
                label = node["label"]
                bucket = temporal if classify(label) == "temporal" else non_temporal
                entry = bucket.setdefault(
                    label, {"occurrences": 0, "svas_containing": 0, "example": node["code"]}
                )
                entry["occurrences"] += 1
                if label not in seen_this_sva:
                    entry["svas_containing"] += 1
                    seen_this_sva.add(label)
                # Prefer a shorter example code span when one turns up later,
                # for a more readable JSON.
                if len(node["code"]) < len(entry["example"]):
                    entry["example"] = node["code"]

    for label, entry in temporal.items():
        entry["already_in_operators_json"] = label in existing_operators
    for label, entry in non_temporal.items():
        entry["already_in_operators_json"] = label in existing_operators

    result = {
        "scan_summary": {
            "source": args.input,
            "total_records": total,
            "no_sva_block": no_sva,
            "multi_property": multi_property,
            "structural_parse_failed": parse_failed,
            "structural_parsed_ok": parsed_ok,
        },
        "temporal_operators": dict(sorted(temporal.items(), key=lambda kv: -kv[1]["occurrences"])),
        "non_temporal_operators_seen": dict(sorted(non_temporal.items(), key=lambda kv: -kv[1]["occurrences"])),
        "keyword_scan_temporal_hits": {k: v for k, v in keyword_hits.items() if v > 0},
        "keyword_scan_zero_hits": [k for k, v in keyword_hits.items() if v == 0],
    }

    with open(args.output, "w") as file:
        json.dump(result, file, indent=2)

    missing = [op for op, e in temporal.items() if not e["already_in_operators_json"]]
    print()
    print(f"total records:            {total}")
    print(f"no parseable sva block:   {no_sva}")
    print(f"multi-property answers:   {multi_property}")
    print(f"structural parse failed:  {parse_failed}")
    print(f"structural parsed ok:     {parsed_ok}")
    print()
    print(f"temporal operators found (structural): {len(temporal)}")
    print(f"  already in {args.existing}: {len(temporal) - len(missing)}")
    print(f"  missing from {args.existing}: {len(missing)}")
    if missing:
        print("  missing operators, by occurrence count:")
        for op in sorted(missing, key=lambda o: -temporal[o]["occurrences"]):
            print(f"    {op:20s} occurrences={temporal[op]['occurrences']:6d}  "
                  f"svas={temporal[op]['svas_containing']:6d}  example: {temporal[op]['example']}")
    if any(keyword_hits.values()):
        print()
        print("keyword-scan hits (constructs sva_graph.py doesn't structurally decompose):")
        for kw, count in sorted(keyword_hits.items(), key=lambda kv: -kv[1]):
            if count:
                print(f"    {kw:20s} raw_occurrences={count}")
    print()
    print(f"Wrote full results to {args.output}")


if __name__ == "__main__":
    main()
