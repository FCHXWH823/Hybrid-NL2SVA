"""Render the FVEval-Verified generation results as a plain-text report.

Side-by-side per record: the problem, the reference assertion, and what each
model produced. Records where a model never emitted an assertion are shown as
<NO ASSERTION ...> with the reason, rather than left blank -- a missing answer
and an empty answer are different failures.
"""

import argparse
import json
import os
from collections import Counter

WIDTH = 100


def load(path):
    if not path or not os.path.isfile(path):
        return {}
    out = {}
    with open(path) as f:
        for line in f:
            if line.strip():
                r = json.loads(line)
                out[r.get("name")] = r
    return out


def idiom(s):
    s = s or ""
    if "!== 1'b1" in s or "!= 1'b1" in s:
        return "negated-conjunction"
    if "|->" in s or "|=>" in s:
        return "implication"
    return "other"


def indent(text, pad="    "):
    if not text:
        return pad + "(none)"
    return "\n".join(pad + line for line in text.strip().splitlines())


def describe_failure(rec, cap):
    """Why there is no assertion -- truncation looks different from a model
    that stopped cleanly without emitting one."""
    completion = rec.get("completion", "") or ""
    if "</think>" not in completion:
        return "<NO ASSERTION - reasoning never terminated; hit the %s-token cap>" % cap
    return "<NO ASSERTION - completed but emitted no ```systemverilog fence>"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ours", default="data/fveval_human_qwen3_8b_think.jsonl")
    parser.add_argument("--ours-label", default="qwen3-8b-codev-sva-ol-dfs-think (ours)")
    parser.add_argument("--baseline", default="data/fveval_human_codev_sva_8b.jsonl")
    parser.add_argument("--baseline-label", default="wyt2000/CodeV-SVA-8B (baseline)")
    parser.add_argument("--ours-cap", default="2048")
    parser.add_argument("--baseline-cap", default="8192")
    parser.add_argument("--output", default="data/fveval_human_results.txt")
    args = parser.parse_args()

    ours = load(args.ours)
    base = load(args.baseline)
    names = list(ours.keys()) or list(base.keys())

    lines = []
    add = lines.append

    add("=" * WIDTH)
    add("FVEval-Verified / nl2sva_human -- SVA generation results")
    add("=" * WIDTH)
    add("")
    add("benchmark : Evaluation/FVEval-Verified/fveval_nl2sva_human.jsonl (%d records)" % len(names))
    add("model A   : %s   [max_new_tokens=%s]" % (args.ours_label, args.ours_cap))
    add("model B   : %s   [max_new_tokens=%s]" % (args.baseline_label, args.baseline_cap))
    add("decoding  : greedy (do_sample=False), qwen3 chat template, enable_thinking=True")
    add("")

    def stats(d, cap):
        n = len(names)
        got = sum(1 for k in names if (d.get(k) or {}).get("extracted_assertion"))
        ex = sum(1 for k in names if (d.get(k) or {}).get("exact_match"))
        tb = sum(1 for k in names
                 if "disable iff (tb_reset)" in ((d.get(k) or {}).get("extracted_assertion") or ""))
        return got, ex, tb, n

    ga, ea, ta, n = stats(ours, args.ours_cap)
    gb, eb, tb_, _ = stats(base, args.baseline_cap)

    add("-" * WIDTH)
    add("SUMMARY")
    add("-" * WIDTH)
    add("%-34s %-22s %-22s" % ("", "A (ours)", "B (baseline)"))
    add("%-34s %-22s %-22s" % ("produced a parseable assertion",
                               "%d/%d (%.1f%%)" % (ga, n, 100.0 * ga / n),
                               "%d/%d (%.1f%%)" % (gb, n, 100.0 * gb / n)))
    add("%-34s %-22s %-22s" % ("exact string match vs reference", str(ea), str(eb)))
    add("%-34s %-22s %-22s" % ("used disable iff (tb_reset)", str(ta), str(tb_)))
    add("")

    gt_idiom = Counter(idiom((ours.get(k) or base.get(k) or {}).get("ground_truth")) for k in names)
    a_idiom = Counter(idiom((ours.get(k) or {}).get("extracted_assertion")) for k in names
                      if (ours.get(k) or {}).get("extracted_assertion"))
    b_idiom = Counter(idiom((base.get(k) or {}).get("extracted_assertion")) for k in names
                      if (base.get(k) or {}).get("extracted_assertion"))
    add("idiom distribution")
    add("%-34s %-10s %-10s %-10s" % ("", "reference", "A", "B"))
    for key in ("negated-conjunction", "implication", "other"):
        add("%-34s %-10d %-10d %-10d" % (key, gt_idiom[key], a_idiom[key], b_idiom[key]))
    add("")
    add("NOTE: exact string match is NOT an accuracy score. The references use the")
    add("negated-conjunction idiom in %d of %d cases; both models emit implications" % (
        gt_idiom['negated-conjunction'], n))
    add("instead. `A |-> B` and `(A && !B) !== 1'b1` are logically equivalent but never")
    add("compare equal as strings. A real score requires JasperGold equivalence")
    add("checking (see jasper_equiv_check.py).")
    add("")

    add("=" * WIDTH)
    add("PER-RECORD RESULTS")
    add("=" * WIDTH)
    for i, name in enumerate(names, 1):
        ra = ours.get(name) or {}
        rb = base.get(name) or {}
        ref = ra.get("ground_truth") or rb.get("ground_truth")
        add("")
        add("-" * WIDTH)
        add("[%d/%d] %s" % (i, len(names), name))
        add("-" * WIDTH)
        add("PROBLEM:")
        add(indent(ra.get("problem") or rb.get("problem")))
        add("")
        add("REFERENCE:")
        add(indent(ref))
        add("")
        add("A  %s:" % args.ours_label)
        add(indent(ra.get("extracted_assertion") or describe_failure(ra, args.ours_cap)))
        add("   [exact_match=%s]" % bool(ra.get("exact_match")))
        add("")
        add("B  %s:" % args.baseline_label)
        add(indent(rb.get("extracted_assertion") or describe_failure(rb, args.baseline_cap)))
        add("   [exact_match=%s]" % bool(rb.get("exact_match")))

    add("")
    add("=" * WIDTH)
    add("END")
    add("=" * WIDTH)

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w") as f:
        f.write("\n".join(lines) + "\n")
    print("wrote %s (%d lines, %.1f KB)"
          % (args.output, len(lines), os.path.getsize(args.output) / 1024))


if __name__ == "__main__":
    main()
