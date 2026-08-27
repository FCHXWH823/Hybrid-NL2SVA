"""Convert a generation run's slim *_sva.jsonl into the LMResult CSV that
score_nl2sva_human.py consumes.

The scorer reads an nl2sva_human-shaped LMResult CSV -- the format
run_rag_on_fveval_benchmarks.py emits -- but raw-model sweeps
(generate_vllm_fveval.py) write {"name", "sva"} JSONL instead. Everything the
scorer needs beyond the assertion itself (golden reference, testbench, the
signal whitelist used for the validity check) already lives in the benchmark
file, so this joins the two by task name rather than asking the generation
side to duplicate benchmark fields it never used.

This step previously existed only as ad-hoc commands run by hand on the
scoring host, which is why the committed *_lmresult.csv files had no script
behind them. Scoring a new model meant reconstructing the column mapping from
an existing CSV. It is a script now so the generation host can produce
scorer-ready input directly, and the two hosts stay in sync through git.

The two splits disagree on how the signal whitelist is stored: human carries
`signals_for_validity` as a JSON list, machine carries `signal_list` as an
already-comma-joined string. Both normalize to the same comma-joined CSV cell
-- joining the string form character-by-character is the obvious trap here.

Usage:
    python3 verilogFinetune/sva_jsonl_to_lmresult.py \
        --sva      verilogFinetune/data/vllm_official/qwen3base8b_human_sva.jsonl \
        --benchmark Evaluation/FVEval-Verified/fveval_nl2sva_human.jsonl \
        --model-name qwen3base8b \
        --output   verilogFinetune/data/vllm_official/qwen3base8b_human_sva_lmresult.csv
"""

import argparse
import csv
import re
import json
import os
import sys

FIELDNAMES = [
    "experiment_id", "task_id", "model_name", "response", "ref_solution",
    "design_rtl", "output_tb", "user_prompt", "cot_response",
    "signals_for_validity",
]


def bare_name(signal):
    """Drop a leading bit-range declaration: "[3:0] sig_A" -> "sig_A".

    Multi-bit signals are stored with their width in both splits, but the
    scorer matches these against identifiers parsed out of the assertion,
    where no width appears. Leaving the prefix on makes every multi-bit
    signal fail the whitelist check -- 34 of 73 human rows and 198 of 283
    machine rows carry one.
    """
    return re.sub(r"^\s*\[[^\]]*\]\s*", "", signal).strip()


def signal_whitelist(record):
    """Comma-joined signal names, from whichever field this split uses."""
    sfv = record.get("signals_for_validity")
    if sfv is None:
        # machine: already "sig_B,[3:0] sig_A". Joining the string form would
        # splice a comma between every character.
        sfv = record.get("signal_list")
        if sfv is None:
            return ""
    if isinstance(sfv, str):
        sfv = sfv.split(",")
    return ",".join(bare_name(s) for s in sfv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sva", required=True, help="*_sva.jsonl from generate_vllm_fveval.py")
    ap.add_argument("--benchmark", required=True, help="the FVEval-Verified jsonl it was generated from")
    ap.add_argument("--model-name", required=True, help="short tag, e.g. qwen3base8b")
    ap.add_argument("--experiment-id", default=None,
                    help="defaults to the output basename minus _sva_lmresult.csv")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    bench = {}
    for line in open(args.benchmark):
        if line.strip():
            r = json.loads(line)
            bench[r["name"]] = r

    rows = [json.loads(l) for l in open(args.sva) if l.strip()]
    experiment_id = args.experiment_id or os.path.basename(args.output).replace("_sva_lmresult.csv", "")

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    missing = 0
    empty = 0
    with open(args.output, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDNAMES)
        w.writeheader()
        for row in rows:
            name = row["name"]
            rec = bench.get(name)
            if rec is None:
                missing += 1
                continue
            sva = row.get("sva") or ""
            if not sva.strip():
                # A failed extraction (e.g. a run that never closed </think>).
                # Kept as a row so the scorer counts it as the failure it is,
                # rather than silently shrinking the denominator.
                empty += 1
            w.writerow({
                "experiment_id": experiment_id,
                "task_id": name,
                "model_name": args.model_name,
                "response": "```systemverilog\n%s\n```" % sva,
                "ref_solution": rec.get("ground_truth", ""),
                "design_rtl": rec.get("testbench", ""),
                "output_tb": rec.get("testbench", ""),
                "user_prompt": rec.get("problem", ""),
                "cot_response": "",
                "signals_for_validity": signal_whitelist(rec),
            })

    print("rows written    : %d" % (len(rows) - missing))
    if empty:
        print("empty SVA cells : %d (kept -- they must score as failures)" % empty)
    if missing:
        print("NOT in benchmark: %d (dropped)" % missing, file=sys.stderr)
    print("wrote           : %s" % args.output)


if __name__ == "__main__":
    main()
