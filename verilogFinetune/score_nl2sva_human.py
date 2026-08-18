"""
Scores an nl2sva_human-shaped LMResult CSV (as produced by
run_codev_sva_ol_dfs_eval.py or Src/MultiRoundPromptwithOperatorsExplanation/
run_rag_on_fveval_benchmarks.py) against its golden ref_solution column via
real Cadence JasperGold equivalence checking, reusing jasper_equiv_check.py
(which itself reuses the FVEval harness's own tcl script + pec.tcle macro,
without importing the harness's broken evaluation.py/config.py/saver.py
import chain -- see jasper_equiv_check.py's own docstring).

Metric definitions mirror NL2SVAHumanEvaluator.calculate_jg_metric in spirit,
for direct comparability with the paper's own SC/FM terminology, but `syntax`
is determined independently now, not inferred from prop_eq_checker's output:
    syntax        = 1 iff the LM's own candidate, embedded directly in the
                    REAL testbench, elaborates cleanly on its own (see
                    jasper_direct_equiv_check.check_sva_elaboration) --
                    decoupled entirely from prop_eq_checker's equivalence
                    check. Confirmed live (FVEval-NL2SVA-Human-0,
                    2026-08-11) that inferring syntax from whether
                    prop_eq_checker's own elaboration succeeded was
                    unreliable in BOTH directions: (a) real elaboration
                    failures (undeclared identifiers, illegal indexing,
                    etc.) routinely don't contain the literal string
                    "syntax error" prop_eq_checker's old regex looked for,
                    so they fell through and got counted as functional
                    mismatches instead of syntax failures; and (b)
                    prop_eq_checker's own wrapper module only declares
                    ports for the row's authorized SIGNAL_LIST, so a
                    candidate referencing a signal that's genuinely real in
                    the testbench but outside that authorized list reads as
                    "not declared" INSIDE prop_eq_checker even though it
                    elaborates perfectly well against the real testbench --
                    a signal-authorization issue, not a syntax one. A
                    direct, independent elaboration check against the real
                    testbench resolves both.
    functionality = 1 only on a "Full equivalence" result (from
                    prop_eq_checker, only run at all once syntax=1)
    func_relaxed  = 1 on "Full equivalence" OR a one-directional "implies" match

Body extraction is more general than the harness's own naive
`.split("tb_reset)")[-1]` (which only recognizes that one literal disable
signal name): it strips the assert/label wrapper, the clocking event, and a
`disable iff (...)` clause independently of which order they appear in --
deliberately not "fixing" a misordered clause, since a genuinely malformed
LM response should surface as a real syntax error here, not be silently
normalized away.

Requires `jg` on PATH and CDS_LIC_FILE set. Usage:
    python3 verilogFinetune/score_nl2sva_human.py \\
        --input Results/fveval_rag_outputs/nl2sva_human_gpt-4o_dynamicrag.csv \\
        --output Results/fveval_rag_outputs/nl2sva_human_gpt-4o_dynamicrag_jgscore.csv
"""
import argparse
import concurrent.futures
import csv
import os
import re
import sys
import threading

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jasper_equiv_check import run_equivalence_check
from jasper_direct_equiv_check import check_sva_elaboration

_LABEL_RE = re.compile(r'^\w+\s*:\s*(?=assert)')
_ASSERT_PROPERTY_RE = re.compile(r'^assert\s+property\s*\(')
_ASSERT_RE = re.compile(r'^assert\s*\(')
_CLOCK_RE = re.compile(r'@\s*\(\s*(?:pos|neg)edge\s+\w+\s*\)')
_DISABLE_RE = re.compile(r'disable\s+iff\s*\([^)]*\)')


def extract_property_body(assertion_text):
    text = assertion_text.strip().replace("\n", "")
    text = _LABEL_RE.sub('', text)
    text = _ASSERT_PROPERTY_RE.sub('', text)
    text = _ASSERT_RE.sub('', text)
    text = _CLOCK_RE.sub('', text)
    text = _DISABLE_RE.sub('', text)
    # re.split, not text.split(');'): a literal ');' substring match misses
    # the closing ") ;" (space before the semicolon) some LM responses use
    # -- confirmed leaving a dangling, unbalanced ")" + ";" in the extracted
    # body, which JasperGold then reports as a real syntax error even
    # though the original response was well-formed.
    text = re.split(r'\)\s*;', text)[0]
    return text.strip()


def parse_code_response(text):
    if "```systemverilog" in text:
        text = text.split("```systemverilog")[-1]
    if "```" in text:
        text = text.split("```")[0]
    return text.strip()


def build_signal_list(user_prompt, output_tb):
    signal_list = re.findall(r"'([^'\s]+)'", user_prompt)
    params = re.findall(r'\b(parameter|localparam)\s+(int\s+|real\s+|bit\s+|\[[^]]+\]\s*)?(\w+)', output_tb)
    signal_list += [m[2] for m in params]
    return signal_list


def calculate_jg_metric(jg_output):
    """Derives functionality/func_relaxed from prop_eq_checker's equivalence
    output ONLY -- callers must already know syntax=1 before calling this
    (see score_row: check_sva_elaboration is the independent, authoritative
    syntax verdict now, not anything inferred from this output)."""
    full_equiv = re.search(r"Full equivalence", jg_output)
    partial_equiv = re.search(r"implies", jg_output)
    if not full_equiv:
        if not partial_equiv:
            return {"functionality": 0.0, "func_relaxed": 0.0}
        return {"functionality": 0.0, "func_relaxed": 1.0}
    return {"functionality": 1.0, "func_relaxed": 1.0}


def score_row(row_index, total, row, sv_dir, timeout, disable_signal="tb_reset"):
    """One row's worth of main()'s per-row body, factored out unchanged so
    it can run inside a thread pool (see main()'s ThreadPoolExecutor loop).
    subprocess.run (inside run_equivalence_check) releases the GIL while jg
    is running, so threads parallelize this despite the GIL -- but unlike
    generation's API calls, each concurrent invocation here is a real
    JasperGold session; keep --workers within your JG license's concurrent-
    session limit. sv_dir is shared but each call writes to its own
    {experiment_id}_{task_id}.sva / jg/{experiment_id}_{task_id}/ subpath
    (see jasper_equiv_check.run_equivalence_check), so concurrent calls for
    different rows don't collide on files.

    Returns the metric dict (with task_id/jg_output_tail merged in)."""
    task_id = row["task_id"]
    print(f"[{row_index + 1}/{total}] task_id={task_id} ...")

    lm_sva = parse_code_response(row["response"])
    lm_body = extract_property_body(lm_sva)
    ref_body = extract_property_body(row["ref_solution"])
    # nl2sva_human_verified rows carry an authoritative signal list
    # (signals_for_validity, comma-joined) written by
    # run_rag_on_fveval_benchmarks.py -- prefer it over the regex
    # heuristic below when present. .get(): CSVs produced before
    # this column existed won't have it at all.
    signals_col = (row.get("signals_for_validity") or "").strip()
    if signals_col:
        signal_list = signals_col.split(",")
    else:
        signal_list = build_signal_list(row["user_prompt"], row["output_tb"])
    # dict.fromkeys(...): dedupe while preserving order. Duplicate
    # entries make JG's prop_eq_checker wrapper module reject the
    # whole check ("ANSI port 'X' cannot be redeclared") -- an
    # elaboration error that reads indistinguishably from a genuine
    # functional mismatch to calculate_jg_metric below. Defensive
    # here too (not just at the writer in
    # run_rag_on_fveval_benchmarks.py) so already-generated CSVs
    # with a duplicated signals_for_validity column still score
    # correctly without needing to re-run generation.
    signal_list = list(dict.fromkeys(signal_list))

    # Authoritative, independent syntax verdict: does the LM's OWN candidate
    # elaborate against the REAL testbench, with no SIGNAL_LIST restriction?
    # See module docstring for why this must be decoupled from
    # prop_eq_checker's own elaboration/equivalence output.
    try:
        syntax_ok, elab_output = check_sva_elaboration(
            testbench=row["design_rtl"], sva_expression=lm_body, sv_dir=sv_dir,
            experiment_id=row["experiment_id"], task_id=f"{task_id}_synccheck", timeout=timeout,
            disable_signal=disable_signal,
        )
    except Exception as error:
        syntax_ok, elab_output = False, f"EXCEPTION: {error}"

    if not syntax_ok:
        # Doesn't even elaborate on its own -- no point spending a second,
        # more expensive JG session on prop_eq_checker's equivalence check;
        # it can only fail too (or, worse, "succeed" for the wrong reason).
        metric = {"syntax": 0.0, "functionality": 0.0, "func_relaxed": 0.0}
        metric["task_id"] = task_id
        metric["jg_output_tail"] = elab_output[-500:]
        print(f"    syntax=0.0 functionality=0.0 func_relaxed=0.0 (real elaboration failure)")
        return metric

    try:
        _, jg_output = run_equivalence_check(
            testbench=row["design_rtl"],
            lm_assertion=lm_body,
            ref_assertion=ref_body,
            signal_list=signal_list,
            sv_dir=sv_dir,
            experiment_id=row["experiment_id"],
            task_id=task_id,
            timeout=timeout,
        )
    except Exception as error:
        jg_output = f"EXCEPTION: {error}"

    metric = calculate_jg_metric(jg_output)
    metric["syntax"] = 1.0
    metric["task_id"] = task_id
    metric["jg_output_tail"] = jg_output[-500:]
    print(f"    syntax={metric['syntax']} functionality={metric['functionality']} func_relaxed={metric['func_relaxed']}")
    return metric


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--sv-dir", default=None, help="Defaults to <output>.jgtmp")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--timeout", type=int, default=90)
    parser.add_argument("--workers", type=int, default=4,
                         help="Number of rows scored concurrently via a thread pool -- each is a "
                              "real JasperGold subprocess (jg), so this maps directly to concurrent "
                              "JG sessions. Keep it within your license's concurrent-session limit; "
                              "when unsure, start low (this default is deliberately conservative, "
                              "lower than generation's --workers default of 6) and raise it once you "
                              "confirm your license tolerates it. Set to 1 for the old fully-"
                              "sequential behavior.")
    parser.add_argument("--disable-signal", default="tb_reset",
                         help="The testbench's disable iff signal, used by the independent syntax "
                              "elaboration check (check_sva_elaboration). Default 'tb_reset' matches "
                              "every nl2sva_human(_verified) row. Pass an empty string for testbenches "
                              "with no reset signal at all (e.g. nl2sva_machine_verified).")
    args = parser.parse_args()

    disable_signal = args.disable_signal or None

    sv_dir = args.sv_dir or f"{args.output}.jgtmp"
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    with open(args.input, newline="") as file:
        rows = list(csv.DictReader(file))
    if args.limit is not None:
        rows = rows[: args.limit]

    fieldnames = ["task_id", "syntax", "functionality", "func_relaxed", "jg_output_tail"]
    results = []
    write_lock = threading.Lock()
    with open(args.output, "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=fieldnames)
        writer.writeheader()

        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(score_row, row_index, len(rows), row, sv_dir, args.timeout, disable_signal)
                for row_index, row in enumerate(rows)
            ]
            for future in concurrent.futures.as_completed(futures):
                metric = future.result()
                with write_lock:
                    writer.writerow(metric)
                    out_file.flush()
                results.append(metric)

    n = len(results)
    if n:
        syntax_rate = sum(r["syntax"] for r in results) / n
        func_rate = sum(r["functionality"] for r in results) / n
        relaxed_rate = sum(r["func_relaxed"] for r in results) / n
        print()
        print(f"n={n}")
        print(f"Syntax Correctness (SC): {sum(r['syntax'] for r in results):.0f}/{n} ({syntax_rate:.1%})")
        print(f"Functionality Match (FM, strict): {sum(r['functionality'] for r in results):.0f}/{n} ({func_rate:.1%})")
        print(f"Functionality Match (relaxed, incl. one-directional implies): {sum(r['func_relaxed'] for r in results):.0f}/{n} ({relaxed_rate:.1%})")
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
