"""Scores the AssertionForge-UART pilot's generated SVAs with JasperGold,
matching QiMeng-CodeV-SVA's Table 5 methodology: #SVA (rows the pipeline
returned a candidate for) / #SynC (elaborates) / #Proven (formally proven
under the real UART design -- no golden reference needed).

Run from the Hybrid-NL2SVA repo root, after run_uart_nl2sva.py:
    python3 end2end_evaluation/score_uart_assertionforge.py --workers 6
"""
import sys, os, csv, argparse, concurrent.futures, threading

sys.path.insert(0, "verilogFinetune")
from jasper_direct_equiv_check import check_sva_proven
from score_nl2sva_human import parse_code_response, extract_property_body

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(_HERE, "results", "assertionforge_uart_gpt-4o_dynamicrag.csv")
DEFAULT_OUTPUT_PATH = os.path.join(_HERE, "results", "assertionforge_uart_gpt-4o_dynamicrag_jgscore.csv")

ap = argparse.ArgumentParser()
ap.add_argument("--input", default=DEFAULT_INPUT_PATH)
ap.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--timeout", type=int, default=90)
ap.add_argument("--limit", type=int, default=None)
cli_args = ap.parse_args()

SV_DIR = f"{cli_args.output}.jgtmp"

with open(cli_args.input) as f:
    rows = list(csv.DictReader(f))
if cli_args.limit:
    rows = rows[: cli_args.limit]
print(f"n={len(rows)}")


def score_one(i, row):
    task_id = row["task_id"]
    raw_testbench = row["output_tb"]
    bare_sva = extract_property_body(parse_code_response(row["response"]))
    try:
        syntax_ok, proven, jg_output = check_sva_proven(
            raw_testbench, bare_sva, SV_DIR, experiment_id="uart_proven",
            task_id=str(i), clock_signal="clock", disable_signal=None,
            timeout=cli_args.timeout,
        )
    except Exception as e:
        return task_id, {"task_id": task_id, "syntax": 0.0, "proven": 0.0, "jg_output_tail": f"EXC: {e}"}
    print(f"[{i+1}/{len(rows)}] {task_id}: syntax={syntax_ok} proven={proven}")
    status_lines = [l for l in jg_output.splitlines() if "PROVENSTATUS" in l or "ERROR" in l]
    return task_id, {
        "task_id": task_id,
        "syntax": 1.0 if syntax_ok else 0.0,
        "proven": 1.0 if proven else 0.0,
        "jg_output_tail": "\n".join(status_lines) or jg_output[-500:],
    }


results = []
write_lock = threading.Lock()
with concurrent.futures.ThreadPoolExecutor(max_workers=cli_args.workers) as executor:
    futures = [executor.submit(score_one, i, row) for i, row in enumerate(rows)]
    for future in concurrent.futures.as_completed(futures):
        task_id, result = future.result()
        with write_lock:
            results.append(result)

os.makedirs(os.path.dirname(cli_args.output) or ".", exist_ok=True)
with open(cli_args.output, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=["task_id", "syntax", "proven", "jg_output_tail"])
    writer.writeheader()
    for r in results:
        writer.writerow(r)

n = len(results)
n_svc = sum(1 for r in results if r["syntax"] == 1.0)
n_proven = sum(1 for r in results if r["proven"] == 1.0)
print(f"\n#SVA={n}  #SynC={n_svc} ({n_svc/n*100:.1f}%)  #Proven={n_proven} ({n_proven/n*100:.1f}%)")
print(f"Wrote {cli_args.output}")
