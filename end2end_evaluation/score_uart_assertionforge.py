"""Scores the AssertionForge-UART pilot's generated SVAs with JasperGold,
matching QiMeng-CodeV-SVA's Table 5 methodology: #SVA (rows the pipeline
returned a candidate for) / #SynC (elaborates) / #Proven (formally proven
under the real UART design -- no golden reference needed).

Run from the Hybrid-NL2SVA repo root, after run_uart_nl2sva.py:
    python3 end2end_evaluation/score_uart_assertionforge.py --workers 6

Scope-aware integration (2026-08-31): before insertion/elaboration, every
bare SVA is passed through signal_scope.qualify_out_of_scope_references --
see that module's docstring and end2end_evaluation/README.md's "A second
gap" / scope-integration sections for the full diagnosis. This mechanically
rewrites a real-but-out-of-scope RTL identifier (e.g. `ce_16`, declared
inside uart_top/uart2bus_top's submodule, not uart2bus_top itself) into a
correctly-qualified hierarchical path (`uart1.ce_16`), and a real macro
name used bare (e.g. `MAIN_ADDR`) into a proper macro invocation
(`` `MAIN_ADDR ``). The combined testbench's own `` `define ``s are also
hoisted to the very front (Verilog's preprocessor is single-pass
top-to-bottom -- a macro `` `define ``d in uart_parser.v, concatenated
LAST in RTL_FILE_ORDER, isn't yet defined at the point earlier in the file
where the assertion is spliced in, even once correctly backtick-qualified,
unless hoisted). Neither fix touches prompt engineering, generation, or
AssertionForge's own NL-plan quality -- pure mechanical integration repair,
orthogonal to (and does not require adopting) the skip_signal_list_note=
False experiment. Validated live: 24/58 originally-syntax-failing rows that
reference a qualifiable name flip to syntax-OK from these two fixes alone."""
import sys, os, csv, re, argparse, concurrent.futures, threading

sys.path.insert(0, "verilogFinetune")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from jasper_direct_equiv_check import check_sva_proven
from score_nl2sva_human import parse_code_response, extract_property_body
from signal_scope import build_signal_scope_map, qualify_out_of_scope_references

_HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT_INPUT_PATH = os.path.join(_HERE, "results", "assertionforge_uart_gpt-4o_dynamicrag.csv")
DEFAULT_OUTPUT_PATH = os.path.join(_HERE, "results", "assertionforge_uart_gpt-4o_dynamicrag_jgscore.csv")
DEFAULT_UART_RTL_DIR = os.path.join(_HERE, "AssertLLM", "rtl", "uart")

# Same order as run_uart_nl2sva.py/generate_uart_raw.py -- uart2bus_top.v
# FIRST since it's the true top of the hierarchy every assertion is spliced
# into.
RTL_FILE_ORDER = [
    "uart2bus_top.v", "uart_top.v", "baud_gen.v", "uart_rx.v", "uart_tx.v", "uart_parser.v",
]

_DEFINE_LINE_RE = re.compile(r"^`define\s+\w+.*$", re.MULTILINE)


def build_combined_testbench(uart_rtl_dir):
    """2026-09-02: fallback for "slim" input CSVs (task_id/signal/
    nl_property/response/signals_for_validity -- see README's slim-CSV
    note) that never embedded the ~32KB output_tb column in the first
    place, e.g. generate_uart_raw.py's CodeV-SVA outputs. The testbench is
    identical for every row regardless of which model generated the SVA,
    so it's cheap to reconstruct once from the RTL files rather than
    requiring it be duplicated into every row of every results CSV."""
    parts = []
    for fname in RTL_FILE_ORDER:
        with open(os.path.join(uart_rtl_dir, fname)) as f:
            parts.append(f.read())
    return "\n\n".join(parts)


def hoist_defines(testbench):
    """Moves every `define line to the front of the combined text (order
    among themselves doesn't matter -- none reference each other here) so
    a macro defined in a file concatenated LATER in RTL_FILE_ORDER is
    already defined by the time the spliced-in assertion (near the very
    front, inside uart2bus_top's body) references it."""
    defines = _DEFINE_LINE_RE.findall(testbench)
    if not defines:
        return testbench
    rest = _DEFINE_LINE_RE.sub("", testbench)
    return "\n".join(defines) + "\n\n" + rest


ap = argparse.ArgumentParser()
ap.add_argument("--input", default=DEFAULT_INPUT_PATH)
ap.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
ap.add_argument("--uart-rtl-dir", default=DEFAULT_UART_RTL_DIR,
                 help="Used to (re-)derive the scope-qualification map (signal_scope.py), and as "
                      "the testbench source for 'slim' input CSVs that have no output_tb column "
                      "(see build_combined_testbench) -- for a full LMRESULT-shaped CSV, the "
                      "testbench still comes from the input CSV's own output_tb column instead.")
ap.add_argument("--workers", type=int, default=6)
ap.add_argument("--timeout", type=int, default=90)
ap.add_argument("--limit", type=int, default=None)
cli_args = ap.parse_args()

SV_DIR = f"{cli_args.output}.jgtmp"

SCOPE_MAP, MACRO_NAMES = build_signal_scope_map(cli_args.uart_rtl_dir)
print(f"Scope map: {len(SCOPE_MAP)} unambiguous internal signals, {len(MACRO_NAMES)} macros")

with open(cli_args.input) as f:
    rows = list(csv.DictReader(f))
if cli_args.limit:
    rows = rows[: cli_args.limit]
print(f"n={len(rows)}")

# "slim" CSVs (task_id/signal/nl_property/response/signals_for_validity)
# have no output_tb column at all -- build it once, up front, rather than
# per-row.
_HAS_OUTPUT_TB = "output_tb" in rows[0] if rows else True
_FALLBACK_TESTBENCH = (
    None if _HAS_OUTPUT_TB else hoist_defines(build_combined_testbench(cli_args.uart_rtl_dir))
)
if not _HAS_OUTPUT_TB:
    print(f"Input has no output_tb column -- reconstructed testbench from {cli_args.uart_rtl_dir}")


def score_one(i, row):
    task_id = row["task_id"]
    raw_testbench = _FALLBACK_TESTBENCH if _FALLBACK_TESTBENCH is not None else hoist_defines(row["output_tb"])
    bare_sva = extract_property_body(parse_code_response(row["response"]))
    qualified_sva = qualify_out_of_scope_references(bare_sva, SCOPE_MAP, MACRO_NAMES)
    try:
        syntax_ok, proven, jg_output = check_sva_proven(
            raw_testbench, qualified_sva, SV_DIR, experiment_id="uart_proven",
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
