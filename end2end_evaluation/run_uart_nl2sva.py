"""Feeds AssertionForge's UART NL properties (nl_plans_uart.txt, Stage 2's
output) through Hybrid-NL2SVA's own pipeline for NL2SVA, using the SAME flag
configuration as nl2sva_machine_verified's best-known config:
--skip-signal-list-note --sor-conservative --only-overlap-implication, no
--ol-nl-grounding (these properties are already signal-grounded, same
rationale as the machine task). Reuses process_row directly rather than
plumbing a whole new --task branch into run_rag_on_fveval_benchmarks.py.

Run from the Hybrid-NL2SVA repo root:
    python3 end2end_evaluation/run_uart_nl2sva.py --workers 6

Prerequisites:
  - AssertLLM cloned somewhere (spec + RTL source for UART and 4 other
    designs): https://github.com/hkust-zhiyao/AssertLLM
    Set UART_RTL_DIR below (or pass --uart-rtl-dir) to
    <AssertLLM clone>/rtl/uart.
  - `jg` (JasperGold) on PATH -- same requirement as the main pipeline.
  - Src/Config.yml with a valid Openai_API_Key (see Src/Config.yml.example).

Re-running: pass --resume to skip task_ids already present in the output
CSV and append only the missing rows (handles OpenAI rate-limit/credit
interruptions without re-doing completed rows or re-spending on them).
"""
import sys, os, re, csv, argparse, yaml, concurrent.futures, threading

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "Src", "MultiRoundPromptwithOperatorsExplanation"))
import run_rag_on_fveval_benchmarks as m
from langchain_openai import ChatOpenAI
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from openai import OpenAI

_HERE = os.path.dirname(os.path.abspath(__file__))
NL_PLANS_PATH = os.path.join(_HERE, "nl_plans_uart.txt")
# AssertLLM is now vendored alongside this script (see ../AssertLLM/README.md
# for its no-LICENSE-file caveat) -- override via --uart-rtl-dir to point at
# a different clone instead.
DEFAULT_UART_RTL_DIR = os.path.join(_HERE, "AssertLLM", "rtl", "uart")

# uart2bus_top.v FIRST: check_sva_elaboration/check_sva_proven insert the
# assertion right before the FIRST `endmodule` in the combined text, and
# uart2bus_top is the true top of the hierarchy (instantiates uart_top, which
# itself instantiates baud_gen/uart_rx/uart_tx; also instantiates uart_parser
# directly) -- all 18 VALID_SIGNALS below are visible in its scope, either as
# its own ports or as internal wires wired to uart1's ports.
RTL_FILE_ORDER = [
    "uart2bus_top.v", "uart_top.v", "baud_gen.v", "uart_rx.v", "uart_tx.v", "uart_parser.v",
]

# Full architectural signal list used for the actual Stage 2 (gen_plan) run
# that produced nl_plans_uart.txt -- kept exactly as run for reproducibility.
VALID_SIGNALS = [
    'baud_clk', 'baud_freq', 'baud_limit', 'ce_16', 'int_address', 'int_gnt', 'int_rd_data',
    'int_read', 'int_req', 'int_wr_data', 'int_write', 'new_rx_data', 'new_tx_data', 'rx_data',
    'ser_in', 'ser_out', 'tx_busy', 'tx_data',
]

# Post-hoc finding (see README): 4 of the 18 are NOT design-controlled and
# skew #Proven low for reasons unrelated to NL2SVA quality --
#   - ser_in, int_rd_data, int_gnt: true free/unconstrained primary INPUTS of
#     uart2bus_top -- JasperGold treats them as able to take any value on any
#     cycle with no environment `assume`s, so a property claiming "this input
#     settles into range X" is essentially unprovable by construction.
#   - ce_16: declared inside uart_top (one hierarchy level below
#     uart2bus_top, our assertion-binding scope) -- not even a valid
#     identifier there, a separate elaboration-scope bug.
# Not excluded from VALID_SIGNALS/the actual run (kept faithful to what was
# generated) -- exposed here for scripts that want to reproduce the
# "design-controlled only" analysis without re-deriving it.
NON_CONTROLLED_SIGNALS = ['ser_in', 'int_rd_data', 'int_gnt', 'ce_16']

DEFAULT_OUTPUT_PATH = os.path.join(_HERE, "results", "assertionforge_uart_gpt-4o_dynamicrag.csv")


def parse_nl_plans(path):
    """Returns [(signal_name, plan_text), ...] in file order."""
    plans = []
    current_signal = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            sig_match = re.match(r"^Signal (\w+):$", line)
            if sig_match:
                current_signal = sig_match.group(1)
                continue
            plan_match = re.match(r"^Plan \d+:\s*(.*)$", line)
            if plan_match and current_signal:
                plans.append((current_signal, plan_match.group(1).strip()))
    return plans


def build_combined_testbench(uart_rtl_dir):
    parts = []
    for fname in RTL_FILE_ORDER:
        with open(os.path.join(uart_rtl_dir, fname)) as f:
            parts.append(f.read())
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--uart-rtl-dir", default=DEFAULT_UART_RTL_DIR,
                     help="Path to AssertLLM's rtl/uart directory")
    ap.add_argument("--output", default=DEFAULT_OUTPUT_PATH)
    ap.add_argument("--config", default="Src/Config.yml")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--resume", action="store_true",
                     help="Skip task_ids already present in --output and append only the missing rows "
                          "(instead of overwriting from scratch) -- for resuming after an API credit/rate-"
                          "limit interruption.")
    cli_args = ap.parse_args()

    plans = parse_nl_plans(NL_PLANS_PATH)
    if cli_args.limit:
        plans = plans[: cli_args.limit]
    print(f"{len(plans)} NL properties loaded from {NL_PLANS_PATH}")

    raw_testbench = build_combined_testbench(cli_args.uart_rtl_dir)
    print(f"Combined UART testbench: {len(raw_testbench)} chars, module order: {RTL_FILE_ORDER}")

    indices = range(len(plans))
    done_ids = set()
    if cli_args.resume and os.path.exists(cli_args.output):
        with open(cli_args.output) as f:
            done_ids = {row["task_id"] for row in csv.DictReader(f)}
        indices = [i for i in indices if not any(
            f"AssertionForge-UART-{i}-" == tid[: len(f"AssertionForge-UART-{i}-")] for tid in done_ids
        )]
        print(f"--resume: {len(done_ids)} rows already in {cli_args.output}, {len(indices)} remaining")

    args = argparse.Namespace(
        task="nl2sva_machine_verified",  # reuses build_verified_machine_user_prompt +
                                          # disable_signal=None, matching the machine config
        csv=None, output=None, config=cli_args.config,
        provider="openai", model_name=None, limit=None, workers=cli_args.workers, max_retries=5,
        no_rag=False, ol_nl_grounding=False, ol_nl_replace_question=False, ol_nl_conservative=False,
        skip_signal_list_note=True, sor_template_timing=False, sor_conservative=True,
        only_overlap_implication=True, clock_signal="clock",
    )

    with open(args.config) as f:
        config = yaml.safe_load(f)
    openai_api_key = config["Openai_API_Key"]
    model_name = config["Model_Name"]  # gpt-4o in the original run

    client = OpenAI(api_key=openai_api_key)
    rich_operator_context = m.load_rich_operator_context()
    step1_jg_sv_dir = f"{cli_args.output}.step1_jgtmp"

    code_store = m.build_rag_system(config["PDF_Txt"], openai_api_key)
    code_retriever = code_store.as_retriever()

    llm = ChatOpenAI(model=model_name, api_key=openai_api_key)

    system_prompt = (
        m.SYSTEM_PROMPT
        + "\n\n" + m.EXPRESSION_ONLY_INSTRUCTION
        + "\n{allowed_signals}"
        + "Use the following pieces of retrieved context to help answer the question.\n\n"
        + "{ol_nl_grounding}"
        + "{keywords_explaination}"
        + "{context}"
    )
    prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
    rag_chain = create_retrieval_chain(code_retriever, create_stuff_documents_chain(llm, prompt))

    escaped_operator_context = rich_operator_context.replace("{", "{{").replace("}", "}}")
    system_prompt_checker = (
        "You are a helpful bot that fixes a real JasperGold elaboration error reported for the "
        "given SVA property expression. "
        + "\n\nSVA Operator Context:\n" + escaped_operator_context
        + "\n\n" + m.EXPRESSION_ONLY_INSTRUCTION
        + "\n{allowed_signals}"
        + "Use the following pieces of retrieved context to help answer the question.\n\n"
        "{context}"
    )
    prompt_checker = ChatPromptTemplate.from_messages([("system", system_prompt_checker), ("human", "{input}")])
    rag_chain_checker = create_retrieval_chain(code_retriever, create_stuff_documents_chain(llm, prompt_checker))

    experiment_id = "assertionforge_uart_gpt-4o_dynamicrag"
    os.makedirs(os.path.dirname(cli_args.output) or ".", exist_ok=True)

    write_lock = threading.Lock()
    file_mode = "a" if (cli_args.resume and done_ids) else "w"
    with open(cli_args.output, file_mode, newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=m.LMRESULT_FIELDNAMES)
        if file_mode == "w":
            writer.writeheader()

        with concurrent.futures.ThreadPoolExecutor(max_workers=cli_args.workers) as executor:
            futures = [
                executor.submit(
                    m.process_row, i, f"AssertionForge-UART-{i}-{plans[i][0]}", raw_testbench, plans[i][1],
                    "", VALID_SIGNALS, args, client, model_name, rich_operator_context,
                    code_retriever, rag_chain, rag_chain_checker, step1_jg_sv_dir, experiment_id,
                )
                for i in indices
            ]
            completed = 0
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                completed += 1
                if row is None:
                    continue
                with write_lock:
                    writer.writerow(row)
                    out_file.flush()
                print(f"    [{completed}/{len(indices)} done] task_id={row['task_id']}")

    print(f"Wrote responses to {cli_args.output}")


if __name__ == "__main__":
    main()
