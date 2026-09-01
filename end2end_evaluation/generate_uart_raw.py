"""Generate SVAs for AssertionForge's UART NL plans with a local model, raw.

run_uart_nl2sva.py drives the same plans through the full RAG+SOR pipeline,
which needs an OpenAI key and `jg`. This is the no-pipeline counterpart: one
prompt, one completion, no retrieval, no grounding, no revision -- the raw-model
column of the comparison, produced on a GPU node with vLLM.

Prompt construction is official_prompter's MACHINE variant, deliberately.
The human variant appends "You should use `tb_reset` as the disable condition
signal." and shows a few-shot whose body is `disable iff (tb_reset)`, but the
UART RTL has no tb_reset -- its reset is `reset` and its clock is `clock`.
Handing the model a template it cannot satisfy is exactly the failure
official_prompter's own docstring calls out as the one "with teeth", so
machine (no disable-iff instruction, plain @(posedge clk) few-shot) is the
only correct choice here.

The testbench slot gets the six UART RTL files concatenated in RTL_FILE_ORDER,
matching run_uart_nl2sva.py: uart2bus_top.v first, because it is the true top
of the hierarchy and the scope every assertion binds into.

Token budget: the concatenated RTL makes the prompt ~9.5k tokens against
Qwen3's 40960 max_position_embeddings, so max_new_tokens cannot be the 32768
used for the FVEval sweeps -- prompt + generation would not fit. 31000 leaves
headroom for the longest plan text.

Usage (from the repo root, inside the vLLM env):
    python3 end2end_evaluation/generate_uart_raw.py \
        --model wyt2000/CodeV-SVA-8B --model-tag codev8b \
        --uart-rtl-dir /scratch/wx2356/AssertLLM/rtl/uart
"""

import argparse
import csv
import json
import os
import re
import sys
import time

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, "..", "verilogFinetune"))

from generate_codev_fveval import extract_sva
from official_prompter import NL2SVA_SYSTEM_PROMPT as SYSTEM_PROMPT, build_user_prompt

# uart2bus_top.v FIRST -- see run_uart_nl2sva.py: assertions are inserted before
# the first `endmodule`, and uart2bus_top is the hierarchy top where all the
# planned signals are visible.
RTL_FILE_ORDER = [
    "uart2bus_top.v", "uart_top.v", "baud_gen.v", "uart_rx.v", "uart_tx.v", "uart_parser.v",
]

# Kept identical to run_uart_nl2sva.py so the slim CSV's signals_for_validity
# column matches the pipeline run's column exactly and the two are comparable.
VALID_SIGNALS = [
    'baud_clk', 'baud_freq', 'baud_limit', 'ce_16', 'int_address', 'int_gnt', 'int_rd_data',
    'int_read', 'int_req', 'int_wr_data', 'int_write', 'new_rx_data', 'new_tx_data', 'rx_data',
    'ser_in', 'ser_out', 'tx_busy', 'tx_data',
]


def parse_nl_plans(path):
    """Returns [(plan_number, signal_name, plan_text), ...] in file order.

    KEEPS the plan number, unlike run_uart_nl2sva.py's parser, which drops it
    and lets the caller use enumerate(). That was equivalent when the file
    still held all 323 plans, but the committed nl_plans_uart.txt has since had
    the four non-design-controlled signals (ce_16, int_gnt, int_rd_data,
    ser_in -- see NON_CONTROLLED_SIGNALS in run_uart_nl2sva.py) filtered out.
    256 plans remain, numbered non-contiguously up to 323.

    Position and plan number therefore no longer agree, and task_id is built
    from the number: enumerate() reproduces only 55 of the 256 task_ids the
    gpt-4o pipeline run recorded, silently mislabeling the other 201 with
    another property's id. Parsing the number gives all 256.
    """
    plans = []
    current_signal = None
    with open(path) as f:
        for line in f:
            line = line.rstrip("\n")
            sig_match = re.match(r"^Signal (\w+):$", line)
            if sig_match:
                current_signal = sig_match.group(1)
                continue
            plan_match = re.match(r"^Plan (\d+):\s*(.*)$", line)
            if plan_match and current_signal:
                plans.append((int(plan_match.group(1)), current_signal,
                              plan_match.group(2).strip()))
    return plans


def build_combined_testbench(uart_rtl_dir):
    parts = []
    for fname in RTL_FILE_ORDER:
        with open(os.path.join(uart_rtl_dir, fname)) as f:
            parts.append(f.read())
    return "\n\n".join(parts)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--model-tag", required=True, help="short name for filenames, e.g. codev8b")
    ap.add_argument("--uart-rtl-dir",
                    default=os.path.join(_HERE, "AssertLLM", "rtl", "uart"),
                    help="AssertLLM's UART RTL. Defaults to the copy vendored in this repo; "
                         "verified byte-identical to an upstream clone.")
    ap.add_argument("--plans", default=os.path.join(_HERE, "nl_plans_uart.txt"))
    ap.add_argument("--out-dir", default=os.path.join(_HERE, "results"))
    ap.add_argument("--max-new-tokens", type=int, default=31000)
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--n", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--max-model-len", type=int, default=40960)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    plans = parse_nl_plans(args.plans)
    if args.limit:
        plans = plans[:args.limit]
    testbench = build_combined_testbench(args.uart_rtl_dir)
    n = len(plans)
    print("model     : %s" % args.model, flush=True)
    print("plans     : %s (%d properties)" % (args.plans, n), flush=True)
    print("rtl       : %s (%d chars combined)" % (args.uart_rtl_dir, len(testbench)), flush=True)

    prompts = [build_user_prompt({"testbench": testbench, "problem": text}, "machine")
               for _num, _sig, text in plans]

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True) for p in prompts]

    prompt_tokens = len(tok(texts[0])["input_ids"])
    print("prompt    : ~%d tokens, %d left for generation under max_model_len=%d"
          % (prompt_tokens, args.max_model_len - prompt_tokens, args.max_model_len), flush=True)
    if prompt_tokens + args.max_new_tokens > args.max_model_len:
        sys.exit("ERROR: prompt (%d) + max_new_tokens (%d) exceeds max_model_len (%d)"
                 % (prompt_tokens, args.max_new_tokens, args.max_model_len))

    llm = LLM(model=args.model,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=args.tensor_parallel_size,
              dtype="bfloat16",
              trust_remote_code=True)
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens, n=args.n, seed=args.seed)

    started = time.time()
    outs = llm.generate(texts, sp)
    elapsed = time.time() - started
    outs = sorted(outs, key=lambda o: int(o.request_id))

    os.makedirs(args.out_dir, exist_ok=True)
    base = "assertionforge_uart_%s_raw" % args.model_tag
    full_path = os.path.join(args.out_dir, base + "_full.jsonl")
    slim_path = os.path.join(args.out_dir, base + "_slim.csv")
    valid_signals = ",".join(VALID_SIGNALS)

    got = truncated = gen_tokens = 0
    # Column order matches assertionforge_uart_gpt-4o_dynamicrag_slim.csv so the
    # raw and pipeline runs can be scored by the same script.
    slim_fields = ["task_id", "signal", "nl_property", "response", "signals_for_validity"]
    with open(full_path, "w") as fh, open(slim_path, "w", newline="") as sf:
        writer = csv.DictWriter(sf, fieldnames=slim_fields)
        writer.writeheader()
        for idx, o in enumerate(outs):
            comp = o.outputs[0]
            completion = comp.text
            is_trunc = comp.finish_reason == "length"
            sva = extract_sva(completion)
            plan_number, signal, text = plans[idx]
            # task_id is keyed on the plan number, not the row position -- see
            # parse_nl_plans. This is what makes the raw and pipeline runs
            # joinable row-for-row.
            task_id = "AssertionForge-UART-%d-%s" % (plan_number - 1, signal)
            got += bool(sva)
            truncated += bool(is_trunc)
            gen_tokens += len(comp.token_ids)
            fh.write(json.dumps({
                "task_id": task_id,
                "signal": signal,
                "nl_property": text,
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": prompts[idx],
                "completion": completion,
                "extracted_sva": sva,
                "truncated": is_trunc,
                "finish_reason": comp.finish_reason,
                "n_generated_tokens": len(comp.token_ids),
            }, ensure_ascii=False) + "\n")
            writer.writerow({
                "task_id": task_id,
                "signal": signal,
                "nl_property": text,
                # The pipeline CSV stores the fenced block, not the bare SVA.
                "response": "```systemverilog\n%s\n```" % (sva or ""),
                "signals_for_validity": valid_signals,
            })

    print()
    print("properties      : %d" % n)
    print("extracted SVA   : %d (%.1f%%)" % (got, 100.0 * got / n))
    print("hit token cap   : %d (%.1f%%)" % (truncated, 100.0 * truncated / n))
    print("generated tokens: %d" % gen_tokens)
    print("wall time       : %.1f s (%.1f s/property)" % (elapsed, elapsed / n))
    print("wrote           : %s" % full_path)
    print("wrote           : %s" % slim_path)


if __name__ == "__main__":
    main()
