"""Run the CodeV-SVA models over the FVEval-Verified benchmarks.

Prompts reproduce wyt2000/CodeV-SVA-datasets (CodeV-SVA-dataset-training-83K)
exactly -- verified against the live dataset, byte for byte:

    system: "You are an AI assistant tasked with formal verification ...\n"
    user:   "Here is the testbench to perform your translation:\n"
            "<testbench, with `// TODO: ASSERTION` before endmodule>\n"
            "Question: Create a SVA assertion that checks: <problem> Use the signals 'a', 'b'.\n"
            "[You should use `tb_reset` as the disable condition signal. ...]\n"
            "Enclose your SVA code with ```systemverilog and ```. ...\n\n"
            "For example,\n```systemverilog\n...\n```\nAnswer:"

Two parts are conditional, because the two FVEval-Verified benchmarks follow
different conventions and hard-coding either one would misprompt the other:

  * the tb_reset line is emitted only when the testbench actually declares a
    tb_reset wire. All 73 human testbenches do (and all 73 references use
    `disable iff (tb_reset)`); none of the 283 machine ones do, and no machine
    reference uses it. Telling the model to disable on a signal that does not
    exist would guarantee a syntax error.
  * the "Use the signals ..." clause is appended only when the problem text
    does not already carry one. 64 of 73 human problems do; 0 of 283 machine
    problems do, so those take it from signal_list.
"""

import argparse
import json
import os
import time

SYSTEM_PROMPT = (
    "You are an AI assistant tasked with formal verification of register transfer level (RTL) designs.\n"
    "Your job is to translate a description of an assertion to concrete SystemVerilog Assertion (SVA) implementation.\n"
)

TB_RESET_LINE = (
    "You should use `tb_reset` as the disable condition signal. "
    "Do not add code to output an error message string.\n"
)

TAIL = (
    "Enclose your SVA code with ```systemverilog and ```. "
    "Only output the code snippet and do NOT output anything else.\n"
    "\nFor example,\n"
    "```systemverilog\n"
    "asrt: assert property (@(posedge clk) disable iff (tb_reset)\n"
    "    (a && b) != 1'b1\n"
    ");\n"
    "```\n"
    "Answer:"
)

FENCE = "```systemverilog"


def with_marker(testbench):
    """`// TODO: ASSERTION` immediately before the final endmodule."""
    tb = testbench.rstrip()
    if "endmodule" in tb:
        head, _, _ = tb.rpartition("endmodule")
        return "%s\n// TODO: ASSERTION\nendmodule" % head.rstrip()
    return "%s\n// TODO: ASSERTION\nendmodule" % tb


def signals_of(record):
    raw = record.get("signals_for_validity") or record.get("signal_list") or ""
    if isinstance(raw, list):
        return [str(s).strip() for s in raw if str(s).strip()]
    raw = str(raw).strip()
    if not raw:
        return []
    if raw.startswith("["):
        try:
            return [str(s).strip() for s in json.loads(raw.replace("'", '"'))]
        except Exception:
            pass
    return [s.strip() for s in raw.split(",") if s.strip()]


def build_user_prompt(record):
    testbench = record["testbench"]
    problem = (record["problem"] or "").strip()

    question = "Question: Create a SVA assertion that checks: %s" % problem
    if "Use the signals" not in problem:
        sigs = signals_of(record)
        if sigs:
            question += " Use the signals %s." % ", ".join("'%s'" % s for s in sigs)
    question += "\n"
    if "tb_reset" in testbench:
        question += TB_RESET_LINE

    return ("Here is the testbench to perform your translation:\n%s\n%s%s"
            % (with_marker(testbench), question, TAIL))


def extract_sva(completion):
    if FENCE not in completion:
        return None
    return completion[completion.rfind(FENCE) + len(FENCE):].split("```")[0].strip() or None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sva-output", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        records = records[:args.limit]
    prompts = [build_user_prompt(r) for r in records]
    n = len(records)
    print("model   : %s" % args.model)
    print("data    : %s (%d records)" % (args.data, n))
    print("tb_reset hint on: %d/%d" % (sum(1 for p in prompts if "disable condition signal" in p), n))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    texts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True) for p in prompts]

    for path in (args.output, args.sva_output):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    got = truncated = 0
    started = time.time()
    sva_f = open(args.sva_output, "w") if args.sva_output else None

    with open(args.output, "w") as out:
        for start in range(0, len(texts), args.batch_size):
            chunk = texts[start:start + args.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(**enc, max_new_tokens=args.max_new_tokens,
                                     do_sample=False, pad_token_id=tok.pad_token_id)
            new = gen[:, enc["input_ids"].shape[-1]:]
            for i in range(len(chunk)):
                idx = start + i
                completion = tok.decode(new[i], skip_special_tokens=True)
                sva = extract_sva(completion)
                # a completion that used the whole budget and never closed its
                # reasoning is a non-termination, not a refusal -- record it
                is_trunc = int(new[i].ne(tok.pad_token_id).sum()) >= args.max_new_tokens \
                    and "</think>" not in completion
                got += bool(sva)
                truncated += bool(is_trunc)
                rec = records[idx]
                out.write(json.dumps({
                    "name": rec.get("name"),
                    "problem": rec.get("problem"),
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": prompts[idx],
                    "completion": completion,
                    "extracted_sva": sva,
                    "truncated": is_trunc,
                    "ground_truth": rec.get("ground_truth"),
                }, ensure_ascii=False) + "\n")
                out.flush()
                if sva_f:
                    sva_f.write(json.dumps({"name": rec.get("name"), "sva": sva},
                                           ensure_ascii=False) + "\n")
                    sva_f.flush()
            done = min(start + args.batch_size, n)
            rate = (time.time() - started) / done
            print("  %4d/%d  sva=%d trunc=%d  (%.1fs/rec, ~%.0fs left)"
                  % (done, n, got, truncated, rate, rate * (n - done)), flush=True)

    if sva_f:
        sva_f.close()
    print()
    print("records        : %d" % n)
    print("extracted SVA  : %d (%.1f%%)" % (got, 100.0 * got / n))
    print("non-terminating: %d (%.1f%%)" % (truncated, 100.0 * truncated / n))
    print("wrote          : %s" % args.output)
    if args.sva_output:
        print("wrote          : %s" % args.sva_output)


if __name__ == "__main__":
    main()
