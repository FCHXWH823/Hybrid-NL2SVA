"""Run the fine-tuned model over every FVEval-Verified datapoint.

Writes one JSONL row per record: the prompts, the raw completion, the
extracted assertion, the ground truth, and match flags.

Reported match rates are *string* comparisons and are a floor, not a score:
an assertion can be semantically equivalent to the reference while differing
textually (operand order, redundant parens, a different label). Real scoring
needs JasperGold equivalence -- see jasper_equiv_check.py. Exact/normalized
match is here to spot format regressions cheaply, not to grade the model.

Batched generation uses left padding, which decoder-only models require: with
right padding the pad tokens land between the prompt and the first generated
token and corrupt the continuation.
"""

import argparse
import json
import os
import re
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_codev_sva_ol_dfs_eval import SYSTEM_PROMPT, build_testbench_with_marker
from run_one_fveval_example import build_user_prompt, DEFAULT_DATA, DEFAULT_MODEL

FENCE = "```systemverilog"


def extract_assertion(completion):
    """Last ```systemverilog fence -- same rule as utils.parse_code_response."""
    if FENCE not in completion:
        return None
    body = completion[completion.rfind(FENCE) + len(FENCE):]
    return body.split("```")[0].strip() or None


def normalize(sva):
    """Collapse whitespace and drop the assertion label, so `asrt:` vs
    `asrt_overflow:` and indentation differences don't count as mismatches."""
    if not sva:
        return ""
    s = re.sub(r"^\s*\w+\s*:\s*", "", sva.strip())
    return re.sub(r"\s+", " ", s).strip().rstrip(";")


def parse_signals(raw):
    if isinstance(raw, str):
        try:
            return json.loads(raw.replace("'", '"'))
        except Exception:
            return [raw]
    return raw


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--output", default="data/fveval_human_qwen3_8b_think.jsonl")
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--no-tb-reset-hint", dest="tb_reset_hint",
                        action="store_false", default=True)
    args = parser.parse_args()

    with open(args.data) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if args.limit:
        records = records[:args.limit]
    print("records: %d" % len(records))

    prompts = []
    for r in records:
        tb = build_testbench_with_marker(r["testbench"])
        prompts.append(build_user_prompt(
            tb, r["problem"], parse_signals(r.get("signals_for_validity")),
            args.tb_reset_hint))

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model, padding_side="left")
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    texts = [
        tok.apply_chat_template(
            [{"role": "system", "content": SYSTEM_PROMPT},
             {"role": "user", "content": p}],
            tokenize=False, add_generation_prompt=True, enable_thinking=True)
        for p in prompts
    ]

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    exact = norm = have_fence = 0
    started = time.time()

    with open(args.output, "w") as out:
        for start in range(0, len(texts), args.batch_size):
            chunk = texts[start:start + args.batch_size]
            enc = tok(chunk, return_tensors="pt", padding=True).to(model.device)
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=args.max_new_tokens, do_sample=False,
                    pad_token_id=tok.pad_token_id)
            completions = [
                tok.decode(gen[i][enc["input_ids"].shape[-1]:], skip_special_tokens=True)
                for i in range(len(chunk))
            ]

            for offset, completion in enumerate(completions):
                idx = start + offset
                rec = records[idx]
                got = extract_assertion(completion)
                gt = rec["ground_truth"]
                is_exact = bool(got) and got.strip() == gt.strip()
                is_norm = bool(got) and normalize(got) == normalize(gt)
                have_fence += bool(got)
                exact += is_exact
                norm += is_norm
                out.write(json.dumps({
                    "name": rec.get("name"),
                    "problem": rec["problem"],
                    "system_prompt": SYSTEM_PROMPT,
                    "user_prompt": prompts[idx],
                    "completion": completion,
                    "extracted_assertion": got,
                    "ground_truth": gt,
                    "exact_match": is_exact,
                    "normalized_match": is_norm,
                }, ensure_ascii=False) + "\n")
                out.flush()

            done = min(start + args.batch_size, len(texts))
            rate = (time.time() - started) / done
            print("  %3d/%d  exact=%d norm=%d  (%.1fs/record, ~%.0fs left)"
                  % (done, len(texts), exact, norm, rate,
                     rate * (len(texts) - done)), flush=True)

    n = len(records)
    print()
    print("=" * 60)
    print("records            : %d" % n)
    print("produced a fence   : %d (%.1f%%)" % (have_fence, 100.0 * have_fence / n))
    print("exact match        : %d (%.1f%%)" % (exact, 100.0 * exact / n))
    print("normalized match   : %d (%.1f%%)" % (norm, 100.0 * norm / n))
    print("wrote              : %s" % args.output)
    print()
    print("NOTE: string match only. Semantically equivalent assertions that differ")
    print("textually count as misses here -- use JasperGold for a real score.")


if __name__ == "__main__":
    main()
