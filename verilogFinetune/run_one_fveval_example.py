"""Build the prompts for one FVEval-Verified datapoint and run the fine-tuned
model on it.

Prompt construction reuses run_codev_sva_ol_dfs_eval.py's SYSTEM_PROMPT and
build_testbench_with_marker() so this matches the project's own eval path
rather than inventing a third format.

One deliberate difference: the tb_reset instruction. run_codev_sva_ol_dfs_eval
drops it, because the CSV benchmarks it targets are inconsistent about the
convention and hard-coding it would bias the model against some of their own
correct answers. FVEval-Verified is not inconsistent -- all 73 human
ground truths use `disable iff (tb_reset)` -- so here the instruction matches
both the benchmark and the training data, and is included by default.
--no-tb-reset-hint reproduces the eval script's wording instead.
"""

import argparse
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from run_codev_sva_ol_dfs_eval import SYSTEM_PROMPT, build_testbench_with_marker

DEFAULT_MODEL = "/scratch/wx2356/verilogFinetune/output/qwen3-8B-codev-sva-ol-dfs-think"
DEFAULT_DATA = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "Evaluation", "FVEval-Verified", "fveval_nl2sva_human.jsonl",
)

EXAMPLE_BLOCK = """Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

For example,
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Answer:"""

TB_RESET_HINT = (
    "You should use `tb_reset` as the disable condition signal. "
    "Do not add code to output an error message string.\n"
)


def build_user_prompt(testbench_with_marker, problem, signals, tb_reset_hint=True):
    """Mirrors the training data's user turn."""
    problem = problem.strip()
    question = "Question: Create a SVA assertion that checks: %s" % problem
    # 64 of the 73 problem fields already end with their own "Use the signals
    # 'a', 'b'..." clause. Appending signals_for_validity unconditionally
    # duplicates it, which never appears in training and wastes context on a
    # contradictory-looking repetition.
    if signals and "Use the signals" not in problem:
        quoted = ", ".join("'%s'" % s for s in signals)
        question += " Use the signals %s." % quoted
    question += "\n"
    if tb_reset_hint:
        question += TB_RESET_HINT
    return (
        "Here is the testbench to perform your translation:\n"
        "%s\n%s%s" % (testbench_with_marker, question, EXAMPLE_BLOCK)
    )


def load_record(path, index=None, name=None):
    with open(path) as f:
        records = [json.loads(line) for line in f if line.strip()]
    if name:
        for r in records:
            if r.get("name") == name:
                return r, len(records)
        sys.exit("no record named %r" % name)
    return records[index or 0], len(records)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default=DEFAULT_DATA)
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument("--name", default=None)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--no-tb-reset-hint", dest="tb_reset_hint",
                        action="store_false", default=True)
    parser.add_argument("--prompt-only", action="store_true",
                        help="print the prompts and exit; no model load")
    args = parser.parse_args()

    record, total = load_record(args.data, args.index, args.name)
    signals = record.get("signals_for_validity")
    if isinstance(signals, str):
        try:
            signals = json.loads(signals.replace("'", '"'))
        except Exception:
            signals = [signals]

    tb = build_testbench_with_marker(record["testbench"])
    user_prompt = build_user_prompt(tb, record["problem"], signals, args.tb_reset_hint)

    bar = "=" * 78
    print(bar); print("RECORD: %s   (%d of %d)" % (record.get("name"), args.index + 1, total)); print(bar)
    print()
    print(bar); print("SYSTEM PROMPT"); print(bar); print(SYSTEM_PROMPT)
    print(bar); print("USER PROMPT"); print(bar); print(user_prompt); print()
    print(bar); print("GROUND TRUTH"); print(bar); print(record["ground_truth"]); print()

    if args.prompt_only:
        return

    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print(bar); print("MODEL: %s" % args.model); print(bar)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]
    text = tok.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
    inputs = tok(text, return_tensors="pt").to(model.device)
    print("prompt tokens: %d" % inputs["input_ids"].shape[-1])

    gen = dict(max_new_tokens=args.max_new_tokens,
               pad_token_id=tok.pad_token_id or tok.eos_token_id)
    if args.temperature and args.temperature > 0:
        gen.update(do_sample=True, temperature=args.temperature)
    else:
        gen.update(do_sample=False)

    with torch.no_grad():
        out = model.generate(**inputs, **gen)
    completion = tok.decode(out[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True)

    print(); print(bar); print("MODEL OUTPUT"); print(bar); print(completion)

    print(); print(bar); print("EXTRACTED ASSERTION (last ```systemverilog fence)"); print(bar)
    marker = "```systemverilog"
    if marker in completion:
        body = completion[completion.rfind(marker) + len(marker):]
        print(body.split("```")[0].strip())
    else:
        print("(no systemverilog fence found)")


if __name__ == "__main__":
    main()
