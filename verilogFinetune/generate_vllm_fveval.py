"""vLLM version of the FVEval-Verified generation sweep.

Prompt construction is imported verbatim from generate_codev_fveval.py, so
prompts are byte-identical to the transformers runs and results stay
comparable. Only the inference engine changes.

Why vLLM: the transformers path ran model.generate() over a fixed batch, which
advances in lockstep until the LONGEST member finishes. Roughly 20% of these
records never terminate (the model loops, repeating a line dozens of times) and
run to the full token cap, so a single runaway forced its 7 batch-mates through
32,768 decode steps each. Measured: 4-6 steps/s, ~1.8 hours per batch of 8, and
5 of 7 batches contained a runaway. vLLM's continuous batching retires each
sequence as it finishes and admits the next one, so a runaway occupies one slot
instead of stalling eight; paged attention also keeps the 32k KV cache cheap.

Truncation is read from vLLM's own finish_reason ("length" = hit the cap)
rather than inferred from token counts and a missing </think>, which is what
the transformers script had to do.
"""

import argparse
import json
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from generate_codev_fveval import extract_sva
from official_prompter import NL2SVA_SYSTEM_PROMPT as SYSTEM_PROMPT, build_user_prompt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=("human", "machine"),
                    help="selects the official prompt variant; they differ in more than the tb_reset line")
    ap.add_argument("--data", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--sva-output", default=None)
    ap.add_argument("--max-new-tokens", type=int, default=32768)
    # Official sampling, from SVAClient/configs/nl2sva_*_local_template_pass_at_k.yaml
    ap.add_argument("--temperature", type=float, default=0.8)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--n", type=int, default=1,
                    help="samples per prompt; official pass@k protocol uses n>1")
    ap.add_argument("--seed", type=int, default=42,
                    help="the official vllm server config pins random_seeds: [42, 42]")
    ap.add_argument("--max-model-len", type=int, default=40960,
                    help="Qwen3's max_position_embeddings; prompt + generation must fit")
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.90)
    ap.add_argument("--tensor-parallel-size", type=int, default=1)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    records = [json.loads(l) for l in open(args.data) if l.strip()]
    if args.limit:
        records = records[:args.limit]
    prompts = [build_user_prompt(r, args.task) for r in records]
    n = len(records)
    print("model   : %s" % args.model, flush=True)
    print("data    : %s (%d records)" % (args.data, n), flush=True)
    print("task    : %s (official CodeV-SVA prompt)" % args.task, flush=True)

    from vllm import LLM, SamplingParams
    from transformers import AutoTokenizer

    tok = AutoTokenizer.from_pretrained(args.model)
    texts = [tok.apply_chat_template(
        [{"role": "system", "content": SYSTEM_PROMPT}, {"role": "user", "content": p}],
        tokenize=False, add_generation_prompt=True, enable_thinking=True) for p in prompts]

    llm = LLM(model=args.model,
              max_model_len=args.max_model_len,
              gpu_memory_utilization=args.gpu_memory_utilization,
              tensor_parallel_size=args.tensor_parallel_size,
              dtype="bfloat16",
              trust_remote_code=True)

    # Official CodeV-SVA sampling: temperature 0.8 / top_p 0.95 (their eval
    # config), NOT greedy. Note this makes a single sample non-deterministic in
    # spirit -- their protocol draws n samples per prompt and reports pass@k.
    # seed is pinned so a given run is at least reproducible.
    sp = SamplingParams(temperature=args.temperature, top_p=args.top_p,
                        max_tokens=args.max_new_tokens, n=args.n, seed=args.seed)

    started = time.time()
    outs = llm.generate(texts, sp)          # vLLM schedules all of them itself
    elapsed = time.time() - started

    # vLLM may return results out of order; index by request id.
    outs = sorted(outs, key=lambda o: int(o.request_id))

    for path in (args.output, args.sva_output):
        if path:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)

    got = truncated = 0
    gen_tokens = 0
    sva_f = open(args.sva_output, "w") if args.sva_output else None
    with open(args.output, "w") as out:
        for idx, o in enumerate(outs):
            comp = o.outputs[0]
            completion = comp.text
            is_trunc = comp.finish_reason == "length"
            sva = extract_sva(completion)
            got += bool(sva)
            truncated += bool(is_trunc)
            gen_tokens += len(comp.token_ids)
            rec = records[idx]
            out.write(json.dumps({
                "name": rec.get("name"),
                "problem": rec.get("problem"),
                "system_prompt": SYSTEM_PROMPT,
                "user_prompt": prompts[idx],
                "completion": completion,
                "extracted_sva": sva,
                "truncated": is_trunc,
                "finish_reason": comp.finish_reason,
                "n_generated_tokens": len(comp.token_ids),
                "ground_truth": rec.get("ground_truth"),
            }, ensure_ascii=False) + "\n")
            if sva_f:
                sva_f.write(json.dumps({"name": rec.get("name"), "sva": sva},
                                       ensure_ascii=False) + "\n")
    if sva_f:
        sva_f.close()

    print()
    print("records         : %d" % n)
    print("extracted SVA   : %d (%.1f%%)" % (got, 100.0 * got / n))
    print("hit token cap   : %d (%.1f%%)" % (truncated, 100.0 * truncated / n))
    print("generated tokens: %d" % gen_tokens)
    print("wall time       : %.1f s (%.1f s/record, %.0f tok/s)"
          % (elapsed, elapsed / n, gen_tokens / elapsed))
    print("wrote           : %s" % args.output)
    if args.sva_output:
        print("wrote           : %s" % args.sva_output)


if __name__ == "__main__":
    main()
