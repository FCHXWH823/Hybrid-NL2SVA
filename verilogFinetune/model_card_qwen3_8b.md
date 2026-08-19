---
license: apache-2.0
base_model: Qwen/Qwen3-8B
library_name: transformers
pipeline_tag: text-generation
language:
  - en
tags:
  - systemverilog
  - sva
  - assertion-generation
  - formal-verification
  - eda
  - nl2sva
  - reasoning
---

# Qwen3-8B — NL2SVA with OL-NL / DFS reasoning

Qwen3-8B fully fine-tuned to translate a natural-language description of a
design property into a SystemVerilog Assertion (SVA), emitting an explicit,
structured derivation inside `<think>...</think>` before the final assertion.

The reasoning is **not** free-form chain of thought. Each assistant turn walks a
fixed three-part structure derived mechanically from the golden assertion's
operator tree:

1. **OL NL** — an *operator-level* restatement of the requirement, grounded in
   the design's real signal names and aligned to the assertion's top-level
   operator structure.
2. **nl-decomposition tree** — a top-down tree, one node per operator, each
   carrying its natural-language piece and why that operator expresses it.
3. **operator-merge-sva tree** — a bottom-up symbolic derivation labelling every
   node `T1, T2, ...`, where each line is written in terms of earlier labels.

## Prompt format

Trained with the `qwen3` chat template. The user turn contains the testbench
module (including a `tb_reset` wire and a `// TODO: ASSERTION` marker) followed
by `Question:` and the property description.

```python
from transformers import AutoModelForCausalLM, AutoTokenizer

model_id = "Jalik/qwen3-8b-codev-sva-ol-dfs-think"
tok = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype="bfloat16", device_map="auto")

messages = [
    {"role": "system", "content":
        "You are an AI assistant tasked with formal verification of register "
        "transfer level (RTL) designs.\nYour job is to translate a description "
        "of an assertion to concrete SystemVerilog Assertion (SVA) implementation."},
    {"role": "user", "content": """Here is the testbench to perform your translation:
module example(input clk, input rst_n, output valid, output fail);
// ... design body ...
wire tb_reset;
assign tb_reset = (rst_n == 1'b0);
// TODO: ASSERTION
endmodule
Question: Create a SVA assertion that checks: The valid and fail signals must never be active at the same time. Use the signals 'fail' and 'valid'.
You should use `tb_reset` as the disable condition signal. Do not add code to output an error message string.
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.
Answer:"""},
]

text = tok.apply_chat_template(messages, tokenize=False, add_generation_prompt=True,
                               enable_thinking=True)
out = model.generate(**tok(text, return_tensors="pt").to(model.device), max_new_tokens=2048)
print(tok.decode(out[0], skip_special_tokens=True))
```

`enable_thinking=True` matters: the model was trained with the reasoning block
present and loss computed over it. Disabling it diverges from training.

## Output shape

```
<think>
OL NL: valid and fail must never be true at the same time.

***nl-decomposition tree***
operator: !
    natural language piece: valid and fail must never be true at the same time.
    reason: The logical NOT (!) negates the conjunction (valid && fail), ...
    └── operator: &&
            ├── operator: null   (leaf: valid)
            └── operator: null   (leaf: fail)

***operator-merge-sva tree***
T1 = fail
T2 = valid
T3 = T2 && T1   [ = valid && fail ]
T4 = !(T3)      [ = !(valid && fail) ]

Final assertion (T4): !(valid && fail)
</think>

```systemverilog
asrt_valid_fail_mutex: assert property (@(posedge clk) disable iff (tb_reset)
    !(valid && fail)
);
```
```

The answer is always the final fenced `systemverilog` block, so downstream
extractors that take the *last* such fence work unchanged.

## Training data

5,000 records sampled from
[`wyt2000/CodeV-SVA-datasets`](https://huggingface.co/datasets/wyt2000/CodeV-SVA-datasets)
(83,195 records), filtered to those whose answer is a single `assert property`
parseable into an operator/signal tree.

The reasoning was generated, not hand-written, and — importantly — **formally
validated**. For each record an OL-NL statement was produced, a *candidate* SVA
was regenerated from that statement alone, and the candidate was checked against
the golden assertion with JasperGold. Only a **`Full equivalence`** result was
accepted; anything weaker (including one-directional implication) was rejected
and the record retried or replaced. So each retained OL-NL statement is known to
be a semantically exact restatement of its assertion, not merely a plausible one.

System and user turns are byte-identical to the source dataset; only the
assistant turn was replaced.

## Training

| | |
|---|---|
| Base | `Qwen/Qwen3-8B` (36 layers, hidden 4096, bf16) |
| Method | Full-parameter SFT (not LoRA) |
| Framework | LLaMA-Factory 0.9.3.dev0, DeepSpeed ZeRO-3 |
| Hardware | 4 × NVIDIA H200 |
| Epochs | 3 (1,875 steps) |
| LR | 1e-5, cosine, warmup ratio 0.1 |
| Effective batch | 8 (1 × 2 grad-accum × 4 GPUs) |
| Max sequence | 16,384 tokens |
| Precision | bfloat16 |
| Final train loss | **0.1725** |
| Runtime | 68 min |

Loss fell from ~1.76 to ~0.12 with gradient norm settling from ~35 to ~0.5.

## Limitations

- **Convention-bound.** Training data consistently uses `tb_reset` as the
  `disable iff` condition and a testbench with a `// TODO: ASSERTION` marker.
  Prompts departing from that shape may degrade.
- **Single-assertion scope.** Trained only on examples whose answer is exactly
  one `assert property`. Multi-assertion or multi-clock requests are out of
  distribution.
- **Not verified end-to-end here.** Training loss is not correctness. The
  reasoning traces in the *training data* were JasperGold-verified, but this
  model card reports no benchmark score for the fine-tuned model itself — treat
  generated assertions as proposals to be formally checked, not as correct by
  construction.
- **Reasoning is generated.** The decomposition trees come from an LLM pipeline;
  the operator structure is mechanical, but the natural-language justifications
  in the `reason:` fields were not individually reviewed.

## License

Released under Apache-2.0, matching the `Qwen/Qwen3-8B` base model. Training data
derives from `wyt2000/CodeV-SVA-datasets`; consult that dataset for its own terms.
