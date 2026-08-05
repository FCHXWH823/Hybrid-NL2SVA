# CodeV-SVA OL-NL + Two-Part Decomposition Framework

Summary of the pipeline that turns `CodeV-SVA-dataset-training-83K.jsonl` into a
fine-tuning dataset whose assistant turns show explicit reasoning (an operator-level
restatement, a top-down decomposition, and a bottom-up symbolic derivation) instead
of free-form chain-of-thought. Original design discussion and full plan:
`/Users/fch/.claude/plans/polished-chasing-wreath.md`.

## Motivation

`CodeV-SVA-dataset-training-83K.jsonl` has 83,195 chat-format `{system, user,
assistant}` records for the RTL+spec → SVA task. Its `Question:` text mixes
abstraction levels:

- operator-level (`nl2sva_machine`-style): "Both sig_F and sig_H are high, or all
  bits of sig_I are high, or sig_B is high."
- abstract/domain-level (`nl2sva_human`-style): "that the counter does not
  underflow."

An existing DFS decomposition pipeline (`generate_dfs_explanation.py` +
`sva_graph.py`, built earlier for `qwen_explanation.jsonl`) produces two clean
reasoning artifacts from an explanation, but only works when that explanation is
already operator-level — it mechanically splits a parent phrase into its
children's phrases, which has no clean split point for an abstract sentence like
"the counter does not underflow." The fix: define **OL NL** (operator-level
natural-language) as a normalization step, applied before the existing pipeline,
so *any* input abstraction level converges to the same operator-aligned form
first.

## The four generation stages

### Stage 1 — eligibility filter + sampling
**`select_codev_sva_sample.py`**

Scans all 83,195 records. A record is eligible if its final ` ```systemverilog ` ```
block (the text after `</think>`) contains exactly one `assert property` and is
parseable by `sva_graph.build_operator_signal_graph()` (the deep operator/signal
tree — going through the sequence and Boolean layers, not just the property
layer). Measured: **99.5% eligible** (82,806 / 83,195) — much higher than the
~87% seen on the older `qwen_explanation.jsonl` corpus, because CodeV-SVA
answers are template-generated and mostly single properties. Draws a fixed-seed
(`--seed 0`) uniform random sample of 5,000 from the eligible pool →
`data/codev_sva_5000_sample.json`.

### Stage 2 — OL-NL transform + formal-equivalence validation
**`generate_ol_nl_explanation.py`**

For each sampled record:

1. **Generate** an OL-NL statement: one LLM call given the original Question
   (any abstraction), the golden SVA, the RTL testbench, and the operator-context
   table (`operators.json`). The model rewrites the description into a statement
   naming only the assertion's real signals, with clauses mapping onto its
   top-level operator structure. Two worked examples are baked into the prompt
   (`(ERROR==1) |-> (PSLVERR==1)` → "When ERROR equals 1, then PSLVERR equals 1
   from the current clock cycle"; and a "the counter does not underflow" →
   fully-grounded-in-real-signals rewrite).

2. **Validate** — *not* by checking that signal names merely appear in the text
   (tried first, dropped: it can't catch a statement that names the right
   signals but gets the logic between them wrong). Instead:
   - Generate a **candidate SVA from the OL-NL statement alone** (testbench +
     statement → SVA, the same task a real NL2SVA model faces at inference
     time) — `generate_candidate_sva()`.
   - Run a **formal equivalence check** between that candidate and the golden
     SVA via JasperGold (`jasper_equiv_check.py` — see below).
   - Only accept the statement on a `Full equivalence` result. Otherwise retry
     (up to `--max-retries`) with the failed candidate SVA shown back to the
     model as corrective feedback (`PROMPT_TEMPLATE_OL_NL_RETRY`).
   - If never verified: the record is dropped and backfilled by the orchestrator
     (Stage 4), not force-included with an unverified statement.

**`jasper_equiv_check.py`** — reuses the *existing* FVEval harness's own tcl
script (`Evaluation/FVRuleLearner/FVEval/tool_scripts/run_jg_nl2sva_human.tcl`)
and its IP-protected `pec.tcle` equivalence-check macro (`prop_eq_checker`) — the
same mechanism `fv_tool_execution.launch_jg_custom_equiv_check` /
`NL2SVAHumanEvaluator` already use for final scoring, so Stage 2's validation bar
matches what Stage 6 evaluation will actually measure. Deliberately does *not*
import `fv_tool_execution.py` itself (that module pulls in the harness's
CLI-oriented `config`/`saver` globals — `Evaluation/FVRuleLearner/src/config.py`
hardcodes `global_task = 'train'` at import time — not something a standalone
script should trigger as a side effect); instead it reimplements just the one
`jg -fpv -batch -tcl ... -define LM_ASSERT_TEXT ... -define REF_ASSERT_TEXT ...`
subprocess call and output parsing, verified line-by-line against
`launch_jg_custom_equiv_check` and the tcl script.

Acceptance rule (`is_equivalent`) is stricter than the harness's own
`calculate_jg_metric`: only "Full equivalence" counts, not a one-directional
"implies" match — Stage 2 needs the OL-NL statement to be exactly right, not
merely a relaxation of the golden SVA, which is what the harness's
`func_relaxed` metric tolerates at scoring time.

**Requires `jg` (JasperGold) on PATH with a valid license.** Not available in
the sandbox this was built in — every pure-Python piece (prompt construction,
assertion-body stripping, signal-list extraction, output-string parsing) was
unit-tested directly; the actual `jg` subprocess call was not. Validate on a
small batch wherever `jg` is actually installed before trusting the full run.

### Stage 3 — two-part decomposition (no new logic)
Reuses, unmodified, the already-built and already-verified pipeline from
`generate_dfs_explanation.py` / `sva_graph.py`:

- `walk_tree()` — one LLM call per *operator* node (never leaves). Each call
  sees only its own operator + operand code and returns (a) its natural-language
  piece, (b) why the operator represents it, (c)/(d) the sub-pieces for each
  operand. Python threads (c)/(d) down as the child's own input piece — this
  mechanical threading is what enforces top-down consistency, not the model's
  memory.
- `render_decomposition_tree()` → **Part 1**: a top-down indented tree
  (operator / piece / reason, `├──`/`└──` branches to leaves; leaves show
  `operator: null`).
- `render_merge_tree()` (in `sva_graph.py`) → **Part 2**: a bottom-up symbolic
  derivation. Every node gets a label `T1, T2, ...` in reverse-DFS order; each
  line states what that label equals in terms of *earlier labels' names*, not
  their full code, so every line stays short except the unavoidable final one.

### Stage 4 — reassembly
**`generate_codev_sva_reasoning_dataset.py`** (the orchestrator; ties Stages
2+3 together and writes the final dataset)

System and user turns are kept byte-for-byte identical to the original CodeV-SVA
record. Only the assistant turn is replaced:

```
OL NL: <grounded, formally-verified statement>

***nl-decomposition tree***
<Part 1>

According to the above natural-language decomposition, we finally derive the SystemVerilog assertion as follows:

***operator-merge-sva tree***
<Part 2>

```systemverilog
<original golden SVA, unchanged>
```
```

That final fenced block is why this stays compatible with the eval harness's
own extractor (`utils.parse_code_response`), which only ever looks at the
*last* ` ```systemverilog ` fence in a response.

Runs with a thread pool (`--workers`, default 24) since each record's several
LLM calls are independent and I/O-bound; a lock-guarded shared file handle
(`SourceFile`) gives random access into the 83K-line source without loading it
all into memory. Records whose OL-NL statement never verifies (or whose
property is a bare signal with no operator — nothing to decompose) are dropped
and backfilled with the next untouched eligible record from the source file, so
the output always has exactly `--sample-size` records. Checkpointed
(`--checkpoint`, defaults to `<output>.checkpoint.jsonl`) so a killed/resumed run
picks up where it left off.

Output: `data/codev_sva_ol_dfs_5000.jsonl`, registered in `data/dataset_info.json`
as `codev_sva_ol_dfs_5000` (sharegpt formatting, since this dataset is
chat-shaped, unlike the project's other alpaca-style `instruction`/`input`/`output`
datasets).

## Downstream: fine-tuning and evaluation

- **Stage 5 — fine-tune**: `train_codev_sva_ol_dfs.sh` +
  `finetune_codev_sva_ol_dfs.sbatch`, mirroring the existing LLaMA-Factory
  full-finetune setup (`train_qwen_prompt_guided_explanation.sh` /
  `finetune_nl2sva.sbatch`) — same base model
  (`deepseek-ai/deepseek-coder-7b-instruct-v1.5`), same hyperparameters, pointed
  at the new dataset.
- **Stage 6 — evaluate**: `run_codev_sva_ol_dfs_eval.py` queries an
  OpenAI-compatible endpoint (e.g. vLLM serving the fine-tuned checkpoint) over
  each of the three benchmark CSVs
  (`Evaluation/FVRuleLearner/FVEval/data_nl2sva/data/nl2sva_human.csv`,
  `nl2sva_machine.csv`, `Evaluation/FVRuleLearner/FVEval/data_1k/module_sva_nl_manual_editing.csv`)
  and writes an `LMResult`-shaped CSV the existing `NL2SVAHumanEvaluator` /
  `NL2SVAMachineEvaluator` classes already consume — no changes to that harness.
  Deliberately does **not** force the training data's "use tb_reset as the
  disable condition" instruction at eval time: the benchmark CSVs' own
  `ref_solution` rows don't follow that convention consistently even within one
  benchmark (e.g. an `nl2sva_machine` reference with no `disable iff` at all
  despite its testbench defining `tb_reset`), so hard-coding it would bias the
  model against some of the benchmark's own correct answers.
  `module_sva_nl_manual_editing.csv` only provides `module_interface` (a bare
  port list, no body) — wrapped into a minimal module with the
  `// TODO: ASSERTION` marker appended directly, since there's no `tb_reset`
  scaffold to preserve. Untested end-to-end (needs a served checkpoint +
  JasperGold, neither available in this sandbox); `output_tb`/`cot_response`
  field assumptions should be double-checked once that infra exists.

## File map

| File | Role |
|---|---|
| `select_codev_sva_sample.py` | Stage 1 — eligibility filter + sampling |
| `generate_ol_nl_explanation.py` | Stage 2 — OL-NL transform + candidate-SVA + JasperGold validation |
| `jasper_equiv_check.py` | JasperGold subprocess wrapper (reused by Stage 2, shares the harness's tcl/pec.tcle) |
| `generate_dfs_explanation.py`, `sva_graph.py` | Stage 3 — existing, unmodified decomposition/derivation pipeline |
| `generate_codev_sva_reasoning_dataset.py` | Stages 2+3+4 orchestrator — full pipeline entrypoint |
| `train_codev_sva_ol_dfs.sh`, `finetune_codev_sva_ol_dfs.sbatch` | Stage 5 — fine-tuning |
| `run_codev_sva_ol_dfs_eval.py` | Stage 6 — benchmark inference/CSV assembly |
| `data/codev_sva_5000_sample.json` | Stage 1 output |
| `data/codev_sva_ol_dfs_5000.jsonl` | Final Stage 4 output (fine-tuning dataset) |

## Status at time of writing

Stages 1–6 are all implemented. The full 5,000-record Stage 2–4 run was started
(40 workers) and reached 428/5,000 completed (checkpointed, safe to resume)
before being intentionally stopped — Stage 2's validation was still using the
cheap signal-name-presence check at that point; it has since been replaced with
the JasperGold-based check described above, so a resumed run will re-validate
every record under the new, stricter criterion. Nothing has been run against a
real JasperGold install or a served fine-tuned checkpoint.
