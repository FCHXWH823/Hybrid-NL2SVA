# End-to-end Spec2SVA: AssertionForge + Hybrid-NL2SVA (UART pilot)

Combines [AssertionForge](https://github.com/NVlabs/AssertionForge) (NVlabs,
LAD 2025) as a **Spec2NL** front-end (spec PDF + RTL → natural-language
verification properties) with this repo's own RAG+SOR pipeline as the
**NL2SVA** back-end, closing the loop from a real specification document all
the way to formally-checked SVAs -- as opposed to Part I's evaluation, which
runs NL2SVA alone against FVEval-Verified's already-curated, single-property-
per-row benchmark rows. Spec/RTL source data comes from
[AssertLLM](https://github.com/hkust-zhiyao/AssertLLM) (hkust-zhiyao),
vendored here too (`AssertLLM/`) -- see the license caveat in the Files
section before reusing it outside this repo.

```
spec.pdf + RTL/ ──[AssertionForge Stage 1: build KG]──▶ knowledge graph
                 ──[AssertionForge Stage 2: gen_plan]──▶ NL properties (nl_plans_uart.txt)
                 ──[Hybrid-NL2SVA: same config as        candidate SVAs
                    nl2sva_machine_verified]──────────▶  (results/*_dynamicrag*.csv)
                 ──[JasperGold: check_sva_proven]──────▶ #SVA / #SynC / #Proven
                                                          (results/*_jgscore.csv)
```

## Results (UART, `AssertLLM`'s spec + RTL)

- **Design**: 6 Verilog modules (`uart2bus_top` real top, `uart_top`,
  `baud_gen`, `uart_rx`, `uart_tx`, `uart_parser`); 246 total signals, 18
  architectural (interface) signals used as `valid_signals`.
- **Spec2NL**: knowledge graph with 1,371 nodes / 1,620 edges (spec + RTL
  fused); **323** natural-language verification properties generated for all
  18 signals in the original run.
  `nl_plans_uart.txt` **has since been filtered down to 256 properties**,
  keeping only the 14 design-controlled signals (dropping `ser_in`/
  `int_rd_data`/`int_gnt`/`ce_16` -- see the methodology-gap note below for
  why). The `results/` CSVs below still reflect the original, unfiltered
  323-property run; re-running `run_uart_nl2sva.py` against the current
  `nl_plans_uart.txt` would reproduce only the 256-row "design-controlled
  only" subset (task_id indices would also shift, since they're assigned by
  position in the parsed plan list). Note some kept-signal properties still
  incidentally *mention* an excluded signal in their NL text (AssertionForge
  groups plans by primary target signal, not by every signal referenced) --
  this filtering wasn't re-applied at that finer grain.
- **NL2SVA**: all 323 properties translated via Hybrid-NL2SVA (gpt-4o), using
  `nl2sva_machine_verified`'s best-known flag configuration
  (`--skip-signal-list-note --sor-conservative --only-overlap-implication`,
  no `--ol-nl-grounding`) plus a `clock_signal="clock"` override (UART's real
  clock port isn't named `clk`).
- **JasperGold** (`check_sva_proven`, `prove -all` + `get_status`, no golden
  SVA needed):

  | Subset | n | #SynC | #Proven |
  |---|---|---|---|
  | All 18 signals | 323 | 225 (69.7%) | 60 (18.6%) |
  | Design-controlled only (excl. `ser_in`/`int_rd_data`/`int_gnt`/`ce_16`; current `nl_plans_uart.txt`) | 256 | 177 (69.1%) | **52 (20.3%)** |
  | Excluded (free inputs / out-of-scope) | 67 | 48 (71.6%) | 8 (11.9%) |

  (All three rows are computed by filtering the original 323-row scored
  results by the signal named in each `task_id` -- not a fresh run against
  the now-256-row `nl_plans_uart.txt` -- but the property text for the kept
  256 is unchanged by that filtering, so the numbers should carry over
  exactly. #SynC is close across all three subsets (~69-72%); the gap is
  almost entirely in #Proven, consistent with the free-input/scope-bug
  explanation below affecting provability, not syntax.)

  For reference, QiMeng-CodeV-SVA's own Table 5 reports, for UART with GPT-4o
  as both Spec2NL and NL2SVA: #SVA=265, #SynC=186 (70.2%), #Proven=54
  (20.4%). Our all-18-signal and design-controlled-only numbers both land
  close to that reference row -- **but this is not a strict apples-to-apples
  comparison**: we independently reconstructed the 18-signal `valid_signals`
  list (the AssertionForge repo's own checked-in config for UART only had a
  2-signal placeholder, `['baud_clk', 'baud_freq']`), and we have no evidence
  the paper's own run used the same signal scope. Treat the closeness as
  evidence our pipeline lands in a plausible, self-consistent range, not as a
  precise benchmark match.

### A real methodology gap, not a generation-quality gap

AssertionForge's own hardcoded few-shot prompt (`get_sva_icl_examples()` in
its `gen_plan.py`) is built entirely around `PWDATA` -- APB's free bus-input
signal -- teaching the model to write properties like *"the input data PWDATA
has a value between 83 and 165 ... 3 clock cycles after reset deasserted"*.
Applied to UART, this produces properties about `int_rd_data`/`int_gnt`
(true free/unconstrained primary inputs of `uart2bus_top`) claiming they
"settle into range X" -- but a primary input is free under JasperGold with no
environment `assume`s, so such a property is essentially unprovable by
construction, regardless of NL2SVA quality. `ce_16` is a separate bug: it's
declared inside `uart_top`, one hierarchy level below `uart2bus_top` (our
assertion-binding scope), so it's not even a valid identifier there.
Excluding these 4 signals lifts #Proven from 18.6%→20.3%. 46.1% of the 165
"syntax-ok but not proven" (cex) rows reference one of these 3 free inputs.

### A second gap: signal hallucination / wrong-scope references (fixed 2026-08-27)

Of the 323-run's 98 syntax-fail rows, `'X' is not declared` errors named 68
distinct identifiers, splitting into two categories (checked against the 6
real RTL files):

- **21 real RTL identifiers used at the wrong scope** (e.g. `ce_16` x34,
  `bit_count` x26, `ce_1` x22, `rx_busy` x20, `main_sm`/`tx_sm` x14 each,
  `data_in_hex_range` x12, `write_req` x10, ...) -- genuine registers/FSM
  state declared *inside* `uart_rx.v`/`uart_tx.v`/`uart_parser.v`/
  `baud_gen.v`, invisible from `uart2bus_top`'s scope (where the assertion
  is actually bound).
- **47 pure hallucinations, not found anywhere in the RTL** (e.g.
  `local_global_clock_freq`, `logic_counter`, `write_req_signal`,
  `some_declared_counter`, `undeclared_signal1`) -- affecting 31/98 (~32%)
  of syntax-fail rows.

**Root cause**: `run_uart_nl2sva.py` set `args.task = "nl2sva_machine_verified"`
to reuse `build_verified_machine_user_prompt`/`disable_signal=None`, which
also silently inherited that task's `--skip-signal-list-note` best-known
setting. That flag is *correct* for FVEval's real `nl2sva_machine_verified`
data (bare port-list testbenches, problem text already names every signal
directly -- see `build_verified_machine_user_prompt`'s own docstring) but
wrong for UART: `skip_signal_list_note=True` means `run_rag_on_fveval_
benchmarks.py`'s `allowed_signals_note` ("you must use ONLY signals from
this list") is never built at all, while the prompt still dumps the FULL
32KB, 6-module UART RTL as context (`build_verified_machine_user_prompt`
includes `raw_testbench` verbatim -- the same text used for JasperGold's
elaboration check later). With no guardrail and every submodule's internals
visible, the model freely reached for whatever register looked semantically
relevant.

**Fix**: `run_uart_nl2sva.py` now sets `skip_signal_list_note=False` and
passes a new `ALLOWED_SIGNALS` constant (`VALID_SIGNALS` minus `ce_16` --
17 signals; unlike `NON_CONTROLLED_SIGNALS`, `ser_in`/`int_rd_data`/
`int_gnt` stay in since they're real, valid identifiers in scope, just
unprovable, which is a different concern from this guardrail's job).
`run_rag_on_fveval_benchmarks.py`'s `clock_signal` threading (see the
Reproducing-section note below, now resolved) was restored at the same
time, so this was validated as a combined fix.

**Validation pilot** (`--signals baud_clk,baud_freq`, 36 of the 256
properties, same 36 rows before/after):

| | n | #SynC | #Proven |
|---|---|---|---|
| Before (old `skip_signal_list_note=True` run) | 36 | 26 (72.2%) | 9 (25.0%) |
| After (fixed) | 36 | **33 (91.7%)** | **13 (36.1%)** |

Files: `results/pilot_signalfix_baudclk_baudfreq.csv` (generation) /
`results/pilot_signalfix_baudclk_baudfreq_jgscore.csv` (JasperGold scores).
The 3 remaining syntax-fail rows after the fix are all genuine SVA-
construction bugs unrelated to signal scope -- `|->` misused inside a
sequence, an invented system function (`$steady_gclk`), and two invented
macros (`` `BAUD_RATE ``/`` `GLOBAL_CLOCK_FREQ `` -- the real ones are
`` `D_BAUD_FREQ ``/`` `D_BAUD_LIMIT ``) -- confirming the fix eliminated
signal-hallucination/wrong-scope failures specifically, not syntax failures
in general.

**Not yet done**: a full rerun of all 256 properties with this fix (the 323/
256-row `results/` CSVs above still reflect the OLD `skip_signal_list_note=
True` config) -- the #SynC/#Proven headline numbers in the Results table
above will move once that's done.

### 0-shot base-model baseline, same 36 properties

Ran the same `baud_clk`/`baud_freq` 36 properties through `--no-rag` (plain
gpt-4o completion, no RAG/SOR/syntax-cleanup) for comparison. First attempt
hit a related-but-distinct format bug: 34/36 responses used a NAMED
`property NAME; ... endproperty` + separate `assert property(NAME) else
$error(...)` block instead of one inline `assert property(...)` --
`extract_property_body`'s regex-based stripping (built for the single-
statement form) can't parse that shape, so #SynC collapsed to 1/36 (2.8%)
purely from the mismatch, not generation quality (same failure shape as an
earlier confirmed qwen3-8b machine-baseline bug). Fixed the same way: for
`--no-rag` + `nl2sva_machine_verified` specifically, `process_row` now
appends an explicit "one inline `assert property(...)`, no named property
block" instruction + worked example to the baseline prompt (`build_verified_
machine_user_prompt`/`SYSTEM_PROMPT` give no output-format guidance at all
otherwise). That alone took the named-property rate to 0/36. Re-scored:

| | n | #SynC | #Proven |
|---|---|---|---|
| 0-shot baseline (format-fixed, still no signal-list guardrail) | 36 | 14 (38.9%) | 5 (13.9%) |
| Full pipeline (clock_signal + skip_signal_list_note fixes) | 36 | **33 (91.7%)** | **13 (36.1%)** |

The baseline's remaining 22 syntax failures are dominated by exactly the
signal-scope/hallucination pattern from "A second gap" above (`ce_16` x16,
`global_clock_freq` x8, `counter` x6, `baud_rate` x6, plus a few undefined
macros) -- expected, since `--no-rag` never gets the `allowed_signals_note`
guardrail regardless of `skip_signal_list_note` (that flag only gates the
RAG/pipeline path; `generate_baseline_sva` doesn't take the note as a
parameter at all). Files: `results/pilot_baseline_baudclk_baudfreq.csv` /
`results/pilot_baseline_baudclk_baudfreq_jgscore.csv`.

## Files

- `run_uart_nl2sva.py` -- feeds `nl_plans_uart.txt` through Hybrid-NL2SVA's
  `process_row` (same machinery the main pipeline uses), writing
  `results/assertionforge_uart_gpt-4o_dynamicrag.csv`. Supports `--resume`
  for backfilling after an API rate-limit/credit interruption, and
  `--signals sig1,sig2` for scoping a run to specific target signals (used
  for the validation pilot below).
- `score_uart_assertionforge.py` -- JasperGold-scores that CSV via
  `check_sva_proven` (new function in
  `verilogFinetune/jasper_direct_equiv_check.py` -- a standalone formal
  proof, `prove -all` + `get_status`, needing no golden reference, unlike
  that module's existing equivalence-checking functions), writing
  `results/assertionforge_uart_gpt-4o_dynamicrag_jgscore.csv`.
- `nl_plans_uart.txt` -- AssertionForge Stage 2's output, grouped by signal,
  **filtered to the 256 properties for the 14 design-controlled signals**
  (originally 323 across all 18; see above). The `results/` CSVs were scored
  from the original unfiltered 323.
- `results/assertionforge_uart_gpt-4o_dynamicrag_slim.csv` -- one row per
  property: `task_id`, `signal`, `nl_property` (the NL text), `response`
  (the generated SVA), `signals_for_validity`. (The full LMRESULT-shaped CSV
  `run_uart_nl2sva.py` writes also embeds the ~32KB combined UART testbench
  in every row, which is regenerable from `assertionforge_patches/` +
  AssertLLM's RTL and wasn't worth committing at ~31MB; this slim version is
  the one actually checked in.)
- `results/assertionforge_uart_gpt-4o_dynamicrag_jgscore.csv` -- per-row
  `syntax`/`proven` verdicts (1.0/0.0) plus a `jg_output_tail` with the raw
  JasperGold status line.
- `AssertionForge/` -- the full [AssertionForge](https://github.com/NVlabs/AssertionForge)
  (NVlabs, LAD 2025) source tree, vendored in-place with our patches already
  applied (not just a diff -- see "Vendoring" below), so no separate clone is
  needed to reproduce the UART pilot. Its own upstream `.git` and `.venv`
  aren't carried over (see "Vendoring"); `LICENSE.txt` (NVIDIA Source Code
  License for AssertionForge -- non-commercial reproduction/redistribution
  with attribution is permitted, which is why the full source can be included
  here, not just isolated patch files) is preserved at its root.
- `AssertLLM/` -- the full [AssertLLM](https://github.com/hkust-zhiyao/AssertLLM)
  (hkust-zhiyao) dataset: spec PDFs + golden RTL for 20 designs, vendored
  verbatim (upstream `.git` and a `.DS_Store` or two dropped; the `spec/
  graph_rag_uart/` GraphRAG cache/output that ended up alongside it in our
  scratch clone -- our own regenerable pipeline byproduct, not part of
  AssertLLM's own dataset -- was dropped too). **License caveat**: unlike
  AssertionForge, AssertLLM's repo ships with **no LICENSE file** at all, so
  unlike the NVIDIA-licensed AssertionForge tree above, there's no explicit
  grant permitting redistribution here -- default copyright applies. Included
  anyway per explicit user instruction, for internal reproducibility; think
  twice before re-publishing this subfolder elsewhere, and attribute
  hkust-zhiyao/AssertLLM if you do use it.

## Vendoring: what we changed in `AssertionForge/`

AssertionForge ships `src/utils_LLM.py` and `src/utils_LLM_client.py` as
empty "add your own LLM client" stubs, and is missing `src/load_result.py`
entirely (only used by a code path this pilot never exercises, but imported
unconditionally by `gen_plan.py`). `AssertionForge/src/` here has our
implementations of all three in place of upstream's. `utils_LLM_client.py`'s
`_HYBRID_NL2SVA_CONFIG` constant points at this repo's `Src/Config.yml`.

`AssertionForge/src/config.py` is the actual config used for the UART Stage
1/2 runs. Its `file_path`/`design_dir`/`input_file_path` for every design
AssertLLM actually provides spec+RTL for (`openMSP430`, `tiny_pairing`,
`uart`, `sockit`) now point at the vendored `AssertLLM/` folder above via a
`_ASSERTLLM_DIR = Path(__file__).resolve().parents[2] / 'AssertLLM'`
constant -- no path editing needed for those. `env_source_path`/
`settings_source_path` (GraphRAG `.env`/`settings.yaml`) and every design's
`KG_path` (a GraphRAG *run output* location, not part of AssertLLM's own
dataset) are still `<path>/<to>/...` placeholders -- see step 1 below. Its
`ROOT = Path(__file__).resolve().parents[N]` line
was also adjusted (`parents[3]`, since this file now lives 3 levels under
this repo's own root) so `config.py`'s `git.Repo(ROOT)` call -- used only for
run-logging metadata -- resolves to *this* repo's `.git` rather than
AssertionForge's own (which isn't vendored, to avoid a nested-git-repo
inside this one). `AssertionForge/logs/` keeps the actual Stage 1/2 run logs
(including `gen_plan_.../nl_plans.txt`, the source `nl_plans_uart.txt` here
was copied from) for provenance.

Not vendored: `.venv/` (a Python 3.11 venv with ~300 packages including
torch/transformers/graphrag, several GB -- rebuild it per step 3 below) and
`.git/` (AssertionForge's own upstream history -- irrelevant once vendored,
and would otherwise create a nested git repo inside this one).

## Reproducing

AssertionForge and AssertLLM are both vendored here now, so no separate
clones are needed -- just:

1. In `AssertionForge/src/config.py`, fill in the `<path>/<to>/...`
   placeholders for `env_source_path`/`settings_source_path` (a GraphRAG
   `.env`/`settings.yaml` template -- `python -m graphrag.index --init --root
   <dir>` generates a starting point; set `.env`'s `GRAPHRAG_API_KEY`, and in
   `settings.yaml` set the model to `gpt-4o` and `snapshots.graphml: true`,
   matching the actual run) and `design_name == 'uart'`'s `KG_path` (points
   at wherever Stage 1 below writes its output).
2. `cd AssertionForge && pip install -r requirements.txt` in a fresh venv --
   as shipped, 3 lines are broken and need removing first: `install==1.3.5`
   (nonexistent PyPI version), `py_sv_parser==0.3.0` (fails to build against
   current maturin/Rust), and the whole `pyautogen`+`llama-index-*` cluster
   (mutually incompatible `openai` version pins; neither is actually imported
   anywhere in `src/`). Then `pip install graphrag==0.3.6 gitpython`
   separately (matching AssertionForge's `python -m graphrag.index` CLI
   expectations; `graphrag` itself isn't pinned in `requirements.txt` at all).
3. Run Stage 1 (`task = 'build_KG'`) then Stage 2 (`task = 'gen_plan'`,
   `subtask = 'actual_gen'`, `generate_SVAs = False`) via `python main.py`
   from `AssertionForge/src/`. Stage 2's log directory (under
   `AssertionForge/logs/`) contains `nl_plans.txt` (copy it in as
   `nl_plans_uart.txt` here, or point `NL_PLANS_PATH` at it).
4. From this repo's root: `python3 end2end_evaluation/run_uart_nl2sva.py
   --workers 6` (defaults to the vendored `AssertLLM/rtl/uart`; override with
   `--uart-rtl-dir` for a different clone; add `--signals sig1,sig2` to scope
   a run to specific target signals), then `python3
   end2end_evaluation/score_uart_assertionforge.py --workers 6`.

   **Update (2026-08-27)**: two real bugs (found while reviewing the
   committed pipeline) are now fixed. (1) `clock_signal` wasn't actually
   threaded through `wrap_property_expression`/`jg_driven_syntax_cleanup`/
   `generate_rag_sva`/`process_row` in `run_rag_on_fveval_benchmarks.py`
   despite `run_uart_nl2sva.py` setting `args.clock_signal = "clock"` --
   restored, all four now accept/pass it correctly. (2) `skip_signal_list_
   note=True` (inherited from `nl2sva_machine_verified`'s best-known config,
   wrongly -- see "A second gap" above) let the model reference out-of-scope/
   hallucinated signal names with no guardrail -- now `False`, with a proper
   `ALLOWED_SIGNALS` list. Both fixes validated together on a 36-row pilot;
   see "A second gap" above for the before/after numbers. The `results/`
   CSVs (323/256-row) still reflect the OLD, unfixed config -- a full rerun
   with the fix hasn't been done yet.

## Extending

Next candidates from QiMeng-CodeV-SVA's Table 5: APB, ETHMAC, OPENMSP430,
SOCKIT (all sourced the same way -- 4 from AssertLLM, APB from OpenCores'
[`apb_mstr`](https://opencores.org/projects/apb_mstr)). Also worth trying:
AssertionForge's own built-in NL2SVA generator (`generate_SVAs = True`) as a
head-to-head baseline against Hybrid-NL2SVA on the *same* Spec2NL properties,
and swapping DeepSeek-V4-Flash / qwen3 in as AssertionForge's Spec2NL
backend instead of gpt-4o.
