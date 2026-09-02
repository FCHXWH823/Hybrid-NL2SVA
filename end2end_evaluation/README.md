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

### A second gap: signal hallucination / wrong-scope references

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

**But most of this traces back further, to the NL plans themselves, not
just the pipeline**: of all 338 undeclared-identifier occurrences across
those 98 rows, **236 (69.8%) have their "core" name literally present in
the corresponding NL plan's own text** -- e.g. the plan for `baud_freq`
reads *"...corresponds to the calculated value using the formula
16*baud_rate / gcd(global_clock_freq, 16*baud_rate)"*, and the model just
faithfully transcribed `baud_rate`/`global_clock_freq` as identifiers --
except neither is a real RTL signal (the real UART only has two *macros*,
`` `D_BAUD_FREQ ``/`` `D_BAUD_LIMIT ``, pre-computed constants, not
separate signals for the clock frequency and baud rate). Only the
remaining 102 (30.2%) are inventions with no textual antecedent at all
(`state_change_valid`, `some_declared_counter`, ...).

AssertionForge's Stage 2 NL plans *look* operator-level (they quote
specific names in the FVEval-machine style), but that quoting is never
validated against the actual RTL -- it's built from spec-PDF prose mixed
with KG-retrieved RTL context, so it freely mixes real signal names,
submodule-internal names out of the assertion's scope, and pure spec-level
concepts (formulas, computed quantities) that were never Verilog
identifiers to begin with. This is a fundamentally different grounding
guarantee than FVEval's `nl2sva_machine_verified`, where every named signal
in the problem text was human-verified real and in-scope -- the assumption
behind that dataset's `--skip-signal-list-note` best-known setting.

**Two candidate fixes, one adopted, one deliberately not**:
- Restoring `run_rag_on_fveval_benchmarks.py`'s `clock_signal` parameter
  threading (lost from the committed pipeline code despite `run_uart_
  nl2sva.py` setting `args.clock_signal="clock"` -- `wrap_property_
  expression` was hardcoding `@(posedge clk)` regardless) -- **adopted**,
  unrelated to the signal-hallucination issue but a real, separate bug.
- Setting `skip_signal_list_note=False` + an `ALLOWED_SIGNALS` guardrail
  list (`VALID_SIGNALS` minus `ce_16`) -- validated on a 36-row pilot
  (`--signals baud_clk,baud_freq`) alongside the `clock_signal` fix:

  | | n | #SynC | #Proven |
  |---|---|---|---|
  | Before (both bugs present) | 36 | 26 (72.2%) | 9 (25.0%) |
  | After (both fixed) | 36 | **33 (91.7%)** | **13 (36.1%)** |

  Measurably helped, and the 3 remaining syntax-fail rows after the fix are
  all genuine SVA-construction bugs unrelated to signal scope (`|->`
  misused inside a sequence, an invented system function `$steady_gclk`,
  two invented macros). **But deliberately NOT adopted as the default** --
  given the 70/30 split above, an explicit allow-list mostly papers over
  AssertionForge's own NL-plan grounding gap rather than fixing it at the
  source, and masks how much of #SynC/#Proven is actually attributable to
  NL2SVA generation quality vs. upstream plan quality. `run_uart_nl2sva.py`
  keeps `skip_signal_list_note=True` (matching `nl2sva_machine_verified`'s
  config) by default; `ALLOWED_SIGNALS`/the pilot files
  (`results/pilot_signalfix_baudclk_baudfreq*.csv`) are kept in the repo as
  a documented, reproducible experiment, not the adopted configuration.

### The "human route" (`--ol-nl-grounding`) — ADOPTED default (2026-08-31)

`nl2sva_human_verified`'s own best-known config includes `--ol-nl-grounding`
(Step 1, `generate_ol_nl_grounding`) -- its prompt explicitly says to
"rewrite the description... naming ONLY signals that actually appear in the
testbench (never invented or paraphrased names)", which reads like a
principled, source-level answer to "A second gap" above (fix the NL text
itself, rather than restrict the final SVA's vocabulary). Tried two variants
on the same 36-row pilot, both with the `clock_signal` fix and
`skip_signal_list_note` left at its default `True`:

| | n | #SynC | #Proven |
|---|---|---|---|
| Baseline (no Step 1) | 36 | 26 (72.2%) | 9 (25.0%) |
| **`--ol-nl-grounding`** (no replace-question) | 36 | 25 (69.4%) | 10 (27.8%) |
| `--ol-nl-grounding --ol-nl-replace-question` | 36 | 23 (63.9%) | 10 (27.8%) |

Neither variant reliably fixes the pattern -- the remaining syntax failures
under both still cite `ce_16`/`counter`/`global_clock_freq`/`baud_rate` at
similar rates to the baseline. Root cause: Step 1's own "actually appear in
the testbench" check is done against the FULL 32KB, 6-module RTL dump, not
against `uart2bus_top`'s actual assertion-binding scope -- so it considers
`ce_16` (declared in `uart_top`, not `uart2bus_top`) "grounded" too, the
same scope-blindness as the underlying generation call. `--ol-nl-replace-
question` additionally introduced NEW structural SVA errors (a fresh
invented identifier `p_master_startl`, several `syntax error near ')'`/
`'##'`/`'|->'` cases) by having the generation call see only Step 1's own
rewrite -- net #SynC went DOWN, not up.

**Decision: `--ol-nl-grounding` (without `--ol-nl-replace-question`)
adopted as `run_uart_nl2sva.py`'s default anyway** (`--no-ol-nl-grounding`
to opt out), despite measuring slightly worse than the no-Step-1 baseline
on this small a sample (69.4% vs 72.2%, within noise at n=36) and clearly
worse than the not-adopted `skip_signal_list_note=False` (91.7%) -- chosen
as the more principled, source-level approach consistent with how
`nl2sva_human_verified` itself is configured, over an explicit downstream
allow-list. `--ol-nl-replace-question` stays off (measurably worse, see
above). The scope-aware integration fix below is adopted independently and
stacks with this -- it doesn't touch prompt engineering or NL-plan quality
at all, so there's no tension between the two decisions.

### Scope-aware integration fix (`signal_scope.py`) — adopted

Separately from the NL-plan-quality debate above: of the original run's 98
syntax-fail rows, 21 distinct "real RTL identifiers used at the wrong
scope" (see "A second gap") are real, valid signals -- just not reachable
as bare identifiers from `uart2bus_top`, the module every assertion gets
spliced into (`ce_16` alone caused 34 failures). `signal_scope.py` fixes
this mechanically, with no prompt/generation changes at all:

1. **`build_signal_scope_map`**: statically parses the 6 RTL files, builds
   `{internal_signal_name: hierarchical_prefix}` for every internal (non-
   port) wire/reg/integer declared in exactly ONE of the 6 modules (28
   found -- e.g. `ce_16` → `uart1.` since it's declared inside `uart_top`,
   `main_sm` → `uart_parser1.`). Deliberately skips genuinely AMBIGUOUS
   names declared identically in *different* sibling modules (`ce_1`/
   `count16`/`bit_count`/`data_buf`, each independently declared in both
   `uart_rx.v` and `uart_tx.v` -- guessing which one would risk silently
   binding to the wrong net). Also collects every `` `define `` macro name
   across the 6 files.
2. **`qualify_out_of_scope_references`**: rewrites a bare reference to a
   mapped internal signal into its qualified path (`ce_16` → `uart1.ce_16`)
   and a bare reference to a known macro into a proper invocation
   (`MAIN_ADDR` → `` `MAIN_ADDR ``, since a bare macro name is just an
   ordinary, unresolvable identifier reference in Verilog, not a macro
   call). Skips anything already preceded by `.`/`` ` `` (won't correct an
   already-wrong hierarchical guess, only adds missing qualification).
3. **`hoist_defines`** (in `score_uart_assertionforge.py`): Verilog macro
   preprocessing is single-pass top-to-bottom, so a `` `define `` in
   `uart_parser.v` (concatenated LAST in `RTL_FILE_ORDER`) isn't yet
   defined at the point earlier in the file where the assertion is spliced
   in -- confirmed live, `` `MAIN_ADDR `` still failed as "undefined macro"
   even once correctly backtick-qualified, until all `` `define `` lines
   are hoisted to the very front of the combined text.

Both fixes are wired into `score_uart_assertionforge.py`, applied
automatically before every JasperGold check. Validated live on the 58
originally-syntax-failing rows (of the 98 total) that reference a name
either fix can address: **0/58 → 24/58 (41.4%) now syntax-OK**, purely from
these two mechanical fixes (21/58 from scope-qualification alone, +3 more
from macro-hoisting). The remaining 34/58 fail for unrelated reasons
confirmed by inspection -- a co-occurring pure hallucination in the same
row, or a genuine SVA-construction bug (e.g. `main_sm == 'hMAIN_ADDR`, the
model gluing a macro name onto a hex-literal base specifier instead of
using `` ` ``).

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
| Pipeline, `clock_signal` fix + the `skip_signal_list_note=False` experiment (not the adopted default -- see "A second gap") | 36 | **33 (91.7%)** | **13 (36.1%)** |

The baseline's remaining 22 syntax failures are dominated by exactly the
signal-scope/hallucination pattern from "A second gap" above (`ce_16` x16,
`global_clock_freq` x8, `counter` x6, `baud_rate` x6, plus a few undefined
macros) -- expected, since `--no-rag` never gets the `allowed_signals_note`
guardrail regardless of `skip_signal_list_note` (that flag only gates the
RAG/pipeline path; `generate_baseline_sva` doesn't take the note as a
parameter at all). Files: `results/pilot_baseline_baudclk_baudfreq.csv` /
`results/pilot_baseline_baudclk_baudfreq_jgscore.csv`.

### Fixing "A second gap" at the source: AssertionForge Stage 2 rework (2026-09-02)

Everything above treats AssertionForge's Stage 2 NL plans as fixed input and
patches the NL2SVA side (`--ol-nl-grounding`, `skip_signal_list_note`,
`signal_scope.py`). This section instead reworks `AssertionForge/src/
gen_plan.py` itself -- a `baud_clk`/`baud_freq`-only pilot (`valid_signals`
still the full 18-signal list; `max_num_signals_process` temporarily capped
to 2 signals for iteration speed) -- to make Stage 2 generate more precise,
signal-grounded plans directly, rather than relying entirely on downstream
cleanup.

**Two-step generation, mirroring Hybrid-NL2SVA's own Stage 1 OL-NL
grounding.** The original `construct_static_nl_prompt` asked one LLM call to
both invent a property AND state it with full operator-level precision in
the same breath. Split into `construct_idea_prompt` (Step 1: free-form
property ideas, no precision rules at all) and `construct_ol_nl_grounding_
prompt` (Step 2: grounds Step 1's ideas into "OL NL" form -- carries the
valid-signal whitelist, an operator-level/sequential-vs-combinational
discipline, vague-qualifier/trailing-rationale/gcd bans, the same `SVA
Operator Context` table Stage 1 itself uses, and worked examples -- literally
the same grounding task Stage 1 performs, just batched over several ideas per
call instead of one-at-a-time). `generate_dynamic_nl_plans` now calls both in
sequence per retrieved context.

**Confirmed negative-priming pattern**, consistent with earlier findings in
this project (naming a specific bad SVA-syntax example in a NL2SVA prompt
tends to make the model reproduce it, not avoid it -- see `Src/` for the
FVEval-side version of this lesson): every prompt revision that named a
*specific* thing to avoid (bad operator syntax, vague words, "abstract
spec-level quantities") measurably made that exact category worse across
several rounds, while revisions that stayed purely positive (list the valid
signals, show one worked example of correct behavior) reliably helped. Full
per-round data lives in git history (`gen_plan.py` commit messages,
2026-09-02) -- not reproduced here since the numbered "regenN" comparison
files themselves were transient debugging artifacts, deleted once the
pipeline settled.

**The hardest residual case traced to real RTL documentation, not noise or
hallucination.** `baud_freq` plans kept citing `baud_rate`/`global_clock_
freq`/`gcd` -- never real signals anywhere in the 6 RTL files -- no matter
how the prompt was reworded, up to and including a dedicated worked example
showing how to drop an ungroundable concept. Root cause: `baud_gen.v`'s own
header comment verbatim documents `baud_freq = 16*baud_rate / gcd(global_
clock_freq, 16*baud_rate)` -- genuine, directly-relevant design
documentation (`baud_freq` is an *input* to `baud_gen`; the comment describes
how an external register-loader is supposed to compute it) that the model has
good reason to want to cite. Since the content is true and salient, not
noise, prompt-level bans couldn't reliably suppress it.

**Mechanical output filter, not prompt engineering, closed this one.**
`filter_invalid_signal_references` (and its index-preserving twin,
`filter_invalid_signal_references_paired`) drops any grounded plan
referencing a snake_case, signal-shaped token not in `valid_signals` --
applied as a pure post-hoc check, never fed back into a prompt, so it carries
none of the negative-priming risk above. Restricted to snake_case
(underscore-joined) tokens specifically to stay high-precision against real
signal names in this codebase, rather than flagging ordinary English words.

**Validated comparison, same 44 underlying properties.** To isolate
grounding's actual contribution from unrelated batch-to-batch sampling
variance, `generate_dynamic_nl_plans` also tracks each grounded plan's
originating raw idea by index (`matched_raw_ideas.txt` / `matched_grounded_
plans.txt`, same `Plan N:` numbering in both files, N referring to the same
underlying idea in each) -- letting the identical 44 properties be tested
both ways instead of two independently-sized/sampled batches (an earlier,
methodologically flawed attempt compared 58 raw ideas against 48 *unrelated*
grounded plans from a different sampling run, and showed only a marginal
difference -- not reproduced here for that reason):

| | n | #SynC | #Proven |
|---|---|---|---|
| Raw ideas (ungrounded Step 1 output) + `--no-rag` 0-shot gpt-4o baseline | 44 | 30 (68.2%) | 15 (34.1%) |
| The SAME 44 ideas, grounded (Step 2 + filter) + full Hybrid-NL2SVA pipeline | 44 | **40 (90.9%)** | **24 (54.5%)** |

+22.7pp #SynC, +20.4pp #Proven from grounding + the full pipeline together,
on identical underlying properties. Files: `nl_plans_uart_matched_rawideas_
regen23_baudclk_baudfreq.txt` / `nl_plans_uart_matched_grounded_regen23_
baudclk_baudfreq.txt` (inputs), `results/assertionforge_uart_regen23_
matched_raw_baseline_jgscore.csv` / `results/assertionforge_uart_regen23_
matched_grounded_pipeline_jgscore.csv` (JasperGold-scored outputs).

**Scope**: `baud_clk`/`baud_freq` only (2/18 signals) -- a pilot to validate
the two-step-generation + mechanical-filter approach before spending it on
the remaining 16 signals and a full `nl_plans_uart.txt` regeneration.
`AssertionForge/src/gen_plan.py`'s `config.py` still has `max_num_signals_
process` temporarily capped and `valid_signals` set for this 2-signal pilot
(see the `# TEMP` comment there) -- restore before any full-scale rerun.

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
  `results/assertionforge_uart_gpt-4o_dynamicrag_jgscore.csv`. Also applies
  the scope-aware integration fixes (`qualify_out_of_scope_references` +
  `hoist_defines`, see "Scope-aware integration fix" above) to every row
  before checking.
- `signal_scope.py` -- the scope-aware integration fix's static-analysis
  module (`build_signal_scope_map`, `qualify_out_of_scope_references`);
  see "Scope-aware integration fix" above for the full writeup.
- `nl_plans_uart.txt` -- AssertionForge Stage 2's output, grouped by signal,
  **filtered to the 256 properties for the 14 design-controlled signals**
  (originally 323 across all 18; see above). The `results/` CSVs were scored
  from the original unfiltered 323.
- `nl_plans_uart_matched_rawideas_regen23_baudclk_baudfreq.txt` /
  `nl_plans_uart_matched_grounded_regen23_baudclk_baudfreq.txt` -- the
  matched-pair, `baud_clk`/`baud_freq`-only pilot from "Fixing 'A second gap'
  at the source" above: 44 properties each, same `Plan N:` numbering across
  both files (N is the same underlying idea, ungrounded vs. grounded). NOT
  the canonical `nl_plans_uart.txt` input -- a standalone comparison
  artifact.
- `results/assertionforge_uart_gpt-4o_dynamicrag_slim.csv` -- one row per
  property: `task_id`, `signal`, `nl_property` (the NL text), `response`
  (the generated SVA), `signals_for_validity`. (The full LMRESULT-shaped CSV
  `run_uart_nl2sva.py` writes also embeds the ~32KB combined UART testbench
  in every row, which is regenerable from the vendored `AssertionForge/` +
  `AssertLLM/`'s RTL and wasn't worth committing at ~31MB; this slim version
  is the one actually checked in.)
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

   **Update (2026-08-31)**: several real fixes/decisions since the
   `results/` CSVs (323/256-row) were generated -- those still reflect the
   ORIGINAL config below. A full rerun with the current defaults hasn't
   been done yet.
   - `clock_signal` wasn't actually threaded through `wrap_property_
     expression`/`jg_driven_syntax_cleanup`/`generate_rag_sva`/`process_row`
     in `run_rag_on_fveval_benchmarks.py` despite `run_uart_nl2sva.py`
     setting `args.clock_signal = "clock"` -- restored (real bug, always
     on).
   - `--ol-nl-grounding` (Step 1) is now ON by default (pass
     `--no-ol-nl-grounding` to turn it off) -- **adopted**, see "The
     'human route'" above. `--ol-nl-replace-question` stays off by
     default -- tested and found to make things worse.
   - `skip_signal_list_note=False` + `ALLOWED_SIGNALS` -- tested,
     measurably helped more than `--ol-nl-grounding`, but **deliberately
     NOT adopted**; `skip_signal_list_note` stays at its default `True`.
     See "A second gap" for why.
   - Scope-aware integration (`signal_scope.py`'s `qualify_out_of_scope_
     references` + `hoist_defines`) is now applied automatically inside
     `score_uart_assertionforge.py` -- **adopted**, orthogonal to the NL-
     plan-quality decisions above (pure mechanical identifier-path repair,
     no prompt/generation changes). See "Scope-aware integration fix".

## Extending

Next candidates from QiMeng-CodeV-SVA's Table 5: APB, ETHMAC, OPENMSP430,
SOCKIT (all sourced the same way -- 4 from AssertLLM, APB from OpenCores'
[`apb_mstr`](https://opencores.org/projects/apb_mstr)). Also worth trying:
AssertionForge's own built-in NL2SVA generator (`generate_SVAs = True`) as a
head-to-head baseline against Hybrid-NL2SVA on the *same* Spec2NL properties,
and swapping DeepSeek-V4-Flash / qwen3 in as AssertionForge's Spec2NL
backend instead of gpt-4o.
