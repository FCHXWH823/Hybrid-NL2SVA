# End-to-end Spec2SVA: AssertionForge + Hybrid-NL2SVA (UART pilot)

Combines [AssertionForge](https://github.com/NVlabs/AssertionForge) (NVlabs,
LAD 2025) as a **Spec2NL** front-end (spec PDF + RTL → natural-language
verification properties) with this repo's own RAG+SOR pipeline as the
**NL2SVA** back-end, closing the loop from a real specification document all
the way to formally-checked SVAs -- as opposed to Part I's evaluation, which
runs NL2SVA alone against FVEval-Verified's already-curated, single-property-
per-row benchmark rows.

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
  | Design-controlled only (excl. `ser_in`/`int_rd_data`/`int_gnt`/`ce_16`) | 256 | — | **52 (20.3%)** |
  | Excluded (free inputs / out-of-scope) | 67 | — | 8 (11.9%) |

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

## Files

- `run_uart_nl2sva.py` -- feeds `nl_plans_uart.txt` through Hybrid-NL2SVA's
  `process_row` (same machinery the main pipeline uses), writing
  `results/assertionforge_uart_gpt-4o_dynamicrag.csv`. Supports `--resume`
  for backfilling after an API rate-limit/credit interruption.
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

## Vendoring: what we changed in `AssertionForge/`

AssertionForge ships `src/utils_LLM.py` and `src/utils_LLM_client.py` as
empty "add your own LLM client" stubs, and is missing `src/load_result.py`
entirely (only used by a code path this pilot never exercises, but imported
unconditionally by `gen_plan.py`). `AssertionForge/src/` here has our
implementations of all three in place of upstream's. `utils_LLM_client.py`'s
`_HYBRID_NL2SVA_CONFIG` constant points at this repo's `Src/Config.yml`.

`AssertionForge/src/config.py` is the actual config used for the UART Stage
1/2 runs, with the AssertLLM/GraphRAG-template paths that were specific to
our scratch environment genericized back to the `<path>/<to>/...`
placeholder style the file already uses for every other design branch --
fill those in for your own AssertLLM clone (see step 1 below) before
re-running Stage 1/2. Its `ROOT = Path(__file__).resolve().parents[N]` line
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

1. Clone [AssertLLM](https://github.com/hkust-zhiyao/AssertLLM) (spec + RTL
   for UART and 4 other designs) somewhere.
2. In `AssertionForge/src/config.py`, fill in the `<path>/<to>/...`
   placeholders for `design_name == 'uart'` (both the `task = 'gen_plan'` and
   `task = 'build_KG'` branches) with your AssertLLM clone's paths, plus
   `env_source_path`/`settings_source_path` (a GraphRAG `.env`/`settings.yaml`
   template -- `python -m graphrag.index --init --root <dir>` generates a
   starting point; set `.env`'s `GRAPHRAG_API_KEY`, and in `settings.yaml` set
   the model to `gpt-4o` and `snapshots.graphml: true`, matching the actual
   run).
3. `cd AssertionForge && pip install -r requirements.txt` in a fresh venv --
   as shipped, 3 lines are broken and need removing first: `install==1.3.5`
   (nonexistent PyPI version), `py_sv_parser==0.3.0` (fails to build against
   current maturin/Rust), and the whole `pyautogen`+`llama-index-*` cluster
   (mutually incompatible `openai` version pins; neither is actually imported
   anywhere in `src/`). Then `pip install graphrag==0.3.6 gitpython`
   separately (matching AssertionForge's `python -m graphrag.index` CLI
   expectations; `graphrag` itself isn't pinned in `requirements.txt` at all).
4. Run Stage 1 (`task = 'build_KG'`) then Stage 2 (`task = 'gen_plan'`,
   `subtask = 'actual_gen'`, `generate_SVAs = False`) via `python main.py`
   from `AssertionForge/src/`. Stage 2's log directory (under
   `AssertionForge/logs/`) contains `nl_plans.txt` (copy it in as
   `nl_plans_uart.txt` here, or point `NL_PLANS_PATH` at it).
5. From this repo's root: `python3 end2end_evaluation/run_uart_nl2sva.py
   --uart-rtl-dir <AssertLLM clone>/rtl/uart --workers 6`, then
   `python3 end2end_evaluation/score_uart_assertionforge.py --workers 6`.

## Extending

Next candidates from QiMeng-CodeV-SVA's Table 5: APB, ETHMAC, OPENMSP430,
SOCKIT (all sourced the same way -- 4 from AssertLLM, APB from OpenCores'
[`apb_mstr`](https://opencores.org/projects/apb_mstr)). Also worth trying:
AssertionForge's own built-in NL2SVA generator (`generate_SVAs = True`) as a
head-to-head baseline against Hybrid-NL2SVA on the *same* Spec2NL properties,
and swapping DeepSeek-V4-Flash / qwen3 in as AssertionForge's Spec2NL
backend instead of gpt-4o.
