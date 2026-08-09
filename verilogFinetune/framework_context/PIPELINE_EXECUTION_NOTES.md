# Pipeline Execution Notes (this server)

Companion to `OL_NL_DFS_FRAMEWORK.md`, which describes the pipeline's design.
This file documents what actually happened running it end-to-end on this
server: environment setup, bugs the run surfaced (and fixes), and the
production run's history. `OL_NL_DFS_FRAMEWORK.md`'s own "Status at time of
writing" section was accurate as of when it was written -- Stages 1-6 were
implemented but "Nothing has been run against a real JasperGold install or a
served fine-tuned checkpoint." Everything below is what changed once both of
those became available here.

## Environment setup

Nothing in this section is pipeline logic -- it's what this specific machine
needed before the pipeline could run at all.

- **JasperGold**: not on `PATH` by default. `module load
  /home/shared/modules/cadence/IC231` puts `jg` on `PATH` (confirmed:
  `2026.03p001 64 bits`). Needed by anything that calls
  `jasper_equiv_check.run_equivalence_check` -- Stage 2's validation loop and
  the Stage 2+3+4 orchestrator.
- **Python packages**: this host had no `pip` at all. Bootstrapped via
  `python3 -m ensurepip --user`, then `pip install --user openai pyslang` (both
  required by every generation script; neither was present). `tiktoken` was
  also installed, for the token-count cost estimation below (not a pipeline
  dependency).
- **Source dataset**: `verilogFinetune/data/CodeV-SVA-dataset-training-83K.jsonl`
  is `.gitignore`d (exceeds GitHub's 100MB limit) and wasn't present after
  cloning. Downloaded from
  `https://huggingface.co/datasets/wyt2000/CodeV-SVA-datasets` and placed at
  the path every script's `--source`/`--input` default already expects.
  Verified: 83,195 lines, `{messages: [system, user, assistant]}` shape,
  matching the framework doc's stated corpus size exactly.

## Bugs found and fixed

All five were found by actually running the pipeline against live JasperGold
and a live model endpoint -- none were visible from reading the code alone,
which is why they survived the original "never executed end-to-end" state.

### 1. `sva_graph.render_merge_tree` and the DFS node-walker assumed unary/binary only

A system-function call with 3 real arguments (e.g. `$past(x, N, gate)`)
produces an operator node with 3 children, which the whole Stage 3 pipeline
had no path for:

- `sva_graph.py`'s `render_merge_tree` only ever formatted `operand[0] {op}
  operand[1]`, silently dropping any 3rd+ operand from the symbolic
  derivation.
- `generate_dfs_explanation.py`'s `walk_tree` only recursed into
  `children[0]`/`children[1]` (gated behind a `binary = len(children) == 2`
  check), so a 3rd child that was itself an operator (not a bare signal) was
  never visited -- leaving a gap in `parsed_by_id` that crashed
  `render_decomposition_tree` with a `KeyError` when it later tried to render
  that node.

**Fix**: generalized both to arbitrary arity -- `render_merge_tree` uses
function-call-style `op(T1, T2, T3, ...)` for anything but exactly 2 operands;
`walk_tree`/`build_node_prompt`/`call_node_llm` now loop over every child and
use a per-operand output-field letter (`c`, `d`, `e`, ...) instead of
hardcoding `c`/`d`. Verified with a synthetic `$past(counter, 1, enable_pulse
&& reset_n)` node (3 children, 3rd itself an operator) -- confirmed the old
code path would `KeyError` on it and the new one renders both trees cleanly.

### 2. `jasper_equiv_check.run_equivalence_check` wrote the `.sva` file relative to the wrong process's cwd

`jg` is invoked with `cwd=fveval_dir` (the FVEval directory), and its tcl
script resolves `${SV_DIR}` *relative to that*, not to the calling Python
process's cwd. With the pipeline's default `--sv-dir`
(`verilogFinetune/data/ol_nl_validation_scratch`, a relative path), the
`.sva` file actually got written under
`.../Hybrid-NL2SVA/verilogFinetune/...`, while `jg` looked for it under
`.../Hybrid-NL2SVA/Evaluation/FVRuleLearner/FVEval/verilogFinetune/...` --
every single equivalence check failed with `ERROR (ESW046): file does not
exist` before any real analysis ran. This read as "not equivalent" (`is_equivalent`
only accepts an explicit `Full equivalence` string) with no distinguishing
signal, so it looked exactly like every OL-NL statement in a 50-record
validation batch being wrong, instead of a path bug -- caught by manually
running the exact same call with full (non-truncated) `jg` output instead of
the pipeline's own 200-char-truncated log line.

**Fix**: `sv_dir = os.path.abspath(sv_dir)` at the top of
`run_equivalence_check`, before the file is written and before it's passed to
`jg`. Verified against the same record that failed before the fix -- now
returns `Full equivalence`.

### 3. The Stage 2+3+4 orchestrator's `worker()` couldn't tell "bad candidate" from "infrastructure failure"

`worker()` treated any exception from `process_one()` (API rate limits,
network errors, account billing failures) identically to a clean `None`
return (the candidate's OL-NL statement genuinely never verified): both
triggered an immediate, permanent backfill -- discard this slot, pull a fresh
record from the source pool, try again. That's correct for a genuine
per-record failure, but catastrophic for a sustained infrastructure failure:
when the DeepSeek account ran out of balance mid-run (see below), every one
of 24 concurrent workers hit an instantly-rejected call, backfilled
immediately, hit the same rejection on the new record, backfilled again, in
a tight loop with no backoff -- racing through the entire remaining eligible
pool of the 83K-record source file until none were left, at which point the
whole process crashed with an unhandled `StopIteration: Ran out of backfill
candidates`.

**Fix**: exceptions from `process_one()` are now retried on the *same*
record with exponential backoff (up to `--max-retries` attempts) before
falling through to backfill; a clean `None` return still backfills
immediately, unchanged. Bounds the blast radius of a transient outage to a
few backoff cycles instead of the entire remaining corpus.

### 4. The backfill search space was a 28-line sliver of an 83,195-line file

`next_backfill_candidate`'s scan cursor was initialized to `max(target_indices)
+ 1` -- "start looking for replacement candidates right after the highest of
the 5,000 originally-sampled indices." For this particular 5,000-record sample
drawn from an ~82.8K-record eligible pool, the highest target index happened
to be 83,166, out of 83,194 total -- so the *entire* searchable range for
every backfill, for the whole run, across every resume, was 28 lines wide.
The other ~77,800 unused-but-eligible records earlier in the file (the vast
majority of the actual pool) were structurally unreachable; the cursor never
looked there and never wrapped around. This produced the same crash signature
as bug #3 (`StopIteration: Ran out of backfill candidates`) but for a
completely different reason -- confirmed by checking that the account balance
was healthy at the time of that particular crash, and that the last several
successful `resolved_index` values before it were all clustered at
82,500-83,166.

**Fix**: cursor now starts at `0`. `used_indices` is already seeded with all
5,000 target indices at startup, so the scan just skips past those (and any
already-consumed backfills) instead of needing a narrower starting point.
Verified: post-fix, backfills immediately began landing on low indices
(e.g. `resolved_index=349, 353, 362-365`) that were previously unreachable in
any run.

### 5. Stage 2's own internal retry loop swallowed API/infra errors too, one layer below fix #3

Fix #3 only helps once an exception actually reaches `worker()`. But
`generate_ol_nl_explanation.generate_ol_nl()` had its *own* try/except around
the OL-NL rewrite API call, one layer further in: it caught the exception,
printed `"OL-NL call failed (...), retrying..."`, and retried *within its own
loop* -- and after exhausting its `max_retries` budget this way, returned a
perfectly normal `(text, False)`. No exception ever propagated to `worker()`,
so fix #3 never engaged. During the next DeepSeek balance outage this is
exactly what happened: **43,864** internal `OL-NL call failed` retries in one
resume (all `Insufficient Balance`), against only **12** exceptions that
actually reached the worker-level safety net -- and each of those thousands
of internally-exhausted attempts still triggered an immediate, no-backoff
backfill, one layer below where fix #3 could see it. Net effect that resume:
15,368 backfill candidates consumed for 569 net new completions.

**Fix**: removed the try/except around the API call inside `generate_ol_nl`'s
loop entirely. API/infra exceptions now propagate all the way up through
`process_one()` to `worker()`'s existing retry-with-backoff, which already
handles this correctly and uniformly for the whole record (Stage 2 *and*
Stage 3 together) -- rather than duplicating (and getting wrong) backoff
logic at every layer that happens to make an API call. The genuine
content-quality retries in the same loop (missing `OL NL:` label, no code
block, JasperGold says not-equivalent) are unchanged -- those really are
per-attempt content issues and correctly consume that loop's own budget.

## Known issue, not yet fixed

Carried over from reviewing `OL_NL_DFS_FRAMEWORK.md` against the code:
Stage 2's `SIGNAL_LIST` (`generate_ol_nl_explanation.build_signal_list`,
parses the golden SVA's tree for leaf identifiers) doesn't match how the real
FVEval harness builds it at Stage 6 scoring time
(`NL2SVAHumanEvaluator.evaluate_jg`, regexes quoted substrings out of the
free-text prompt instead). The two can diverge, which means Stage 2's
validation bar isn't guaranteed to match what Stage 6 will actually measure
for the same record, despite that being the explicit design goal. Not
addressed here -- flagged for a follow-up decision (switch Stage 2 to the
harness's method, or document the deviation as deliberate).

## Multi-provider support added

`generate_ol_nl_explanation.py`, `generate_dfs_explanation.py`, and
`generate_codev_sva_reasoning_dataset.py` gained a shared `--provider
{openai,deepseek}` flag (`add_provider_arg`/`build_llm_client`, added to
`generate_prompt_guided_explanation.py` since all three already import from
it). `deepseek` points the client at `https://api.deepseek.com` with
`DeepSeek_API_Key`, mirroring the convention already used throughout
`Src/DeepSeek/`. The orchestrator runs Stage 2 and Stage 3 through the same
client/model already, so this one flag covers both.

API key resolution order: `DEEPSEEK_API_KEY`/`OPENAI_API_KEY` environment
variable first, `Src/Config.yml` second -- so a key never has to be committed
to that (git-tracked) file to be used.

Confirmed live: `deepseek-v4-pro` resolves against `api.deepseek.com`
(current published pricing: input $0.435/1M tokens cache-miss, $0.003625/1M
cache-hit; output $0.87/1M tokens).

## Production run history

1. **50-record validation batch, 4 workers** -- failed 50/50 before fix #2
   above (see fix #2 for root cause); succeeded 50/50 after the fix (7
   backfills, 14% drop rate). Spot-checked output quality directly: coherent
   OL-NL statements, correctly nested decomposition/merge trees, valid final
   SVA. Written to `data/codev_sva_ol_dfs_validation50.jsonl` (not part of
   the registered training dataset -- a throwaway validation artifact).
2. **Full 5,000-record run, 24 workers, launched** against
   `data/codev_sva_5000_sample.json` / the downloaded 83K source, writing to
   `data/codev_sva_ol_dfs_5000.jsonl` (checkpointed to
   `data/codev_sva_ol_dfs_5000.checkpoint.jsonl`). Reached 1,345/5,000 with a
   healthy ~5% backfill rate and zero errors, then the DeepSeek account
   balance ran out mid-run -- see fix #3 for what that triggered. The
   process crashed; checkpoint preserved all 1,345 completed records.
3. Applied fix #3, balance topped up, **resumed** -- reached 3,700/5,000
   before crashing again. This time the balance outage was long enough that
   even fix #3's bounded backoff couldn't prevent eventual pool exhaustion
   (it only slows a sustained outage, it can't survive one indefinitely);
   real per-record cost also turned out ~4x higher than the original
   token-based estimate (~¥0.085/record empirically, vs. ~$0.0028 estimated).
4. Balance topped up again, resumed -- crashed almost immediately, this time
   with the balance confirmed *healthy*. Root-caused to fix #4's bug (the
   28-line backfill sliver); applied fix #4.
5. Resumed again -- reached 4,831/5,000 with 15,368 backfills consumed for
   only 569 net completions (the huge discrepancy was the tell). Root-caused
   to fix #5's bug (Stage 2's own internal retry loop swallowing infra
   errors one layer below fix #3); applied fix #5; stopped the process
   manually rather than let it keep burning the pool once diagnosed.
6. Balance topped up a third time, resumed for the last 169 records --
   **completed cleanly**, zero errors, 197 backfills (routine
   verification-driven, not infra-driven).

**Final result**: `data/codev_sva_ol_dfs_5000.jsonl`, 5,000/5,000 records.
Integrity-checked directly: all 5,000 lines are valid JSON, correct
`[system, user, assistant]` role order, every assistant turn has both the
`OL NL:` prefix and a final ` ```systemverilog ` block. Already registered in
`data/dataset_info.json` as `codev_sva_ol_dfs_5000` (sharegpt format), ready
for Stage 5 (`train_codev_sva_ol_dfs.sh` / `finetune_codev_sva_ol_dfs.sbatch`)
with no further changes needed.

The pattern across steps 2-6 is worth naming: each crash looked identical
from the outside (`StopIteration: Ran out of backfill candidates`) but had a
different root cause -- a genuine sustained outage, a search-range bug, and a
second infra-error-handling gap one layer below the first fix. Checking the
*actual* signal (account balance, backfill-to-completion ratio, where in the
file the failures clustered) before assuming "it's the same issue again" is
what found #4 and #5; assuming the fix from #3 covered everything would have
missed both.

### Checking status (while a run is in progress)

```bash
wc -l verilogFinetune/data/codev_sva_ol_dfs_5000.checkpoint.jsonl   # completed count (ground truth -- explicitly flushed)
ps -p "$(cat verilogFinetune/data/full5000.pid)"                    # still running?
tail -n 50 verilogFinetune/data/full5000.log                        # recent activity (stdout is buffered when
                                                                      # redirected to a file, so this can lag
                                                                      # the checkpoint count)
```
