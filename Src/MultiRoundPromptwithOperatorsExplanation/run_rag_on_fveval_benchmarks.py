"""
Runs the customized RAG pipeline (dynamic-splitting code database +
HybridRetrieval + SVA operator-based rechecking, as in
Dynamic-RAG-Openai-4o-mini-MultiRoundPrompted-Assertion-Generation-1assert1iteration.py)
over one of the three FVEval benchmark CSVs and writes an LMResult-shaped CSV
that the existing FVEval harness (Evaluation/FVRuleLearner/FVEval/run_evaluation.py
-- NL2SVAHumanEvaluator / NL2SVAMachineEvaluator) can consume as-is, via its
`llm_output_dir` glob of `*{model_name}_*.csv`. No changes are made to that
harness; this script only produces its expected input, mirroring exactly how
verilogFinetune/run_codev_sva_ol_dfs_eval.py does the same thing for the
fine-tuned-model research line -- this is the RAG-pipeline counterpart.

The three benchmarks (real row counts, not raw wc -l on the multiline CSVs):
    nl2sva_human                  79 rows  (real testbench + tb_reset scaffold)
    nl2sva_machine                300 rows (synthetic dummy testbench)
    module_sva_nl_manual_editing  1000 rows (bare module_interface, no testbench --
                                              this is the "nl2sva_opencore" scoring
                                              task; its evaluator never reads
                                              output_tb, so the same testbench-with-
                                              marker convention used for the other
                                              two is harmless here too)

Prompt shape (system prompt, testbench-with-marker + "Question: ..." + output
format instructions) is reused verbatim from run_codev_sva_ol_dfs_eval.py so the
harness's response-parsing regexes (which expect the same "posedge clk)" /
"tb_reset)" conventions) see the same shape of answer regardless of which
research line produced it. What's added on top, per row:
    1. HybridRetrieval: a global-semantic retrieval pass (the user prompt itself
       queries the persisted code-centric-chunk database) plus a keyword/
       operator-guided pass (extract_keywords + extract_related_operators_of_keyword,
       same as the main pipeline script), combined into one generation call.
    2. SVA operator-based rechecking via the bottom-up explanation-merge tree
       (sva_tree/explanation_merge_tree.py): the generated SVA is parsed into its
       operator/signal tree and a node-by-node natural-language meaning is
       composed bottom-up, then compared against the row's own NL prompt --
       falls back to no rechecking context for the ~15% of SVAs sva_graph.py
       can't parse (same fallback as the main pipeline script).
    3. A few syntax-only cleanup passes (unmatched parentheses etc.).

Usage:
    python3 Src/MultiRoundPromptwithOperatorsExplanation/run_rag_on_fveval_benchmarks.py \\
        --task nl2sva_human \\
        --limit 10
"""
import argparse
import concurrent.futures
import csv
import json
import os
import re
import sys
import threading

import yaml
from openai import OpenAI

# rag_database's sqlite3 patch must run before anything imports chromadb --
# import it first, ahead of extract_keywords.py's own Chroma import.
from rag_database import build_rag_system

from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from extract_keywords import extract_keywords, extract_related_operators_of_keyword

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_REPO_ROOT = os.path.join(_THIS_DIR, "..", "..")
sys.path.insert(0, os.path.join(_REPO_ROOT, "sva_tree"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "verilogFinetune"))
sys.path.insert(0, os.path.join(_REPO_ROOT, "Evaluation", "FVRuleLearner", "FVEval"))
from explanation_merge_tree import build_and_render_explanation_merge_tree
from run_codev_sva_ol_dfs_eval import (
    SYSTEM_PROMPT,
    QUESTION_TEMPLATE,
    build_testbench_with_marker,
    build_user_prompt,
    iter_rows,
)
# fv_eval's own prompt string constants (pure data, no imports of their own --
# unlike fv_eval.evaluation/fv_tool_execution, safe to import directly). Used
# to build the *exact* FVEval-0-shot prompt shape for nl2sva_human, after
# discovering run_codev_sva_ol_dfs_eval.py's QUESTION_TEMPLATE (borrowed from
# a different, training-data-shaped task) was missing the official prompt's
# "Do not add code to output an error message string." instruction -- and a
# real generated response (counter_2) hit exactly that failure mode as a
# result. TODO: nl2sva_machine/module_sva_nl_manual_editing still use the
# borrowed QUESTION_TEMPLATE below; give them the same official-prompt
# treatment (prompts_nl2sva_machine.py / benchmark_launcher.py's
# NL2SVAMachineLauncher) before trusting their functionality-match numbers.
from fv_eval import prompts_nl2sva_human

# jasper_direct_equiv_check.check_sva_elaboration: real JasperGold
# elaboration-only check (no equivalence, no proof) used by generate_rag_sva's
# syntax-cleanup loop -- see that function's docstring.
from jasper_direct_equiv_check import check_sva_elaboration, summarize_elaboration_errors
# score_nl2sva_human's more robust body/signal extraction (handles any
# disable-iff clause, not just the literal "tb_reset)" jasper_equiv_check.py
# itself looks for) -- reused here instead of duplicated.
from score_nl2sva_human import extract_property_body, build_signal_list

DEFAULT_CSV_PATHS = {
    "nl2sva_human": "Evaluation/FVRuleLearner/FVEval/data_nl2sva/data/nl2sva_human.csv",
    "nl2sva_machine": "Evaluation/FVRuleLearner/FVEval/data_nl2sva/data/nl2sva_machine.csv",
    "module_sva_nl_manual_editing": "Evaluation/FVRuleLearner/FVEval/data_1k/module_sva_nl_manual_editing.csv",
    # Human-expert-corrected nl2sva_human (73 of the original 79 rows -- 6
    # erroneous ones removed outright), from wyt2000/FVEval-Verified on
    # HuggingFace. Confirmed this fixes real annotation defects found this
    # session: e.g. the arbiter "holds onto grants" rows now list 'busy' and
    # 'last_gnt' among the prompt's "use these signals", where the original
    # nl2sva_human silently omitted them despite the golden requiring both.
    "nl2sva_human_verified": "Evaluation/FVEval-Verified/fveval_nl2sva_human.jsonl",
    # Human-expert-corrected nl2sva_machine (283 rows), same wyt2000/
    # FVEval-Verified source as nl2sva_human_verified. Structurally
    # different from the two nl2sva_human variants in ways that matter to
    # this pipeline: the "problem" text here is ALREADY operator-level NL
    # (names real signals directly, e.g. "sig_F and sig_H are high"), so
    # Step 1 grounding is redundant by construction -- just don't pass
    # --ol-nl-grounding for this task, same as any other row where it's
    # not wanted. Its bare dummy testbenches (module ports only, no
    # internal logic) declare no reset signal at all -- confirmed 0/283
    # rows mention "reset" anywhere -- and golden answers have no disable
    # iff clause, so this task is threaded through with disable_signal=None
    # everywhere a complete SVA gets assembled or elaboration-checked
    # (wrap_property_expression, check_sva_elaboration), unlike the fixed
    # `tb_reset` convention every nl2sva_human(_verified) row uses.
    "nl2sva_machine_verified": "Evaluation/FVEval-Verified/fveval_nl2sva_machine.jsonl",
}

LMRESULT_FIELDNAMES = [
    "experiment_id", "task_id", "model_name", "response", "ref_solution",
    "design_rtl", "output_tb", "user_prompt", "cot_response", "signals_for_validity",
]

# nl2sva_human-specific: every one of the 79 testbenches declares a signal
# named exactly `tb_reset` (verified against every ref_solution's disable
# iff clause) for use in the assertion's disable iff -- but many of those
# same testbenches also declare derived/delayed variants (tb_reset_d1,
# tb_reset_d2, tb_reset_1_cycle_pulse_shadow, ...) that are relevant to the
# property's own logic. Real generations (e.g. counter_0) have picked one of
# those derived variants for the disable iff clause itself, which is a
# genuine functional mismatch against the golden, not just a style
# difference. Not applicable to nl2sva_machine/module_sva_nl_manual_editing,
# whose testbenches don't follow this convention -- only injected when
# args.task == "nl2sva_human".
# NL2SVA_HUMAN_RESET_NOTE (REMOVED 2026-08-10): used to tell the model
# "use exactly `tb_reset` in the assertion's disable iff (...) clause" --
# written for the OLD design, before the expression-only redesign below,
# where the model wrote the full statement including disable iff itself.
# Under the current architecture the model NEVER writes disable iff at all
# (wrap_property_expression adds `disable iff (tb_reset)` mechanically,
# always correctly, with zero model involvement) -- so the note had
# nothing legitimate left to do, and confirmed live it was actively
# harmful: it explicitly primed the model with the name `tb_reset` and a
# reason to reference it, directly undermining allowed_signals_note's
# "reference ONLY these signals" instruction whenever tb_reset wasn't
# itself one of the row's approved signals (FVEval-NL2SVA-Human-37: the
# model added a redundant `&& !tb_reset` inside the bare expression,
# duplicating what disable iff already handles, weakening the property
# just enough to turn a full equivalence into a one-directional-only
# match). generate_baseline_sva (official FVEval-0-shot fidelity) never
# used this note in the first place, so removing it here has no effect
# there.

# Our RAG+OL-NL flow (Step 1, Stage 2 RAG generation, Stage 3 SOR/cleanup --
# NOT generate_baseline_sva, which stays on the official FVEval-0-shot
# prompt/output shape for baseline fidelity) has the model generate ONLY the
# bare property expression, never the `assert property (@(posedge clk)
# disable iff (tb_reset) ...);` wrapper -- the clock/reset/label are always
# supplied mechanically via wrap_property_expression below. This eliminates,
# by construction rather than by hoping the model complies with a prompt
# instruction, the whole class of bugs already confirmed this session: using
# the wrong reset signal in disable iff (tb_reset vs
# tb_reset_1_cycle_pulse_shadow), and SOR/cleanup reordering or otherwise
# mangling the clock/disable-iff clause.
EXPRESSION_ONLY_INSTRUCTION = (
    "IMPORTANT OVERRIDE: even if an example elsewhere shows a complete `assert property (...)` "
    "statement, you must output ONLY the bare property expression itself -- the boolean/temporal-"
    "logic condition. Do NOT include a label, the `assert property (` / `);` wrapper, the "
    "`@(posedge clk)` clocking event, or a `disable iff (...)` clause. Those are added separately "
    "from a fixed template; including them yourself is unnecessary and risks getting the clock or "
    "reset signal name wrong."
)


def wrap_property_expression(expression, label="asrt", disable_signal="tb_reset"):
    """Mechanically builds a complete, well-formed SVA from a bare property
    expression, using this benchmark's fixed clock convention (every row's
    testbench exposes `clk`) -- the model is never trusted to produce the
    clock/disable-iff/label itself in our RAG+OL-NL flow.

    disable_signal: nl2sva_human(_verified) testbenches all expose exactly
    `tb_reset` for this. Pass None for testbenches with no reset signal at
    all (e.g. nl2sva_machine_verified's bare dummy modules -- confirmed
    none of its 283 rows declare one, and golden answers there have no
    disable iff clause either) to omit the clause entirely."""
    disable_clause = f" disable iff ({disable_signal})" if disable_signal else ""
    return f"{label}: assert property (@(posedge clk){disable_clause}\n    {expression.strip()}\n);"


def load_rich_operator_context(path="sva_temporal_operators.json"):
    """sva_temporal_operators.json (38 entries: type/original/
    natural_langage_explanation/example_usgae -- field names as spelled in
    the file) is far richer than the old operators.json (11 entries, bare
    name: one-liner) -- notably it actually includes `strong`/`weak`, which
    operators.json doesn't have at all. The failure-mode analysis of the
    plain 0-shot baseline found a systematic pattern of the model omitting
    `strong(##[0:$] ...)` for "eventually" claims; operators.json literally
    cannot teach it that operator exists. This is now the ONLY operator
    table used anywhere in this pipeline -- Step 1's SVA-generation calls,
    OL-NL grounding, HybridRetrieval's keyword-guided path
    (extract_related_operators_of_keyword), and SOR rechecking (both the
    explanation-merge-tree and the final recheck completion) -- the old
    operators.json-based operator_context has been fully retired."""
    with open(path) as file:
        data = json.load(file)
    lines = []
    for op, entry in data.items():
        lines.append(
            f"{op} ({entry['type']}): {entry['natural_langage_explanation']} "
            f"Example: {entry['example_usgae']}"
        )
    return "\n".join(lines)


def parse_code_response(text):
    """Strips a ```systemverilog ... ``` fence down to just the code, same
    convention as FVEval/fv_eval/utils.py's parse_code_response (duplicated
    here rather than imported, since fv_eval.evaluation -- unlike this
    module -- pulls in the harness's CLI-oriented config/saver globals as an
    import-time side effect)."""
    if "```systemverilog" in text:
        text = text.split("```systemverilog")[-1]
    if "```" in text:
        text = text.split("```")[0]
    return text.strip()


OL_NL_SYSTEM_PROMPT = (
    "You are a helpful bot that rewrites natural-language descriptions of SystemVerilog "
    "assertions into a precise, signal-grounded, operator-level form, following the "
    "requested format exactly."
)

# Adapted from verilogFinetune/generate_ol_nl_explanation.py's PROMPT_TEMPLATE_OL_NL,
# for inference time: that version takes the GOLDEN SVA as a direct input (it's a
# dataset-construction tool -- the rewrite is built and validated against a known-
# correct answer). We don't have a golden answer at inference time -- using
# ref_solution here to help generate would invalidate the evaluation -- so this
# grounds purely from the testbench, with no golden-equivalence validation loop.
# One best-effort call; if it fails to parse, the caller falls back to prompt_text.
#
# The worked example below is entirely invented (verified to not appear in any
# of the three FVEval benchmark CSVs -- grepped for its signal names first) --
# NOT copied from generate_ol_nl_explanation.py's own worked example, which
# turned out to be verbatim nl2sva_human's counter_1 row: reusing it here would
# have shown the model counter_1's own golden answer, disguised as NL, while
# grounding counter_1 itself, and primed every other counter_* row too via the
# shared "counter does not underflow" phrasing (nl2sva_human's own prompt text).
PROMPT_TEMPLATE_OL_NL_NO_GOLDEN = """You are given a SystemVerilog RTL testbench and a natural-language \
description of a property that should hold on it. The description may be written at any level of \
abstraction -- sometimes it already names signals and operators directly, sometimes it only \
describes the high-level intent without naming every signal involved.

Your job: rewrite the description as an "OL NL" (operator-level natural-language) statement -- one \
that names ONLY signals that actually appear in the testbench (never invented or paraphrased names), \
and whose clauses describe the property's structure as precisely and concretely as possible (e.g. for \
an implication-shaped property, state the antecedent condition, then the consequent, in that order). \
You are NOT given the correct assertion -- infer the grounded meaning from the testbench and the \
description alone, using standard RTL/verification idioms.

Worked example (abstract/domain-level input, grounded in the real signals of an unrelated design):
 Testbench (excerpt): a module with a status flag pending, its previous-cycle value pending_d1, a \
clear request clear_vld, and a saturation guard sat_guard.
 Description given: that the pending flag never gets stuck. Use the signals 'pending', 'pending_d1', \
'clear_vld', and 'sat_guard'.
 OL NL: It must never be the case that pending_d1 is asserted and clear_vld is not asserted and \
pending is still asserted and sat_guard is not asserted

Now do the same for this:

RTL testbench (for grounding signal names/widths/parameters only -- do not describe the RTL itself):
 {testbench}

Description given (any abstraction level):
 {question}

Output exactly one labeled line, in plain text (not JSON), and nothing else:

OL NL: <the operator-level, signal-grounded statement>
"""

_OL_NL_PATTERN = re.compile(r"OL NL:\s*(.*)", re.DOTALL)

# --ol-nl-conservative: appended to Step 1's extra_note when set. NOT a
# default -- reverse-engineering golden phrasing into a prompt rule is
# against this project's own standard (see EVENTUALLY_TIMING_NOTE's
# rejection); this note instead just asks Step 1 to be more literal, and
# is opt-in/togglable pending further testing.
OL_NL_CONSERVATIVE_NOTE = (
    "Stay as faithful and accurate to the original description as possible. Do not invent, "
    "infer, or imagine additional structure, conditions, or behavior that the description "
    "does not actually state -- only ground the description in real signal names and restate "
    "its own logic, without adding meaning, timing, or structure that wasn't there."
)

# --sor-conservative: appended to run_sor_recheck's system message when set.
# Motivation: confirmed live (2026-08-14, FVEval-NL2SVA-Machine-120/-280,
# --sor-template-timing 2-pass trials) that SOR's own revision, when it does
# decide a change is warranted, isn't reliably minimal -- e.g. dropping the
# `|->`/`|=>` implication operator entirely down to a bare sequence
# concatenation, or inserting a spurious `##1 1'b1` clause nobody asked
# for, rather than touching only the specific Tn node actually responsible
# for the flagged mismatch. Same principle as OL_NL_CONSERVATIVE_NOTE
# (opt-in, not reverse-engineered from any golden answer -- just a general
# minimal-edit instruction), applied to SOR's revision step instead of
# Step 1's grounding step.
SOR_CONSERVATIVE_NOTE = (
    "When revising, change ONLY the minimal part of the expression responsible for a genuine, "
    "clearly-identified mismatch -- do not restructure, drop, or add operators, clauses, or "
    "sub-expressions beyond what that specific mismatch requires. Do not invent additional "
    "structure, conditions, or timing that the property description does not actually state."
)

# --only-overlap-implication: appended to both Stage 2's generation note and
# SOR's extra_note when set. Sidesteps the whole |->/|=> confusion this
# session repeatedly traced (e.g. FVEval-NL2SVA-Machine-18/-106/-120/-205:
# the model writes `A |=> ##N B`, double-counting `|=>`'s own implicit
# +1-cycle advance on top of the explicit ##N, landing on N+1 total cycles
# instead of golden's N) at its root: rather than teaching the model to
# correctly juggle TWO delay-encoding conventions that compose additively
# (|=>'s hidden +1, plus ##N's explicit +N), this eliminates one of the two
# conventions entirely. If the model only ever writes `|->` and always
# spells the FULL total delay out explicitly via `##N`, there is nothing
# left to add together -- it only has to get one number right, not "N,
# given that this operator itself already silently contributes 1." Opt-in,
# not a default -- untested at scale yet.
ONLY_OVERLAP_IMPLICATION_NOTE = (
    "Always use the overlapped implication operator `|->` for property implications -- never use "
    "the nonoverlapped implication operator `|=>`. If the property requires a delay before the "
    "consequent (e.g. \"N cycles later\", \"at the next clock cycle\"), express that delay "
    "explicitly as `##N` (or a range `##[M:N]`) inside the consequent, using the TOTAL number of "
    "clock cycles from the antecedent -- e.g. write `A |-> ##1 B` for \"B holds one cycle later\", "
    "not `A |=> B`. Do NOT use `|=>`, which implicitly adds one extra cycle on top of any explicit "
    "`##N` you also write, making the total easy to miscount."
)


def generate_ol_nl_grounding(client, model_name, prompt_text, testbench, operator_context, extra_note=""):
    """Best-effort, no-golden OL-NL grounding call -- this IS Step 1: called
    once, used directly, no independent second candidate and no formal
    check against one (see the removed-code note above for why). Returns
    the grounded restatement, or prompt_text unchanged if the model's
    reply doesn't contain a parseable 'OL NL:' line.

    operator_context goes in the SYSTEM message, not repeated in the user
    prompt template. extra_note (e.g. a row's authoritative allowed-signals
    note) is appended after it."""
    prompt = PROMPT_TEMPLATE_OL_NL_NO_GOLDEN.format(testbench=testbench, question=prompt_text)
    system_msg = OL_NL_SYSTEM_PROMPT + "\n\nSVA Operator Context:\n" + operator_context
    if extra_note:
        system_msg += "\n\n" + extra_note
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": prompt},
        ],
    )
    match = _OL_NL_PATTERN.search(completion.choices[0].message.content)
    if not match:
        return prompt_text
    return match.group(1).strip().splitlines()[0].strip()


_QUOTED_SIGNAL_NAME_RE = re.compile(r"'([^'\s]+)'")


def extract_named_signals(user_prompt):
    """Just the signal names explicitly quoted in the question's own "Use
    the signals '...'" phrasing -- deliberately NOT build_signal_list's
    parameter/localparam scan, which exists for a DIFFERENT purpose
    (making sure prop_eq_checker's SIGNAL_LIST declares every parameter a
    golden answer might reference during scoring, not constraining what
    the model itself should use during generation). Using the param-
    inclusive list for allowed_signals_note instead produces a bloated,
    noisy instruction telling the model to use testbench-wide parameters
    that have nothing to do with this row's actual property -- confirmed
    real (FVEval-NL2SVA-Human-37 pulled in fsm_width/num_of_states/
    num_of_times_initial_state_repeats this way, unrelated to "should not
    remain in the same state")."""
    return _QUOTED_SIGNAL_NAME_RE.findall(user_prompt)


SIGNAL_DESCRIPTION_SYSTEM_PROMPT = (
    "You are a helpful bot that writes a brief, precise natural-language description of what each "
    "given RTL signal represents, based strictly on its declaration and usage in the testbench, "
    "following the requested format exactly."
)

PROMPT_TEMPLATE_SIGNAL_DESCRIPTIONS = """RTL testbench:
{testbench}

For each of the following signals, write ONE brief, precise sentence describing what it represents, \
based strictly on its declaration and how it's used in the testbench above -- do not guess or invent \
behavior the code doesn't show:
{signal_list}

Output exactly one line per signal, in this format, and nothing else:
<signal_name>: <description>
"""

_SIGNAL_DESCRIPTION_LINE_RE = re.compile(r'^(\w+)\s*:\s*(.+)$', re.MULTILINE)


def describe_signals(client, model_name, raw_testbench, signal_list):
    """Preprocessing step, run ONCE per row before Step 1: derives a brief
    NL description of each authoritative signal straight from the
    testbench, so allowed_signals_note can tell the model what each signal
    MEANS, not just its bare name. Confirmed real gap (FVEval-NL2SVA-
    Human-37): a bare name list alone didn't stop the model from inventing/
    using an unapproved shadow register (fsm_state_d1) instead of correctly
    using $past() on the approved fsm_state signal -- it had no per-signal
    guidance steering it toward the right approach, just a "don't invent
    names" instruction with no positive signal-meaning content.

    Returns {signal_name: description}. A signal the model's reply doesn't
    cover (or that isn't in signal_list) is simply left out -- callers fall
    back to the bare name for it, same as before this preprocessing step
    existed."""
    if not signal_list:
        return {}
    prompt = PROMPT_TEMPLATE_SIGNAL_DESCRIPTIONS.format(
        testbench=raw_testbench, signal_list="\n".join(signal_list)
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SIGNAL_DESCRIPTION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = completion.choices[0].message.content
    signal_set = set(signal_list)
    return {name: desc.strip() for name, desc in _SIGNAL_DESCRIPTION_LINE_RE.findall(raw) if name in signal_set}


_RECHECK_OL_NL_LINE_PATTERN = re.compile(r"OL NL:\s*(.*)")


def generate_sva_direct(client, model_name, user_prompt, rich_operator_context, extra_note=""):
    """Plain single-shot SVA generation -- no retrieval, no rechecking --
    but WITH the rich sva_temporal_operators.json operator context, unlike
    generate_baseline_sva (which has none at all). Used for both halves of
    Step 1's self-consistency check: sva_orig_ques (from the real question)
    and sva_ol_nl (from a candidate OL-NL restatement), built with an
    otherwise-identical prompt so the only thing that can make them diverge
    is the question text itself.

    extra_note, when non-empty (e.g. a row's authoritative allowed-signals
    note), is appended to the system message verbatim -- used for task-
    specific generation guidance that shouldn't be baked into the shared
    SYSTEM_PROMPT, which is also used unmodified by generate_baseline_sva
    for 0-shot-baseline fidelity.

    Returns a BARE property expression (EXPRESSION_ONLY_INSTRUCTION), not a
    complete `assert property (...)` statement -- the caller is responsible
    for wrap_property_expression-ing it when a complete SVA is needed.
    extract_property_body normalizes away any wrapper the model adds
    despite the instruction, so this is correct regardless of compliance."""
    system_msg = (
        SYSTEM_PROMPT + "\n\nSVA Operator Context:\n" + rich_operator_context
        + "\n\n" + EXPRESSION_ONLY_INSTRUCTION
    )
    if extra_note:
        system_msg += "\n\n" + extra_note
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
    )
    return extract_property_body(parse_code_response(completion.choices[0].message.content))


# generate_and_validate_ol_nl (REMOVED 2026-08-10): used to independently
# generate sva_orig_ques (direct from the question) and sva_ol_nl (via an
# OL-NL restatement), then formally check them against each other via
# JasperGold, retrying/reconciling/falling back based on whether they
# agreed. Removed after extensive live testing (FVEval-NL2SVA-Human-4,
# -11, -15, -20, and the full "type A"/"type D" buckets) established the
# whole self-consistency-checking premise was NET HARMFUL, not just
# unhelpful, relative to simply using the OL-NL-derived restatement
# directly with no check at all:
#   - The two paths oscillating never converging within the retry budget
#     (Human-4, -11), even though a correct answer was sitting in one of
#     the two drafts the whole time -- the "fix" for each round's flagged
#     side reliably reintroduced the mismatch the OTHER side had just been
#     fixed away from.
#   - A later "joint reconciliation" redesign fixed the oscillation
#     structurally, but introduced a WORSE failure mode: it sometimes
#     confidently converged BOTH sides onto the wrong candidate even when
#     the OL-NL path alone already had the right answer (Human-15: a
#     working `!(rd_pop && fifo_empty)` got reconciled into a broken
#     `$fell(rd_pop) |-> $past(!fifo_empty)`), with no remaining
#     disagreement signal left to catch it.
#   - A direct head-to-head test (same 6 "type D" rows, 3 strategies) found
#     "always just use sva_ol_nl, unverified" beat both the reconciliation
#     design (5/6 vs multiple false positives) and "fall back on any
#     mismatch" (5/6 vs 3 rows losing their grounding entirely) -- across
#     every trace examined, sva_orig_ques (the direct-from-NL path) was
#     the one making structural mistakes (backwards implications, spurious
#     |=>/$stable/$changed embellishments) far more often than the OL-NL
#     path, so "check them against each other" was mostly just risking a
#     good OL-NL answer on a worse independent draft.
# Step 1 is now just generate_ol_nl_grounding, called once, used directly
# -- no sva_orig_ques, no JasperGold check, no retries. See main() below.


def build_hybrid_retrieval_context(code_retriever, prompt_text):
    """Keyword/operator-guided retrieval path of HybridRetrieval: split the NL
    property description into operation-level phrases, map each to the most
    relevant SVA operator, and retrieve database chunks about that operator."""
    keywords = extract_keywords(prompt_text)
    op_explanations = extract_related_operators_of_keyword(keywords)
    checking_str = ""
    for op_explanation in op_explanations:
        checking_str += f"{op_explanation}\n\n"
        for doc in code_retriever.invoke(op_explanation):
            checking_str += doc.page_content + "\n\n"
        checking_str += "\n"
    return checking_str


def run_sor_recheck(client, model_name, sva_text, ol_nl_text, operator_context, max_retries, extra_note="", sor_template_timing=False, sor_conservative=False):
    """Runs ONE SOR (SVA operator-based rechecking) pass: builds the
    explanation-merge-tree context (or falls back to no tree-based context
    when sva_graph.py can't parse sva_text), and asks the model to either
    confirm sva_text is already correct or produce a revised version.

    sva_text is a BARE property expression in and out (our RAG+OL-NL flow
    never has the model handle the assert property/clock/disable iff
    wrapper -- see EXPRESSION_ONLY_INSTRUCTION).

    sor_template_timing: see build_rechecking_context.
    sor_conservative: appends SOR_CONSERVATIVE_NOTE to the system message,
    instructing a flagged revision to change only the minimal part
    responsible for the mismatch. Opt-in, not a default."""
    recheck_context = build_rechecking_context(
        client, model_name, sva_text, operator_context, max_retries, sor_template_timing
    )
    recheck_instruction = (
        "If it is already correct, repeat it verbatim -- do not introduce changes (e.g. adding a "
        "cycle delay like ##1 or |=>, or swapping to a superficially-similar operator) that aren't "
        "actually needed. "
    ) + (
        (
            "If there is a genuine mismatch, point to the specific Tn node responsible, list the "
            if recheck_context is not None else
            "If there is a genuine mismatch, please list the "
        ) + (
            "differences, and output a corrected property expression (just the boolean/temporal-"
            "logic condition -- no assert property/clock/disable iff wrapper) enclosed in "
            "```systemverilog and ```.\n"
        )
    )
    recheck_system_msg = (
        "You are a helpful bot to modify an SVA property expression based on the given description.\n\n"
        "SVA Operator Context:\n" + operator_context + "\n\n" + EXPRESSION_ONLY_INSTRUCTION
    )
    if extra_note:
        recheck_system_msg += "\n\n" + extra_note
    if sor_conservative:
        recheck_system_msg += "\n\n" + SOR_CONSERVATIVE_NOTE
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": recheck_system_msg},
            {"role": "user", "content": (
                f"Given the desired property description:\n{ol_nl_text}\n\n"
                f"please check whether the generated SVA property expression below operates "
                f"with the correct logic and timing (i.e., clock cycle):\n{sva_text}\n\n"
                f"{recheck_context or ''}\n{recheck_instruction}"
            )},
        ],
    )
    revised = extract_property_body(parse_code_response(completion.choices[0].message.content))
    # Same guard as generate_rag_sva's syntax-cleanup loop: if the model
    # answers with a conversational non-answer instead of code, keep the
    # input sva_text rather than overwrite it with garbage.
    return revised if looks_like_property_expression(revised) else sva_text


def build_rechecking_context(client, model_name, sva_text, operator_context, max_retries, sor_template_timing=False):
    """SVA operator-based rechecking context: the bottom-up explanation-merge
    tree of the generated SVA, or None if sva_graph.py can't parse it (~15%
    of the corpus -- same fallback the main pipeline script uses).

    sor_template_timing (--sor-template-timing): gives `|->`/`|=>`/`##N`
    nodes a fixed, deterministic natural-language template instead of an
    LLM-composed one -- see explanation_merge_tree.py's use_templates
    docstring for why (confirmed live: the LLM composition silently drops
    `|=>`'s own implicit +1-cycle advance when merged with a nested `##N`,
    and SOR's rechecking step can't then detect the resulting off-by-one).
    Opt-in, not a default -- an earlier attempt at fixing this same bug via
    a merge-node prompt tweak caused a replicated regression; this is
    untested at full scale yet."""
    try:
        merge_tree_str = build_and_render_explanation_merge_tree(
            client, model_name, sva_text, operator_context, max_retries=max_retries,
            use_templates=sor_template_timing,
        )
    except ValueError:
        return None
    return (
        "The following is a derived, node-by-node breakdown of what the generated "
        f"assertion actually means, built mechanically from its parsed syntax tree. "
        "Each `Tn` line shows one subexpression, the SVA operator that merges its "
        f"operand(s) into it, and the resulting natural-language meaning:\n\n{merge_tree_str}"
    )


def build_official_nl2sva_human_user_prompt(raw_testbench, prompt_text):
    """Reproduces NL2SVAHumanLauncher.generate_user_prompt_prefix +
    generate_question_prompt's exact concatenation for num_icl_examples=0
    (FVEval-0-shot) -- see benchmark_launcher.py:829-851. Notably: the RAW
    testbench (no injection marker), and SVAGEN_QUESTION_POSTAMBLE's "Do not
    add code to output an error message string." instruction, which
    run_codev_sva_ol_dfs_eval.py's QUESTION_TEMPLATE (used for the other two
    tasks below) doesn't have."""
    user_prompt_prefix = "\n\n" + prompts_nl2sva_human.SVAGEN_TB_PREAMBLE + "\n" + raw_testbench
    question_prompt = (
        prompts_nl2sva_human.SVAGEN_QUESTION_PREAMBLE + prompt_text + "\n"
        + prompts_nl2sva_human.SVAGEN_QUESTION_POSTAMBLE
    )
    return user_prompt_prefix + "\n" + question_prompt


def build_verified_machine_user_prompt(raw_testbench, prompt_text):
    """nl2sva_machine_verified's human-message shape: unlike
    build_official_nl2sva_human_user_prompt, there's no "official FVEval-
    0-shot" prompt to reproduce for this task (its own harness launcher
    uses the SAME borrowed, disable-iff-priming QUESTION_TEMPLATE this repo
    already flags as wrong for nl2sva_machine -- see the module docstring's
    TODO), so this is a minimal, dataset-appropriate prompt instead: just
    the testbench and the (already operator-level) question. No
    "disable iff (tb_reset)" worked example, no injection marker (this
    dataset's testbenches are bare port-list modules with no internal logic
    to mark) -- output-format instructions (EXPRESSION_ONLY_INSTRUCTION)
    live in the system message, not duplicated here."""
    return (
        "Here is the testbench to perform your translation:\n"
        f"{raw_testbench}\n"
        f"Question: Create a SVA assertion that checks: {prompt_text}\n"
    )


_SIGNAL_WIDTH_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")


def iter_verified_nl2sva_machine_rows(jsonl_path):
    """Same 5-tuple shape as iter_verified_nl2sva_human_rows. signal_list
    here is always populated (unlike signals_for_validity, which is null
    for 15/73 human rows) and comma-joined with occasional bit-width
    prefixes (e.g. "sig_C,[3:0] sig_A") -- stripped down to bare
    identifiers the same way."""
    with open(jsonl_path) as file:
        for line in file:
            row = json.loads(line)
            signals = [_SIGNAL_WIDTH_PREFIX_RE.sub("", s).strip() for s in row["signal_list"].split(",")]
            yield row["name"], row["testbench"], row["problem"], row["ground_truth"], signals


def iter_verified_nl2sva_human_rows(jsonl_path):
    """Same 4-tuple shape as run_codev_sva_ol_dfs_eval.iter_rows, plus a 5th
    field: the authoritative signals_for_validity list (with any bit-width
    prefix like '[5:0] ' stripped down to the bare identifier -- the JG
    SIGNAL_LIST macro just wants comma-joined names, same convention as
    build_signal_list's existing quoted-name/parameter extraction). Confirmed
    signals_for_validity does NOT include testbench parameters (e.g. `max`)
    even when the golden references them, so callers still need to union
    this with a parameter/localparam scan of the testbench, same as before."""
    with open(jsonl_path) as file:
        for line in file:
            row = json.loads(line)
            # 15/73 rows have signals_for_validity: null in the source JSONL
            # (confirmed, e.g. FVEval-NL2SVA-Human-20) -- treat as "none
            # provided" rather than crash; the parameter/localparam scan
            # callers union this with still applies.
            signals = [_SIGNAL_WIDTH_PREFIX_RE.sub("", s).strip() for s in (row["signals_for_validity"] or [])]
            yield row["name"], row["testbench"], row["problem"], row["ground_truth"], signals


def iter_task_rows(task, csv_path):
    """Normalizes both row shapes to one 5-tuple: (task_id, raw_testbench,
    prompt_text, ref_solution, signals_for_validity), with
    signals_for_validity left as None for the two tasks that don't have an
    authoritative list (build_signal_list's regex heuristic is used for
    those instead, unchanged)."""
    if task == "nl2sva_human_verified":
        yield from iter_verified_nl2sva_human_rows(csv_path)
    elif task == "nl2sva_machine_verified":
        yield from iter_verified_nl2sva_machine_rows(csv_path)
    else:
        for task_id, raw_testbench, prompt_text, ref_solution in iter_rows(csv_path, task):
            yield task_id, raw_testbench, prompt_text, ref_solution, None


_SVA_TOKEN_RE = re.compile(r'\|->|\|=>|##|\$\w+|[=!<>]=|&&|\|\||[()]')
_BARE_IDENTIFIER_RE = re.compile(r'^\w+$')
# A period followed by whitespace and a capital letter is a sentence
# boundary -- something no real SVA property expression ever contains
# (periods in this dataset only appear, without following whitespace, in
# hierarchical references like `module.signal`; there are no decimal
# literals). Confirmed live (2026-08-11): a SOR recheck completion that
# QUOTES the correct expression inline while explaining it ("The generated
# SVA property expression `!(a == 0 && b)` accurately captures the
# condition...This means...") contains real SVA tokens as substrings, so
# the token-presence check alone accepted the whole paragraph as if it
# were the answer -- got mechanically wrapped into a nonsense `assert
# property`. Checked FIRST, ahead of the token-presence check, since a
# prose verdict should override an incidental token match.
_PROSE_SENTENCE_RE = re.compile(r'\.\s+[A-Z]')


def looks_like_property_expression(text):
    """Guard against the syntax-cleanup checker LLM returning a conversational
    non-answer (e.g. "Please provide the SVA property expression that needs
    to be checked and corrected." or "The generated SVA property expression
    is as follows:") instead of actual code. Confirmed live (2026-08-10,
    nl2sva_human_verified counterexample-validation Step-1-only runs, TWO
    distinct phrasings) that with no guard, such a response gets silently
    accepted as sva_text, mechanically wrapped via wrap_property_expression,
    and counted as a real (malformed) generated SVA -- inflating
    syntax-failure counts with a pipeline artifact rather than a genuine
    generation error.

    A phrase blocklist proved too fragile (a first fix catching "please
    provide" missed this second, differently-worded non-answer) -- checks
    STRUCTURE instead: any real property expression contains at least one
    SVA/boolean operator, a `$system` call, or parens, since even the
    simplest realistic properties compare/combine signals. A bare single
    identifier (e.g. just `count`) is also accepted as a rare-but-legitimate
    edge case. Prose sentences (English words, punctuation, no such tokens)
    are rejected regardless of exact wording -- and prose that happens to
    QUOTE real SVA tokens inline (confirmed live -- see _PROSE_SENTENCE_RE)
    is rejected too, via a sentence-boundary check that fires ahead of the
    token-presence check."""
    text = text.strip()
    if not text:
        return False
    if _PROSE_SENTENCE_RE.search(text):
        return False
    if _BARE_IDENTIFIER_RE.match(text):
        return True
    return bool(_SVA_TOKEN_RE.search(text))


def jg_driven_syntax_cleanup(rag_chain_checker, raw_testbench, sva_text, sv_dir, experiment_id, task_id, label, allowed_signals_note="", disable_signal="tb_reset"):
    """Up to 3 rounds of real-JasperGold-elaboration-driven syntax cleanup
    on sva_text (see check_sva_elaboration). Each round FIRST checks with a
    real `jg` elaboration whether sva_text actually has a problem at all --
    if it elaborates cleanly, returns immediately with no LLM call, no risk
    of "fixing" something that was never broken. Only on a genuine
    elaboration failure does the model get asked to fix anything, given the
    REAL JasperGold error text and an explicit "fix ONLY this" instruction.

    allowed_signals_note: the row's authoritative signal list, passed as the
    checker chain's {allowed_signals} template variable (a per-row dynamic
    slot -- unlike the operator table, which is static/global and baked
    directly into system_prompt_checker at chain-construction time in
    main()). Without this, a fix for one real problem (e.g. an undeclared
    identifier) could easily introduce or reintroduce a DIFFERENT one: a
    real-but-unauthorized testbench signal, which check_sva_elaboration
    itself can't catch either (see its docstring / score_nl2sva_human.py's
    module docstring for the FVEval-NL2SVA-Human-0 case this describes).

    Factored out so generate_rag_sva can run this TWICE -- once on Stage 2's
    initial generation (before SOR, since SOR is a functional recheck that
    only makes sense against something that already elaborates), and once
    again on SOR's output (after SOR, since SOR can itself introduce a new
    elaboration error while "fixing" something functional -- confirmed live,
    2026-08-11, FVEval-NL2SVA-Human-69: SOR combined two implications with
    an invalid `else` inside a property; nothing re-checked its output
    afterward in the single-pass version, so it went straight through to
    the final answer broken).

    label: a short string distinguishing which pass this is (e.g.
    "presor"/"postsor") so the two passes' JG scratch files/task_ids don't
    collide when both run for the same row."""
    for attempt in range(3):
        ok, jg_output = check_sva_elaboration(
            raw_testbench, sva_text, sv_dir or "/tmp/syntax_cleanup_jgtmp",
            experiment_id=experiment_id or "syntax_cleanup", task_id=f"{task_id}_{label}{attempt}",
            disable_signal=disable_signal,
        )
        if ok:
            break
        error_summary = summarize_elaboration_errors(jg_output)
        checker_prompt = (
            "JasperGold reported a real elaboration error for the following SVA property "
            f"expression:\n{sva_text}\n\n"
            f"JasperGold error output:\n{error_summary}\n\n"
            "Fix ONLY the specific error(s) reported above -- do not otherwise change the "
            "property's meaning or introduce unrelated changes. Do NOT split the property "
            "into multiple separate statements or multiple code blocks, even if the fix "
            "involves an operator like `or`/`and` that combines two conditions -- the "
            "corrected property must still be exactly ONE single expression. Output ONLY "
            "that one corrected property expression (no assert property/clock/disable iff "
            "wrapper -- just the boolean/temporal-logic condition), enclosed in EXACTLY ONE "
            "```systemverilog ... ``` block -- nothing before or after it, and no second "
            "code block.\n"
        )
        checker_result = rag_chain_checker.invoke({
            "input": checker_prompt,
            "allowed_signals": (allowed_signals_note + "\n\n") if allowed_signals_note else "",
        })["answer"]
        candidate = extract_property_body(parse_code_response(checker_result))
        if not looks_like_property_expression(candidate):
            # Checker returned a conversational non-answer instead of code --
            # keep the prior sva_text rather than overwrite it with garbage.
            continue
        sva_text = candidate
    return sva_text


def generate_rag_sva(
    client, model_name, rag_chain, rag_chain_checker, code_retriever,
    operator_context, user_prompt, prompt_text, ol_nl_text, max_retries,
    question_replaced=False, extra_note="", allowed_signals_note="",
    sv_dir=None, experiment_id=None, task_id=None, raw_testbench=None,
    disable_signal="tb_reset", sor_template_timing=False, sor_conservative=False,
    only_overlap_implication=False,
):
    """Runs one FVEval row through the full pipeline: HybridRetrieval-augmented
    generation, THEN a JasperGold-elaboration-driven syntax cleanup pass,
    THEN SOR (SVA operator-based rechecking), THEN a SECOND syntax cleanup
    pass. Cleanup runs before SOR deliberately -- SOR is a functional/
    semantic recheck, which only makes sense against an expression that
    already elaborates -- and again after SOR, since SOR can itself
    introduce a new elaboration error while fixing something functional
    (confirmed live, FVEval-NL2SVA-Human-69: SOR combined two implications
    with an invalid `else` inside a property). Returns (final_sva_text,
    initial_response_text).

    raw_testbench: the row's real RTL testbench, used ONLY by the syntax-
    cleanup loop's real JasperGold elaboration checks (check_sva_elaboration)
    -- required whenever this function is actually called (not --no-rag).

    ol_nl_text is the description used for HybridRetrieval's query and the
    rechecking step -- either prompt_text unchanged, or a best-effort OL-NL
    grounded restatement of it (generate_ol_nl_grounding), depending on
    whether --ol-nl-grounding is set. user_prompt is the official FVEval
    prompt shape used for the actual generation call's human message; by
    default it still carries the original, unmodified prompt_text as its
    question (grounding only augments the system message). If
    question_replaced is True, the caller has already substituted ol_nl_text
    as user_prompt's question itself (--ol-nl-replace-question), so the
    grounding is skipped here to avoid stating it twice.

    only_overlap_implication (--only-overlap-implication): appends
    ONLY_OVERLAP_IMPLICATION_NOTE to both Stage 2's generation note and
    SOR's extra_note. Opt-in, not a default."""
    checking_str = build_hybrid_retrieval_context(code_retriever, ol_nl_text)
    ol_nl_context = (
        f"Grounded, signal-level restatement of the property description: {ol_nl_text}\n\n"
        if ol_nl_text != prompt_text and not question_replaced else ""
    )
    step2_extra_note = allowed_signals_note
    if only_overlap_implication:
        step2_extra_note = (
            (step2_extra_note + "\n\n") if step2_extra_note else ""
        ) + ONLY_OVERLAP_IMPLICATION_NOTE
    llm_result = rag_chain.invoke({
        "keywords_explaination": checking_str,
        "ol_nl_grounding": ol_nl_context,
        "allowed_signals": (step2_extra_note + "\n\n") if step2_extra_note else "",
        "input": user_prompt,
    })
    initial_response = llm_result["answer"]
    # extract_property_body normalizes away any assert property/clock/
    # disable iff wrapper the model adds despite EXPRESSION_ONLY_INSTRUCTION
    # (soft instruction, not guaranteed) -- sva_text is a bare expression
    # from here on, throughout SOR and syntax cleanup; wrapped back into a
    # complete SVA only once, right before this function returns.
    sva_text = extract_property_body(parse_code_response(initial_response))

    # JasperGold-elaboration-driven syntax cleanup runs FIRST, before SOR --
    # SOR is a functional/semantic recheck (does this match the intended
    # meaning?), which only makes sense to run against an expression that
    # actually elaborates in the first place; fixing elaboration errors
    # after SOR would mean SOR spent its one pass reasoning about something
    # that might not even be valid SystemVerilog yet. See
    # jg_driven_syntax_cleanup's docstring for why it also runs a SECOND
    # time after SOR, below.
    sva_text = jg_driven_syntax_cleanup(
        rag_chain_checker, raw_testbench, sva_text, sv_dir, experiment_id, task_id, "presor",
        allowed_signals_note=allowed_signals_note, disable_signal=disable_signal,
    )

    sor_extra_note = extra_note
    if only_overlap_implication:
        sor_extra_note = (
            (sor_extra_note + "\n\n") if sor_extra_note else ""
        ) + ONLY_OVERLAP_IMPLICATION_NOTE
    sva_text = run_sor_recheck(
        client, model_name, sva_text, ol_nl_text, operator_context, max_retries, extra_note=sor_extra_note,
        sor_template_timing=sor_template_timing, sor_conservative=sor_conservative,
    )

    # Second pass: SOR can itself introduce a new elaboration error while
    # "fixing" something functional -- confirmed live (2026-08-11,
    # FVEval-NL2SVA-Human-69) that SOR combined two implications with an
    # invalid `else` inside a property, and with only the pre-SOR pass above,
    # nothing re-checked SOR's own output before it became the final answer.
    sva_text = jg_driven_syntax_cleanup(
        rag_chain_checker, raw_testbench, sva_text, sv_dir, experiment_id, task_id, "postsor",
        allowed_signals_note=allowed_signals_note, disable_signal=disable_signal,
    )

    # The only place a complete SVA gets assembled in this whole function --
    # mechanically, from a fixed template, never trusting the model to get
    # the clock/disable-iff/label right itself.
    return wrap_property_expression(sva_text, disable_signal=disable_signal), initial_response


def generate_baseline_sva(client, model_name, user_prompt):
    """Plain single-shot generation: no retrieval, no rechecking, no syntax-
    cleanup passes -- just SYSTEM_PROMPT + the official user prompt, one
    completion call. Reproduces the FVEval-0-shot baseline
    (NL2SVAHumanLauncher with num_icl_examples=0) as closely as possible
    while still going through this script's own CSV/scoring pipeline, so the
    comparison against the RAG-augmented run is apples-to-apples: identical
    scoring code, identical body-extraction, only generation itself differs."""
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
    )
    response_text = completion.choices[0].message.content
    return parse_code_response(response_text), response_text


def process_row(
    row_index, task_id, raw_testbench, prompt_text, ref_solution, signals_for_validity,
    args, client, model_name, rich_operator_context, code_retriever, rag_chain,
    rag_chain_checker, step1_jg_sv_dir, experiment_id,
):
    """One row's worth of main()'s per-row body, factored out unchanged so it
    can run inside a thread pool (see main()'s ThreadPoolExecutor loop) --
    the whole pipeline is I/O-bound (OpenAI API calls, Chroma vector
    lookups, `jg` subprocesses), not CPU-bound, so threads (not processes)
    parallelize it fine despite the GIL: each blocking call releases the
    GIL while waiting on network/
    subprocess I/O. All shared objects touched here (client, code_retriever,
    rag_chain, rag_chain_checker) are called read-only/statelessly per-
    invocation -- OpenAI's client and LangChain's LCEL Runnables don't hold
    mutable per-call state, and Chroma similarity search is a read-only
    query -- so no locking is needed around them specifically (only around
    the shared CSV writer in main(), since concurrent writes to one file
    handle need serializing).

    Returns the LMRESULT_FIELDNAMES-shaped row dict on success, or None on
    failure (after printing the same "failed (...), skipping" message
    main()'s try/except used to print inline)."""
    print(f"[{row_index + 1}] task_id={task_id} ...")
    try:
        # Built unconditionally, using the REAL original question --
        # needed both as Step 1's self-consistency anchor
        # (sva_orig_ques) and, whenever the question isn't being
        # replaced, as the prompt actually used for generation.
        if args.task in ("nl2sva_human", "nl2sva_human_verified"):
            # Official FVEval-0-shot shape: raw testbench, no marker.
            design_rtl = raw_testbench
            user_prompt_orig = build_official_nl2sva_human_user_prompt(raw_testbench, prompt_text)
        elif args.task == "nl2sva_machine_verified":
            # Bare dummy testbenches (port declarations only, no internal
            # logic) -- no injection marker needed, no reason to borrow the
            # disable-iff-priming QUESTION_TEMPLATE either (see
            # build_verified_machine_user_prompt's docstring).
            design_rtl = raw_testbench
            user_prompt_orig = build_verified_machine_user_prompt(raw_testbench, prompt_text)
        else:
            # TODO: give nl2sva_machine / module_sva_nl_manual_editing the
            # same official-prompt treatment -- still borrowing
            # run_codev_sva_ol_dfs_eval.py's QUESTION_TEMPLATE here.
            design_rtl = build_testbench_with_marker(raw_testbench)
            user_prompt_orig = build_user_prompt(design_rtl, prompt_text)

        # nl2sva_human_verified supplies an authoritative signal list
        # (signals_for_validity, unioned with a parameter/localparam
        # scan) instead of build_signal_list's regex-over-prompt-text
        # heuristic -- None for the other two tasks, which fall back
        # to that heuristic alone below (row_signal_list's own
        # ternary).
        #
        # nl2sva_machine_verified is deliberately NOT unioned with
        # build_signal_list: confirmed 0/283 of its bare dummy testbenches
        # declare any parameter/localparam (the only thing that scan adds
        # beyond quoted names), while its problem text routinely quotes
        # BIT VALUES rather than signal names (e.g. "an odd number of '1'
        # bits") -- build_signal_list's quoted-name regex can't tell the
        # difference, so the union injected a bogus "1" entry into
        # SIGNAL_LIST for ~27/283 rows, which broke JasperGold's
        # prop_eq_checker wrapper outright (syntax error) and surfaced as
        # a false-negative "functional mismatch" -- confirmed live
        # (FVEval-NL2SVA-Machine-48, 2026-08-13).
        if args.task == "nl2sva_machine_verified":
            row_signal_list = signals_for_validity
        else:
            row_signal_list = (
                list(dict.fromkeys(signals_for_validity + build_signal_list(user_prompt_orig, raw_testbench)))
                if signals_for_validity is not None
                else build_signal_list(user_prompt_orig, raw_testbench)
            )

        # --skip-signal-list-note: nl2sva_machine_verified's "problem" text
        # already names every real signal directly (e.g. "sig_F and sig_H
        # are high"), so this note -- and the describe_signals call that
        # feeds it, an extra LLM call per row -- is redundant there. Kept
        # as a togglable flag rather than auto-detected by task, so a
        # human.jsonl rerun stays byte-identical to before this option
        # existed.
        allowed_signals_note = ""
        if not args.skip_signal_list_note:
            question_signal_list = signals_for_validity if signals_for_validity else extract_named_signals(user_prompt_orig)
            signal_descriptions = describe_signals(client, model_name, raw_testbench, question_signal_list)
            signal_list_str = "; ".join(
                f"'{s}' ({signal_descriptions[s]})" if s in signal_descriptions else f"'{s}'"
                for s in question_signal_list
            )
            allowed_signals_note = (
                "This row's authoritative signal list -- you must use ONLY signals from this "
                "list; do not invent or substitute a different, merely-real testbench signal "
                "in their place, even if it looks related. Each signal, with its meaning "
                "grounded in the testbench: " + signal_list_str + ". "
                "If the testbench declares a parameter/localparam for a bound or constant you "
                "need (e.g. a max/min/width value), you MUST reference that parameter's name "
                "directly -- do NOT hardcode a literal constant instead, even if its numeric "
                "value would be the same."
                if question_signal_list else ""
            )

        ol_nl_text = prompt_text
        if args.ol_nl_grounding and not args.no_rag:
            # No formal check -- see the removed-code note above for
            # why generating+verifying an independent second candidate
            # (the old generate_and_validate_ol_nl) was dropped in
            # favor of just using this directly.
            step1_extra_note = allowed_signals_note
            if args.ol_nl_conservative:
                step1_extra_note = (
                    (step1_extra_note + "\n\n") if step1_extra_note else ""
                ) + OL_NL_CONSERVATIVE_NOTE
            ol_nl_text = generate_ol_nl_grounding(
                client, model_name, prompt_text, raw_testbench,
                rich_operator_context, extra_note=step1_extra_note,
            )

        if args.ol_nl_replace_question:
            if args.task in ("nl2sva_human", "nl2sva_human_verified"):
                user_prompt = build_official_nl2sva_human_user_prompt(raw_testbench, ol_nl_text)
            elif args.task == "nl2sva_machine_verified":
                user_prompt = build_verified_machine_user_prompt(raw_testbench, ol_nl_text)
            else:
                user_prompt = build_user_prompt(design_rtl, ol_nl_text)
        else:
            user_prompt = user_prompt_orig

        # nl2sva_machine_verified's bare dummy testbenches declare no reset
        # signal at all (confirmed 0/283 rows) and golden answers have no
        # disable iff clause -- see DEFAULT_CSV_PATHS's comment.
        disable_signal = None if args.task == "nl2sva_machine_verified" else "tb_reset"

        if args.no_rag:
            sva_text, initial_response = generate_baseline_sva(client, model_name, user_prompt)
        else:
            sva_text, initial_response = generate_rag_sva(
                client, model_name, rag_chain, rag_chain_checker, code_retriever,
                rich_operator_context, user_prompt, prompt_text, ol_nl_text, args.max_retries,
                question_replaced=args.ol_nl_replace_question, extra_note=allowed_signals_note,
                allowed_signals_note=allowed_signals_note,
                sv_dir=step1_jg_sv_dir, experiment_id=experiment_id, task_id=task_id,
                raw_testbench=raw_testbench, disable_signal=disable_signal,
                sor_template_timing=args.sor_template_timing, sor_conservative=args.sor_conservative,
                only_overlap_implication=args.only_overlap_implication,
            )
    except Exception as error:
        print(f"    failed ({error}), skipping")
        return None

    return {
        "experiment_id": experiment_id,
        "task_id": task_id,
        "model_name": model_name,
        "response": f"```systemverilog\n{sva_text}\n```",
        "ref_solution": ref_solution,
        "design_rtl": design_rtl,
        "output_tb": design_rtl,
        "user_prompt": user_prompt,
        "cot_response": initial_response,
        # row_signal_list (signals_for_validity unioned with the
        # testbench's parameter/localparam scan), not the bare
        # signals_for_validity -- goldens routinely reference
        # parameters (e.g. `max`) that signals_for_validity alone
        # doesn't list (confirmed for FVEval-NL2SVA-Human-0), and
        # scoring reads this column back verbatim as the JG
        # SIGNAL_LIST, so leaving params out surfaces as a JG
        # elaboration error, not a genuine functional mismatch.
        "signals_for_validity": ",".join(row_signal_list) if row_signal_list is not None else "",
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=list(DEFAULT_CSV_PATHS))
    parser.add_argument("--csv", default=None, help="Defaults to the standard FVEval path for --task")
    parser.add_argument("--output", default=None,
                         help="Defaults to Results/fveval_rag_outputs/{task}_{model_name}_{dynamicrag|baseline0shot}.csv")
    parser.add_argument("--config", default="Src/Config.yml")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--workers", type=int, default=6,
                         help="Number of rows processed concurrently via a thread pool (the pipeline "
                              "is I/O-bound -- OpenAI API calls, Chroma lookups, `jg` subprocesses -- "
                              "so threads parallelize it despite the GIL). Set to 1 for the old "
                              "fully-sequential behavior. Each row's syntax-cleanup passes spawn real "
                              "JasperGold subprocesses -- keep this low enough to stay under your JG "
                              "license's concurrent-session limit.")
    parser.add_argument("--max-retries", type=int, default=5)
    parser.add_argument("--no-rag", action="store_true",
                         help="Skip HybridRetrieval/rechecking/syntax-cleanup entirely -- plain "
                              "single-shot 0-shot generation, reproducing the FVEval-0-shot baseline "
                              "for a controlled, same-scorer comparison against the RAG-augmented run.")
    parser.add_argument("--ol-nl-grounding", action="store_true",
                         help="Before HybridRetrieval, rewrite the row's (often abstract, e.g. "
                              "nl2sva_human-style) description into a signal-grounded, operator-level "
                              "restatement (no golden SVA used -- best-effort, single-pass, no "
                              "validation loop), and use that for retrieval + rechecking. By default "
                              "the official generation prompt's question is untouched (grounding only "
                              "augments the system message); see --ol-nl-replace-question. Ignored "
                              "when --no-rag is set.")
    parser.add_argument("--ol-nl-replace-question", action="store_true",
                         help="Requires --ol-nl-grounding. Instead of appending the grounded "
                              "restatement to the system message alongside the original question, "
                              "substitute it AS the question in the generation call's human message. "
                              "Motivation: chat models tend to weight the human message as the primary "
                              "instruction, so an ambiguous original phrasing sitting there may win out "
                              "over a clarification relegated to the system message.")
    parser.add_argument("--ol-nl-conservative", action="store_true",
                         help="Requires --ol-nl-grounding. Appends OL_NL_CONSERVATIVE_NOTE to Step 1's "
                              "extra_note, instructing it to stay literal and not invent/infer additional "
                              "structure, conditions, or behavior beyond what the description states. "
                              "Opt-in, not a default -- tested on a small sample with inconclusive results; "
                              "unlike EVENTUALLY_TIMING_NOTE (rejected), this note is NOT reverse-engineered "
                              "from golden phrasing, just a general faithfulness instruction.")
    parser.add_argument("--skip-signal-list-note", action="store_true",
                         help="Don't build/inject allowed_signals_note (the 'use only these signals' "
                              "instruction + per-signal descriptions) at all, and skip the describe_signals "
                              "call that feeds it. Off by default -- nl2sva_human(_verified) rows benefit "
                              "from it since their prompt text often doesn't name every signal explicitly. "
                              "Intended for nl2sva_machine_verified, whose problem text already names every "
                              "real signal directly, making the note (and its extra LLM call) redundant.")
    parser.add_argument("--sor-template-timing", action="store_true",
                         help="Gives SOR's explanation-merge-tree a fixed, deterministic (LLM-free) "
                              "natural-language template (from sva_temporal_operators.json's own "
                              "template_unary/template_binary fields, covering all 46 documented "
                              "operators) instead of an LLM-composed nl_piece, wherever the node's "
                              "operator has one. Confirmed live: LLM composition silently drops `|=>`'s "
                              "own implicit +1-cycle advance when merged with a nested `##N` consequent, "
                              "and SOR can't then detect the resulting off-by-one cycle error. Any "
                              "operator NOT in the table (booleans, reductions, comparisons, arbitrary "
                              "system functions) is unaffected. Opt-in, not a default -- untested at "
                              "full benchmark scale yet; a related earlier fix attempt (a merge-node "
                              "prompt tweak, not this) caused a replicated regression and was reverted.")
    parser.add_argument("--sor-conservative", action="store_true",
                         help="Appends SOR_CONSERVATIVE_NOTE to SOR's recheck system message: when SOR "
                              "does flag a genuine mismatch, change ONLY the minimal part responsible -- "
                              "do not restructure, drop, or add operators/clauses beyond what that "
                              "specific mismatch requires. Confirmed live (surfaced most clearly under "
                              "--sor-template-timing, which gives SOR more genuine mismatches to act on "
                              "in the first place) that SOR's own revision isn't reliably minimal on its "
                              "own -- e.g. dropping the `|->`/`|=>` operator entirely, or inserting a "
                              "spurious clause nobody asked for. Independent of --sor-template-timing at "
                              "the code level; not required, just where the problem was found. Opt-in, "
                              "not a default -- untested at scale yet.")
    parser.add_argument("--only-overlap-implication", action="store_true",
                         help="Appends ONLY_OVERLAP_IMPLICATION_NOTE to both Stage 2's generation note "
                              "and SOR's extra_note: always use `|->`, never `|=>`; express any needed "
                              "delay explicitly as `##N`/`##[M:N]` using the TOTAL cycle count. Sidesteps "
                              "the repeatedly-confirmed |=>+##N double-counting bug (model writes `A |=> "
                              "##N B`, silently landing on N+1 total cycles instead of golden's N) at its "
                              "root, by eliminating one of the two delay-encoding conventions the model "
                              "has to correctly combine, rather than trying to teach it to combine them "
                              "correctly. Opt-in, not a default -- untested at scale yet.")
    args = parser.parse_args()

    if args.ol_nl_replace_question and not args.ol_nl_grounding:
        parser.error("--ol-nl-replace-question requires --ol-nl-grounding")

    if args.ol_nl_conservative and not args.ol_nl_grounding:
        parser.error("--ol-nl-conservative requires --ol-nl-grounding")

    if not args.no_rag:
        import shutil
        if shutil.which("jg") is None:
            parser.error(
                "generate_rag_sva's syntax-cleanup loop now checks each candidate with a real "
                "JasperGold elaboration (analyze+elaborate, see jasper_direct_equiv_check."
                "check_sva_elaboration) before ever asking the model to 'fix' anything -- `jg` "
                "was not found on PATH. Add JasperGold's bin/ to PATH (and set CDS_LIC_FILE) "
                "before running. (Only --no-rag, the plain 0-shot baseline with no retrieval/"
                "rechecking/cleanup at all, doesn't need `jg`.)"
            )

    csv_path = args.csv or DEFAULT_CSV_PATHS[args.task]

    with open(args.config) as file:
        config = yaml.safe_load(file)
    openai_api_key = config["Openai_API_Key"]
    model_name = config["Model_Name"]

    if args.no_rag:
        default_suffix = "baseline0shot"
    elif args.ol_nl_grounding and args.ol_nl_replace_question:
        default_suffix = "dynamicrag_olnl_replaceq"
    elif args.ol_nl_grounding:
        default_suffix = "dynamicrag_olnl"
    else:
        default_suffix = "dynamicrag"
    output_path = args.output or f"Results/fveval_rag_outputs/{args.task}_{model_name}_{default_suffix}.csv"
    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)

    client = OpenAI(api_key=openai_api_key)

    rich_operator_context = None
    step1_jg_sv_dir = None
    code_retriever = None
    rag_chain = None
    rag_chain_checker = None
    if not args.no_rag:
        # Sole operator table for the whole pipeline now (sva_temporal_operators.json,
        # 38 entries) -- used for HybridRetrieval's keyword-guided path, SOR
        # rechecking (merge-tree + final recheck completion), and, when
        # --ol-nl-grounding is set, Step 1's SVA-generation/grounding calls too.
        rich_operator_context = load_rich_operator_context()
        # Shared JG scratch dir -- used by generate_rag_sva's syntax-cleanup
        # loop's real elaboration checks (any RAG path).
        step1_jg_sv_dir = f"{output_path}.step1_jgtmp"

        code_store = build_rag_system(config["PDF_Txt"], openai_api_key)
        code_retriever = code_store.as_retriever()

        llm = ChatOpenAI(model=model_name, api_key=openai_api_key)

        system_prompt = (
            SYSTEM_PROMPT
            + "\n\n" + EXPRESSION_ONLY_INSTRUCTION
            + "\n{allowed_signals}"
            + "Use the following pieces of retrieved context to help answer the question.\n\n"
            + "{ol_nl_grounding}"
            + "{keywords_explaination}"
            + "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        rag_chain = create_retrieval_chain(code_retriever, create_stuff_documents_chain(llm, prompt))

        # ChatPromptTemplate parses `{...}` in ANY literal template text as a
        # variable placeholder, not just in the explicit {input}/{context}
        # slots -- confirmed live (2026-08-11, FVEval-NL2SVA-Human-69) that
        # embedding rich_operator_context here unescaped crashes with
        # KeyError "missing variables {'hold, busy, cont_gnt'}" the moment
        # the checker is actually invoked, because one of the operator
        # table's own worked examples is `$onehot0({hold, busy, cont_gnt})`.
        # Doubling every brace is the standard str.format() escape (`{{`/
        # `}}` render as literal `{`/`}`), applied only to this copy -- the
        # unescaped rich_operator_context is still used as-is everywhere
        # else (Step 1, SOR, HybridRetrieval), none of which go through a
        # ChatPromptTemplate.
        escaped_operator_context = rich_operator_context.replace("{", "{{").replace("}", "}}")
        system_prompt_checker = (
            "You are a helpful bot that fixes a real JasperGold elaboration error reported for the "
            "given SVA property expression. "
            + "\n\nSVA Operator Context:\n" + escaped_operator_context
            + "\n\n" + EXPRESSION_ONLY_INSTRUCTION
            + "\n{allowed_signals}"
            + "Use the following pieces of retrieved context to help answer the question.\n\n"
            "{context}"
        )
        prompt_checker = ChatPromptTemplate.from_messages([("system", system_prompt_checker), ("human", "{input}")])
        rag_chain_checker = create_retrieval_chain(code_retriever, create_stuff_documents_chain(llm, prompt_checker))

    experiment_id = os.path.basename(csv_path).rsplit(".", 1)[0]

    rows = list(iter_task_rows(args.task, csv_path))
    if args.limit is not None:
        rows = rows[: args.limit]

    write_lock = threading.Lock()
    completed_count = 0

    with open(output_path, "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=LMRESULT_FIELDNAMES)
        writer.writeheader()

        # Threads, not processes: this whole pipeline is I/O-bound (OpenAI
        # API calls, Chroma vector lookups, optionally `jg` subprocesses),
        # so each blocking call releases the GIL while waiting -- no need
        # for the pickling/reinit overhead multiprocessing would add just
        # to share client/code_retriever/rag_chain across workers. Results
        # complete out of row order under concurrency; only the CSV row
        # order changes (harmless -- scoring reads task_id per row, not
        # position), not correctness.
        with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = [
                executor.submit(
                    process_row, row_index, task_id, raw_testbench, prompt_text, ref_solution,
                    signals_for_validity, args, client, model_name, rich_operator_context,
                    code_retriever, rag_chain, rag_chain_checker, step1_jg_sv_dir, experiment_id,
                )
                for row_index, (task_id, raw_testbench, prompt_text, ref_solution, signals_for_validity) in enumerate(rows)
            ]
            for future in concurrent.futures.as_completed(futures):
                row = future.result()
                completed_count += 1
                if row is None:
                    continue
                with write_lock:
                    writer.writerow(row)
                    out_file.flush()
                print(f"    [{completed_count}/{len(rows)} done] task_id={row['task_id']}")

    print(f"Wrote responses to {output_path}")


if __name__ == "__main__":
    main()
