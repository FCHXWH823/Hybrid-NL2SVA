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
import csv
import json
import os
import re
import sys

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

# jasper_equiv_check.run_equivalence_check: the real JasperGold formal
# equivalence check used to validate OL-NL against a plain-generation anchor
# (see generate_and_validate_ol_nl below) -- requires `jg` on PATH and
# CDS_LIC_FILE set, but only when --ol-nl-grounding is actually used.
from jasper_equiv_check import run_equivalence_check
# jasper_counterexample_check.run_counterexample_checks: alternative Step 1
# validation mechanism (--counterexample-validation) -- checks sva_ol_nl
# against concrete example scenarios derived from the ORIGINAL NL directly,
# instead of against a second independently-generated SVA. See
# generate_and_validate_ol_nl_counterexample below.
from jasper_counterexample_check import run_counterexample_checks
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
NL2SVA_HUMAN_RESET_NOTE = (
    "Note: this testbench declares a signal named exactly `tb_reset` "
    "specifically for use in the assertion's disable iff (...) clause. "
    "Always use exactly `tb_reset` there, never a derived, delayed, or "
    "pulse/shadow variant of it (e.g. tb_reset_d1, tb_reset_d2, "
    "tb_reset_1_cycle_pulse_shadow) -- even if such a variant is what the "
    "property's own logic needs to reference elsewhere in the assertion body."
)


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


def generate_ol_nl_grounding(client, model_name, prompt_text, testbench, operator_context):
    """Best-effort, no-golden OL-NL grounding call (used only for the first
    attempt -- see generate_and_validate_ol_nl for how later rounds revise
    it). Returns the grounded restatement, or prompt_text unchanged if the
    model's reply doesn't contain a parseable 'OL NL:' line.

    operator_context goes in the SYSTEM message, not repeated in the user
    prompt template."""
    prompt = PROMPT_TEMPLATE_OL_NL_NO_GOLDEN.format(testbench=testbench, question=prompt_text)
    system_msg = OL_NL_SYSTEM_PROMPT + "\n\nSVA Operator Context:\n" + operator_context
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


RECHECK_SVA_ORIG_SYSTEM_PROMPT = (
    "You are a helpful bot that checks a SystemVerilog assertion against its natural-language "
    "description and corrects it if there is a mismatch, following the requested format exactly."
)

PROMPT_TEMPLATE_RECHECK_SVA_ORIG = """Given the desired property description:
{description}

please check whether the following SystemVerilog assertion correctly and completely implements it:
{sva}

If it is already correct, repeat it verbatim. If there is a mismatch, output a corrected, complete \
SystemVerilog assertion instead. Enclose your SVA code with ```systemverilog and ```. Only output the \
code snippet and do NOT output anything else."""


def recheck_sva_orig_ques(client, model_name, prompt_text, sva_orig_ques):
    """Path A's independent self-check: does sva_orig_ques (generated
    directly from the real question) actually match that question? Unlike
    the old design, sva_orig_ques is never treated as a fixed, unquestioned
    anchor -- it gets exactly the same chance to self-correct as the OL-NL
    path does, since we've seen real cases (e.g. counter_0/counter_1) where
    the plain-question generation itself was the flawed one, not the
    grounding. Always returns a full assertion (falls back to the input
    unchanged if the reply doesn't parse)."""
    prompt = PROMPT_TEMPLATE_RECHECK_SVA_ORIG.format(description=prompt_text, sva=sva_orig_ques)
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": RECHECK_SVA_ORIG_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    revised = parse_code_response(completion.choices[0].message.content)
    return revised or sva_orig_ques


RECHECK_OL_NL_SYSTEM_PROMPT = (
    "You are a helpful bot that checks an OL-NL restatement and its corresponding SystemVerilog "
    "assertion against the original description, correcting either or both if there is a mismatch, "
    "following the requested format exactly."
)

PROMPT_TEMPLATE_RECHECK_OL_NL = """Original property description:
{prompt_text}

Your OL-NL (operator-level, signal-grounded) restatement of it:
{ol_nl_text}

The SystemVerilog assertion generated from that restatement:
{sva_ol_nl}

Please check two things: (1) does the OL-NL restatement faithfully and completely capture the \
original description's intent (never inventing or dropping a condition), and (2) does the \
SystemVerilog assertion correctly and completely implement the OL-NL restatement. If both are \
already correct, repeat them unchanged. If either has a mismatch, output a corrected version of \
both, in exactly this format and nothing else:

OL NL: <corrected restatement>
```systemverilog
<corrected assertion>
```"""

_RECHECK_OL_NL_LINE_PATTERN = re.compile(r"OL NL:\s*(.*)")


def recheck_ol_nl_path(client, model_name, prompt_text, ol_nl_text, sva_ol_nl):
    """Path B's independent self-check: does ol_nl_text still faithfully
    restate prompt_text, and does sva_ol_nl still faithfully implement
    ol_nl_text? May revise either or both. Returns (ol_nl_text, sva_ol_nl),
    falling back to the inputs unchanged for whichever half doesn't parse."""
    prompt = PROMPT_TEMPLATE_RECHECK_OL_NL.format(
        prompt_text=prompt_text, ol_nl_text=ol_nl_text, sva_ol_nl=sva_ol_nl
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": RECHECK_OL_NL_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = completion.choices[0].message.content
    ol_nl_match = _RECHECK_OL_NL_LINE_PATTERN.search(raw)
    new_ol_nl = ol_nl_match.group(1).strip().splitlines()[0].strip() if ol_nl_match else ol_nl_text
    new_sva = parse_code_response(raw)
    return new_ol_nl, (new_sva or sva_ol_nl)


def generate_sva_direct(client, model_name, user_prompt, rich_operator_context, extra_note=""):
    """Plain single-shot SVA generation -- no retrieval, no rechecking --
    but WITH the rich sva_temporal_operators.json operator context, unlike
    generate_baseline_sva (which has none at all). Used for both halves of
    Step 1's self-consistency check: sva_orig_ques (from the real question)
    and sva_ol_nl (from a candidate OL-NL restatement), built with an
    otherwise-identical prompt so the only thing that can make them diverge
    is the question text itself.

    extra_note, when non-empty (e.g. NL2SVA_HUMAN_RESET_NOTE), is appended
    to the system message verbatim -- used for task-specific generation
    guidance that shouldn't be baked into the shared SYSTEM_PROMPT, which
    is also used unmodified by generate_baseline_sva for 0-shot-baseline
    fidelity."""
    system_msg = SYSTEM_PROMPT + "\n\nSVA Operator Context:\n" + rich_operator_context
    if extra_note:
        system_msg += "\n\n" + extra_note
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": system_msg},
            {"role": "user", "content": user_prompt},
        ],
    )
    return parse_code_response(completion.choices[0].message.content)


def generate_and_validate_ol_nl(
    client, model_name, prompt_text, raw_testbench, user_prompt_orig,
    rich_operator_context, sv_dir, experiment_id, task_id, max_retries=3,
    extra_note="", signal_list_override=None,
):
    """Step 1, redesigned: generate sva_orig_ques directly from the real
    question, and an OL-NL restatement plus the SVA it produces (sva_ol_nl);
    check the two SVAs for real JasperGold formal equivalence -- our only
    available inference-time self-consistency anchor, since there's no
    golden to validate against.

    On a mismatch, BOTH paths get an independent chance to self-correct
    each round (recheck_sva_orig_ques, recheck_ol_nl_path) before
    verification is redone -- sva_orig_ques is never treated as a fixed,
    unquestioned anchor the way the previous design treated it, since real
    cases (counter_0/counter_1) showed the plain-question generation itself
    can be the flawed one, not just the grounding.

    Accepts the OL-NL text the first time the two SVAs prove equivalent;
    falls back to prompt_text unchanged (i.e. no grounding effect
    downstream) if it never converges within max_retries rounds.

    Returns (ol_nl_text, verified: bool).
    """
    sva_orig_ques = generate_sva_direct(client, model_name, user_prompt_orig, rich_operator_context, extra_note)
    ol_nl_text = generate_ol_nl_grounding(client, model_name, prompt_text, raw_testbench, rich_operator_context)
    user_prompt_ol_nl = build_official_nl2sva_human_user_prompt(raw_testbench, ol_nl_text)
    sva_ol_nl = generate_sva_direct(client, model_name, user_prompt_ol_nl, rich_operator_context, extra_note)

    # signal_list_override, when given (nl2sva_human_verified's authoritative
    # signals_for_validity + a parameter/localparam scan), replaces the
    # regex-over-prompt-text heuristic below -- see build_signal_list's own
    # docstring/callers for why that heuristic exists at all for the other
    # tasks (no authoritative list is available there).
    signal_list = signal_list_override if signal_list_override is not None else build_signal_list(user_prompt_orig, raw_testbench)

    for attempt in range(max_retries):
        equivalent, _jg_output = run_equivalence_check(
            testbench=raw_testbench,
            lm_assertion=extract_property_body(sva_orig_ques),
            ref_assertion=extract_property_body(sva_ol_nl),
            signal_list=signal_list,
            sv_dir=sv_dir,
            experiment_id=experiment_id,
            task_id=f"{task_id}_step1iter{attempt}",
        )
        if equivalent:
            return ol_nl_text, True

        # Not equivalent -- let both paths independently self-correct
        # against their own reference point, then redo verification.
        sva_orig_ques = recheck_sva_orig_ques(client, model_name, prompt_text, sva_orig_ques)
        ol_nl_text, sva_ol_nl = recheck_ol_nl_path(client, model_name, prompt_text, ol_nl_text, sva_ol_nl)

    return prompt_text, False


# ---------------------------------------------------------------------------
# Step 1, Plan-1 alternative: counterexample-scenario validation
# (--counterexample-validation). Does NOT replace generate_and_validate_ol_nl
# above (the dual-path self-consistency check remains the default) -- this is
# a selectable alternative mechanism. See jasper_counterexample_check.py's
# module docstring for the full formal-verification mechanism and its
# empirically-confirmed vacuity pitfall (an unreachable scenario makes any
# implication built on it vacuously "proven" regardless of whether sva_ol_nl
# is actually correct -- confirmed live against counter_0/width=1 on
# 2026-08-08, e.g. `count==2` against a 1-bit `count` signal).
# ---------------------------------------------------------------------------

_PARAM_NAME_RE = re.compile(r'\b(?:parameter|localparam)\s+(?:int\s+|real\s+|bit\s+|\[[^\]]+\]\s*)?(\w+)')


def extract_param_names(testbench):
    return list(dict.fromkeys(_PARAM_NAME_RE.findall(testbench)))


COUNTEREXAMPLE_SYSTEM_PROMPT = (
    "You are a helpful bot that derives concrete example scenarios (signal-value traces) from a "
    "natural-language hardware property description, used to formally validate a candidate SVA "
    "implementation of that description, following the requested format exactly."
)

PROMPT_TEMPLATE_COUNTEREXAMPLE_SCENARIOS = """You are given a SystemVerilog RTL testbench and a natural-language \
description of a property that should hold on it. Your job is to derive a small set of concrete example \
scenarios that a verification engineer would use to sanity-check a candidate SVA implementation of this \
property -- WITHOUT being shown any candidate implementation yourself.

RTL testbench (for grounding signal names/widths only -- do not describe the RTL itself):
 {testbench}

The following identifiers are PARAMETERS (fixed constants for this testbench instance) -- you may \
reference them in comparisons (e.g. `count > max`), but NEVER assign them a specific value yourself, \
since they cannot be set at runtime:
 {param_names}

Property description:
 {question}

For each scenario, specify:
- One or more numbered cycles (Cycle 0, Cycle 1, ...), each a comma-separated list of concrete \
signal_name==value assignments (or !=, <, >, <=, >= comparisons against a parameter). Only reference \
signals that actually appear in the testbench (never invented or paraphrased names). Prefer using the \
testbench's own precomputed delayed/history signals (e.g. names ending in _d1, _d2, or similar "previous \
cycle" shadow registers) to express history in a SINGLE cycle where such a signal already exists, rather \
than spelling out a multi-cycle trace -- only use multiple cycles when no such precomputed signal captures \
what you need.
- Exactly one verdict, labeled "Expected: hold" (the property's intent, applied to this exact scenario, \
must be satisfied) or "Expected: violate" (this exact scenario is a case the property's intent says must \
NOT be allowed to happen).

Produce a MIX of both kinds -- at least 2 "hold" scenarios and at least 2 "violate" scenarios, covering \
distinct, meaningfully different situations (not trivial variations of the same one).

Output ONLY the following, in plain text, and nothing else:

Scenario 1:
Expected: <hold|violate>
Cycle 0: <signal==value, signal==value, ...>
Cycle 1: <signal==value, ...>    (omit if single-cycle)

Scenario 2:
...
"""

_SCENARIO_BLOCK_RE = re.compile(r'Scenario\s+\d+\s*:\s*\n(.*?)(?=\nScenario\s+\d+\s*:|\Z)', re.DOTALL)
_EXPECTED_RE = re.compile(r'Expected:\s*(hold|violate)', re.IGNORECASE)
_CYCLE_LINE_RE = re.compile(r'Cycle\s+(\d+)\s*:\s*(.+)')


def parse_counterexample_scenarios(text):
    """Parses PROMPT_TEMPLATE_COUNTEREXAMPLE_SCENARIOS's labeled-block format
    into scenario dicts consumable by jasper_counterexample_check.
    run_counterexample_checks: {"id", "cycle_conditions": [str, ...],
    "expected": "hold"|"violate"}. Malformed blocks (no parseable Expected:
    label, or no Cycle lines) are silently skipped -- same
    tolerance-over-strictness convention as this file's other labeled-text
    parsers (e.g. build_explanation_merge_tree's per-node retry/degrade)."""
    scenarios = []
    for i, block in enumerate(_SCENARIO_BLOCK_RE.findall(text), start=1):
        expected_match = _EXPECTED_RE.search(block)
        if not expected_match:
            continue
        cycles = sorted(
            ((int(num), cond.strip()) for num, cond in _CYCLE_LINE_RE.findall(block) if cond.strip()),
            key=lambda pair: pair[0],
        )
        if not cycles:
            continue
        cycle_conditions = [cond.replace(",", " &&") for _, cond in cycles]
        scenarios.append({
            "id": f"s{i}",
            "cycle_conditions": cycle_conditions,
            "expected": expected_match.group(1).lower(),
        })
    return scenarios


def generate_counterexample_scenarios(client, model_name, prompt_text, testbench):
    """Best-effort scenario generation from the RAW original NL only (never
    shown any candidate SVA/OL-NL) -- that independence is the whole point
    of this alternative to self-consistency validation. Returns a (possibly
    empty) list of scenario dicts; an empty list means this row can't be
    counterexample-validated (caller falls back to unverified)."""
    param_names = extract_param_names(testbench) or ["(none)"]
    prompt = PROMPT_TEMPLATE_COUNTEREXAMPLE_SCENARIOS.format(
        testbench=testbench, question=prompt_text, param_names=", ".join(param_names)
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": COUNTEREXAMPLE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    return parse_counterexample_scenarios(completion.choices[0].message.content)


RECHECK_OL_NL_COUNTEREXAMPLE_SYSTEM_PROMPT = (
    "You are a helpful bot that revises an OL-NL (operator-level, signal-grounded) restatement and its "
    "corresponding SystemVerilog assertion, given specific example scenarios it failed to handle "
    "correctly, following the requested format exactly."
)

PROMPT_TEMPLATE_RECHECK_OL_NL_COUNTEREXAMPLE = """Original property description:
{prompt_text}

Your OL-NL (operator-level, signal-grounded) restatement of it:
{ol_nl_text}

The SystemVerilog assertion generated from that restatement:
{sva_ol_nl}

This restatement/assertion pair was checked against concrete example scenarios derived independently \
from the original description, and did NOT correctly handle the following:
{failure_report}

Please revise the OL-NL restatement and/or the assertion so it correctly handles these scenarios, without \
breaking any scenario it already handled correctly. Output exactly this format and nothing else:

OL NL: <corrected restatement>
```systemverilog
<corrected assertion>
```"""


def build_counterexample_failure_report(results):
    lines = []
    for res in results:
        if res["passed"]:
            continue
        if not res["reachable"]:
            lines.append(f"- Scenario {res['id']}: could not be checked (the scenario itself is not "
                          f"reachable in this testbench -- ignore this one).")
            continue
        lines.append(
            f"- Scenario {res['id']} (expected the property to {res['expected']}): NOT correctly "
            f"handled (JasperGold status: {res['result']})."
        )
    return "\n".join(lines)


def recheck_ol_nl_path_counterexample(client, model_name, prompt_text, ol_nl_text, sva_ol_nl, results):
    """Counterexample-validation variant of recheck_ol_nl_path: same output
    contract, but the feedback names the SPECIFIC scenarios that failed
    (and why), not just "there's a mismatch against another SVA"."""
    failure_report = build_counterexample_failure_report(results)
    prompt = PROMPT_TEMPLATE_RECHECK_OL_NL_COUNTEREXAMPLE.format(
        prompt_text=prompt_text, ol_nl_text=ol_nl_text, sva_ol_nl=sva_ol_nl, failure_report=failure_report
    )
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": RECHECK_OL_NL_COUNTEREXAMPLE_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    raw = completion.choices[0].message.content
    ol_nl_match = _RECHECK_OL_NL_LINE_PATTERN.search(raw)
    new_ol_nl = ol_nl_match.group(1).strip().splitlines()[0].strip() if ol_nl_match else ol_nl_text
    new_sva = parse_code_response(raw)
    return new_ol_nl, (new_sva or sva_ol_nl)


def generate_and_validate_ol_nl_counterexample(
    client, model_name, prompt_text, raw_testbench, user_prompt_orig,
    rich_operator_context, sv_dir, experiment_id, task_id, max_retries=3,
    extra_note="",
):
    """Step 1, Plan-1 alternative (--counterexample-validation): instead of
    self-consistency against an independently-generated sva_orig_ques,
    validates sva_ol_nl against concrete example scenarios derived straight
    from the ORIGINAL NL (never shown any candidate SVA) -- catches a
    shared blind spot between generation paths that self-consistency alone
    cannot, at the cost of needing a scenario-generation call plus a
    heavier per-row JasperGold check. See jasper_counterexample_check.py
    for the underlying formal mechanism.

    Falls back to (prompt_text, False) if scenario generation yields
    nothing parseable, or if verification hasn't succeeded within
    max_retries rounds.

    Returns (ol_nl_text, verified: bool).
    """
    scenarios = generate_counterexample_scenarios(client, model_name, prompt_text, raw_testbench)
    if not scenarios:
        return prompt_text, False

    ol_nl_text = generate_ol_nl_grounding(client, model_name, prompt_text, raw_testbench, rich_operator_context)
    user_prompt_ol_nl = build_official_nl2sva_human_user_prompt(raw_testbench, ol_nl_text)
    sva_ol_nl = generate_sva_direct(client, model_name, user_prompt_ol_nl, rich_operator_context, extra_note)

    for attempt in range(max_retries):
        results, _jg_output = run_counterexample_checks(
            testbench=raw_testbench,
            scenarios=scenarios,
            sva_ol_nl_body=extract_property_body(sva_ol_nl),
            sv_dir=sv_dir,
            experiment_id=experiment_id,
            task_id=f"{task_id}_cexiter{attempt}",
        )
        reachable_results = [r for r in results if r["reachable"]]
        if reachable_results and all(r["passed"] for r in reachable_results):
            return ol_nl_text, True

        ol_nl_text, sva_ol_nl = recheck_ol_nl_path_counterexample(
            client, model_name, prompt_text, ol_nl_text, sva_ol_nl, results
        )

    return prompt_text, False


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


def build_rechecking_context(client, model_name, sva_text, operator_context, max_retries):
    """SVA operator-based rechecking context: the bottom-up explanation-merge
    tree of the generated SVA, or None if sva_graph.py can't parse it (~15%
    of the corpus -- same fallback the main pipeline script uses)."""
    try:
        merge_tree_str = build_and_render_explanation_merge_tree(
            client, model_name, sva_text, operator_context, max_retries=max_retries
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


_SIGNAL_WIDTH_PREFIX_RE = re.compile(r"^\s*\[[^\]]+\]\s*")


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
    else:
        for task_id, raw_testbench, prompt_text, ref_solution in iter_rows(csv_path, task):
            yield task_id, raw_testbench, prompt_text, ref_solution, None


def generate_rag_sva(
    client, model_name, rag_chain, rag_chain_checker, code_retriever,
    operator_context, user_prompt, prompt_text, ol_nl_text, max_retries,
    question_replaced=False, extra_note="",
):
    """Runs one FVEval row through the full pipeline: HybridRetrieval-augmented
    generation, SVA operator-based rechecking, then a few syntax-only cleanup
    passes. Returns (final_sva_text, initial_response_text).

    ol_nl_text is the description used for HybridRetrieval's query and the
    rechecking step -- either prompt_text unchanged, or a best-effort OL-NL
    grounded restatement of it (generate_ol_nl_grounding), depending on
    whether --ol-nl-grounding is set. user_prompt is the official FVEval
    prompt shape used for the actual generation call's human message; by
    default it still carries the original, unmodified prompt_text as its
    question (grounding only augments the system message). If
    question_replaced is True, the caller has already substituted ol_nl_text
    as user_prompt's question itself (--ol-nl-replace-question), so the
    grounding is skipped here to avoid stating it twice."""
    checking_str = build_hybrid_retrieval_context(code_retriever, ol_nl_text)
    ol_nl_context = (
        f"Grounded, signal-level restatement of the property description: {ol_nl_text}\n\n"
        if ol_nl_text != prompt_text and not question_replaced else ""
    )
    llm_result = rag_chain.invoke({
        "keywords_explaination": checking_str,
        "ol_nl_grounding": ol_nl_context,
        "input": user_prompt,
    })
    initial_response = llm_result["answer"]
    sva_text = parse_code_response(initial_response)

    recheck_context = build_rechecking_context(client, model_name, sva_text, operator_context, max_retries)
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
            "differences, and output a corrected, complete SystemVerilog assertion (the "
            "full `assert property (...) ;` statement, including the clocking event and "
            "any disable condition) enclosed in ```systemverilog and ```.\n"
        )
    )
    recheck_system_msg = (
        "You are a helpful bot to modify a SystemVerilog assertion based on the given description.\n\n"
        "SVA Operator Context:\n" + operator_context
    )
    if extra_note:
        recheck_system_msg += "\n\n" + extra_note
    completion = client.chat.completions.create(
        model=model_name,
        messages=[
            {"role": "system", "content": recheck_system_msg},
            {"role": "user", "content": (
                f"Given the desired property description:\n{ol_nl_text}\n\n"
                f"please check whether the generated SystemVerilog assertion below operates "
                f"with the correct logic and timing (i.e., clock cycle):\n{sva_text}\n\n"
                f"{recheck_context or ''}\n{recheck_instruction}"
            )},
        ],
    )
    sva_text = parse_code_response(completion.choices[0].message.content)

    for _ in range(3):
        checker_prompt = (
            "Please correct the following SystemVerilog assertion if it has syntax "
            f"errors (such as unmatched parentheses):\n{sva_text}\n"
            "Output ONLY the corrected assertion as a complete `assert property (...) ;` "
            "statement, enclosed in ```systemverilog and ```.\n"
        )
        checker_result = rag_chain_checker.invoke({"input": checker_prompt})["answer"]
        sva_text = parse_code_response(checker_result)

    return sva_text, initial_response


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


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--task", required=True, choices=list(DEFAULT_CSV_PATHS))
    parser.add_argument("--csv", default=None, help="Defaults to the standard FVEval path for --task")
    parser.add_argument("--output", default=None,
                         help="Defaults to Results/fveval_rag_outputs/{task}_{model_name}_{dynamicrag|baseline0shot}.csv")
    parser.add_argument("--config", default="Src/Config.yml")
    parser.add_argument("--limit", type=int, default=None)
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
    parser.add_argument("--counterexample-validation", action="store_true",
                         help="Requires --ol-nl-grounding. Selects an alternative Step 1 validation "
                              "mechanism: instead of self-consistency against an independently-generated "
                              "sva_orig_ques (the default, generate_and_validate_ol_nl), validates "
                              "sva_ol_nl against concrete example scenarios derived straight from the "
                              "ORIGINAL NL (never shown any candidate SVA) via real JasperGold "
                              "assert+prove checks -- see generate_and_validate_ol_nl_counterexample and "
                              "jasper_counterexample_check.py. Catches a shared blind spot between "
                              "generation paths that self-consistency alone can't, at extra per-row cost.")
    args = parser.parse_args()

    if args.ol_nl_replace_question and not args.ol_nl_grounding:
        parser.error("--ol-nl-replace-question requires --ol-nl-grounding")

    if args.counterexample_validation and not args.ol_nl_grounding:
        parser.error("--counterexample-validation requires --ol-nl-grounding")

    if args.ol_nl_grounding and not args.no_rag:
        import shutil
        if shutil.which("jg") is None:
            parser.error(
                "--ol-nl-grounding now validates each OL-NL restatement against a "
                "plain-generation anchor via real JasperGold equivalence checking "
                "(generate_and_validate_ol_nl) -- `jg` was not found on PATH. Add "
                "JasperGold's bin/ to PATH (and set CDS_LIC_FILE) before running."
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
        if args.ol_nl_grounding:
            step1_jg_sv_dir = f"{output_path}.step1_jgtmp"

        code_store = build_rag_system(config["PDF_Txt"], openai_api_key)
        code_retriever = code_store.as_retriever()

        llm = ChatOpenAI(model=model_name, api_key=openai_api_key)

        task_reset_note = NL2SVA_HUMAN_RESET_NOTE if args.task in ("nl2sva_human", "nl2sva_human_verified") else ""
        system_prompt = (
            SYSTEM_PROMPT
            + (("\n\n" + task_reset_note) if task_reset_note else "")
            + "\nUse the following pieces of retrieved context to help answer the question.\n\n"
            + "{ol_nl_grounding}"
            + "{keywords_explaination}"
            + "{context}"
        )
        prompt = ChatPromptTemplate.from_messages([("system", system_prompt), ("human", "{input}")])
        rag_chain = create_retrieval_chain(code_retriever, create_stuff_documents_chain(llm, prompt))

        system_prompt_checker = (
            "You are a helpful bot that checks the syntax correctness of the given SystemVerilog "
            "assertion and corrects it if there are syntax errors, such as unmatched parentheses. "
            + (("\n" + task_reset_note + "\n") if task_reset_note else "")
            + "Use the following pieces of retrieved context to help answer the question.\n\n"
            "{context}"
        )
        prompt_checker = ChatPromptTemplate.from_messages([("system", system_prompt_checker), ("human", "{input}")])
        rag_chain_checker = create_retrieval_chain(code_retriever, create_stuff_documents_chain(llm, prompt_checker))

    experiment_id = os.path.basename(csv_path).rsplit(".", 1)[0]

    with open(output_path, "w", newline="") as out_file:
        writer = csv.DictWriter(out_file, fieldnames=LMRESULT_FIELDNAMES)
        writer.writeheader()

        for row_index, (task_id, raw_testbench, prompt_text, ref_solution, signals_for_validity) in enumerate(iter_task_rows(args.task, csv_path)):
            if args.limit is not None and row_index >= args.limit:
                break

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
                else:
                    # TODO: give nl2sva_machine / module_sva_nl_manual_editing the
                    # same official-prompt treatment -- still borrowing
                    # run_codev_sva_ol_dfs_eval.py's QUESTION_TEMPLATE here.
                    design_rtl = build_testbench_with_marker(raw_testbench)
                    user_prompt_orig = build_user_prompt(design_rtl, prompt_text)

                # nl2sva_human_verified supplies an authoritative signal list
                # (signals_for_validity, unioned with a parameter/localparam
                # scan) instead of build_signal_list's regex-over-prompt-text
                # heuristic -- None for the other two tasks, which still fall
                # back to that heuristic inside generate_and_validate_ol_nl.
                # build_signal_list(user_prompt_orig, ...) here (not ""):
                # 15/73 verified rows have no signals_for_validity at all
                # (see iter_verified_nl2sva_human_rows) -- for those, union
                # in the same quoted-name-in-prompt heuristic the other two
                # tasks rely on entirely, not just the parameter scan alone.
                # dict.fromkeys(...): dedupe while preserving order.
                # signals_for_validity and build_signal_list's quoted-name
                # heuristic draw from the same "Use the signals '...'"
                # phrasing in the problem text, so they overlap almost
                # completely -- an undeduped union produces duplicate
                # entries in JG's SIGNAL_LIST, which prop_eq_checker's
                # generated wrapper module then rejects outright ("ANSI
                # port 'count' cannot be redeclared"), an elaboration error
                # that calculate_jg_metric silently misclassifies as a
                # functional mismatch (confirmed: this was corrupting the
                # large majority of nl2sva_human_verified rows).
                row_signal_list = (
                    list(dict.fromkeys(signals_for_validity + build_signal_list(user_prompt_orig, raw_testbench)))
                    if signals_for_validity is not None else None
                )

                ol_nl_text = prompt_text
                ol_nl_verified = None
                if args.ol_nl_grounding and not args.no_rag:
                    if args.counterexample_validation:
                        ol_nl_text, ol_nl_verified = generate_and_validate_ol_nl_counterexample(
                            client, model_name, prompt_text, raw_testbench, user_prompt_orig,
                            rich_operator_context, step1_jg_sv_dir, experiment_id, task_id,
                            extra_note=task_reset_note,
                        )
                    else:
                        ol_nl_text, ol_nl_verified = generate_and_validate_ol_nl(
                            client, model_name, prompt_text, raw_testbench, user_prompt_orig,
                            rich_operator_context, step1_jg_sv_dir, experiment_id, task_id,
                            extra_note=task_reset_note, signal_list_override=row_signal_list,
                        )
                    print(f"    OL-NL {'verified' if ol_nl_verified else 'NOT verified (fell back to original question)'}")

                if args.ol_nl_replace_question:
                    if args.task in ("nl2sva_human", "nl2sva_human_verified"):
                        user_prompt = build_official_nl2sva_human_user_prompt(raw_testbench, ol_nl_text)
                    else:
                        user_prompt = build_user_prompt(design_rtl, ol_nl_text)
                else:
                    user_prompt = user_prompt_orig

                if args.no_rag:
                    sva_text, initial_response = generate_baseline_sva(client, model_name, user_prompt)
                else:
                    sva_text, initial_response = generate_rag_sva(
                        client, model_name, rag_chain, rag_chain_checker, code_retriever,
                        rich_operator_context, user_prompt, prompt_text, ol_nl_text, args.max_retries,
                        question_replaced=args.ol_nl_replace_question, extra_note=task_reset_note,
                    )
            except Exception as error:
                print(f"    failed ({error}), skipping")
                continue

            writer.writerow({
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
            })
            out_file.flush()

    print(f"Wrote responses to {output_path}")


if __name__ == "__main__":
    main()
