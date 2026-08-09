"""
Stage 2 of the CodeV-SVA OL-NL + two-part decomposition pipeline (see the plan
in /Users/fch/.claude/plans/polished-chasing-wreath.md).

Defines "OL NL" (operator-level natural-language) description: a rewrite of a
CodeV-SVA question -- which may be written at any abstraction level, from
literal/operator-level ("Both sig_F and sig_H are high...") to abstract/
domain-level ("that the counter does not underflow") -- into a statement that
is grounded in the SVA's actual signal names and maps clause-by-clause onto
its parsed operator tree. This is what makes the existing DFS decomposition
pipeline (generate_dfs_explanation.py, built for qwen_explanation.jsonl's
already-operator-level explanations) applicable to CodeV-SVA's mixed-
abstraction questions: normalize first, then reuse that pipeline unmodified.

Validation: the OL-NL statement is only accepted once a *freshly generated*
candidate SVA -- produced from the OL-NL text alone, the same way a real
NL2SVA model would -- is formally proven equivalent to the golden SVA via
JasperGold (jasper_equiv_check.py, reusing the existing FVEval harness's own
tcl scripts). A cheap string-match check (does every leaf signal name appear
in the text) was tried first and dropped: it can't catch a statement that
names all the right signals but gets the logic between them wrong, which is
exactly the failure mode that matters here.

This module is meant to be imported by generate_codev_sva_reasoning_dataset.py
(Stage 2+3+4 orchestrator), but can also run standalone against a few example
records for spot-checking:

    python verilogFinetune/generate_ol_nl_explanation.py \\
        --input verilogFinetune/data/CodeV-SVA-dataset-training-83K.jsonl \\
        --indices 30354,60725 \\
        --config Src/Config.yml \\
        --sv-dir /tmp/ol_nl_validation
"""
import argparse
import json
import os
import re
import sys
import time

import yaml

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from generate_prompt_guided_explanation import add_provider_arg, build_llm_client, load_operator_context
from jasper_equiv_check import run_equivalence_check
from sva_graph import build_operator_signal_graph, dfs_nodes

SYSTEM_PROMPT = (
    "You are a helpful bot that rewrites natural-language descriptions of SystemVerilog "
    "assertions into a precise, signal-grounded, operator-level form, following the "
    "requested format exactly."
)

PROMPT_TEMPLATE_OL_NL = """You are given a SystemVerilog assertion, the RTL testbench it belongs to, \
and a natural-language description of it. The description may be written at any level of \
abstraction -- sometimes it already names signals and operators directly, sometimes it only \
describes the high-level intent (e.g. "that the counter does not underflow") without naming \
every signal involved.

Your job: rewrite the description as an "OL NL" (operator-level natural-language) statement \
-- one that names ONLY the assertion's actual signals (never invented or paraphrased names), \
and whose clauses map directly onto the assertion's top-level structure (e.g. for an \
implication, state the antecedent condition, then the consequent, in that order).

Two worked examples of this transformation:

Example 1 (already close to operator-level, just needs tightening):
 SVA: (ERROR == 1) |-> (PSLVERR == 1)
 OL NL: When ERROR equals 1, then PSLVERR equals 1 from the current clock cycle

Example 2 (abstract/domain-level input, must be grounded in the real signals):
 SVA: asrt: assert property (@(posedge clk) disable iff (tb_reset)
    ((count_d1 === min) && !jump_vld_d1 && ((count < min) || (count >= max)) && !tb_reset_1_cycle_pulse_shadow) !== 1'b1
 );
 Description given: that the counter does not underflow. Use the signals 'count', 'count_d1', \
'jump_vld_d1', and 'tb_reset_1_cycle_pulse_shadow'.
 OL NL: It must never be the case that count_d1 equals min and jump_vld_d1 is not asserted and \
(count is less than min or count is greater than or equal to max) and \
tb_reset_1_cycle_pulse_shadow is not asserted

Now do the same for this assertion:

SVA:
 {sva}

RTL testbench (for grounding signal names/widths/parameters only -- do not describe the RTL itself):
 {testbench}

Description given (any abstraction level):
 {question}

SVA Operator Context:
 {operator_context}

Output exactly one labeled line, in plain text (not JSON), and nothing else:

OL NL: <the operator-level, signal-grounded statement>
"""

PROMPT_TEMPLATE_OL_NL_RETRY = PROMPT_TEMPLATE_OL_NL + """
Your previous attempt was: {previous_ol_nl}

A candidate SVA generated from that statement (by a separate model, given only the statement \
and the testbench -- not the golden SVA) was:
{candidate_sva}

That candidate was checked against the golden SVA above with a formal equivalence tool and was \
NOT proven equivalent. Revise the OL NL statement so that it captures the golden SVA's exact \
logic -- not just its signals -- precisely enough that a model translating only the statement \
back into SVA would reconstruct something equivalent to the original.
"""

CANDIDATE_SVA_SYSTEM_PROMPT = (
    "You are an AI assistant tasked with formal verification of register transfer level (RTL) designs.\n"
    "Your job is to translate a description of an assertion to concrete SystemVerilog Assertion (SVA) implementation.\n"
)

PROMPT_TEMPLATE_CANDIDATE_SVA = """Here is the testbench to perform your translation:
{testbench}
Question: Create a SVA assertion that checks: {ol_nl}
Enclose your SVA code with ```systemverilog and ```. Only output the code snippet and do NOT output anything else.

For example,
```systemverilog
asrt: assert property (@(posedge clk) disable iff (tb_reset)
    (a && b) != 1'b1
);
```
Answer:"""

_OL_NL_PATTERN = re.compile(r"OL NL:\s*(.*)", re.DOTALL)

_IDENTIFIER_RE = re.compile(r"^[A-Za-z_]\w*")

_CANDIDATE_SVA_CODE_RE = re.compile(r"```systemverilog\s*(.*?)```", re.DOTALL)

_PARAMETER_RE = re.compile(r"\b(?:parameter|localparam)\s+(?:int\s+|real\s+|bit\s+|\[[^]]+\]\s*)?(\w+)")


def split_user_content(user_content):
    """Splits a CodeV-SVA-dataset-training-83K user turn into (testbench,
    question) -- the testbench (RTL with the `// TODO: ASSERTION` injection
    point) and the trimmed `Question: ...` text (boilerplate output-format
    instructions and the few-shot example dropped, since only the actual
    requirement is needed as input to the OL-NL transform)."""
    q_idx = user_content.find("Question:")
    testbench = user_content[:q_idx].replace(
        "Here is the testbench to perform your translation:\n", ""
    ).strip()
    question_full = user_content[q_idx:].strip()
    enclose_idx = question_full.find("Enclose your SVA")
    question = (question_full[:enclose_idx] if enclose_idx != -1 else question_full).strip()
    return testbench, question


def parse_ol_nl_response(text):
    match = _OL_NL_PATTERN.search(text)
    if not match:
        return None
    return match.group(1).strip().splitlines()[0].strip()


def leaf_signal_identifiers(root):
    """Base identifier of every signal leaf in the parsed tree (e.g. 'Q[3:0]' ->
    'Q'), skipping pure-numeric-literal leaves and placeholders, since those
    aren't expected to appear verbatim in prose."""
    identifiers = []
    for node in dfs_nodes(root):
        if node["type"] != "signal":
            continue
        match = _IDENTIFIER_RE.match(node["label"])
        if match:
            identifiers.append(match.group(0))
    return identifiers


def build_signal_list(root, testbench):
    """Signal names to pass to the JasperGold equivalence checker's
    SIGNAL_LIST define: every leaf signal in the golden SVA's parsed tree,
    plus any module parameters (a golden SVA often compares a signal against
    a parameter, e.g. `count >= max`), mirroring
    NL2SVAHumanEvaluator.evaluate_jg's own signal_list construction
    (leaf/quoted signal names + parameter regex over the testbench)."""
    identifiers = list(dict.fromkeys(leaf_signal_identifiers(root)))
    for match in _PARAMETER_RE.finditer(testbench):
        name = match.group(1)
        if name not in identifiers:
            identifiers.append(name)
    return identifiers


def build_ol_nl_prompt(question, sva, testbench, operator_context, correction=None):
    if correction:
        return PROMPT_TEMPLATE_OL_NL_RETRY.format(
            sva=sva,
            testbench=testbench,
            question=question,
            operator_context=operator_context,
            previous_ol_nl=correction["previous_ol_nl"],
            candidate_sva=correction["candidate_sva"],
        )
    return PROMPT_TEMPLATE_OL_NL.format(
        sva=sva,
        testbench=testbench,
        question=question,
        operator_context=operator_context,
    )


def generate_candidate_sva(client, model, ol_nl_text, testbench):
    """Generates a candidate SVA from the OL-NL statement alone (plus the
    testbench, for signal/width context) -- the same task a real NL2SVA model
    faces at inference time -- so its formal equivalence to the golden SVA is
    a real test of whether the OL-NL statement fully captures the assertion's
    semantics, not just its vocabulary."""
    prompt = PROMPT_TEMPLATE_CANDIDATE_SVA.format(testbench=testbench, ol_nl=ol_nl_text)
    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": CANDIDATE_SVA_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
    )
    match = _CANDIDATE_SVA_CODE_RE.search(completion.choices[0].message.content)
    return match.group(1).strip() if match else None


def generate_ol_nl(client, model, question, sva, testbench, operator_context, max_retries, sv_dir, task_id):
    """Returns (ol_nl_text, verified). verified is False if no candidate SVA
    generated from the OL-NL statement was proven formally equivalent to the
    golden SVA within max_retries attempts -- callers decide whether to use
    the best-effort text anyway or drop the record.

    Deliberately does NOT catch exceptions from the API calls here (unlike
    the missing-label/no-code-block/not-equivalent cases below, which are
    genuine content-quality retries and consume this function's own
    max_retries budget): an API/infra failure (rate limit, network error,
    billing) is not evidence this record's statement is bad, so it shouldn't
    be spent against the same budget or -- worse -- exhaust it and return
    verified=False, which looks identical to a real verification failure and
    causes the caller to permanently drop the record for backfill. Letting
    it propagate means the caller's own retry-with-backoff (generate_
    codev_sva_reasoning_dataset.worker()) handles it once, uniformly, for
    the whole record (Stage 2 and Stage 3 together) instead of duplicating
    -- and getting subtly wrong -- backoff logic at every layer that happens
    to make an API call. (An earlier version of this function caught and
    retried API errors right here, which meant a sustained outage silently
    exhausted this loop's budget for every single record, every time,
    indistinguishable from genuine bad statements -- see
    PIPELINE_EXECUTION_NOTES.md.)

    sv_dir/task_id are passed straight through to jasper_equiv_check --
    task_id should be unique per concurrent caller (e.g. the source record's
    index) so parallel JasperGold invocations don't collide on temp files."""
    root = build_operator_signal_graph(sva)
    signal_list = build_signal_list(root, testbench)
    correction = None
    last_text = None

    for attempt in range(max_retries):
        prompt = build_ol_nl_prompt(question, sva, testbench, operator_context, correction)
        completion = client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        )
        text = parse_ol_nl_response(completion.choices[0].message.content)

        if text is None:
            print("    OL-NL reply missing 'OL NL:' label, retrying...")
            time.sleep(2 ** attempt)
            continue

        last_text = text
        candidate_sva = generate_candidate_sva(client, model, text, testbench)
        if candidate_sva is None:
            print("    candidate-SVA generation returned no code block, retrying...")
            correction = {"previous_ol_nl": text, "candidate_sva": "(no code block returned)"}
            continue

        equivalent, jg_output = run_equivalence_check(
            testbench=testbench,
            lm_assertion=candidate_sva,
            ref_assertion=sva,
            signal_list=signal_list,
            sv_dir=sv_dir,
            task_id=str(task_id),
        )
        if equivalent:
            return text, True

        print(f"    candidate SVA not formally equivalent to golden SVA (jg: {jg_output[:200]!r}), retrying...")
        correction = {"previous_ol_nl": text, "candidate_sva": candidate_sva}

    return last_text, False


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="verilogFinetune/data/CodeV-SVA-dataset-training-83K.jsonl")
    parser.add_argument("--indices", required=True, help="Comma-separated line indices to spot-check")
    parser.add_argument("--operators", default="operators.json")
    parser.add_argument("--config", default="Src/Config.yml")
    parser.add_argument("--model", default="o4-mini")
    add_provider_arg(parser)
    parser.add_argument("--max-retries", type=int, default=3)
    parser.add_argument("--sv-dir", default="/tmp/ol_nl_validation", help="Scratch dir for JasperGold temp files")
    args = parser.parse_args()

    with open(args.config) as file:
        config = yaml.safe_load(file)
    client = build_llm_client(config, args.provider)
    operator_context = load_operator_context(args.operators)

    targets = set(int(i) for i in args.indices.split(","))
    with open(args.input) as file:
        for index, line in enumerate(file):
            if index not in targets:
                continue
            record = json.loads(line)
            user = record["messages"][1]["content"]
            assistant = record["messages"][2]["content"]

            testbench, question = split_user_content(user)

            final_part = assistant.split("</think>", 1)[1] if "</think>" in assistant else assistant
            sva_match = re.search(r"```systemverilog\s*(.*?)```", final_part, re.DOTALL)
            sva = sva_match.group(1).strip()

            print(f"{'='*80}\nindex {index}\n{'='*80}")
            print("question:", question)
            print("sva:", sva)
            ol_nl, verified = generate_ol_nl(
                client, args.model, question, sva, testbench, operator_context,
                args.max_retries, args.sv_dir, task_id=index,
            )
            print(f"OL NL ({'verified equivalent' if verified else 'NOT verified equivalent'}): {ol_nl}")
            print()


if __name__ == "__main__":
    main()
