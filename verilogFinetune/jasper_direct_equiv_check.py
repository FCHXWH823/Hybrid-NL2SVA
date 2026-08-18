"""
Direct JasperGold formal equivalence checking between an LM-generated SVA
property expression and its golden reference, via `iff` (property-level
biconditional) + prove -- bypassing the official FVEval harness's
prop_eq_checker macro (Evaluation/FVRuleLearner/FVEval/tool_scripts/pec/
pec.tcle) entirely.

Motivation: prop_eq_checker is IP-protected/encrypted (no source access),
and has a CONFIRMED real limitation (2026-08-10, nl2sva_human_verified
FVEval-NL2SVA-Human-1's `count >= min` vs `count >= 0`, where `min`
elaborates to the literal constant 0 in that testbench): a direct proof of
`(count>=min) <-> (count>=0)` and `(count>=min) iff (count>=0)` both
succeed (`proven`) -- confirmed genuinely, fully equivalent -- yet
prop_eq_checker, given the SAME two expressions each wrapped in an
identical `|->` antecedent, reports only a one-directional "implies"
relationship, not full equivalence. That means calculate_jg_metric's
strict `functionality` score was UNDER-COUNTING genuinely-correct answers
whenever they happened to trip this blind spot -- a scoring-tool
limitation, not a generation-quality one. Since prop_eq_checker's source
can't be inspected or patched, this reimplements equivalence checking
directly and empirically more reliably instead.

`iff` (not `<->`) is used deliberately: our SVAs are PROPERTIES, often
containing temporal/sequence operators (|->, |=>, strong(...), etc.), and
`iff` is SVA's property-level biconditional (property_expr1 iff
property_expr2 -- "the two properties have the same truth outcome",
per this repo's own sva_temporal_operators.json), correctly composing with
such operators; `<->` is a boolean/expression-level operator not
guaranteed to accept property-typed operands.

Mechanism (embedding named assert properties directly into the testbench,
`clock`/`reset -none`, `prove -all`, `get_status <module>.<name>`) mirrors
jasper_counterexample_check.py, empirically validated the same way before
being wired into any pipeline code -- confirmed live against real `jg`:
the known-equivalent case above proves `iff` as expected, and a
deliberately-wrong case correctly comes back with a real counterexample
(`cex`), not a false positive.

The two one-directional "relaxed" checks use `not (...) or (...)`, NOT
`|->`/`|=>` -- confirmed live (VERI-1243 "operator |-> is only allowed in a
property, not in sequence") that `(lm) |-> (ref)` fails to even elaborate
whenever lm/ref are themselves implication properties (the typical NL2SVA
shape: `antecedent |-> consequent`), since `|->`'s antecedent must be a
SEQUENCE, and a property containing `|->` is not one. `not`/`or` are
property-level connectives (this is literally how `iff` itself is defined
per the LRM), so they compose correctly regardless of what lm/ref contain.
Re-validated live after this fix: both the corrected one-directional checks
and `iff` prove correctly on a real multi-row testbench, and still
correctly report `cex` (not a false positive) on a deliberately-wrong pair.
"""
import os
import re
import subprocess

_MODULE_NAME_RE = re.compile(r'\bmodule\s+(\w+)')
_STATUS_LINE_RE = re.compile(r'^DIRECTEQSTATUS (\S+) (\S+)\s*$')

_IFF_NAME = "direct_iff"
_LM_IMPLIES_REF_NAME = "direct_lm_implies_ref"
_REF_IMPLIES_LM_NAME = "direct_ref_implies_lm"
_ALL_NAMES = [_IFF_NAME, _LM_IMPLIES_REF_NAME, _REF_IMPLIES_LM_NAME]


def run_direct_equivalence_check(
    testbench,
    lm_assertion,
    ref_assertion,
    sv_dir,
    experiment_id="direct_equiv",
    task_id="0",
    clock_signal="clk",
    timeout=90,
):
    """testbench: an elaborable module with NO assertion inside (the two
    property expressions are supplied separately). lm_assertion/
    ref_assertion: BARE property expressions (no assert property/clock/
    disable iff wrapper -- e.g. already run through score_nl2sva_human.
    extract_property_body / run_rag_on_fveval_benchmarks.wrap_property_
    expression's inverse).

    Returns (metric: dict, raw_output: str) where metric matches
    calculate_jg_metric's exact fields/semantics:
      syntax        = 1.0 unless elaboration/analysis failed
      functionality = 1.0 only if the `iff` property is proven (full
                      equivalence)
      func_relaxed  = 1.0 if functionality, OR either one-directional
                      implication (lm_assertion |-> ref_assertion, or the
                      reverse) is proven
    """
    module_match = _MODULE_NAME_RE.search(testbench)
    if not module_match:
        raise ValueError("Could not find a `module <name>` declaration in the testbench")
    module_name = module_match.group(1)

    lm = lm_assertion.strip()
    ref = ref_assertion.strip()
    # `|->`/`|=>` require a SEQUENCE on the left-hand side, not a property --
    # confirmed live (VERI-1243 "operator |-> is only allowed in a property,
    # not in sequence") that this breaks whenever lm/ref are themselves
    # implication properties (the common NL2SVA shape: `antecedent |->
    # consequent`), since that makes the antecedent a property, not a
    # sequence. `not (...) or (...)` is the property-level connective used,
    # which is exactly how `iff` itself is defined per the LRM, so it
    # composes correctly regardless of what lm/ref contain.
    block = (
        f"{_IFF_NAME}: assert property (@(posedge {clock_signal}) ({lm}) iff ({ref}));\n"
        f"{_LM_IMPLIES_REF_NAME}: assert property (@(posedge {clock_signal}) not ({lm}) or ({ref}));\n"
        f"{_REF_IMPLIES_LM_NAME}: assert property (@(posedge {clock_signal}) not ({ref}) or ({lm}));\n"
    )
    if "endmodule" in testbench:
        tb_text = testbench.replace("endmodule", block + "endmodule", 1)
    else:
        tb_text = testbench + "\n" + block

    sv_dir = os.path.abspath(sv_dir)
    os.makedirs(sv_dir, exist_ok=True)
    sva_path = os.path.join(sv_dir, f"{experiment_id}_{task_id}.sva")
    with open(sva_path, "w") as file:
        file.write(tb_text)

    status_puts = "\n".join(f'puts "DIRECTEQSTATUS {n} [get_status {module_name}.{n}]"' for n in _ALL_NAMES)
    tcl_content = f"""clear -all
analyze -clear
analyze -sv12 {sva_path}
elaborate
clock {clock_signal}
reset -none
prove -all
{status_puts}
exit
"""
    tcl_path = os.path.join(sv_dir, f"{experiment_id}_{task_id}.tcl")
    with open(tcl_path, "w") as file:
        file.write(tcl_content)

    tmp_jg_proj_dir = os.path.join(sv_dir, "jg_direct_equiv", f"{experiment_id}_{task_id}")

    jg_command = ["jg", "-fpv", "-batch", "-tcl", tcl_path, "-proj", tmp_jg_proj_dir, "-allow_unsupported_OS"]
    try:
        result = subprocess.run(jg_command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return {"syntax": 0.0, "functionality": 0.0, "func_relaxed": 0.0}, f"TIMEOUT: JasperGold execution exceeded {timeout}s"

    output = result.stdout.strip()
    statuses = {}
    for line in output.splitlines():
        m = _STATUS_LINE_RE.match(line.strip())
        if m:
            statuses[m.group(1)] = m.group(2)

    if not all(n in statuses for n in _ALL_NAMES):
        # Elaboration/analysis never completed -- a real syntax error in
        # either lm_assertion or ref_assertion (or a batch-wide failure;
        # confirmed live that ONE malformed property in the same file
        # blocks elaboration of the whole module, so ALL statuses go
        # missing together, not just the broken one).
        return {
            "syntax": 0.0, "functionality": 0.0, "func_relaxed": 0.0,
            "lm_implies_ref": False, "ref_implies_lm": False,
        }, output

    iff_proven = statuses[_IFF_NAME] == "proven"
    lm_implies_ref = statuses[_LM_IMPLIES_REF_NAME] == "proven"
    ref_implies_lm = statuses[_REF_IMPLIES_LM_NAME] == "proven"

    metric = {
        "syntax": 1.0,
        "functionality": 1.0 if iff_proven else 0.0,
        "func_relaxed": 1.0 if (iff_proven or lm_implies_ref or ref_implies_lm) else 0.0,
        # Directional detail beyond the FVEval-standard fields above -- lets
        # a caller give useful feedback on a mismatch ("candidate is too
        # strong/weak relative to the other side") without a full
        # value-level counterexample trace, which JasperGold's batch-mode
        # get_status doesn't expose. lm_implies_ref: lm_assertion's truth
        # set is a subset of ref_assertion's (lm is at least as strong).
        "lm_implies_ref": lm_implies_ref,
        "ref_implies_lm": ref_implies_lm,
    }
    return metric, output


_ERROR_LINE_RE = re.compile(r'^\[?ERROR\b.*$', re.MULTILINE)


def summarize_elaboration_errors(jg_output, max_lines=8):
    """Pulls out just the `[ERROR ...]`/`ERROR (...)` lines from a raw JG
    console log, for feeding back to an LLM as targeted correction context
    -- the full log is mostly boilerplate (version banners, INFO lines,
    per-file analysis progress) that only dilutes the one thing a fix
    prompt actually needs: what specifically went wrong."""
    lines = _ERROR_LINE_RE.findall(jg_output)
    if not lines:
        return jg_output[-500:]
    return "\n".join(lines[:max_lines])


def check_sva_elaboration(testbench, sva_expression, sv_dir, experiment_id="elab_check", task_id="0", clock_signal="clk", disable_signal="tb_reset", timeout=60):
    """Real JasperGold elaboration-only check (analyze + elaborate, no
    clocking/reset/prove -- we don't care about proving anything here, just
    whether the design elaborates at all) for ONE candidate SVA property
    expression against the REAL testbench.

    disable_signal: the testbench's disable iff signal (this benchmark's
    fixed nl2sva_human(_verified) convention is `tb_reset`). Pass None for
    testbenches that declare no reset signal at all (e.g.
    nl2sva_machine_verified's bare dummy modules) to omit the disable iff
    clause entirely -- with the default, such a testbench would fail
    elaboration on an undeclared `tb_reset` identifier regardless of
    whether sva_expression itself is valid.

    Motivation: generate_rag_sva's syntax-cleanup loop used to ask the model
    "does this have syntax errors?" with no ground truth at all -- confirmed
    live (FVEval-NL2SVA-Human-8) that an ungrounded "helpful" checker can
    silently corrupt an already-correct expression (there it dropped a `|`
    reduction-OR operator) while fixing nothing real, since it has no way to
    know whether a problem genuinely exists. This gives it one: real
    elaboration errors surface here exactly the same way they do during
    full scoring (VERI-1128 "not declared", VERI-1011 "cannot index into
    non-array type", VERI-2061 "unresolved external task/function
    reference", etc. -- all confirmed live across many traces this
    session), so a caller can react to a REAL error, or skip straight past
    the fix step entirely when there isn't one.

    sva_expression: a BARE property expression (no assert property/clock/
    disable iff wrapper -- wrapped here the same way wrap_property_
    expression does, assuming this benchmark's fixed clk/tb_reset
    convention).

    Returns (ok: bool, jg_output: str)."""
    module_match = _MODULE_NAME_RE.search(testbench)
    if not module_match:
        raise ValueError("Could not find a `module <name>` declaration in the testbench")

    disable_clause = f" disable iff ({disable_signal})" if disable_signal else ""
    block = (
        f"elabcheck: assert property (@(posedge {clock_signal}){disable_clause}\n"
        f"    {sva_expression.strip()}\n);\n"
    )
    if "endmodule" in testbench:
        tb_text = testbench.replace("endmodule", block + "endmodule", 1)
    else:
        tb_text = testbench + "\n" + block

    sv_dir = os.path.abspath(sv_dir)
    os.makedirs(sv_dir, exist_ok=True)
    sva_path = os.path.join(sv_dir, f"{experiment_id}_{task_id}.sva")
    with open(sva_path, "w") as file:
        file.write(tb_text)

    tcl_content = f"""clear -all
analyze -clear
analyze -sv12 {sva_path}
elaborate
exit
"""
    tcl_path = os.path.join(sv_dir, f"{experiment_id}_{task_id}.tcl")
    with open(tcl_path, "w") as file:
        file.write(tcl_content)

    tmp_jg_proj_dir = os.path.join(sv_dir, "jg_elab_check", f"{experiment_id}_{task_id}")

    jg_command = ["jg", "-fpv", "-batch", "-tcl", tcl_path, "-proj", tmp_jg_proj_dir, "-allow_unsupported_OS"]
    try:
        result = subprocess.run(jg_command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: JasperGold execution exceeded {timeout}s"

    output = result.stdout.strip()
    ok = not re.search(r'\[ERROR', output)
    return ok, output
