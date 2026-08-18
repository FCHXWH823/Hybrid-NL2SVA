"""
Counterexample-scenario-based validation of an OL-NL-derived SVA
(sva_ol_nl), as an alternative Step 1 mechanism to the JG-equivalence
self-consistency check in generate_and_validate_ol_nl (run_rag_on_fveval_
benchmarks.py) -- selected via --counterexample-validation.

Where the self-consistency check can only catch cases where two
independently-generated SVAs (sva_orig_ques vs sva_ol_nl) DISAGREE, this
checks sva_ol_nl against concrete example scenarios derived straight from
the ORIGINAL natural-language description (never shown any candidate SVA),
so it isn't subject to the same shared-blind-spot risk.

POSITIVE-ONLY, NO TESTBENCH (2026-08-10 redesign): scenarios are all
"should hold" cases -- no LLM-labeled "violate" scenarios anymore, removing
the mislabeling risk that drove a real regression earlier this session
(count==max wrongly labeled "violate" -- max is a legal boundary value, not
an overflow condition -- which corrupted an already-correct answer through
several rounds of unnecessary SOR "fixes").

Each scenario becomes ONE formal claim: does the concrete example imply the
candidate holds? `(scenario_seq) |-> (sva_ol_nl_body)`. No testbench/RTL is
combined in -- confirmed by reading run_jg_nl2sva_human.tcl (the official
FVEval scoring script): it elaborates the testbench, then `clear -all`s
that elaboration completely BEFORE calling prop_eq_checker, which works
from just the two bare assertion strings and a signal_list.

PLAIN SIGNAL LIST (2026-08-10, explicit instruction -- matching
prop_eq_checker's own convention exactly, not just its spirit): the caller
provides the authoritative signal_list directly (this pipeline already
builds one per row -- signals_for_validity / row_signal_list in main()),
and every name in it becomes a plain, generic `logic` declaration here --
no width lookup, no parameter-vs-signal distinction, mirroring pec.md's own
documented usage (`set signal_list [split $SIGNAL_LIST ","]` ->
`prop_eq_checker $LM_ASSERT_TEXT $REF_ASSERT_TEXT "" "" $signal_list`,
where every entry is just a name, clk/rst included or omitted by the
caller).

NOTE, explicitly accepted: this deliberately reintroduces the same
trade-off that caused prop_eq_checker's own confirmed limitation (missing
full equivalence between `count>=min` and `count>=0` when `min` elaborates
to 0) -- if signal_list contains a parameter name, its real value is
thrown away in favor of an unconstrained free variable, same as any other
signal. An earlier version of this module avoided that by parsing the
testbench for parameters' real defining expressions and signals' real
declared widths; that machinery was removed in favor of this simpler,
caller-provided-list convention. `clk` is always added automatically
(needed for our own explicit `@(posedge clk)` usage, which prop_eq_checker
itself doesn't need since it strips the clocking event before ever seeing
the assertion bodies).

NO reachability cover and NO tautology guard (2026-08-10, explicit
instruction -- simplicity over the extra per-row JG cost/complexity those
added): this means a scenario that happens to be unsatisfiable (e.g.
count==2 against a 1-bit count) will vacuously "pass" regardless of whether
sva_ol_nl is actually correct, and an over-permissive/tautological
candidate (e.g. a hidden 1'b1) can pass every scenario undetected -- known,
accepted trade-offs, not oversights. See git history for the earlier
reachability-cover + taut-guard version if this needs revisiting.

KNOWN LIMITATION (accepted, 2026-08-10): because signals are fully free/
abstract (no real DUT behavior), this cannot catch a |-> vs |=> timing bug
whenever the consequent happens to be tautological given the testbench's
real parameter values -- confirmed live against counter_tb (width=1):
`count <= max` is always true regardless of when it's evaluated, since
count's next-cycle value is completely unconstrained rather than governed
by real increment/decrement logic, so a deliberately-mistimed candidate
(|=> instead of the correct |->) scores identically to the correct one.
Accepted trade-off: still catches most other bug classes (wrong signals,
wrong comparisons, dropped conditions, malformed logic); only misses timing
bugs on the subset of rows whose consequent is coincidentally tautological.

All scenarios for one row are embedded together and checked in a single
JasperGold invocation -- elaboration/licensing overhead is per-process, not
per-property, and jg startup dominates runtime for small properties like
these.

Mechanism (clock/reset declaration, `prove -all`, `get_status <module>.
<name>`) mirrors jasper_direct_equiv_check.py, empirically validated live
against real `jg` before being wired into any pipeline code.
"""
import os
import re
import subprocess


def build_abstract_declarations(signal_list):
    """Every provided name becomes a plain, generic `logic` declaration --
    no width, no parameter special-casing -- matching prop_eq_checker's own
    signal_list convention (see module docstring). `clk` is always added,
    de-duplicated, even if the caller's list already includes it."""
    names = list(dict.fromkeys([*signal_list, "clk"]))
    return "\n".join(f"logic {name};" for name in names)


def build_scenario_sequence(cycle_conditions):
    """cycle_conditions: ordered list of boolean-expression strings, one
    per cycle (oldest first), concatenated with ##1 delays. A single-cycle
    scenario is just cycle_conditions == [one string] -- no ##1 needed."""
    return " ##1 ".join(f"({c})" for c in cycle_conditions)


def build_abstract_module_with_scenario_checks(
    signal_list, scenarios, sva_ol_nl_body, module_name="abstract_scenario_check", clock_signal="clk"
):
    """scenarios: list of dicts {"id": str, "cycle_conditions": [str, ...]}
    (all treated as "should hold" -- see module docstring). id must be a
    valid, unique SV identifier fragment (used directly in property names).

    signal_list: the row's authoritative signal names (caller-provided --
    see module docstring). The returned module is entirely self-contained,
    built fresh -- no testbench is elaborated or embedded.

    Returns (module_text, module_name, claim_names).
    """
    decls = build_abstract_declarations(signal_list)

    lines = [f"module {module_name};", decls]
    claim_names = []
    for sc in scenarios:
        seq = build_scenario_sequence(sc["cycle_conditions"])
        claim_name = f"claim_{sc['id']}"
        claim_names.append(claim_name)
        lines.append(f"{claim_name}: assert property (@(posedge {clock_signal}) ({seq}) |-> ({sva_ol_nl_body}));")
    lines.append("endmodule")
    return "\n".join(lines) + "\n", module_name, claim_names


_STATUS_LINE_RE = re.compile(r'^JGCEXSTATUS (\S+) (\S+)\s*$')


def run_counterexample_checks(
    signal_list,
    scenarios,
    sva_ol_nl_body,
    sv_dir,
    experiment_id="ol_nl_counterexample_validation",
    task_id="0",
    clock_signal="clk",
    timeout=120,
):
    """Runs all of `scenarios` against `sva_ol_nl_body` in one JasperGold
    invocation (a fresh, self-contained abstract module -- see
    build_abstract_module_with_scenario_checks). signal_list: the row's
    authoritative signal names (e.g. main()'s row_signal_list).

    Returns (results, raw_output): results is one dict per scenario, in
    order -- {"id", "result": str, "passed": bool} -- where `result` is
    JasperGold's raw get_status string for the claim property and `passed`
    is True only when it's "proven". No reachability/vacuity check and no
    tautology guard (see module docstring for the accepted trade-off)."""
    tb_text, module_name, claim_names = build_abstract_module_with_scenario_checks(
        signal_list, scenarios, sva_ol_nl_body, clock_signal=clock_signal
    )

    sv_dir = os.path.abspath(sv_dir)
    os.makedirs(sv_dir, exist_ok=True)
    sva_path = os.path.join(sv_dir, f"{experiment_id}_{task_id}.sva")
    with open(sva_path, "w") as file:
        file.write(tb_text)

    status_puts = "\n".join(f'puts "JGCEXSTATUS {n} [get_status {module_name}.{n}]"' for n in claim_names)
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

    tmp_jg_proj_dir = os.path.join(sv_dir, "jg_cex", f"{experiment_id}_{task_id}")

    jg_command = ["jg", "-fpv", "-batch", "-tcl", tcl_path, "-proj", tmp_jg_proj_dir, "-allow_unsupported_OS"]
    try:
        result = subprocess.run(jg_command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        output = f"TIMEOUT: JasperGold execution exceeded {timeout}s"
        statuses = {}
    else:
        output = result.stdout.strip()
        statuses = {}
        for line in output.splitlines():
            m = _STATUS_LINE_RE.match(line.strip())
            if m:
                statuses[m.group(1)] = m.group(2)

    results = []
    for sc, claim_name in zip(scenarios, claim_names):
        claim_status = statuses.get(claim_name, "unknown")
        results.append({"id": sc["id"], "result": claim_status, "passed": claim_status == "proven"})
    return results, output
