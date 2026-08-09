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

For each scenario (a concrete, possibly multi-cycle signal-value trace,
plus whether the property should hold or be violated at the final cycle),
builds two named formal properties against the real testbench:
  1. A `cover` of the scenario's own sequence alone -- confirms the
     scenario is actually REACHABLE given the testbench's real signal
     widths/parameters. This is NOT optional: confirmed empirically
     (2026-08-08, counter_0/width=1) that a scenario referencing an
     out-of-range value (count==2 against a 1-bit count signal) makes the
     scenario's antecedent unsatisfiable, so ANY implication built on top
     of it -- including one testing a deliberately-wrong sva_ol_nl -- is
     vacuously "proven" regardless of whether sva_ol_nl is actually
     correct. Skipping this check would silently validate wrong SVAs.
  2. The actual claim: `scenario_seq |-> sva_ol_nl_body` (a "should hold"
     scenario) or `scenario_seq |-> not (sva_ol_nl_body)` (a "should
     violate" scenario -- SVA's `not` legally negates a whole property,
     confirmed working via a live smoke test against a deliberately wrong,
     always-true SVA once the scenario was made reachable).

All scenarios for one row are embedded together and checked in a single
JasperGold invocation -- elaboration/licensing overhead is per-process,
not per-property, and jg startup dominates runtime for small properties
like these.

Mechanism (clock/reset declaration, `prove -all`, `get_status <module>.
<name>`) was empirically validated live against real `jg` before being
wired into any pipeline code -- see .tmp/jg_counterexample_check_dev/ for
the raw exploratory session. Confirmed necessary: `clock <clk>` +
`reset -none` before `prove -all` (pec.tcle's equivalence path handles
this internally; this bypasses pec.tcle entirely, so it must be done
explicitly here or JasperGold errors out with ECK040 "no clocks declared").
"""
import os
import re
import subprocess

_MODULE_NAME_RE = re.compile(r'\bmodule\s+(\w+)')


def build_scenario_sequence(cycle_conditions):
    """cycle_conditions: ordered list of boolean-expression strings, one
    per cycle (oldest first), concatenated with ##1 delays. A single-cycle
    scenario is just cycle_conditions == [one string] -- no ##1 needed."""
    return " ##1 ".join(f"({c})" for c in cycle_conditions)


def build_testbench_with_scenario_checks(raw_testbench, scenarios, sva_ol_nl_body, clock_signal="clk"):
    """scenarios: list of dicts {"id": str, "cycle_conditions": [str, ...],
    "expected": "hold" | "violate"}. id must be a valid, unique SV
    identifier fragment (used directly in property names).

    Returns (testbench_text, module_name, cover_names, claim_names).
    """
    module_match = _MODULE_NAME_RE.search(raw_testbench)
    if not module_match:
        raise ValueError("Could not find a `module <name>` declaration in the testbench")
    module_name = module_match.group(1)

    lines = []
    cover_names = []
    claim_names = []
    for sc in scenarios:
        seq = build_scenario_sequence(sc["cycle_conditions"])
        cover_name = f"cov_{sc['id']}"
        claim_name = f"claim_{sc['id']}"
        cover_names.append(cover_name)
        claim_names.append(claim_name)
        lines.append(f"{cover_name}: cover property (@(posedge {clock_signal}) ({seq}));")
        if sc["expected"] == "hold":
            claim = f"({seq}) |-> ({sva_ol_nl_body})"
        else:
            claim = f"({seq}) |-> not ({sva_ol_nl_body})"
        lines.append(f"{claim_name}: assert property (@(posedge {clock_signal}) {claim});")
    block = "\n".join(lines) + "\n"

    if "endmodule" in raw_testbench:
        tb_text = raw_testbench.replace("endmodule", block + "endmodule", 1)
    else:
        tb_text = raw_testbench + "\n" + block
    return tb_text, module_name, cover_names, claim_names


_STATUS_LINE_RE = re.compile(r'^JGCEXSTATUS (\S+) (\S+)\s*$')


def run_counterexample_checks(
    testbench,
    scenarios,
    sva_ol_nl_body,
    sv_dir,
    experiment_id="ol_nl_counterexample_validation",
    task_id="0",
    clock_signal="clk",
    timeout=180,
):
    """Runs all of `scenarios` against `sva_ol_nl_body` in one JasperGold
    invocation. Returns a list of dicts, one per scenario, in the same
    order as `scenarios`:
      {"id", "expected", "reachable": bool, "result": str, "passed": bool}
    where `result` is JasperGold's raw get_status string for the claim
    property (e.g. "proven", "cex", "undetermined") and `passed` is True
    only when the scenario was reachable AND the claim was proven (the
    claim text already encodes the expected hold/violate direction via
    its plain/negated form, so "proven" always means "correct" here).

    A scenario with reachable=False must be treated as inconclusive, not
    as a pass -- see this module's docstring for why (vacuity)."""
    tb_text, module_name, cover_names, claim_names = build_testbench_with_scenario_checks(
        testbench, scenarios, sva_ol_nl_body, clock_signal
    )

    sv_dir = os.path.abspath(sv_dir)
    os.makedirs(sv_dir, exist_ok=True)
    sva_path = os.path.join(sv_dir, f"{experiment_id}_{task_id}.sva")
    with open(sva_path, "w") as file:
        file.write(tb_text)

    all_names = [n for pair in zip(cover_names, claim_names) for n in pair]
    status_puts = "\n".join(f'puts "JGCEXSTATUS {n} [get_status {module_name}.{n}]"' for n in all_names)
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
    for sc, cover_name, claim_name in zip(scenarios, cover_names, claim_names):
        cover_status = statuses.get(cover_name, "unknown")
        claim_status = statuses.get(claim_name, "unknown")
        reachable = cover_status == "covered"
        passed = reachable and claim_status == "proven"
        results.append({
            "id": sc["id"],
            "expected": sc["expected"],
            "reachable": reachable,
            "result": claim_status,
            "passed": passed,
        })
    return results, output
