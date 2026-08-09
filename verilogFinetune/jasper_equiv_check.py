"""
Formal property-equivalence checking via Cadence JasperGold, reusing the
existing FVEval harness's own tcl scripts and IP-protected equivalence-check
macro (Evaluation/FVRuleLearner/FVEval/tool_scripts/run_jg_nl2sva_human.tcl +
tool_scripts/pec/pec.tcle's `prop_eq_checker`) -- the same mechanism
Evaluation/FVRuleLearner/FVEval/fv_eval/fv_tool_execution.py's
launch_jg_custom_equiv_check and evaluation.py's NL2SVAHumanEvaluator already
use for final scoring.

This does NOT import fv_tool_execution.py itself: that module does `from
config import FLAGS` / `from saver import saver` at import time, pulling in
the NVIDIA harness's own CLI-oriented global config and logging state
(Evaluation/FVRuleLearner/src/config.py hardcodes e.g. `global_task =
'train'`), which isn't meant to be initialized from an unrelated standalone
script, and FVEval/ has no __init__.py so it's only importable as a namespace
package in the first place. Instead this reimplements just the one `jg`
subprocess invocation + stdout parsing, calling the exact same tcl script and
pec.tcle macro, so results stay consistent with whatever Stage 6 evaluation
will eventually run.

Requires `jg` (JasperGold) on PATH and a valid license. Not available in the
sandbox this was developed in -- built directly from reading
fv_tool_execution.launch_jg_custom_equiv_check, run_jg_nl2sva_human.tcl, and
NL2SVAHumanEvaluator.calculate_jg_metric, but never executed end-to-end.
Confirm against a real `jg` install before relying on it.
"""
import os
import re
import shutil
import subprocess

_FVRULE_LEARNER_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "Evaluation", "FVRuleLearner")
)
DEFAULT_TCL_FILE_PATH = os.path.join(_FVRULE_LEARNER_ROOT, "FVEval", "tool_scripts", "run_jg_nl2sva_human.tcl")


_RAW_ASSERTION_PREFIX_RE = re.compile(r'^\s*(\w+\s*:\s*)?assert\s*(property\s*)?\(')


def strip_assertion_body(assertion_text):
    """Mirrors NL2SVAHumanEvaluator's own extraction: run_jg_nl2sva_human.tcl's
    prop_eq_checker wants just the property body, not the
    `assert property (... disable iff (tb_reset)` wrapper or the trailing
    `);` -- matching exactly how Stage 6 scoring will strip both the
    LM-generated and reference assertion text before comparing them.

    No-ops on text that's already a bare property body (callers like
    score_nl2sva_human.py/run_rag_on_fveval_benchmarks.py pre-extract via
    extract_property_body's more robust, order-independent stripping before
    calling run_equivalence_check at all). Without this guard, the naive
    "tb_reset)" search below matches ANY occurrence of that substring, not
    just a disable iff clause -- confirmed corrupting a real response whose
    property body legitimately referenced tb_reset as a boolean operand
    (`!(jump_vld_d1 || tb_reset) |-> ...`), silently truncating it down to
    ` |-> ...` and surfacing as a JasperGold syntax error, not the intended
    functional-equivalence result."""
    text = assertion_text.strip().replace("\n", "")
    if not _RAW_ASSERTION_PREFIX_RE.match(text):
        return text
    if "tb_reset)" in text:
        text = text.split("tb_reset)")[-1].strip()
    # re.split, not text.split(");"): misses a ") ;" (space before the
    # semicolon), leaving a dangling unbalanced close -- same fix as
    # score_nl2sva_human.extract_property_body.
    return re.split(r'\)\s*;', text)[0].strip()


def is_equivalent(jg_output):
    """Same acceptance rule as NL2SVAHumanEvaluator.calculate_jg_metric, but
    stricter: only a "Full equivalence" result counts, not a one-directional
    "implies" match. Stage 2 wants confirmation that the OL-NL statement's
    candidate SVA captures the *exact* semantics of the golden assertion, not
    merely a relaxation of it."""
    if re.search(r"syntax error", jg_output):
        return False
    return bool(re.search(r"Full equivalence", jg_output))


def run_equivalence_check(
    testbench,
    lm_assertion,
    ref_assertion,
    signal_list,
    sv_dir,
    experiment_id="codev_sva_ol_nl_validation",
    task_id="0",
    tcl_file_path=DEFAULT_TCL_FILE_PATH,
    timeout=60,
):
    """Writes `testbench` (an elaborable module with NO assertion inside --
    the two assertions are supplied separately, matching
    run_jg_nl2sva_human.tcl's `analyze -sv12 ${SV_DIR}/${EXP_ID}_${TASK_ID}.sva`
    + `prop_eq_checker $LM_ASSERT_TEXT $REF_ASSERT_TEXT` contract) to
    {sv_dir}/{experiment_id}_{task_id}.sva, then invokes JasperGold's custom
    property-equivalence checker against it.

    Returns (equivalent: bool, raw_jg_output: str).
    """
    # jg itself runs with cwd=fveval_dir (below), and the tcl script resolves
    # ${SV_DIR} relative to THAT cwd, not this process's -- a relative sv_dir
    # would make `analyze -sv12 ${SV_DIR}/...` look in the wrong place and
    # fail with "file does not exist" before any real analysis runs, which
    # reads indistinguishably from a genuine "not equivalent" result.
    sv_dir = os.path.abspath(sv_dir)
    os.makedirs(sv_dir, exist_ok=True)
    sva_path = os.path.join(sv_dir, f"{experiment_id}_{task_id}.sva")
    with open(sva_path, "w") as file:
        file.write(testbench)

    tmp_jg_proj_dir = os.path.join(sv_dir, "jg", f"{experiment_id}_{task_id}")
    if os.path.isdir(tmp_jg_proj_dir):
        shutil.rmtree(tmp_jg_proj_dir)

    jg_command = [
        "jg", "-fpv", "-batch", "-tcl", tcl_file_path,
        "-define", "LM_ASSERT_TEXT", strip_assertion_body(lm_assertion),
        "-define", "REF_ASSERT_TEXT", strip_assertion_body(ref_assertion),
        # dict.fromkeys(...): dedupe while preserving order -- a duplicate
        # entry makes prop_eq_checker's generated wrapper module reject the
        # whole check ("ANSI port 'X' cannot be redeclared"), an
        # elaboration error indistinguishable from a genuine functional
        # mismatch to callers just pattern-matching the output. Centralized
        # here (not just at each caller) since every caller ultimately
        # funnels through this one SIGNAL_LIST join.
        "-define", "SIGNAL_LIST", ",".join(dict.fromkeys(signal_list)),
        "-define", "EXP_ID", experiment_id,
        "-define", "TASK_ID", task_id,
        "-define", "SV_DIR", sv_dir,
        "-proj", tmp_jg_proj_dir,
        "-allow_unsupported_OS",
    ]
    fveval_dir = os.path.abspath(os.path.join(os.path.dirname(tcl_file_path), ".."))

    try:
        result = subprocess.run(jg_command, cwd=fveval_dir, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired:
        return False, f"TIMEOUT: JasperGold execution exceeded {timeout}s"

    output = result.stdout.strip()
    return is_equivalent(output), output
