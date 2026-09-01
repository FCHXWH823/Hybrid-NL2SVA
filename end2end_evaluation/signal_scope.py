"""Static scope analysis for UART's 6-module RTL hierarchy, used to fix a
real integration bug: run_uart_nl2sva.py/score_uart_assertionforge.py bind
every generated SVA into uart2bus_top's own module body (see check_sva_
elaboration/check_sva_proven -- they splice the assertion in right before
the first `endmodule`, and uart2bus_top.v is listed first in RTL_FILE_ORDER
specifically so that's uart2bus_top's own body), but AssertionForge's NL
plans sometimes reference a real RTL signal that's declared *inside* a
submodule (uart_top/baud_gen/uart_rx/uart_tx/uart_parser), not visible as a
bare identifier from uart2bus_top's own scope -- confirmed empirically:
`ce_16` alone caused 34/98 of the original run's syntax failures. See
end2end_evaluation/README.md's "A second gap" section for the full
diagnosis.

qualify_out_of_scope_references() mechanically rewrites bare references to
these real-but-out-of-scope identifiers into correctly-qualified
hierarchical paths (e.g. `ce_16` -> `uart1.ce_16`) or macro invocations
(e.g. `MAIN_ADDR` -> `` `MAIN_ADDR ``), so a generated SVA that legitimately
needs one of these can still elaborate against the real design. This is
Approach A from the two options discussed for fixing this (vs. Approach B,
a full `bind`-at-the-correct-module refactor) -- deliberately the lower-
risk, more surgical one: it only touches identifier text, not how/where
the assertion gets bound, and it's orthogonal to (does not touch, does not
require adopting) the separately-discussed and NOT-adopted
skip_signal_list_note=False fix -- this is pure mechanical scope repair on
identifiers that are already real, valid RTL signals; it does nothing for
the ~30% of failures that are pure hallucinations with no RTL antecedent
at all.

Deliberately conservative about ambiguity: `ce_1`/`count16`/`bit_count`/
`data_buf` are each declared -- same name, but a genuinely DIFFERENT net --
in both uart_rx.v and uart_tx.v. Guessing which one a bare reference meant
would risk silently binding to the WRONG net (elaborates fine, but checks
the wrong signal) rather than failing loudly as before, which is worse.
These are left unqualified on purpose; see AMBIGUOUS_INTERNAL_NAMES.
"""
import os
import re
from collections import Counter

# Hierarchical path prefix from uart2bus_top's own scope to reach each
# file's own module instance. Empty string for uart2bus_top.v itself (no
# qualification needed -- these are already bare-reachable, whether as
# uart2bus_top's own ports or as internal wires it declares itself to
# connect to uart1/uart_parser1's identically-named ports). Matches
# run_uart_nl2sva.py's RTL_FILE_ORDER; instance names (uart1, baud_gen_1,
# uart_rx_1, uart_tx_1, uart_parser1) are real, read directly from the RTL
# (uart2bus_top.v's/uart_top.v's own instantiation statements) -- not
# something a generic parser can derive without real elaboration, so this
# one part IS hand-specified, same as RTL_FILE_ORDER itself.
SCOPE_PREFIX_BY_FILE = {
    "uart2bus_top.v": "",
    "uart_top.v": "uart1.",
    "baud_gen.v": "uart1.baud_gen_1.",
    "uart_rx.v": "uart1.uart_rx_1.",
    "uart_tx.v": "uart1.uart_tx_1.",
    "uart_parser.v": "uart_parser1.",
}

_PORT_DECL_RE = re.compile(r'\b(?:input|output|inout)\s+(?:reg|wire)?\s*(?:\[[^\]]+\])?\s*(\w+)\s*;')
_INTERNAL_DECL_RE = re.compile(
    r'^\s*(?:wire|reg|integer)\s+(?:signed\s+)?(?:\[[^\]]+\])?\s*([\w, ]+)\s*;', re.MULTILINE
)
_DEFINE_RE = re.compile(r'^`define\s+(\w+)', re.MULTILINE)


def build_signal_scope_map(uart_rtl_dir):
    """Parses the 6 real RTL files fresh each call (cheap -- 6 small files)
    rather than hand-listing signal names, so this stays correct if the
    vendored AssertLLM RTL ever changes. Returns (scope_map, macro_names):

    scope_map: {bare_identifier: hierarchical_prefix} for every internal
    (non-port) wire/reg/integer that's declared in exactly ONE of the 6
    files and is NOT already bare-reachable from uart2bus_top's own scope.

    macro_names: the set of all `define names across the 6 files (Verilog
    macros are compilation-unit-global, not module-scoped, so no ambiguity
    concern the way internal signals have -- confirmed no name collides
    across files here)."""
    internals_by_file = {}
    macro_names = set()

    for fname, prefix in SCOPE_PREFIX_BY_FILE.items():
        with open(os.path.join(uart_rtl_dir, fname)) as f:
            text = f.read()
        ports = set(_PORT_DECL_RE.findall(text))
        internal = set()
        for m in _INTERNAL_DECL_RE.finditer(text):
            internal.update(n.strip() for n in m.group(1).split(","))
        internal -= ports  # old-style modules re-declare ports with wire/reg -- those aren't "internal"
        internals_by_file[fname] = internal
        macro_names.update(_DEFINE_RE.findall(text))

    counts = Counter()
    for names in internals_by_file.values():
        for n in names:
            counts[n] += 1

    scope_map = {}
    for fname, names in internals_by_file.items():
        prefix = SCOPE_PREFIX_BY_FILE[fname]
        if not prefix:
            continue  # uart2bus_top.v itself -- already bare-reachable, nothing to qualify
        for n in names:
            if counts[n] == 1:  # unambiguous: an internal declaration in exactly one module
                scope_map[n] = prefix
    return scope_map, macro_names


AMBIGUOUS_INTERNAL_NAMES = frozenset({"ce_1", "count16", "bit_count", "data_buf"})

_IDENT_RE = re.compile(r"\b\w+\b")


def qualify_out_of_scope_references(sva_text, scope_map, macro_names):
    """Rewrites bare identifiers in sva_text: a known internal signal name
    (unambiguous, per scope_map) gets its hierarchical prefix prepended; a
    known macro name gets a backtick prepended (bare `MAIN_ADDR` is a plain-
    identifier reference Verilog resolves against declared signals, not a
    macro invocation -- confirmed live, this is exactly why the model's own
    bare `MAIN_ADDR`/`TX_LO_NIB`/`CHAR_CR`/`CHAR_LF` references failed
    elaboration despite naming real, existing macros). Skips anything
    already immediately preceded by `.` or `` ` `` (already qualified/
    invoked -- including the model's own, possibly-wrong, hierarchical
    guesses; this only adds qualification, never corrects a wrong one)."""
    def repl(match):
        word = match.group(0)
        preceding = sva_text[max(0, match.start() - 1):match.start()]
        if preceding in (".", "`"):
            return word
        if word in macro_names:
            return "`" + word
        if word in scope_map:
            return scope_map[word] + word
        return word

    return _IDENT_RE.sub(repl, sva_text)
