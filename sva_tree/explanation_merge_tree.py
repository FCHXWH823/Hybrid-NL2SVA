"""
Builds a bottom-up, code-grounded natural-language explanation of an SVA's
operator/signal tree (sva_graph.py) -- the inverse of
verilogFinetune/generate_dfs_explanation.py's walk_tree(), which needs a
known-correct golden explanation to split top-down. Here there is no golden
explanation to split: the LLM-generated SVA under review might itself be
wrong, so each node's meaning is instead *composed* bottom-up from its own
code and its already-derived operand meanings, with no assumption that the
result matches anyone's intent. The root's composed meaning is exactly what a
rechecking step should compare against the original natural-language
property: a mismatch there points at the specific operator node responsible,
instead of requiring the checking LLM to holistically re-read raw SVA syntax
against a generic operator glossary.

Two node types coming out of sva_graph.py:
    operator - gets one LLM call, given its own code/operator and its
               already-composed operand meanings (never the rest of the
               tree) -- UNLESS use_templates=True and the operator is one
               of the fixed-template operators below, in which case no LLM
               call happens at all (see _TEMPLATE_OPERATORS).
    signal   - no LLM call; its natural-language piece is its own code
               verbatim (an identifier or literal has no meaning to compose).

Walk order is the same reversed-DFS order sva_graph.render_merge_tree()
already uses: every node's children (and their full subtrees) are visited,
and thus have entries in pieces_by_id, before the node itself.
"""
import json
import os
import re
import time

from sva_graph import build_operator_signal_graph, dfs_nodes

# use_templates=True (--sor-template-timing): fixed, LLM-free natural-
# language templates for the operators documented in
# sva_temporal_operators.json, opt-in via
# build_and_render_explanation_merge_tree's use_templates parameter.
#
# Motivation: confirmed live (2026-08-14, FVEval-NL2SVA-Machine-11/-18/-120,
# full-283 nl2sva_machine_verified trials) that the LLM-composed nl_piece
# for a `|=>` node combining a `##N` consequent silently drops `|=>`'s own
# implicit +1-cycle advance when paraphrasing into one fluent sentence --
# e.g. "sig_G != ... |=> ##4 sig_J" got composed as "...then, 4 clock ticks
# later, sig_J must..." (the correct total is 5 cycles from the antecedent,
# not 4), and SOR's rechecking step then compares that already-wrong
# derived meaning against the question's "four cycles later" phrasing and
# finds no discrepancy, since both read as "4 clock ticks later." An
# earlier attempt to fix this by asking the merge-node LLM call to more
# carefully reason about operator semantics (see git history) caused a
# separate, REPLICATED regression across the whole nl2sva_machine_verified
# benchmark (77.0% vs 82.0% FM-strict, confirmed on two independent full
# trials) and was reverted.
#
# This is a structurally different fix: an operator's own effect on its
# operands is deterministic, not a matter of LLM judgment, so it's baked
# directly into a fixed template string (sva_temporal_operators.json's own
# "template_unary"/"template_binary" fields, written from that operator's
# already-documented natural_langage_explanation) instead of being composed
# by an LLM call that could paraphrase it away. Nesting composes correctly
# by construction, since every level's own contribution stays permanently,
# literally present in the final text -- there is no free-form generation
# step where it could be silently dropped.
#
# Deliberately data-driven and general, not special-cased per operator: no
# operator name appears anywhere in this module's code (only in the JSON).
# An earlier version specifically detected and numerically summed chains of
# consecutive `##N` nodes (e.g. "##1 ##1 X" -> "2 clock cycles") -- correct,
# but a bespoke arithmetic rule that applied to exactly one operator family
# and doesn't generalize. This version applies each node's own template
# independently and lets them nest, same as every other operator here (so a
# `##1 ##1 X` chain again renders as two separately-nested "occurring
# exactly 1 clock cycle(s)..." clauses rather than one precomputed "2" --
# an accepted, deliberate tradeoff for a uniform mechanism).
#
# Any operator NOT covered by the JSON (booleans, reductions, comparisons,
# arbitrary system functions, ...) falls through to the existing
# call_merge_node_llm path completely unchanged, so this can't regress
# anything outside the SVA-temporal-operator vocabulary it targets.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_DEFAULT_OPERATOR_TABLE_PATH = os.path.join(_THIS_DIR, "..", "sva_temporal_operators.json")

# Maps a real node label (e.g. "##2", "always[3:7]") to its
# sva_temporal_operators.json key (e.g. "##N", "always[M:N]") plus any
# embedded numeric parameter(s) -- covers every parameterized operator in
# the table. Every other entry's JSON key already equals its real-world
# label verbatim (e.g. `|->`, `strong`, `$rose`), so no pattern is needed
# for those; see _match_operator_key.
_PARAMETERIZED_LABEL_PATTERNS = [
    (re.compile(r'^##(\d+)$'), "##N", ("n",)),
    (re.compile(r'^##\[(\d+):(\$|\d+)\]$'), "##[M:N]", ("m", "n")),
    (re.compile(r'^\[\*(\d+)\]$'), "[*N]", ("n",)),
    (re.compile(r'^\[\*(\d+):(\$|\d+)\]$'), "[*M:N]", ("m", "n")),
    (re.compile(r'^\[->(\d+):(\$|\d+)\]$'), "[->M:N]", ("m", "n")),
    (re.compile(r'^\[=(\d+):(\$|\d+)\]$'), "[=M:N]", ("m", "n")),
    (re.compile(r'^nexttime\[(\d+)\]$'), "nexttime[N]", ("n",)),
    (re.compile(r'^s_nexttime\[(\d+)\]$'), "s_nexttime[N]", ("n",)),
    (re.compile(r'^always\[(\d+):(\$|\d+)\]$'), "always[M:N]", ("m", "n")),
    (re.compile(r'^s_always\[(\d+):(\$|\d+)\]$'), "s_always[M:N]", ("m", "n")),
    (re.compile(r'^eventually\[(\d+):(\$|\d+)\]$'), "eventually[M:N]", ("m", "n")),
    (re.compile(r'^s_eventually\[(\d+):(\$|\d+)\]$'), "s_eventually[M:N]", ("m", "n")),
]


def load_operator_templates(path=_DEFAULT_OPERATOR_TABLE_PATH):
    """Loads sva_temporal_operators.json's per-operator "template_unary"/
    "template_binary" fields into a lookup dict: {operator_key:
    {"template_unary": ..., "template_binary": ...}} (only the keys that
    entry actually has). Operator keys use the JSON's own placeholder
    spelling (e.g. "##N", "always[M:N]") -- see _match_operator_key for how
    a real node label like "##2" or "always[3:7]" maps back to one."""
    with open(path) as file:
        data = json.load(file)
    return {
        op: {k: v for k, v in entry.items() if k in ("template_unary", "template_binary")}
        for op, entry in data.items()
    }


def _match_operator_key(label, operator_templates):
    """Returns (json_key, params) for a real node label, or (None, {}) if
    it matches no templated operator. Tries an exact/literal match first
    (the majority of operators), then the parameterized patterns above."""
    if label in operator_templates:
        return label, {}
    for pattern, json_key, param_names in _PARAMETERIZED_LABEL_PATTERNS:
        match = pattern.match(label)
        if match:
            return json_key, dict(zip(param_names, match.groups()))
    return None, {}

# PROMPT_TEMPLATE_MERGE_NODE's own (b) instruction ("a single, self-contained
# sentence stating exactly what "{code}" means") reliably makes the LLM echo
# that framing back almost verbatim -- 'The expression "X" means that Y' /
# 'The expression "X" is true if Y' -- for every LLM-composed node. Harmless
# on its own, but confirmed live (2026-08-14) that chaining several such
# pieces together inside a |->/|=> template produces a needlessly verbose,
# clunky sentence (three stacked "The expression '...' means/is true if..."
# clauses in a row). Only stripped here, when embedding an operand into a
# template -- the underlying LLM prompt/output is untouched, so a
# non-templated node's own displayed nl_piece looks exactly as it always
# has.
_EXPRESSION_BOILERPLATE_RE = re.compile(
    r'^(?:The\s+expression\s+)?"[^"]*"\s+'
    r'(?:means(?:\s+that)?|is\s+true\s+(?:if|when)|evaluates?\s+to\s+true\s+'
    r'(?:if(?:\s+and\s+only\s+if)?|when))\s+',
    re.IGNORECASE,
)


def _normalize_operand_piece(text):
    """Strips the recurring 'The expression "X" means/is true if' prefix
    (see _EXPRESSION_BOILERPLATE_RE) and one trailing period, so an
    LLM-composed operand piece slots grammatically into a template instead
    of producing a run-on with a stray "., " in the middle -- confirmed
    live (2026-08-14, FVEval-NL2SVA-Machine-120) that raw insertion made
    the rendered tree borderline unreadable. Harmless when applied to a
    template-composed piece too (never matches, no-op)."""
    text = text.strip()
    text = _EXPRESSION_BOILERPLATE_RE.sub("", text)
    return text[:-1] if text.endswith(".") else text


def render_template_nl_piece(node, pieces_by_id, operator_templates):
    """Deterministic, LLM-free nl_piece for a node whose operator has a
    template in sva_temporal_operators.json (see load_operator_templates).
    Returns None when the operator has no template there, or the node's
    actual operand count doesn't match a template this operator has (e.g.
    a `$past` call with 3 real arguments), so the caller falls back to
    call_merge_node_llm unchanged. No operator-specific branch lives here
    at all -- the JSON is the sole source of both which operators are
    covered and how their operands combine."""
    json_key, params = _match_operator_key(node["label"], operator_templates)
    if json_key is None:
        return None
    templates = operator_templates[json_key]
    operand_pieces = [_normalize_operand_piece(pieces_by_id[c["id"]]["nl_piece"]) for c in node["children"]]

    if len(operand_pieces) == 1 and "template_unary" in templates:
        return templates["template_unary"].format(op1=operand_pieces[0], **params)
    if len(operand_pieces) == 2 and "template_binary" in templates:
        return templates["template_binary"].format(op1=operand_pieces[0], op2=operand_pieces[1], **params)
    return None

NODE_SYSTEM_PROMPT = (
    "You are a helpful bot that explains one node of an already-parsed SystemVerilog "
    "assertion syntax tree, by composing the meaning of its operand(s), following the "
    "requested format exactly."
)

PROMPT_TEMPLATE_MERGE_NODE = """You are explaining ONE node of a SystemVerilog assertion's already-parsed \
syntax tree, working bottom-up. You are not shown the rest of the tree, and you must \
not assume anything about the assertion's intended purpose or correctness -- only \
this node's own operator and its operand(s)' already-derived meanings, given below, \
are relevant. Do not judge whether the code is correct; just state what it means.

This node's code:
 {code}

This node's operator: {operator}

{operand_lines}

Output exactly the following labeled lines in plain text (not JSON):

(a) Operator reasoning: <why the operator "{operator}" combines the operand meaning(s) above into this node's meaning>
(b) Natural language piece: <a single, self-contained sentence stating exactly what "{code}" means, built by combining the operand meanings via the operator's semantics>
"""

_FIELD_PATTERN = re.compile(r"\(([ab])\)[^:\n]*:\s*(.*?)(?=\n\s*\([ab]\)|\Z)", re.DOTALL)


def parse_node_response(text):
    """Extract {"a": reasoning, "b": nl_piece} from the model's labeled
    plain-text reply."""
    return {letter: content.strip() for letter, content in _FIELD_PATTERN.findall(text)}


def _code_of(node):
    return node["code"] if node["type"] == "operator" else node["label"]


def build_merge_node_prompt(node, pieces_by_id):
    children = node["children"]
    operand_lines = "\n".join(
        f"Operand {i + 1} code: {_code_of(child)}\n"
        f"Operand {i + 1} meaning: {pieces_by_id[child['id']]['nl_piece']}"
        for i, child in enumerate(children)
    )
    return PROMPT_TEMPLATE_MERGE_NODE.format(
        code=node["code"],
        operator=node["label"],
        operand_lines=operand_lines,
    )


def call_merge_node_llm(client, model, node, pieces_by_id, operator_context, max_retries):
    """Calls the LLM to compose one operator node's meaning from its
    operands' already-derived meanings. Retries on API failure or a reply
    missing a required field; after max_retries, degrades to a naive
    function-call-style join of the operand meanings rather than losing the
    subtree.

    operator_context goes in the SYSTEM message (stable across every node
    call for a given tree), not repeated in each per-node user prompt."""
    prompt = build_merge_node_prompt(node, pieces_by_id)
    system_msg = NODE_SYSTEM_PROMPT + "\n\nSVA Operator Context:\n" + operator_context
    last_error = None
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system_msg},
                    {"role": "user", "content": prompt},
                ],
            )
            parsed = parse_node_response(completion.choices[0].message.content)
            if {"a", "b"}.issubset(parsed):
                return parsed
            print("    merge-node reply missing field(s), retrying...")
        except Exception as error:
            last_error = error
            print(f"    merge-node call failed ({error}), retrying...")
        time.sleep(2 ** attempt)

    print(f"    giving up on this merge node after {max_retries} attempts; "
          f"degrading to a verbatim join (last_error={last_error})")
    operand_meanings = [pieces_by_id[c["id"]]["nl_piece"] for c in node["children"]]
    fallback_piece = f"{node['label']}(" + ", ".join(operand_meanings) + ")"
    return {"a": "(reasoning unavailable -- extraction failed after retries)", "b": fallback_piece}


def build_explanation_merge_tree(client, model, root, operator_context, max_retries=5, use_templates=False):
    """Walks root's tree bottom-up (children before parents), returning
    pieces_by_id: {node_id: {"sva_piece", "nl_piece", "merge_operator"}} for
    every node in the tree, operator and signal alike. A degenerate root
    (bare signal, no operator) is handled the same way as any other leaf --
    no special-casing needed, unlike the top-down walk_tree().

    use_templates: when True, any node whose operator is covered by
    sva_temporal_operators.json's template fields gets a deterministic,
    LLM-free nl_piece (see render_template_nl_piece) instead of an LLM
    call -- every other operator is unaffected."""
    operator_templates = load_operator_templates() if use_templates else None
    order = list(reversed(dfs_nodes(root)))
    pieces_by_id = {}
    for node in order:
        if node["type"] == "signal":
            pieces_by_id[node["id"]] = {
                "sva_piece": node["label"],
                "nl_piece": node["label"],
                "merge_operator": None,
            }
            continue
        if use_templates:
            templated = render_template_nl_piece(node, pieces_by_id, operator_templates)
            if templated is not None:
                pieces_by_id[node["id"]] = {
                    "sva_piece": node["code"],
                    "nl_piece": templated,
                    "merge_operator": node["label"],
                }
                continue
        parsed = call_merge_node_llm(client, model, node, pieces_by_id, operator_context, max_retries)
        pieces_by_id[node["id"]] = {
            "sva_piece": node["code"],
            "nl_piece": parsed["b"],
            "merge_operator": node["label"],
        }
    return pieces_by_id


def render_explanation_merge_tree(root, pieces_by_id):
    """Renders the T1, T2, ... symbolic derivation (same labeling scheme and
    walk order as sva_graph.render_merge_tree()), but each block only ever
    shows the 3 fields a rechecking prompt needs: the exact SVA piece, its
    composed natural-language meaning, and the operator that merged its
    operands into it. The LLM call's internal-only reasoning field is never
    surfaced here."""
    order = list(reversed(dfs_nodes(root)))
    label_of = {}
    lines = []
    for i, node in enumerate(order, start=1):
        label = f"T{i}"
        label_of[node["id"]] = label
        piece = pieces_by_id[node["id"]]

        if node["type"] == "signal":
            lines.append(f"{label} = {piece['sva_piece']}")
        else:
            operand_labels = [label_of[c["id"]] for c in node["children"]]
            if len(operand_labels) == 2:
                expr = f"{operand_labels[0]} {node['label']} {operand_labels[1]}"
            else:
                expr = f"{node['label']}({', '.join(operand_labels)})"
            lines.append(f"{label} = {expr}")

        merge_operator = piece["merge_operator"] if piece["merge_operator"] is not None else "null"
        lines.append(f"    sva piece: {piece['sva_piece']}")
        lines.append(f"    merge operator: {merge_operator}")
        lines.append(f"    nl piece: {piece['nl_piece']}")
        lines.append("")

    root_piece = pieces_by_id[root["id"]]
    lines.append(f"Final derived meaning of the generated assertion ({label_of[root['id']]}): {root_piece['nl_piece']}")
    return "\n".join(lines)


def build_and_render_explanation_merge_tree(client, model, sva_text, operator_context, max_retries=5, use_templates=False):
    """Convenience entrypoint for callers (e.g. the rechecking flow) that
    just want the rendered tree text for a piece of SVA source.

    Raises ValueError when sva_graph.py can't parse sva_text (~15% of the
    corpus; see verilogFinetune/check_parser_coverage.py) so the caller can
    fall back to a non-tree-based prompt for that case, the same way
    generate_dfs_explanation.py falls back for its own top-down tree.

    use_templates: see build_explanation_merge_tree.
    """
    root = build_operator_signal_graph(sva_text)
    pieces_by_id = build_explanation_merge_tree(client, model, root, operator_context, max_retries, use_templates)
    return render_explanation_merge_tree(root, pieces_by_id)
