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
               tree).
    signal   - no LLM call; its natural-language piece is its own code
               verbatim (an identifier or literal has no meaning to compose).

Walk order is the same reversed-DFS order sva_graph.render_merge_tree()
already uses: every node's children (and their full subtrees) are visited,
and thus have entries in pieces_by_id, before the node itself.
"""
import re
import time

from sva_graph import build_operator_signal_graph, dfs_nodes

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


def build_explanation_merge_tree(client, model, root, operator_context, max_retries=5):
    """Walks root's tree bottom-up (children before parents), returning
    pieces_by_id: {node_id: {"sva_piece", "nl_piece", "merge_operator"}} for
    every node in the tree, operator and signal alike. A degenerate root
    (bare signal, no operator) is handled the same way as any other leaf --
    no special-casing needed, unlike the top-down walk_tree()."""
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


def build_and_render_explanation_merge_tree(client, model, sva_text, operator_context, max_retries=5):
    """Convenience entrypoint for callers (e.g. the rechecking flow) that
    just want the rendered tree text for a piece of SVA source.

    Raises ValueError when sva_graph.py can't parse sva_text (~15% of the
    corpus; see verilogFinetune/check_parser_coverage.py) so the caller can
    fall back to a non-tree-based prompt for that case, the same way
    generate_dfs_explanation.py falls back for its own top-down tree.
    """
    root = build_operator_signal_graph(sva_text)
    pieces_by_id = build_explanation_merge_tree(client, model, root, operator_context, max_retries)
    return render_explanation_merge_tree(root, pieces_by_id)
