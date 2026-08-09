"""
Builds a fully-decomposed operator/signal graph for an SVA's property
expression, going deeper than sva_parser.py (which deliberately stops at
sequence leaves to match the paper's Fig. 12 recursion). This module keeps
recursing through the sequence and Boolean layers too, so the result has only
two node types:

    operator - any SVA/Boolean operator: |->, |=>, and, or, not, s_eventually,
               ##N, and/or/intersect at the sequence layer, &&, ||, !, ==, !=,
               <, <=, >, >=, and system functions like $past/$rose/$fell/
               $stable/$onehot/$onehot0/$isunknown/$changed (the function name
               is the operator, its arguments are children).
    signal   - identifiers (plain or indexed, e.g. "dqm[0]") and literal
               constants (there's no third node type, so a numeric literal
               like the "2" in $past(addr, 2) is rendered as a signal node).

Clocking, disable iff, the assert/property wrapper, and parentheses are all
dropped entirely -- they never become nodes.

Every operator node also carries the exact source text of the subtree it
represents (its "code" field), e.g. the node for the second ##N in
"##2 A ##1 B" carries the full "##2 A ##1 B", not just "##1" -- to_dot()
renders this as a smaller second line under the operator symbol.

Unlike sva_parser.py, this module never raises on an unrecognized construct:
anything it doesn't have a specific rule for falls back to a signal-node leaf
containing that subtree's raw text, so a graph is always produced.
"""
import pyslang

from sva_parser import (
    find_first_node,
    restore_macros,
    substitute_macros,
)

_next_id = 0


def _new_id():
    global _next_id
    _next_id += 1
    return _next_id


def _operator(label, children, code):
    return {"id": _new_id(), "type": "operator", "label": label, "children": children, "code": code}


def _signal(label):
    return {"id": _new_id(), "type": "signal", "label": label, "children": []}


def _unwrap_delayed_sequence(node, restore):
    """##N chains (DelayedSequenceExpr) are flattened by pyslang into an
    optional `.first` sequence plus a list of `.elements`, each carrying its
    own delay and following sequence: "A ##2 B ##1 C" -> first=A,
    elements=[(2,B), (1,C)]. Per the IEEE grammar, ##n is unary-prefix when
    nothing precedes it and binary-infix once a real sequence_expr does:
        sequence_expr ::= ##n sequence_expr | sequence_expr ##n sequence_expr
    so "A ##2 B ##1 C" parses as (A ##2 B) ##1 C -- ##1's left operand is the
    single sequence_expr value already formed by "A ##2 B", not two separate
    siblings. Fold left-associatively to match that exactly, tracking the
    exact source text of each partial chain alongside it so every ##N node
    can be labeled with the code it actually represents."""
    elements = list(node.elements)
    if node.first is not None:
        chain = _node_to_graph(node.first, restore)
        code_so_far = restore_macros(str(node.first).strip(), restore)
    else:
        # Chain starts directly with a delay (e.g. "##2 ack ##1 done"): the
        # prefix rule applies to the first element, giving a unary ##N node.
        first_element = elements.pop(0)
        delay_label = f"##{str(first_element.delayVal).strip()}"
        term_code = restore_macros(str(first_element.expr).strip(), restore)
        code_so_far = f"{delay_label} {term_code}"
        chain = _operator(delay_label, [_node_to_graph(first_element.expr, restore)], code_so_far)

    for element in elements:
        delay_label = f"##{str(element.delayVal).strip()}"
        term_code = restore_macros(str(element.expr).strip(), restore)
        code_so_far = f"{code_so_far} {delay_label} {term_code}"
        chain = _operator(delay_label, [chain, _node_to_graph(element.expr, restore)], code_so_far)
    return chain


def _node_to_graph(node, restore):
    if node is None:
        # Defensive: an optional/omitted sub-expression somewhere pyslang
        # allows None (e.g. an omitted argument). There's nothing to show.
        return _signal("(omitted)")

    kind = str(node.kind)

    # Binary operators exposing .left/.op/.right or .left/.operatorToken/.right.
    # (ScopedNameSyntax, e.g. "s_mchk6.ended", also has .left/.right but no
    # operator -- it's a single hierarchical signal reference, not a binary op,
    # so it must fall through to the base-case leaf below instead.)
    if hasattr(node, "left") and hasattr(node, "right") and (hasattr(node, "op") or hasattr(node, "operatorToken")):
        op_attr = node.op if hasattr(node, "op") else node.operatorToken
        label = restore_macros(str(op_attr).strip(), restore)
        code = restore_macros(str(node).strip(), restore)
        return _operator(label, [_node_to_graph(node.left, restore), _node_to_graph(node.right, restore)], code)

    # Delay chains: seq ##N seq ##M seq ...
    if kind == "SyntaxKind.DelayedSequenceExpr":
        return _unwrap_delayed_sequence(node, restore)

    # System function / task calls: $past(x, N), $rose(x), $stable(x), ...
    # $past/$rose/etc. also accept an optional trailing clocking-event
    # argument (e.g. $past(a, 2, , @(posedge clk2))); that argument -- and any
    # fully-omitted argument between commas -- is dropped rather than made
    # into a node, consistent with excluding clock nodes everywhere else.
    if kind == "SyntaxKind.InvocationExpression":
        label = restore_macros(str(node.left).strip(), restore)
        code = restore_macros(str(node).strip(), restore)
        children = []
        for item in node.arguments.parameters:
            if str(item.kind) != "SyntaxKind.OrderedArgument":
                continue
            if item.expr is None or str(item.expr.kind) == "SyntaxKind.ClockingPropertyExpr":
                continue
            children.append(_node_to_graph(item.expr, restore))
        return _operator(label, children, code)

    # Unary operators: !x, not(p), s_eventually p, strong(seq)/weak(seq), ...
    if hasattr(node, "operand") and hasattr(node, "operatorToken"):
        label = restore_macros(str(node.operatorToken).strip(), restore)
        code = restore_macros(str(node).strip(), restore)
        return _operator(label, [_node_to_graph(node.operand, restore)], code)
    if kind == "SyntaxKind.UnaryPropertyExpr":
        label = restore_macros(str(node.op).strip(), restore)
        code = restore_macros(str(node).strip(), restore)
        return _operator(label, [_node_to_graph(node.expr, restore)], code)
    if kind == "SyntaxKind.StrongWeakPropertyExpr":
        label = restore_macros(str(node.keyword).strip(), restore)
        code = restore_macros(str(node).strip(), restore)
        return _operator(label, [_node_to_graph(node.expr, restore)], code)

    # Transparent wrappers: parens, SimplePropertyExpr/SimpleSequenceExpr,
    # embedded clock overrides -- none of these are operators or signals.
    if hasattr(node, "expr"):
        return _node_to_graph(node.expr, restore)
    if hasattr(node, "expression"):
        return _node_to_graph(node.expression, restore)

    # Base case: identifiers (plain or indexed), literals, anything else
    # unrecognized -- captured verbatim as a signal-type leaf.
    return _signal(restore_macros(str(node).strip(), restore))


def build_operator_signal_graph(sva_text):
    """Parse sva_text and return the root of a two-node-type (operator/signal)
    tree describing only its property/sequence/Boolean-layer structure."""
    global _next_id
    _next_id = 0

    substituted, restore = substitute_macros(sva_text.strip())
    if "assert" not in substituted or "property" not in substituted:
        substituted = f"assert property ({substituted});"
    elif not substituted.rstrip().endswith(";"):
        substituted += ";"

    source = f"module __sva_graph_scratch_module;\n  {substituted}\nendmodule\n"
    tree = pyslang.syntax.SyntaxTree.fromText(source)

    errors = [d for d in tree.diagnostics if d.isError]
    if errors:
        raise ValueError(f"Parse error(s): {[str(d.code) for d in errors]}")

    spec = find_first_node(tree.root, "SyntaxKind.PropertySpec")
    if spec is None:
        raise ValueError("No property spec found (immediate assertion?)")

    return _node_to_graph(spec.expr, restore)


def dfs_nodes(root):
    """Pre-order (root-first) depth-first traversal of the tree."""
    order = []

    def visit(node):
        order.append(node)
        for child in node["children"]:
            visit(child)

    visit(root)
    return order


def render_dfs_structure(root):
    """Render the tree as a numbered, depth-first list -- ground truth for a
    prompt that asks an LLM to explain the tree node by node in that same
    order, referencing operand node numbers instead of re-deriving structure."""
    nodes = dfs_nodes(root)
    index_of = {node["id"]: i for i, node in enumerate(nodes, start=1)}
    lines = []
    for i, node in enumerate(nodes, start=1):
        if node["type"] == "signal":
            lines.append(f'Node {i} [signal, leaf]: {node["label"]}')
        else:
            operand_refs = [f"Node {index_of[child['id']]}" for child in node["children"]]
            lines.append(
                f'Node {i} [operator "{node["label"]}"], code: {node["code"]}, '
                f'operand(s): {", ".join(operand_refs)}'
            )
    return "\n".join(lines)


def render_merge_tree(node):
    """A symbolic derivation, like a math proof's "let T1 = ..., T2 = ...":
    every node in the tree (walked in the same reverse order as before,
    verified against dfs_nodes()) gets a short label T1, T2, ..., assigned
    in that order. Each line just states what that label equals, referring
    to its operand(s) by their *own* labels rather than repeating their full
    code -- "T4 = T3 == T2", not "T4 = out == $past(in)" -- with the actual
    code for that step shown once, in brackets, for a reader who doesn't
    want to trace the chain by hand.

    This sidesteps the whole class of problems every earlier design ran
    into (padded columns, corner glyphs, fixed indentation): none of them
    could avoid *some* line growing as wide as the full accumulated SVA
    code, because they all showed that full code inline at every step. Once
    a step only ever needs to name its *own* operator and its operands'
    labels, every line is short and uniform by construction -- the only
    line that must show the complete assertion is the very last one, which
    is unavoidable regardless of format. (On the largest real example, every
    intermediate line stays short; only the final line reaches ~380
    characters, from the assertion text itself.)

    No indentation, no connectors, nothing to misalign -- the label
    references alone fully encode the tree structure, the same way named
    intermediate variables encode a computation without needing to draw
    lines between them."""
    order = list(reversed(dfs_nodes(node)))
    label_of = {}
    lines = []
    for i, n in enumerate(order, start=1):
        label = f"T{i}"
        label_of[n["id"]] = label
        if n["type"] == "signal":
            lines.append(f"{label} = {n['label']}")
            continue
        operand_labels = [label_of[c["id"]] for c in n["children"]]
        if len(operand_labels) == 2:
            expr = f"{operand_labels[0]} {n['label']} {operand_labels[1]}"
        else:
            # Unary, or 3+ operands (e.g. $past(x, N, gate)): function-call
            # style generalizes cleanly to any arity, including 1.
            expr = f"{n['label']}({', '.join(operand_labels)})"
        lines.append(f"{label} = {expr}   [ = {n['code']} ]")
    lines.append("")
    lines.append(f"Final assertion ({label_of[node['id']]}): {node['code']}")
    return lines


def _html_escape(text):
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def to_dot(root, graph_name="sva"):
    """Render the operator/signal tree as Graphviz DOT source. Operator nodes
    get an HTML-like label with the operator symbol on top and the exact
    source code of the subtree it represents underneath, in a smaller font."""
    lines = [f"digraph {graph_name} {{", '  rankdir="TB";', '  node [fontname="Helvetica"];']

    def visit(node):
        if node["type"] == "operator":
            style = 'shape=box, style="rounded,filled", fillcolor="#3B6EA5"'
            label = (
                "<<TABLE BORDER=\"0\" CELLBORDER=\"0\" CELLSPACING=\"1\">"
                f'<TR><TD><FONT COLOR="white" POINT-SIZE="14"><B>{_html_escape(node["label"])}</B></FONT></TD></TR>'
                f'<TR><TD><FONT COLOR="#DCE6F0" POINT-SIZE="9">{_html_escape(node["code"])}</FONT></TD></TR>'
                "</TABLE>>"
            )
            lines.append(f'  n{node["id"]} [label={label}, {style}];')
        else:
            style = 'shape=ellipse, style=filled, fillcolor="#F2C14E", fontcolor="black"'
            label = node["label"].replace('"', '\\"')
            lines.append(f'  n{node["id"]} [label="{label}", {style}];')
        for child in node["children"]:
            lines.append(f'  n{node["id"]} -> n{child["id"]};')
            visit(child)

    visit(root)
    lines.append("}")
    return "\n".join(lines)
