"""
Deterministic structural parser for the property layer of a concurrent SVA, built
on pyslang (a real SystemVerilog frontend) instead of asking an LLM to infer the
syntax tree itself.

Per the paper's four-layer model (Preliminary.tex, subsec:SVA):
    Boolean layer -> Sequence layer -> Property layer -> Verification layer
and per Fig. 12's own recursion (Step 1/Step 3), only the *property* layer is ever
recursively decomposed: property operators (|->, |=>, and, or, not, s_eventually,
...) combine property/sequence operands, but once a branch bottoms out at a bare
sequence expression, it is translated as a single opaque unit (e.g. "##1 req" or
"$rose(req)") rather than broken down further. So this parser only needs to
recurse through property-layer nodes and stop at the first sequence-layer node it
hits, capturing its exact source text as a leaf.

parse_sva_property() returns a dict:
    {
        "clock": "@(posedge clk)" | None,
        "disable": "disable iff (rst)" | None,
        "property": <node>,
    }
where a <node> is either
    {"kind": "sequence", "text": "..."}
or
    {"kind": "property", "operator": "|->", "operands": [<node>, ...]}

Raises UnsupportedSVAConstruct if the assertion uses a property-layer construct
outside this recursive model (case/conditional properties, clocking overrides,
strong/weak wrappers, abort/accept-on properties, etc.) or fails to parse
outright, so callers can fall back to the LLM-only prompt for that example.
"""
import re

import pyslang

_MACRO_PATTERN = re.compile(r"`(\w+)")
_ASSERT_PROPERTY_PATTERN = re.compile(r"\bassert\s+property\b", re.IGNORECASE)

_BINARY_PROPERTY_KINDS = {
    "SyntaxKind.ImplicationPropertyExpr",
    "SyntaxKind.AndPropertyExpr",
    "SyntaxKind.OrPropertyExpr",
    "SyntaxKind.ImpliesPropertyExpr",
    "SyntaxKind.UntilPropertyExpr",
    "SyntaxKind.UntilWithPropertyExpr",
}
_UNARY_PROPERTY_KINDS = {"SyntaxKind.UnaryPropertyExpr"}
_PAREN_PROPERTY_KIND = "SyntaxKind.ParenthesizedPropertyExpr"
_SIMPLE_PROPERTY_KIND = "SyntaxKind.SimplePropertyExpr"
# strong(seq)/weak(seq) qualify a sequence, not a property; per the paper's
# recursion, sequences are opaque leaves, so these bottom out just like
# SimplePropertyExpr.
_SEQUENCE_LEAF_KINDS = {"SyntaxKind.StrongWeakPropertyExpr"}
# @(...) clock overrides nested inside a property expression don't change the
# property/sequence structure itself; unwrap transparently like parens.
_TRANSPARENT_WRAPPER_KINDS = {_PAREN_PROPERTY_KIND, "SyntaxKind.ClockingPropertyExpr"}


class UnsupportedSVAConstruct(Exception):
    pass


def _placeholder_for(macro_name, index):
    return f"macroph{index}{re.sub(r'[^0-9A-Za-z]', '', macro_name)}"


def substitute_macros(text):
    """Replace `macro tokens with plain identifiers so slang's preprocessor never
    has to resolve them, then return the substituted text plus a restore map."""
    names = sorted(set(_MACRO_PATTERN.findall(text)))
    restore = {}
    substituted = text
    for index, name in enumerate(names):
        placeholder = _placeholder_for(name, index)
        restore[placeholder] = f"`{name}"
        substituted = re.sub(rf"`{re.escape(name)}\b", placeholder, substituted)
    return substituted, restore


def restore_macros(text, restore):
    for placeholder, original in restore.items():
        text = re.sub(rf"\b{re.escape(placeholder)}\b", original, text)
    return text


def find_first_node(node, kind_name):
    if node is None:
        return None
    if str(getattr(node, "kind", "")) == kind_name:
        return node
    try:
        length = len(node)
    except TypeError:
        return None
    for i in range(length):
        try:
            child = node[i]
        except Exception:
            continue
        result = find_first_node(child, kind_name)
        if result is not None:
            return result
    return None


def _node_to_property(node, restore):
    kind = str(node.kind)

    if kind in _BINARY_PROPERTY_KINDS:
        return {
            "kind": "property",
            "operator": str(node.op).strip(),
            "operands": [
                _node_to_property(node.left, restore),
                _node_to_property(node.right, restore),
            ],
        }

    if kind in _UNARY_PROPERTY_KINDS:
        return {
            "kind": "property",
            "operator": str(node.op).strip(),
            "operands": [_node_to_property(node.expr, restore)],
        }

    if kind in _TRANSPARENT_WRAPPER_KINDS:
        return _node_to_property(node.expr, restore)

    if kind == _SIMPLE_PROPERTY_KIND or kind in _SEQUENCE_LEAF_KINDS:
        return {"kind": "sequence", "text": restore_macros(str(node).strip(), restore)}

    raise UnsupportedSVAConstruct(f"Unsupported property-layer construct: {kind}")


def parse_sva_property(sva_text):
    """Parse a concurrent SVA (with or without the `assert property (...);`
    wrapper) into the clock/disable/property-tree structure described above."""
    substituted, restore = substitute_macros(sva_text.strip())

    if _ASSERT_PROPERTY_PATTERN.search(substituted):
        statement = substituted
        if not statement.rstrip().endswith(";"):
            statement += ";"
    else:
        statement = f"assert property ({substituted});"

    source = f"module __sva_parser_scratch_module;\n  {statement}\nendmodule\n"
    tree = pyslang.syntax.SyntaxTree.fromText(source)

    errors = [d for d in tree.diagnostics if d.isError]
    if errors:
        raise UnsupportedSVAConstruct(f"Parse error(s): {[str(d.code) for d in errors]}")

    spec = find_first_node(tree.root, "SyntaxKind.PropertySpec")
    if spec is None:
        raise UnsupportedSVAConstruct("No property spec found (immediate assertion?)")

    clock = restore_macros(str(spec.clocking).strip(), restore) if spec.clocking else None
    disable = restore_macros(str(spec.disable).strip(), restore) if spec.disable else None

    return {
        "clock": clock,
        "disable": disable,
        "property": _node_to_property(spec.expr, restore),
    }


def render_tree(node, indent=0):
    """Render a property node into an indented outline suitable for embedding in
    an LLM prompt as ground-truth structure."""
    pad = "  " * indent
    if node["kind"] == "sequence":
        return f"{pad}- [sequence, leaf] {node['text']}"
    lines = [f"{pad}- [property] operator: {node['operator']} ({len(node['operands'])} operand(s))"]
    for i, operand in enumerate(node["operands"], start=1):
        lines.append(f"{pad}  operand {i}:")
        lines.append(render_tree(operand, indent + 2))
    return "\n".join(lines)


def render_parsed_sva(parsed):
    lines = []
    lines.append(f"Clock: {parsed['clock'] or '(none)'}")
    lines.append(f"Disable condition: {parsed['disable'] or '(none)'}")
    lines.append("Property structure (top-level first):")
    lines.append(render_tree(parsed["property"]))
    return "\n".join(lines)
