# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from utils_gen_plan import count_tokens_in_file
import os
import re
import sys
import networkx as nx
from config import FLAGS
from saver import saver

# Import rtl_kg functions
from rtl_kg import extract_rtl_knowledge, build_knowledge_graph

print = saver.log_info


def refine_kg_from_rtl(spec_kg: nx.Graph) -> nx.Graph:
    """
    Refine the given specification Knowledge Graph by linking it with RTL information.

    Args:
        spec_kg (nx.Graph): Existing specification Knowledge Graph

    Returns:
        nx.Graph: Combined and linked Knowledge Graph
    """
    print(f"Starting Knowledge Graph refinement with RTL from {FLAGS.design_dir}")
    print(
        f"Initial specification KG has {len(spec_kg.nodes())} nodes and {len(spec_kg.edges())} edges"
    )

    # Add additional diagnostic info about spec_kg
    print("\n=== Specification KG Analysis ===")
    node_types = {}
    for _, data in spec_kg.nodes(data=True):
        node_type = data.get('type', 'unknown')
        if node_type not in node_types:
            node_types[node_type] = 0
        node_types[node_type] += 1

    print("Node types in specification KG:")
    for node_type, count in node_types.items():
        print(f"  - {node_type}: {count} nodes")

    # Check if spec_kg has 'text' attributes that we can use for linking
    text_nodes = sum(1 for _, data in spec_kg.nodes(data=True) if 'text' in data)
    print(
        f"Specification KG has {text_nodes} nodes with 'text' attributes for matching"
    )

    try:
        # Create a new combined graph that will contain both spec and RTL information
        combined_kg, rtl_knowledge = link_spec_and_rtl_graphs(spec_kg, FLAGS.design_dir)

        # Analyze the connectivity in the combined graph
        analyze_graph_connectivity(combined_kg)
    except Exception as e:
        print(f"Error during KG refinement: {str(e)}")
        import traceback

        traceback.print_exc()
        print("Returning original KG without refinement")
        return spec_kg

    print(
        f"Combined KG now has {len(combined_kg.nodes())} nodes and {len(combined_kg.edges())} edges"
    )
    
    return combined_kg, rtl_knowledge


def link_spec_and_rtl_graphs(spec_kg: nx.Graph, design_dir: str) -> nx.Graph:
    """
    Link the specification KG with RTL KG.

    Args:
        spec_kg (nx.Graph): Specification Knowledge Graph
        design_dir (str): Path to the design directory containing RTL files

    Returns:
        nx.Graph: Combined Knowledge Graph with links between spec and RTL
    """
    # Extract RTL knowledge using the existing function
    rtl_knowledge = extract_rtl_knowledge(design_dir, output_dir=None, verbose=True)

    # Build RTL knowledge graph using the existing function
    rtl_kg = build_knowledge_graph(rtl_knowledge)
    print(
        f"Built RTL KG with {rtl_kg.number_of_nodes()} nodes and {rtl_kg.number_of_edges()} edges"
    )

    # Create a combined KG by first copying the spec KG
    combined_kg = spec_kg.copy()

    # Add all nodes and edges from RTL KG to the combined KG
    # We need to make sure node IDs don't conflict
    rtl_node_mapping = {}  # Maps RTL node IDs to new IDs in the combined graph

    # Add RTL nodes to combined graph with a prefix to avoid conflicts
    for node, data in rtl_kg.nodes(data=True):
        # Create a new node ID with rtl_ prefix
        new_node_id = f"rtl_{node}"
        rtl_node_mapping[node] = new_node_id

        # Add node to combined graph
        combined_kg.add_node(new_node_id, **data)

        # Add an attribute to indicate this is an RTL node
        combined_kg.nodes[new_node_id]['source'] = 'rtl'

    # Add RTL edges to combined graph
    new_edges = 0
    for u, v, data in rtl_kg.edges(data=True):
        combined_kg.add_edge(rtl_node_mapping[u], rtl_node_mapping[v], **data)
        new_edges += 1

    print(
        f"Added {len(rtl_node_mapping)} RTL nodes and {new_edges} RTL edges to combined KG"
    )

    # Now create links between spec KG and RTL KG
    link_count = link_modules_to_spec(combined_kg, rtl_node_mapping)
    print(f"Created {link_count} links between specification and RTL nodes")

    # Add additional links based on signal name matching
    signal_link_count = link_signals_to_spec(combined_kg, rtl_node_mapping)
    print(f"Created {signal_link_count} additional links based on signal name matching")

    # Ensure graph connectivity by adding a root node if necessary
    ensure_graph_connectivity(combined_kg)

    return combined_kg, rtl_knowledge


def link_modules_to_spec(combined_kg: nx.Graph, rtl_node_mapping: dict) -> int:
    """
    Link module nodes from RTL KG to relevant nodes in the spec KG.

    Args:
        combined_kg (nx.Graph): Combined Knowledge Graph
        rtl_node_mapping (dict): Mapping from original RTL node IDs to new IDs

    Returns:
        int: Number of links created
    """
    link_count = 0
    created_links = []  # Store details of created links

    # Identify RTL module nodes
    rtl_module_nodes = [
        node
        for node, data in combined_kg.nodes(data=True)
        if data.get('source') == 'rtl' and data.get('type') == 'module'
    ]

    # Identify spec nodes
    spec_nodes = [
        node
        for node, data in combined_kg.nodes(data=True)
        if data.get('source') != 'rtl'
    ]

    # Create a "design" node as a bridge between spec and RTL if it doesn't exist
    design_node = "design_root"
    if design_node not in combined_kg:
        combined_kg.add_node(
            design_node,
            type="root",
            name="Design Root",
            description="Root node connecting specification and RTL",
        )

        # Connect all spec nodes to the design node
        for spec_node in spec_nodes:
            combined_kg.add_edge(
                design_node, spec_node, relationship="contains_spec", weight=0.5
            )
            link_count += 1

            # Copy all node attributes for debugging
            node_attrs = {k: v for k, v in combined_kg.nodes[spec_node].items()}

            created_links.append(
                {
                    "source": design_node,
                    "target": spec_node,
                    "relationship": "contains_spec",
                    "source_type": "root",
                    "target_type": node_attrs.get('type', 'unknown'),
                    "target_attrs": node_attrs,
                }
            )

    # Connect all RTL module nodes to the design node
    for module_node in rtl_module_nodes:
        combined_kg.add_edge(
            design_node, module_node, relationship="contains_implementation", weight=0.5
        )
        link_count += 1

        # Copy all node attributes for debugging
        node_attrs = {k: v for k, v in combined_kg.nodes[module_node].items()}

        created_links.append(
            {
                "source": design_node,
                "target": module_node,
                "relationship": "contains_implementation",
                "source_type": "root",
                "target_type": node_attrs.get('type', 'module'),
                "target_attrs": node_attrs,
            }
        )

        # Extract module name
        module_name = combined_kg.nodes[module_node].get('name', '')
        if not module_name:
            continue

        # Find matching spec nodes based on text similarity
        for spec_node in spec_nodes:
            if 'text' in combined_kg.nodes[spec_node]:
                spec_text = combined_kg.nodes[spec_node].get('text', '')

                # If module name appears in the specification text
                if re.search(
                    r'\b' + re.escape(module_name) + r'\b', spec_text, re.IGNORECASE
                ):
                    combined_kg.add_edge(
                        spec_node,
                        module_node,
                        relationship="describes",
                        weight=1.0,
                        match_type="name_in_text",
                    )
                    link_count += 1

                    # Copy all node attributes for debugging
                    source_attrs = {
                        k: v for k, v in combined_kg.nodes[spec_node].items()
                    }
                    target_attrs = {
                        k: v for k, v in combined_kg.nodes[module_node].items()
                    }

                    created_links.append(
                        {
                            "source": spec_node,
                            "target": module_node,
                            "relationship": "describes",
                            "module_name": module_name,
                            "spec_text_excerpt": (
                                spec_text[:50] + "..."
                                if len(spec_text) > 50
                                else spec_text
                            ),
                            "source_attrs": source_attrs,
                            "target_attrs": target_attrs,
                        }
                    )
                    print(
                        f"Linked spec node to RTL module: {spec_node} --describes--> {module_node}"
                    )

    # Print summary of created links
    print("\n=== Link Summary ===")
    print(f"Total links created: {link_count}")

    # Group links by relationship type
    relationship_counts = {}
    for link in created_links:
        rel = link["relationship"]
        if rel not in relationship_counts:
            relationship_counts[rel] = 0
        relationship_counts[rel] += 1

    for rel, count in relationship_counts.items():
        print(f"- {rel}: {count} links")

    # Print examples of each type of link with detailed node information
    print("\n=== Link Examples ===")
    for rel in relationship_counts.keys():
        examples = [link for link in created_links if link["relationship"] == rel][
            :3
        ]  # Get up to 3 examples
        print(f"\nExamples of '{rel}' links:")
        for i, example in enumerate(examples, 1):
            source = example["source"]
            target = example["target"]
            print(f"  {i}. {source} --{rel}--> {target}")

            # Print additional details based on link type
            if rel == "describes":
                print(f"     Module: {example['module_name']}")
                print(f"     Spec text: {example['spec_text_excerpt']}")

                # Print more detailed source node info
                source_attrs = example.get("source_attrs", {})
                print(f"     Source node details:")
                for k, v in source_attrs.items():
                    if k != 'text':  # Skip long text fields
                        print(f"       {k}: {v}")

                # Print more detailed target node info
                target_attrs = example.get("target_attrs", {})
                print(f"     Target node (module) details:")
                for k, v in target_attrs.items():
                    print(f"       {k}: {v}")
            else:
                # For other relationship types
                if "target_attrs" in example:
                    target_attrs = example["target_attrs"]
                    print(f"     Target node details:")
                    for k, v in target_attrs.items():
                        if k != 'text' and k != 'data':  # Skip long fields
                            print(f"       {k}: {v}")

    return link_count

    return link_count


def link_signals_to_spec(combined_kg: nx.Graph, rtl_node_mapping: dict) -> int:
    """
    Link signal nodes from RTL KG to relevant nodes in the spec KG based on name matching.

    Args:
        combined_kg (nx.Graph): Combined Knowledge Graph
        rtl_node_mapping (dict): Mapping from original RTL node IDs to new IDs

    Returns:
        int: Number of links created
    """
    link_count = 0
    created_links = []  # Store details of created links

    # Identify RTL signal nodes (ports and internal signals)
    rtl_signal_nodes = [
        node
        for node, data in combined_kg.nodes(data=True)
        if data.get('source') == 'rtl' and data.get('type') in ['port', 'signal']
    ]

    # Identify spec nodes
    spec_nodes = [
        node
        for node, data in combined_kg.nodes(data=True)
        if data.get('source') != 'rtl'
    ]

    print(
        f"\nAttempting to link {len(rtl_signal_nodes)} RTL signals to {len(spec_nodes)} spec nodes"
    )

    # For each signal, try to find matching spec nodes
    for signal_node in rtl_signal_nodes:
        signal_name = combined_kg.nodes[signal_node].get('name', '')
        if not signal_name:
            continue

        # Find matching spec nodes
        for spec_node in spec_nodes:
            if 'text' in combined_kg.nodes[spec_node]:
                spec_text = combined_kg.nodes[spec_node].get('text', '')

                # If signal name appears in the specification text
                if re.search(
                    r'\b' + re.escape(signal_name) + r'\b', spec_text, re.IGNORECASE
                ):
                    combined_kg.add_edge(
                        spec_node,
                        signal_node,
                        relationship="references",
                        weight=1.0,
                        match_type="signal_in_text",
                    )
                    link_count += 1
                    created_links.append(
                        {
                            "source": spec_node,
                            "target": signal_node,
                            "relationship": "references",
                            "signal_name": signal_name,
                            "spec_text_excerpt": (
                                spec_text[:50] + "..."
                                if len(spec_text) > 50
                                else spec_text
                            ),
                        }
                    )
                    print(
                        f"Linked spec node to RTL signal: {spec_node} --references--> {signal_node}"
                    )

    # Print summary of signal links
    if link_count > 0:
        print("\n=== Signal Link Summary ===")
        print(f"Total signal links created: {link_count}")

        # Print examples of signal links
        print("\n=== Signal Link Examples ===")
        for i, link in enumerate(created_links[:5], 1):  # Show up to 5 examples
            source = link["source"]
            target = link["target"]
            signal_name = link["signal_name"]
            spec_text = link["spec_text_excerpt"]
            print(f"  {i}. {source} --references--> {target}")
            print(f"     Signal: {signal_name}")
            print(f"     Spec text: {spec_text}")
            print('')

    return link_count


def ensure_graph_connectivity(kg: nx.Graph) -> None:
    """
    Ensure that the graph is connected by adding necessary edges.

    Args:
        kg (nx.Graph): Knowledge Graph
    """
    # Check if the graph is already connected
    if nx.is_connected(kg.to_undirected()):
        print("Graph is already connected")
        return

    # Find connected components
    components = list(nx.connected_components(kg.to_undirected()))
    print(f"Found {len(components)} disconnected components in the graph")

    if len(components) <= 1:
        return

    # Create or find a root node
    root_node = "knowledge_root"
    if root_node not in kg:
        kg.add_node(
            root_node,
            type="root",
            name="Knowledge Root",
            description="Root node ensuring graph connectivity",
        )

    # Connect all components to the root node
    for component in components:
        # Take the first node from each component
        component_node = list(component)[0]

        # Skip if this node is already connected to the root
        if kg.has_edge(root_node, component_node) or kg.has_edge(
            component_node, root_node
        ):
            continue

        # Connect to the root
        kg.add_edge(root_node, component_node, relationship="connects", weight=0.1)
        print(
            f"Connected component to root: {root_node} --connects--> {component_node}"
        )


def analyze_graph_connectivity(kg: nx.Graph) -> None:
    """
    Analyze and print information about graph connectivity.

    Args:
        kg (nx.Graph): Knowledge Graph
    """
    # Check overall connectivity
    undirected = kg.to_undirected()
    is_connected = nx.is_connected(undirected)
    print(f"Graph is{''.join(' not' if not is_connected else '')} connected")

    # Find connected components
    components = list(nx.connected_components(undirected))
    print(f"Number of connected components: {len(components)}")

    # Find bridges between RTL and spec
    rtl_nodes = {
        node for node, data in kg.nodes(data=True) if data.get('source') == 'rtl'
    }

    spec_nodes = {
        node for node, data in kg.nodes(data=True) if data.get('source') != 'rtl'
    }

    bridges = []
    for u, v in kg.edges():
        if (u in rtl_nodes and v in spec_nodes) or (u in spec_nodes and v in rtl_nodes):
            bridges.append((u, v))

    print(f"Number of bridges between RTL and spec: {len(bridges)}")
    for i, (u, v) in enumerate(bridges[:10]):  # Print first 10 bridges
        # Get node details
        u_data = kg.nodes[u]
        v_data = kg.nodes[v]
        u_type = u_data.get('type', 'unknown')
        v_type = v_data.get('type', 'unknown')
        u_name = u_data.get('name', u)
        v_name = v_data.get('name', v)

        print(f"  Bridge {i+1}: {u} ({u_type}: {u_name}) ---> {v} ({v_type}: {v_name})")

    if len(bridges) > 10:
        print(f"  ... and {len(bridges) - 10} more")

    # Find high-degree nodes (hubs)
    degrees = dict(kg.degree())
    sorted_degrees = sorted(degrees.items(), key=lambda x: x[1], reverse=True)

    print("\n=== High-Degree Nodes (Hubs) ===")
    for node, degree in sorted_degrees[:5]:  # Top 5 hubs
        node_data = kg.nodes[node]
        node_type = node_data.get('type', 'unknown')
        node_name = node_data.get('name', node)
        node_source = node_data.get('source', 'spec')

        print(f"  Node: {node} ({node_type}: {node_name})")
        print(f"  Degree: {degree} connections")
        print(f"  Source: {node_source}")
        print(f"  Connected to:")

        # Show a sample of connections
        neighbors = list(kg.neighbors(node))[:5]  # First 5 neighbors
        for neighbor in neighbors:
            neighbor_data = kg.nodes[neighbor]
            neighbor_type = neighbor_data.get('type', 'unknown')
            neighbor_name = neighbor_data.get('name', neighbor)
            print(f"    - {neighbor} ({neighbor_type}: {neighbor_name})")

        if len(list(kg.neighbors(node))) > 5:
            print(f"    - ... and {len(list(kg.neighbors(node))) - 5} more")
        print('')
