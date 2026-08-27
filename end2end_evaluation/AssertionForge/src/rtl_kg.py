# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

#!/usr/bin/env python3

from saver import saver
print = saver.log_info


import os
import sys
import json
import networkx as nx
import matplotlib.pyplot as plt
from rtl_analyzer import RTLAnalyzer


def extract_rtl_knowledge(design_dir, output_dir=None, verbose=False):
    """
    Extract knowledge from RTL design files and return structured data.

    Args:
        design_dir (str): Directory containing RTL design files
        output_dir (str, optional): Directory to save output files
        verbose (bool): Whether to print verbose output

    Returns:
        dict: Structured RTL knowledge
    """
    # Create output directory if specified
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)

    # Run RTL analyzer
    analyzer = RTLAnalyzer(design_dir, verbose)
    analyzer.analyze_design()

    # Extract structured knowledge from the analyzer
    rtl_knowledge = {
        # Design overview
        'design_info': {
            'num_files': len(analyzer.file_info),
            'num_modules': len(analyzer.module_info),
            'num_primary_signals': len(analyzer.primary_signals),
            'design_dir': design_dir,
        },
        # Module information
        'modules': analyzer.module_info,
        # File information
        'files': analyzer.file_info,
        # FSM information
        'fsm_info': analyzer.fsm_info,
        # Protocol patterns identified in the design
        'protocol_patterns': getattr(analyzer, 'protocol_patterns', {}),
        # Verification points extracted from analysis
        'verification_points': extract_verification_points(analyzer),
        # Signal type information
        'signal_types': analyzer.signal_type_info,
        # Primary signals (I/O ports)
        'primary_signals': list(analyzer.primary_signals),
    }

    # Export the data flow graph if requested
    if output_dir:
        # Export data flow graph as GraphML for later use
        nx.write_graphml(
            analyzer.data_flow_graph,
            os.path.join(output_dir, "data_flow_graph.graphml"),
        )

        # Also save a JSON serializable version of the graph
        graph_data = {
            'nodes': list(analyzer.data_flow_graph.nodes(data=True)),
            'edges': list(analyzer.data_flow_graph.edges(data=True)),
        }
        rtl_knowledge['data_flow_graph'] = graph_data

        # Save the complete knowledge as JSON
        with open(os.path.join(output_dir, "rtl_knowledge.json"), 'w') as f:
            # Convert non-serializable objects to strings
            json_data = make_json_serializable(rtl_knowledge)
            json.dump(json_data, f, indent=2)

    rtl_knowledge['combined_content'] = analyzer.combined_content

    return rtl_knowledge


def extract_verification_points(analyzer):
    """Extract verification points from the analyzer"""
    # If the analyzer has already generated verification suggestions, use those
    if (
        hasattr(analyzer, 'verification_suggestions')
        and analyzer.verification_suggestions
    ):
        verification_points = []
        for module_suggestion in analyzer.verification_suggestions:
            module_name = module_suggestion.get('module', '')
            suggestions = module_suggestion.get('suggestions', [])

            for suggestion in suggestions:
                verification_points.append(
                    {
                        'type': suggestion.get('type', 'unknown'),
                        'module': module_name,
                        'signals': suggestion.get('signals', []),
                        'description': suggestion.get('description', ''),
                        'suggestion': suggestion.get('suggestion', ''),
                    }
                )

        return verification_points

    # Fallback: Generate verification points using protocol patterns
    verification_points = []
    protocol_patterns = getattr(analyzer, 'protocol_patterns', {})

    for module_name, patterns in protocol_patterns.items():
        # Reset verification if reset signals exist
        if patterns.get('reset_signals', []):
            verification_points.append(
                {
                    'type': 'reset_behavior',
                    'module': module_name,
                    'signals': patterns['reset_signals'],
                    'description': 'Verify proper reset behavior',
                    'suggestion': 'Check all outputs go to defined state when reset is active',
                }
            )

        # Data stability verification if data and clock signals exist
        if patterns.get('data_signals', []) and patterns.get('clock_signals', []):
            verification_points.append(
                {
                    'type': 'data_stability',
                    'module': module_name,
                    'signals': patterns['data_signals'] + patterns['clock_signals'],
                    'description': 'Verify data signal stability',
                    'suggestion': 'Check data signals maintain stable values when being sampled',
                }
            )

        # Handshaking verification if handshaking signals exist
        if patterns.get('handshaking_signals', []):
            verification_points.append(
                {
                    'type': 'handshaking_protocol',
                    'module': module_name,
                    'signals': patterns['handshaking_signals'],
                    'description': 'Verify handshaking protocol',
                    'suggestion': 'Check proper sequencing of handshaking signals',
                }
            )

    return verification_points


def build_knowledge_graph(rtl_knowledge):
    """
    Build a knowledge graph from RTL knowledge.

    Args:
        rtl_knowledge (dict): RTL knowledge from extract_rtl_knowledge

    Returns:
        networkx.MultiDiGraph: Knowledge graph
    """
    import re
    import networkx as nx

    kg = nx.MultiDiGraph()

    # Create maps for node IDs
    node_id_map = {}  # Maps original ID to integer ID
    reverse_id_map = {}  # Maps integer ID to original ID
    next_node_id = 0  # Counter for generating node IDs

    # Helper to get or create integer ID for a node
    def get_node_id(original_id):
        nonlocal next_node_id
        if original_id not in node_id_map:
            node_id_map[original_id] = next_node_id
            reverse_id_map[next_node_id] = original_id
            next_node_id += 1
        return node_id_map[original_id]

    # [Existing code for adding module nodes, port nodes, etc.]

    # Add module nodes
    module_nodes = {}  # Keep track of module integer IDs
    for module_name, module_data in rtl_knowledge['modules'].items():
        original_id = f"module:{module_name}"
        node_id = get_node_id(original_id)
        module_nodes[module_name] = node_id

        kg.add_node(
            node_id,
            type="module",
            name=module_name,
            original_id=original_id,
            data=module_data,
        )

    # Add port nodes and connect to module
    port_nodes = {}  # Keep track of port integer IDs
    for module_name, module_data in rtl_knowledge['modules'].items():
        module_id = module_nodes[module_name]

        for port_name, port_data in module_data.get('ports', {}).items():
            original_id = f"port:{module_name}.{port_name}"
            node_id = get_node_id(original_id)
            port_nodes[original_id] = node_id

            kg.add_node(
                node_id,
                type="port",
                name=port_name,
                module=module_name,
                original_id=original_id,
                direction=port_data.get('direction'),
                width=port_data.get('width'),
            )

            # Connect port to module
            if port_data.get('direction') == 'input':
                kg.add_edge(
                    node_id, module_id, relation="input_to", id=f"e{len(kg.edges())}"
                )
            else:
                kg.add_edge(
                    module_id, node_id, relation="outputs", id=f"e{len(kg.edges())}"
                )

    # Add module instantiation relationships
    for module_name, module_data in rtl_knowledge['modules'].items():
        module_id = module_nodes[module_name]

        for instance_name, instance_data in module_data.get('instances', {}).items():
            instance_module = instance_data['module']
            if instance_module in module_nodes:
                instance_id = module_nodes[instance_module]
                kg.add_edge(
                    module_id,
                    instance_id,
                    relation="instantiates",
                    instance_name=instance_name,
                    id=f"e{len(kg.edges())}",
                )

                # Add port connections
                for port_name, connected_signal in instance_data.get(
                    'connections', {}
                ).items():
                    if connected_signal:
                        source_orig_id = f"port:{module_name}.{connected_signal}"
                        target_orig_id = f"port:{instance_module}.{port_name}"

                        if (
                            source_orig_id in port_nodes
                            and target_orig_id in port_nodes
                        ):
                            source_id = port_nodes[source_orig_id]
                            target_id = port_nodes[target_orig_id]

                            # Add connection based on port direction
                            port_in_child = (
                                rtl_knowledge['modules']
                                .get(instance_module, {})
                                .get('ports', {})
                                .get(port_name, {})
                            )
                            if port_in_child.get('direction') == 'input':
                                kg.add_edge(
                                    source_id,
                                    target_id,
                                    relation="drives",
                                    id=f"e{len(kg.edges())}",
                                )
                            else:
                                kg.add_edge(
                                    target_id,
                                    source_id,
                                    relation="drives",
                                    id=f"e{len(kg.edges())}",
                                )

    # Add FSM nodes and relationships
    fsm_counter = 0
    for fsm_id, fsm_data in rtl_knowledge['fsm_info'].items():
        module_name = fsm_data.get('module', 'unknown')
        original_id = f"fsm:{module_name}_fsm_{fsm_counter}"
        node_id = get_node_id(original_id)
        fsm_counter += 1

        # Truncate the code for storing as an attribute
        code_str = str(fsm_id)
        if len(code_str) > 100:
            code_preview = code_str[:100] + "..."
        else:
            code_preview = code_str

        kg.add_node(
            node_id,
            type="fsm",
            name=f"{module_name}_fsm_{fsm_counter-1}",
            module=module_name,
            original_id=original_id,
            code_preview=code_preview,
            data=fsm_data,
        )

        # Connect FSM to its module
        if module_name and module_name in module_nodes:
            module_id = module_nodes[module_name]
            kg.add_edge(
                node_id, module_id, relation="part_of", id=f"e{len(kg.edges())}"
            )

    # Add protocol pattern nodes
    for module_name, patterns in rtl_knowledge.get('protocol_patterns', {}).items():
        if module_name not in module_nodes:
            continue

        module_id = module_nodes[module_name]

        for pattern_type, signals in patterns.items():
            if signals:  # Only add if there are signals
                original_id = f"pattern:{module_name}.{pattern_type}"
                node_id = get_node_id(original_id)

                kg.add_node(
                    node_id,
                    type="protocol_pattern",
                    pattern_type=pattern_type,
                    module=module_name,
                    original_id=original_id,
                )

                # Connect to module
                kg.add_edge(
                    node_id, module_id, relation="found_in", id=f"e{len(kg.edges())}"
                )

                # Connect to signals
                for signal in signals:
                    signal_orig_id = f"port:{module_name}.{signal}"
                    if signal_orig_id in port_nodes:
                        signal_id = port_nodes[signal_orig_id]
                        kg.add_edge(
                            node_id,
                            signal_id,
                            relation="includes",
                            id=f"e{len(kg.edges())}",
                        )

    # Add verification point nodes
    for i, vp in enumerate(rtl_knowledge['verification_points']):
        module_name = vp['module']
        if module_name not in module_nodes:
            continue

        original_id = f"verif_point:{vp['type']}_{i}"
        node_id = get_node_id(original_id)

        kg.add_node(
            node_id,
            type="verification_point",
            name=vp['type'],
            module=module_name,
            original_id=original_id,
            description=vp['description'],
            suggestion=vp['suggestion'],
        )

        # Connect to relevant module
        module_id = module_nodes[module_name]
        kg.add_edge(node_id, module_id, relation="targets", id=f"e{len(kg.edges())}")

        # Connect to relevant signals
        for signal in vp['signals']:
            signal_orig_id = f"port:{module_name}.{signal}"
            if signal_orig_id in port_nodes:
                signal_id = port_nodes[signal_orig_id]
                kg.add_edge(
                    node_id, signal_id, relation="involves", id=f"e{len(kg.edges())}"
                )

    # Add new nodes and edges for control flow structures
    if 'control_flow' in rtl_knowledge:
        for module_name, cf_data in rtl_knowledge['control_flow'].items():
            if module_name not in module_nodes:
                continue

            module_id = module_nodes[module_name]

            # Add if statement nodes
            for i, if_stmt in enumerate(cf_data.get('if_statements', [])):
                original_id = f"if:{module_name}.{i}"
                node_id = get_node_id(original_id)

                kg.add_node(
                    node_id,
                    type="control_flow",
                    subtype="if_statement",
                    name=f"{module_name}_if_{i}",
                    module=module_name,
                    original_id=original_id,
                    condition=if_stmt.get('condition', ''),
                )

                # Connect to module
                kg.add_edge(
                    node_id, module_id, relation="part_of", id=f"e{len(kg.edges())}"
                )

                # Connect to signals used in condition (if available)
                for signal_name in if_stmt.get('signals_in_condition', []):
                    signal_orig_id = f"port:{module_name}.{signal_name}"
                    if signal_orig_id in port_nodes:
                        signal_id = port_nodes[signal_orig_id]
                        kg.add_edge(
                            node_id,
                            signal_id,
                            relation="references",
                            id=f"e{len(kg.edges())}",
                        )

            # Add case statement nodes
            for i, case_stmt in enumerate(cf_data.get('case_statements', [])):
                original_id = f"case:{module_name}.{i}"
                node_id = get_node_id(original_id)

                kg.add_node(
                    node_id,
                    type="control_flow",
                    subtype="case_statement",
                    name=f"{module_name}_case_{i}",
                    module=module_name,
                    original_id=original_id,
                    case_expression=case_stmt.get('case_expression', ''),
                )

                # Connect to module
                kg.add_edge(
                    node_id, module_id, relation="part_of", id=f"e{len(kg.edges())}"
                )

                # Connect to the case expression signal if it's a port
                case_expr = case_stmt.get('case_expression', '')
                if case_expr:
                    signal_orig_id = f"port:{module_name}.{case_expr}"
                    if signal_orig_id in port_nodes:
                        signal_id = port_nodes[signal_orig_id]
                        kg.add_edge(
                            node_id,
                            signal_id,
                            relation="switches_on",
                            id=f"e{len(kg.edges())}",
                        )

            # Add loop nodes
            for i, loop in enumerate(cf_data.get('loops', [])):
                original_id = f"loop:{module_name}.{i}"
                node_id = get_node_id(original_id)

                kg.add_node(
                    node_id,
                    type="control_flow",
                    subtype=loop.get('type', 'unknown'),
                    name=f"{module_name}_{loop.get('type', 'loop')}_{i}",
                    module=module_name,
                    original_id=original_id,
                )

                # Connect to module
                kg.add_edge(
                    node_id, module_id, relation="part_of", id=f"e{len(kg.edges())}"
                )

    # Add assignment relationship nodes and edges
    for module_name, module_data in rtl_knowledge['modules'].items():
        if module_name not in module_nodes:
            continue

        module_id = module_nodes[module_name]

        for i, assignment in enumerate(module_data.get('assignments', [])):
            original_id = f"assign:{module_name}.{i}"
            node_id = get_node_id(original_id)

            # LHS and RHS signals
            lhs = assignment.get('lhs', '')
            rhs = assignment.get('rhs', '')
            rhs_signals = assignment.get('rhs_signals', [])

            kg.add_node(
                node_id,
                type="assignment",
                name=f"{lhs}_assignment",
                module=module_name,
                original_id=original_id,
                assignment_type=assignment.get('type', 'unknown'),
                expression=f"{lhs} = {rhs}",
            )

            # Connect to module
            kg.add_edge(
                node_id, module_id, relation="part_of", id=f"e{len(kg.edges())}"
            )

            # Connect to LHS signal (target)
            lhs_orig_id = f"port:{module_name}.{lhs}"
            if lhs_orig_id in port_nodes:
                lhs_id = port_nodes[lhs_orig_id]
                kg.add_edge(
                    node_id, lhs_id, relation="assigns_to", id=f"e{len(kg.edges())}"
                )

            # Connect from RHS signals (sources)
            for signal_name in rhs_signals:
                rhs_orig_id = f"port:{module_name}.{signal_name}"
                if rhs_orig_id in port_nodes:
                    rhs_id = port_nodes[rhs_orig_id]
                    kg.add_edge(
                        rhs_id, node_id, relation="used_in", id=f"e{len(kg.edges())}"
                    )

    # Add detailed signal attribute nodes
    signal_nodes = {}  # Track internal signals (non-ports)

    if 'signal_attributes' in rtl_knowledge:
        for module_name, signals in rtl_knowledge['signal_attributes'].items():
            if module_name not in module_nodes:
                continue

            module_id = module_nodes[module_name]

            for signal_name, attr in signals.items():
                # Check if this is already a port
                port_orig_id = f"port:{module_name}.{signal_name}"
                if port_orig_id in port_nodes:
                    # Update port node with additional attributes
                    port_id = port_nodes[port_orig_id]
                    for key, value in attr.items():
                        if key not in [
                            'direction',
                            'width',
                        ]:  # Don't overwrite existing port attributes
                            kg.nodes[port_id][key] = value
                else:
                    # Create a new internal signal node
                    original_id = f"signal:{module_name}.{signal_name}"
                    node_id = get_node_id(original_id)
                    signal_nodes[original_id] = node_id

                    kg.add_node(
                        node_id,
                        type="signal",
                        name=signal_name,
                        module=module_name,
                        original_id=original_id,
                        signal_type=attr.get('type', 'unknown'),
                        width=attr.get('width', '1'),
                        signed=attr.get('signed', False),
                    )

                    # Add initial value if present
                    if 'initial_value' in attr:
                        kg.nodes[node_id]['initial_value'] = attr['initial_value']

                    # Connect to module
                    kg.add_edge(
                        node_id, module_id, relation="part_of", id=f"e{len(kg.edges())}"
                    )

    # Store the ID mappings as graph attributes
    kg.graph['node_id_map'] = node_id_map
    kg.graph['reverse_id_map'] = reverse_id_map

    # Print statistics
    print(
        f"Built knowledge graph with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges"
    )

    return kg


def export_graph_to_graphml(kg, output_path, simplify=False):
    """
    Export a knowledge graph to GraphML format, handling serialization issues.

    Args:
        kg (networkx.Graph): Knowledge graph to export
        output_path (str): Path to the output GraphML file
        simplify (bool): Whether to simplify the graph for better compatibility
    """
    import os
    import json
    import networkx as nx

    # If we need a simplified version for tools like Gephi
    if simplify:
        export_kg = nx.MultiDiGraph()

        # Process nodes
        for node, data in kg.nodes(data=True):
            # Keep only simple attributes
            node_data = {
                'type': data.get('type', 'unknown'),
                'label': data.get('name', str(node)),
                'module': data.get('module', ''),
            }

            # Add node
            export_kg.add_node(node, **node_data)

        # Process edges with explicit IDs
        for i, (u, v, key, data) in enumerate(kg.edges(data=True, keys=True)):
            # Create simple edge data
            edge_data = {
                'id': f"e{i}",
                'type': data.get('relation', 'unknown'),
                'label': data.get('relation', 'unknown'),
            }

            # Add edge
            export_kg.add_edge(u, v, key=key, **edge_data)
    else:
        # Use the original graph but process to make sure it's serializable
        export_kg = nx.MultiDiGraph()

        # Process nodes
        for node, data in kg.nodes(data=True):
            # Convert complex attributes to JSON strings
            node_data = {}
            for k, v in data.items():
                if isinstance(v, (dict, list, tuple, set)):
                    node_data[k] = json.dumps(make_json_serializable(v))
                elif v is None:
                    node_data[k] = ""
                else:
                    node_data[k] = v

            # Add node
            export_kg.add_node(node, **node_data)

        # Process edges
        for i, (u, v, key, data) in enumerate(kg.edges(data=True, keys=True)):
            # Convert complex attributes to JSON strings
            edge_data = {'id': f"e{i}"}
            for k, v in data.items():
                if isinstance(v, (dict, list, tuple, set)):
                    edge_data[k] = json.dumps(make_json_serializable(v))
                elif v is None:
                    edge_data[k] = ""
                else:
                    edge_data[k] = v

            # Add edge
            export_kg.add_edge(u, v, key=key, **edge_data)

    # Print graph statistics to verify edge preservation
    print(f"Original graph: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
    print(
        f"Export graph: {export_kg.number_of_nodes()} nodes, {export_kg.number_of_edges()} edges"
    )

    # Save the graph
    try:
        nx.write_graphml(export_kg, output_path)
        print(f"Successfully wrote GraphML file to {output_path}")
        return True
    except Exception as e:
        print(f"Error writing GraphML: {e}")

        # Try with minimal attributes as fallback
        try:
            # Create an even more simplified graph
            minimal_kg = nx.MultiDiGraph()

            # Copy nodes with minimal attributes
            for node in export_kg.nodes():
                minimal_kg.add_node(node, label=str(node))

            # Copy edges with just IDs
            for i, (u, v, k) in enumerate(export_kg.edges(keys=True)):
                minimal_kg.add_edge(u, v, key=k, id=f"e{i}")

            # Save this minimal version
            nx.write_graphml(minimal_kg, output_path)
            print(f"Wrote minimal GraphML file to {output_path}")
            return True
        except Exception as e:
            print(f"Failed to write even minimal GraphML: {e}")
            return False


def save_knowledge_graph(kg, output_path, output_dir):
    """
    Save knowledge graph to a file in GraphML format.
    Uses the already-assigned integer IDs.

    Args:
        kg (networkx.Graph): Knowledge graph to save
        output_path (str): Path to save the GraphML file
        output_dir (str): Directory for other output files
    """
    import os
    import json
    import networkx as nx

    # Print statistics for verification
    print(
        f"Saving graph with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges"
    )

    # Try to save using NetworkX's GraphML writer
    try:
        nx.write_graphml(kg, output_path)
        print(f"Knowledge graph saved to GraphML at {output_path}")
    except Exception as e:
        print(f"Warning: Failed to save as GraphML: {str(e)}")

        # Save as JSON as backup
        json_output_path = os.path.join(output_dir, "knowledge_graph.json")

        # Convert to node-link format
        data = nx.node_link_data(kg)

        # Make JSON serializable
        def make_serializable(obj):
            if isinstance(obj, dict):
                return {str(k): make_serializable(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [make_serializable(i) for i in obj]
            elif obj is None:
                return None
            elif isinstance(obj, (int, float, bool, str)):
                return obj
            else:
                return str(obj)

        json_data = make_serializable(data)

        with open(json_output_path, 'w') as f:
            json.dump(json_data, f, indent=2)

        print(f"Knowledge graph saved as JSON to {json_output_path}")

        # Try to create a minimal GraphML file as last resort
        try:
            minimal_kg = nx.DiGraph()
            for node in kg.nodes():
                minimal_kg.add_node(node, id=str(node))
            for u, v in kg.edges():
                minimal_kg.add_edge(u, v)
            nx.write_graphml(minimal_kg, output_path)
            print(f"Created minimal GraphML file at {output_path}")
        except Exception as e:
            print(f"Failed to create even a minimal GraphML file: {str(e)}")


def make_json_serializable(obj):
    """Make an object JSON serializable by converting problematic types"""
    if obj is None:
        return None
    elif isinstance(obj, dict):
        return {k: make_json_serializable(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, tuple):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, set):
        return [make_json_serializable(i) for i in obj]
    elif isinstance(obj, (int, float, str, bool)):
        return obj
    else:
        return str(obj)


def save_ultra_simplified_gephi_kg(kg, output_path):
    """
    Create a simplified version of knowledge graph for Gephi visualization.
    Uses the integer IDs already assigned to the graph.

    Args:
        kg (networkx.Graph): Knowledge graph to save
        output_path (str): Path to save the GraphML file
    """
    import networkx as nx

    # Create a simplified graph for Gephi
    gephi_kg = nx.DiGraph()  # Use simple DiGraph for maximum compatibility

    # Copy nodes with simplified attributes
    for node, data in kg.nodes(data=True):
        # Get basic attributes
        node_type = data.get('type', 'unknown')
        node_label = data.get('name', str(node))

        # Create simplified node data
        node_data = {
            'id': str(node),
            'label': node_label,
            'type': node_type,
            'original_id': data.get('original_id', ''),
        }

        # Add node
        gephi_kg.add_node(node, **node_data)

    # Copy edges with simplified attributes
    edge_counter = 0
    for u, v, data in kg.edges(data=True):
        # Create simplified edge data
        edge_data = {
            'id': f"e{edge_counter}",
            'type': data.get('relation', 'unknown'),
            'label': data.get('relation', 'unknown'),
        }

        # Add edge
        gephi_kg.add_edge(u, v, **edge_data)
        edge_counter += 1

    # Print statistics
    print(f"Original graph: {kg.number_of_nodes()} nodes, {kg.number_of_edges()} edges")
    print(
        f"Gephi graph: {gephi_kg.number_of_nodes()} nodes, {gephi_kg.number_of_edges()} edges"
    )

    # Try to save the graph
    try:
        nx.write_graphml(gephi_kg, output_path)
        print(f"Gephi-compatible graph saved to {output_path}")
        return True
    except Exception as e:
        print(f"Error saving Gephi graph: {e}")

        # Try with even more minimal attributes
        try:
            minimal_kg = nx.DiGraph()
            for node in gephi_kg.nodes():
                minimal_kg.add_node(node, label=str(node))
            for i, (u, v) in enumerate(gephi_kg.edges()):
                minimal_kg.add_edge(u, v, id=f"e{i}")
            nx.write_graphml(minimal_kg, output_path)
            print(f"Created minimal Gephi GraphML file at {output_path}")
            return True
        except Exception as e:
            print(f"Failed to create even a minimal Gephi GraphML file: {str(e)}")
            return False


def write_graphml_with_unique_edge_ids(G, path):
    """
    Write a graph to GraphML format with guaranteed unique edge IDs.
    This function ensures that each edge has a unique ID in the GraphML file,
    which is essential for proper visualization in Gephi.

    Args:
        G (networkx.Graph): Graph to write
        path (str): Path to save the GraphML file
    """
    import networkx as nx
    from xml.dom import minidom
    import os

    # First check if the graph already has unique edge IDs
    needs_ids = False
    edge_ids = {}

    # Handle both MultiGraph and simple Graph types
    if hasattr(G, 'is_multigraph') and G.is_multigraph():
        # For multigraphs
        for u, v, key, data in G.edges(data=True, keys=True):
            if 'id' not in data:
                needs_ids = True
                break
            edge_id = str(data['id'])
            if edge_id in edge_ids:
                needs_ids = True
                break
            edge_ids[edge_id] = True
    else:
        # For simple graphs
        for u, v, data in G.edges(data=True):
            if 'id' not in data:
                needs_ids = True
                break
            edge_id = str(data['id'])
            if edge_id in edge_ids:
                needs_ids = True
                break
            edge_ids[edge_id] = True

    # If we need to add IDs, do it before writing
    if needs_ids:
        G_with_ids = G.copy()

        if hasattr(G, 'is_multigraph') and G.is_multigraph():
            # For multigraphs
            for i, (u, v, key) in enumerate(G.edges(keys=True)):
                G_with_ids[u][v][key]['id'] = f"e{i}"
        else:
            # For simple graphs
            for i, (u, v) in enumerate(G.edges()):
                G_with_ids[u][v]['id'] = f"e{i}"

        G = G_with_ids

    # Now write to GraphML
    temp_path = path + ".temp"
    nx.write_graphml(G, temp_path)

    # Process the file to make sure edge IDs are set as attributes and as the 'id' attribute
    doc = minidom.parse(temp_path)
    edges = doc.getElementsByTagName('edge')

    for edge in edges:
        # Make sure the edge has an 'id' attribute
        if not edge.hasAttribute('id'):
            # Look for an id in the data
            edge_data = edge.getElementsByTagName('data')
            id_value = None
            for data in edge_data:
                if data.hasAttribute('key') and data.getAttribute('key').endswith(
                    '.id'
                ):
                    id_value = data.firstChild.nodeValue
                    break

            # If found, use it, otherwise generate a new one
            if id_value:
                edge.setAttribute('id', id_value)
            else:
                # Generate a unique ID
                edge_id = (
                    f"e{edge.getAttribute('source')}_{edge.getAttribute('target')}"
                )
                edge.setAttribute('id', edge_id)

    # Write the modified document
    with open(path, 'w') as f:
        f.write(doc.toxml())

    # Clean up temporary file
    os.remove(temp_path)

    print(f"Wrote GraphML file with unique edge IDs to: {path}")


def main():
    import argparse
    import os

    parser = argparse.ArgumentParser(
        description='Extract knowledge from RTL design and build a knowledge graph'
    )
    parser.add_argument(
        '--design_dir',
        type=str,
        required=True,
        help='Directory containing RTL design files',
    )
    parser.add_argument(
        '--output_dir', type=str, default=None, help='Directory to save output files'
    )
    parser.add_argument('--verbose', action='store_true', help='Enable verbose output')
    args = parser.parse_args()

    # Extract RTL knowledge
    rtl_knowledge = extract_rtl_knowledge(
        args.design_dir, args.output_dir, args.verbose
    )

    # Example usage:
    kg = build_knowledge_graph(rtl_knowledge)
    print(
        f"Built graph with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges"
    )

    # Export for analysis tools
    export_graph_to_graphml(kg, "full_graph.graphml")

    # Save knowledge graph using our custom function
    if args.output_dir:
        kg_output_path = os.path.join(args.output_dir, "knowledge_graph.graphml")
        save_knowledge_graph(kg, kg_output_path, args.output_dir)

        # Generate visualization if output directory is specified
        try:
            plt.figure(figsize=(20, 16))
            pos = nx.spring_layout(kg)
            node_colors = {
                'module': 'lightblue',
                'port': 'lightgreen',
                'fsm': 'salmon',
                'verification_point': 'yellow',
                'protocol_pattern': 'orange',
            }

            # Color nodes by type
            colors = [
                node_colors.get(data.get('type'), 'gray')
                for _, data in kg.nodes(data=True)
            ]

            nx.draw(
                kg,
                pos,
                with_labels=True,
                node_color=colors,
                node_size=1000,
                font_size=8,
                arrows=True,
            )
            plt.title("RTL Knowledge Graph")
            plt.savefig(os.path.join(args.output_dir, "knowledge_graph.png"))
            plt.close()
        except Exception as e:
            print(f"Warning: Failed to generate KG visualization: {str(e)}")

    print(
        f"Successfully built knowledge graph with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges"
    )

    # Generate a Gephi-compatible version of the knowledge graph
    if args.output_dir:
        # Main simplified graph
        ultra_gephi_path = os.path.join(args.output_dir, "ultra_simplified_kg.graphml")
        ultra_gephi_kg = save_ultra_simplified_gephi_kg(kg, ultra_gephi_path)

    if args.output_dir:
        print(f"Results saved to {args.output_dir}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
