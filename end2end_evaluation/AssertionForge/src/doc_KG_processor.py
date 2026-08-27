# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from kg_traversal import KGTraversal
from context_generator_rag import RAGContextGenerator
from context_generator_path import PathBasedContextGenerator
from context_generator_BFS import LocalExpansionContextGenerator
from context_generator_rw import GuidedRandomWalkContextGenerator
from context_pruner import ContextResult
from config import FLAGS
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import networkx as nx
from sentence_transformers import SentenceTransformer
from saver import saver
import networkx.algorithms.community as nx_comm

# from networkx.algorithms import isomorphism  # instead of vf2graph_matcher
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

print = saver.log_info


def create_context_generators(
    spec_text: str, kg: Optional[Dict], valid_signals: List[str], rtl_knowledge
) -> Dict[str, object]:
    """
    Factory function to create context generators based on config settings.

    Args:
        spec_text (str): The specification text
        kg (Optional[Dict]): Knowledge graph if available
        valid_signals (List[str], optional): List of valid signal names to map to KG nodes

    Returns:
        Dict[str, object]: Dictionary of initialized context generators
    """
    context_generators = {}
    signal_to_node_map = {}

    # Print KG structure for debugging
    if kg:
        print("\n=== KG Structure ===")
        if isinstance(kg, dict):
            print(f"KG is a dictionary with keys: {list(kg.keys())}")
            for key, value in kg.items():
                if isinstance(value, list):
                    print(f"  {key}: list with {len(value)} items")
                    if value and len(value) > 0:
                        print(f"    First item type: {type(value[0])}")
                        if isinstance(value[0], dict):
                            print(f"    First item keys: {list(value[0].keys())}")
                else:
                    print(f"  {key}: {type(value)}")
        elif isinstance(kg, nx.Graph):
            print(
                f"KG is a NetworkX graph with {kg.number_of_nodes()} nodes and {kg.number_of_edges()} edges"
            )
            # Print sample node attributes
            if kg.number_of_nodes() > 0:
                sample_node = list(kg.nodes())[0]
                print(f"  Sample node: {sample_node}")
                print(f"  Sample node attributes: {kg.nodes[sample_node]}")
        else:
            print(f"KG is of type {type(kg)}")
    else:
        print("KG is None!")

    # Build signal name to node ID mapping if KG is available and valid_signals provided
    if kg and valid_signals:
        print(
            f"\nBuilding signal name to node ID mapping for {len(valid_signals)} signals..."
        )

        # Handle different KG structures
        if isinstance(kg, nx.Graph):
            # Convert NetworkX graph to dictionary format for the mapping function
            kg_dict = {
                'nodes': [
                    {'id': node, 'attributes': kg.nodes[node]} for node in kg.nodes()
                ],
                'edges': [
                    {'source': u, 'target': v, 'attributes': kg[u][v]}
                    for u, v in kg.edges()
                ],
            }
            signal_to_node_map = build_signal_to_node_mapping(kg_dict, valid_signals)
        else:
            # Assume it's already in the expected dictionary format
            signal_to_node_map = build_signal_to_node_mapping(kg, valid_signals)

        print(f"Created mappings for {len(signal_to_node_map)} signals")

        # Print all mappings for debugging
        for signal, nodes in signal_to_node_map.items():
            print(f"Signal '{signal}' mapped to {len(nodes)} nodes: {nodes}")

    # Initialize RAG generator if enabled
    if FLAGS.dynamic_prompt_settings['rag']['enabled']:

        context_generators['rag'] = RAGContextGenerator(
            spec_text=spec_text,
            chunk_sizes=FLAGS.dynamic_prompt_settings['rag']['chunk_sizes'],
            overlap_ratios=FLAGS.dynamic_prompt_settings['rag']['overlap_ratios'],
            rtl_code=rtl_knowledge['combined_content'] if rtl_knowledge is not None else None
        )

    # Initialize graph-based generators if KG is available
    if kg:
        # Create a fake mapping if we don't have one but have valid signals
        if not signal_to_node_map and valid_signals and isinstance(kg, nx.Graph):
            print("\nCreating fallback signal mapping by direct node name search...")
            fallback_map = {}
            for signal in valid_signals:
                for node, attrs in kg.nodes(data=True):
                    # Try to match signal name against node attributes
                    node_name = attrs.get('name', '')
                    if signal == node_name or (node_name.endswith(f".{signal}")):
                        if signal not in fallback_map:
                            fallback_map[signal] = []
                        fallback_map[signal].append(node)
                        print(
                            f"Found fallback match! Signal '{signal}' -> node {node} (name: {node_name})"
                        )

            if fallback_map:
                print(f"Created fallback mapping for {len(fallback_map)} signals")
                signal_to_node_map = fallback_map

        kg_traversal = KGTraversal(kg)

        # Add signal mapping to KG traversal for access by all generators
        kg_traversal.signal_to_node_map = signal_to_node_map

        # Path-based generator
        if FLAGS.dynamic_prompt_settings['path_based']['enabled']:
            context_generators['path'] = PathBasedContextGenerator(kg_traversal)

        # Motif-based generator
        if FLAGS.dynamic_prompt_settings['motif']['enabled']:
            context_generators['motif'] = MotifContextGenerator(kg_traversal)

        # Community-based generator
        if FLAGS.dynamic_prompt_settings['community']['enabled']:
            context_generators['community'] = CommunityContextGenerator(kg_traversal)

        # Local expansion-based generator
        if FLAGS.dynamic_prompt_settings['local_expansion']['enabled']:
            context_generators['local_expansion'] = LocalExpansionContextGenerator(
                kg_traversal
            )

        # Guided random walk generator (NEW)
        if FLAGS.dynamic_prompt_settings['guided_random_walk']['enabled']:
            context_generators['guided_random_walk'] = GuidedRandomWalkContextGenerator(
                kg_traversal
            )

    return context_generators


def build_signal_to_node_mapping(
    kg: Dict, valid_signals: List[str]
) -> Dict[str, List[str]]:
    """
    Build a mapping from signal names to node IDs in the knowledge graph.

    Args:
        kg (Dict): Knowledge graph
        valid_signals (List[str]): List of valid signal names

    Returns:
        Dict[str, List[str]]: Mapping from signal names to lists of node IDs
    """
    signal_map = {}

    print(f"\n=== Building signal to node mapping for {len(valid_signals)} signals ===")
    print(f"Valid signals: {valid_signals}")
    print(f"KG structure: {list(kg.keys())}")
    print(f"Number of nodes in KG: {len(kg.get('nodes', []))}")

    # Create a set for O(1) lookups
    valid_signal_set = set(valid_signals)

    # Debug counters
    node_types_count = {}
    signal_mentions_by_type = {signal: {} for signal in valid_signals}

    # First, scan all nodes to get a sense of what we're working with
    for node in kg.get('nodes', []):
        node_id = node.get('id', 'unknown')
        node_attrs = node.get('attributes', {})

        # Count node types
        node_type = node_attrs.get('type', 'unknown')
        if node_type not in node_types_count:
            node_types_count[node_type] = 0
        node_types_count[node_type] += 1

        # If this node has a name, check if it contains any of our signals
        node_name = node_attrs.get('name', '')
        if node_name:
            for signal in valid_signal_set:
                if signal in node_name:
                    if node_type not in signal_mentions_by_type[signal]:
                        signal_mentions_by_type[signal][node_type] = 0
                    signal_mentions_by_type[signal][node_type] += 1

    print("\nNode type distribution in KG:")
    for node_type, count in node_types_count.items():
        print(f"  {node_type}: {count} nodes")

    print("\nSignal mentions by node type:")
    for signal, type_counts in signal_mentions_by_type.items():
        if type_counts:
            print(f"  {signal}: {type_counts}")
        else:
            print(f"  {signal}: No mentions found in node names")

    # Iterate through all nodes in the graph to build the actual mapping
    for node in kg.get('nodes', []):
        node_id = node.get('id', 'unknown')
        node_attrs = node.get('attributes', {})

        # Debug output for a few nodes that might be signals
        node_type = node_attrs.get('type', 'unknown')
        node_name = node_attrs.get('name', '')
        if node_type in ['port', 'signal', 'assignment'] and node_name:
            is_likely_signal = False
            for signal in valid_signal_set:
                if signal in node_name:
                    is_likely_signal = True
                    break

            # if is_likely_signal:
            #     print(f"\nPotential signal node found:")
            #     print(f"  ID: {node_id}")
            #     print(f"  Type: {node_type}")
            #     print(f"  Name: {node_name}")
            #     print(f"  Attributes: {node_attrs}")

        # Check if this is a signal-related node (port, signal, or assignment)
        if 'type' in node_attrs and node_attrs['type'] in [
            'port',
            'signal',
            'assignment',
        ]:
            # Look for signal name in the node attributes
            node_name = node_attrs.get('name', '')

            # For assignments, the name might be like "baud_clk_assignment" - extract just the signal name
            signal_base_name = node_name
            if node_attrs['type'] == 'assignment' and '_assignment' in node_name:
                signal_base_name = node_name.replace('_assignment', '')

            # Check if this node name matches a valid signal name
            for signal in valid_signal_set:
                # Try exact match first
                if signal == signal_base_name:
                    if signal not in signal_map:
                        signal_map[signal] = []

                    signal_map[signal].append(node_id)
                    # print(
                    #     f"Exact match! Signal '{signal}' mapped to node {node_id} (type: {node_attrs['type']}, name: {node_name})"
                    # )
                    continue

                # Then try partial match (signal is part of the name)
                # if signal in signal_base_name:
                # print(
                #     f"Partial match: Signal '{signal}' found in node name '{signal_base_name}'"
                # )
                # print(f"  Node ID: {node_id}, Type: {node_attrs['type']}")
                # print(f"  Should we add this mapping? (Currently not adding)")

            # Check other attributes that might contain signal names
            for attr_name, attr_value in node_attrs.items():
                # Expression might contain the signal name
                if attr_name in ['expression'] and isinstance(attr_value, str):
                    for signal in valid_signal_set:
                        # Use word boundary matching to avoid partial matches
                        import re

                        pattern = r'\b' + re.escape(signal) + r'\b'
                        if re.search(pattern, attr_value):
                            if signal not in signal_map:
                                signal_map[signal] = []

                            if node_id not in signal_map[signal]:
                                signal_map[signal].append(node_id)
                                # print(
                                #     f"Expression match! Signal '{signal}' found in expression: '{attr_value}'"
                                # )
                                # print(
                                #     f"  Node ID: {node_id}, Type: {node_attrs['type']}"
                                # )

    # Check if we found mappings for all valid signals
    print("\n=== Signal mapping results ===")
    for signal in valid_signals:
        if signal in signal_map:
            print(
                f"Signal '{signal}' mapped to {len(signal_map[signal])} nodes: {signal_map[signal]}"
            )
        else:
            print(f"WARNING: No node mapping found for signal '{signal}'")

    return signal_map


class MotifContextGenerator:
    """
    Detects and analyzes structural motifs in graphs using VF2 algorithm,
    anchored around specified interface signals.
    """

    def __init__(self, kg_traversal: KGTraversal):
        self.kg_traversal = kg_traversal
        self.G = self.kg_traversal.graph
        # Access signal mapping if available
        self.signal_to_node_map = getattr(kg_traversal, 'signal_to_node_map', {})

    def get_contexts(self, start_nodes: List[str]) -> List[ContextResult]:
        """
        Generate contexts from discovered structural patterns around start nodes.

        Args:
            start_nodes: List of interface signals to anchor the motif search
        """
        contexts = []

        # Convert signal names to actual graph node IDs if we have a mapping
        mapped_nodes = []
        for start_node in start_nodes:
            if start_node in self.signal_to_node_map:
                mapped_nodes.extend(self.signal_to_node_map[start_node])
                print(
                    f"Mapped signal '{start_node}' to {len(self.signal_to_node_map[start_node])} nodes"
                )
            else:
                # Still try with the original name (might be a node ID already)
                mapped_nodes.append(start_node)
                print(f"No mapping found for '{start_node}', using as-is")

        # Remove duplicates
        mapped_nodes = list(set(mapped_nodes))
        print(f"Using {len(mapped_nodes)} mapped nodes for motif context generation")

        # Skip processing if no valid nodes are found in the graph
        valid_nodes = [node for node in mapped_nodes if node in self.G]
        if not valid_nodes:
            print("No valid nodes found in graph for motif detection")
            return contexts

        # Find motifs that include the start nodes
        print("Finding cycle patterns...")
        cycle_patterns = self._find_cycles(valid_nodes)
        print(f"Found {len(cycle_patterns)} cycle patterns")

        print("Finding hub patterns...")
        hub_patterns = self._find_hubs(valid_nodes)
        print(f"Found {len(hub_patterns)} hub patterns")

        print("Finding dense subgraph patterns...")
        dense_subgraphs = self._find_dense_subgraphs(valid_nodes)
        print(f"Found {len(dense_subgraphs)} dense subgraph patterns")

        # Convert patterns to contexts
        for pattern_type, patterns, score in [
            ('cycle', cycle_patterns, 0.8),
            ('hub', hub_patterns, 0.7),
            ('dense', dense_subgraphs, 0.6),
        ]:
            for pattern in patterns:
                importance = self._analyze_pattern_importance(pattern['nodes'])
                final_score = (score + importance) / 2

                # Use enhanced description method for richer context
                context_text = self._describe_enhanced_pattern(pattern_type, pattern)

                contexts.append(
                    ContextResult(
                        text=context_text,
                        score=final_score,
                        source_type='motif',
                        metadata={
                            'type': pattern_type,
                            'nodes': pattern['nodes'],
                            'metrics': pattern['metrics'],
                            'start_node': pattern['start_node'],
                        },
                    )
                )

        print(f"Generated {len(contexts)} motif-based contexts")
        return contexts

    def _find_cycles(self, start_nodes: List[str], max_length: int = 5) -> List[Dict]:
        """Find cyclic patterns that include start nodes"""
        cycles = []

        for start_node in start_nodes:
            try:
                # Find cycles containing this start node
                for cycle in nx.simple_cycles(self.G):
                    if start_node in cycle and len(cycle) <= max_length:
                        metrics = {
                            'length': len(cycle),
                            'edge_density': self._calculate_edge_density(cycle),
                            'avg_degree': self._calculate_avg_degree(cycle),
                        }
                        cycles.append(
                            {
                                'nodes': cycle,
                                'metrics': metrics,
                                'start_node': start_node,
                            }
                        )
            except nx.NetworkXNoCycle:
                continue
            except Exception as e:
                print(f"Error finding cycles for node {start_node}: {str(e)}")
                continue

        return cycles

    def _find_hubs(
        self, start_nodes: List[str], min_degree_ratio: float = 1.5
    ) -> List[Dict]:
        """Find hub patterns connected to start nodes"""
        hubs = []

        try:
            degree_seq = [d for n, d in self.G.degree()]
            avg_degree = sum(degree_seq) / len(degree_seq) if degree_seq else 0

            for start_node in start_nodes:
                # Check if start_node itself is a hub
                if self.G.degree(start_node) > avg_degree * min_degree_ratio:
                    neighbors = list(self.G.neighbors(start_node))
                    metrics = {
                        'degree': self.G.degree(start_node),
                        'degree_ratio': self.G.degree(start_node) / avg_degree,
                        'neighbor_avg_degree': self._calculate_avg_degree(neighbors),
                    }
                    hubs.append(
                        {
                            'nodes': [start_node] + neighbors,
                            'center': start_node,
                            'metrics': metrics,
                            'start_node': start_node,
                        }
                    )

                # Check if start_node is connected to a hub
                for neighbor in self.G.neighbors(start_node):
                    if (
                        self.G.degree(neighbor) > avg_degree * min_degree_ratio
                        and neighbor not in start_nodes
                    ):  # Avoid duplicate hubs
                        hub_neighbors = list(self.G.neighbors(neighbor))
                        metrics = {
                            'degree': self.G.degree(neighbor),
                            'degree_ratio': self.G.degree(neighbor) / avg_degree,
                            'neighbor_avg_degree': self._calculate_avg_degree(
                                hub_neighbors
                            ),
                        }
                        hubs.append(
                            {
                                'nodes': [neighbor] + hub_neighbors,
                                'center': neighbor,
                                'metrics': metrics,
                                'start_node': start_node,
                            }
                        )
        except Exception as e:
            print(f"Error finding hubs: {str(e)}")

        return hubs

    def _find_dense_subgraphs(
        self, start_nodes: List[str], min_size: int = 3
    ) -> List[Dict]:
        """Find dense subgraphs containing start nodes"""
        dense_subgraphs = []

        # For each start node, find its local community
        for start_node in start_nodes:
            try:
                # Get the community containing the start node
                communities = nx_comm.louvain_communities(self.G, weight=None)
                for community in communities:
                    if start_node in community and len(community) >= min_size:
                        subgraph = self.G.subgraph(community)
                        density = nx.density(subgraph)

                        if density > 0.5:
                            metrics = {
                                'size': len(community),
                                'density': density,
                                'avg_degree': self._calculate_avg_degree(community),
                            }
                            dense_subgraphs.append(
                                {
                                    'nodes': list(community),
                                    'metrics': metrics,
                                    'start_node': start_node,
                                }
                            )
            except Exception as e:
                print(f"Warning: Dense subgraph detection failed for {start_node}: {e}")

        return dense_subgraphs

    def _analyze_pattern_importance(self, nodes: List[str]) -> float:
        """Analyze the structural importance of a pattern"""
        structural_score = self._calculate_structural_score(nodes)
        connectivity_score = self._calculate_connectivity_score(nodes)
        centrality_score = self._calculate_centrality_score(nodes)

        return (structural_score + connectivity_score + centrality_score) / 3

    def _calculate_structural_score(self, nodes: List[str]) -> float:
        """Calculate structural importance based on graph metrics"""
        try:
            subgraph = self.G.subgraph(nodes)
            metrics = {
                'density': nx.density(subgraph),
                'avg_degree': sum(dict(subgraph.degree()).values()) / len(nodes),
                'connectivity': nx.average_node_connectivity(subgraph),
            }
            return sum(metrics.values()) / len(metrics)
        except Exception:
            return 0.0

    def _calculate_connectivity_score(self, nodes: List[str]) -> float:
        """Calculate importance based on connectivity patterns"""
        total_edges = self.G.number_of_edges()
        node_edges = sum(self.G.degree(n) for n in nodes)
        return min(1.0, node_edges / (2 * total_edges) if total_edges > 0 else 0)

    def _calculate_centrality_score(self, nodes: List[str]) -> float:
        """Calculate importance based on centrality measures"""
        try:
            centrality_measures = {
                'degree': nx.degree_centrality(self.G),
                'betweenness': nx.betweenness_centrality(self.G),
                'closeness': nx.closeness_centrality(self.G),
            }

            scores = []
            for measure in centrality_measures.values():
                node_scores = [measure[n] for n in nodes if n in measure]
                if node_scores:
                    scores.append(sum(node_scores) / len(node_scores))

            return sum(scores) / len(scores) if scores else 0.0
        except Exception:
            return 0.0

    def _calculate_edge_density(self, nodes: List[str]) -> float:
        """Calculate edge density for a set of nodes"""
        subgraph = self.G.subgraph(nodes)
        n = len(nodes)
        max_edges = n * (n - 1) / 2
        return len(subgraph.edges()) / max_edges if max_edges > 0 else 0

    def _calculate_avg_degree(self, nodes: List[str]) -> float:
        """Calculate average degree for a set of nodes"""
        degrees = [self.G.degree(n) for n in nodes]
        return sum(degrees) / len(degrees) if degrees else 0

    def _describe_enhanced_pattern(self, pattern_type: str, pattern: Dict) -> str:
        """Generate a detailed description of the discovered pattern"""
        G = self.G
        nodes = pattern['nodes']
        start_node = pattern['start_node']

        # Get details about the central node
        central_node_info = G.nodes[start_node]
        central_node_type = central_node_info.get('type', 'unknown')
        central_node_name = central_node_info.get('name', start_node)
        central_node_module = central_node_info.get('module', 'unknown')

        # Common header for all pattern types
        header = (
            f"MOTIF ANALYSIS: {pattern_type.upper()} PATTERN FOR {central_node_name}"
        )

        if pattern_type == 'cycle':
            # For cycle patterns, describe the cycle path and relationships
            cycle_nodes = pattern['nodes']
            desc_parts = [
                header,
                f"Signal: {central_node_name} ({central_node_type}) in module {central_node_module}",
                f"Cycle length: {len(cycle_nodes)} nodes with density {pattern['metrics']['edge_density']:.2f}",
            ]

            # Describe the cycle path
            cycle_path = []
            for i, node in enumerate(cycle_nodes):
                node_info = G.nodes[node]
                node_type = node_info.get('type', 'unknown')
                node_name = node_info.get('name', node)
                node_module = node_info.get('module', '')

                if node_module:
                    cycle_path.append(f"{node_name} ({node_type} in {node_module})")
                else:
                    cycle_path.append(f"{node_name} ({node_type})")

            # Format cycle as A → B → C → ... → A
            desc_parts.append(
                "Cycle path: " + " → ".join(cycle_path) + f" → {cycle_path[0]}"
            )

            # Interpret the cycle's meaning for hardware design
            if 'assignment' in [G.nodes[n].get('type', '') for n in cycle_nodes]:
                desc_parts.append(
                    "NOTE: This cycle contains signal assignments, suggesting a feedback loop or sequential logic."
                )
            elif central_node_type == 'port':
                desc_parts.append(
                    "NOTE: This cycle includes ports, suggesting a communication or synchronization pattern."
                )

            return "\n".join(desc_parts)

        elif pattern_type == 'hub':
            # For hub patterns, describe the hub node and its connections
            center = pattern.get('center', start_node)
            center_info = G.nodes[center]
            center_type = center_info.get('type', 'unknown')
            center_name = center_info.get('name', center)
            center_module = center_info.get('module', 'unknown')

            hub_neighbors = [n for n in pattern['nodes'] if n != center]

            desc_parts = [
                header,
                f"Hub signal: {center_name} ({center_type}) in module {center_module}",
                f"Hub degree: {pattern['metrics']['degree']} connections ({pattern['metrics']['degree_ratio']:.1f}x average)",
            ]

            # Categorize neighbors by type
            neighbor_types = {}
            for n in hub_neighbors[:20]:  # Limit to first 20 neighbors
                n_type = G.nodes[n].get('type', 'unknown')
                if n_type not in neighbor_types:
                    neighbor_types[n_type] = []
                neighbor_types[n_type].append(G.nodes[n].get('name', n))

            # List of neighbors by type
            desc_parts.append("Hub connects to:")
            for n_type, names in neighbor_types.items():
                names_str = ", ".join(names[:7])
                if len(names) > 7:
                    names_str += f" and {len(names) - 7} more"
                desc_parts.append(f"  - {len(names)} {n_type}s: {names_str}")

            # Interpret the hub's role
            if center_type == 'port':
                desc_parts.append(
                    "NOTE: This is a port hub pattern, suggesting this signal is central to module communication."
                )
            elif center_type == 'assignment':
                desc_parts.append(
                    "NOTE: This assignment hub suggests a control or status signal used in multiple conditions."
                )

            return "\n".join(desc_parts)

        elif pattern_type == 'dense':
            # For dense subgraphs, highlight the tightly connected components
            subgraph_nodes = pattern['nodes']

            # Count nodes by type
            node_types = {}
            node_modules = {}
            for node in subgraph_nodes:
                node_info = G.nodes[node]
                node_type = node_info.get('type', 'unknown')
                node_module = node_info.get('module', 'unknown')

                if node_type not in node_types:
                    node_types[node_type] = 0
                node_types[node_type] += 1

                if node_module and node_module not in node_modules:
                    node_modules[node_module] = 0
                if node_module:
                    node_modules[node_module] += 1

            desc_parts = [
                header,
                f"Signal: {central_node_name} ({central_node_type}) in module {central_node_module}",
                f"Dense subgraph size: {len(subgraph_nodes)} nodes with density {pattern['metrics']['density']:.2f}",
            ]

            # Node type breakdown
            desc_parts.append("Node composition:")
            for n_type, count in node_types.items():
                desc_parts.append(
                    f"  - {count} {n_type}s ({count/len(subgraph_nodes)*100:.1f}%)"
                )

            # Module breakdown (if relevant)
            if len(node_modules) > 1:
                desc_parts.append("Module distribution:")
                for module, count in sorted(
                    node_modules.items(), key=lambda x: x[1], reverse=True
                ):
                    if count > 2:  # Only list modules with significant presence
                        desc_parts.append(f"  - {count} nodes in '{module}'")

            # Interpret the dense subgraph
            if len(node_types.get('port', 0)) > 3:
                desc_parts.append(
                    "NOTE: This dense subgraph contains many ports, suggesting an interface or protocol implementation."
                )
            elif node_types.get('assignment', 0) > 3:
                desc_parts.append(
                    "NOTE: This dense subgraph contains many assignments, suggesting complex signal processing logic."
                )

            return "\n".join(desc_parts)

        return f"Found {pattern_type} pattern involving {len(pattern['nodes'])} nodes around signal {central_node_name}"


class CommunityContextGenerator:
    """
    Generates community-based contexts anchored around interface signals.
    """

    def __init__(self, kg_traversal: KGTraversal):
        self.kg_traversal = kg_traversal
        self.G = self.kg_traversal.graph
        # Access signal mapping if available
        self.signal_to_node_map = getattr(kg_traversal, 'signal_to_node_map', {})

    def get_contexts(self, start_nodes: List[str]) -> List[ContextResult]:
        """
        Generate community contexts that include the start nodes.

        Args:
            start_nodes: List of interface signals to anchor the community detection
        """
        contexts = []

        # Convert signal names to actual graph node IDs if we have a mapping
        mapped_nodes = []
        for start_node in start_nodes:
            if start_node in self.signal_to_node_map:
                mapped_nodes.extend(self.signal_to_node_map[start_node])
                print(
                    f"Mapped signal '{start_node}' to {len(self.signal_to_node_map[start_node])} nodes"
                )
            else:
                # Still try with the original name (might be a node ID already)
                mapped_nodes.append(start_node)
                print(f"No mapping found for '{start_node}', using as-is")

        # Remove duplicates
        mapped_nodes = list(set(mapped_nodes))
        print(
            f"Using {len(mapped_nodes)} mapped nodes for community context generation"
        )

        # Skip nodes that are not in the graph
        valid_nodes = [node for node in mapped_nodes if node in self.G]
        if not valid_nodes:
            print("No valid nodes found in graph for community detection")
            return contexts

        # Check if any nodes are in the graph
        print(f"Found {len(valid_nodes)}/{len(mapped_nodes)} nodes in the graph")

        try:
            # Detect communities using a safer method
            communities = self._detect_communities_safely()
            print(f"Detected {len(communities)} communities")

            for node_id in valid_nodes:
                community_found = False
                for i, community in enumerate(communities):
                    if node_id in community:
                        community_found = True
                        try:
                            metrics = self._calculate_community_metrics(community)
                            desc = self._describe_enhanced_community(community, node_id)

                            # Score based on community quality and start node centrality
                            base_score = len(community) / self.G.number_of_nodes()
                            centrality_score = self._calculate_node_centrality(
                                node_id, community
                            )
                            final_score = (base_score + centrality_score) / 2

                            contexts.append(
                                ContextResult(
                                    text=desc,
                                    score=final_score,
                                    source_type='community',
                                    metadata={
                                        'community_id': i,
                                        'size': len(community),
                                        'metrics': metrics,
                                        'start_node': node_id,
                                    },
                                )
                            )
                        except Exception as e:
                            print(
                                f"Error processing community {i} for node {node_id}: {str(e)}"
                            )

                if not community_found:
                    print(f"No community found for node {node_id}")
                    # Try to create a small community based on node neighborhood
                    try:
                        neighbors = list(self.G.neighbors(node_id))
                        if neighbors:
                            local_community = set(
                                [node_id] + neighbors[:20]
                            )  # Limit size
                            metrics = self._calculate_community_metrics(local_community)
                            desc = self._describe_enhanced_community(
                                local_community, node_id
                            )

                            contexts.append(
                                ContextResult(
                                    text=f"[FALLBACK] {desc}",
                                    score=0.4,  # Lower score for fallback
                                    source_type='community',
                                    metadata={
                                        'community_id': 'fallback',
                                        'size': len(local_community),
                                        'metrics': metrics,
                                        'start_node': node_id,
                                        'is_fallback': True,
                                    },
                                )
                            )
                            print(
                                f"Created fallback community for {node_id} with {len(local_community)} nodes"
                            )
                    except Exception as e:
                        print(
                            f"Could not create fallback community for {node_id}: {str(e)}"
                        )

        except Exception as e:
            print(f"Error in community detection: {str(e)}")
            # Attempt a simpler approach as fallback
            try:
                for node_id in valid_nodes:
                    # Create a simple neighborhood community
                    neighbors = list(self.G.neighbors(node_id))
                    if neighbors:
                        local_community = set([node_id] + neighbors[:20])  # Limit size
                        metrics = self._calculate_community_metrics(local_community)
                        desc = self._describe_enhanced_community(
                            local_community, node_id
                        )

                        contexts.append(
                            ContextResult(
                                text=f"[FALLBACK] {desc}",
                                score=0.3,  # Lower score for fallback
                                source_type='community',
                                metadata={
                                    'community_id': 'fallback',
                                    'size': len(local_community),
                                    'metrics': metrics,
                                    'start_node': node_id,
                                    'is_fallback': True,
                                },
                            )
                        )
                        print(
                            f"Created emergency fallback community for {node_id} with {len(local_community)} nodes"
                        )
            except Exception as e:
                print(f"Failed to create even simple communities: {str(e)}")

        print(f"Generated {len(contexts)} community-based contexts")
        return contexts

    def _detect_communities_safely(self) -> List[Set[str]]:
        """Detect communities using a safer method that handles edge weight issues"""
        # First, inspect if there are any edge weight issues
        has_weight_issues = False

        # Check a sample of edges for weight issues
        edge_sample = list(self.G.edges(data=True))[:100]
        for u, v, data in edge_sample:
            for key, value in data.items():
                if isinstance(value, str) and key != 'relationship' and key != 'id':
                    has_weight_issues = True
                    # print(
                    #     f"Found problematic edge weight: {key}={value} (type: {type(value)}"
                    # )

        if has_weight_issues:
            print("Edge weight issues detected, creating unweighted copy of graph")
            # Create an unweighted copy of the graph
            G_unweighted = nx.Graph()
            G_unweighted.add_nodes_from(self.G.nodes(data=True))
            G_unweighted.add_edges_from(self.G.edges())

            # Use alternative community detection method
            try:
                # Try greedy modularity maximization instead
                return list(nx_comm.greedy_modularity_communities(G_unweighted))
            except Exception as e:
                print(f"Greedy modularity failed: {str(e)}")
                # Fall back to connected components if community detection fails
                return [set(comp) for comp in nx.connected_components(G_unweighted)]
        else:
            # Try Louvain with weight=None to ignore weights
            return list(nx_comm.louvain_communities(self.G, weight=None))

    def _calculate_community_metrics(self, community: Set[str]) -> Dict:
        """Calculate metrics for a community"""
        subgraph = self.G.subgraph(community)
        return {
            'density': nx.density(subgraph),
            'avg_degree': sum(dict(subgraph.degree()).values()) / len(community),
            'clustering_coeff': nx.average_clustering(subgraph),
        }

    def _calculate_node_centrality(self, node: str, community: Set[str]) -> float:
        """Calculate the centrality of a node within its community"""
        subgraph = self.G.subgraph(community)
        try:
            centrality = nx.degree_centrality(
                subgraph
            )  # Use degree centrality instead of betweenness
            return centrality.get(node, 0.0)
        except Exception as e:
            print(f"Centrality calculation failed: {str(e)}")
            return 0.0

    def _describe_enhanced_community(self, community: Set[str], start_node: str) -> str:
        """Generate a detailed description of the community focused on the start node"""
        G = self.G

        try:
            # Start with basic information about the central node
            central_node_info = G.nodes[start_node]
            central_node_type = central_node_info.get('type', 'unknown')
            central_node_name = central_node_info.get('name', start_node)
            central_node_module = central_node_info.get('module', 'unknown')

            desc_parts = [
                f"COMMUNITY ANALYSIS FOR SIGNAL: {central_node_name} (type: {central_node_type})",
                f"Located in module: {central_node_module}",
            ]

            # Add community size and metrics
            subgraph = G.subgraph(community)
            metrics = self._calculate_community_metrics(community)
            desc_parts.append(
                f"Community size: {len(community)} nodes with density={metrics['density']:.2f}, "
                f"clustering coefficient={metrics['clustering_coeff']:.2f}"
            )

            # Find related signals in the community
            signal_nodes = []
            module_nodes = []
            assignment_nodes = []
            module_counts = {}

            for node in community:
                node_info = G.nodes[node]
                node_type = node_info.get('type', 'unknown')
                node_name = node_info.get('name', node)
                node_module = node_info.get('module', None)

                if node_type in ['port', 'signal']:
                    signal_nodes.append((node, node_name, node_module))
                elif node_type == 'module':
                    module_nodes.append((node, node_name))
                    if node_name not in module_counts:
                        module_counts[node_name] = 0
                    module_counts[node_name] += 1
                elif node_type == 'assignment':
                    assignment_nodes.append((node, node_name, node_module))

            # List important modules in the community
            if module_nodes:
                modules_str = ", ".join([f"{name}" for _, name in module_nodes[:5]])
                if len(module_nodes) > 5:
                    modules_str += f" and {len(module_nodes) - 5} more"
                desc_parts.append(f"Related modules: {modules_str}")

            # List related signals (focus on ports and signals)
            if signal_nodes:
                # Try to find the most relevant signals (not the central one)
                other_signals = [
                    name
                    for _, name, _ in signal_nodes
                    if name != central_node_name and len(name) > 0
                ]
                if other_signals:
                    signals_str = ", ".join(other_signals[:7])
                    if len(other_signals) > 7:
                        signals_str += f" and {len(other_signals) - 7} more"
                    desc_parts.append(f"Related signals: {signals_str}")

            # Describe key relationships/edges from the central node
            central_node_neighbors = list(G.neighbors(start_node))
            if central_node_neighbors:
                relationships = []
                for neighbor in central_node_neighbors[:5]:  # Limit to 5 neighbors
                    neighbor_info = G.nodes[neighbor]
                    neighbor_type = neighbor_info.get('type', 'unknown')
                    neighbor_name = neighbor_info.get('name', neighbor)

                    # Get edge data to understand relationship
                    edge_data = G.get_edge_data(start_node, neighbor)
                    relation = edge_data.get(
                        'relationship', edge_data.get('relation', 'connected to')
                    )

                    relationships.append(
                        f"{central_node_name} {relation} {neighbor_name} ({neighbor_type})"
                    )

                if relationships:
                    desc_parts.append("Key relationships:")
                    desc_parts.extend([f"  - {rel}" for rel in relationships])

            # If there are assignments, describe them
            relevant_assignments = []
            for _, name, module in assignment_nodes:
                if central_node_name in name:
                    relevant_assignments.append((name, module))

            if relevant_assignments:
                desc_parts.append("Signal assignments:")
                for name, module in relevant_assignments[:3]:  # Limit to 3 assignments
                    desc_parts.append(f"  - {name} in module {module}")
                if len(relevant_assignments) > 3:
                    desc_parts.append(
                        f"  - Plus {len(relevant_assignments) - 3} more assignments"
                    )

            # Add a conclusion about the signal's role based on community analysis
            # Look for clues in the community structure
            if len(central_node_neighbors) > 5:
                desc_parts.append(
                    f"NOTE: {central_node_name} appears to be a hub signal with many connections."
                )
            elif central_node_type == 'port' and central_node_module:
                desc_parts.append(
                    f"NOTE: {central_node_name} is a port in the {central_node_module} module."
                )

            return "\n".join(desc_parts)

        except Exception as e:
            # Simplified fallback description
            print(f"Error generating enhanced community description: {str(e)}")
            return (
                f"Community of size {len(community)} containing signal {start_node}.\n"
                f"This signal appears to be connected to {len(list(G.neighbors(start_node)))} other nodes."
            )
