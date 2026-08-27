# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from kg_traversal import KGTraversal


import networkx as nx
from typing import List, Dict, Set, Tuple, Optional
from context_pruner import ContextResult
from config import FLAGS
import numpy as np
from saver import saver

print = saver.log_info


class LocalExpansionContextGenerator:
    """
    Generates local context by expanding outward from interface signals using BFS.
    """

    def __init__(self, kg_traversal: 'KGTraversal'):
        self.kg_traversal = kg_traversal
        self.G = self.kg_traversal.graph
        # Access signal mapping if available
        self.signal_to_node_map = getattr(kg_traversal, 'signal_to_node_map', {})
        # Default expansion depth
        self.max_depth = FLAGS.dynamic_prompt_settings['local_expansion']['max_depth']

    def get_contexts(self, start_nodes: List[str]) -> List[ContextResult]:
        """
        Generate local expansion contexts around start nodes using BFS.

        Args:
            start_nodes: List of interface signals to use as expansion starting points
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
            f"Using {len(mapped_nodes)} mapped nodes for local expansion context generation"
        )

        # Skip nodes that are not in the graph
        valid_nodes = [node for node in mapped_nodes if node in self.G]
        if not valid_nodes:
            print("No valid nodes found in graph for local expansion")
            return contexts

        print(f"Found {len(valid_nodes)}/{len(mapped_nodes)} nodes in the graph")

        # Process each valid node
        for start_node in valid_nodes:
            try:
                # Perform BFS expansion up to max_depth
                local_subgraph = self._bfs_expansion(start_node, self.max_depth)

                if not local_subgraph:
                    print(f"No local subgraph found for node {start_node}")
                    continue

                # Calculate metrics and generate description
                metrics = self._calculate_subgraph_metrics(local_subgraph)
                description = self._describe_local_subgraph(
                    local_subgraph, start_node, metrics
                )

                # Calculate score based on subgraph quality
                score = self._calculate_context_score(
                    local_subgraph, start_node, metrics
                )

                contexts.append(
                    ContextResult(
                        text=description,
                        score=score,
                        source_type='local_expansion',
                        metadata={
                            'start_node': start_node,
                            'expansion_depth': self.max_depth,
                            'subgraph_size': len(local_subgraph),
                            'metrics': metrics,
                        },
                    )
                )
                print(
                    f"Generated local expansion context for {start_node} with {len(local_subgraph)} nodes"
                )

            except Exception as e:
                print(
                    f"Error processing local expansion for node {start_node}: {str(e)}"
                )
                import traceback

                traceback.print_exc()

        print(f"Generated {len(contexts)} local expansion contexts")
        return contexts

    def _bfs_expansion(self, start_node: str, max_depth: int) -> Set[str]:
        """
        Perform BFS expansion from start_node up to max_depth.

        Args:
            start_node: Starting node for BFS
            max_depth: Maximum depth for BFS expansion

        Returns:
            Set of nodes in the local subgraph
        """
        visited = {start_node}
        frontier = {start_node}
        depth = 0

        while frontier and depth < max_depth:
            next_frontier = set()
            for node in frontier:
                for neighbor in self.G.neighbors(node):
                    if neighbor not in visited:
                        next_frontier.add(neighbor)
                        visited.add(neighbor)

            frontier = next_frontier
            depth += 1

            # Optional: limit expansion size to prevent explosion
            max_subgraph_size = FLAGS.dynamic_prompt_settings['local_expansion'].get(
                'max_subgraph_size', 100
            )
            if len(visited) >= max_subgraph_size:
                print(
                    f"Reached maximum subgraph size ({max_subgraph_size}) during BFS expansion"
                )
                break

        return visited

    def _calculate_subgraph_metrics(self, nodes: Set[str]) -> Dict:
        """
        Calculate metrics for the local subgraph.

        Args:
            nodes: Set of nodes in the subgraph

        Returns:
            Dictionary of metrics
        """
        subgraph = self.G.subgraph(nodes)

        # Basic metrics
        metrics = {
            'num_nodes': len(nodes),
            'num_edges': subgraph.number_of_edges(),
            'density': nx.density(subgraph),
        }

        # Node type distribution
        node_types = {}
        for node in nodes:
            node_type = self.G.nodes[node].get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1
        metrics['node_type_distribution'] = node_types

        # Try to compute clustering coefficient (may fail for some graphs)
        try:
            metrics['clustering_coefficient'] = nx.average_clustering(subgraph)
        except:
            metrics['clustering_coefficient'] = 0

        return metrics

    def _calculate_context_score(
        self, nodes: Set[str], start_node: str, metrics: Dict
    ) -> float:
        """
        Calculate a relevance score for the context.

        Args:
            nodes: Set of nodes in the subgraph
            start_node: The starting node for the expansion
            metrics: Subgraph metrics

        Returns:
            Relevance score between 0.0 and 1.0
        """
        # Base score based on subgraph size and richness
        base_score = min(0.5, metrics['num_nodes'] / 100.0)

        # Add points for diversity of node types
        diversity_score = min(0.2, len(metrics['node_type_distribution']) / 10.0)

        # Add points for density (denser subgraphs may contain more useful information)
        density_score = min(0.3, metrics['density'])

        # Final score
        return base_score + diversity_score + density_score

    def _describe_local_subgraph(
        self, nodes: Set[str], start_node: str, metrics: Dict
    ) -> str:
        """
        Generate a detailed description of the local subgraph.

        Args:
            nodes: Set of nodes in the subgraph
            start_node: The starting node for the expansion
            metrics: Subgraph metrics

        Returns:
            Textual description of the subgraph
        """
        G = self.G

        # Get info about the central node
        start_info = G.nodes[start_node]
        start_type = start_info.get('type', 'unknown')
        start_name = start_info.get('name', start_node)
        start_module = start_info.get('module', 'unknown')

        # Header
        parts = [
            f"LOCAL SUBGRAPH ANALYSIS FOR {start_name} ({start_type})",
            f"Located in module: {start_module}",
            f"Subgraph size: {len(nodes)} nodes, {metrics['num_edges']} edges",
            f"Density: {metrics['density']:.3f}, Clustering: {metrics['clustering_coefficient']:.3f}",
            "",
        ]

        # Node type distribution
        parts.append("Node type distribution:")
        for node_type, count in metrics['node_type_distribution'].items():
            parts.append(f"  - {node_type}: {count} nodes")
        parts.append("")

        # Identify significant nodes by type
        signal_nodes = []
        module_nodes = []
        register_nodes = []

        for node in nodes:
            if node == start_node:
                continue

            node_info = G.nodes[node]
            node_type = node_info.get('type', 'unknown')
            node_name = node_info.get('name', node)

            if node_type in ['port', 'signal']:
                signal_nodes.append((node, node_name))
            elif node_type == 'module':
                module_nodes.append((node, node_name))
            elif node_type == 'register':
                register_nodes.append((node, node_name))

        # List important signals
        if signal_nodes:
            parts.append("Key signals in this subgraph:")
            for _, name in signal_nodes[:5]:  # Limit to 5
                parts.append(f"  - {name}")
            if len(signal_nodes) > 5:
                parts.append(f"  - and {len(signal_nodes) - 5} more signals")
            parts.append("")

        # List modules
        if module_nodes:
            parts.append("Modules connected to this signal:")
            for _, name in module_nodes[:3]:  # Limit to 3
                parts.append(f"  - {name}")
            if len(module_nodes) > 3:
                parts.append(f"  - and {len(module_nodes) - 3} more modules")
            parts.append("")

        # List registers if present
        if register_nodes:
            parts.append("Related registers:")
            for _, name in register_nodes[:3]:  # Limit to 3
                parts.append(f"  - {name}")
            if len(register_nodes) > 3:
                parts.append(f"  - and {len(register_nodes) - 3} more registers")
            parts.append("")

        # Direct neighbors analysis
        neighbors = list(G.neighbors(start_node))
        if neighbors:
            parts.append("Direct connections:")
            for neighbor in neighbors[:7]:  # Limit to 7
                neighbor_info = G.nodes[neighbor]
                neighbor_type = neighbor_info.get('type', 'unknown')
                neighbor_name = neighbor_info.get('name', neighbor)

                # Get relationship info from edge
                edge_data = G.get_edge_data(start_node, neighbor)
                relationship = "connected to"
                if edge_data:
                    # Handle different edge data formats
                    if isinstance(edge_data, dict):
                        if any(isinstance(edge_data.get(k), dict) for k in edge_data):
                            # Multiple edges between same nodes
                            for k, e in edge_data.items():
                                if isinstance(e, dict) and (
                                    'relationship' in e or 'relation' in e
                                ):
                                    relationship = e.get(
                                        'relationship',
                                        e.get('relation', 'connected to'),
                                    )
                                    break
                        else:
                            # Single edge
                            relationship = edge_data.get(
                                'relationship',
                                edge_data.get('relation', 'connected to'),
                            )

                parts.append(
                    f"  - {start_name} {relationship} {neighbor_name} ({neighbor_type})"
                )

            if len(neighbors) > 7:
                parts.append(f"  - and {len(neighbors) - 7} more connections")
            parts.append("")

        # Add analysis insights
        parts.append("Analysis insights:")

        # Check if it's a hub
        if len(neighbors) > 5:
            parts.append(
                f"  - {start_name} appears to be a hub node with {len(neighbors)} direct connections"
            )

        # Check for control signal patterns
        control_keywords = ['clk', 'clock', 'reset', 'enable', 'valid', 'ready']
        if any(kw in start_name.lower() for kw in control_keywords):
            parts.append(f"  - {start_name} appears to be a control signal")

            # Specific signal type identification
            if 'clk' in start_name.lower() or 'clock' in start_name.lower():
                parts.append(
                    "  - This is a clock signal, important for timing verification"
                )
            elif 'reset' in start_name.lower():
                parts.append(
                    "  - This is a reset signal, important for initialization verification"
                )
            elif 'enable' in start_name.lower():
                parts.append(
                    "  - This is an enable signal, controls functionality activation"
                )
            elif 'valid' in start_name.lower() or 'ready' in start_name.lower():
                parts.append("  - This appears to be part of a handshaking interface")

        # Check for data signals
        data_keywords = ['data', 'addr', 'value']
        if any(kw in start_name.lower() for kw in data_keywords):
            parts.append(f"  - {start_name} appears to be a data signal")

        return "\n".join(parts)
