# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from kg_traversal import KGTraversal


from context_pruner import ContextResult
from config import FLAGS
import numpy as np
import networkx as nx
from saver import saver

# from networkx.algorithms import isomorphism  # instead of vf2graph_matcher
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

print = saver.log_info


class PathBasedContextGenerator:
    """Extracts path-based contexts from a graph using structural and statistical analysis"""

    def __init__(self, kg_traversal: KGTraversal):
        self.kg_traversal = kg_traversal
        self.G = self.kg_traversal.graph
        # Access signal mapping if available
        self.signal_to_node_map = getattr(kg_traversal, 'signal_to_node_map', {})
        # Calculate global graph metrics once during initialization
        self.global_metrics = self._calculate_global_metrics()

    def get_contexts(self, start_nodes: List[str]) -> List[ContextResult]:
        """Generate context from significant paths in the graph"""
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

        # Remove duplicates and check if they exist in the graph
        mapped_nodes = list(set(mapped_nodes))
        valid_nodes = [node for node in mapped_nodes if node in self.G]
        print(
            f"Using {len(valid_nodes)}/{len(mapped_nodes)} valid nodes for path context generation"
        )

        if not valid_nodes:
            print("No valid nodes found in graph for path analysis")
            return contexts

        for node_id in valid_nodes:
            # Skip if node is not in graph
            if node_id not in self.G:
                print(f"Warning: Node {node_id} not found in graph")
                continue

            significant_paths = self._find_significant_paths(node_id)
            print(
                f"Found {len(significant_paths)} significant paths for node {node_id}"
            )

            for path, path_metrics in significant_paths:
                path_importance = self._calculate_path_importance(path, path_metrics)
                # Use enhanced path description for better context
                path_desc = self._describe_enhanced_path(path, path_metrics)

                contexts.append(
                    ContextResult(
                        text=path_desc,
                        score=path_importance,
                        source_type='path',
                        metadata={
                            'path_length': len(path),
                            'start_node': node_id,
                            'metrics': path_metrics,
                        },
                    )
                )
            # except Exception as e:
            #     print(f"Error processing paths for node {node_id}: {str(e)}")
            #     import traceback

            #     traceback.print_exc()

        print(f"Generated {len(contexts)} path-based contexts")
        return contexts

    def _calculate_global_metrics(self) -> Dict:
        """Calculate global graph metrics for reference"""
        metrics = {
            'avg_degree': sum(dict(self.G.degree()).values())
            / self.G.number_of_nodes(),
            'density': nx.density(self.G),
            'avg_clustering': nx.average_clustering(self.G),
        }

        try:
            # These might fail for some graph types
            metrics.update(
                {
                    'centrality': nx.degree_centrality(self.G),
                    'betweenness': nx.betweenness_centrality(self.G),
                }
            )
        except Exception:
            pass

        return metrics

    def _find_significant_paths(self, start_node: str) -> List[Tuple[List[str], Dict]]:
        """Find structurally significant paths from start_node"""
        significant_paths = []

        # Find potential endpoint nodes based on structural properties
        try:
            target_nodes = self._identify_potential_endpoints(start_node)

            for target in target_nodes:
                if target == start_node:
                    continue

                try:
                    # Find all simple paths between start_node and target
                    max_depth = FLAGS.dynamic_prompt_settings['path_based']['max_depth']
                    all_paths = list(
                        nx.all_simple_paths(
                            self.G, start_node, target, cutoff=max_depth
                        )
                    )

                    for path in all_paths:
                        path_metrics = self._calculate_path_metrics(path)
                        if self._is_significant_path(path_metrics):
                            significant_paths.append((path, path_metrics))

                except nx.NetworkXNoPath:
                    # No path between these nodes
                    continue
                except Exception as e:
                    print(
                        f"Error finding paths from {start_node} to {target}: {str(e)}"
                    )
                    continue

        except Exception as e:
            print(f"Error in _find_significant_paths for {start_node}: {str(e)}")
            import traceback

            traceback.print_exc()

        return significant_paths

    def _identify_potential_endpoints(self, start_node: str) -> Set[str]:
        """Identify potential endpoint nodes based on structural properties"""
        endpoints = set()

        try:
            # Get a subset of nodes to analyze (for performance)
            # First get N-hop neighborhood
            neighborhood = {start_node}
            frontier = {start_node}
            for _ in range(3):  # 3-hop neighborhood
                new_frontier = set()
                for node in frontier:
                    new_frontier.update(self.G.neighbors(node))
                frontier = new_frontier - neighborhood
                neighborhood.update(frontier)
                if len(neighborhood) > 200:  # Limit size
                    break

            # Add some random nodes for diversity
            other_nodes = set(self.G.nodes()) - neighborhood
            if len(other_nodes) > 20:
                import random

                neighborhood.update(random.sample(list(other_nodes), 20))

            # Analyze this subset for potential endpoints
            for node in neighborhood:
                if node == start_node:
                    continue

                # Check various structural indicators
                node_degree = self.G.degree(node)
                try:
                    node_clustering = nx.clustering(self.G, node)
                except:
                    node_clustering = 0

                # Node is interesting if it deviates from average metrics
                if (
                    node_degree > self.global_metrics['avg_degree']
                    or node_clustering > self.global_metrics['avg_clustering']
                ):
                    endpoints.add(node)

                # Check if node is a local hub
                neighbors = set(self.G.neighbors(node))
                if neighbors:
                    neighbor_degrees = [self.G.degree(n) for n in neighbors]
                    avg_neighbor_degree = (
                        sum(neighbor_degrees) / len(neighbor_degrees)
                        if neighbor_degrees
                        else 0
                    )
                    if node_degree > avg_neighbor_degree:
                        endpoints.add(node)

                # Check if node has interesting attributes
                node_attrs = self.G.nodes[node]
                node_type = node_attrs.get('type', 'unknown')

                # Prioritize certain node types
                if node_type in ['port', 'module', 'assignment']:
                    endpoints.add(node)

                # Look for nodes that might have similar names to start node
                start_name = self.G.nodes[start_node].get('name', '')
                node_name = node_attrs.get('name', '')
                if (
                    start_name
                    and node_name
                    and start_name in node_name
                    or node_name in start_name
                ):
                    endpoints.add(node)

        except Exception as e:
            print(f"Error identifying endpoints: {str(e)}")

        # Limit the number of endpoints to avoid combinatorial explosion
        endpoints_list = list(endpoints)
        if len(endpoints_list) > 30:
            import random

            endpoints_list = random.sample(endpoints_list, 30)
            endpoints = set(endpoints_list)

        print(f"  Found {len(endpoints)} potential endpoints for {start_node}")
        return endpoints

    def _calculate_path_metrics(self, path: List[str]) -> Dict:
        """Calculate various metrics for a path"""
        subgraph = self.G.subgraph(path)

        metrics = {
            'length': len(path),
            'edge_density': nx.density(subgraph),
            'path_clustering': nx.average_clustering(subgraph),
            'degrees': [self.G.degree(n) for n in path],
        }

        # Calculate path-specific centrality if possible
        try:
            metrics['path_centrality'] = [
                self.global_metrics['centrality'].get(n, 0) for n in path
            ]
            metrics['path_betweenness'] = [
                self.global_metrics['betweenness'].get(n, 0) for n in path
            ]
        except Exception:
            pass

        return metrics

    def _is_significant_path(self, metrics: Dict) -> bool:
        """Determine if a path is significant based on its metrics"""
        # Path is significant if it meets any of these criteria
        return (
            # High average node degree relative to global average
            sum(metrics['degrees']) / metrics['length']
            > self.global_metrics['avg_degree']
            or
            # High clustering coefficient
            metrics['path_clustering'] > self.global_metrics['avg_clustering']
            or
            # High edge density
            metrics['edge_density'] > self.global_metrics['density']
        )

    def _calculate_path_importance(self, path: List[str], metrics: Dict) -> float:
        """Calculate the overall importance score for a path"""
        scores = []

        # Length-based score (favor moderate lengths)
        length_score = 1.0 / (abs(len(path) - 3) + 1)  # Peak at length 3
        scores.append(length_score)

        # Density score
        scores.append(metrics['edge_density'])

        # Clustering score
        clustering_ratio = (
            metrics['path_clustering'] / self.global_metrics['avg_clustering']
            if self.global_metrics['avg_clustering'] > 0
            else 0
        )
        scores.append(min(1.0, clustering_ratio))

        # Centrality score if available
        if 'path_centrality' in metrics:
            avg_centrality = sum(metrics['path_centrality']) / len(
                metrics['path_centrality']
            )
            scores.append(avg_centrality)

        return sum(scores) / len(scores)

    def _describe_enhanced_path(self, path: List[str], metrics: Dict) -> str:
        """Generate a detailed description of the path with node relationships"""
        G = self.G

        # Get representation style from FLAGS
        rep_style = FLAGS.dynamic_prompt_settings.get('path_based', {}).get(
            'representation_style', 'standard'
        )

        # Get info about source and target nodes
        source_node = path[0]
        target_node = path[-1]
        source_info = G.nodes[source_node]
        target_info = G.nodes[target_node]

        source_type = source_info.get('type', 'unknown')
        source_name = source_info.get('name', source_node)
        source_module = source_info.get('module', 'unknown')

        target_type = target_info.get('type', 'unknown')
        target_name = target_info.get('name', target_node)
        target_module = target_info.get('module', 'unknown')

        # Generate different representations based on style
        if rep_style == 'concise':
            return self._generate_concise_path_description(
                path,
                source_name,
                target_name,
                source_type,
                target_type,
                source_module,
                target_module,
                metrics,
            )
        elif rep_style == 'detailed':
            return self._generate_detailed_path_description(
                path,
                source_name,
                target_name,
                source_type,
                target_type,
                source_module,
                target_module,
                metrics,
            )
        elif rep_style == 'verification_focused':
            return self._generate_verification_focused_path_description(
                path,
                source_name,
                target_name,
                source_type,
                target_type,
                source_module,
                target_module,
                metrics,
            )
        else:  # 'standard' or default

            # Create header
            header_parts = [
                f"PATH ANALYSIS FROM {source_name} TO {target_name}",
                f"Path length: {len(path)} nodes with clustering {metrics['path_clustering']:.2f}",
            ]

            # Add source/target details
            header_parts.append(
                f"Source: {source_name} ({source_type}) in module {source_module}"
            )
            header_parts.append(
                f"Target: {target_name} ({target_type}) in module {target_module}"
            )

            # Create the relationship chain
            relationship_parts = ["Relationship chain:"]
            for i in range(len(path) - 1):
                curr_node = path[i]
                next_node = path[i + 1]

                # Get node attributes
                curr_info = G.nodes[curr_node]
                next_info = G.nodes[next_node]

                curr_type = curr_info.get('type', 'unknown')
                curr_name = curr_info.get('name', curr_node)
                curr_module = curr_info.get('module', '')

                next_type = next_info.get('type', 'unknown')
                next_name = next_info.get('name', next_node)
                next_module = next_info.get('module', '')

                # Get edge attributes
                edge_data = G.get_edge_data(curr_node, next_node)
                if edge_data:
                    # There might be multiple parallel edges
                    # Take the first one or one with relationship info
                    if isinstance(edge_data, dict) and not any(
                        k.isdigit() for k in edge_data.keys()
                    ):
                        # Single edge - use it directly
                        relationship = edge_data.get(
                            'relationship', edge_data.get('relation', '→')
                        )
                    else:
                        # Multiple edges - find one with relationship info
                        for k, e in edge_data.items():
                            if 'relationship' in e or 'relation' in e:
                                relationship = e.get(
                                    'relationship', e.get('relation', '→')
                                )
                                break
                        else:
                            relationship = '→'
                else:
                    relationship = '→'

                # Format node names with module info if available
                curr_fmt = f"{curr_name} ({curr_type}"
                if curr_module:
                    curr_fmt += f" in {curr_module}"
                curr_fmt += ")"

                next_fmt = f"{next_name} ({next_type}"
                if next_module:
                    next_fmt += f" in {next_module}"
                next_fmt += ")"

                relationship_parts.append(f"  {curr_fmt} {relationship} {next_fmt}")

            # Analyze the path for signal flow patterns
            analysis_parts = ["Analysis:"]

            # Check signal flow from ports through assignments
            port_count = sum(1 for n in path if G.nodes[n].get('type') == 'port')
            assignment_count = sum(
                1 for n in path if G.nodes[n].get('type') == 'assignment'
            )
            module_count = sum(1 for n in path if G.nodes[n].get('type') == 'module')

            if port_count > 0 and assignment_count > 0:
                analysis_parts.append(
                    f"  • Signal flow path with {port_count} ports and {assignment_count} assignments"
                )

            if module_count > 0:
                module_names = [
                    G.nodes[n].get('name', '')
                    for n in path
                    if G.nodes[n].get('type') == 'module'
                ]
                module_str = ", ".join(module_names)
                analysis_parts.append(f"  • Cross-module path connecting: {module_str}")

            # Check if it's a source-to-sink path
            if source_type == 'port' and target_type == 'port':
                src_direction = source_info.get('direction', '')
                tgt_direction = target_info.get('direction', '')
                if src_direction == 'input' and tgt_direction == 'output':
                    analysis_parts.append("  • Complete input-to-output signal path")

            # Check for signal transformation
            if source_name in target_name or target_name in source_name:
                analysis_parts.append(
                    f"  • Signal transformation path between related signals: {source_name} and {target_name}"
                )

            # Join all parts
            all_parts = header_parts + [""] + relationship_parts + [""] + analysis_parts
            return "\n".join(all_parts)

    def _generate_concise_path_description(
        self,
        path,
        source_name,
        target_name,
        source_type,
        target_type,
        source_module,
        target_module,
        metrics,
    ):
        """Generate a concise path description for pruning stage"""
        return (
            f"PATH: {source_name} → {target_name} "
            f"(length: {len(path)}, modules: {source_module}/{target_module})"
        )

    def _generate_detailed_path_description(
        self,
        path,
        source_name,
        target_name,
        source_type,
        target_type,
        source_module,
        target_module,
        metrics,
    ):
        """Generate a very detailed path description with all node and edge data"""
        G = self.G
        parts = [
            f"DETAILED PATH: {source_name} → {target_name}",
            f"Path length: {len(path)} nodes with clustering {metrics['path_clustering']:.2f}",
        ]

        # Add full attributes for every node
        for i, node in enumerate(path):
            parts.append(f"Node {i+1}: {G.nodes[node].get('name', node)}")
            # Add all node attributes
            for key, value in G.nodes[node].items():
                parts.append(f"  - {key}: {value}")

            # Add edge information if not the last node
            if i < len(path) - 1:
                next_node = path[i + 1]
                parts.append(f"  Edge to {G.nodes[next_node].get('name', next_node)}:")
                edge_data = G.get_edge_data(node, next_node)
                for key, value in edge_data.items():
                    if isinstance(value, dict):
                        for k, v in value.items():
                            parts.append(f"    - {k}: {v}")
                    else:
                        parts.append(f"    - {key}: {value}")

        return "\n".join(parts)

    def _generate_verification_focused_path_description(
        self,
        path,
        source_name,
        target_name,
        source_type,
        target_type,
        source_module,
        target_module,
        metrics,
    ):
        """Generate a description focused on verification implications"""
        G = self.G

        # Get signal flow details
        signal_flow = []
        for i in range(len(path) - 1):
            curr_node, next_node = path[i], path[i + 1]
            curr_name = G.nodes[curr_node].get('name', curr_node)
            next_name = G.nodes[next_node].get('name', next_node)

            # Get edge relationship
            edge_data = G.get_edge_data(curr_node, next_node)
            relation = "→"
            if edge_data:
                for k, v in edge_data.items():
                    if isinstance(v, dict) and 'relationship' in v:
                        relation = v['relationship']
                        break

            signal_flow.append(f"{curr_name} {relation} {next_name}")

        # Build verification implications
        implications = []

        # Check timing implications
        if 'clock' in source_name.lower() or any(
            'clock' in G.nodes[n].get('name', '').lower() for n in path
        ):
            implications.append(
                "• Timing relationship: Changes must respect clock domain rules"
            )

        # Check for control signals
        control_keywords = ['enable', 'reset', 'valid', 'ready', 'busy', 'new']
        if any(
            kw in source_name.lower() or kw in target_name.lower()
            for kw in control_keywords
        ):
            implications.append(
                "• Control dependency: Verify correct sequencing of control signals"
            )

        # Check for data signals
        data_keywords = ['data', 'addr', 'value']
        if any(
            kw in source_name.lower() or kw in target_name.lower()
            for kw in control_keywords
        ):
            implications.append(
                "• Data integrity: Verify data remains stable during transfers"
            )

        # Check for cross-module implications
        if source_module != target_module:
            implications.append(
                f"• Cross-module interface: Verify handshaking between {source_module} and {target_module}"
            )

        # Format the final output
        parts = [
            f"VERIFICATION-FOCUSED PATH: {source_name} → {target_name}",
            f"Path length: {len(path)} nodes across {len(set(G.nodes[n].get('module', '') for n in path))} modules",
            "",
            "Signal Flow:",
            "  " + " → ".join(signal_flow),
            "",
            "Verification Implications:",
        ]

        parts.extend(
            implications if implications else ["  • No specific implications detected"]
        )

        return "\n".join(parts)
