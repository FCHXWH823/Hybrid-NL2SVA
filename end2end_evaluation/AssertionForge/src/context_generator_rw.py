# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

import networkx as nx
import numpy as np
import random
from typing import List, Dict, Set, Tuple, Optional, DefaultDict
from collections import defaultdict, Counter
from context_pruner import ContextResult
from config import FLAGS
from saver import saver
import heapq

print = saver.log_info


class GuidedRandomWalkContextGenerator:
    """
    Generates context by performing guided random walks from interface signals.
    Uses a biased random walk algorithm that preferentially explores paths
    toward other interface signals.
    """

    def __init__(self, kg_traversal: 'KGTraversal'):
        self.kg_traversal = kg_traversal
        self.G = self.kg_traversal.graph
        # Access signal mapping if available
        self.signal_to_node_map = getattr(kg_traversal, 'signal_to_node_map', {})
        # Get settings from config
        config = FLAGS.dynamic_prompt_settings['guided_random_walk']
        self.num_walks = config.get('num_walks', 10)
        self.walk_budget = config.get('walk_budget', 20)
        self.teleport_prob = config.get('teleport_probability', 0.1)
        self.alpha = config.get('local_importance_weight', 0.3)
        self.beta = config.get('direction_weight', 0.5)
        self.gamma = config.get('discovery_weight', 0.2)
        self.max_targets = config.get('max_targets_per_walk', 3)

        # Pre-computed data structures
        self._signal_distance_map = None
        self._gateway_nodes = None
        self._node_importance = None

    def get_contexts(self, start_nodes: List[str]) -> List[ContextResult]:
        """
        Generate contexts using guided random walks from start nodes.

        Args:
            start_nodes: List of interface signals to use as walk starting points
        """
        contexts = []

        # Map signal names to actual graph nodes
        mapped_nodes = []
        for start_node in start_nodes:
            if start_node in self.signal_to_node_map:
                mapped_nodes.extend(self.signal_to_node_map[start_node])
                print(
                    f"Mapped signal '{start_node}' to {len(self.signal_to_node_map[start_node])} nodes"
                )
            else:
                # Try with the original name (might be a node ID already)
                mapped_nodes.append(start_node)
                print(f"No mapping found for '{start_node}', using as-is")

        # Remove duplicates
        mapped_nodes = list(set(mapped_nodes))
        print(
            f"Using {len(mapped_nodes)} mapped nodes for guided random walk context generation"
        )

        # Skip nodes that aren't in the graph
        valid_nodes = [node for node in mapped_nodes if node in self.G]
        if not valid_nodes:
            print("No valid nodes found in graph for guided random walks")
            return contexts

        print(f"Found {len(valid_nodes)}/{len(mapped_nodes)} nodes in the graph")

        # Check if we have enough interface signals for guided walks
        signal_nodes = list(self.signal_to_node_map.keys())
        if len(signal_nodes) < 2:
            print("Not enough interface signals for guided random walks")
            return contexts

        # Pre-compute distances and gateway nodes if needed
        self._precompute_signal_distances(valid_nodes)
        self._identify_gateway_nodes()
        self._compute_node_importance()

        # Process each valid node as a starting point
        for focus_node in valid_nodes:
            try:
                # Get the corresponding signal name for this node
                focus_signal = self._get_signal_for_node(focus_node)
                if not focus_signal:
                    focus_signal = "unknown"

                # print(
                #     f"Performing guided random walks from {focus_node} (signal: {focus_signal})"
                # )

                # Get other interface signals as potential targets
                other_signals = [s for s in signal_nodes if s != focus_signal]

                # Perform multiple random walks
                paths = []
                for i in range(self.num_walks):
                    path, discovered = self._guided_random_walk(
                        focus_node, other_signals, self.walk_budget
                    )
                    if path and len(path) > 1:
                        paths.append((path, discovered))
                        # print(
                        #     f"Walk {i+1}: Found path with {len(path)} nodes, discovered {len(discovered)} signals"
                        # )

                # Filter and sort paths
                filtered_paths = self._filter_and_rank_paths(paths)

                # Generate contexts from paths
                for path, discovered in filtered_paths[
                    : FLAGS.dynamic_prompt_settings['guided_random_walk'].get(
                        'max_contexts_per_signal', 5
                    )
                ]:
                    path_metrics = self._calculate_path_metrics(path)
                    description = self._describe_path(
                        path, focus_node, discovered, path_metrics
                    )

                    # Score based on path quality and number of signals discovered
                    score = (
                        len(discovered) / len(other_signals) * 0.6
                        + path_metrics.get('quality_score', 0) * 0.4
                    )

                    contexts.append(
                        ContextResult(
                            text=description,
                            score=score,
                            source_type='guided_random_walk',
                            metadata={
                                'focus_node': focus_node,
                                'path_length': len(path),
                                'discovered_signals': list(discovered),
                                'metrics': path_metrics,
                            },
                        )
                    )

                # print(
                #     f"Generated {len(filtered_paths[:5])} contexts from random walks for {focus_node}"
                # )

            except Exception as e:
                print(
                    f"Error processing guided random walks for node {focus_node}: {str(e)}"
                )
                import traceback

                traceback.print_exc()

        print(f"Generated {len(contexts)} guided random walk contexts in total")
        return contexts

    def _precompute_signal_distances(self, focus_nodes: List[str]) -> None:
        """
        Pre-compute distances between interface signals.

        Args:
            focus_nodes: List of nodes we're focusing on for this analysis
        """
        print("Pre-computing distances between interface signals...")

        # Initialize distance map
        self._signal_distance_map = defaultdict(
            lambda: defaultdict(lambda: defaultdict(lambda: float('inf')))
        )

        # Get all signal nodes
        signal_nodes = {}
        for signal, nodes in self.signal_to_node_map.items():
            for node in nodes:
                if node in self.G:
                    signal_nodes[node] = signal

        # For each focus node, compute shortest paths to other signal nodes
        for focus_node in focus_nodes:
            # Skip if node is not in graph
            if focus_node not in self.G:
                continue

            # Compute shortest paths from this node to all others
            try:
                # Use single source shortest path for efficiency
                distances = nx.single_source_shortest_path_length(self.G, focus_node)
                paths = nx.single_source_shortest_path(self.G, focus_node)

                # Store distances and next hops for other signal nodes
                for target_node, target_signal in signal_nodes.items():
                    if target_node != focus_node and target_node in distances:
                        focus_signal = signal_nodes.get(focus_node, "unknown")
                        dist = distances[target_node]

                        # Store distance
                        self._signal_distance_map[focus_signal][target_signal][
                            'distance'
                        ] = dist

                        # Store next hop in path
                        if target_node in paths and len(paths[target_node]) > 1:
                            next_hop = paths[target_node][
                                1
                            ]  # First node after focus_node
                            self._signal_distance_map[focus_signal][target_signal][
                                'next_hop'
                            ] = next_hop

                            # Store complete path
                            self._signal_distance_map[focus_signal][target_signal][
                                'path'
                            ] = paths[target_node]

            except Exception as e:
                print(f"Error computing shortest paths from {focus_node}: {str(e)}")

        print(
            f"Computed distances between {len(self._signal_distance_map)} source signals"
        )

    def _identify_gateway_nodes(self) -> None:
        """
        Identify gateway nodes that appear frequently in shortest paths between interface signals.
        """
        print("Identifying gateway nodes between interface signals...")

        # Skip if we don't have distance map
        if not self._signal_distance_map:
            print("No signal distance map available, skipping gateway identification")
            self._gateway_nodes = {}
            return

        # Count occurrences of nodes in paths between signals
        node_counts = Counter()
        path_count = 0

        # For each source-target signal pair, count nodes in the shortest path
        for source in self._signal_distance_map:
            for target in self._signal_distance_map[source]:
                path = self._signal_distance_map[source][target].get('path')
                if path and len(path) > 2:  # Ignore source and target
                    path_count += 1
                    for node in path[1:-1]:  # Exclude source and target
                        node_counts[node] += 1

        # Calculate gateway score for each node
        gateway_scores = {}
        for node, count in node_counts.items():
            gateway_scores[node] = count / max(path_count, 1)

        # Store the top gateway nodes for each signal pair
        self._gateway_nodes = defaultdict(dict)
        max_gateways = 5  # Store top 5 gateways per signal pair

        for source in self._signal_distance_map:
            for target in self._signal_distance_map[source]:
                path = self._signal_distance_map[source][target].get('path')
                if path and len(path) > 2:
                    # Get internal nodes in this path
                    internal_nodes = path[1:-1]

                    # Score them by general gateway score
                    scored_nodes = [
                        (node, gateway_scores.get(node, 0)) for node in internal_nodes
                    ]

                    # Sort by score and keep top nodes
                    self._gateway_nodes[source][target] = [
                        node
                        for node, _ in sorted(scored_nodes, key=lambda x: -x[1])[
                            :max_gateways
                        ]
                    ]

        print(f"Identified gateway nodes for {len(self._gateway_nodes)} signal pairs")

    def _compute_node_importance(self) -> None:
        """
        Compute importance scores for all nodes based on type, degree, etc.
        """
        print("Computing node importance scores...")

        self._node_importance = {}

        # Calculate base importance for each node
        for node in self.G.nodes():
            # Get node attributes
            attrs = self.G.nodes[node]
            node_type = attrs.get('type', 'unknown')

            # Base score from node degree (normalized)
            degree = self.G.degree(node)
            max_degree = max(dict(self.G.degree()).values()) if self.G else 1
            degree_score = degree / max_degree

            # Type importance (prioritize certain node types)
            type_score = 0.0
            if node_type in ['port', 'signal']:
                type_score = 0.9
            elif node_type in ['register', 'fsm_state']:
                type_score = 0.8
            elif node_type in ['module', 'instance']:
                type_score = 0.7
            elif node_type in ['assignment']:
                type_score = 0.6

            # Combined importance score (weighted sum)
            importance = 0.4 * degree_score + 0.6 * type_score

            # Store importance score
            self._node_importance[node] = importance

        print(f"Computed importance scores for {len(self._node_importance)} nodes")

    def _guided_random_walk(
        self, start_node: str, target_signals: List[str], budget: int
    ) -> Tuple[List[str], Set[str]]:
        """
        Perform a guided random walk from start_node toward target signals.

        Args:
            start_node: Starting node for the walk
            target_signals: List of signals to guide the walk toward
            budget: Maximum number of steps in the walk

        Returns:
            Tuple of (path, discovered_signals)
        """
        # Initialize
        current_node = start_node
        path = [current_node]
        visited = {current_node}

        # Select a subset of target signals for this walk
        num_targets = min(self.max_targets, len(target_signals))
        if num_targets > 0:
            selected_targets = set(random.sample(target_signals, num_targets))
        else:
            selected_targets = set(target_signals)

        discovered_signals = set()
        remaining_budget = budget

        # Start random walk
        while remaining_budget > 0 and selected_targets:
            # Get neighbors of current node
            neighbors = list(self.G.neighbors(current_node))

            # Filter out already visited nodes if possible
            unvisited = [n for n in neighbors if n not in visited]
            if unvisited:
                candidates = unvisited
            else:
                candidates = neighbors

            # If no candidates, terminate walk
            if not candidates:
                break

            # Decide whether to teleport
            if random.random() < self.teleport_prob and selected_targets:
                # Try to teleport to a gateway node
                gateway = self._select_gateway(current_node, selected_targets)
                if gateway and gateway != current_node and gateway not in visited:
                    current_node = gateway
                    path.append(current_node)
                    visited.add(current_node)
                    remaining_budget -= 1
                    continue

            # Compute transition probabilities for each candidate
            probs = self._compute_transition_probabilities(
                current_node, candidates, selected_targets, visited
            )

            # If all probabilities are zero, choose randomly
            if sum(probs) == 0:
                next_node = random.choice(candidates)
            else:
                # Normalize probabilities
                probs = [p / sum(probs) for p in probs]
                next_node = random.choices(candidates, weights=probs, k=1)[0]

            # Move to next node
            current_node = next_node
            path.append(current_node)
            visited.add(current_node)

            # Check if we discovered a target signal
            signal = self._get_signal_for_node(current_node)
            if signal and signal in selected_targets:
                discovered_signals.add(signal)
                selected_targets.remove(signal)

            remaining_budget -= 1

        return path, discovered_signals

    def _compute_transition_probabilities(
        self,
        current_node: str,
        candidates: List[str],
        targets: Set[str],
        visited: Set[str],
    ) -> List[float]:
        """
        Compute transition probabilities for random walk candidates.

        Args:
            current_node: Current node in the walk
            candidates: Candidate next nodes
            targets: Target signals we're trying to reach
            visited: Already visited nodes

        Returns:
            List of probabilities corresponding to candidates
        """
        probs = []

        # Get current signal
        current_signal = self._get_signal_for_node(current_node)

        for candidate in candidates:
            # 1. Local importance component
            local_importance = self._node_importance.get(candidate, 0.1)

            # 2. Direction score component (higher for nodes toward targets)
            direction_score = 0.0

            if current_signal and targets:
                # Check if candidate is on path to any target
                for target in targets:
                    next_hop = (
                        self._signal_distance_map.get(current_signal, {})
                        .get(target, {})
                        .get('next_hop')
                    )
                    if next_hop == candidate:
                        direction_score += 1.0

                # Normalize by number of targets
                direction_score = direction_score / len(targets) if targets else 0

            # 3. Discovery component (prefer unvisited nodes)
            discovery_score = 0.0 if candidate in visited else 1.0

            # Combine scores with weights
            prob = (
                self.alpha * local_importance
                + self.beta * direction_score
                + self.gamma * discovery_score
            )

            probs.append(max(0.01, prob))  # Ensure minimum probability

        return probs

    def _select_gateway(self, current_node: str, targets: Set[str]) -> Optional[str]:
        """
        Select a gateway node to teleport to.

        Args:
            current_node: Current node in the walk
            targets: Target signals we're trying to reach

        Returns:
            Gateway node or None
        """
        current_signal = self._get_signal_for_node(current_node)
        if not current_signal:
            return None

        # Collect gateway candidates for all targets
        gateway_candidates = []

        for target in targets:
            gateways = self._gateway_nodes.get(current_signal, {}).get(target, [])
            for gateway in gateways:
                if gateway != current_node and gateway in self.G:
                    gateway_candidates.append(gateway)

        # If we have gateway candidates, select one randomly
        if gateway_candidates:
            return random.choice(gateway_candidates)

        return None

    def _filter_and_rank_paths(
        self, paths: List[Tuple[List[str], Set[str]]]
    ) -> List[Tuple[List[str], Set[str]]]:
        """
        Filter and rank paths based on quality and signal coverage.

        Args:
            paths: List of (path, discovered_signals) tuples

        Returns:
            Filtered and ranked paths
        """
        # Remove duplicate paths
        unique_paths = []
        seen_paths = set()

        for path, discovered in paths:
            path_key = tuple(path)
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                unique_paths.append((path, discovered))

        # Filter out paths that don't discover any signals
        signal_paths = [
            (path, discovered) for path, discovered in unique_paths if discovered
        ]

        # If we don't have any paths that reach signals, return original paths
        if not signal_paths and unique_paths:
            return unique_paths

        # Score and rank paths
        scored_paths = []
        for path, discovered in signal_paths:
            # Score based on:
            # 1. Number of signals discovered
            # 2. Path length (shorter is better)
            # 3. Node type diversity

            # Basic quality score
            path_metrics = self._calculate_path_metrics(path)
            quality_score = path_metrics.get('quality_score', 0)

            # Overall score
            score = (
                len(discovered) * 10  # Heavily weight signal discovery
                + quality_score
                + (1.0 / len(path))  # Prefer shorter paths
            )

            scored_paths.append((score, path, discovered))

        # Sort by score (descending) and return paths
        scored_paths.sort(reverse=True)
        return [(path, discovered) for _, path, discovered in scored_paths]

    def _calculate_path_metrics(self, path: List[str]) -> Dict:
        """
        Calculate metrics for a path.

        Args:
            path: List of nodes in the path

        Returns:
            Dictionary of metrics
        """
        if not path:
            return {'quality_score': 0}

        # Calculate basic path properties
        length = len(path)

        # Node type distribution
        node_types = {}
        for node in path:
            node_type = self.G.nodes[node].get('type', 'unknown')
            node_types[node_type] = node_types.get(node_type, 0) + 1

        # Calculate diversity score (more types is better)
        diversity = len(node_types) / max(5, length / 2)  # Normalize

        # Calculate importance score (average of node importance)
        importance = sum(self._node_importance.get(node, 0) for node in path) / length

        # Calculate edge quality (based on relationship types)
        edge_quality = 0.0
        edge_count = 0

        for i in range(length - 1):
            source, target = path[i], path[i + 1]
            if self.G.has_edge(source, target):
                edge_count += 1

                # Check edge attributes for relationship quality
                edge_data = self.G.get_edge_data(source, target)
                if isinstance(edge_data, dict):
                    # Prioritize certain relationship types
                    rel = edge_data.get('relationship', '')
                    if rel in ['assigns', 'drives', 'connects']:
                        edge_quality += 1.0
                    elif rel in ['contains', 'has_port']:
                        edge_quality += 0.8
                    else:
                        edge_quality += 0.5
                else:
                    edge_quality += 0.5

        # Normalize edge quality
        if edge_count > 0:
            edge_quality /= edge_count

        # Combine into overall quality score
        quality_score = diversity * 0.3 + importance * 0.3 + edge_quality * 0.4

        return {
            'length': length,
            'node_types': node_types,
            'diversity': diversity,
            'importance': importance,
            'edge_quality': edge_quality,
            'quality_score': quality_score,
        }

    def _describe_path(
        self,
        path: List[str],
        focus_node: str,
        discovered_signals: Set[str],
        metrics: Dict,
    ) -> str:
        """
        Generate a detailed description of a random walk path.

        Args:
            path: List of nodes in the path
            focus_node: The starting node
            discovered_signals: Set of signals discovered along this path
            metrics: Path metrics

        Returns:
            Textual description of the path
        """
        G = self.G

        # Get info about the focus node
        focus_info = G.nodes[focus_node]
        focus_type = focus_info.get('type', 'unknown')
        focus_name = focus_info.get('name', focus_node)
        focus_module = focus_info.get('module', 'unknown')

        # Get signal name
        focus_signal = self._get_signal_for_node(focus_node) or focus_name

        # Create header
        parts = [
            f"GUIDED RANDOM WALK FROM {focus_signal} ({focus_type})",
            f"Located in module: {focus_module}",
            f"Path length: {len(path)} nodes, discovered signals: {', '.join(discovered_signals)}",
            "",
            "Signal flow path:",
        ]

        # Enhanced path description
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

            # Get edge relationship
            edge_data = G.get_edge_data(curr_node, next_node)
            relationship = "→"
            if edge_data:
                if isinstance(edge_data, dict):
                    relationship = edge_data.get(
                        'relationship', edge_data.get('relation', '→')
                    )

            # Format node descriptions with improved clarity
            curr_desc = f"{curr_name}"
            if curr_type:
                curr_desc += f" ({curr_type}"
                if curr_module:
                    curr_desc += f" in {curr_module}"
                curr_desc += ")"

            next_desc = f"{next_name}"
            if next_type:
                next_desc += f" ({next_type}"
                if next_module:
                    next_desc += f" in {next_module}"
                next_desc += ")"

            # Use meaningful descriptions for relationships
            rel_description = self._get_relationship_description(relationship)

            # Add relationship to output with improved formatting and explanation
            parts.append(f"  {curr_desc} {rel_description} {next_desc}")

        # Add discovered signal details with improved formatting
        if discovered_signals:
            parts.append("\nDiscovered interface signals:")
            for signal in discovered_signals:
                nodes = self.signal_to_node_map.get(signal, [])
                if nodes:
                    node = nodes[0]  # Take first node for info
                    if node in G.nodes:
                        node_info = G.nodes[node]
                        node_type = node_info.get('type', 'unknown')
                        node_module = node_info.get('module', 'unknown')
                        parts.append(f"  - {signal} ({node_type} in {node_module})")
                else:
                    parts.append(f"  - {signal}")

        # Add path analysis section with better explanations
        parts.append("\nPath Analysis:")

        # Node type distribution with clearer format
        node_types = metrics.get('node_types', {})
        if node_types:
            type_parts = []
            for node_type, count in node_types.items():
                type_parts.append(f"'{node_type}': {count} nodes")
            parts.append(f"  Node type distribution: {{{', '.join(type_parts)}}}")

        # Check for specific patterns with more informative descriptions
        signal_nodes = [n for n in path if G.nodes[n].get('type') in ['port', 'signal']]
        if len(signal_nodes) > 1:
            parts.append(
                f"  Contains {len(signal_nodes)} signal nodes, suggesting signal propagation path"
            )

        register_nodes = [n for n in path if G.nodes[n].get('type') == 'register']
        if register_nodes:
            parts.append(
                f"  Contains {len(register_nodes)} register nodes, suggesting state-dependent behavior"
            )

        # Add quality assessment
        quality_score = metrics.get('quality_score', 0)
        if quality_score > 0.7:
            parts.append(
                "  This path has high verification value due to multiple signal interactions"
            )
        elif quality_score > 0.5:
            parts.append("  This path has moderate verification value")
        else:
            parts.append(
                "  This path provides basic context but may need supplementation"
            )

        # Add metrics display
        metrics_str = ", ".join(
            [
                f"{k}={v:.2f}" if isinstance(v, float) else f"{k}={v}"
                for k, v in metrics.items()
                if k not in ['node_types']
            ]
        )
        parts.append(f"(metrics: {metrics_str})")

        return "\n".join(parts)

    def _get_relationship_description(self, relationship):
        """
        Convert technical relationship identifiers into human-readable descriptions.

        Args:
            relationship: The relationship identifier from the edge

        Returns:
            Human-readable description of the relationship
        """
        relationship_map = {
            "→": "connects to",
            "assigns": "assigns to",
            "assigns_to": "assigns to",
            "connects": "connects to",
            "connects_to": "connects to",
            "drives": "drives",
            "part_of": "part of",
            "contains": "contains",
            "has_port": "has port",
            "includes": "includes",
            "instantiates": "instantiates",
            "used_in": "used in",
            "outputs": "outputs to",
            "inputs": "receives input from",
            "input_to": "input to",
            "outputs_to": "outputs to",
            "found_in": "found in",
            "involves": "involves",
            "targets": "targets",
            "contains_spec": "specified in",
            "contains_implementation": "implements",
            "→": "relates to",
        }

        return relationship_map.get(relationship, relationship)

    def _get_signal_for_node(self, node: str) -> Optional[str]:
        """
        Find the interface signal name for a node.

        Args:
            node: Node ID to lookup

        Returns:
            Signal name or None
        """
        for signal, nodes in self.signal_to_node_map.items():
            if node in nodes:
                return signal
        return None
