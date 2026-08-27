# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.

from config import FLAGS
import numpy as np
import networkx as nx
from saver import saver

# from networkx.algorithms import isomorphism  # instead of vf2graph_matcher
from typing import List, Dict, Set, Tuple, Optional
from dataclasses import dataclass

print = saver.log_info


class KGTraversal:
    def __init__(self, kg):
        self.kg = kg
        self._build_graph()

    def _build_graph(self):
        self.graph = nx.Graph()
        for node in self.kg['nodes']:
            self.graph.add_node(node['id'], **node['attributes'])
        for edge in self.kg['edges']:
            self.graph.add_edge(edge['source'], edge['target'], **edge['attributes'])

    def traverse(self, start_node, max_depth=2):
        visited_nodes = set()
        visited_edges = set()
        result_nodes = []
        result_edges = []
        self._dfs(
            start_node,
            max_depth,
            0,
            visited_nodes,
            visited_edges,
            result_nodes,
            result_edges,
        )
        return result_nodes, result_edges

    def _dfs(
        self,
        node,
        max_depth,
        current_depth,
        visited_nodes,
        visited_edges,
        result_nodes,
        result_edges,
    ):
        if current_depth > max_depth or node in visited_nodes:
            return
        visited_nodes.add(node)
        result_nodes.append(node)
        for neighbor in self.graph.neighbors(node):
            if (node, neighbor) not in visited_edges and (
                neighbor,
                node,
            ) not in visited_edges:
                visited_edges.add((node, neighbor))
                result_edges.append((node, neighbor, self.graph[node][neighbor]))
            self._dfs(
                neighbor,
                max_depth,
                current_depth + 1,
                visited_nodes,
                visited_edges,
                result_nodes,
                result_edges,
            )

    def get_node_info(self, node_id):
        if node_id in self.graph:
            return {'id': node_id, 'attributes': self.graph.nodes[node_id]}
        return None

    def get_edge_info(self, source, target):
        if self.graph.has_edge(source, target):
            return {
                'source': source,
                'target': target,
                'attributes': self.graph[source][target],
            }
        return None
