from typing import Dict, List, Optional, Tuple

import networkx as nx
import numpy as np
import pydot
from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node


def dag_adj_to_graph(
    dag_adj: np.ndarray,
    adj_type: str = "upper_triangular",
) -> GeneralGraph:
    """Converts an adjacency matrix to a causallearn.GeneralGraph"""

    nodes = list[Node]([GraphNode(f"X{i}") for i in range(dag_adj.shape[0])])
    graph = GeneralGraph(nodes)

    if adj_type == "lower_triangular":
        for i in range(dag_adj.shape[0]):
            for j in range(dag_adj.shape[1]):
                if dag_adj[i, j] != 0 and not np.isnan(dag_adj[i, j]):
                    graph.add_directed_edge(graph.nodes[j], graph.nodes[i])
    elif adj_type == "upper_triangular":
        for i in range(dag_adj.shape[0]):
            for j in range(dag_adj.shape[1]):
                if dag_adj[i, j] != 0 and not np.isnan(dag_adj[i, j]):
                    graph.add_directed_edge(graph.nodes[i], graph.nodes[j])

    return graph


def get_edge_adjacency_matrix(graph: GeneralGraph) -> np.ndarray:
    """
    Creates a symmetrical adjacency matrix from a graph, ignoring edge directions.

    Parameters:
    -----------
    graph : GeneralGraph
        The input graph

    Returns:
    --------
    np.ndarray
        A symmetrical adjacency matrix where 1 indicates an edge between nodes
    """
    nodes = graph.get_nodes().copy()
    n = len(nodes)
    edge_adj_matrix = np.zeros((n, n), dtype=int)

    # For each pair of nodes, set matrix values to 1 if they're adjacent
    for i in range(n):
        for j in range(n):
            if graph.is_adjacent_to(nodes[i], nodes[j]):
                edge_adj_matrix[i, j] = 1
                edge_adj_matrix[j, i] = 1  # Make it symmetrical

    return edge_adj_matrix


def get_graph_skeleton(graph: GeneralGraph) -> GeneralGraph:
    """
    Creates and returns the skeleton (undirected graph) from a GeneralGraph.

    Parameters:
    -----------
    graph : GeneralGraph
        The original graph

    Returns:
    --------
    GeneralGraph
        The skeleton with only undirected edges
    """
    # Create a new graph with the same nodes
    nodes = graph.get_nodes().copy()
    skeleton = GeneralGraph(nodes)

    # For each pair of nodes, add an undirected edge if they're adjacent in the original graph
    for i in range(len(nodes)):
        for j in range(i + 1, len(nodes)):
            if graph.is_adjacent_to(nodes[i], nodes[j]):
                # Create an undirected edge directly
                edge = Edge(nodes[i], nodes[j], Endpoint.TAIL, Endpoint.TAIL)
                skeleton.add_edge(edge)

    return skeleton


def networkx_to_edge_dict(graph: nx.DiGraph) -> dict:
    nodes = sorted([str(node) for node in graph.nodes()])
    edge_dict = {}
    for i, source in enumerate(nodes):
        for j, target in enumerate(nodes[i + 1 :]):
            edge_key = (f"{source}", f"{target}")
            if graph.has_edge(source, target):
                edge_dict[edge_key] = "source->target"
            elif graph.has_edge(target, source):
                edge_dict[edge_key] = "target->source"
            else:
                edge_dict[edge_key] = "no_edge"

    return edge_dict
