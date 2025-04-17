from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.Edge import Edge
from causallearn.graph.Node import Node

from typing import List, Optional, Dict, Tuple
import pydot
import numpy as np


def dag_adj_to_graph(
    dag_adj: np.ndarray,
    adj_type: str = "upper_triangular",
) -> GeneralGraph:
    nodes = list[Node]([GraphNode(f"X{i}") for i in range(1, dag_adj.shape[0] + 1)])
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


def to_pydot_label_edges(
    G: GeneralGraph,
    edges: Optional[List["Edge"]] = None,
    labels: Optional[List[str]] = None,
    title: str = "",
    dpi: float = 200,
    # Generic edge_labels dict: (node1_id, node2_id) -> str (the label)
    edge_labels: Optional[Dict[Tuple[int, int], str]] = None,
) -> pydot.Dot:
    """
    Convert a Graph (from causal-learn) into a pydot Dot object.

    Parameters
    ----------
    G : Graph
        A graph object from causal-learn
    edges : list of Edge, optional
        If not provided, will use G.get_graph_edges()
    labels : list of str, optional
        Node labels to override Graph node names
    title : str, optional
        Title (name) of the resulting pydot graph
    dpi : float, optional
        The dots-per-inch setting for the figure
    edge_labels : dict, optional
        A mapping from (node1_id, node2_id) to a string label.
        For instance, "3.5" or "2" or "80%"—whatever you want to show on that edge.

    Returns
    -------
    pydot_g : pydot.Dot
        A DOT-format pydot graph object.
    """

    # Gather the nodes
    nodes = G.get_nodes()
    if labels is not None:
        assert len(labels) == len(nodes), (
            f"Number of labels ({len(labels)}) does not match "
            f"number of nodes ({len(nodes)})."
        )

    # Create the pydot graph
    pydot_g = pydot.Dot(title, graph_type="digraph", fontsize="18")
    pydot_g.set("dpi", str(dpi))

    # Add pydot Nodes
    for i, node in enumerate(nodes):
        # If we have override labels, use them; otherwise use node.get_name()
        node_label = labels[i] if labels else node.get_name()
        shape = "square" if node.get_node_type().name == "LATENT" else "ellipse"
        pydot_node = pydot.Node(i, label=node_label, shape=shape)
        pydot_g.add_node(pydot_node)

    # If user didn't pass edges, get them from the Graph
    if edges is None:
        edges = G.get_graph_edges()

    # A helper to convert causal-learn's endpoints into pydot edge arrow styles
    def get_g_arrow_type(endpoint: "Endpoint") -> str:
        if endpoint.name == "TAIL":
            return "none"
        elif endpoint.name == "ARROW":
            return "normal"
        elif endpoint.name == "CIRCLE":
            return "odot"
        else:
            raise NotImplementedError(f"Unsupported endpoint: {endpoint}")

    # Build pydot Edges
    for edge in edges:
        node1 = edge.get_node1()
        node2 = edge.get_node2()

        # Convert from node to index
        node1_id = nodes.index(node1)
        node2_id = nodes.index(node2)

        # Build the pydot edge
        dot_edge = pydot.Edge(
            node1_id,
            node2_id,
            dir="both",  # We let arrowtail and arrowhead define directions
            arrowtail=get_g_arrow_type(edge.get_endpoint1()),
            arrowhead=get_g_arrow_type(edge.get_endpoint2()),
        )

        # Example: highlight special properties
        if "dd" in edge.properties:  # Possibly Edge.Property.dd
            dot_edge.set_color("green3")
        if "nl" in edge.properties:  # Possibly Edge.Property.nl
            dot_edge.set_penwidth("2.0")

        # If we have a label for this edge, attach it
        if edge_labels is not None:
            # Because the edges might be i->j or j->i, do the lookups consistently
            # If your graph is undirected, you might store label as (min, max).
            if (node1_id, node2_id) in edge_labels:
                dot_edge.set_label(edge_labels[(node1_id, node2_id)])
            elif (node2_id, node1_id) in edge_labels:
                dot_edge.set_label(edge_labels[(node2_id, node1_id)])

        pydot_g.add_edge(dot_edge)

    return pydot_g
