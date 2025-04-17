from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode
from causallearn.graph.Node import Node
import re


def str_to_edge_probabilities(graph_str: str) -> GeneralGraph:
    """
    Parse a tetrad graph string into a dictionary structure with edge probabilities.

    Args:
        graph_str (str): String representation of the tetrad graph

    Returns:
        dict: Dictionary with node tuple keys and probability dictionary values
        dict: Dictionary of nodes present in the graph
    """

    def _extract_nodes(graph_str: str):
        sections = graph_str.split("\n\n")
        graph_nodes_section = sections[0].replace("Graph Nodes:\n", "")
        return [node for node in graph_nodes_section.split(";") if node]

    def _initialize_edge_probs(nodes):
        # Initialize all possible edges probabilities
        edge_probs = {}
        for i, source in enumerate(nodes):
            for target in nodes[
                i + 1 :
            ]:  # Only consider unique pairs: (X1,X3) but not (X3,X1)
                edge_probs[(source, target)] = {
                    "no_edge": 1.0,
                    "undirected": 0.0,
                    "source->target": 0.0,
                    "target->source": 0.0,
                    "edge": 0.0,
                }
        return edge_probs

    nodes = _extract_nodes(graph_str)
    edge_probs = _initialize_edge_probs(nodes)

    edges_section = graph_str.split("\n\n")[1].replace("Graph Edges:\n", "")

    edge_lines = edges_section.split("\n")
    for line in edge_lines:
        edge_info = line.split(". ", 1)[1]
        edge_components = edge_info.split(" ", 3)
        source = edge_components[0]
        target = edge_components[2]

        try:
            if int(source[1:]) > int(target[1:]):
                source, target = target, source
        except ValueError:
            raise ValueError(
                f"Could not convert node indices to integers: {source[1:]}, {target[1:]}"
            )

        edge_key = (source, target)

        prob_parts = edge_info.split(";")
        for part in prob_parts:
            # Find the probability value after the colon
            match = re.search(r":([0-9.]+)", part)
            if not match:
                continue

            prob_value = float(match.group(1))
            if "[no edge]" in part:
                edge_probs[edge_key]["no_edge"] = prob_value
            elif "[edge]" in part:
                edge_probs[edge_key]["edge"] = prob_value
                edge_probs[edge_key]["no_edge"] = round(1 - prob_value, 4)
            elif (f"{source} --- {target}" in part) or (
                f"{target} --- {source}" in part
            ):
                edge_probs[edge_key]["undirected"] = prob_value
            elif f"{source} --> {target}" in part:
                edge_probs[edge_key]["source->target"] += prob_value
            elif f"{source} <-- {target}" in part:
                edge_probs[edge_key]["target->source"] += prob_value
            elif f"{target} --> {source}" in part:
                edge_probs[edge_key]["target->source"] += prob_value
            elif f"{target} <-- {source}" in part:
                edge_probs[edge_key]["source->target"] += prob_value

    return edge_probs


def str_to_edge_dict(graph_string: str) -> dict:
    """
    Extracts and processes edge information from a graph string representation to determine edge types between nodes.

    This function parses a string representation of a graph that contains node and edge information,
    and returns a dictionary mapping node pairs to their edge types/directions.
    """

    def _extract_nodes(graph_string: str):
        sections = graph_string.split("\n\n")
        graph_nodes_section = sections[0].replace("Graph Nodes:\n", "")
        return [node for node in graph_nodes_section.split(";") if node]

    def _initialize_edge_types(nodes):
        # Initialize all possible edges probabilities
        edge_types = {}
        for i, source in enumerate(nodes):
            for target in nodes[
                i + 1 :
            ]:  # Only consider unique pairs: (X1,X3) but not (X3,X1)
                edge_types[(source, target)] = "no_edge"
        return edge_types

    nodes = _extract_nodes(graph_string)
    chosen_edges = _initialize_edge_types(nodes)

    edges_section = graph_string.split("\n\n")[1].replace("Graph Edges:\n", "")

    edge_lines = edges_section.split("\n")
    for line in edge_lines:
        edge_info = line.split(" [", 1)[0][3:].strip()
        edge_components = edge_info.split(" ", 3)
        source = edge_components[0]
        connection = edge_components[1]
        target = edge_components[2]

        try:
            if int(source[1:]) > int(target[1:]):
                source, target = target, source
        except ValueError:
            raise ValueError(
                f"Could not convert node indices to integers: {source[1:]}, {target[1:]}"
            )

        edge_key = (source, target)

        if (f"{source} --- {target}" in edge_info) or (
            f"{target} --- {source}" in edge_info
        ):
            chosen_edges[edge_key] = "undirected"
        elif f"{source} --> {target}" in edge_info:
            chosen_edges[edge_key] = "source->target"
        elif f"{source} <-- {target}" in edge_info:
            chosen_edges[edge_key] = "target->source"
        elif f"{target} --> {source}" in edge_info:
            chosen_edges[edge_key] = "target->source"
        elif f"{target} <-- {source}" in edge_info:
            chosen_edges[edge_key] = "source->target"
        elif f"{source} ... {target}" in edge_info:
            chosen_edges[edge_key] = "no_edge"
        elif f"{target} ... {source}" in edge_info:
            chosen_edges[edge_key] = "no_edge"

    return chosen_edges


def str_to_general_graph(graph_str: str) -> GeneralGraph:
    """Convert a string representation to a GeneralGraph object.
    Function from causal-learn library with a filter to exclude edges

    Args:
        graph_str: String representation of a graph in the same format as expected in txt2generalgraph

    Returns:
        A GeneralGraph object constructed from the string
    """

    def _to_endpoint(s: str) -> Endpoint:
        if s == "o":
            return Endpoint.CIRCLE
        elif s == ">":
            return Endpoint.ARROW
        elif s == "-":
            return Endpoint.TAIL
        else:
            print(f"Invalid endpoint type: {s}"), NotImplementedError

    def _mod_endpoint(edge: Edge, z: Node, end: Endpoint):
        if edge.get_node1() == z:
            edge.set_endpoint1(end)
        elif edge.get_node2() == z:
            edge.set_endpoint2(end)
        else:
            raise ValueError("z not in edge")

    g = GeneralGraph([])
    node_map = {}

    lines = graph_str.strip().split("\n")
    next_nodes_line = False

    for line in lines:
        line = line.strip()
        words = line.split()

        if len(words) > 1 and words[1] == "Nodes:":
            next_nodes_line = True
        elif len(line) > 0 and next_nodes_line:
            next_nodes_line = False
            nodes = line.split(";")
            for node in nodes:
                node_map[node] = GraphNode(node)
                g.add_node(node_map[node])
        elif len(words) > 0 and words[0][-1] == ".":
            next_nodes_line = False
            node1 = words[1]
            node2 = words[3]
            end1 = words[2][0]
            end2 = words[2][-1]
            if end1 != "." and end2 != ".":  # Added filter of no-edges
                if end1 == "<":
                    end1 = ">"
                end1 = _to_endpoint(end1)
                end2 = _to_endpoint(end2)
                edge = Edge(
                    node_map[node1],
                    node_map[node2],
                    Endpoint.CIRCLE,
                    Endpoint.CIRCLE,
                )
                _mod_endpoint(edge, node_map[node1], end1)
                _mod_endpoint(edge, node_map[node2], end2)
                g.add_edge(edge)

    return g


def txt_to_general_graph(filename: str) -> GeneralGraph:
    """Convert a text file representation to a GeneralGraph object.

    Args:
        filename: Path to file containing graph representation

    Returns:
        A GeneralGraph object constructed from the file content
    """
    with open(filename, "r") as file:
        graph_str = file.read()

    return str_to_general_graph(graph_str)
