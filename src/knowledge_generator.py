"""
Knowledge Generator Module

This module provides functionality to generate temporal knowledge constraints
for causal discovery algorithms based on the true graph structure.
"""

import logging

import networkx as nx

logger = logging.getLogger(__name__)


def generate_temporal_knowledge(true_graph: nx.DiGraph, num_tiers: int) -> str:
    """
    Generate temporal knowledge file content in Tetrad format based on the true graph structure.

    Parameters:
    -----------
    true_graph : nx.DiGraph
        NetworkX DiGraph representing the true DAG
    num_tiers : int
        Number of temporal tiers to create

    Returns:
    --------
    str
        String containing knowledge file content in Tetrad format

    Raises:
    -------
    ValueError
        If num_tiers > number_of_variables or num_tiers <= 0
    """
    # Validate input parameters
    num_variables = len(true_graph.nodes())

    if num_tiers <= 0:
        raise ValueError(f"num_tiers must be positive, got {num_tiers}")

    if num_tiers > num_variables:
        raise ValueError(
            f"num_tiers ({num_tiers}) cannot be greater than number of variables ({num_variables})"
        )

    # Get topological sort of the graph for temporal ordering
    try:
        topological_order = list(nx.topological_sort(true_graph))
    except nx.NetworkXError:
        # If graph has cycles, use node order as fallback
        raise ValueError("Graph contains cycles, cannot generate temporal knowledge")

    # Distribute variables evenly across tiers
    base_size = num_variables // num_tiers
    remainder = num_variables % num_tiers

    # Build tier assignments
    tiers = []
    current_index = 0

    for tier_num in range(num_tiers):
        # First 'remainder' tiers get one extra variable each
        tier_size = base_size + (1 if tier_num < remainder else 0)
        tier_variables = topological_order[current_index : current_index + tier_size]
        tiers.append(tier_variables)
        current_index += tier_size

    # Generate Tetrad knowledge file format
    knowledge_content = "/knowledge\n"
    knowledge_content += "addtemporal\n\n"

    # Add temporal tiers
    for tier_num, tier_variables in enumerate(tiers, 1):
        tier_str = " ".join(str(var) for var in tier_variables)
        knowledge_content += f"{tier_num}  {tier_str}\n"

    # Add empty sections for forbidden and required direct effects
    knowledge_content += "\nforbiddirect\n\n"
    knowledge_content += "requiredirect\n"

    logger.info(f"Generated temporal knowledge with {num_tiers} tiers")
    logger.info(f"Tier distribution: {[len(tier) for tier in tiers]}")

    return knowledge_content


def save_knowledge_file(knowledge_content: str, output_path: str) -> None:
    """
    Save knowledge content to a file at the specified path.

    Parameters:
    -----------
    knowledge_content : str
        String with knowledge file content
    output_path : str
        Path to save the knowledge file

    Returns:
    --------
    None
    """
    try:
        with open(output_path, "w") as f:
            f.write(knowledge_content)
        logger.info(f"Knowledge file saved to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save knowledge file to {output_path}: {e}")
        raise
