from typing import Dict, List, Literal, Optional, Tuple, Union
from dataclasses import dataclass

import networkx as nx
import numpy as np
import pandas as pd


@dataclass
class ErrorConfig:
    """Configuration for error distribution of a variable."""

    type: Literal["normal", "uniform"]
    params: Union[Tuple[float, float], Dict[str, float]]

    def __post_init__(self):
        """Validate and standardize parameters after initialization."""
        if self.type == "normal":
            if isinstance(self.params, tuple) and len(self.params) == 2:
                self.params = {"mean": self.params[0], "std": self.params[1]}
            elif not (
                isinstance(self.params, dict)
                and "mean" in self.params
                and "std" in self.params
            ):
                raise ValueError("Normal error requires 'mean' and 'std' parameters")

        elif self.type == "uniform":
            if isinstance(self.params, tuple) and len(self.params) == 2:
                self.params = {"a": self.params[0], "b": self.params[1]}
            elif not (
                isinstance(self.params, dict)
                and "a" in self.params
                and "b" in self.params
            ):
                raise ValueError("Uniform error requires 'a' and 'b' parameters")
        else:
            raise ValueError(
                f"Unsupported error type: {self.type}. Use 'normal' or 'uniform'"
            )

    def generate_error(self, size: int) -> np.ndarray:
        """Generate random errors based on the configured distribution."""
        if self.type == "normal":
            return np.random.normal(self.params["mean"], self.params["std"], size)
        elif self.type == "uniform":
            return np.random.uniform(self.params["a"], self.params["b"], size)
        raise ValueError(f"Unsupported error type: {self.type}")


class SyntheticDataGenerator:
    """
    Class for generating synthetic data with linear relationships between variables.

    This generator creates data based on a causal graph represented by an adjacency matrix,
    where each node's value is a linear combination of its parent nodes plus a specified error term.
    """

    def __init__(
        self,
        n_nodes: int,
        adjacency_matrix: Optional[np.ndarray] = None,
        error_configs: Optional[List[ErrorConfig]] = None,
        seed: Optional[int] = None,
    ):
        """
        Initialize the synthetic data generator.

        Args:
            n_nodes: Number of nodes (variables) in the system
            adjacency_matrix: Optional adjacency matrix where element (i,j) represents
                              the weight of the relationship from node i to node j.
                              If None, a random sparse matrix will be generated.
            error_configs: Optional list of error configurations for each node.
                           If None, default normal distributions will be used.
            seed: Random seed for reproducibility
        """
        self.n_nodes = n_nodes

        # Set random seed if provided
        if seed is not None:
            np.random.seed(seed)

        # Initialize adjacency matrix
        if adjacency_matrix is None:
            # Generate a random sparse adjacency matrix with ~20% of possible edges
            self.adjacency_matrix = np.zeros((n_nodes, n_nodes))
            for i in range(n_nodes):
                for j in range(i + 1, n_nodes):  # Only upper triangular to avoid cycles
                    if np.random.random() < 0.2:
                        self.adjacency_matrix[i, j] = np.random.uniform(-2, 2)
        else:
            if adjacency_matrix.shape != (n_nodes, n_nodes):
                raise ValueError(
                    f"Adjacency matrix shape {adjacency_matrix.shape} does not match n_nodes {n_nodes}"
                )
            self.adjacency_matrix = adjacency_matrix

        # Ensure no self-loops
        np.fill_diagonal(self.adjacency_matrix, 0)

        # Check for cycles in the graph
        if not self._is_dag():
            raise ValueError(
                "The adjacency matrix must represent a directed acyclic graph (DAG)"
            )

        # Initialize error configurations
        if error_configs is None:
            self.error_configs = [
                ErrorConfig("normal", {"mean": 0, "std": 1}) for _ in range(n_nodes)
            ]
        else:
            if len(error_configs) != n_nodes:
                raise ValueError(
                    f"Number of error configs ({len(error_configs)}) must match n_nodes ({n_nodes})"
                )
            self.error_configs = error_configs

        # Topological ordering of nodes for generation
        self.topo_order = self._topological_sort()

        # Initialize weights dict to store the actual coefficients used
        self.weights = {}

    def _is_dag(self) -> bool:
        """Check if the adjacency matrix represents a directed acyclic graph (DAG)."""
        # Create a directed graph
        G = nx.DiGraph(self.adjacency_matrix)

        # Check for cycles
        try:
            nx.find_cycle(G)
            return False
        except nx.NetworkXNoCycle:
            return True

    def _topological_sort(self) -> List[int]:
        """Perform topological sort on the graph to determine generation order."""
        G = nx.DiGraph(self.adjacency_matrix)
        return list(nx.topological_sort(G))

    def generate_data(self, n_samples: int) -> pd.DataFrame:
        """
        Generate synthetic data based on the defined linear relationships.

        Args:
            n_samples: Number of samples to generate

        Returns:
            DataFrame containing the generated data
        """
        # Initialize data array
        data = np.zeros((n_samples, self.n_nodes))

        # Generate data for each node in topological order
        for node in self.topo_order:
            # Get parent nodes (nodes that have an edge to this node)
            parents = np.where(self.adjacency_matrix[:, node] != 0)[0]

            # Initialize with error term
            data[:, node] = self.error_configs[node].generate_error(n_samples)

            # Add linear combination of parent values
            for parent in parents:
                weight = self.adjacency_matrix[parent, node]
                data[:, node] += weight * data[:, parent]

                # Store the weight in the weights dictionary
                if node not in self.weights:
                    self.weights[node] = {}
                self.weights[node][parent] = weight

        return pd.DataFrame(data)
