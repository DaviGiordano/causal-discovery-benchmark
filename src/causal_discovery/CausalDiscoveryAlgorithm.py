from causallearn.graph.Edge import Edge
from causallearn.graph.Endpoint import Endpoint
import numpy as np
from abc import ABC, abstractmethod
import logging
from typing import Dict

from causallearn.graph.GeneralGraph import GeneralGraph
from src.graph_aux import get_edge_adjacency_matrix, get_graph_skeleton
from src.logging_config import setup_logging
from src.metrics import Metrics
from src.visualization import Plotter


class CausalDiscoveryAlgorithm(ABC):
    def __init__(self, config_params):
        self.config_params = config_params

        # Not all graphs may have this. I'll leave unset
        self.est_adj: np.ndarray = np.ndarray([])
        self.est_edge_adj: np.ndarray = np.ndarray([])

        self.est_graph: GeneralGraph = GeneralGraph([])
        self.est_graph_skeleton: GeneralGraph = GeneralGraph([])
        self.is_trained = False

        self.graph_string: str = ""
        self.edge_probabilities: dict = {}
        self.est_edges_dict: dict = {}
        self.est_dotgraph: str = ""

    @abstractmethod
    def train(self, data: np.ndarray) -> None:
        """Learns causal structure from data"""
        pass

    def _set_auxiliary_results(self):

        self.est_edge_adj = get_edge_adjacency_matrix(graph=self.est_graph)
        self.est_graph_skeleton = get_graph_skeleton(graph=self.est_graph)
        self.is_trained = True
