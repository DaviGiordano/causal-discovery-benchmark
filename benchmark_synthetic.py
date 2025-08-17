#!/usr/bin/env python3
"""
Synthetic Data Causal Discovery Benchmark Script

This script performs comprehensive benchmarking of causal discovery algorithms
on synthetic datasets with various configurations including bootstrap resampling.
"""
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import networkx as nx
import numpy as np
import pandas as pd
import yaml
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.graph.GraphNode import GraphNode

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

from src.causal_discovery.tetrad_algorithms import (
    BOSSTetrad,
    DAGMATetrad,
    DirectLiNGAMTetrad,
    FGESTetrad,
    GRASPTetrad,
    PCTetrad,
)
from src.data.acyclic_graph_generator_more_mechanisms import AcyclicGraphGenerator
from src.graph_aux import get_graph_skeleton
from src.logging_config import setup_logging
from src.metrics import Metrics


class SyntheticBenchmark:
    """Comprehensive benchmark for synthetic data causal discovery."""

    def __init__(self, results_file: str = "benchmark_results.csv"):
        self.results_file = results_file
        self.results = []

        # Setup logging
        setup_logging("benchmark.log")
        self.logger = logging.getLogger(__name__)

        # Algorithm mapping
        self.algorithm_classes = {
            "PC": PCTetrad,
            "FGES": FGESTetrad,
            "Direct LiNGAM": DirectLiNGAMTetrad,
            "DAGMA": DAGMATetrad,
            "GRASP": GRASPTetrad,
            "BOSS": BOSSTetrad,
        }

        # Load algorithm configurations
        self.algorithm_configs = self._load_algorithm_configs()

    def _load_algorithm_configs(self) -> Dict[str, Dict]:
        """Load algorithm configurations from YAML file."""
        config_path = "configs/algorithms.yaml"
        with open(config_path, "r") as f:
            configs = yaml.safe_load(f)

        # Map algorithm names to configurations
        algorithm_configs = {}
        for key, config in configs.items():
            algorithm_name = config["algorithm_name"]
            if algorithm_name == "pc_tetrad":
                algorithm_configs["PC"] = config
            elif algorithm_name == "fges_tetrad":
                algorithm_configs["FGES"] = config
            elif algorithm_name == "directlingam_tetrad":
                algorithm_configs["Direct LiNGAM"] = config
            elif algorithm_name == "dagma_tetrad":
                algorithm_configs["DAGMA"] = config
            elif algorithm_name == "grasp_tetrad":
                algorithm_configs["GRASP"] = config
            elif algorithm_name == "boss_tetrad":
                algorithm_configs["BOSS"] = config

        return algorithm_configs

    def _generate_dataset_configs(self) -> List[Dict]:
        """Generate all dataset configurations for the benchmark."""
        configs = []

        # Dataset parameters
        nodes_list = [10, 100, 1000]
        expected_degree_list = [2, 5, 10]
        mechanism_settings = [
            (["linear"], [1.0]),
            (["polynomial"], [1.0]),
            (["sigmoid_add"], [1.0]),
            (["nn"], [1.0]),
            (["binary"], [1.0]),
            (["linear", "binary"], [0.5, 0.5]),
            (["polynomial", "binary"], [0.5, 0.5]),
            (["sigmoid_add", "binary"], [0.5, 0.5]),
            (["nn", "binary"], [0.5, 0.5]),
        ]

        for nodes in nodes_list:
            for expected_degree in expected_degree_list:
                for mech_pool, mech_prob in mechanism_settings:
                    gen_params = {
                        "nodes": nodes,
                        "npoints": 1000,
                        "parents_max": -1,
                        "expected_degree": expected_degree,
                        "dag_type": "erdos",
                        "noise_kind": "gaussian",
                        "noise_coeff": 0.4,
                        "mechanism_pool": tuple(mech_pool),
                        "mechanism_prob": tuple(mech_prob),
                        "root_pool": ("gmm",),
                        "root_prob": (1.0,),
                        "force_binary_root": False,
                        "binary_root_p": 0,
                        "random_state": 42,
                    }

                    gen_name = (
                        f"erdos_n{nodes}_d{expected_degree}_"
                        f"rootgmm_mech{'-'.join(mech_pool)}_noisegaussian"
                    )

                    configs.append(
                        {
                            "gen_name": gen_name,
                            "gen_params": gen_params,
                        }
                    )

        return configs

    def _run_single_benchmark(
        self, dataset_config: Dict, algorithms: List[str]
    ) -> List[Dict]:
        """Run benchmark for a single dataset configuration using Tetrad's internal bootstrap."""
        self.logger.info(f"Generating dataset: {dataset_config['gen_name']}")

        # Generate dataset
        generator = AcyclicGraphGenerator(**dataset_config["gen_params"])
        data, graph, metadata = generator.generate(return_metadata=True)
        # Verify graph is DAG using networkx
        # Get true graph and edges
        true_graph_dict = metadata.to_dict()
        true_adjacency = true_graph_dict["adjacency_matrix"]

        # Convert adjacency matrix to graph format for metrics
        true_graph_obj = GeneralGraph([])
        n_nodes = true_adjacency.shape[0]

        # Add nodes
        for i in range(n_nodes):
            node = GraphNode(f"X{i+1}")
            true_graph_obj.add_node(node)

        # Add edges based on adjacency matrix
        for i in range(n_nodes):
            for j in range(n_nodes):
                if true_adjacency[i, j] == 1:
                    from causallearn.graph.Edge import Edge
                    from causallearn.graph.Endpoint import Endpoint

                    edge = Edge(
                        true_graph_obj.get_node(f"X{i+1}"),
                        true_graph_obj.get_node(f"X{j+1}"),
                        Endpoint.ARROW,
                        Endpoint.TAIL,
                    )
                    true_graph_obj.add_edge(edge)

        # Create true edges dictionary
        true_edges_dict = {}
        for i in range(n_nodes):
            for j in range(n_nodes):
                if i != j:
                    edge_key = f"X{i+1}---X{j+1}"
                    if true_adjacency[i, j] == 1:
                        true_edges_dict[edge_key] = "source->target"
                    else:
                        true_edges_dict[edge_key] = "no_edge"

        results = []

        # Run each algorithm once - Tetrad will handle bootstrap internally
        for algorithm_name in algorithms:
            self.logger.info(f"  Running {algorithm_name}")

            result = self._run_algorithm_on_data(
                algorithm_name, data, true_graph_obj, true_edges_dict
            )

            # Add metadata
            result.update(
                {
                    "dataset_name": dataset_config["gen_name"],
                    "nodes": dataset_config["gen_params"]["nodes"],
                    "expected_degree": dataset_config["gen_params"]["expected_degree"],
                    "mechanism": "-".join(
                        dataset_config["gen_params"]["mechanism_pool"]
                    ),
                    "bootstrap_idx": 0,  # Single run since Tetrad handles bootstrap
                    "timestamp": datetime.now().isoformat(),
                }
            )

            results.append(result)

            # Save results incrementally
            self._save_results_incrementally(result)

        return results

    def _run_algorithm_on_data(
        self, algorithm_name: str, data: np.ndarray, true_graph, true_edges_dict: Dict
    ) -> Dict[str, Any]:
        """Run a single algorithm on a dataset and compute metrics."""
        training_time = -1

        # Get algorithm configuration
        config = self.algorithm_configs[algorithm_name]
        algorithm_class = self.algorithm_classes[algorithm_name]

        # Initialize algorithm
        algorithm = algorithm_class(config)

        # Train algorithm
        start_time = time.time()
        algorithm.train(data)
        training_time = time.time() - start_time

        # Compute metrics
        metrics = Metrics(
            training_time=training_time,
            true_graph=true_graph,
            est_graph=algorithm.est_graph,
            est_edges_dict=algorithm.est_edges_dict,
            true_edges_dict=true_edges_dict,
            edge_probabilities=algorithm.edge_probabilities,
        )

        result_metrics = metrics.get_result_metrics()

        return {
            "algorithm": algorithm_name,
            "training_time": training_time,
            "skeleton_shd": result_metrics["skeleton_shd"],
            "skeleton_precision": result_metrics["adjacency"]["precision"],
            "skeleton_recall": result_metrics["adjacency"]["recall"],
            "skeleton_f1": result_metrics["adjacency"]["f1"],
            "shd": result_metrics["shd"],
            "precision": result_metrics["arrow"]["precision"],
            "recall": result_metrics["arrow"]["recall"],
            "f1": result_metrics["arrow"]["f1"],
            "skeleton_confidence": result_metrics["average_frequency"],
            "confidence": result_metrics["average_edge_frequency"],
            "expected_calibration_error": result_metrics["ece"],
            "brier_score": result_metrics["brier"],
            "success": True,
            "error": "",
        }

    def _run_bootstrap_benchmark(
        self, dataset_config: Dict, algorithms: List[str]
    ) -> List[Dict]:
        """Run benchmark for a single dataset configuration using Tetrad's internal bootstrap."""
        return self._run_single_benchmark(dataset_config, algorithms)

    def _save_results_incrementally(self, result: Dict):
        """Save a single result to CSV file incrementally."""
        df_result = pd.DataFrame([result])

        if os.path.exists(self.results_file):
            # Append to existing file
            df_result.to_csv(self.results_file, mode="a", header=False, index=False)
        else:
            # Create new file
            df_result.to_csv(self.results_file, index=False)

    def run_benchmark(self):
        """Run the complete benchmark."""
        self.logger.info("Starting synthetic data causal discovery benchmark")

        # Dataset configurations
        dataset_configs = self._generate_dataset_configs()

        # Algorithms to test
        algorithms = ["PC", "FGES", "Direct LiNGAM", "DAGMA", "GRASP", "BOSS"]

        self.logger.info(f"Total dataset configurations: {len(dataset_configs)}")
        self.logger.info(f"Algorithms: {algorithms}")
        self.logger.info(f"Results will be saved to: {self.results_file}")

        # Run benchmark for each dataset configuration
        for i, dataset_config in enumerate(dataset_configs):
            self.logger.info(f"Processing dataset {i+1}/{len(dataset_configs)}")

            try:
                results = self._run_bootstrap_benchmark(dataset_config, algorithms)
                self.results.extend(results)

                self.logger.info(f"Completed dataset {i+1}/{len(dataset_configs)}")

            except Exception as e:
                self.logger.error(
                    f"Error processing dataset {dataset_config['gen_name']}: {str(e)}"
                )
                continue

        self.logger.info("Benchmark completed!")
        self.logger.info(f"Results saved to: {self.results_file}")

        # Print summary statistics
        self._print_summary()

    def _print_summary(self):
        """Print summary statistics of the benchmark results."""
        if not os.path.exists(self.results_file):
            self.logger.warning("No results file found for summary")
            return

        df = pd.read_csv(self.results_file)


def main():
    """Main function to run the benchmark."""
    # Set random seed for reproducibility
    np.random.seed(42)

    # Create benchmark instance
    benchmark = SyntheticBenchmark()

    # Run benchmark
    benchmark.run_benchmark()


if __name__ == "__main__":
    main()
