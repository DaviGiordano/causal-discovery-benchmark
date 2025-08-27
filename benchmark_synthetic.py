# Minimal script for testing synthetic data generation and causal discovery using PCTetrad
import argparse
import fcntl
import json
import logging
import os
import pathlib
import sys
import time

import numpy as np
import pandas as pd
import yaml

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
from src.graph_aux import dag_adj_to_graph, networkx_to_edge_dict
from src.logging_config import setup_logging
from src.metrics import Metrics, edge_confusion_matrix


def load_configs():
    """Load generation and algorithm configurations from YAML files."""
    # Load generation configurations
    with open("configs/generation.yaml", "r") as f:
        generation_configs = yaml.safe_load(f)

    # Load algorithm configurations
    with open("configs/algorithms.yaml", "r") as f:
        algorithm_configs = yaml.safe_load(f)

    return generation_configs, algorithm_configs


def get_algorithm_class(algorithm_name):
    """Get the algorithm class based on the algorithm name."""
    algorithm_mapping = {
        "pc_tetrad": PCTetrad,
        "fges_tetrad": FGESTetrad,
        "directlingam_tetrad": DirectLiNGAMTetrad,
        "dagma_tetrad": DAGMATetrad,
        "grasp_tetrad": GRASPTetrad,
        "boss_tetrad": BOSSTetrad,
    }

    # Extract base algorithm name (remove suffix like _pd2)
    base_name = algorithm_name.split("_")[0] + "_tetrad"

    if base_name in algorithm_mapping:
        return algorithm_mapping[base_name]
    else:
        raise ValueError(f"Unknown algorithm: {algorithm_name}")


def save_generated_data(data, graph, metadata, gen_params, output_path):
    """Save generated data, graph, and metadata to files if they don't exist."""
    data_csv_path = output_path / "data.csv"
    # Save the generated data as CSV
    if hasattr(data, "to_csv"):
        data.to_csv(data_csv_path, index=False)
    else:
        # Convert numpy array to DataFrame and save
        data_np = np.asarray(data)
        df_data = pd.DataFrame(data_np)
        df_data.to_csv(data_csv_path, index=False)

    # Save graph edges as CSV
    graph_csv_path = output_path / "graph_edges.csv"
    edges_list = list(graph.edges)
    df_edges = pd.DataFrame(edges_list, columns=["source", "target"])
    df_edges.to_csv(graph_csv_path, index=False)

    # Save metadata as JSON
    metadata_path = output_path / "metadata.json"
    metadata_dict = {
        "mechanisms": metadata.mechanisms,
        "variable_types": metadata.variable_types,
        "binary_root": metadata.binary_root,
        "generation_params": gen_params,
        "adjacency_matrix": metadata.adjacency_matrix.tolist(),
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata_dict, f, indent=2)

    if logging.getLogger().isEnabledFor(logging.INFO):
        logging.info(f"Saved generated data to {data_csv_path}")
        logging.info(f"Saved graph edges to {graph_csv_path}")
        logging.info(f"Saved metadata to {metadata_path}")
    else:
        print(f"Saved generated data to {data_csv_path}")
        print(f"Saved graph edges to {graph_csv_path}")
        print(f"Saved metadata to {metadata_path}")


def save_experiment_files(
    output_path,
    generated_graph,
    true_edges_dict,
    est_edges_dict,
    edge_probabilities,
    metrics_results,
):
    """Save experiment results to JSON files."""
    # Convert edge dictionaries to string keys for JSON serialization
    generated_graph_str = {"graph": [str(edge) for edge in list(generated_graph.edges)]}
    true_edges_str = {str(k): v for k, v in true_edges_dict.items()}
    est_edges_str = {str(k): v for k, v in est_edges_dict.items()}
    edge_probs_str = (
        {str(k): v for k, v in edge_probabilities.items()} if edge_probabilities else {}
    )
    cm = edge_confusion_matrix(est_edges_dict, true_edges_dict)

    # Save files
    with open(output_path / "generated_graph.json", "w") as f:
        json.dump(generated_graph_str, f, indent=2)

    with open(output_path / "true_edges_dict.json", "w") as f:
        json.dump(true_edges_str, f, indent=2)

    with open(output_path / "est_edges_dict.json", "w") as f:
        json.dump(est_edges_str, f, indent=2)

    with open(output_path / "edge_probabilities.json", "w") as f:
        json.dump(edge_probs_str, f, indent=2)

    with open(output_path / "metrics_results.json", "w") as f:
        json.dump(metrics_results, f, indent=2)

    cm.to_csv(output_path / "edge_confusion_matrix.csv", index=True)


def append_to_csv(csv_path, result_data):
    """Append experiment result to CSV file with file locking for concurrent access."""
    # Convert result data to DataFrame
    df_result = pd.DataFrame([result_data])

    # Use file locking for thread-safe appending
    with open(csv_path, "a") as f:
        fcntl.flock(f.fileno(), fcntl.LOCK_EX)  # Exclusive lock
        try:
            # Check if file is empty to write headers
            f.seek(0, 2)  # Seek to end
            if f.tell() == 0:
                df_result.to_csv(f, index=False)
            else:
                df_result.to_csv(f, header=False, index=False)
        finally:
            fcntl.flock(f.fileno(), fcntl.LOCK_UN)  # Release lock


def run_single_experiment(
    generation_name,
    algorithm_name,
    generation_configs,
    algorithm_configs,
    output_path=None,
    csv_path=None,
):
    """Run a single experiment with given generation and algorithm configurations."""
    if output_path:
        logging.info(f"Running experiment: {generation_name} with {algorithm_name}")
    else:
        print(f"\n{'='*60}")
        print(f"Running experiment: {generation_name} with {algorithm_name}")
        print(f"{'='*60}")

    # Get configurations
    if generation_name not in generation_configs:
        raise ValueError(f"Generation configuration '{generation_name}' not found")

    if algorithm_name not in algorithm_configs:
        raise ValueError(f"Algorithm configuration '{algorithm_name}' not found")

    gen_params = generation_configs[generation_name]
    alg_config = algorithm_configs[algorithm_name]

    # Convert lists to tuples for the generator
    gen_params["mechanism_pool"] = tuple(gen_params["mechanism_pool"])
    gen_params["mechanism_prob"] = tuple(gen_params["mechanism_prob"])
    gen_params["root_pool"] = tuple(gen_params["root_pool"])
    gen_params["root_prob"] = tuple(gen_params["root_prob"])

    # 1. Generate dataset
    gen = AcyclicGraphGenerator(**gen_params)
    data, graph, metadata = gen.generate(return_metadata=True)  # type: ignore

    # Save generated data, graph, and metadata if output path is specified and data.csv doesn't exist
    if output_path:
        save_generated_data(data, graph, metadata, gen_params, output_path)

    # 2. Create true graph from adjacency matrix
    true_adj = metadata.adjacency_matrix
    true_graph = dag_adj_to_graph(true_adj, "line_to_column")
    true_edges_dict = networkx_to_edge_dict(graph)

    # 3. Run the algorithm
    algorithm_class = get_algorithm_class(alg_config["algorithm_name"])
    algorithm = algorithm_class(alg_config)

    # Train model and measure time
    start_time = time.time()
    algorithm.train(data)
    training_time = time.time() - start_time

    # 4. Calculate metrics
    metrics = Metrics(
        training_time=training_time,
        true_graph=true_graph,
        est_graph=algorithm.est_graph,
        est_edges_dict=algorithm.est_edges_dict,
        true_edges_dict=true_edges_dict,
        edge_probabilities=algorithm.edge_probabilities,
    )

    metrics_results = metrics.get_result_metrics()
    cm = edge_confusion_matrix(algorithm.est_edges_dict, true_edges_dict)
    # 5. Save files if output path is specified
    if output_path:
        save_experiment_files(
            output_path,
            graph,
            true_edges_dict,
            algorithm.est_edges_dict,
            algorithm.edge_probabilities,
            metrics_results,
        )

    # 6. Print results
    if output_path:
        logging.info(
            f"Dataset: {gen_params['nodes']} nodes, {gen_params['npoints']} samples"
        )
        logging.info(f"Expected degree: {gen_params['expected_degree']}")
        logging.info(f"Mechanisms: {gen_params['mechanism_pool']}")
        logging.info(f"Algorithm: {alg_config['algorithm_name']}")
        logging.info(f"Training time: {training_time:.4f} seconds")
        logging.info(f"Edge confusion matrix: {cm}")

        logging.info("=== METRICS ===")
        adj_metrics = metrics_results["adjacency"]
        logging.info(
            f"Adjacency - Precision: {adj_metrics['precision']:.4f}, Recall: {adj_metrics['recall']:.4f}, F1: {adj_metrics['f1']:.4f}"
        )

        arrow_metrics = metrics_results["arrow"]
        logging.info(
            f"Arrow - Precision: {arrow_metrics['precision']:.4f}, Recall: {arrow_metrics['recall']:.4f}, F1: {arrow_metrics['f1']:.4f}"
        )

        logging.info(
            f"Distance - SHD: {metrics_results['shd']}, Skeleton SHD: {metrics_results['skeleton_shd']}"
        )

        if algorithm.edge_probabilities:
            logging.info(
                f"Confidence - Avg Freq: {metrics_results['average_frequency']:.4f}, Brier: {metrics_results['brier']:.4f}, ECE: {metrics_results['ece']:.4f}"
            )
    else:
        print(f"Dataset: {gen_params['nodes']} nodes, {gen_params['npoints']} samples")
        print(f"Expected degree: {gen_params['expected_degree']}")
        print(f"Mechanisms: {gen_params['mechanism_pool']}")
        print(f"Algorithm: {alg_config['algorithm_name']}")
        print(f"Training time: {training_time:.4f} seconds")
        print()

        print("=== METRICS ===")
        print("Adjacency Metrics:")
        adj_metrics = metrics_results["adjacency"]
        print(f"  Precision: {adj_metrics['precision']:.4f}")
        print(f"  Recall: {adj_metrics['recall']:.4f}")
        print(f"  F1: {adj_metrics['f1']:.4f}")
        print()

        print("Arrow Metrics:")
        arrow_metrics = metrics_results["arrow"]
        print(f"  Precision: {arrow_metrics['precision']:.4f}")
        print(f"  Recall: {arrow_metrics['recall']:.4f}")
        print(f"  F1: {arrow_metrics['f1']:.4f}")
        print()

        print("Distance Metrics:")
        print(f"  SHD: {metrics_results['shd']}")
        print(f"  Skeleton SHD: {metrics_results['skeleton_shd']}")
        print()

        if algorithm.edge_probabilities:
            print("Confidence Metrics:")
            print(f"  Average Frequency: {metrics_results['average_frequency']:.4f}")
            print(f"  Brier Score: {metrics_results['brier']:.4f}")
            print(f"  Expected Calibration Error: {metrics_results['ece']:.4f}")
            print()

    # 7. Prepare result data for CSV
    result_data = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "dataset_name": generation_name,
        "algorithm_name": algorithm_name,
        "nodes": gen_params["nodes"],
        "npoints": gen_params["npoints"],
        "expected_degree": gen_params["expected_degree"],
        "true_num_edges": len(graph.edges),
        "mechanism": "-".join(gen_params["mechanism_pool"]),
        "training_time": training_time,
        "skeleton_shd": metrics_results["skeleton_shd"],
        "skeleton_precision": adj_metrics["precision"],
        "skeleton_recall": adj_metrics["recall"],
        "skeleton_f1": adj_metrics["f1"],
        "shd": metrics_results["shd"],
        "precision": arrow_metrics["precision"],
        "recall": arrow_metrics["recall"],
        "f1": arrow_metrics["f1"],
        "normalized_skeleton_shd": metrics_results["normalized_skeleton_shd"],
        "possible_edges_normalized_shd": metrics_results[
            "possible_edges_normalized_shd"
        ],
        "true_support_normalized_shd": metrics_results["true_support_normalized_shd"],
        "est_support_normalized_shd": metrics_results["est_support_normalized_shd"],
        "average_frequency": metrics_results["average_frequency"],
        "median_frequency": metrics_results["median_frequency"],
        "min_frequency": metrics_results["min_frequency"],
        "average_edge_frequency": metrics_results["average_edge_frequency"],
        "brier": metrics_results["brier"],
        "ece": metrics_results["ece"],
    }

    # 8. Append to CSV if specified
    if csv_path:
        append_to_csv(csv_path, result_data)

    return result_data


def list_available_configs():
    """List all available generation and algorithm configurations."""
    generation_configs, algorithm_configs = load_configs()

    print("Available generation configurations:")
    print("-" * 40)
    for gen_name in sorted(generation_configs.keys()):
        gen_params = generation_configs[gen_name]
        print(f"  {gen_name}")
        print(
            f"    Nodes: {gen_params['nodes']}, Degree: {gen_params['expected_degree']}, Mechanisms: {gen_params['mechanism_pool']}"
        )

    print("\nAvailable algorithm configurations:")
    print("-" * 40)
    for alg_name in sorted(algorithm_configs.keys()):
        alg_config = algorithm_configs[alg_name]
        print(f"  {alg_name} ({alg_config['algorithm_name']})")


def main():
    """Main function to run experiments with command line arguments."""
    parser = argparse.ArgumentParser(description="Run causal discovery experiments")
    parser.add_argument("--generation", "-g", help="Generation configuration name")
    parser.add_argument("--algorithm", "-a", help="Algorithm configuration name")
    parser.add_argument(
        "--list", "-l", action="store_true", help="List all available configurations"
    )
    parser.add_argument(
        "--output-dir", "-o", help="Output directory for logs and results"
    )
    parser.add_argument(
        "--retries", "-r", type=int, default=3, help="Number of retry attempts"
    )
    parser.add_argument(
        "--csv", "-c", help="CSV file path to record experiment results (appending)"
    )

    args = parser.parse_args()

    if args.list:
        list_available_configs()
        return

    if not args.generation or not args.algorithm:
        parser.error(
            "Both --generation and --algorithm are required (use --list to see available options)"
        )

    # Setup output directory and logging if specified
    output_path = None
    if args.output_dir:
        output_path = (
            pathlib.Path(args.output_dir) / str(args.generation) / str(args.algorithm)
        )
        output_path.mkdir(parents=True, exist_ok=True)

        # Setup logging
        logging_fpath = str(output_path / "output.log")
        setup_logging(logging_fpath)

        # Capture warnings as log messages
        logging.captureWarnings(True)

        logging.info("Starting causal discovery experiment")
        logging.info("=" * 60)
    else:
        print("Starting causal discovery experiment")
        print("=" * 60)

    try:
        # Load configurations
        generation_configs, algorithm_configs = load_configs()

        # Run experiment with retry logic
        retries = 0
        while retries < args.retries:
            try:
                if output_path:
                    logging.info(
                        f"Running experiment (Attempt {retries+1}) for {args.algorithm} on {args.generation}"
                    )
                else:
                    print(
                        f"Running experiment (Attempt {retries+1}) for {args.algorithm} on {args.generation}"
                    )

                result = run_single_experiment(
                    args.generation,
                    args.algorithm,
                    generation_configs,
                    algorithm_configs,
                    output_path,
                    args.csv,
                )

                if output_path:
                    logging.info(f"Completed algorithm: {args.algorithm}")
                    logging.info("=" * 60)
                    logging.info("EXPERIMENT COMPLETED")
                    logging.info("=" * 60)
                    logging.info(f"Generation: {result['dataset_name']}")
                    logging.info(f"Algorithm: {result['algorithm_name']}")
                    logging.info(
                        f"Training time: {result['training_time']:.4f} seconds"
                    )
                    logging.info(f"Skeleton F1: {result['skeleton_f1']:.4f}")
                    logging.info(f"SHD: {result['shd']}")
                else:
                    print(f"\n{'='*60}")
                    print("EXPERIMENT COMPLETED")
                    print(f"{'='*60}")
                    print(f"Generation: {result['dataset_name']}")
                    print(f"Algorithm: {result['algorithm_name']}")
                    print(f"Training time: {result['training_time']:.4f} seconds")
                    print(f"Skeleton F1: {result['skeleton_f1']:.4f}")
                    print(f"SHD: {result['shd']}")

                break  # Success, exit retry loop

            except Exception as e:
                retries += 1
                if output_path:
                    logging.warning(
                        f"Attempt {retries} failed for {args.algorithm} on {args.generation}: {e}"
                    )
                    if retries >= args.retries:
                        logging.error(
                            f"Experiment failed after {args.retries} attempts for {args.algorithm} on {args.generation}"
                        )
                else:
                    print(f"Attempt {retries} failed: {e}")
                    if retries >= args.retries:
                        print(f"Experiment failed after {args.retries} attempts")
                        raise

    except Exception as e:
        if output_path:
            logging.exception("Unhandled exception.")
        else:
            print(f"Error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()
