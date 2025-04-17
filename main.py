import os
import pathlib
import time
import logging
from tqdm import tqdm
from dotenv import load_dotenv
from src.parse_tetrad_string import str_to_edge_dict
from src.algorithm_choice import get_discovery_algorithm
from causallearn.graph.GeneralGraph import GeneralGraph
from src.metrics import Metrics
from src.causal_discovery.causallearn_algorithms import (
    PCAlgorithm,
    FCIAlgorithm,
    GESAlgorithm,
    ExactSearchAlgorithm,
    ICALiNGAMAlgorithm,
    DirectLiNGAMAlgorithm,
    GRaSPAlgorithm,
    BossAlgorithm,
)
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend

# from src.causal_discovery.castle_algorithms import (
#     NOTEARSAlgorithm,
#     DAGGNNAlgorithm,
#     CORLAlgorithm,
#     GraNDAGAlgorithm,
# )
from src.load_parse import load_csv, load_json, load_txt, load_yaml, parse_arguments
from src.graph_aux import dag_adj_to_graph
from src.visualization import Plotter
from src.results_writer import write_experiment_results
from src.logging_config import setup_logging
from src.mlflow_logger import MLflowLogger
from flatten_dict import flatten
import json
import numpy as np

ALL_ALGORITHMS_CONFIGS = "./configs/algorithms.yaml"
ALL_DATA_CONFIGS = "./configs/dataset.yaml"


def plot_results(
    true_graph: GeneralGraph,
    est_graph: GeneralGraph,
    est_dotgraph: str,
    metrics: Metrics,
    output_path: pathlib.Path,
):

    # Generate and save plots
    plotter = Plotter()
    plotter.plot_confusion_comparison(
        metrics_data=metrics.get_result_metrics(),
        title=f"Confusion Matrices - {algorithm_tag} - {dataset_tag}",
        fpath=f"{output_path}/confusion_matrices.png",
    )
    plotter.plot_graph(
        title=f"True Graph - {dataset_tag}",
        graph=true_graph,
        fpath=f"{output_path}/true_graph.png",
    )
    plotter.plot_graph(
        title=f"Estimated Graph - {algorithm_tag} - {dataset_tag}",
        graph=est_graph,
        fpath=f"{output_path}/est_graph.png",
    )
    plotter.plot_graph_comparison(
        graph1=true_graph,
        graph2=est_graph,
        fpath=f"{output_path}/graph_comparison.png",
        title=f"Graph Comparison - {algorithm_tag} - {dataset_tag}",
    )
    plotter.plot_pydot(
        est_dotgraph,
        title=f"Edge probabilities - {algorithm_tag} - {dataset_tag}",
        fpath=f"{output_path}/edge_probabilities.png",
    )


def run_experiment(
    algorithm_tag: str,
    dataset_tag: str,
    experiment_name: str,
    output_path: pathlib.Path,
):
    # Setup logging
    load_dotenv(override=True)

    try:
        # Setup MLflow logging
        mlflow_logger = MLflowLogger(
            experiment_name=experiment_name,
            mlflow_tracking_uri=os.getenv("MLFLOW_TRACKING_URI"),
        )

        # Load data and ground truth adj matrix
        data_params = load_yaml(ALL_DATA_CONFIGS)[dataset_tag]
        data = load_csv(data_params["train_fpath"])
        true_adj = load_csv(data_params["true_adj_fpath"])
        true_graph = dag_adj_to_graph(true_adj, "upper_triangular")
        true_edges_dict = load_json(data_params["true_edges_dict"])
        # Resample data if larger than 2000 samples
        if len(data) > 2000:
            logging.info(
                f"Dataset size {len(data)} exceeds 2000, resampling to 2000 samples"
            )
            np.random.seed(42)
            sample_indices = np.random.choice(len(data), size=2000, replace=False)
            data = data[sample_indices]
        # Load selected model
        config_params = load_yaml(ALL_ALGORITHMS_CONFIGS)[algorithm_tag]
        model = get_discovery_algorithm(**config_params)

        # Train model to discover causal structure and measure time
        start_time = time.time()
        model.train(data)
        training_time = time.time() - start_time

        # Evaluate and get metrics
        metrics = Metrics(
            training_time,
            true_graph,
            model.est_graph,
            model.est_edges_dict,
            true_edges_dict,
            model.edge_probabilities,
        )
        metrics_results = metrics.get_result_metrics()

        params_to_log = flatten(config_params, reducer="dot")
        params_to_log["dataset"] = dataset_tag
        params_to_log["dataset_length"] = len(data)

        plot_results(
            true_graph,
            model.est_graph,
            model.est_dotgraph,
            metrics,
            output_path,
        )

        # Save edge probabilities to JSON
        edge_probs_str_keys = {str(k): v for k, v in model.edge_probabilities.items()}
        with open(output_path / "edge_probabilities.json", "w") as f:
            json.dump(edge_probs_str_keys, f, indent=4)

        # Write graph strings
        with open(output_path / "graph_strings.txt", "w") as f:
            f.write("== True graph ==\n")
            f.write(str(true_graph))
            f.write("\n\n== Estimated graph ==\n")
            f.write(str(model.est_graph))
            f.write("\n\n== Confidence per edge ==\n")
            f.write(model.graph_string)

        # Write experiment parameters and results to JSON
        write_experiment_results(
            output_path=output_path,
            params=params_to_log,
            metrics=metrics_results,
        )
        # Log to MLflow
        mlflow_logger.log_run(
            run_name=algorithm_tag,
            params=params_to_log,
            metrics=metrics_results,
            artifacts_dir=output_path,
        )
    except Exception:
        logging.exception("Unhandled exception.")  # Logs full traceback
        raise


if __name__ == "__main__":

    dataset_tags = (
        # # "adult_dataset",
        # "ruta_synth_normal_4000",
        # "ruta_synth_uniform_4000",
        # # "ruta_synth_normal_100",
        # # "ruta_synth_uniform_1000",
        # # "ruta_synth_normal_1000",
        # # "ruta_synth_uniform_10000",
        # # "ruta_synth_normal_10000",
        # # "csuite_cat_chain",
        # # "csuite_cat_collider",
        # # "csuite_cat_to_cts",
        # # "csuite_cts_to_cat",
        # # "csuite_linexp",
        # # "csuite_lingauss",
        # # "csuite_nonlingauss",
        # "csuite_nonlin_simpson",
        # # "csuite_symprod_simpson",
        # # "csuite_weak_arrows",
        # "csuite_weak_arrows_binary_t",
        # # "csuite_large_backdoor",
        # "csuite_large_backdoor_binary_t",
        # "csuite_mixed_simpson",
        "csuite_mixed_confounding",
    )
    algorithm_tags = [
        # "pc_tetrad_01",
        # "pc_tetrad_05",
        "pc_tetrad_10",
        # "fges_tetrad_pd1",
        # "fges_tetrad_pd2",
        # "fges_tetrad_pd4",
        # "boss_tetrad_pd1",
        # "boss_tetrad_pd2",
        # "boss_tetrad_pd4",
        # "grasp_tetrad_pd1",
        # "grasp_tetrad_pd2",
        # "grasp_tetrad_pd4",
        # "directlingam_pd1",
        # "directlingam_pd2",
        # "directlingam_pd4",
        # "dagma_tetrad_pd1",
        # "dagma_tetrad_pd2",
        # "dagma_tetrad_pd4",
    ]

    experiment_name = "resampled_focused_experiments_tetrad"
    MAX_RETRIES = 2

    for dataset_tag in tqdm(dataset_tags):
        for algorithm_tag in algorithm_tags:

            # Setup output directory
            output_path = pathlib.Path(
                f"./results/{experiment_name}/{dataset_tag}/{algorithm_tag}"
            )
            output_path.mkdir(parents=True, exist_ok=True)

            # Setup logging
            logging_fpath = str(output_path / "output.log")
            setup_logging(logging_fpath)

            # Capture warnings as log messages
            logging.captureWarnings(True)

            retries = 0
            while retries < MAX_RETRIES:
                try:
                    logging.info(
                        f"Running experiment (Attempt {retries+1}) for {algorithm_tag} on {dataset_tag}"
                    )
                    run_experiment(
                        algorithm_tag=algorithm_tag,
                        dataset_tag=dataset_tag,
                        experiment_name=experiment_name,
                        output_path=output_path,
                    )
                    logging.info(f"Completed algorithm: {algorithm_tag}")
                    break  # Success, exit retry loop

                except Exception as e:
                    retries += 1
                    logging.warning(
                        f"Attempt {retries} failed for {algorithm_tag} on {dataset_tag}: {e}"
                    )
                    if retries >= MAX_RETRIES:
                        logging.error(
                            f"Experiment failed after {MAX_RETRIES} attempts for {algorithm_tag} on {dataset_tag}"
                        )
