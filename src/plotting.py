from src.visualization import Plotter
from src.metrics import Metrics
from causallearn.graph.GeneralGraph import GeneralGraph
import pathlib


def plot_results(
    true_graph: GeneralGraph,
    est_graph: GeneralGraph,
    est_dotgraph: str,
    metrics: Metrics,
    output_path: pathlib.Path,
    algorithm_tag: str,
    dataset_tag: str,
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
