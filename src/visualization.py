import pathlib
from causallearn.graph.GeneralGraph import GeneralGraph
from causallearn.utils.GraphUtils import GraphUtils
from matplotlib.figure import Figure
from matplotlib.axes import Axes
from src.metrics import Metrics
from typing import Optional
from io import BytesIO
from PIL import Image

import matplotlib.pyplot as plt
import seaborn as sns
import pydot


class Plotter:
    def __init__(self):
        """Initialize plotter with metrics object."""

    def plot_confusion(
        self,
        cm: list,
        title: str = "Confusion Matrix",
        xlabel: str = "Predicted Label",
        ylabel: str = "Actual Label",
        xticklabels: list = ["Predicted Positive", "Predicted Negative"],
        yticklabels: list = ["Actual Positive", "Actual Negative"],
        ax: Optional[Axes] = None,
    ):
        """Plot a single confusion matrix.

        Args:
            cm: 2x2 confusion matrix
            title: Title for the plot
            xlabel: Label for x-axis
            ylabel: Label for y-axis
            xticklabels: Labels for x-axis ticks
            yticklabels: Labels for y-axis ticks
            ax: Optional matplotlib axes to plot on

        Returns:
            Figure and axes objects
        """
        if ax is None:
            fig = plt.figure(figsize=(6, 5))
            ax = plt.gca()
        else:
            fig = ax.figure

        sns.heatmap(
            cm,
            annot=True,
            cmap="Blues",
            xticklabels=xticklabels,
            yticklabels=yticklabels,
            ax=ax,
        )
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        return fig, ax

    def plot_confusion_comparison(
        self,
        metrics_data,
        title: str = "Edge and Arrow Confusion Matrices",
        fpath: Optional[str] = None,
    ) -> Figure:
        """Plot comparison of confusion matrices for adjacency and arrows.

        Args:
            title: Title for the overall figure
            fpath: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """

        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(18, 5))

        # Plot adjacency confusion matrix
        adj_metrics = metrics_data["adjacency"]
        self.plot_confusion(
            adj_metrics["confusion_matrix"],
            title=f"Edge CM | precision={adj_metrics['precision']} | recall={adj_metrics['recall']}",
            xlabel="Predicted Edges",
            ylabel="Actual Edges",
            xticklabels=["", ""],
            yticklabels=["", ""],
            ax=ax1,
        )

        # Plot arrow confusion matrix
        arrow_metrics = metrics_data["arrow"]
        self.plot_confusion(
            arrow_metrics["confusion_matrix"],
            title=f"Arrow CM | precision={arrow_metrics['precision']} | recall={arrow_metrics['recall']}",
            xlabel="Predicted Arrows",
            ylabel="Actual Arrows",
            xticklabels=["", ""],
            yticklabels=["", ""],
            ax=ax2,
        )

        # Plot arrow_ce confusion matrix
        arrow_ce_metrics = metrics_data["arrow_ce"]
        self.plot_confusion(
            arrow_ce_metrics["confusion_matrix"],
            title=f"Arrow_ce CM | precision={arrow_ce_metrics['precision']} | recall={arrow_ce_metrics['recall']}",
            xlabel="Predicted Arrows",
            ylabel="Actual Arrows",
            xticklabels=["", ""],
            yticklabels=["", ""],
            ax=ax3,
        )

        fig.suptitle(title, fontsize=14)
        plt.tight_layout()

        if fpath:
            fig.savefig(fpath)
            plt.close(fig)

        return fig

    def plot_graph(
        self,
        graph: GeneralGraph,
        title: str = "Graph",
        ax: Optional[Axes] = None,
        fpath: Optional[str] = None,
    ) -> Figure:
        """Plot a single graph.

        Args:
            graph: Graph to plot
            title: Title for the plot
            ax: Optional matplotlib axes to plot on
            fpath: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        # Convert graph to image
        try:
            pyd = GraphUtils.to_pydot(graph)
        except AttributeError as e:
            raise e

        pyd.set_rankdir("LR")
        try:
            graph_img = Image.open(BytesIO(pyd.create_png()))
        except AttributeError as e:
            raise e

        if ax is None:
            fig = plt.figure(figsize=(6, 5))
            ax = plt.gca()
        else:
            fig = ax.figure

        # Plot graph
        ax.imshow(graph_img)
        ax.axis("off")
        ax.set_title(title)

        if fpath:
            fig.savefig(fpath)
            plt.close(fig)

        return fig

    def plot_graph_comparison(
        self,
        graph1: GeneralGraph,
        graph2: GeneralGraph,
        fpath: Optional[str] = None,
        title: str = "Graph comparison",
    ) -> Figure:
        """Plot comparison between true and estimated graphs.

        Args:
            graph1: First graph (typically true graph)
            graph2: Second graph (typically estimated graph)
            fpath: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        # Create figure with two subplots
        fig, (ax_left, ax_right) = plt.subplots(1, 2, figsize=(12, 6))
        fig.subplots_adjust(wspace=0.05)

        # Add divider
        divider = fig.add_axes((0.5, 0.1, 0.01, 0.8))
        divider.axvline(0.5, color="black", linewidth=1)
        divider.axis("off")

        # Plot both graphs using plot_graph
        self.plot_graph(graph1, title="True Graph", ax=ax_left)
        self.plot_graph(graph2, title="Estimated Graph", ax=ax_right)

        fig.suptitle(title, fontsize=14)

        if fpath:
            fig.savefig(fpath)
            plt.close(fig)

        return fig

    def plot_labeled_graph(
        self,
        graph: GeneralGraph,
        edge_labels: dict,
        title: str = "Labeled Graph",
        fpath: Optional[str] = None,
    ) -> Figure:
        """Plot a graph with edge labels.

        Args:
            graph: Graph to plot
            edge_labels: Dictionary mapping (i,j) tuples to label strings
            title: Title for the plot
            fpath: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        # Convert graph to pydot with labels
        from src.graph_aux import to_pydot_label_edges

        pyd = to_pydot_label_edges(G=graph, edge_labels=edge_labels, title=title)
        graph_img = Image.open(BytesIO(pyd.create_png()))

        # Create figure
        fig = plt.figure(figsize=(8, 6))
        ax = plt.gca()

        # Plot graph
        ax.imshow(graph_img)
        ax.axis("off")
        ax.set_title(title)

        if fpath:
            fig.savefig(fpath)
            plt.close(fig)

        return fig

    def plot_pydot(
        self,
        dotgraph: str,
        title: str = "Graph",
        ax: Optional[Axes] = None,
        fpath: Optional[str] = None,
    ) -> Figure:
        """Plot a graph from dot string representation.

        Args:
            dotgraph: DOT string representation of the graph
            title: Title for the plot
            ax: Optional matplotlib axes to plot on
            fpath: Optional path to save the figure

        Returns:
            Matplotlib figure object
        """
        # Convert dot string to image
        graph = pydot.graph_from_dot_data(dotgraph)[0]
        graph_img = Image.open(BytesIO(graph.create_png()))

        if ax is None:
            fig = plt.figure(figsize=(6, 5))
            ax = plt.gca()
        else:
            fig = ax.figure

        # Plot graph
        ax.imshow(graph_img)
        ax.axis("off")
        ax.set_title(title)

        if fpath:
            fig.savefig(fpath)
            plt.close(fig)

        return fig


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
