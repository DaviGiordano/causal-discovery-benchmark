#!/usr/bin/env python3
"""
Causal Discovery Algorithm Performance Analysis Script

This script performs comprehensive statistical analysis comparing different causal discovery algorithms
across multiple performance metrics using Friedman tests and post-hoc comparisons.

Adapted from the synthetic data analysis methodology for causal discovery evaluation.

Author: Analysis Script Generator
Date: 2025
"""

import argparse
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2


@dataclass
class MetricConfig:
    """Configuration for a performance metric"""

    key: str
    name: str
    lower_better: bool


@dataclass
class FriedmanResult:
    """Results from Friedman test"""

    n_datasets: int
    n_algorithms: int
    rank_sums: List[float]
    algorithm_names: List[str]
    friedman_statistic: float
    degrees_freedom: int
    p_value: float
    kendalls_w: float
    critical_value: float
    significant: bool
    average_ranks: List[float]


@dataclass
class PostHocComparison:
    """Single pairwise comparison result"""

    algorithm1: str
    algorithm2: str
    avg_rank1: float
    avg_rank2: float
    difference: float
    significant: bool


@dataclass
class PostHocResult:
    """Results from post-hoc analysis"""

    comparisons: List[PostHocComparison]
    critical_difference: float


@dataclass
class DescriptiveStats:
    """Descriptive statistics for an algorithm"""

    algorithm: str
    values: List[float]
    count: int
    mean: float
    median: float
    std_dev: float
    min_val: float
    max_val: float
    q1: float
    q3: float


@dataclass
class MetricAnalysisResult:
    """Complete analysis results for a single metric"""

    metric_config: MetricConfig
    descriptive_stats: List[DescriptiveStats]
    friedman_result: FriedmanResult
    posthoc_result: Optional[PostHocResult]


class CausalDiscoveryAnalyzer:
    """Main class for performing statistical analysis on causal discovery performance data"""

    def __init__(self, csv_path: str):
        """
        Initialize analyzer with data from CSV file

        Args:
            csv_path: Path to the CSV file containing causal discovery performance data
        """
        self.csv_path = Path(csv_path)
        self.data = None
        self.processed_data = {}
        self.metrics = [
            # Structural accuracy metrics (lower is better)
            MetricConfig("shd", "Structural Hamming Distance", True),
            MetricConfig("skeleton_shd", "Skeleton Hamming Distance", True),
            MetricConfig("normalized_skeleton_shd", "Normalized Skeleton SHD", True),
            # Precision/Recall metrics (higher is better)
            MetricConfig("skeleton_f1", "Skeleton F1 Score", False),
            MetricConfig("f1", "Directed F1 Score", False),
            MetricConfig("skeleton_precision", "Skeleton Precision", False),
            MetricConfig("skeleton_recall", "Skeleton Recall", False),
            # Calibration metrics (lower is better)
            MetricConfig("brier", "Brier Score", True),
            MetricConfig("ece", "Expected Calibration Error", True),
            # Timing metric (lower is better)
            MetricConfig("training_time", "Training Time", True),
        ]

    def load_data(self) -> None:
        """Load and preprocess data from CSV file"""
        try:
            self.data = pd.read_csv(self.csv_path)
            print(
                f"Loaded data with {len(self.data)} rows and {len(self.data.columns)} columns"
            )
        except Exception as e:
            raise FileNotFoundError(f"Could not load data from {self.csv_path}: {e}")

        # Clean algorithm and dataset columns
        self.data["algorithm_name"] = (
            self.data["algorithm_name"].astype(str).str.strip()
        )
        self.data["dataset_name"] = self.data["dataset_name"].astype(str).str.strip()

    def preprocess_data(self) -> None:
        """Preprocess data for analysis"""
        if self.data is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        # Group data by dataset and algorithm
        data_by_dataset = {}
        algorithms = set()

        for _, row in self.data.iterrows():
            dataset = row["dataset_name"]
            algorithm = row["algorithm_name"]

            if pd.isna(dataset) or pd.isna(algorithm):
                continue

            algorithms.add(algorithm)

            if dataset not in data_by_dataset:
                data_by_dataset[dataset] = {}

            # Store all metric values for this dataset-algorithm combination
            metric_values = {}
            for metric in self.metrics:
                value = row[metric.key]
                # Handle NaN values in F1 scores (can occur when precision and recall are both 0)
                if pd.isna(value) and metric.key in ["f1", "skeleton_f1"]:
                    value = 0.0
                metric_values[metric.key] = value

            data_by_dataset[dataset][algorithm] = metric_values

        # Filter to complete datasets (all algorithms present with valid data)
        algorithms_list = sorted(list(algorithms))
        complete_datasets = []

        for dataset, algorithms_data in data_by_dataset.items():
            if all(algorithm in algorithms_data for algorithm in algorithms_list):
                # Check if all metrics have valid data
                valid = True
                for algorithm in algorithms_list:
                    algorithm_data = algorithms_data[algorithm]
                    for metric in self.metrics:
                        value = algorithm_data[metric.key]
                        if pd.isna(value) or not np.isfinite(value):
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    complete_datasets.append(dataset)

        self.processed_data = {
            "data_by_dataset": data_by_dataset,
            "algorithms": algorithms_list,
            "complete_datasets": complete_datasets,
        }

        print(
            f"Found {len(complete_datasets)} complete datasets with all {len(algorithms_list)} algorithms"
        )
        print(f"Algorithms: {', '.join(algorithms_list)}")

    def calculate_descriptive_stats(self, metric_key: str) -> List[DescriptiveStats]:
        """Calculate descriptive statistics for each algorithm for a given metric"""
        stats_list = []

        for algorithm in self.processed_data["algorithms"]:
            values = []

            for dataset in self.processed_data["complete_datasets"]:
                value = self.processed_data["data_by_dataset"][dataset][algorithm][
                    metric_key
                ]
                if not pd.isna(value) and np.isfinite(value):
                    values.append(value)

            if len(values) == 0:
                continue

            values = np.array(values)
            stats_list.append(
                DescriptiveStats(
                    algorithm=algorithm,
                    values=values.tolist(),
                    count=len(values),
                    mean=np.mean(values),
                    median=np.median(values),
                    std_dev=np.std(values, ddof=1) if len(values) > 1 else 0.0,
                    min_val=np.min(values),
                    max_val=np.max(values),
                    q1=np.percentile(values, 25),
                    q3=np.percentile(values, 75),
                )
            )

        return stats_list

    def perform_friedman_test(
        self, metric_key: str, lower_better: bool
    ) -> FriedmanResult:
        """Perform Friedman test for a given metric"""
        algorithms = self.processed_data["algorithms"]
        datasets = self.processed_data["complete_datasets"]
        data_by_dataset = self.processed_data["data_by_dataset"]

        n_datasets = len(datasets)
        n_algorithms = len(algorithms)

        if n_datasets == 0 or n_algorithms < 2:
            raise ValueError("Need at least 2 algorithms and some complete datasets")

        # Calculate ranks for each dataset
        rank_sums = np.zeros(n_algorithms)

        for dataset in datasets:
            values = []
            for algorithm in algorithms:
                value = data_by_dataset[dataset][algorithm][metric_key]
                values.append((algorithm, value))

            # Sort by value (ascending for lower-is-better, descending for higher-is-better)
            values.sort(key=lambda x: x[1] if lower_better else -x[1])

            # Assign ranks (handle ties by averaging)
            ranks = np.zeros(n_algorithms)
            i = 0
            while i < len(values):
                j = i
                # Find tied values
                while j < len(values) and values[j][1] == values[i][1]:
                    j += 1

                # Assign average rank
                avg_rank = (i + j + 1) / 2
                for k in range(i, j):
                    algorithm_idx = algorithms.index(values[k][0])
                    ranks[algorithm_idx] = avg_rank

                i = j

            rank_sums += ranks

        # Calculate Friedman statistic (standard formula)
        sum_rank_squares = np.sum(rank_sums**2)
        friedman_stat = (
            12 / (n_datasets * n_algorithms * (n_algorithms + 1))
        ) * sum_rank_squares - 3 * n_datasets * (n_algorithms + 1)

        # Degrees of freedom and critical value
        df = n_algorithms - 1
        critical_value = chi2.ppf(0.95, df)  # α = 0.05
        p_value = 1 - chi2.cdf(friedman_stat, df)
        significant = friedman_stat > critical_value

        average_ranks = rank_sums / n_datasets
        kendalls_w = friedman_stat / (n_datasets * (n_algorithms - 1))

        return FriedmanResult(
            n_datasets=n_datasets,
            n_algorithms=n_algorithms,
            rank_sums=rank_sums.tolist(),
            algorithm_names=algorithms.copy(),
            friedman_statistic=friedman_stat,
            degrees_freedom=df,
            p_value=p_value,
            kendalls_w=kendalls_w,
            critical_value=critical_value,
            significant=significant,
            average_ranks=average_ranks.tolist(),
        )

    def perform_posthoc_analysis(
        self, friedman_result: FriedmanResult
    ) -> PostHocResult:
        """Perform post-hoc pairwise comparisons (Dunn's test approximation)"""
        if not friedman_result.significant or friedman_result.n_algorithms < 3:
            return None

        algorithms = friedman_result.algorithm_names
        n_datasets = friedman_result.n_datasets
        n_algorithms = friedman_result.n_algorithms
        average_ranks = friedman_result.average_ranks

        # Critical difference for Dunn's test (approximation)
        # Using Bonferroni correction for multiple comparisons
        n_comparisons = n_algorithms * (n_algorithms - 1) // 2
        alpha_corrected = 0.05 / n_comparisons
        z_critical = stats.norm.ppf(1 - alpha_corrected / 2)
        critical_difference = z_critical * np.sqrt(
            (n_algorithms * (n_algorithms + 1)) / (6 * n_datasets)
        )

        comparisons = []
        for i in range(n_algorithms):
            for j in range(i + 1, n_algorithms):
                diff = abs(average_ranks[i] - average_ranks[j])
                significant = diff > critical_difference

                comparisons.append(
                    PostHocComparison(
                        algorithm1=algorithms[i],
                        algorithm2=algorithms[j],
                        avg_rank1=average_ranks[i],
                        avg_rank2=average_ranks[j],
                        difference=diff,
                        significant=significant,
                    )
                )

        return PostHocResult(
            comparisons=comparisons, critical_difference=critical_difference
        )

    def analyze_metric(self, metric: MetricConfig) -> MetricAnalysisResult:
        """Perform complete analysis for a single metric"""
        print(f"\nAnalyzing {metric.name}...")

        # Calculate descriptive statistics
        descriptive_stats = self.calculate_descriptive_stats(metric.key)

        # Perform Friedman test
        friedman_result = self.perform_friedman_test(metric.key, metric.lower_better)

        # Perform post-hoc analysis if appropriate
        posthoc_result = None
        if friedman_result.significant and friedman_result.n_algorithms >= 3:
            posthoc_result = self.perform_posthoc_analysis(friedman_result)

        return MetricAnalysisResult(
            metric_config=metric,
            descriptive_stats=descriptive_stats,
            friedman_result=friedman_result,
            posthoc_result=posthoc_result,
        )

    def run_complete_analysis(self) -> Dict[str, MetricAnalysisResult]:
        """Run complete analysis for all metrics"""
        if not self.processed_data:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        results = {}
        for metric in self.metrics:
            try:
                results[metric.key] = self.analyze_metric(metric)
            except Exception as e:
                print(f"Warning: Could not analyze {metric.name}: {e}")
                continue

        return results


class CausalDiscoveryResultsPrinter:
    """Class responsible for printing analysis results in a user-friendly format"""

    def __init__(self, results: Dict[str, MetricAnalysisResult]):
        """
        Initialize printer with analysis results

        Args:
            results: Dictionary mapping metric keys to analysis results
        """
        self.results = results

    def print_dataset_summary(self) -> None:
        """Print summary of the dataset"""
        if not self.results:
            return

        # Get summary info from any result
        first_result = next(iter(self.results.values()))
        n_datasets = first_result.friedman_result.n_datasets
        algorithms = first_result.friedman_result.algorithm_names

        print("=" * 80)
        print("CAUSAL DISCOVERY ALGORITHM PERFORMANCE ANALYSIS")
        print("=" * 80)
        print(f"Total datasets analyzed: {n_datasets}")
        print(f"Causal discovery algorithms: {', '.join(algorithms)}")
        print(f"Metrics analyzed: {len(self.results)}")
        print()

    def print_metric_analysis(self, metric_key: str) -> None:
        """Print detailed analysis for a single metric"""
        if metric_key not in self.results:
            print(f"No results found for metric: {metric_key}")
            return

        result = self.results[metric_key]
        metric = result.metric_config

        print("-" * 60)
        print(f"ANALYSIS: {metric.name}")
        print(f"({'Lower is better' if metric.lower_better else 'Higher is better'})")
        print("-" * 60)

        # Descriptive Statistics
        print("\n1. DESCRIPTIVE STATISTICS")
        print("-" * 30)
        stats_df_data = []
        for stat in result.descriptive_stats:
            stats_df_data.append(
                {
                    "Algorithm": stat.algorithm,
                    "Count": stat.count,
                    "Mean": f"{stat.mean:.4f}",
                    "Median": f"{stat.median:.4f}",
                    "Std Dev": f"{stat.std_dev:.4f}",
                    "Min": f"{stat.min_val:.4f}",
                    "Max": f"{stat.max_val:.4f}",
                }
            )

        # Sort by mean (best first)
        stats_df_data.sort(
            key=lambda x: float(x["Mean"]), reverse=not metric.lower_better
        )

        # Print as formatted table
        headers = ["Algorithm", "Count", "Mean", "Median", "Std Dev", "Min", "Max"]
        col_widths = [
            max(
                len(str(row.get(h, "")))
                for row in [dict(zip(headers, headers))] + stats_df_data
            )
            + 2
            for h in headers
        ]

        # Header
        header_row = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_row)
        print("-" * len(header_row))

        # Data rows
        for i, row in enumerate(stats_df_data):
            row_str = "".join(
                str(row.get(h, "")).ljust(w) for h, w in zip(headers, col_widths)
            )
            if i == 0:  # Mark best performer
                print(f"* {row_str}")
            else:
                print(f"  {row_str}")

        # Friedman Test Results
        print(f"\n2. FRIEDMAN TEST RESULTS")
        print("-" * 30)
        fr = result.friedman_result
        print(f"Friedman Statistic: {fr.friedman_statistic:.4f}")
        print(f"Degrees of Freedom: {fr.degrees_freedom}")
        print(f"Critical Value (α=0.05): {fr.critical_value:.4f}")
        print(f"P-value: {fr.p_value:.6f}")
        print(f"Effect size (Kendall's W): {fr.kendalls_w:.6f}")
        print(f"Significant: {'Yes ✓' if fr.significant else 'No'}")

        print(f"\nAverage Ranks (lower rank = better performance):")
        rank_data = list(zip(fr.algorithm_names, fr.average_ranks))
        rank_data.sort(key=lambda x: x[1])  # Sort by rank
        for i, (algorithm, rank) in enumerate(rank_data):
            marker = "★" if i == 0 else " "
            print(f"  {marker} {algorithm}: {rank:.3f}")

        # Post-hoc Analysis
        if result.posthoc_result:
            print(f"\n3. POST-HOC ANALYSIS (Dunn's Test)")
            print("-" * 30)
            pr = result.posthoc_result
            print(f"Critical Difference: {pr.critical_difference:.4f}")
            print(f"Pairwise Comparisons:")

            significant_comparisons = [c for c in pr.comparisons if c.significant]
            non_significant = [c for c in pr.comparisons if not c.significant]

            if significant_comparisons:
                print("  Significant differences:")
                for comp in significant_comparisons:
                    print(
                        f"    ✓ {comp.algorithm1} vs {comp.algorithm2}: "
                        f"ranks {comp.avg_rank1:.3f} vs {comp.avg_rank2:.3f} "
                        f"(diff: {comp.difference:.3f})"
                    )

            if non_significant:
                print("  Non-significant differences:")
                for comp in non_significant:
                    print(
                        f"      {comp.algorithm1} vs {comp.algorithm2}: "
                        f"ranks {comp.avg_rank1:.3f} vs {comp.avg_rank2:.3f} "
                        f"(diff: {comp.difference:.3f})"
                    )

        elif result.friedman_result.significant:
            print(f"\n3. POST-HOC ANALYSIS")
            print("-" * 30)
            print(
                "Post-hoc analysis not performed (need ≥3 algorithms for meaningful comparisons)"
            )

        # Interpretation
        print(f"\n4. INTERPRETATION")
        print("-" * 30)
        if fr.significant:
            best_algorithm = rank_data[0][0]  # Algorithm with lowest rank
            print(f"✓ Significant differences detected (p = {fr.p_value:.6f})")
            print(f"✓ Best performing algorithm: {best_algorithm}")

            if result.posthoc_result:
                sig_count = len(
                    [c for c in result.posthoc_result.comparisons if c.significant]
                )
                total_count = len(result.posthoc_result.comparisons)
                print(
                    f"✓ {sig_count}/{total_count} pairwise comparisons are significant after correction"
                )
        else:
            print(f"⚠ No significant differences detected (p = {fr.p_value:.6f})")
            print("⚠ All algorithms perform similarly for this metric")

        print()

    def print_cross_metric_summary(self) -> None:
        """Print summary comparing results across all metrics"""
        print("=" * 80)
        print("CROSS-METRIC SUMMARY")
        print("=" * 80)

        # Create summary table
        summary_data = []
        for metric_key, result in self.results.items():
            # Find best algorithm (lowest average rank)
            fr = result.friedman_result
            best_idx = np.argmin(fr.average_ranks)
            best_algorithm = fr.algorithm_names[best_idx]

            # Count significant pairwise comparisons
            sig_comparisons = 0
            total_comparisons = 0
            if result.posthoc_result:
                sig_comparisons = len(
                    [c for c in result.posthoc_result.comparisons if c.significant]
                )
                total_comparisons = len(result.posthoc_result.comparisons)

            summary_data.append(
                {
                    "Metric": result.metric_config.name,
                    "Best Algorithm": best_algorithm,
                    "Friedman Sig": "Yes ✓" if fr.significant else "No",
                    "P-value": f"{fr.p_value:.4f}",
                    "Pairwise Sig": (
                        f"{sig_comparisons}/{total_comparisons}"
                        if total_comparisons > 0
                        else "N/A"
                    ),
                }
            )

        # Print summary table
        headers = [
            "Metric",
            "Best Algorithm",
            "Friedman Sig",
            "P-value",
            "Pairwise Sig",
        ]
        col_widths = [
            max(
                len(str(row.get(h, "")))
                for row in [dict(zip(headers, headers))] + summary_data
            )
            + 2
            for h in headers
        ]

        header_row = "".join(h.ljust(w) for h, w in zip(headers, col_widths))
        print(header_row)
        print("-" * len(header_row))

        for row in summary_data:
            row_str = "".join(
                str(row.get(h, "")).ljust(w) for h, w in zip(headers, col_widths)
            )
            print(row_str)

        print()

        # Overall conclusions
        print("OVERALL CONCLUSIONS:")
        print("-" * 20)

        # Algorithm performance across metrics
        algorithm_wins = {}
        significant_metrics = 0

        for result in self.results.values():
            if result.friedman_result.significant:
                significant_metrics += 1
                best_idx = np.argmin(result.friedman_result.average_ranks)
                best_algorithm = result.friedman_result.algorithm_names[best_idx]
                algorithm_wins[best_algorithm] = (
                    algorithm_wins.get(best_algorithm, 0) + 1
                )

        if algorithm_wins:
            print(
                f"• {significant_metrics}/{len(self.results)} metrics show significant differences"
            )
            print("• Algorithm performance summary:")
            for algorithm, wins in sorted(
                algorithm_wins.items(), key=lambda x: x[1], reverse=True
            ):
                print(f"  - {algorithm}: best in {wins} metric(s)")
        else:
            print("• No metrics show significant differences between algorithms")
            print("• All algorithms perform similarly across all evaluated metrics")

        print()

    def print_complete_report(self) -> None:
        """Print complete analysis report"""
        self.print_dataset_summary()

        for metric_key in self.results.keys():
            self.print_metric_analysis(metric_key)

        self.print_cross_metric_summary()


def main():
    """Main function to run the analysis"""
    parser = argparse.ArgumentParser(
        description="Causal Discovery Algorithm Performance Analysis"
    )
    parser.add_argument(
        "csv_file",
        help="Path to the CSV file containing causal discovery performance data",
    )
    parser.add_argument(
        "--metric",
        help="Analyze only specific metric",
        choices=[
            "shd",
            "skeleton_shd",
            "normalized_skeleton_shd",
            "skeleton_f1",
            "f1",
            "skeleton_precision",
            "skeleton_recall",
            "brier",
            "ece",
            "training_time",
        ],
    )
    parser.add_argument(
        "--summary-only", action="store_true", help="Print only cross-metric summary"
    )

    args = parser.parse_args()

    try:
        # Initialize analyzer
        analyzer = CausalDiscoveryAnalyzer(args.csv_file)

        # Load and preprocess data
        print("Loading causal discovery performance data...")
        analyzer.load_data()

        print("Preprocessing data...")
        analyzer.preprocess_data()

        # Run analysis
        print("Running statistical analysis...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            results = analyzer.run_complete_analysis()

        # Print results
        printer = CausalDiscoveryResultsPrinter(results)

        if args.summary_only:
            printer.print_dataset_summary()
            printer.print_cross_metric_summary()
        elif args.metric:
            printer.print_dataset_summary()
            printer.print_metric_analysis(args.metric)
        else:
            printer.print_complete_report()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
