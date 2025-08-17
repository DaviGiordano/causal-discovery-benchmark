#!/usr/bin/env python3
"""
Dataset Difficulty Analysis Script for Causal Discovery

This script performs comprehensive statistical analysis to identify which datasets are "harder"
or "easier" for causal discovery algorithms based on their performance metrics.

The analysis uses multiple approaches:
1. Aggregate performance scores across all algorithms per dataset
2. Friedman tests to rank datasets by difficulty
3. Variance analysis to identify datasets with high algorithm disagreement
4. Outlier detection for exceptionally hard/easy datasets

Author: Dataset Analysis Script Generator
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
    weight: float = 1.0  # Weight for composite scoring


@dataclass
class DatasetDifficultyStats:
    """Statistics for dataset difficulty"""

    dataset_name: str
    algorithm_count: int
    metric_scores: Dict[str, float]  # Average scores per metric
    composite_score: float  # Weighted composite difficulty score
    rank_position: float  # Average rank across metrics
    performance_variance: float  # Variance in algorithm performance
    worst_algorithm: str  # Algorithm that performed worst
    best_algorithm: str  # Algorithm that performed best
    performance_spread: float  # Difference between best and worst


@dataclass
class DatasetFriedmanResult:
    """Results from Friedman test on datasets"""

    n_algorithms: int
    n_datasets: int
    rank_sums: List[float]
    dataset_names: List[str]
    friedman_statistic: float
    degrees_freedom: int
    p_value: float
    kendalls_w: float
    critical_value: float
    significant: bool
    average_ranks: List[float]


@dataclass
class DifficultyAnalysisResult:
    """Complete difficulty analysis results"""

    metric_config: MetricConfig
    dataset_stats: List[DatasetDifficultyStats]
    friedman_result: DatasetFriedmanResult
    difficulty_ranking: List[Tuple[str, float]]  # (dataset_name, difficulty_score)
    hardest_datasets: List[str]
    easiest_datasets: List[str]


class DatasetDifficultyAnalyzer:
    """Main class for analyzing dataset difficulty in causal discovery"""

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
            # Structural accuracy metrics (lower is better, higher weight)
            MetricConfig("shd", "Structural Hamming Distance", True, 2.0),
            MetricConfig("skeleton_shd", "Skeleton Hamming Distance", True, 1.5),
            MetricConfig(
                "normalized_skeleton_shd", "Normalized Skeleton SHD", True, 1.5
            ),
            # Precision/Recall metrics (higher is better)
            MetricConfig("skeleton_f1", "Skeleton F1 Score", False, 1.5),
            MetricConfig("f1", "Directed F1 Score", False, 2.0),
            MetricConfig("skeleton_precision", "Skeleton Precision", False, 1.0),
            MetricConfig("skeleton_recall", "Skeleton Recall", False, 1.0),
            # Calibration metrics (lower is better)
            MetricConfig("brier", "Brier Score", True, 1.0),
            MetricConfig("ece", "Expected Calibration Error", True, 1.0),
            # Timing metric (lower is better, lower weight for difficulty assessment)
            MetricConfig("training_time", "Training Time", True, 0.5),
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

        # Group data by algorithm and dataset (inverse of algorithm analysis)
        data_by_algorithm = {}
        datasets = set()

        for _, row in self.data.iterrows():
            dataset = row["dataset_name"]
            algorithm = row["algorithm_name"]

            if pd.isna(dataset) or pd.isna(algorithm):
                continue

            datasets.add(dataset)

            if algorithm not in data_by_algorithm:
                data_by_algorithm[algorithm] = {}

            # Store all metric values for this algorithm-dataset combination
            metric_values = {}
            for metric in self.metrics:
                value = row[metric.key]
                # Handle NaN values in F1 scores
                if pd.isna(value) and metric.key in ["f1", "skeleton_f1"]:
                    if metric.key == "f1":
                        precision = row.get("precision", 0)
                        recall = row.get("recall", 0)
                        value = 0.0 if (precision == 0 and recall == 0) else np.nan
                    elif metric.key == "skeleton_f1":
                        precision = row.get("skeleton_precision", 0)
                        recall = row.get("skeleton_recall", 0)
                        value = 0.0 if (precision == 0 and recall == 0) else np.nan

                metric_values[metric.key] = value

            data_by_algorithm[algorithm][dataset] = metric_values

        # Filter to complete algorithms (all datasets present with valid data)
        datasets_list = sorted(list(datasets))
        complete_algorithms = []

        for algorithm, datasets_data in data_by_algorithm.items():
            if all(dataset in datasets_data for dataset in datasets_list):
                # Check if all metrics have valid data
                valid = True
                for dataset in datasets_list:
                    dataset_data = datasets_data[dataset]
                    for metric in self.metrics:
                        value = dataset_data[metric.key]
                        if pd.isna(value) or not np.isfinite(value):
                            valid = False
                            break
                    if not valid:
                        break

                if valid:
                    complete_algorithms.append(algorithm)

        self.processed_data = {
            "data_by_algorithm": data_by_algorithm,
            "datasets": datasets_list,
            "complete_algorithms": complete_algorithms,
        }

        print(
            f"Found {len(complete_algorithms)} complete algorithms across {len(datasets_list)} datasets"
        )
        print(f"Datasets: {', '.join(datasets_list)}")
        print(f"Algorithms: {', '.join(complete_algorithms)}")

    def calculate_dataset_difficulty_stats(
        self, metric: MetricConfig
    ) -> List[DatasetDifficultyStats]:
        """Calculate difficulty statistics for each dataset for a given metric"""
        stats_list = []
        datasets = self.processed_data["datasets"]
        algorithms = self.processed_data["complete_algorithms"]
        data_by_algorithm = self.processed_data["data_by_algorithm"]

        for dataset in datasets:
            values = []
            algorithm_performances = {}

            # Collect all algorithm performances on this dataset
            for algorithm in algorithms:
                value = data_by_algorithm[algorithm][dataset][metric.key]
                if not pd.isna(value) and np.isfinite(value):
                    values.append(value)
                    algorithm_performances[algorithm] = value

            if len(values) == 0:
                continue

            values = np.array(values)

            # Calculate statistics
            if metric.lower_better:
                # For "lower is better" metrics, higher values indicate harder datasets
                avg_score = np.mean(values)
                worst_algorithm = max(
                    algorithm_performances.keys(),
                    key=lambda x: algorithm_performances[x],
                )
                best_algorithm = min(
                    algorithm_performances.keys(),
                    key=lambda x: algorithm_performances[x],
                )
                performance_spread = np.max(values) - np.min(values)
            else:
                # For "higher is better" metrics, lower values indicate harder datasets
                avg_score = np.mean(values)
                worst_algorithm = min(
                    algorithm_performances.keys(),
                    key=lambda x: algorithm_performances[x],
                )
                best_algorithm = max(
                    algorithm_performances.keys(),
                    key=lambda x: algorithm_performances[x],
                )
                performance_spread = np.max(values) - np.min(values)

            # Calculate composite difficulty score (normalized)
            if metric.lower_better:
                # Higher average score = harder dataset
                composite_score = avg_score
            else:
                # Lower average score = harder dataset, so invert
                composite_score = (
                    1.0 - avg_score if avg_score <= 1.0 else 1.0 / (1.0 + avg_score)
                )

            stats_list.append(
                DatasetDifficultyStats(
                    dataset_name=dataset,
                    algorithm_count=len(values),
                    metric_scores={metric.key: avg_score},
                    composite_score=composite_score,
                    rank_position=0.0,  # Will be calculated later
                    performance_variance=(
                        np.var(values, ddof=1) if len(values) > 1 else 0.0
                    ),
                    worst_algorithm=worst_algorithm,
                    best_algorithm=best_algorithm,
                    performance_spread=performance_spread,
                )
            )

        return stats_list

    def perform_dataset_friedman_test(
        self, metric: MetricConfig
    ) -> DatasetFriedmanResult:
        """Perform Friedman test to rank datasets by difficulty"""
        datasets = self.processed_data["datasets"]
        algorithms = self.processed_data["complete_algorithms"]
        data_by_algorithm = self.processed_data["data_by_algorithm"]

        n_datasets = len(datasets)
        n_algorithms = len(algorithms)

        if n_algorithms == 0 or n_datasets < 2:
            raise ValueError("Need at least 2 datasets and some complete algorithms")

        # Calculate ranks for each algorithm (datasets are being ranked)
        rank_sums = np.zeros(n_datasets)

        for algorithm in algorithms:
            values = []
            for dataset in datasets:
                value = data_by_algorithm[algorithm][dataset][metric.key]
                values.append((dataset, value))

            # Sort by value
            # For difficulty ranking: worse performance = higher rank (harder dataset)
            if metric.lower_better:
                # Higher values are worse, so sort descending (hardest first)
                values.sort(key=lambda x: x[1], reverse=True)
            else:
                # Lower values are worse, so sort ascending (hardest first)
                values.sort(key=lambda x: x[1])

            # Assign ranks (handle ties by averaging)
            ranks = np.zeros(n_datasets)
            i = 0
            while i < len(values):
                j = i
                # Find tied values
                while j < len(values) and values[j][1] == values[i][1]:
                    j += 1

                # Assign average rank
                avg_rank = (i + j + 1) / 2
                for k in range(i, j):
                    dataset_idx = datasets.index(values[k][0])
                    ranks[dataset_idx] = avg_rank

                i = j

            rank_sums += ranks

        # Calculate Friedman statistic
        sum_rank_squares = np.sum(rank_sums**2)
        friedman_stat = (
            12 / (n_algorithms * n_datasets * (n_datasets + 1))
        ) * sum_rank_squares - 3 * n_algorithms * (n_datasets + 1)

        # Degrees of freedom and critical value
        df = n_datasets - 1
        critical_value = chi2.ppf(0.95, df)  # α = 0.05
        p_value = 1 - chi2.cdf(friedman_stat, df)
        significant = friedman_stat > critical_value

        average_ranks = rank_sums / n_algorithms
        kendalls_w = friedman_stat / (n_algorithms * (n_datasets - 1))

        return DatasetFriedmanResult(
            n_algorithms=n_algorithms,
            n_datasets=n_datasets,
            rank_sums=rank_sums.tolist(),
            dataset_names=datasets.copy(),
            friedman_statistic=friedman_stat,
            degrees_freedom=df,
            p_value=p_value,
            kendalls_w=kendalls_w,
            critical_value=critical_value,
            significant=significant,
            average_ranks=average_ranks.tolist(),
        )

    def analyze_metric_difficulty(
        self, metric: MetricConfig
    ) -> DifficultyAnalysisResult:
        """Perform complete difficulty analysis for a single metric"""
        print(f"\nAnalyzing dataset difficulty for {metric.name}...")

        # Calculate dataset difficulty statistics
        dataset_stats = self.calculate_dataset_difficulty_stats(metric)

        # Perform Friedman test to rank datasets
        friedman_result = self.perform_dataset_friedman_test(metric)

        # Update rank positions in dataset stats
        for i, dataset in enumerate(friedman_result.dataset_names):
            for stat in dataset_stats:
                if stat.dataset_name == dataset:
                    stat.rank_position = friedman_result.average_ranks[i]
                    break

        # Create difficulty ranking based on composite scores
        difficulty_ranking = [
            (stat.dataset_name, stat.composite_score) for stat in dataset_stats
        ]
        difficulty_ranking.sort(key=lambda x: x[1], reverse=True)  # Hardest first

        # Identify hardest and easiest datasets (top/bottom 20% or at least 1)
        n_extreme = max(1, len(difficulty_ranking) // 5)
        hardest_datasets = [name for name, _ in difficulty_ranking[:n_extreme]]
        easiest_datasets = [name for name, _ in difficulty_ranking[-n_extreme:]]

        return DifficultyAnalysisResult(
            metric_config=metric,
            dataset_stats=dataset_stats,
            friedman_result=friedman_result,
            difficulty_ranking=difficulty_ranking,
            hardest_datasets=hardest_datasets,
            easiest_datasets=easiest_datasets,
        )

    def calculate_overall_difficulty_ranking(self) -> List[Tuple[str, float]]:
        """Calculate overall difficulty ranking across all metrics"""
        datasets = self.processed_data["datasets"]
        algorithms = self.processed_data["complete_algorithms"]
        data_by_algorithm = self.processed_data["data_by_algorithm"]

        overall_scores = {}

        for dataset in datasets:
            weighted_score = 0.0
            total_weight = 0.0

            for metric in self.metrics:
                values = []
                for algorithm in algorithms:
                    value = data_by_algorithm[algorithm][dataset][metric.key]
                    if not pd.isna(value) and np.isfinite(value):
                        values.append(value)

                if len(values) == 0:
                    continue

                # Normalize score for this metric
                avg_score = np.mean(values)
                if metric.lower_better:
                    # Higher values indicate harder datasets
                    normalized_score = avg_score
                else:
                    # Lower values indicate harder datasets
                    max_possible = (
                        1.0  # Assume metrics are bounded [0,1] or handle differently
                    )
                    normalized_score = (
                        max_possible - avg_score
                        if avg_score <= max_possible
                        else 1.0 / (1.0 + avg_score)
                    )

                weighted_score += normalized_score * metric.weight
                total_weight += metric.weight

            if total_weight > 0:
                overall_scores[dataset] = weighted_score / total_weight

        # Sort by difficulty (hardest first)
        return sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)

    def run_complete_analysis(self) -> Dict[str, DifficultyAnalysisResult]:
        """Run complete difficulty analysis for all metrics"""
        if not self.processed_data:
            raise ValueError("Data not preprocessed. Call preprocess_data() first.")

        results = {}
        for metric in self.metrics:
            try:
                results[metric.key] = self.analyze_metric_difficulty(metric)
            except Exception as e:
                print(f"Warning: Could not analyze {metric.name}: {e}")
                continue

        return results


class DatasetDifficultyPrinter:
    """Class responsible for printing dataset difficulty analysis results"""

    def __init__(
        self,
        results: Dict[str, DifficultyAnalysisResult],
        analyzer: DatasetDifficultyAnalyzer,
    ):
        """
        Initialize printer with analysis results

        Args:
            results: Dictionary mapping metric keys to analysis results
            analyzer: The analyzer instance for additional computations
        """
        self.results = results
        self.analyzer = analyzer

    def print_analysis_summary(self) -> None:
        """Print summary of the analysis"""
        if not self.results:
            return

        # Get summary info from any result
        first_result = next(iter(self.results.values()))
        n_algorithms = first_result.friedman_result.n_algorithms
        n_datasets = first_result.friedman_result.n_datasets

        print("=" * 80)
        print("DATASET DIFFICULTY ANALYSIS FOR CAUSAL DISCOVERY")
        print("=" * 80)
        print(f"Total algorithms analyzed: {n_algorithms}")
        print(f"Datasets evaluated: {n_datasets}")
        print(f"Metrics analyzed: {len(self.results)}")
        print(
            "Analysis perspective: Which datasets are harder/easier for causal discovery"
        )
        print()

    def print_metric_difficulty_analysis(self, metric_key: str) -> None:
        """Print detailed difficulty analysis for a single metric"""
        if metric_key not in self.results:
            print(f"No results found for metric: {metric_key}")
            return

        result = self.results[metric_key]
        metric = result.metric_config

        print("-" * 60)
        print(f"DATASET DIFFICULTY: {metric.name}")
        print(
            f"({'Lower scores = better performance' if metric.lower_better else 'Higher scores = better performance'})"
        )
        print("-" * 60)

        # Dataset Difficulty Statistics
        print("\n1. DATASET DIFFICULTY RANKING")
        print("-" * 30)

        # Sort by average rank (hardest first)
        sorted_stats = sorted(
            result.dataset_stats, key=lambda x: x.rank_position, reverse=True
        )

        print(
            f"{'Rank':<6} {'Dataset':<25} {'Avg Score':<12} {'Variance':<12} {'Spread':<12} {'Worst Algo':<15}"
        )
        print("-" * 85)

        for i, stat in enumerate(sorted_stats):
            rank_marker = (
                "🔴"
                if stat.dataset_name in result.hardest_datasets
                else "🟢" if stat.dataset_name in result.easiest_datasets else "  "
            )
            avg_score = list(stat.metric_scores.values())[0]
            print(
                f"{i+1:<4}{rank_marker} {stat.dataset_name:<25} {avg_score:<12.4f} {stat.performance_variance:<12.4f} {stat.performance_spread:<12.4f} {stat.worst_algorithm:<15}"
            )

        # Friedman Test Results
        print(f"\n2. STATISTICAL SIGNIFICANCE TEST")
        print("-" * 30)
        fr = result.friedman_result
        print(f"Friedman Statistic: {fr.friedman_statistic:.4f}")
        print(f"P-value: {fr.p_value:.6f}")
        print(f"Effect size (Kendall's W): {fr.kendalls_w:.6f}")
        print(f"Significant differences: {'Yes ✓' if fr.significant else 'No'}")

        if fr.significant:
            print(f"\n✓ Datasets show significantly different difficulty levels")
        else:
            print(f"\n⚠ No significant differences in dataset difficulty detected")

        # Extreme Cases
        print(f"\n3. EXTREME CASES")
        print("-" * 30)
        print(f"Hardest datasets ({len(result.hardest_datasets)}):")
        for dataset in result.hardest_datasets:
            # Find stats for this dataset
            dataset_stat = next(
                s for s in result.dataset_stats if s.dataset_name == dataset
            )
            avg_score = list(dataset_stat.metric_scores.values())[0]
            print(
                f"  🔴 {dataset}: score {avg_score:.4f}, variance {dataset_stat.performance_variance:.4f}"
            )

        print(f"\nEasiest datasets ({len(result.easiest_datasets)}):")
        for dataset in result.easiest_datasets:
            dataset_stat = next(
                s for s in result.dataset_stats if s.dataset_name == dataset
            )
            avg_score = list(dataset_stat.metric_scores.values())[0]
            print(
                f"  🟢 {dataset}: score {avg_score:.4f}, variance {dataset_stat.performance_variance:.4f}"
            )

        # Algorithm Agreement Analysis
        print(f"\n4. ALGORITHM AGREEMENT ANALYSIS")
        print("-" * 30)
        high_variance_datasets = sorted(
            result.dataset_stats, key=lambda x: x.performance_variance, reverse=True
        )[:3]
        low_variance_datasets = sorted(
            result.dataset_stats, key=lambda x: x.performance_variance
        )[:3]

        print("Datasets with highest algorithm disagreement (high variance):")
        for stat in high_variance_datasets:
            print(f"  📊 {stat.dataset_name}: variance {stat.performance_variance:.4f}")
            print(f"     Best: {stat.best_algorithm}, Worst: {stat.worst_algorithm}")

        print("\nDatasets with highest algorithm agreement (low variance):")
        for stat in low_variance_datasets:
            print(f"  📈 {stat.dataset_name}: variance {stat.performance_variance:.4f}")
            print(f"     Best: {stat.best_algorithm}, Worst: {stat.worst_algorithm}")

        print()

    def print_overall_difficulty_ranking(self) -> None:
        """Print overall difficulty ranking across all metrics"""
        print("=" * 80)
        print("OVERALL DATASET DIFFICULTY RANKING")
        print("=" * 80)

        overall_ranking = self.analyzer.calculate_overall_difficulty_ranking()

        print("Datasets ranked by overall difficulty (weighted across all metrics):")
        print(f"{'Rank':<6} {'Dataset':<30} {'Difficulty Score':<18} {'Category'}")
        print("-" * 70)

        n_datasets = len(overall_ranking)
        for i, (dataset, score) in enumerate(overall_ranking):
            if i < n_datasets // 3:
                category = "🔴 Hard"
            elif i < 2 * n_datasets // 3:
                category = "🟡 Medium"
            else:
                category = "🟢 Easy"

            print(f"{i+1:<6} {dataset:<30} {score:<18.4f} {category}")

        print()

    def print_cross_metric_summary(self) -> None:
        """Print summary comparing results across all metrics"""
        print("=" * 80)
        print("CROSS-METRIC DIFFICULTY SUMMARY")
        print("=" * 80)

        # Collect hardest/easiest datasets across metrics
        all_hardest = set()
        all_easiest = set()
        significant_metrics = 0

        for result in self.results.values():
            if result.friedman_result.significant:
                significant_metrics += 1
            all_hardest.update(result.hardest_datasets)
            all_easiest.update(result.easiest_datasets)

        print(
            f"📊 {significant_metrics}/{len(self.results)} metrics show significant difficulty differences"
        )
        print()

        # Consistently hard datasets
        consistently_hard = {}
        consistently_easy = {}

        for dataset in all_hardest:
            count = sum(
                1
                for result in self.results.values()
                if dataset in result.hardest_datasets
            )
            consistently_hard[dataset] = count

        for dataset in all_easiest:
            count = sum(
                1
                for result in self.results.values()
                if dataset in result.easiest_datasets
            )
            consistently_easy[dataset] = count

        if consistently_hard:
            print("🔴 CONSISTENTLY DIFFICULT DATASETS:")
            for dataset, count in sorted(
                consistently_hard.items(), key=lambda x: x[1], reverse=True
            ):
                print(
                    f"  {dataset}: ranked as hardest in {count}/{len(self.results)} metrics"
                )
            print()

        if consistently_easy:
            print("🟢 CONSISTENTLY EASY DATASETS:")
            for dataset, count in sorted(
                consistently_easy.items(), key=lambda x: x[1], reverse=True
            ):
                print(
                    f"  {dataset}: ranked as easiest in {count}/{len(self.results)} metrics"
                )
            print()

        # Summary statistics
        print("SUMMARY INSIGHTS:")
        print("-" * 20)

        if significant_metrics > len(self.results) // 2:
            print(
                "✓ Most metrics show significant difficulty differences between datasets"
            )
            print("✓ Dataset difficulty is a meaningful concept for causal discovery")
        else:
            print("⚠ Few metrics show significant difficulty differences")
            print("⚠ Datasets may be relatively similar in difficulty")

        if consistently_hard or consistently_easy:
            print(
                "✓ Some datasets are consistently harder/easier across multiple metrics"
            )
            print(
                "✓ These datasets could be used as benchmarks for algorithm evaluation"
            )
        else:
            print("⚠ No datasets are consistently hard/easy across metrics")
            print("⚠ Difficulty may be metric-specific")

        print()

    def print_complete_report(self) -> None:
        """Print complete difficulty analysis report"""
        self.print_analysis_summary()

        for metric_key in self.results.keys():
            self.print_metric_difficulty_analysis(metric_key)

        self.print_overall_difficulty_ranking()
        self.print_cross_metric_summary()


def main():
    """Main function to run the dataset difficulty analysis"""
    parser = argparse.ArgumentParser(
        description="Dataset Difficulty Analysis for Causal Discovery Algorithms"
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
        analyzer = DatasetDifficultyAnalyzer(args.csv_file)

        # Load and preprocess data
        print("Loading causal discovery performance data...")
        analyzer.load_data()

        print("Preprocessing data for difficulty analysis...")
        analyzer.preprocess_data()

        # Run analysis
        print("Running dataset difficulty analysis...")
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=RuntimeWarning)
            results = analyzer.run_complete_analysis()

        # Print results
        printer = DatasetDifficultyPrinter(results, analyzer)

        if args.summary_only:
            printer.print_analysis_summary()
            printer.print_overall_difficulty_ranking()
            printer.print_cross_metric_summary()
        elif args.metric:
            printer.print_analysis_summary()
            printer.print_metric_difficulty_analysis(args.metric)
        else:
            printer.print_complete_report()

    except Exception as e:
        print(f"Error: {e}")
        return 1

    return 0


if __name__ == "__main__":
    exit(main())
