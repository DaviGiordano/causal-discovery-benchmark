import json
import logging

import numpy as np
import pandas as pd

from src.causal_discovery.CausalDiscoveryAlgorithm import CausalDiscoveryAlgorithm
from src.parse_tetrad_string import (
    str_to_edge_dict,
    str_to_edge_probabilities,
    str_to_general_graph,
)
from src.py_tetrad.pytetrad.tools.TetradSearch import TetradSearch

logger = logging.getLogger(__name__)

# Functions and default configuration parameters for independence test methods in Tetrad.
# Only used when a required parameter is not configured in configs/algorithm_params.yaml
TEST_METHODS = {
    "fisherz": {
        "func_name": "use_fisher_z",
        "default_params": {
            "alpha": 0.01,
            "use_for_mc": False,
            "singularity_lambda": 0.0,
        },
    },
    "conditional_gaussian_test": {
        "func_name": "use_conditional_gaussian_test",
        "default_params": {
            "alpha": 0.01,
            "discretize": True,
            "num_categories_to_discretize": 3,
            "use_for_mc": False,
        },
    },
    "degenerate_gaussian_test": {
        "func_name": "use_degenerate_gaussian_test",
        "default_params": {
            "alpha": 0.01,
            "use_for_mc": False,
            "singularity_lambda": 0.0,
        },
    },
}

# Functions and default configuration parameters for scoring methods in Tetrad.
# Only used when a required parameter is not configured in configs/algorithm_params.yaml
SCORE_METHODS = {
    "conditional_gaussian_score": {
        "func_name": "use_conditional_gaussian_score",
        "default_params": {
            "penalty_discount": 1,
            "discretize": True,
            "num_categories_to_discretize": 3,
            "structure_prior": 0,
        },
    },
    "degenerate_gaussian_score": {
        "func_name": "use_degenerate_gaussian_score",
        "default_params": {
            "penalty_discount": 1,
            "structure_prior": 0,
            "singularity_lambda": 0.0,
        },
    },
}


class TetradAlgorithm(CausalDiscoveryAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)

    def _set_test_and_score(self, search):
        """Set test, score, or both based on configuration."""

        test_name = self.config_params.get("test_name", "")
        score_name = self.config_params.get("score_name", "")

        if test_name:
            test_params = self.config_params.get("test_params", {})
            self._configure_method(search, test_name, test_params, "test")

        if score_name:
            score_params = self.config_params.get("score_params", {})
            self._configure_method(search, score_name, score_params, "score")

        log_message = f"Configured {'test' if test_name else ''}"
        log_message += f"{' and ' if test_name and score_name else ''}"
        log_message += f"{'score' if score_name else ''}"
        log_message += " (none)" if not (test_name or score_name) else ""
        logger.info(log_message)

    def _configure_method(self, search, method_name, method_params, method_type):
        """Configure a specific test or score method with its parameters."""

        methods_map = SCORE_METHODS if method_type == "score" else TEST_METHODS

        if method_name not in methods_map:
            error_msg = f"{method_type.capitalize()} method '{method_name}' not found."
            logger.error(error_msg)
            raise ValueError(error_msg)

        func_info = methods_map[method_name]
        func_name = func_info["func_name"]
        default_params = func_info.get("default_params", {})
        final_params = default_params.copy()
        final_params.update(method_params)

        try:
            logger.info(
                f"Configuring {method_type} '{method_name}' with params:\n"
                f"{json.dumps(final_params, indent=2)}"
            )

            method = getattr(search, func_name)
            method(**final_params)

        except Exception as e:
            error_msg = (
                f"Error configuring {method_type} '{method_name}': {str(e)}\n"
                f"Config: {self.config_params}"
            )
            logger.error(error_msg)
            raise ValueError(error_msg)

    def _set_bootstrap(self, search):

        bootstrap_strategy = self.config_params.get("bootstrap_strategy", "")

        if bootstrap_strategy == "bootstrap_100_100":
            number_resampling = 100
            percent_resample_size = 100
            with_replacement = True
            add_original = False
            resampling_ensemble = 3
            seed = 42
        elif bootstrap_strategy == "bootstrap_2_100":
            number_resampling = 2
            percent_resample_size = 100
            with_replacement = True
            add_original = False
            resampling_ensemble = 3
            seed = 42
        elif bootstrap_strategy == "jackknife90":
            number_resampling = 10
            percent_resample_size = 90
            with_replacement = False
            add_original = True
            resampling_ensemble = 1
            seed = 42
        else:
            raise ValueError(
                f"'{bootstrap_strategy}' Bootstrap strategy is not implemented"
            )

        bootstrap_params = {
            "numberResampling": number_resampling,
            "percent_resample_size": percent_resample_size,
            "add_original": add_original,
            "with_replacement": with_replacement,
            "resampling_ensemble": resampling_ensemble,
            "seed": seed,
        }

        try:
            logger.info(
                f"Configuring bootstrap with params:\n{json.dumps(bootstrap_params, indent=2)}"
            )
            search.set_bootstrapping(**bootstrap_params)
        except Exception as ex:
            logger.error(
                f"Exception: {ex.message()}",
                f"Config params: {self.config_params}",
            )
            raise ex

    def _algorithm_specific_train(self, search: TetradSearch) -> TetradSearch:
        raise NotImplementedError()

    def train(self, data: np.ndarray, node_names: list = []) -> None:
        """Train the algorithm on the provided data."""
        if not node_names:
            node_names = [f"X{i}" for i in range(data.shape[1])]

        df = pd.DataFrame(data, columns=node_names)

        search = TetradSearch(df)
        search.set_verbose(False)

        self._set_test_and_score(search)

        if self.config_params.get("bootstrap_strategy"):
            self._set_bootstrap(search)

        self._algorithm_specific_train(search)

        self.graph_string = str(search.get_string())
        self.est_dotgraph = search.get_dot()
        self.est_graph = str_to_general_graph(self.graph_string)
        self.edge_probabilities = str_to_edge_probabilities(self.graph_string)
        self.est_edges_dict = str_to_edge_dict(self.graph_string)
        self._set_auxiliary_results()


class PCTetrad(TetradAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(
            f"Instantiated PC algorithm with params:\n{json.dumps(self.config_params, indent=2)}"
        )

    def _algorithm_specific_train(self, search: TetradSearch) -> TetradSearch:
        algorithm_params = self.config_params.get("algorithm_params", {})
        conflict_rule = algorithm_params.get("conflict_rule", 1)
        depth = algorithm_params.get("depth", -1)
        stable_fas = algorithm_params.get("stable_fas", True)
        guarantee_cpdag = algorithm_params.get("guarantee_cpdag", False)

        params_dict = {
            "conflict_rule": conflict_rule,
            "depth": depth,
            "stable_fas": stable_fas,
            "guarantee_cpdag": guarantee_cpdag,
        }

        try:
            logger.info(f"Running PC with params:\n{json.dumps(params_dict, indent=2)}")
            search.run_pc(**params_dict)

        except Exception as ex:
            logger.error(
                f"Exception: {ex.message()}",
                f"Config params: {self.config_params}",
            )
            raise ex

        return search


class BOSSTetrad(TetradAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(
            f"Instantiated BOSS algorithm with params:\n{json.dumps(self.config_params, indent=2)}"
        )

    def _algorithm_specific_train(self, search: TetradSearch) -> TetradSearch:
        algorithm_params = self.config_params.get("algorithm_params", {})

        num_starts = algorithm_params.get("num_starts", 1)
        use_bes = algorithm_params.get("use_bes", False)
        time_lag = algorithm_params.get("time_lag", 0)
        use_data_order = algorithm_params.get("use_data_order", True)
        output_cpdag = algorithm_params.get("output_cpdag", True)

        params_dict = {
            "num_starts": num_starts,
            "use_bes": use_bes,
            "time_lag": time_lag,
            "use_data_order": use_data_order,
            "output_cpdag": output_cpdag,
        }

        try:
            logger.info(
                f"Running BOSS with params:\n{json.dumps(params_dict, indent=2)}"
            )
            search.run_boss(**params_dict)

        except Exception as ex:
            logger.error(
                f"Exception: {ex.message()}",
                f"Config params: {self.config_params}",
            )
            raise ex

        return search


class FGESTetrad(TetradAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(
            f"Instantiated FGES algorithm with params:\n{json.dumps(self.config_params, indent=2)}"
        )

    def _algorithm_specific_train(self, search: TetradSearch) -> TetradSearch:
        algorithm_params = self.config_params.get("algorithm_params", {})

        symmetric_first_step = algorithm_params.get("symmetric_first_step", False)
        max_degree = algorithm_params.get("max_degree", -1)
        parallelized = algorithm_params.get("parallelized", False)
        faithfulness_assumed = algorithm_params.get("faithfulness_assumed", False)

        params_dict = {
            "symmetric_first_step": symmetric_first_step,
            "max_degree": max_degree,
            "parallelized": parallelized,
            "faithfulness_assumed": faithfulness_assumed,
        }

        try:
            logger.info(
                f"Running FGES with params:\n{json.dumps(params_dict, indent=2)}"
            )
            search.run_fges(**params_dict)

        except Exception as ex:
            logger.error(
                f"Exception: {ex.message()}",
                f"Config params: {self.config_params}",
            )
            raise ex

        return search


class GRASPTetrad(TetradAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(
            f"Instantiated GRASP algorithm with params:\n{json.dumps(self.config_params, indent=2)}"
        )

    def _algorithm_specific_train(self, search: TetradSearch) -> TetradSearch:
        algorithm_params = self.config_params.get("algorithm_params", {})

        covered_depth = algorithm_params.get("covered_depth", 4)
        singular_depth = algorithm_params.get("singular_depth", 1)
        nonsingular_depth = algorithm_params.get("nonsingular_depth", 1)
        ordered_alg = algorithm_params.get("ordered_alg", False)
        raskutti_uhler = algorithm_params.get("raskutti_uhler", False)
        use_data_order = algorithm_params.get("use_data_order", True)
        num_starts = algorithm_params.get("num_starts", 1)

        params_dict = {
            "covered_depth": covered_depth,
            "singular_depth": singular_depth,
            "nonsingular_depth": nonsingular_depth,
            "ordered_alg": ordered_alg,
            "raskutti_uhler": raskutti_uhler,
            "use_data_order": use_data_order,
            "num_starts": num_starts,
        }

        try:
            logger.info(
                f"Running GRASP with params:\n{json.dumps(params_dict, indent=2)}"
            )
            search.run_grasp(**params_dict)

        except Exception as ex:
            logger.error(
                f"Exception: {ex.message()}",
                f"Config params: {self.config_params}",
            )
            raise ex

        return search


class DAGMATetrad(TetradAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(
            f"Instantiated DAGMA algorithm with params:\n{json.dumps(self.config_params, indent=2)}"
        )

    def _algorithm_specific_train(self, search: TetradSearch) -> TetradSearch:
        algorithm_params = self.config_params.get("algorithm_params", {})
        dagma_lambda = algorithm_params.get("dagma_lambda", 0.05)
        w_threshold = algorithm_params.get("w_threshold", 0.1)
        cpdag = algorithm_params.get("cpdag", True)

        params_dict = {
            "dagma_lambda": dagma_lambda,
            "w_threshold": w_threshold,
            "cpdag": cpdag,
        }

        try:
            logger.info(
                f"Running DAGMA with params:\n{json.dumps(params_dict, indent=2)}"
            )
            search.run_dagma(**params_dict)

        except Exception as ex:
            logger.error(
                f"Exception: {ex.message()}",
                f"Config params: {self.config_params}",
            )
            raise ex

        return search


class DirectLiNGAMTetrad(TetradAlgorithm):
    def __init__(self, config_params):
        super().__init__(config_params)
        logger.info(
            f"Instantiated Direct-LiNGAM algorithm with params:\n{json.dumps(self.config_params, indent=2)}"
        )

    def _algorithm_specific_train(self, search: TetradSearch) -> TetradSearch:
        try:
            logger.info(f"Running Direct-LiNGAM with default parameters")
            search.run_direct_lingam()

        except Exception as ex:
            logger.error(
                f"Exception: {ex.message()}",
                f"Config params: {self.config_params}",
            )
            raise ex

        return search
