from src.causal_discovery.CausalDiscoveryAlgorithm import \
    CausalDiscoveryAlgorithm
from src.causal_discovery.tetrad_algorithms import (BOSSTetrad, DAGMATetrad,
                                                    DirectLiNGAMTetrad,
                                                    FGESTetrad, GRASPTetrad,
                                                    PCTetrad)


def get_discovery_algorithm(**config_params) -> CausalDiscoveryAlgorithm:
    algorithm_name = config_params.pop("algorithm_name", "")

    if algorithm_name == "pc_tetrad":
        return PCTetrad(config_params)
    elif algorithm_name == "fges_tetrad":
        return FGESTetrad(config_params)
    elif algorithm_name == "boss_tetrad":
        return BOSSTetrad(config_params)
    elif algorithm_name == "grasp_tetrad":
        return GRASPTetrad(config_params)
    elif algorithm_name == "dagma_tetrad":
        return DAGMATetrad(config_params)
    elif algorithm_name == "directlingam_tetrad":
        return DirectLiNGAMTetrad(config_params)
    else:
        raise NotImplementedError(f"{algorithm_name} was not yet implemented")
