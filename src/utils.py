"""
Utility functions for the causal discovery benchmark.
"""

import os

import yaml


def write_generation_configs(output_file: str = "configs/generation.yaml"):
    """Write all possible generation configurations to a YAML file."""

    # Dataset parameters from benchmark_synthetic.py
    nodes_degree_points_list = [
        (5, 2, 1000),
        (10, 2, 1000),
        (20, 2, 1000),
        (40, 2, 1000),
        (50, 2, 1000),
    ]
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

    configs = {}

    for nodes, expected_degree, points in nodes_degree_points_list:
        for mech_pool, mech_prob in mechanism_settings:
            gen_name = (
                f"erdos_n{nodes}_d{expected_degree}_"
                f"rootgmm_mech{'-'.join(mech_pool)}_noisegaussian"
            )

            gen_params = {
                "nodes": nodes,
                "npoints": points,
                "parents_max": -1,
                "expected_degree": expected_degree,
                "dag_type": "erdos",
                "noise_kind": "gaussian",
                "noise_coeff": 0.4,
                "mechanism_pool": mech_pool,
                "mechanism_prob": mech_prob,
                "root_pool": ["gmm"],
                "root_prob": [1.0],
                "force_binary_root": False,
                "binary_root_p": 0,
                "random_state": 42,
            }

            configs[gen_name] = gen_params

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Write to YAML file
    with open(output_file, "w") as f:
        yaml.dump(configs, f, default_flow_style=False, indent=2)

    print(f"Generated {len(configs)} configurations in {output_file}")
    return configs


if __name__ == "__main__":
    write_generation_configs()
