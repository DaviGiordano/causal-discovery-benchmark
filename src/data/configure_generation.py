def get_param_dicts():
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

    root_settings = [
        (["gmm"], [1.0]),
        (["bernoulli"], [1.0]),
    ]

    noises = ["gaussian", "uniform"]

    fits = ["linear", "lgbm", "diffusion", "causalflow"]
    configs = []
    for mech_pool, mech_prob in mechanism_settings:
        for root_pool, root_prob in root_settings:
            for noise in noises:
                for fit in fits:
                    gen_params = {
                        "nodes": 100,
                        "npoints": 10000,
                        "parents_max": -1,
                        "expected_degree": 5,
                        "dag_type": "erdos",
                        "noise_kind": noise,
                        "noise_coeff": 0.4,
                        "mechanism_pool": tuple(mech_pool),
                        "mechanism_prob": tuple(mech_prob),
                        "root_pool": tuple(root_pool),
                        "root_prob": tuple(root_prob),
                        "force_binary_root": True,
                        "binary_root_p": 0.5,
                        "random_state": 42,
                    }
                    gen_name = (
                        f"{gen_params['dag_type']}"
                        f"_n{gen_params['nodes']}"
                        f"_d{gen_params['expected_degree']}"
                        f"_root{'-'.join(root_pool)}"
                        f"_mech{'-'.join(mech_pool)}"
                        f"_noise{noise}"
                        # f"_fit{fit}"
                    )

                    configs.append(
                        {
                            "gen_name": gen_name,
                            "gen_params": gen_params,
                            "fit": fit,
                        }
                    )

    return configs
