"""
This class is heavily inspired by Acyclic Graph Generator
from https://fentechsolutions.github.io/CausalDiscoveryToolbox/html/index.html
"""

import random
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence

import networkx as nx
import numpy as np
import pandas as pd
from sklearn.preprocessing import scale

from .causal_mechanisms import (
    Binary_Mechanism,
    GaussianProcessAdd_Mechanism,
    GaussianProcessMix_Mechanism,
    LinearMechanism,
    NN_Mechanism,
    Polynomial_Mechanism,
    SigmoidAM_Mechanism,
    SigmoidMix_Mechanism,
    bernoulli_cause,
    gaussian_cause,
    gmm_cause,
    normal_noise,
    uniform_noise,
)

MECH_MAP = {
    "linear": LinearMechanism,
    "polynomial": Polynomial_Mechanism,
    "sigmoid_add": SigmoidAM_Mechanism,
    "sigmoid_mix": SigmoidMix_Mechanism,
    "gp_add": GaussianProcessAdd_Mechanism,
    "gp_mix": GaussianProcessMix_Mechanism,
    "nn": NN_Mechanism,
    "binary": Binary_Mechanism,
}
ROOT_GEN_MAP = {
    "gmm": gmm_cause,
    "gaussian": gaussian_cause,
    "bernoulli": bernoulli_cause,
}


@dataclass(frozen=True)
class GenerationMetadata:
    adjacency_matrix: np.ndarray
    mechanisms: Dict[str, str]
    variable_types: Dict[str, str]
    binary_root: str

    def to_dict(self) -> Dict[str, object]:
        return {
            "adjacency_matrix": self.adjacency_matrix.copy(),
            "mechanisms": dict(self.mechanisms),
            "variable_types": dict(self.variable_types),
            "binary_root": self.binary_root,
        }

    def __str__(self) -> str:
        n_nodes = int(self.adjacency_matrix.shape[0])
        n_edges = int(self.adjacency_matrix.sum())
        mech_counts: Dict[str, int] = {}
        for m in self.mechanisms.values():
            mech_counts[m] = mech_counts.get(m, 0) + 1
        mech_summary = ", ".join(f"{k}:{v}" for k, v in sorted(mech_counts.items()))
        return (
            f"GenerationMetadata(nodes={n_nodes}, edges={n_edges}, "
            f"binary_root={self.binary_root}, mechanisms=[{mech_summary}])"
        )


class AcyclicGraphGenerator:
    """Synthetic DAG data generator with pluggable causal mechanisms."""

    def __init__(
        self,
        nodes: int = 20,
        npoints: int = 500,
        parents_max: int = 5,
        expected_degree: int = 3,
        dag_type: str = "erdos",
        noise_kind: str = "gaussian",
        noise_coeff: float = 0.4,
        mechanism_pool: Sequence[str] = ("linear",),
        mechanism_prob: Optional[Sequence[float]] = None,
        root_pool: Sequence[str] = ("gmm",),
        root_prob: Optional[Sequence[float]] = None,
        force_binary_root: bool = True,
        binary_root_p: float = 0.5,
        random_state: Optional[int] = None,
    ):
        self.nodes = nodes
        self.npoints = npoints
        self.parents_max = parents_max
        self.expected_degree = expected_degree
        self.dag_type = dag_type
        self.noise_kind = noise_kind
        self.noise_coeff = noise_coeff
        self.mechanism_pool = list(mechanism_pool)
        self.mechanism_prob = mechanism_prob
        self.root_pool = list(root_pool)
        self.root_prob = root_prob
        self.force_binary_root = force_binary_root
        self.binary_root_p = binary_root_p
        self.random_state = random_state

        if self.random_state is not None:
            random.seed(self.random_state)
            np.random.seed(self.random_state)

        self._noise = {
            "gaussian": normal_noise,
            "uniform": uniform_noise,
        }.get(self.noise_kind, self.noise_kind)

        self.adjacency_matrix = np.zeros((self.nodes, self.nodes), dtype=int)
        self._cfuncs: List[Callable] = [None] * self.nodes
        self._variable_types: List[str] = ["cont"] * self.nodes
        self._mechanism_names: List[str] = [None] * self.nodes
        self._binary_root_name: Optional[str] = None

    @staticmethod
    def _bernoulli_cause(pts, binary_root_p):
        return bernoulli_cause(pts, binary_root_p)

    def _choose_binary_root(self) -> int:
        roots = np.where(~self.adjacency_matrix.any(axis=0))[0]
        idx = np.random.choice(roots)
        self._binary_root_name = f"V{idx}"
        return idx

    def _init_dag(self) -> None:
        A = self.adjacency_matrix  # alias

        if self.dag_type == "default":
            for j in range(1, self.nodes):
                k = np.random.randint(0, min(self.parents_max, j) + 1)
                parents = np.random.choice(range(j), k, replace=False)
                A[parents, j] = 1

        elif self.dag_type == "erdos":
            n_edges = self.expected_degree * self.nodes
            p_conn = 2 * n_edges / (self.nodes * (self.nodes - 1))
            order = np.random.permutation(self.nodes)
            for i, node in enumerate(order[:-1]):
                potential = order[i + 1 :]
                k = np.random.binomial(len(potential), p_conn)
                parents = np.random.choice(potential, k, replace=False)
                A[parents, node] = 1
        else:
            raise ValueError(f"Unsupported dag_type: {self.dag_type}")

        # Has cycles. Regenerate
        if list(nx.simple_cycles(nx.DiGraph(A))):
            self.adjacency_matrix[:] = 0
            self._init_dag()

    def _init_variables(self) -> None:
        self._init_dag()
        bin_root = self._choose_binary_root() if self.force_binary_root else None

        for i in range(self.nodes):
            n_par = int(self.adjacency_matrix[:, i].sum())

            # roots
            if n_par == 0:
                if i == bin_root:
                    gen = lambda pts, p=self.binary_root_p: bernoulli_cause(pts, p)
                    root_name = "bernoulli"
                    self._variable_types[i] = "binary"
                    self._mechanism_names[i] = "binary"
                else:
                    root_name = np.random.choice(self.root_pool, p=self.root_prob)
                    gen = ROOT_GEN_MAP[root_name]
                    self._mechanism_names[i] = gen.__name__

                self._cfuncs[i] = gen

            # non‑roots
            else:
                mech_name = np.random.choice(self.mechanism_pool, p=self.mechanism_prob)
                mech_cls = MECH_MAP[mech_name]
                self._cfuncs[i] = mech_cls(
                    n_par, self.npoints, self._noise, noise_coeff=self.noise_coeff
                )
                self._mechanism_names[i] = mech_name
                if str(mech_name) == "binary":
                    self._variable_types[i] = "binary"

    def generate(self, rescale: bool = True, return_metadata: bool = False):
        if self._cfuncs[0] is None:
            self._init_variables()

        df = pd.DataFrame(
            index=range(self.npoints), columns=[f"V{i}" for i in range(self.nodes)]
        )
        order = nx.topological_sort(nx.DiGraph(self.adjacency_matrix))

        for i in order:
            parents = np.where(self.adjacency_matrix[:, i])[0]

            if parents.size == 0:
                col = self._cfuncs[i](self.npoints)
            else:
                X = df.iloc[:, list(parents)].values
                if X.ndim == 1:
                    X = X[:, None]
                col = self._cfuncs[i](X)

            col = np.squeeze(col)

            if rescale and not (self._variable_types[i] == "binary"):
                col = scale(col)

            df[f"V{i}"] = col

        graph = nx.relabel_nodes(
            nx.DiGraph(self.adjacency_matrix), {i: f"V{i}" for i in range(self.nodes)}
        )

        if return_metadata:
            meta = GenerationMetadata(
                adjacency_matrix=self.adjacency_matrix.copy(),
                mechanisms={
                    n: self._mechanism_names[i] for i, n in enumerate(graph.nodes)
                },
                variable_types={
                    n: self._variable_types[i] for i, n in enumerate(graph.nodes)
                },
                binary_root=self._binary_root_name,
            )
            return df, graph, meta
        return df, graph
