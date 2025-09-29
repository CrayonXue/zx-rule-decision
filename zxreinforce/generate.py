# Holds resetters for the ZX env returning 
# (colors, angles, selected_node, source, target, selected_edges)

import numpy as np
from typing import Optional, List, Set, Union
from fractions import Fraction
from typing import Union
import os, pickle
import sys
sys.path.append('../pyzx_copy')
import pyzx_copy as zx_copy
from pyzx_copy.utils import heuristic_fixed_pairs_node_disjoint, broken_paths_for_unrouted_pairs, to_networkx_graph


class Resetter_Circuit():
    def __init__(self, 
                 num_qubits_min:int,
                 num_qubits_max:int,
                 min_gates:int,
                 max_gates:int,
                 p_t: float = 0.2,
                 p_h: float = 0.2,
                 seed:Optional[int]=None):
        """generate a random circuit: pyzx circuit then convert to graph"""
        self.num_qubits_min = num_qubits_min
        self.num_qubits_max = num_qubits_max
        self.min_gates = min_gates
        self.max_gates = max_gates
        self.p_t = p_t
        self.p_h = p_h
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self)->tuple:
        n_qubits = self.rng.integers(low=self.num_qubits_min, high=self.num_qubits_max+1)
        n_gates = self.rng.integers(low=self.min_gates, high=self.max_gates+1)
        self.circuit: zx_copy.Circuit = zx_copy.generate.CNOT_HAD_PHASE_circuit(n_qubits, n_gates, p_had=self.p_h, p_t=self.p_t)
        self.graph: zx_copy.Graph = self.circuit.to_graph().copy()
        zx_copy.full_reduce(self.graph)
        return self.graph, self.circuit
    

def num_nodes_left(graph):
        G_nx = to_networkx_graph(graph)
        pairs = list(zip(graph.inputs(), graph.outputs()))  # fixed (si, ti)

        # path connecting inputs to outputs
        continuous_paths_dict = heuristic_fixed_pairs_node_disjoint(G_nx, pairs, iterations=10, restarts=5, seed=42)
        used_nodes = set().union(*continuous_paths_dict.values())
        left_nodes_by_continuous = set(graph.vertices()) - used_nodes # nodes not in any continuous path
        return len(left_nodes_by_continuous)

def count_high_degree_nodes(graph, degree_threshold: int = 3) -> int:
    return sum(1 for v in graph.vertices() if len(graph.neighbors(v)) > degree_threshold)

def graph_is_done(g: zx_copy.Graph) -> bool:
    return (num_nodes_left(g) == 0 and count_high_degree_nodes(g) == 0)
    

def build_graph_bank(
    out_path: str,
    keep_limit: int = 1000,
    num_qubits_min: int = 2,
    num_qubits_max: int = 6,
    min_gates: int = 5,
    max_gates: int = 30,
    p_t: float = 0.2,
    p_h: float = 0.2,
    seed: int = 123,
):
    rng = np.random.default_rng(seed)
    resetter = Resetter_Circuit(
        num_qubits_min=num_qubits_min,
        num_qubits_max=num_qubits_max,
        min_gates=min_gates,
        max_gates=max_gates,
        p_t=p_t,
        p_h=p_h,
        seed=seed,
    )

    kept = []
    i=0
    while i < keep_limit:
        g, c = resetter.reset()          # pyzx graph + circuit
        g = g.copy()                     # defensive copy
        # optional: ensure any normalization you want is applied once here
        # zx_copy.full_reduce(g)  # already called in Resetter_Circuit.reset()

        if graph_is_done(g):
            continue
        kept.append(g)
        i += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        # Highest protocol for speed and compactness
        pickle.dump(kept, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(kept)} graphs to {out_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a bank of random ZX graphs to use as env resetter.")
    parser.add_argument("--out_path", type=str, required=True, help="Ex:./data Path to save the graph bank (pickle file).")
    parser.add_argument("--keep_limit", type=int, default=1000, help="Number of graphs to keep in the bank.")
    parser.add_argument("--num_qubits_min", type=int, default=2)
    parser.add_argument("--num_qubits_max", type=int, default=6)
    parser.add_argument("--min_gates", type=int, default=5)
    parser.add_argument("--max_gates", type=int, default=30)
    parser.add_argument("--p_t", type=float, default=0.2)
    parser.add_argument("--p_h", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=123)

    args = parser.parse_args()
    exp_name = f"GraphBank_nq[{args.num_qubits_min}-{args.num_qubits_max}]_gates[{args.min_gates}-{args.max_gates}]_length{args.keep_limit}.pkl"
    out_path = os.path.join(args.out_path, exp_name)
    build_graph_bank(
        out_path=out_path,
        keep_limit=args.keep_limit,
        num_qubits_min=args.num_qubits_min,
        num_qubits_max=args.num_qubits_max,
        min_gates=args.min_gates,
        max_gates=args.max_gates,
        p_t=args.p_t,
        p_h=args.p_h,
        seed=args.seed,
    )