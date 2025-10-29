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
                 p_tofolli: float = 0.0,
                 p_cp: float = 0.0,
                 p_t: float = 0.2,
                 p_h: float = 0.2,
                 p_s: float = 0.2,
                 p_cnot: float = 0.2,
                 p_not: float = 0.2,
                 seed:Optional[int]=None):
        """generate a random circuit: pyzx circuit then convert to graph"""
        self.num_qubits_min = num_qubits_min
        self.num_qubits_max = num_qubits_max
        self.min_gates = min_gates
        self.max_gates = max_gates
        self.p_tofolli = p_tofolli
        self.p_cp = p_cp
        self.p_t = p_t
        self.p_h = p_h
        self.p_s = p_s 
        self.p_cnot = p_cnot
        self.p_not = p_not
        self.seed = seed
        self.rng = np.random.default_rng(seed)

    def reset(self)->tuple:
        n_qubits = self.rng.integers(low=self.num_qubits_min, high=self.num_qubits_max+1)
        n_gates = self.rng.integers(low=self.min_gates, high=self.max_gates+1)
        fresh_seed = int(self.rng.integers(0, 2**31 - 1))
        self.circuit: zx_copy.Circuit = zx_copy.generate.my_circuit(n_qubits, n_gates,p_tofolli=self.p_tofolli, p_cp = self.p_cp, p_had=self.p_h, p_t=self.p_t, p_s=self.p_s, p_cnot=self.p_cnot, p_not = self.p_not, seed=fresh_seed)
        self.graph: zx_copy.Graph = self.circuit.to_graph()
        zx_copy.full_reduce(self.graph)
        return self.graph, self.circuit
    

def num_nodes_left(graph):
    G_nx = to_networkx_graph(graph)
    pairs = list(zip(graph.inputs(), graph.outputs()))  # fixed (si, ti)

    # path connecting inputs to outputs
    continuous_paths_dict = heuristic_fixed_pairs_node_disjoint(G_nx, pairs, iterations=5, restarts=3, seed=42)
    used_nodes = set().union(*continuous_paths_dict.values())
    left_nodes_by_continuous = set(graph.vertices()) - used_nodes # nodes not in any continuous path

    broken = broken_paths_for_unrouted_pairs(
        G_nx, continuous_paths_dict, pairs,
        include_endpoints_as_used=True,  # set False if endpoints of accepted paths may be reused
        edge_cost=1e-3,                  # tie-breaker toward shorter routes
        forbid_cost=1.0                  # cost per used (forbidden) node
    )
    broken_paths = []
    for (s,t), info in broken.items():
        broken_paths.append(sum(info["segments"], []))
    left_nodes_by_continuous_and_broken = left_nodes_by_continuous - set().union(*broken_paths) # nodes not in any continuous or broken path

    return len(left_nodes_by_continuous),len(left_nodes_by_continuous_and_broken)


def count_high_degree_nodes(graph, degree_threshold: int = 3) -> int:
    return sum(1 for v in graph.vertices() if len(graph.neighbors(v)) > degree_threshold)


def compute_dense_reward(graph) -> float:
    # Your existing dense reward formula, but callable at any time
    n_spiders = len(graph.vertices())
    n = max(1, n_spiders)
    current_left_nodes_by_continuous, current_left_nodes_by_continuous_and_broken = num_nodes_left(graph)
    current_high_degree_nodes = count_high_degree_nodes(graph)
    return (
        0.45*(n_spiders - current_left_nodes_by_continuous)/n
        + 0.10*(n_spiders - current_left_nodes_by_continuous_and_broken)/n
        + 0.45*(n_spiders - current_high_degree_nodes)/n
    )


def graph_is_done(g: zx_copy.Graph) -> bool:
    current_left_nodes_by_continuous, current_left_nodes_by_continuous_and_broken = num_nodes_left(g)
    current_high_degree_nodes = count_high_degree_nodes(g)
    return (current_left_nodes_by_continuous == 0 and current_high_degree_nodes == 0)

def build_graph_bank(
    out_path: str,
    circuits_out_path: Optional[str] = None,
    keep_limit: int = 1000,
    num_qubits_min: int = 2,
    num_qubits_max: int = 6,
    min_gates: int = 5,
    max_gates: int = 30,
    p_tofolli: float = 0.2,
    p_cp: float = 0.0,
    p_s: float = 0.1,
    p_cnot: float = 0.2,
    p_not: float = 0.1,
    p_t: float = 0.2,
    p_h: float = 0.2,
    max_reward: float = 0.9,
    seed: int = 123,
    ):
    rng = np.random.default_rng(seed)
    fresh_seed = int(rng.integers(0, 2**31 - 1))

    resetter = Resetter_Circuit(
        num_qubits_min=num_qubits_min,
        num_qubits_max=num_qubits_max,
        min_gates=min_gates,
        max_gates=max_gates,
        p_tofolli=p_tofolli,
        p_cp=p_cp,
        p_t=p_t,
        p_h=p_h,
        p_s=p_s,
        p_cnot=p_cnot,
        p_not=p_not,
        seed=fresh_seed,
    )

    kept = []
    circuits = []
    i=0
    while i < keep_limit:
        g, c = resetter.reset() 
        if compute_dense_reward(g) > max_reward:
            continue
        kept.append(g)
        circuits.append(c)
        i += 1

    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "wb") as f:
        # Highest protocol for speed and compactness
        pickle.dump(kept, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(kept)} graphs to {out_path}")

    # Derive circuits path if not provided
    if circuits_out_path is None:
        base, ext = os.path.splitext(out_path)
        circuits_out_path = base + "_circuits" + (ext if ext else ".pkl")
    os.makedirs(os.path.dirname(circuits_out_path) or ".", exist_ok=True)
    with open(circuits_out_path, "wb") as f:
        pickle.dump(circuits, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(circuits)} circuits to {circuits_out_path}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build a bank of random ZX graphs to use as env resetter.")
    parser.add_argument("--out_path", type=str, required=True, help="Ex:./data Path to save the graph bank (pickle file).")
    parser.add_argument("--keep_limit", type=int, default=1000, help="Number of graphs to keep in the bank.")
    parser.add_argument("--num_qubits_min", type=int, default=2)
    parser.add_argument("--num_qubits_max", type=int, default=6)
    parser.add_argument("--min_gates", type=int, default=5)
    parser.add_argument("--max_gates", type=int, default=30)
    parser.add_argument("--p_tofolli", type=float, default=0.2)
    parser.add_argument("--p_cp", type=float, default=0.0)
    parser.add_argument("--p_t", type=float, default=0.2)
    parser.add_argument("--p_h", type=float, default=0.2)
    parser.add_argument("--p_s", type=float, default=0.2)
    parser.add_argument("--p_cnot", type=float, default=0.2)
    parser.add_argument("--p_not", type=float, default=0.2)
    parser.add_argument("--max_reward", type=float, default=0.9, help="Max dense reward to accept a graph into the bank.")
    parser.add_argument("--seed", type=int, default=123)


    args = parser.parse_args()
    exp_name = f"GraphBank_nq[{args.num_qubits_min}-{args.num_qubits_max}]_gates[{args.min_gates}-{args.max_gates}]_T{args.p_t}_H{args.p_h}_S{args.p_s}_CX{args.p_cnot}_X{args.p_not}_TOF{args.p_tofolli}_CP{args.p_cp}_length{args.keep_limit}"
    
    out_path = os.path.join(args.out_path, f"{exp_name}.pkl")
    circuits_out_path = os.path.join(args.out_path, f"{exp_name}_circuits.pkl")
    build_graph_bank(
        out_path=out_path,
        keep_limit=args.keep_limit,
        num_qubits_min=args.num_qubits_min,
        num_qubits_max=args.num_qubits_max,
        min_gates=args.min_gates,
        max_gates=args.max_gates,
        p_tofolli=args.p_tofolli,
        p_cp=args.p_cp,
        p_t=args.p_t,
        p_h=args.p_h,
        p_s=args.p_s,
        p_cnot=args.p_cnot,
        p_not=args.p_not,
        max_reward=args.max_reward,
        seed=args.seed,
        circuits_out_path=circuits_out_path,
    )